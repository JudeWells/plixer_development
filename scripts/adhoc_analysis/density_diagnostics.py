"""What kind of density does a Poc2Mol checkpoint produce, and how much of it is the pocket?

Two diagnostics that Dice alone cannot give you, both model-agnostic. They were written
while evaluating a flow-matching Poc2Mol against the regression one, but nothing here is
specific to either -- any model that maps a protein grid to a ligand grid can be passed in.

1. POCKET DECOMPOSITION. Score the prediction against the true ligand three ways:

       correct pocket   what the model actually achieves
       shuffled pocket  a real pocket, but the wrong one -- isolates the generic
                        "a ligand of about this size sits near the box centre" signal
       zero pocket      the pocket-blind floor

   The gaps decompose the score. Measured on poc2mol_v2_11ch_ep576 (2026-08-12, 1019 val
   pockets): floor 0.2116, +0.041 from any pocket, +0.255 from the CORRECT pocket. So the
   pocket-specific term is the dominant one -- worth knowing before blaming conditioning
   for anything.

2. STRAY-DENSITY AMPLITUDE PROFILE. Count predicted voxels above several thresholds in
   places the true grid is EMPTY. This separates two failure modes that every aggregate
   metric conflates:

       many voxels, low amplitude  = a diffuse haze. Cheap under MSE (squaring), expensive
                                     under mass-based metrics like on_target/empty_frac.
       few voxels, high amplitude  = confident hallucination. The reverse.

   The same 11ch checkpoint puts 646 voxels per pocket above 0.5 in empty space at mean
   amplitude 0.106. Dice, MSE and the emission metrics each go blind to one of these, which
   is why three metrics can disagree about which of two models is better.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/density_diagnostics.py \\
        --ckpt checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt \\
        --data poc2mol_hiqbind_v2_11ch --out_channels 11
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import hydra  # noqa: E402

from src.data.common.voxelization.batched import BatchedVoxelizer  # noqa: E402
from src.models.pytorch3dunet import ResidualUNetSE3D  # noqa: E402
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE  # noqa: E402


def pooled_soft_dice(a, b, eps=1e-8):
    """Per-sample soft Dice pooled over channels -- the same function the rotation control
    uses, so numbers from the two scripts are directly comparable."""
    dims = tuple(range(1, a.dim()))
    num = 2.0 * (a * b).sum(dim=dims)
    den = (a * a).sum(dim=dims) + (b * b).sum(dim=dims)
    return (num / (den + eps)).cpu().numpy()


def load_batches(data_config, batch_size, n_batches, device):
    """Voxelised (protein, ligand) pairs from a DATA config's validation split.

    A data config rather than an experiment, because the channel scheme belongs to the
    checkpoint: an 11-channel model must be scored against 11-channel targets, and no
    experiment on this branch composes the v2 11ch poc2mol data.

    ⚠️ `data.config.batch_size` controls the TRAIN loader. ComplexDataModule's validation
    loader uses `data.val_batch_size` (or min(4, batch_size) when unset), so the number of
    pockets is n_batches x THAT, not n_batches x batch_size. Getting this wrong silently
    compares numbers computed on different subsets.
    """
    root = os.path.dirname(os.path.dirname(_HERE))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"data={data_config}", "data.num_workers=4",
            f"data.config.batch_size={batch_size}",
            # Set the VAL batch size explicitly rather than inheriting min(4, batch_size).
            # Without this, --batch_size silently does not apply to validation and
            # --n_batches * 4 pockets get scored -- the trap described above.
            f"+data.val_batch_size={batch_size}",
            "paths.output_dir=/tmp/density_diag",
            "paths.img_save_dir=/tmp/density_diag/img"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    n_protein = len(dm.config.protein_channels) if dm.config.has_protein else 0
    vx = BatchedVoxelizer(
        dm.config, cutoff_ratio=dm.config.get("voxel_cutoff_ratio", 2.0),
        aggregation=dm.config.get("voxel_aggregation", "max"),
        radius_scale=dm.config.get("voxel_radius_scale", 1.0)).to(device)

    loader = dm.val_dataloader()
    loader = loader[0] if isinstance(loader, (list, tuple)) else loader
    out = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        grid = vx(batch["atom_xyz"].to(device), batch["atom_radius"].to(device),
                  batch["atom_slot"].to(device), int(batch["batch_size"]),
                  int(batch["n_channels"]))
        out.append((grid[:, :n_protein].float(), grid[:, n_protein:].float()))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="poc2mol_hiqbind_v2_11ch",
                        help="DATA config supplying the val split; its ligand channel count "
                             "must match --out_channels")
    parser.add_argument("--out_channels", type=int, default=11)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_batches", type=int, default=10**6, help="default: whole split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    state = {k[len("model."):]: v for k, v in state.get("state_dict", state).items()
             if k.startswith("model.")}
    net = ResidualUNetSE3D(
        in_channels=args.in_channels, out_channels=args.out_channels,
        basic_module=ResNetBlockSE, f_maps=64, num_levels=5,
        layer_order="gcr", num_groups=8).to(device).eval()
    incompatible = net.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"WARNING state_dict mismatch: missing={incompatible.missing_keys[:4]} "
              f"unexpected={incompatible.unexpected_keys[:4]}")

    batches = load_batches(args.data, args.batch_size, args.n_batches, device)
    n = sum(l.shape[0] for _, l in batches)
    n_lig = batches[0][1].shape[1]
    if n_lig != args.out_channels:
        raise SystemExit(
            f"channel mismatch: --data {args.data} produces {n_lig} ligand channels but the "
            f"model was built with out_channels={args.out_channels}. Pick the data config "
            f"matching the checkpoint's scheme (poc2mol_hiqbind_v2_9ch / _v2_11ch)."
        )
    print(f"{args.ckpt}\n{n} pockets from {args.data} val, {n_lig} ligand channels\n")

    results = {"ckpt": args.ckpt, "n_pockets": int(n)}

    # ---------------------------------------------------------- 1. pocket decomposition
    print("POCKET DECOMPOSITION (dice against the true ligand)")
    print(f"{'pocket fed to model':>22} {'dice':>8}")
    scores = {}
    with torch.no_grad():
        for label in ("correct", "shuffled", "zeros"):
            d = []
            for b, (prot, lig) in enumerate(batches):
                if label == "correct":
                    p_in = prot
                elif label == "shuffled":
                    idx = torch.randperm(prot.shape[0],
                                         generator=torch.Generator().manual_seed(args.seed + b))
                    p_in = prot[idx]
                else:
                    p_in = torch.zeros_like(prot)
                d.append(pooled_soft_dice(torch.sigmoid(net(p_in)).float(), lig))
            scores[label] = float(np.concatenate(d).mean())
            print(f"{label:>22} {scores[label]:>8.4f}")
    results["decomposition"] = scores
    print(f"\n  pocket-blind floor      : {scores['zeros']:.4f}")
    print(f"  + any pocket            : {scores['shuffled'] - scores['zeros']:+.4f}")
    print(f"  + the CORRECT pocket    : {scores['correct'] - scores['shuffled']:+.4f}")

    # ------------------------------------------------------ 2. stray-density profile
    thresholds = (0.01, 0.05, 0.2, 0.5)
    counts = np.zeros(len(thresholds))
    amp = 0.0
    npk = 0
    with torch.no_grad():
        for prot, lig in batches:
            p = torch.sigmoid(net(prot)).float()
            stray = p[lig <= 0.05]
            for i, thr in enumerate(thresholds):
                counts[i] += float((stray > thr).sum())
            amp += float(stray[stray > 0.01].mean()) * prot.shape[0]
            npk += prot.shape[0]
    counts /= npk
    print("\nSTRAY DENSITY per pocket (voxels where the true grid is empty)")
    print("  " + "  ".join(f">{t}: {c:,.0f}" for t, c in zip(thresholds, counts)))
    print(f"  mean amplitude of stray voxels: {amp / npk:.4f}")
    print("\n  many + low amplitude = diffuse haze (cheap under MSE, costly under on_target)")
    print("  few + high amplitude = confident hallucination (the reverse)")
    results["stray_per_pocket"] = {str(t): float(c) for t, c in zip(thresholds, counts)}
    results["stray_mean_amplitude"] = amp / npk

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
