"""Per-channel over/under-emission of a Poc2Mol checkpoint.

CLAUDE.md §12b estimated over-emission by comparing predicted mass/atom against the mass an
ideal vdW sphere would deposit (48.8 for carbon at 0.75 A voxels). That reference is only
approximate: max-aggregation and atom overlap mean the true deposited mass per atom is not the
analytic single-sphere value. Here the reference is the TRUE VOXELISED LIGAND on the same
grid, so the comparison is exact and needs no idealisation.

Four readouts per channel:

  ratio          total predicted mass / total true mass. >1 over-emission, <1 under.
  empty_mass     mean predicted mass in samples whose TRUE channel is completely empty.
                 This is the decisive one. An all-zero target channel gives Dice a numerator
                 that is identically zero, hence NO gradient (§3c) -- the only pressure to
                 keep it empty is BCE, diluted by 1/(N*C*D*H*W). Mass sitting here is mass the
                 loss is nearly blind to, and it is the mechanism §12b inferred but did not
                 measure directly.
  slope/intercept  OLS of predicted mass on true mass. A large intercept with a small slope
                 is a constant smear: the channel emits regardless of what the ligand holds.
  r              correlation, for reference against §12b.

Usage:
    python scripts/adhoc_analysis/poc2mol_channel_emission.py \
        --run_dir logs/poc2mol_v2_ch11/runs/... --checkpoint <ckpt> [--output x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in ckpt:
        model.model.load_state_dict({k.replace("model.", "", 1): v
                                     for k, v in ckpt["state_dict"].items()
                                     if k.startswith("model.")})
    else:
        model.load_state_dict(ckpt)
    model.eval().to(args.device)

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    names = list(cfg.data.config.ligand_channel_names)

    pred_mass, true_mass, overlap_mass = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            batch = dm.on_after_batch_transfer(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["protein"])
            pred = out["predicted_ligand_voxels"].float()
            true = batch["ligand"].float()
            pred_mass.append(pred.sum(dim=(2, 3, 4)).cpu().numpy())
            true_mass.append(true.sum(dim=(2, 3, 4)).cpu().numpy())
            # predicted mass that lands where the true channel has real occupancy
            overlap_mass.append((pred * (true > 0.05)).sum(dim=(2, 3, 4)).cpu().numpy())

    pred_mass = np.concatenate(pred_mass, 0)
    true_mass = np.concatenate(true_mass, 0)
    overlap_mass = np.concatenate(overlap_mass, 0)
    n, C = pred_mass.shape
    print(f"\n{n} samples, {C} ligand channels\n")

    hdr = (f"{'channel':<20}{'ratio':>8}{'empty%':>8}{'empty_mass':>12}"
           f"{'occ_mass':>10}{'slope':>8}{'intcpt':>9}{'r':>7}{'on_target':>11}")
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for c in range(C):
        pm, tm = pred_mass[:, c], true_mass[:, c]
        empty = tm <= 0
        ratio = pm.sum() / tm.sum() if tm.sum() > 0 else float("nan")
        empty_mass = float(pm[empty].mean()) if empty.any() else 0.0
        occ_mass = float(pm[~empty].mean()) if (~empty).any() else 0.0
        if tm.std() > 1e-9:
            slope, intercept = np.polyfit(tm, pm, 1)
            r = float(np.corrcoef(pm, tm)[0, 1])
        else:
            slope = intercept = r = float("nan")
        # fraction of predicted mass that lands on real ligand density
        on_target = float(overlap_mass[:, c].sum() / pm.sum()) if pm.sum() > 0 else float("nan")
        rows[names[c] if c < len(names) else f"ch{c}"] = {
            "ratio": float(ratio), "empty_fraction": float(empty.mean()),
            "empty_mass": empty_mass, "occupied_mass": occ_mass,
            "slope": float(slope), "intercept": float(intercept), "r": r,
            "on_target_fraction": on_target,
        }
        nm = names[c] if c < len(names) else f"ch{c}"
        print(f"{nm:<20}{ratio:>8.2f}{100*empty.mean():>7.1f}%{empty_mass:>12.1f}"
              f"{occ_mass:>10.1f}{slope:>8.2f}{intercept:>9.1f}{r:>7.3f}{on_target:>10.1%}")

    print("\nratio      >1 = over-emitting that channel overall")
    print("empty%     how often the TRUE channel is completely empty")
    print("empty_mass predicted mass emitted into a channel the ligand leaves EMPTY.")
    print("           Dice has zero gradient there (§3c), so this is mass the loss barely sees.")
    print("intcpt     predicted mass at true mass 0, from OLS -- a constant smear")
    print("on_target  share of predicted mass landing on real ligand density (>0.05)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"n_samples": int(n), "channels": rows}, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
