"""Checkpoint x augmentation x (mass + decoder) fusion on likelihood ranking.

§20 ensembled the COMPOSITION readout across Poc2Mol checkpoints and augmentations
(0.7253 single -> 0.7578 both) and closed by naming the untested axis: the mass x decoder
fusion of §12d. Those two readouts correlate at only r = +0.024 within a pocket, against 0.715
among ensemble members, so fusion is the one axis likely to be genuinely additive on top of an
ensemble rather than averaging the same noise. This script runs all three axes together.

Each MEMBER is one end-to-end checkpoint, which is convenient rather than clever: an
`EndToEndPoc2Smiles` checkpoint carries both models, so a member's decoder is automatically
paired with the exact upstream it was trained against. Arms Z/H/I differ only in that
upstream, so a mixed member list ensembles over densities and decoders at once.

Per member x augmentation replicate it computes two (pocket x candidate) matrices:

    decoder      mean per-token log-likelihood of each candidate SMILES under the pocket's
                 predicted density -- the deployed score
    composition  negative distance between the density's per-channel summed occupancy
                 (calibrated to atom-count units) and each candidate's per-channel heavy-atom
                 counts. Parameter-free, pose-free, and spatially blind: it uses ONLY the
                 per-channel mass, discarding where in the box that mass sits

THREE TRAPS, all of which have already cost a run in this project:

1. ⚠️ AUGMENTATION IS A NO-OP ON A DETERMINISTIC VAL SET. This pipeline's val datasets are
   `rotate: false, translation: 0.0` by design, so replicates would be bitwise identical and
   the axis would silently measure nothing (§20). `--rotate` turns it on, and the script
   prints max|mass(aug0) - mass(aug1)| and shouts if it is zero.
2. ⚠️ NORMALISE MEMBERS THE WAY THE METRIC DOES. Every member matrix is COLUMN z-normalised
   before averaging, matching `likelihood_auc_znorm`. Row-standardising instead inflated every
   figure in an earlier §20 run by ~+0.005 relative to its own baseline. It also matters for
   fusion: decoder log-likelihoods and negative distances have wildly different dynamic
   ranges, so blending them raw would be a decoder-only readout wearing a blend's name.
3. ⚠️ SUBSET SELECTION OVER MEMBERS IS NOISE-FITTING (§20: greedy 0.7600, honest split-half
   0.7586, average-everything 0.7578, against a member sd of 0.0092). This script averages
   everything and does not offer a greedy option.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/fusion_ensemble.py \\
        --checkpoints a.ckpt,b.ckpt,c.ckpt --n_aug 4 --rotate --output fusion.json
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
from omegaconf import open_dict  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from src.utils.likelihood_eval import per_pocket_auc, znormalise_columns  # noqa: E402
from scripts.adhoc_analysis.poc2mol_scheme_discrimination import channel_counts  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", required=True,
                   help="comma-separated end-to-end checkpoints; each is one ensemble member")
    p.add_argument("--experiment", default="e2e_z_frozen",
                   help="experiment config supplying the architecture and the val panel")
    p.add_argument("--n_aug", type=int, default=4)
    p.add_argument("--rotate", action="store_true",
                   help="REQUIRED for the augmentation axis to be real; val is deterministic")
    p.add_argument("--translation", type=float, default=6.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    p.add_argument("--save_matrices", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("PROJECT_ROOT", root)
    checkpoints = [c for c in args.checkpoints.split(",") if c]

    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"experiment={args.experiment}",
            "data.num_workers=4",
            "paths.output_dir=/tmp/fusion", "paths.img_save_dir=/tmp/fusion/img",
        ])
    os.makedirs("/tmp/fusion/img", exist_ok=True)

    # Trap 1. Without this the replicates are identical and the augmentation axis is a lie.
    if args.rotate:
        with open_dict(cfg):
            cfg.data.val_datasets.roc_auc_plinder.complex_dataset.rotate = True
            cfg.data.val_datasets.roc_auc_plinder.complex_dataset.translation = args.translation
        print(f"augmentation ENABLED: rotate=True translation={args.translation}")
    else:
        print("augmentation DISABLED -- val is deterministic, the aug axis will be a NO-OP")

    channels = {int(k): list(v) for k, v in cfg.data.config.ligand_channels.items()}
    catch_all = bool(cfg.data.config.get("ligand_last_channel_is_catch_all", True))

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()[0]          # roc_auc_plinder, the decoy panel
    panel = list(datamodule.val_datasets["roc_auc_plinder"].decoy_smiles_list)

    model = hydra.utils.instantiate(cfg.model).to(args.device).eval()
    tokenizer = model.tokenizer

    # Candidate channel counts -- the composition readout's target, from SMILES alone.
    counts = {}
    for smiles in panel:
        mol = Chem.MolFromSmiles(smiles)
        counts[smiles] = channel_counts(mol, channels, catch_all) if mol is not None else None
    valid_columns = np.array([counts[s] is not None for s in panel])
    n_channels = len(channels)
    candidate_counts = np.stack([
        counts[s] if counts[s] is not None else np.zeros(n_channels) for s in panel
    ])
    print(f"panel: {len(panel)} candidates, {int(valid_columns.sum())} parse under RDKit")

    decoder_members, mass_members = {}, {}
    binder_order = None

    for ci, checkpoint in enumerate(checkpoints):
        state = torch.load(checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(state.get("state_dict", state), strict=False)
        missing = [k for k in incompatible.missing_keys
                   if not k.startswith(("val_", "train_", "test_"))]
        if missing:
            print(f"  ⚠️ ckpt{ci} missing {len(missing)} keys, e.g. {missing[:3]}")
        model.eval()

        for a in range(args.n_aug):
            # Same seed at a given aug index across members, so member-vs-member differences
            # are not contaminated by different random rotations.
            torch.manual_seed(4242 + a)
            np.random.seed(4242 + a)

            rows, mass_rows, binders = [], [], []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                    batch = datamodule.on_after_batch_transfer(batch)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pixel_values, info = model.build_pixel_values(batch, training=False)

                    # Spatially blind readout: collapse each channel to a single number.
                    density = info["predicted"].float()
                    mass_rows.append(density.sum(dim=(2, 3, 4)).cpu().numpy())

                    cand_ids = batch["candidate_tokens"]["input_ids"].to(args.device)
                    masked = cand_ids.clone()
                    masked[masked == tokenizer.pad_token_id] = -100
                    gather_idx = masked.clone()
                    gather_idx[gather_idx == -100] = 0
                    keep = masked != -100
                    seq_len = keep.sum(dim=1).clamp(min=1)

                    for i in range(pixel_values.size(0)):
                        repeated = pixel_values[i:i + 1].repeat(
                            cand_ids.size(0), *([1] * (pixel_values.dim() - 1)))
                        logits = model(repeated, labels=cand_ids).logits
                        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                        token_lp = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1) * keep
                        rows.append((token_lp.sum(1) / seq_len).cpu().numpy())
                    binders.extend(int(x) for x in batch["binder_indices"])

            decoder_members[(ci, a)] = np.stack(rows)
            mass_members[(ci, a)] = np.concatenate(mass_rows, axis=0)
            if binder_order is None:
                binder_order = binders
            elif binders != binder_order:
                raise RuntimeError("pocket order changed between replicates -- not alignable")
            print(f"  ckpt{ci} aug{a}: {decoder_members[(ci, a)].shape} decoder matrix", flush=True)

    # Trap 1, verified rather than assumed.
    if args.n_aug > 1:
        delta = float(np.abs(mass_members[(0, 0)] - mass_members[(0, 1)]).max())
        flag = "  ⚠️ ZERO -- augmentation is NOT varying, the axis is a no-op" if delta == 0 else ""
        print(f"\nmax |mass(aug0) - mass(aug1)| = {delta:.4f}{flag}")

    positive = np.zeros((len(binder_order), len(panel)), dtype=bool)
    for row, binder in enumerate(binder_order):
        target = panel[binder]
        positive[row] = np.array([s == target for s in panel])
    true_counts = np.stack([candidate_counts[b] for b in binder_order])

    def composition_matrix(mass):
        """Per-channel least-squares calibration to atom-count units, then negative distance.

        Distance, not cosine: dividing a vector by its sum is a scalar rescale and cosine is
        already scale-invariant, so a cosine readout is blind to SIZE -- which §12a measured
        as the larger half of the signal (size-only 0.668 vs composition-only 0.638).
        """
        denominator = (true_counts * true_counts).sum(axis=0)
        scale = np.where(denominator > 0,
                         (mass * true_counts).sum(axis=0) / np.maximum(denominator, 1e-9), 0.0)
        calibrated = mass / np.where(scale > 0, scale, 1.0)
        return -np.linalg.norm(calibrated[:, None, :] - candidate_counts[None, :, :], axis=2)

    keys = sorted(decoder_members)
    # Trap 2: column z-norm, the same transform the metric applies, before anything is averaged
    # or blended.
    decoder_z = {k: znormalise_columns(decoder_members[k]) for k in keys}
    composition_z = {k: znormalise_columns(composition_matrix(mass_members[k])) for k in keys}

    def auc(matrix):
        return per_pocket_auc(matrix, positive, valid_columns)[0]

    def mean_of(source, subset):
        return np.mean([source[k] for k in subset], axis=0)

    n_models, n_aug = len(checkpoints), args.n_aug
    results = {"n_pockets": len(binder_order), "n_candidates": len(panel),
               "rotate": bool(args.rotate), "n_models": n_models, "n_aug": n_aug,
               "checkpoints": checkpoints}

    print("\n" + "=" * 78)
    print("SINGLE MEMBERS (mean over all model x aug members)")
    print("=" * 78)
    for name, source in (("decoder", decoder_z), ("composition", composition_z)):
        singles = [auc(source[k]) for k in keys]
        results[f"single_{name}_mean"] = float(np.mean(singles))
        results[f"single_{name}_std"] = float(np.std(singles))
        print(f"  {name:<12} {np.mean(singles):.4f}  (sd {np.std(singles):.4f}, "
              f"range {np.min(singles):.4f}-{np.max(singles):.4f})")

    print("\n" + "=" * 78)
    print("ENSEMBLING, PER READOUT")
    print("=" * 78)
    ensembles = {}
    for name, source in (("decoder", decoder_z), ("composition", composition_z)):
        aug_only = float(np.mean([auc(mean_of(source, [(c, a) for a in range(n_aug)]))
                                  for c in range(n_models)]))
        model_only = float(np.mean([auc(mean_of(source, [(c, a) for c in range(n_models)]))
                                    for a in range(n_aug)]))
        both = mean_of(source, keys)
        ensembles[name] = both
        base = results[f"single_{name}_mean"]
        results[f"{name}_aug_ensemble"] = aug_only
        results[f"{name}_model_ensemble"] = model_only
        results[f"{name}_full_ensemble"] = float(auc(both))
        print(f"  {name}")
        print(f"    single                {base:.4f}")
        print(f"    + augmentation (x{n_aug})   {aug_only:.4f}   delta {aug_only - base:+.4f}")
        print(f"    + checkpoint (x{n_models})     {model_only:.4f}   delta {model_only - base:+.4f}")
        print(f"    + both                {auc(both):.4f}   delta {auc(both) - base:+.4f}")

    print("\n" + "=" * 78)
    print("FUSION -- blend of the two fully-ensembled readouts")
    print("  (both already column z-normed, so w is a genuine mixing weight)")
    print("=" * 78)
    blend = []
    for w in np.arange(0.0, 1.0001, 0.1):
        value = float(auc((1 - w) * ensembles["decoder"] + w * ensembles["composition"]))
        blend.append({"w_composition": round(float(w), 2), "auc": value})
        marker = ""
        if abs(w) < 1e-9:
            marker = "  <- decoder only"
        elif abs(w - 1.0) < 1e-9:
            marker = "  <- composition only"
        print(f"  w_composition = {w:.1f}   AUC {value:.4f}{marker}")
    results["fusion_curve"] = blend
    best = max(blend, key=lambda d: d["auc"])
    results["fusion_best"] = best
    print(f"\n  BEST: w_composition = {best['w_composition']}  ->  AUC {best['auc']:.4f}")

    # How independent are the two readouts really? §12d measured +0.024 within-pocket; if it
    # has drifted, the fusion gain should move with it.
    rows_corr = []
    for p in range(ensembles["decoder"].shape[0]):
        a, b = ensembles["decoder"][p], ensembles["composition"][p]
        if a.std() > 1e-9 and b.std() > 1e-9:
            rows_corr.append(float(np.corrcoef(a, b)[0, 1]))
    results["within_pocket_corr_decoder_composition"] = float(np.mean(rows_corr))
    print(f"\n  within-pocket corr(decoder, composition) = {np.mean(rows_corr):+.4f}"
          f"   (§12d measured +0.024)")

    if args.save_matrices:
        np.savez_compressed(
            args.save_matrices,
            decoder=np.stack([decoder_z[k] for k in keys]),
            composition=np.stack([composition_z[k] for k in keys]),
            member_keys=np.array([f"ckpt{c}_aug{a}" for c, a in keys]),
            checkpoints=np.array(checkpoints), positive=positive,
            valid_columns=valid_columns, panel=np.array(panel),
        )
        print(f"\nsaved member matrices -> {args.save_matrices}")

    if args.output:
        json.dump(results, open(args.output, "w"), indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
