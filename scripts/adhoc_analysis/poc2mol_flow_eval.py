"""Evaluate the generative (flow-matching) Poc2Mol against the regression yardstick.

The regression model's number to beat is **dice(pred, true) = 0.5027**, measured on the
HiQBind v2 validation split this script also uses. report.md 12g's widely-quoted **0.596**
is a DIFFERENT measurement -- a different checkpoint on 104 PLINDER-panel pockets -- and is
not comparable to anything scored on HiQBind val, though its rotation-ladder calibration
(~27 deg equivalent) still describes that checkpoint on that set. This script uses the same
pooled soft Dice, imported from ``poc2mol_rotation_control.py`` rather than reimplemented,
so the numbers cannot drift apart.

A generative model has three axes the regression model does not, and all three are swept
here because none of them can be argued from first principles:

* **guidance scale** -- classifier-free guidance sharpens the pocket dependence. Above some
  value it starts producing over-saturated density; `emission_ratio` is what reveals that.
* **ODE steps** -- more steps cost linearly and stop paying at some point.
* **draws per pocket** -- the model defines a *distribution*, so mean Dice and best-of-N
  Dice answer different questions. Mean Dice is the honest single-shot number and the one
  comparable to 0.596; best-of-N is what a downstream pipeline that screens candidates
  would actually see.

The rotation control is then run at the chosen setting. It is the test that distinguishes
"the density is sharp" from "the density is in the right place": a sample is scored against
the true pose and against rotated copies of the same molecule, so composition and size are
identical by construction and only the pose differs.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/poc2mol_flow_eval.py \
        --ckpt logs/flow_poc2mol_hiqbind/runs/<...>/checkpoints/<...>.ckpt \
        --n_batches 8 --guidance 1.0 2.0 3.0 --steps 25 50 --n_samples 4
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
sys.path.insert(0, _HERE)

import hydra  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

# Same Dice and the same rotation helper the regression yardstick uses. Importing rather
# than copying is deliberate: a divergent Dice definition would silently invalidate every
# comparison against 0.596.
from poc2mol_rotation_control import rotate_ligand, soft_dice  # noqa: E402

from src.data.common.voxelization.batched import BatchedVoxelizer  # noqa: E402
from src.models.poc2mol_flow import Poc2MolFlow  # noqa: E402

# Measured 2026-08-11 on the HiQBind v2 val split (128 pockets, sem 0.009) with THIS Dice,
# from checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt. Use this, not the 0.596 in
# report.md 12g: that was a different checkpoint on 104 PLINDER-panel pockets, so it is not
# comparable to anything scored on HiQBind val.
REGRESSION_YARDSTICK = 0.5027
ROTATION_ANGLES = [15, 30, 60, 90, 180, "random"]


def emission_stats(predicted: torch.Tensor, true: torch.Tensor) -> dict:
    """The three non-floor-bound diagnostics from report.md 18b, on sampled density."""
    total = predicted.sum().clamp(min=1e-6)
    occupied = true > 0.05
    empty_channel = true.sum(dim=(2, 3, 4)) <= 0
    return {
        "emission_ratio": float(total / true.sum().clamp(min=1e-6)),
        "on_target": float((predicted * occupied).sum() / total),
        "empty_frac": float((predicted.sum(dim=(2, 3, 4)) * empty_channel).sum() / total),
    }


def load_model(ckpt_path: str, device) -> Poc2MolFlow:
    """Rebuild the model from the checkpoint's own hyperparameters.

    Scheme-agnostic on purpose: the channel counts come from the checkpoint rather than
    from whatever config happens to be current, so an 11-channel checkpoint cannot be
    silently evaluated under a 9-channel config.
    """
    model = Poc2MolFlow.load_from_checkpoint(ckpt_path, map_location="cpu")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="Poc2MolFlow checkpoint")
    parser.add_argument("--experiment", default="flow_poc2mol_hiqbind",
                        help="experiment config supplying the DATA (model comes from --ckpt)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=8,
                        help="-1 for the whole split")
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--steps", type=int, nargs="+", default=[50])
    parser.add_argument("--sampler", default="heun", choices=["euler", "heun"])
    parser.add_argument("--n_samples", type=int, default=1,
                        help="draws per pocket; >1 also reports best-of-N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_rotation_control", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.dirname(os.path.dirname(_HERE))
    os.environ.setdefault("PROJECT_ROOT", root)

    model = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt}")
    print(f"  ligand channels  = {model.n_ligand_channels}")
    print(f"  protein channels = {model.n_protein_channels}")
    print(f"  cond dropout     = {model.cond_dropout_prob} "
          f"({'guidance available' if model.cond_dropout_prob > 0 else 'NO unconditional branch -- guidance will be meaningless'})")

    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"experiment={args.experiment}", "data.num_workers=4",
            f"data.config.batch_size={args.batch_size}",
            "paths.output_dir=/tmp/flow_eval", "paths.img_save_dir=/tmp/flow_eval/img",
        ])

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit" if args.split == "val" else "test")
    n_protein = datamodule.batch_builder.n_protein_channels
    voxel_config = datamodule.config
    voxelizer = BatchedVoxelizer(
        voxel_config,
        cutoff_ratio=voxel_config.get("voxel_cutoff_ratio", 2.0),
        aggregation=voxel_config.get("voxel_aggregation", "max"),
        radius_scale=voxel_config.get("voxel_radius_scale", 1.0),
    ).to(device)

    loader = (datamodule.val_dataloader() if args.split == "val"
              else datamodule.test_dataloader())

    # Grids are built once per batch and reused across every (guidance, steps) setting, so
    # the settings are compared on identical pockets AND identical noise draws.
    batches = []
    for i, batch in enumerate(loader):
        if args.n_batches >= 0 and i >= args.n_batches:
            break
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()})
    print(f"{len(batches)} batches of up to {args.batch_size} pockets\n")

    results = {"ckpt": args.ckpt, "split": args.split, "sampler": args.sampler,
               "regression_yardstick": REGRESSION_YARDSTICK, "settings": {}}

    header = (f"{'guidance':>9} {'steps':>6} {'dice':>8} {'sem':>7} {'best-of-N':>10} "
              f"{'ratio':>8} {'on_tgt':>8} {'empty':>8} {'trajRMS':>8} {'oor':>7}")
    print(header)
    print("-" * len(header))
    print("trajRMS ~1 = healthy trajectory (>2 drifting, >10 diverging); "
          "oor = fraction of voxels outside [-0.1, 1.1] before clamping")

    best = None
    for steps in args.steps:
        for w in args.guidance:
            dice_mean, dice_best, stats = [], [], []
            for b, batch in enumerate(batches):
                grid = voxelizer(batch["atom_xyz"], batch["atom_radius"],
                                 batch["atom_slot"], int(batch["batch_size"]),
                                 int(batch["n_channels"]))
                protein = grid[:, :n_protein].float()
                true = grid[:, n_protein:].float()

                per_draw = []
                for draw in range(args.n_samples):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(args.seed + 1000 * draw + b)
                    with torch.no_grad():
                        predicted, traj = model.sample(
                            protein=protein, n_steps=steps, guidance_scale=w,
                            sampler=args.sampler, generator=generator, return_stats=True,
                        )
                    per_draw.append(soft_dice(predicted, true))
                    if draw == 0:
                        entry = emission_stats(predicted, true)
                        # Divergence watchdog, same quantities the training run logs.
                        # traj_rms_max ~1 is healthy whatever the model has learned; it is
                        # what separates "sampler ran away" from "not trained yet".
                        entry.update({k: traj[k] for k in ("traj_rms_max", "out_of_range")})
                        stats.append(entry)
                per_draw = np.stack(per_draw)            # (n_samples, n_pockets)
                dice_mean.append(per_draw.mean(axis=0))
                dice_best.append(per_draw.max(axis=0))

            dice_mean = np.concatenate(dice_mean)
            dice_best = np.concatenate(dice_best)
            aggregated = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            sem = float(dice_mean.std(ddof=1) / np.sqrt(dice_mean.size)) if dice_mean.size > 1 else 0.0

            entry = {"guidance": w, "steps": steps, "n_pockets": int(dice_mean.size),
                     "dice_mean": float(dice_mean.mean()), "dice_sem": sem,
                     "dice_best_of_n": float(dice_best.mean()),
                     "n_samples": args.n_samples, **aggregated}
            results["settings"][f"w{w}_s{steps}"] = entry
            print(f"{w:>9.2f} {steps:>6d} {entry['dice_mean']:>8.4f} {sem:>7.4f} "
                  f"{entry['dice_best_of_n']:>10.4f} {aggregated['emission_ratio']:>8.3f} "
                  f"{aggregated['on_target']:>8.3f} {aggregated['empty_frac']:>8.3f} "
                  f"{aggregated['traj_rms_max']:>8.2f} {aggregated['out_of_range']:>7.3f}")

            if best is None or entry["dice_mean"] > best["dice_mean"]:
                best = entry

    print(f"\nbest single-shot dice = {best['dice_mean']:.4f} at guidance {best['guidance']}, "
          f"{best['steps']} steps")
    print(f"regression Poc2Mol     = {REGRESSION_YARDSTICK:.4f}  "
          f"({'BEATEN' if best['dice_mean'] > REGRESSION_YARDSTICK else 'not beaten'})")
    print("emission_ratio 1.0 is calibrated; the regression model runs 1.33-10.4x per channel.")
    results["best"] = best

    # ---------------------------------------------------------------- rotation control
    if not args.skip_rotation_control:
        print("\nrotation control at the best setting -- is the density in the right PLACE?")
        rng = np.random.default_rng(args.seed)
        pred_true, collected = [], {a: {"pred_rot": [], "true_rot": []} for a in ROTATION_ANGLES}

        for b, batch in enumerate(batches):
            atom_xyz, atom_radius, atom_slot = (batch["atom_xyz"], batch["atom_radius"],
                                                batch["atom_slot"])
            n_samples = int(batch["batch_size"])
            n_channels = int(batch["n_channels"])
            grid = voxelizer(atom_xyz, atom_radius, atom_slot, n_samples, n_channels)
            protein, true = grid[:, :n_protein].float(), grid[:, n_protein:].float()

            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + b)
            with torch.no_grad():
                predicted = model.sample(
                    protein=protein, n_steps=best["steps"], guidance_scale=best["guidance"],
                    sampler=args.sampler, generator=generator,
                )
            pred_true.append(soft_dice(predicted, true))

            for angle in ROTATION_ANGLES:
                if angle == "random":
                    mats = Rotation.random(n_samples,
                                           random_state=int(rng.integers(1 << 30))).as_matrix()
                else:
                    axes = rng.normal(size=(n_samples, 3))
                    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
                    mats = Rotation.from_rotvec(axes * np.deg2rad(angle)).as_matrix()
                rotated = voxelizer(
                    rotate_ligand(atom_xyz, atom_slot, n_channels, n_protein, mats),
                    atom_radius, atom_slot, n_samples, n_channels,
                )[:, n_protein:].float()
                collected[angle]["pred_rot"].append(soft_dice(predicted, rotated))
                collected[angle]["true_rot"].append(soft_dice(true, rotated))

        pred_true = np.concatenate(pred_true)
        print(f"\n{'angle':>8} {'dice(sample,rot)':>17} {'drop':>8} {'dice(true,rot)':>15} {'win rate':>10}")
        per_angle = {}
        for angle in ROTATION_ANGLES:
            pred_rot = np.concatenate(collected[angle]["pred_rot"])
            true_rot = np.concatenate(collected[angle]["true_rot"])
            win = float((pred_true > pred_rot).mean())
            per_angle[str(angle)] = {"dice_pred_rot": float(pred_rot.mean()),
                                     "drop": float(pred_true.mean() - pred_rot.mean()),
                                     "dice_true_rot": float(true_rot.mean()),
                                     "win_rate": win}
            print(f"{str(angle):>8} {pred_rot.mean():>17.4f} "
                  f"{pred_true.mean() - pred_rot.mean():>8.4f} {true_rot.mean():>15.4f} {win:>10.3f}")
        results["rotation_control"] = {"dice_pred_vs_true": float(pred_true.mean()),
                                       "per_angle": per_angle}
        print("\nThe regression model loses this test past 60 deg (0.347 vs 0.334, win rate < 0.5):")
        print("its blurred density overlaps a WRONG pose better than the true one. A generative")
        print("model that has learned the pose should keep the win rate near 1.0 at every angle.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
