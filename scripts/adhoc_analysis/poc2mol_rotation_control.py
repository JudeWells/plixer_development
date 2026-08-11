"""Does Poc2Mol's predicted density localise atoms, or only get the composition right?

The problem with testing spatial fidelity against decoy molecules is that decoys have no pose in
this pocket's frame, so any comparison confounds "wrong molecule" with "wrong pose". This test
sidesteps that completely: compare the prediction against the TRUE ligand in its true pose and
against a ROTATED COPY OF THE SAME MOLECULE.

Composition and size are then identical by construction -- same atoms, same counts, same channel
occupancy totals -- so any difference in agreement is purely spatial. No docking, no decoys.

Rotation is applied to the ligand atom coordinates before voxelisation, about the box centre
(the coordinate frame is already centred there, so a rotation about the origin is exactly that).
Protein atoms are left alone; only ligand channels are scored.

Reported per rotation angle:
    dice(pred, true)     -- the reference, identical for every angle
    dice(pred, rotated)  -- falls with angle iff the prediction is spatially specific
    dice(true, rotated)  -- how much the rotation actually changed the grid. Without this the
                            test has no scale: a near-spherical ligand barely changes under
                            rotation, so a small drop would be unremarkable.
    win rate             -- fraction of pockets where dice(pred,true) > dice(pred,rotated),
                            a paired sign test that does not care about absolute Dice levels.

Usage:
    python scripts/adhoc_analysis/poc2mol_rotation_control.py [--output x.json]
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
from scipy.spatial.transform import Rotation  # noqa: E402

from src.data.common.voxelization.batched import BatchedVoxelizer  # noqa: E402

ANGLES = [15, 30, 45, 60, 90, 120, 180, "random"]


def soft_dice(a, b, eps=1e-8):
    """Per-sample soft Dice over the ligand channels, pooled across channels.

    Pooled rather than per-channel-averaged on purpose: CLAUDE.md 3c showed that averaging over
    channels lets the many empty ones dominate (an all-zero target channel contributes a fixed
    1.0 to the loss regardless of prediction), which would swamp the spatial effect being
    measured here.
    """
    dims = tuple(range(1, a.dim()))
    num = 2.0 * (a * b).sum(dim=dims)
    den = (a * a).sum(dim=dims) + (b * b).sum(dim=dims)
    return (num / (den + eps)).cpu().numpy()


def rotate_ligand(atom_xyz, atom_slot, n_channels, n_protein, rotations):
    """Rotate each sample's LIGAND atoms about the box centre; leave protein atoms alone."""
    xyz = atom_xyz.clone()
    sample = torch.div(atom_slot, n_channels, rounding_mode="floor")
    channel = atom_slot % n_channels
    is_ligand = channel >= n_protein
    for s, matrix in enumerate(rotations):
        mask = is_ligand & (sample == s)
        if not bool(mask.any()):
            continue
        R = torch.as_tensor(matrix, dtype=xyz.dtype, device=xyz.device)
        xyz[mask] = xyz[mask] @ R.T
    return xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            "experiment=exp1_s3_protein", "data.num_workers=4",
            f"data.config.batch_size={args.batch_size}",
            f"data.config.val_batch_size={args.batch_size}",
            "data.config.target_samples_per_batch=32",
            "paths.output_dir=/tmp/rot", "paths.img_save_dir=/tmp/rot/img",
        ])
    os.makedirs("/tmp/rot/img", exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    n_protein = datamodule.voxel_builder.n_protein_channels
    voxel_config = datamodule.config
    voxelizer = None

    rng = np.random.default_rng(args.seed)
    collected = {a: {"pred_rot": [], "true_rot": []} for a in ANGLES}
    pred_true = []

    loader = datamodule.val_dataloader()[0]
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            atom_xyz = batch["atom_xyz"].clone()
            atom_radius = batch["atom_radius"].clone()
            atom_slot = batch["atom_slot"].clone()
            n_samples = int(batch["batch_size"])
            n_channels = int(batch["n_channels"])

            if voxelizer is None:
                voxelizer = BatchedVoxelizer(
                    voxel_config,
                    cutoff_ratio=voxel_config.get("voxel_cutoff_ratio", 2.0),
                    aggregation=voxel_config.get("voxel_aggregation", "max"),
                    radius_scale=voxel_config.get("voxel_radius_scale", 1.0),
                ).to(atom_xyz.device)

            # Prediction. training=False forces fraction 1.0 -> Poc2Mol's output, not the
            # ground truth. This consumes the atom_* keys, hence the copies above.
            out = datamodule.voxel_builder(dict(batch), apply_quality_filter=False,
                                           training=False, global_step=0)
            predicted = out["pixel_values"][:, :n_channels - n_protein].float()

            true_grid = voxelizer(atom_xyz, atom_radius, atom_slot,
                                  n_samples, n_channels)[:, n_protein:].float()
            pred_true.append(soft_dice(predicted, true_grid))

            for angle in ANGLES:
                if angle == "random":
                    mats = Rotation.random(n_samples, random_state=int(rng.integers(1 << 30))).as_matrix()
                else:
                    axes = rng.normal(size=(n_samples, 3))
                    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
                    mats = Rotation.from_rotvec(axes * np.deg2rad(angle)).as_matrix()
                rotated_xyz = rotate_ligand(atom_xyz, atom_slot, n_channels, n_protein, mats)
                rotated_grid = voxelizer(rotated_xyz, atom_radius, atom_slot,
                                         n_samples, n_channels)[:, n_protein:].float()
                collected[angle]["pred_rot"].append(soft_dice(predicted, rotated_grid))
                collected[angle]["true_rot"].append(soft_dice(true_grid, rotated_grid))

    pred_true = np.concatenate(pred_true)
    results = {"n_pockets": int(pred_true.size),
               "dice_pred_vs_true": float(pred_true.mean())}
    print(f"\nn pockets = {pred_true.size}")
    print(f"dice(pred, TRUE pose) = {pred_true.mean():.4f}   (reference, same for every row)\n")
    print(f"{'angle':>8} {'dice(pred,rot)':>15} {'drop':>8} {'dice(true,rot)':>15} {'win rate':>10}")

    per_angle = {}
    for angle in ANGLES:
        pred_rot = np.concatenate(collected[angle]["pred_rot"])
        true_rot = np.concatenate(collected[angle]["true_rot"])
        win = float((pred_true > pred_rot).mean())
        per_angle[str(angle)] = {"dice_pred_rot": float(pred_rot.mean()),
                                 "drop": float(pred_true.mean() - pred_rot.mean()),
                                 "dice_true_rot": float(true_rot.mean()),
                                 "win_rate": win}
        print(f"{str(angle):>8} {pred_rot.mean():>15.4f} {pred_true.mean()-pred_rot.mean():>8.4f} "
              f"{true_rot.mean():>15.4f} {win:>10.3f}")
    results["per_angle"] = per_angle

    print("\nwin rate = fraction of pockets where the TRUE pose beats the rotated copy.")
    print("0.5 means the prediction carries no pose information; 1.0 means it is fully specific.")
    print("dice(true,rot) is the scale: if it stays near 1.0 the rotation barely changed the")
    print("grid, and a small drop in dice(pred,rot) would not be evidence of anything.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
