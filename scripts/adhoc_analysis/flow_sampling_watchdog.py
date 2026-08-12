"""Does this flow checkpoint still SAMPLE on-distribution? Ask without touching the run.

The failure this exists for: the training loss falls nicely while sampling walks off the
data manifold and returns a nonsensical grid. `Poc2MolFlow` logs the watchdog metrics during
its own validation, but a run launched without them -- or one you simply do not want to
restart -- can be interrogated from its checkpoints instead. `last.ckpt` is rewritten at
every validation, so polling it gives the same signal a few minutes behind.

Works for both stages, because it reads the CHANNEL LAYOUT FROM THE CHECKPOINT rather than
from a config: an unconditional (ZINC) model has protein_channels = 0 and its data has no
protein channels either, a pocket-conditioned one has both.

What to look at, in order:

    traj_rms_max   ~1 is healthy WHATEVER the model has learned -- every point of the
                   probability path has RMS ~1 in model space. >2 drifting, >10 diverging.
    out_of_range   fraction of voxels outside [-0.1, 1.1] occupancy BEFORE clamping.
    occupied_frac  against the data's own value on the same batch.
    restore_dice   sampling started from the TRUE path at t=0.5. If this is healthy while
                   sample_dice is ~0, the velocity field is fine and the TRAJECTORY is the
                   problem: more steps, less guidance, or model.sample_clamp.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/flow_sampling_watchdog.py \\
        --ckpt logs/flow_zinc_pretrain/runs/<...>/checkpoints/last.ckpt \\
        --data zinc [--watch 600]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import hydra  # noqa: E402

from src.data.common.voxelization.batched import BatchedVoxelizer  # noqa: E402
from src.models.poc2mol_flow import Poc2MolFlow, pooled_soft_dice  # noqa: E402

EXPERIMENTS = {
    "zinc": "flow_zinc_pretrain",
    "hiqbind": "flow_poc2mol_hiqbind",
}


def load_flow_checkpoint(path: str, device, retries: int = 6, delay: int = 20):
    """Load a checkpoint a LIVE run may be rewriting underneath us.

    `last.ckpt` is 1.9 GB and takes seconds to write, so reading it at the wrong moment
    gives "PytorchStreamReader failed reading zip archive: failed finding central
    directory". Waiting for the size to settle and then working from a copy makes polling
    a running job safe; a genuinely corrupt file still raises after the retries.
    """
    tmp = os.path.join(tempfile.gettempdir(), f"flow_ckpt_{os.getpid()}.ckpt")
    last_error = None
    for attempt in range(retries):
        try:
            size = os.path.getsize(path)
            time.sleep(2)
            if os.path.getsize(path) != size or size == 0:
                raise RuntimeError("checkpoint is still being written")
            shutil.copy2(path, tmp)
            step = int(torch.load(tmp, map_location="cpu").get("global_step", -1))
            model = Poc2MolFlow.load_from_checkpoint(tmp, map_location="cpu").to(device).eval()
            return model, step
        except Exception as exc:  # noqa: BLE001 -- report the last one if all attempts fail
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(f"could not load {path} after {retries} attempts: {last_error}")


def build_batches(kind: str, batch_size: int, n_batches: int, device):
    """Voxelised validation batches, in the same deterministic order the run sees."""
    root = os.path.dirname(os.path.dirname(_HERE))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"experiment={EXPERIMENTS[kind]}", "data.num_workers=4",
            f"data.config.batch_size={batch_size}",
            "paths.output_dir=/tmp/flow_watchdog",
            "paths.img_save_dir=/tmp/flow_watchdog/img",
        ])
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    n_protein = datamodule.batch_builder.n_protein_channels
    voxel_config = datamodule.config
    voxelizer = BatchedVoxelizer(
        voxel_config,
        cutoff_ratio=voxel_config.get("voxel_cutoff_ratio", 2.0),
        aggregation=voxel_config.get("voxel_aggregation", "max"),
        radius_scale=voxel_config.get("voxel_radius_scale", 1.0),
    ).to(device)

    # The ligand-only datamodule reports n_protein_channels for the ZEROS it pads onto the
    # batch; the grid it voxelises carries ligand channels only. Splitting the grid at
    # n_protein there would silently hand four ligand channels over as "protein".
    has_protein_in_grid = bool(voxel_config.get("has_protein", True))

    # NOTE: `batch_size` sets data.config.batch_size, which controls the TRAIN loader.
    # ComplexDataModule's val loader uses data.val_batch_size (32 in the flow experiments),
    # so the number of pockets is n_batches * val_batch_size, NOT n_batches * batch_size.
    # Getting this wrong produced a 128-pocket baseline compared against 256-pocket flow
    # numbers on 2026-08-11, which understated the gap by 0.028.
    batches = []
    loader = datamodule.val_dataloader()
    loader = loader[0] if isinstance(loader, (list, tuple)) else loader
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        grid = voxelizer(
            batch["atom_xyz"].to(device), batch["atom_radius"].to(device),
            batch["atom_slot"].to(device), int(batch["batch_size"]),
            int(batch["n_channels"]),
        )
        if has_protein_in_grid and n_protein:
            protein, ligand = grid[:, :n_protein].float(), grid[:, n_protein:].float()
        else:
            ligand = grid.float()
            protein = (ligand.new_zeros(ligand.shape[0], n_protein, *ligand.shape[2:])
                       if n_protein else None)
        batches.append((protein, ligand))
    return batches


def probe(model, batches, n_steps: int, guidance: float, restore_t: float, seed: int):
    device = model.device
    totals = {k: 0.0 for k in ("traj_rms_max", "rms_ratio", "out_of_range", "max_abs_occ",
                               "occupied_frac", "data_occupied_frac", "sample_dice",
                               "restore_dice")}
    for b, (protein, ligand) in enumerate(batches):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + b)
        x0 = torch.randn(
            (ligand.shape[0], model.n_ligand_channels, *ligand.shape[2:]),
            device=device, dtype=torch.float32, generator=generator,
        )
        sample, stats = model.sample(protein=protein, x0=x0, n_steps=n_steps,
                                     guidance_scale=guidance, return_stats=True)
        x1_true = model.to_model_space(ligand)
        restored = model.sample(
            protein=protein, x0=(1.0 - restore_t) * x0 + restore_t * x1_true,
            t_start=restore_t, n_steps=n_steps, guidance_scale=guidance,
        )
        true_rms = float(x1_true.pow(2).mean().sqrt().clamp(min=1e-6))

        totals["traj_rms_max"] += stats["traj_rms_max"]
        totals["rms_ratio"] += stats["final_rms"] / true_rms
        totals["out_of_range"] += stats["out_of_range"]
        totals["max_abs_occ"] += stats["max_abs_occ"]
        totals["occupied_frac"] += float((sample > 0.5).float().mean())
        totals["data_occupied_frac"] += float((ligand > 0.5).float().mean())
        totals["sample_dice"] += float(pooled_soft_dice(sample, ligand).mean())
        totals["restore_dice"] += float(pooled_soft_dice(restored, ligand).mean())
    return {k: v / len(batches) for k, v in totals.items()}


def verdict(r: dict, pair_dice: float = 0.0) -> str:
    """One line, ordered by how badly you would want to know it.

    `pair_dice` is what two DIFFERENT true molecules score against each other, which is the
    level an unconditional sampler should reach: it says "a real molecule, just not that
    one". Below it with the right total mass means the density is there but not yet
    arranged like a molecule -- a training-progress signal, not a divergence.
    """
    if r["traj_rms_max"] > 10:
        return "DIVERGING -- the ODE state is running away"
    if r["traj_rms_max"] > 2:
        return "DRIFTING -- trajectory scale is above the healthy band"
    if r["out_of_range"] > 0.1:
        return "off-distribution -- many voxels outside plausible occupancy"

    mass_ratio = r["occupied_frac"] / max(r["data_occupied_frac"], 1e-9)
    if not 0.4 < mass_ratio < 2.5:
        return f"on-distribution in scale, but emits {mass_ratio:.1f}x the data's occupied mass"
    if r["restore_dice"] > 3 * max(r["sample_dice"], 1e-6) and r["sample_dice"] < 0.05:
        return "field OK, full trajectory weak (early training, or too few steps)"
    if pair_dice and r["sample_dice"] < 0.5 * pair_dice:
        return f"healthy; mass right, structure still forming (dice {r['sample_dice']:.3f} vs {pair_dice:.3f} target)"
    return "on-distribution"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="zinc", choices=sorted(EXPERIMENTS))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--restore_t", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", type=int, default=0,
                        help="seconds between re-checks; 0 = once. Re-reads the checkpoint "
                             "each time, so point it at last.ckpt of a live run.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = build_batches(args.data, args.batch_size, args.n_batches, device)
    print(f"{len(batches)} batches of {args.batch_size} from the {args.data} val split")

    # Calibrates the `dice` column, which is otherwise uninterpretable for an UNCONDITIONAL
    # model: it scores a sample against one specific held-out molecule, and a perfectly good
    # sample is simply a different molecule. This is what "a real molecule, but not that
    # one" scores -- the ceiling an unconditional model should approach, and the floor a
    # pocket-conditioned one must clearly beat.
    pair_dice_mean, pair_dice, n = 0.0, 0.0, 0
    for _, ligand in batches:
        if ligand.shape[0] < 2:
            continue
        shifted = torch.roll(ligand, 1, dims=0)
        pair_dice += float(pooled_soft_dice(ligand, shifted).mean())
        n += 1
    if n:
        pair_dice_mean = pair_dice / n
        print(f"reference: dice between two DIFFERENT true molecules = {pair_dice_mean:.4f}")
        print("  an unconditional model should approach that; a conditioned one must beat it\n")
    else:
        print()

    header = (f"{'when':>8} {'step':>7} {'trajRMS':>8} {'rmsRatio':>9} {'oor':>6} {'maxOcc':>7} "
              f"{'occFrac':>8} {'dataOcc':>8} {'dice':>7} {'restore':>8}  verdict")
    print(header)
    print("-" * len(header))

    while True:
        model, step = load_flow_checkpoint(args.ckpt, device)
        with torch.no_grad():
            r = probe(model, batches, args.steps, args.guidance, args.restore_t, args.seed)
        print(f"{time.strftime('%H:%M:%S'):>8} {step:>7} {r['traj_rms_max']:>8.3f} "
              f"{r['rms_ratio']:>9.3f} {r['out_of_range']:>6.3f} {r['max_abs_occ']:>7.2f} "
              f"{r['occupied_frac']:>8.4f} {r['data_occupied_frac']:>8.4f} "
              f"{r['sample_dice']:>7.4f} {r['restore_dice']:>8.4f}  {verdict(r, pair_dice_mean)}", flush=True)
        del model
        torch.cuda.empty_cache()
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
