"""Draw what the flow model actually generates: sampled ligand voxel grids, side by side.

Numbers say whether a sample is on-distribution; only a picture says whether it looks like a
molecule. This renders, for each validation example, the TRUE ligand next to several
independent draws from the model, on a shared set of axes so shapes are directly comparable.

Two modes, because the two stages answer different questions:

    --data hiqbind   pocket-conditioned. Each row is one pocket, rendered as a translucent
                     grey shell (the same idea as show_3d_voxel_lig_in_protein), with the
                     true ligand in column 0 and independent samples beside it. The question
                     is whether the samples land in the pocket, in the right place.
    --data zinc      unconditional. Column 0 is a REAL molecule for reference; the rest are
                     samples drawn from noise. There is no pairing to expect -- the question
                     is only whether they look like molecules.

Colours follow LIGAND_CHANNELS_V2 (11 channels). Voxels are thresholded at --threshold, the
same 0.5 the repo's other voxel plots use, so an empty panel is a real finding rather than a
rendering choice; the per-panel voxel count is printed in the title either way.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/flow_visualise_samples.py \\
        --ckpt <path> --data zinc --n_examples 4 --n_samples 3 --out /tmp/zinc_samples.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from flow_sampling_watchdog import build_batches, load_flow_checkpoint  # noqa: E402

from src.data.common.voxelization.config import LIGAND_CHANNEL_NAMES_V2  # noqa: E402
from src.models.poc2mol_flow import Poc2MolFlow, pooled_soft_dice  # noqa: E402

# One colour per LIGAND_CHANNELS_V2 entry. Carbon greens, nitrogen blues, oxygen reds, so
# related channels read as related at a glance.
CHANNEL_COLOURS = [
    "green",        # 0  carbon_aliphatic
    "limegreen",    # 1  carbon_aromatic
    "blue",         # 2  nitrogen_with_h
    "royalblue",    # 3  nitrogen_no_h
    "red",          # 4  oxygen_with_h
    "orangered",    # 5  oxygen_no_h
    "gold",         # 6  sulfur
    "saddlebrown",  # 7  halogen
    "cyan",         # 8  fluorine
    "magenta",      # 9  hbond_acceptor
    "purple",       # 10 other
]


def draw_grid(ax, grid, threshold, protein=None, alpha=0.45):
    """Render one (C, X, Y, Z) occupancy grid into a 3-D axis. Returns the voxel count."""
    ax.set_axis_off()
    ax.grid(False)
    if protein is not None:
        shell = (protein > threshold).any(axis=0)
        if shell.any():
            # Very low opacity: the pocket is context, not the subject. Without it the
            # samples float in an empty cube and "is it in the pocket" is unanswerable.
            ax.voxels(shell, facecolors=(0.5, 0.5, 0.5, 0.045), edgecolors=(0.5, 0.5, 0.5, 0.02))

    total = 0
    occupied = grid > threshold
    for channel in range(grid.shape[0]):
        mask = occupied[channel]
        n = int(mask.sum())
        if not n:
            continue
        total += n
        colour = mcolors.to_rgba(CHANNEL_COLOURS[channel % len(CHANNEL_COLOURS)], alpha)
        ax.voxels(mask, facecolors=colour, edgecolors=(0, 0, 0, 0.06))

    dims = grid.shape[1:]
    ax.set_xlim(0, dims[0]); ax.set_ylim(0, dims[1]); ax.set_zlim(0, dims[2])
    ax.set_box_aspect((1, 1, 1))
    return total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="zinc", choices=["zinc", "hiqbind"])
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=3, help="draws per example")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show_pocket", action="store_true",
                        help="hiqbind only: draw the pocket shell behind every panel")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tolerates a live run rewriting last.ckpt mid-read.
    model, step = load_flow_checkpoint(args.ckpt, device)
    print(f"loaded {args.ckpt} (global_step {step})")

    batches = build_batches(args.data, args.n_examples, 1, device)
    protein, ligand = batches[0]
    protein = protein[: args.n_examples] if protein is not None else None
    ligand = ligand[: args.n_examples]
    n_examples = ligand.shape[0]
    conditional = args.data == "hiqbind"

    # One shared noise draw per (example, sample) so the panels are reproducible.
    samples, dices = [], []
    with torch.no_grad():
        for draw in range(args.n_samples):
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 1000 * draw)
            x0 = torch.randn(
                (n_examples, model.n_ligand_channels, *ligand.shape[2:]),
                device=device, dtype=torch.float32, generator=generator,
            )
            sample = model.sample(protein=protein, x0=x0, n_steps=args.steps,
                                  guidance_scale=args.guidance)
            samples.append(sample.float().cpu().numpy())
            dices.append(pooled_soft_dice(sample, ligand).cpu().numpy())

    true_np = ligand.cpu().numpy()
    protein_np = protein.cpu().numpy() if (protein is not None and args.show_pocket) else None

    n_cols = 1 + args.n_samples
    fig = plt.figure(figsize=(3.4 * n_cols, 3.6 * n_examples))
    for row in range(n_examples):
        ax = fig.add_subplot(n_examples, n_cols, row * n_cols + 1, projection="3d")
        n = draw_grid(ax, true_np[row], args.threshold,
                      protein_np[row] if protein_np is not None else None)
        ax.view_init(elev=args.elev, azim=args.azim)
        label = "true ligand" if conditional else "real molecule"
        ax.set_title(f"{label}\n{n} voxels", fontsize=9)

        for col in range(args.n_samples):
            ax = fig.add_subplot(n_examples, n_cols, row * n_cols + col + 2, projection="3d")
            n = draw_grid(ax, samples[col][row], args.threshold,
                          protein_np[row] if protein_np is not None else None)
            ax.view_init(elev=args.elev, azim=args.azim)
            # Dice is only a quality score when the model is conditioned on this pocket;
            # unconditionally it just measures overlap with an unrelated molecule.
            note = (f"dice {dices[col][row]:.3f}" if conditional
                    else f"vs unrelated {dices[col][row]:.3f}")
            ax.set_title(f"sample {col + 1}\n{n} voxels, {note}", fontsize=9)

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                          markerfacecolor=CHANNEL_COLOURS[i], markeredgecolor="none",
                          label=name)
               for i, name in enumerate(LIGAND_CHANNEL_NAMES_V2)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)

    title = args.title or (
        f"{'pocket-conditioned' if conditional else 'unconditional'} flow samples "
        f"({args.data}, step {step}, {args.steps} steps"
        + (f", guidance {args.guidance}" if conditional and args.guidance != 1.0 else "") + ")"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}")

    occupied = [float((s > args.threshold).mean()) for s in samples]
    print(f"occupied fraction: samples {np.mean(occupied):.5f} "
          f"vs data {float((true_np > args.threshold).mean()):.5f}")


if __name__ == "__main__":
    main()
