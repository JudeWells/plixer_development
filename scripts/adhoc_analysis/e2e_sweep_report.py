"""Read an end-to-end sweep off W&B and print the contrasts it was designed to measure.

The four arms form a ladder, each rung adding exactly one thing to the one below, so the
useful quantities are DIFFERENCES between adjacent rungs, not the raw numbers:

    D - Z   effect of merely unfreezing Poc2Mol (it trains on its own voxel loss)
    B - D   effect of the LANGUAGE-MODELLING GRADIENT reaching Poc2Mol  <- the measurement
    C - B   effect of loosening the density anchor from 1.0 to 0.1

Two habits this script enforces, both from CLAUDE.md:

* Report BOTH the best value and the final one. "Best val X" carries a maximum-selection
  bias, and a big best-minus-final gap means the arm peaked and then degraded -- which is
  the expected signature of a density drifting away from being a density.
* Never read the AUC without the Dice beside it. An AUC win bought by destroying the
  density is not a win; it means Poc2Mol has become a private code for the decoder.
  val/poc2mol/dice uses density_diagnostics.pooled_soft_dice's definition, so it is
  directly comparable to the 0.5027 baseline.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/e2e_sweep_report.py
    ./venvPlixer/bin/python scripts/adhoc_analysis/e2e_sweep_report.py --runs Z=abc123 B=def456
"""
from __future__ import annotations

import argparse

import numpy as np
import wandb

# Order matters: it is the ladder, bottom rung first.
#
# ROUND 2 (2026-08-12 16:38). The live sweep. Arm C (anchor 0.1) is replaced by arm A
# (anchor 3.0), which pulls the balance the other way -- see e2e_a_anchored_hard.yaml.
DEFAULT_RUNS = [
    ("Z-frozen", "uxufh43w"),
    ("D-control", "1e3ptob7"),
    ("B-balanced", "0d5u8ufc"),
    ("A-anchored3", "1l41jx1d"),
]

# ROUND 1 (2026-08-12 15:54). ⚠️ TRUNCATED and not directly comparable: val_check_interval
# was 250 MICRO-batches against accumulate_grad_batches=4, so validation ran every 62
# optimiser steps and `patience: 12` meant 750 steps rather than 3000. Every arm was stopped
# between step 1312 and 1374, none reached the LR anneal that begins at step 2000, and each
# had ~21 validation draws feeding its "best" value instead of ~5. Read as an early-training
# snapshot only. Pass with --runs to inspect:
#     --runs Z-frozen=ro3ztphj D-control=0ovikvsz B-balanced=cegjk664 C-lm-dominant=7wthzn9n
ROUND_1_RUNS = [
    ("Z-frozen", "ro3ztphj"),
    ("D-control", "0ovikvsz"),
    ("B-balanced", "cegjk664"),
    ("C-lm-dominant", "7wthzn9n"),
]

AUC = "val/likelihood_auc_znorm"
# dataloader_idx_1 is hiqbind_val -- the full 1019-pocket split, i.e. the one comparable to
# the 0.5027 figure. idx_0 is the 104-pocket PLINDER panel and runs lower on a smaller set.
DICE = "val/poc2mol/dice/dataloader_idx_1"
KEYS = [
    AUC, DICE,
    "val/poc2mol/dice/dataloader_idx_0",
    "val/poc2mol/loss", "val/poc2mol/tanimoto", "val/poc2mol/exact_match",
    "val/poc2mol/emission_ratio/dataloader_idx_1",
    "val/poc2mol/on_target/dataloader_idx_1",
    "val/zinc/loss", "val/zinc/exact_match",
    "trainer/global_step", "_runtime",
]


def series(history, key):
    """(step, value) pairs for one key, dropping the rows where it was not logged."""
    out = [(row.get("trainer/global_step"), row.get(key)) for row in history]
    out = [(s, v) for s, v in out
           if s is not None and v is not None and not (isinstance(v, float) and np.isnan(v))]
    return sorted(out)


def fetch(entity_project, runs):
    api = wandb.Api()
    data = {}
    for name, run_id in runs:
        run = api.run(f"{entity_project}/{run_id}")
        history = run.history(keys=KEYS, samples=5000, pandas=False)
        data[name] = {
            "state": run.state,
            "history": history,
            "summary": run.summary,
            "auc": series(history, AUC),
            "dice": series(history, DICE),
        }
    return data


def fmt(value, width=7, places=4):
    return " " * width if value is None else f"{value:{width}.{places}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_project", default="cath/voxelSmiles")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="name=run_id pairs; defaults to the 2026-08-12 sweep")
    args = parser.parse_args()

    runs = DEFAULT_RUNS
    if args.runs:
        runs = [tuple(spec.split("=", 1)) for spec in args.runs]

    data = fetch(args.entity_project, runs)

    print("=" * 92)
    print("PER-ARM SUMMARY   (dice is pooled soft Dice on hiqbind_val; baseline 0.5027)")
    print("=" * 92)
    header = (f"{'arm':<15}{'state':<10}{'step':>6}{'bestAUC':>9}{'@step':>7}"
              f"{'finalAUC':>9}{'dice@best':>10}{'finalDice':>10}{'zincLoss':>9}")
    print(header)
    print("-" * 92)

    best = {}
    for name, _ in runs:
        entry = data[name]
        auc, dice = entry["auc"], entry["dice"]
        if not auc:
            print(f"{name:<15}{entry['state']:<10}{'-':>6}  no validation logged yet")
            continue
        best_step, best_auc = max(auc, key=lambda p: p[1])
        final_step, final_auc = auc[-1]
        dice_at_best = next((v for s, v in dice if s == best_step), None)
        final_dice = dice[-1][1] if dice else None
        zinc = entry["summary"].get("val/zinc/loss")
        best[name] = {"auc": best_auc, "step": best_step, "final_auc": final_auc,
                      "dice": dice_at_best, "final_dice": final_dice}
        print(f"{name:<15}{entry['state']:<10}{final_step:>6}{fmt(best_auc,9)}{best_step:>7}"
              f"{fmt(final_auc,9)}{fmt(dice_at_best,10)}{fmt(final_dice,10)}"
              f"{fmt(zinc,9,5) if zinc is not None else '':>9}")

    print()
    print("=" * 92)
    print("LADDER CONTRASTS   (each rung adds exactly one thing to the one below)")
    print("=" * 92)
    contrasts = [
        ("D-control", "Z-frozen", "unfreezing Poc2Mol (voxel loss only)"),
        ("B-balanced", "D-control", "*** THE LM GRADIENT REACHING POC2MOL ***"),
        ("C-lm-dominant", "B-balanced", "loosening the density anchor 1.0 -> 0.1"),
    ]
    for upper, lower, label in contrasts:
        if upper in best and lower in best:
            d_auc = best[upper]["auc"] - best[lower]["auc"]
            d_dice = ((best[upper]["dice"] or 0) - (best[lower]["dice"] or 0))
            print(f"  {label}")
            print(f"      dAUC = {d_auc:+.4f}    dDice = {d_dice:+.4f}   "
                  f"({upper} {best[upper]['auc']:.4f} vs {lower} {best[lower]['auc']:.4f})")
    print()
    print("  Reference points: the metric is noisy at +-0.02 between adjacent checks, so a")
    print("  contrast below ~0.02 is not a result. Parameter-free composition readout = 0.7615.")

    print()
    print("=" * 92)
    print("AUC / DICE TRAJECTORIES")
    print("=" * 92)
    for name, _ in runs:
        entry = data[name]
        if not entry["auc"]:
            continue
        dice_map = dict(entry["dice"])
        print(f"\n{name}")
        print(f"  {'step':>6} {'auc':>8} {'dice':>8}")
        for step, value in entry["auc"]:
            marker = "  <- best" if best.get(name, {}).get("step") == step else ""
            print(f"  {step:>6} {value:>8.4f} {fmt(dice_map.get(step), 8)}{marker}")


if __name__ == "__main__":
    main()
