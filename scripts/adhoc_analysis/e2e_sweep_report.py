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
# ROUND 3 (2026-08-12 17:20). The live sweep. Early stopping disabled so every arm runs the
# full 4000 steps and completes the LR anneal (§23f). Arm A (anchor 3.0) is replaced by arm E
# (poc2mol_lr 1e-5): round 2 showed anchor strength does nothing (A-B = -0.004) while both
# moving-upstream arms ended with Dice ABOVE the frozen baseline yet worse AUC, which points
# at a moving-target problem rather than a density-quality one. See e2e_e_slow_upstream.yaml.
DEFAULT_RUNS = [
    ("Z-frozen", "5vcgm5tc"),
    ("D-control", "cidxmees"),
    ("B-balanced", "en8a9lnx"),
    ("E-slow-upstream", "su6pa5bf"),
]

# ROUND 2 (2026-08-12 16:38). Complete, but arms early-stopped at different steps (§23f):
# Z at 2999, B at 2499, D and A at the full 3999. Inspect with:
#   --runs Z-frozen=uxufh43w D-control=1e3ptob7 B-balanced=0d5u8ufc A-anchored3=1l41jx1d
ROUND_2_RUNS = [
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


def smooth(values, window=3):
    """Centred rolling mean, shrinking the window at the ends rather than dropping points."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


def check_noise(values):
    """Per-check standard deviation, estimated from successive differences.

    Var(x_t - x_{t-1}) = 2*sigma^2 when the checks are independent around a slowly-moving
    level, so sigma = sd(diff)/sqrt(2). Using the raw sd instead would confuse the metric's
    noise with the training trend it is sitting on.
    """
    if len(values) < 3:
        return None
    return float(np.std(np.diff(values), ddof=1) / np.sqrt(2))


def selection_bias(sigma, n_draws):
    """Roughly how much a MAXIMUM over n noisy draws exceeds the underlying level.

    E[max of n standard normals] ~ sqrt(2*ln(n)) for moderate n. This is why comparing
    arms on "best AUC" compares luck as much as quality, and why the smoothed peak is the
    number to read (CLAUDE.md §6).
    """
    if sigma is None or n_draws < 2:
        return None
    return float(sigma * np.sqrt(2 * np.log(n_draws)))


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

    print("=" * 100)
    print("PER-ARM SUMMARY   (dice is pooled soft Dice on hiqbind_val; baseline 0.5027)")
    print("  smoothAUC = peak of a centred 3-check rolling mean. READ THIS ONE, not rawBest:")
    print("  rawBest is a maximum over ~16 noisy draws and is inflated by roughly `bias`.")
    print("=" * 100)
    header = (f"{'arm':<14}{'state':<9}{'step':>6}{'rawBest':>9}{'smoothAUC':>10}{'@step':>7}"
              f"{'sigma':>7}{'bias':>7}{'peakDice':>9}{'lastDice':>9}{'zinc':>8}")
    print(header)
    print("-" * 100)

    best = {}
    for name, _ in runs:
        entry = data[name]
        auc, dice = entry["auc"], entry["dice"]
        if not auc:
            print(f"{name:<14}{entry['state']:<9}{'-':>6}  no validation logged yet")
            continue
        steps = [s for s, _ in auc]
        values = [v for _, v in auc]
        smoothed = smooth(values)
        raw_best = max(values)
        peak_ix = int(np.argmax(smoothed))
        sigma = check_noise(values)
        bias = selection_bias(sigma, len(values))
        dice_map = dict(dice)
        # Dice at the SMOOTHED peak -- the checkpoint that step actually corresponds to.
        peak_dice = dice_map.get(steps[peak_ix])
        last_dice = dice[-1][1] if dice else None
        zinc = entry["summary"].get("val/zinc/loss")
        best[name] = {"auc": smoothed[peak_ix], "raw": raw_best, "step": steps[peak_ix],
                      "dice": peak_dice, "final_dice": last_dice, "sigma": sigma}
        print(f"{name:<14}{entry['state']:<9}{steps[-1]:>6}{fmt(raw_best,9)}"
              f"{fmt(smoothed[peak_ix],10)}{steps[peak_ix]:>7}"
              f"{fmt(sigma,7,4) if sigma else '':>7}{fmt(bias,7,4) if bias else '':>7}"
              f"{fmt(peak_dice,9)}{fmt(last_dice,9)}"
              f"{fmt(zinc,8,5) if zinc is not None else '':>8}")

    print()
    print("=" * 92)
    print("LADDER CONTRASTS   (each rung adds exactly one thing to the one below)")
    print("=" * 92)
    # The third rung differs between rounds: round 1 loosened the anchor, round 2 tightens
    # it. Both are stated against arm B, so whichever is present is the one that prints.
    contrasts = [
        ("D-control", "Z-frozen", "unfreezing Poc2Mol (voxel loss only)"),
        ("B-balanced", "D-control", "*** THE LM GRADIENT REACHING POC2MOL ***"),
        ("A-anchored3", "B-balanced", "tightening the density anchor 1.0 -> 3.0"),
        ("C-lm-dominant", "B-balanced", "loosening the density anchor 1.0 -> 0.1"),
        ("E-slow-upstream", "B-balanced", "slowing the upstream 1e-4 -> 1e-5"),
        ("E-slow-upstream", "Z-frozen", "slow end-to-end vs the frozen baseline"),
        ("A-anchored3", "Z-frozen", "anchored end-to-end vs the frozen baseline"),
        # Density arms: all three upstreams frozen and stationary, same decoder init, so
        # these contrasts are about the DENSITY alone with no moving target anywhere.
        ("G-Ddensity", "Z-frozen", "further voxel-only upstream training (density frozen)"),
        ("F-Bdensity", "G-Ddensity", "*** WHAT THE LM GRADIENT ADDED TO THE DENSITY ***"),
        ("F-Bdensity", "Z-frozen", "end-to-end density vs original, both frozen"),
    ]
    for upper, lower, label in contrasts:
        if upper in best and lower in best:
            d_auc = best[upper]["auc"] - best[lower]["auc"]
            d_dice = ((best[upper]["dice"] or 0) - (best[lower]["dice"] or 0))
            print(f"  {label}")
            print(f"      dAUC = {d_auc:+.4f}    dDice = {d_dice:+.4f}   "
                  f"({upper} {best[upper]['auc']:.4f} vs {lower} {best[lower]['auc']:.4f})")
    print()
    sigmas = [b["sigma"] for b in best.values() if b.get("sigma")]
    if sigmas:
        pooled = float(np.mean(sigmas))
        print(f"  Per-check sigma across arms = {pooled:.4f} (WITHIN a run, between validations).")
    # ⚠️ Do NOT turn the per-check sigma into a significance threshold. That is what this
    # script used to do, propagating it as 1.96*sigma*sqrt(2/3) to get floors of 0.020-0.024,
    # and it was far too conservative -- it labelled every real effect "noise" for most of a
    # day. The smoothed PEAK is a much more stable statistic than the per-check scatter
    # implies, because smoothing plus taking a maximum over 16 checks averages most of that
    # scatter away.
    #
    # Measured directly on seed replicates (2026-08-12, 4000 steps each):
    #     Z frozen    seed 42  0.7601   seed 43  0.7618   spread 0.0017
    #     B balanced  seed 42  0.7531   seed 43  0.7515   spread 0.0016
    # so the between-seed spread of the smoothed peak is ~0.002, an order of magnitude below
    # the per-check sigma. THAT is the yardstick for comparing arms.
    print("  Between-seed spread of the SMOOTHED PEAK is ~0.002 (measured, Z and B, 2 seeds).")
    print("  Judge contrasts against ~0.002-0.005, NOT against the per-check sigma above.")
    print("  Caveat: 2 seeds is a crude spread estimate, and only Z/B have replicates.")
    print("  Parameter-free composition readout = 0.7615 (§14d).")

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
