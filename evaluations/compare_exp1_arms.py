"""Aggregate `evaluate_exp1_arm.py` outputs into the experiment-1 comparison table.

Takes the per-run JSON files, groups them by arm, and reports mean +/- standard deviation
across seeds with the baseline-to-protein delta. Seeds are the point: a single-seed delta on
likelihood AUC is not distinguishable from run-to-run noise, which is why the experiment was
specified as matched-budget and seed-replicated.

Usage:
    python evaluations/compare_exp1_arms.py evaluation_results/exp1/*.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

# (key, human label, higher_is_better)
METRICS = [
    ("teacher_forced_loss", "teacher-forced loss", False),
    ("token_accuracy", "token accuracy", True),
    ("validity", "validity", True),
    ("uniqueness", "uniqueness", True),
    ("tanimoto_to_true_mean", "Tanimoto to true ligand", True),
    ("likelihood_auc_znorm", "likelihood AUC (z-norm)", True),
    ("likelihood_auc_raw", "likelihood AUC (raw)", True),
]


def arm_of(record):
    if record.get("mask_protein"):
        return "protein (masked)"
    return "protein" if record.get("inject_protein") else "baseline"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON files from evaluate_exp1_arm.py")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in args.results:
        with open(path) as handle:
            record = json.load(handle)
        grouped[arm_of(record)].append(record)

    arms = [a for a in ("baseline", "protein", "protein (masked)") if a in grouped]
    print(f"{'metric':26s} " + " ".join(f"{a:>22s}" for a in arms) + f" {'delta':>12s}")
    print("-" * (27 + 23 * len(arms) + 13))

    for key, label, higher_better in METRICS:
        cells, means = [], {}
        for arm in arms:
            vals = [r[key] for r in grouped[arm] if r.get(key) is not None]
            if not vals:
                cells.append(f"{'-':>22s}")
                continue
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            means[arm] = mean
            cells.append(f"{mean:>13.4f} +/-{sd:6.4f}" if len(vals) > 1 else f"{mean:>22.4f}")

        delta = ""
        if "baseline" in means and "protein" in means:
            diff = means["protein"] - means["baseline"]
            if abs(diff) < 1e-9:
                mark = "="
            else:
                mark = "W" if (diff > 0) == higher_better else "L"
            delta = f"{diff:+.4f} {mark}"
        print(f"{label:26s} " + " ".join(cells) + f" {delta:>12s}")

    print()
    for arm in arms:
        print(f"  {arm}: n={len(grouped[arm])} run(s)")
    if any(len(v) < 2 for v in grouped.values()):
        print("\n  WARNING: fewer than 2 seeds in at least one arm. The delta cannot be")
        print("  separated from run-to-run noise -- treat it as directional only.")
    print("\n  Quote likelihood AUC (z-norm), not raw: the raw metric is ~84% ligand-size")
    print("  nuisance and a pocket-blind baseline scores exactly 0.500 (CLAUDE.md 5.1).")
    print("  'protein (masked)' is the within-model ablation: if it recovers baseline,")
    print("  the model genuinely uses the pocket; if it matches the unmasked protein arm,")
    print("  the protein channels are being ignored.")


if __name__ == "__main__":
    main()
