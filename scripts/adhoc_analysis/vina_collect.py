"""Score the Vina cross-docking as a hit-vs-decoy ranking, alongside Plixer on the same panel.

Vina reports binding free energy in kcal/mol, so LOWER is better; the score is negated to make
higher-is-better and compose with the same `per_pocket_auc` used for Plixer and Boltz-2.

⚠️ MISSING CELLS ARE SKIPPED, NOT IMPUTED. The Boltz-2 run demonstrated why: its phase A scores
every pocket's true ligand and completes long before the decoys, so filling absent cells with the
row minimum -- which is safe when cells are missing at random -- pinned the negatives to the floor
and read AUC 0.947 where the honest value was 0.717. Docking failures here are not random either
(they correlate with ligand size and flexibility), so the same rule applies: a cell that was never
computed contributes to nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.likelihood_eval import per_pocket_auc, znormalise_columns   # noqa: E402


def row_auc(matrix, positive, min_decoys):
    """Per-pocket AUC over observed cells only; returns (mean_auc, n_pockets_scored)."""
    aucs = []
    for i in range(matrix.shape[0]):
        have = ~np.isnan(matrix[i])
        pos = matrix[i][have & positive[i]]
        neg = matrix[i][have & ~positive[i]]
        if len(pos) == 0 or len(neg) < min_decoys:
            continue
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        aucs.append(float(wins / (len(pos) * len(neg))))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def znorm_ignoring_nan(matrix):
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (matrix - mean) / sd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", default="../vina_bench/plinder")
    parser.add_argument("--chrono_csv", default="data/test_set_chronological_split.csv")
    parser.add_argument("--plixer_matrices", default="results/bench/*.npz")
    parser.add_argument("--min_decoys", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    meta = json.load(open(f"{args.work}/manifest.json"))
    ids, panel_ids = meta["system_ids"], meta["panel_ids"]
    chrono = pd.read_csv(args.chrono_csv).set_index("system_id")
    panel = [chrono.loc[p].smiles for p in panel_ids]
    row_of = {s: i for i, s in enumerate(ids)}
    col_of = {p: j for j, p in enumerate(panel_ids)}

    matrix = np.full((len(ids), len(panel_ids)), np.nan)
    errors = 0
    for path in glob.glob(f"{args.work}/scores/*.json"):
        blob = json.load(open(path))
        i, j = row_of.get(blob["system_id"]), col_of.get(blob["candidate"])
        if i is None or j is None:
            continue
        value = blob["score"]
        if isinstance(value, str):          # "ERROR:..." -- a failed docking, not a weak one
            errors += 1
            continue
        matrix[i, j] = -float(value)        # kcal/mol, lower is better -> negate
    observed = int((~np.isnan(matrix)).sum())
    print(f"Vina: {observed} scores ({observed / matrix.size:.1%} of "
          f"{len(ids)}x{len(panel_ids)}), {errors} docking failures")

    positive = np.zeros_like(matrix, dtype=bool)
    for i, sid in enumerate(ids):
        target = chrono.loc[sid].smiles
        positive[i] = np.array([s == target for s in panel])

    raw, n_raw = row_auc(matrix, positive, args.min_decoys)
    znorm, n_z = row_auc(znorm_ignoring_nan(matrix), positive, args.min_decoys)
    print(f"  Vina  raw AUC {raw:.4f}   znorm AUC {znorm:.4f}   ({n_z} pockets)")

    results = {"vina": {"raw": raw, "znorm": znorm, "pockets": n_z,
                        "observed_cells": observed, "errors": errors}}

    # Plixer on the identical pockets x candidates, sliced from the precomputed 943x943 matrix.
    paths = sorted(glob.glob(args.plixer_matrices))
    if paths:
        members = [np.load(p, allow_pickle=True) for p in paths]
        all_ids = [str(s) for s in members[0]["system_ids"]]
        all_panel = [str(s) for s in members[0]["panel"]]
        rows = [all_ids.index(s) for s in ids if s in all_ids]
        cols = [all_panel.index(s) for s in panel if s in all_panel]
        if len(rows) == len(ids) and len(cols) == len(panel):
            stack = np.stack([znormalise_columns(m["decoder"][0])[np.ix_(rows, cols)]
                              for m in members])
            for label, mat in [("single", stack[0]), (f"ensemble({len(members)})",
                                                      stack.mean(axis=0))]:
                # Same per-row treatment, so a pocket Vina could not score is excluded from
                # BOTH -- otherwise the two models would be averaged over different pockets.
                masked = np.where(np.isnan(matrix), np.nan, mat)
                auc, n = row_auc(masked, positive, args.min_decoys)
                print(f"  Plixer {label:12s} znorm AUC {auc:.4f}   ({n} pockets, matched cells)")
                results.setdefault("plixer", {})[label] = auc
        else:
            print(f"  ⚠️ could align only {len(rows)}/{len(ids)} pockets, "
                  f"{len(cols)}/{len(panel)} candidates -- Plixer comparison skipped")

    if args.output:
        json.dump(results, open(args.output, "w"), indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
