"""Score Gnina alongside Plixer, Boltz-2 and Vina on one common panel.

Gnina reports three numbers per pair and they rank differently, so all three are scored rather
than one being picked silently:

  cnn_affinity   predicted pK, higher better -- Gnina's recommended screening readout. Strongly
                 ligand-intrinsic (r = +0.69 with heavy-atom count), so compare on the
                 z-normalised column, not the raw one.
  cnn_score      CNN pose quality 0-1, higher better -- answers "is the pose right", NOT
                 "does it bind", so it is expected to rank worse
  affinity       Gnina's OWN default empirical score in kcal/mol, LOWER better, negated here

⚠️ Gnina docks the same receptors, ligand conformers and boxes as `vina_benchmark.py`, but it is
run with defaults and `--scoring default` is NOT `--scoring vina`. The Gnina-vs-Vina difference
is therefore "the two tools as normally run", not an isolated scoring-function experiment: they
agree at Spearman 0.75 on shared poses and Gnina fails to place a ligand that Vina places on
6.8% of pairs (0.1% the reverse). Do not regenerate the inputs.

Cells missing from ANY method are dropped from ALL of them, and nothing is ever imputed -- see
`comparator_report.py` for the 0.947 artifact that rule exists to prevent.
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

from scripts.adhoc_analysis.comparator_report import (           # noqa: E402
    bootstrap, pocket_auc, znorm)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", default="../gnina_bench/plinder")
    parser.add_argument("--vina_work", default="../vina_bench/plinder")
    parser.add_argument("--comparators", default="results/bench/comparators_plinder107.npz")
    parser.add_argument("--chrono_csv", default="data/test_set_chronological_split.csv")
    parser.add_argument("--min_decoys", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--output", default="results/bench/comparators4_plinder107.npz")
    args = parser.parse_args()

    prior = np.load(args.comparators, allow_pickle=True)
    ids = [str(s) for s in prior["system_ids"]]
    panel_ids = [str(s) for s in prior["panel"]]
    row_of = {s: i for i, s in enumerate(ids)}
    col_of = {s: j for j, s in enumerate(panel_ids)}

    shape = (len(ids), len(panel_ids))
    gnina = {key: np.full(shape, np.nan) for key in ("cnn_affinity", "cnn_score", "affinity")}
    errors = 0
    for path in glob.glob(f"{args.work}/scores/*.json"):
        blob = json.load(open(path))
        i, j = row_of.get(blob["system_id"]), col_of.get(blob["candidate"])
        if i is None or j is None:
            continue
        if "error" in blob or "cnn_affinity" not in blob:
            errors += 1
            continue
        gnina["cnn_affinity"][i, j] = blob["cnn_affinity"]
        gnina["cnn_score"][i, j] = blob["cnn_score"]
        gnina["affinity"][i, j] = -blob["affinity"]        # kcal/mol, lower better -> negate
    observed_g = ~np.isnan(gnina["cnn_affinity"])
    print(f"Gnina: {int(observed_g.sum())} scores "
          f"({observed_g.mean():.1%} of {shape[0]}x{shape[1]}), {errors} failures")

    positive = prior["positive"]
    observed = prior["observed"] & observed_g          # cells every method has
    models = {
        "Plixer ensemble(6)": prior["plixer_ens"],
        "Boltz-2 (binary)": prior["boltz"],
        "Gnina CNNaffinity": gnina["cnn_affinity"],
        "Gnina CNNscore": gnina["cnn_score"],
        "Gnina affinity": gnina["affinity"],
        "AutoDock Vina": prior["vina"],
    }
    raw = {k: np.where(observed, v, np.nan) for k, v in models.items()}
    models = {k: znorm(v) for k, v in raw.items()}

    per_pocket = {k: [pocket_auc(v, positive, i, args.min_decoys) for i in range(shape[0])]
                  for k, v in models.items()}
    keep = [i for i in range(shape[0]) if all(per_pocket[k][i] is not None for k in models)]
    auc = {k: np.array([per_pocket[k][i] for i in keep]) for k in models}
    # Raw AUC is reported alongside because the gap between them is diagnostic, not cosmetic:
    # a readout that is mostly a per-ligand offset (Boltz-2's regression head, S23b) gains a lot
    # from column z-normalisation, while a genuinely pocket-specific one gains little.
    auc_raw = {k: np.array([a for a in (pocket_auc(v, positive, i, args.min_decoys)
                                        for i in keep) if a is not None])
               for k, v in raw.items()}

    print(f"\npanel: {len(keep)} pockets x {shape[1]} candidates, "
          f"{int(observed[keep].sum())} cells scored by every method\n")
    for name, values in sorted(auc.items(), key=lambda kv: -kv[1].mean()):
        print(f"  {name:22s} raw AUC {auc_raw[name].mean():.4f}   znorm AUC {values.mean():.4f}")

    reference = "Plixer ensemble(6)"
    print(f"\npaired bootstrap vs {reference} ({args.resamples} resamples over pockets):")
    for name in models:
        if name == reference:
            continue
        delta = auc[reference] - auc[name]
        _, lo, hi, p = bootstrap(delta, args.resamples)
        print(f"  vs {name:22s} {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(>0)={p:.3f}  wins {(delta > 0).sum()}/{len(keep)}")

    # Same inputs, but Gnina at defaults optimises its own empirical function, so this is a
    # tool-vs-tool comparison, NOT an isolated scoring-function experiment. See the docstring.
    delta = auc["Gnina CNNaffinity"] - auc["AutoDock Vina"]
    _, lo, hi, p = bootstrap(delta, args.resamples)
    print(f"\n  Gnina CNNaffinity - Vina = {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={p:.3f}     (same inputs, but both tools at their own defaults)")

    print("\nper-pocket AUC correlation with Plixer (near-zero => fusable):")
    for name in models:
        if name != reference:
            print(f"  r({reference}, {name}) = "
                  f"{np.corrcoef(auc[reference], auc[name])[0, 1]:+.3f}")

    # --- fusion, leave-one-pocket-out weights ------------------------------------------
    keys = [reference, "Boltz-2 (binary)", "AutoDock Vina", "Gnina CNNaffinity"]
    stack = [models[k] for k in keys]
    grid = []
    step = 0.1
    for wb in np.arange(0, 0.55, step):
        for wv in np.arange(0, 0.55, step):
            for wg in np.arange(0, 0.55, step):
                wp = 1 - wb - wv - wg
                if wp >= 0.2:
                    grid.append((wp, wb, wv, wg))
    cache = {g: np.array([pocket_auc(sum(w * m for w, m in zip(g, stack)),
                                     positive, i, args.min_decoys) for i in keep])
             for g in grid}
    held_out = np.empty(len(keep))
    selected = []
    for i in range(len(keep)):
        mask = np.ones(len(keep), bool)
        mask[i] = False
        best = max(grid, key=lambda g: cache[g][mask].mean())
        held_out[i] = cache[best][i]
        selected.append(best)
    print(f"\nfour-way fusion {tuple(keys)}:")
    print(f"  Plixer alone                        {auc[reference].mean():.4f}")
    print(f"  weights tuned ON the panel          "
          f"{max(cache.values(), key=lambda a: a.mean()).mean():.4f}   (optimistic bound)")
    print(f"  weights by leave-one-pocket-out     {held_out.mean():.4f}   (honest)")
    modal = max(set(selected), key=selected.count)
    print(f"  LOO selected {len(set(selected))} distinct triple(s); modal "
          f"{tuple(round(x, 2) for x in modal)} in {selected.count(modal)}/{len(keep)} folds")
    delta = held_out - auc[reference]
    _, lo, hi, p = bootstrap(delta, args.resamples)
    print(f"  LOO fusion - Plixer = {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={p:.3f}  wins {(delta > 0).sum()}/{len(keep)}")

    np.savez(args.output, system_ids=np.array(ids), panel=np.array(panel_ids),
             positive=positive, observed=observed,
             plixer_ens=prior["plixer_ens"], boltz=prior["boltz"], vina=prior["vina"],
             gnina_cnn_affinity=gnina["cnn_affinity"], gnina_cnn_score=gnina["cnn_score"],
             gnina_affinity=gnina["affinity"], keep=np.array(keep))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
