"""Recompute the external-comparator benchmark from the saved score matrices. No GPU.

Plixer, Boltz-2 and AutoDock Vina rank the SAME 107-candidate panel for each PLINDER pocket.
Every number in the report comes out of `comparators_plinder107.npz`, which stores one matrix
per method plus the positive and observed masks.

TWO RULES THAT THE NUMBERS DEPEND ON
------------------------------------
1. **Cells missing from any method are dropped from all of them.** Boltz-2 lost one pocket to a
   missing MSA and Vina failed on 107 scattered pairs; if each method were averaged over its own
   observed cells the three would be scored on different problems. The intersection is 105
   pockets x ~107 candidates.

2. **Missing cells are never imputed.** Filling them with a row extremum inflated the Boltz-2
   pilot from 0.717 to 0.947, because Boltz scores all the true ligands before any decoy, so an
   imputed cell was systematically a decoy pinned to the floor.

Fusion weights are chosen by LEAVE-ONE-POCKET-OUT, so no pocket contributes to picking the
weights it is then scored under. The tuned-on-panel figure is printed alongside as the
optimistic bound; on this data they coincide, because all folds select the same weights.
"""
from __future__ import annotations

import argparse

import numpy as np


def znorm(matrix):
    """Column z-normalisation, NaN-safe. Removes the ligand-intrinsic term that otherwise
    dominates the raw likelihood (84% of its variance, r = -0.758 with heavy-atom count)."""
    mean = np.nanmean(matrix, axis=0)
    sd = np.nanstd(matrix, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (matrix - mean) / sd


def pocket_auc(matrix, positive, index, min_decoys):
    """AUC for one pocket over observed cells only; None if too few decoys survived."""
    have = ~np.isnan(matrix[index])
    pos = matrix[index][have & positive[index]]
    neg = matrix[index][have & ~positive[index]]
    if len(pos) == 0 or len(neg) < min_decoys:
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def bootstrap(delta, resamples, seed=0):
    rng = np.random.default_rng(seed)
    n = len(delta)
    draws = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return draws.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), (draws > 0).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrices", default="results/bench/comparators_plinder107.npz")
    parser.add_argument("--min_decoys", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=4000)
    args = parser.parse_args()

    blob = np.load(args.matrices, allow_pickle=True)
    positive = blob["positive"]
    observed = blob["observed"]           # cells scored by BOTH comparators
    models = {
        "Plixer ensemble(6)": blob["plixer_ens"],
        "Boltz-2 (binary)": blob["boltz"],
        "AutoDock Vina": blob["vina"],
    }
    models = {k: znorm(np.where(observed, v, np.nan)) for k, v in models.items()}

    per_pocket = {k: [pocket_auc(v, positive, i, args.min_decoys) for i in range(v.shape[0])]
                  for k, v in models.items()}
    keep = [i for i in range(positive.shape[0])
            if all(per_pocket[k][i] is not None for k in models)]
    auc = {k: np.array([per_pocket[k][i] for i in keep]) for k in models}

    print(f"panel: {len(keep)} pockets x {positive.shape[1]} candidates, "
          f"{int(observed[keep].sum())} cells scored by all methods\n")
    for name in models:
        print(f"  {name:22s} znorm AUC {auc[name].mean():.4f}")

    print(f"\npaired bootstrap over pockets ({args.resamples} resamples):")
    reference = "Plixer ensemble(6)"
    for name in models:
        if name == reference:
            continue
        delta = auc[reference] - auc[name]
        mean, lo, hi, p = bootstrap(delta, args.resamples)
        print(f"  Plixer - {name:20s} {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(>0)={p:.3f}  wins {(delta > 0).sum()}/{len(keep)}")

    print("\nper-pocket AUC correlation (near-zero => complementary, fusable):")
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = np.corrcoef(auc[names[i]], auc[names[j]])[0, 1]
            print(f"  r({names[i]}, {names[j]}) = {r:+.3f}")

    # --- fusion -------------------------------------------------------------------------
    plixer, boltz, vina = (models[k] for k in
                           ["Plixer ensemble(6)", "Boltz-2 (binary)", "AutoDock Vina"])
    grid = [(1 - wb - wv, wb, wv)
            for wb in np.arange(0, 0.65, 0.05) for wv in np.arange(0, 0.65, 0.05)
            if 1 - wb - wv >= 0.2]
    cache = {g: np.array([pocket_auc(g[0] * plixer + g[1] * boltz + g[2] * vina,
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
    tuned = max(cache.values(), key=lambda a: a.mean()).mean()

    print("\nfusion of all three, column-z-normalised then blended:")
    print(f"  Plixer alone                        {auc[reference].mean():.4f}")
    print(f"  weights tuned ON the panel          {tuned:.4f}   (optimistic bound)")
    print(f"  weights by leave-one-pocket-out     {held_out.mean():.4f}   (honest)")
    unique = {g for g in selected}
    print(f"  LOO selected {len(unique)} distinct weight triple(s); modal "
          f"{tuple(round(x, 2) for x in max(set(selected), key=selected.count))}")
    delta = held_out - auc[reference]
    mean, lo, hi, p = bootstrap(delta, args.resamples)
    print(f"  LOO fusion - Plixer = {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={p:.3f}  wins {(delta > 0).sum()}/{len(keep)}")


if __name__ == "__main__":
    main()
