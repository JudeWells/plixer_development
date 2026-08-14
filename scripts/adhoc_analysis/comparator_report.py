"""Recompute the external-comparator benchmark from the saved score matrices. No GPU.

Plixer, Boltz-2, AutoDock Vina and Gnina rank the SAME 107-candidate panel for each PLINDER
pocket. Everything comes out of one .npz holding a matrix per method plus the positive and
observed masks. Gnina's three readouts are picked up automatically when present, so this script
reproduces both `comparators_plinder107.npz` (3 comparators, 105 pockets) and
`comparators4_plinder107.npz` (4 comparators, 103 pockets).

THREE RULES THAT THE NUMBERS DEPEND ON
--------------------------------------
1. **Cells missing from any method are dropped from all of them.** Boltz-2 lost one pocket to a
   missing MSA, Vina failed on 107 scattered pairs, and Gnina rejects 2 ligands whose Meeko
   PDBQTs carry macrocycle glue atoms (CG0/G0). Averaging each method over its own observed
   cells would score them on different problems.

2. **Missing cells are never imputed.** Filling them with a row extremum inflated the Boltz-2
   pilot from 0.717 to 0.947, because Boltz scores all the true ligands before any decoy, so an
   imputed cell was systematically a decoy pinned to the floor.

3. **Fusion selection is NESTED.** The comparator subset AND the blend weights are chosen inside
   the leave-one-pocket-out loop. Picking the best subset by comparing whole-panel LOO scores and
   then quoting that subset's LOO number leaks the held-out pocket into the subset choice -- the
   maximum-selection error of S6, one level up. It is worth +0.007 here (0.8711 vs 0.8641), so
   the distinction is not academic.

Raw AUC is printed beside the z-normalised one because the gap diagnoses a ligand-intrinsic
readout: Boltz-2's regression head and Gnina's CNNaffinity gain hugely from z-normalisation,
while Gnina's CNNscore barely moves.
"""
from __future__ import annotations

import argparse
import itertools

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
    observed = blob["observed"]           # cells scored by EVERY comparator present
    models = {
        "Plixer ensemble(6)": blob["plixer_ens"],
        "Boltz-2 (binary)": blob["boltz"],
        "AutoDock Vina": blob["vina"],
    }
    # The 4-way matrix file adds Gnina's three readouts. Detected rather than required, so this
    # script reproduces both the 3-comparator and the 4-comparator tables from their own file.
    for key, label in [("gnina_cnn_score", "Gnina CNNscore"),
                       ("gnina_cnn_affinity", "Gnina CNNaffinity"),
                       ("gnina_affinity", "Gnina affinity")]:
        if key in blob.files:
            models[label] = blob[key]
    raw = {k: np.where(observed, v, np.nan) for k, v in models.items()}
    models = {k: znorm(v) for k, v in raw.items()}

    per_pocket = {k: [pocket_auc(v, positive, i, args.min_decoys) for i in range(v.shape[0])]
                  for k, v in models.items()}
    keep = [i for i in range(positive.shape[0])
            if all(per_pocket[k][i] is not None for k in models)]
    auc = {k: np.array([per_pocket[k][i] for i in keep]) for k in models}

    # Raw AUC is reported alongside because the gap is diagnostic, not cosmetic: a readout that
    # is mostly a per-ligand offset (Boltz-2's regression head, Gnina CNNaffinity) gains a lot
    # from column z-normalisation, while a genuinely pocket-specific one gains little.
    auc_raw = {k: np.array([pocket_auc(v, positive, i, args.min_decoys) for i in keep])
               for k, v in raw.items()}

    print(f"panel: {len(keep)} pockets x {positive.shape[1]} candidates, "
          f"{int(observed[keep].sum())} cells scored by all methods\n")
    for name, values in sorted(auc.items(), key=lambda kv: -kv[1].mean()):
        print(f"  {name:22s} raw AUC {auc_raw[name].mean():.4f}   "
              f"znorm AUC {values.mean():.4f}")

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

    # --- fusion, with NESTED leave-one-pocket-out selection ------------------------------
    # Both the comparator SUBSET and the blend weights are chosen inside the LOO loop. Choosing
    # the subset first by comparing whole-panel LOO scores, then reporting that subset's LOO
    # number, leaks the held-out pocket into the subset choice and is optimistically biased --
    # it is the same maximum-selection error S6 warns about, one level up.
    others = [k for k in models if k != reference]
    candidates = {}
    for size in (1, 2, 3):
        for combo in itertools.combinations(others, size):
            stack = [models[reference]] + [models[c] for c in combo]
            for weights in itertools.product(*[np.arange(0, 0.65, 0.1)] * size):
                if 1 - sum(weights) < 0.2:
                    continue
                blend = (1 - sum(weights),) + weights
                candidates[(combo, blend)] = np.array(
                    [pocket_auc(sum(w * m for w, m in zip(blend, stack)),
                                positive, i, args.min_decoys) for i in keep])

    held_out = np.empty(len(keep))
    selected = []
    for i in range(len(keep)):
        mask = np.ones(len(keep), bool)
        mask[i] = False
        best = max(candidates, key=lambda k: candidates[k][mask].mean())
        held_out[i] = candidates[best][i]
        selected.append(best)
    tuned = max(candidates.values(), key=lambda a: a.mean()).mean()

    print(f"\nfusion over {len(candidates)} (subset, weight) candidates, "
          f"column-z-normalised then blended:")
    print(f"  Plixer alone                          {auc[reference].mean():.4f}")
    print(f"  selected ON the panel                 {tuned:.4f}   (optimistic bound)")
    print(f"  NESTED leave-one-pocket-out           {held_out.mean():.4f}   (honest)")
    modal = max(set(selected), key=selected.count)
    print(f"  modal pick: Plixer + {' + '.join(modal[0])} at "
          f"{tuple(round(x, 2) for x in modal[1])} in {selected.count(modal)}/{len(keep)} folds")
    delta = held_out - auc[reference]
    mean, lo, hi, p = bootstrap(delta, args.resamples)
    print(f"  fusion - Plixer = {delta.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={p:.3f}  wins {(delta > 0).sum()}/{len(keep)}")


if __name__ == "__main__":
    main()
