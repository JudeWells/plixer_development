"""Combine member matrices from several fusion runs, with a paired bootstrap over pockets.

`fusion_ensemble.py --save_matrices` writes each member's (pocket x candidate) matrix already
column z-normalised, alongside the panel, the positive mask and the valid-column mask. When two
runs share all three of those, their members are interchangeable and any subset can be
ensembled in numpy -- no GPU, no re-scoring. That is what makes it possible to ask "what if we
pooled our checkpoints with the packaged bundle's?" without owning the bundle's weights.

⚠️ Alignment is checked, not assumed. Combining matrices whose rows are different pockets, or
whose columns are a differently-ordered panel, produces a number that looks fine and means
nothing.

⚠️ Augmentation replicates are matched by index: `fusion_ensemble.py` seeds
`torch.manual_seed(4242 + a)` per augmentation, so aug `a` is the same rotation in every run.
Pooling across runs therefore pools like with like.

WHY THE BOOTSTRAP. Every AUC here comes from ONE 104-pocket panel. The differences being
compared (+0.017 between ensembles) are smaller than the differences this project has already
seen evaporate under replication. Resampling pockets with replacement gives the panel-sampling
component of the uncertainty, and doing it PAIRED -- the same resampled pockets for both
ensembles -- removes the shared pocket-difficulty variance, which is most of it. It does not
capture training-seed variance, so it is a lower bound on the true uncertainty.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.likelihood_eval import per_pocket_auc          # noqa: E402


def load_sets(specs):
    out = {}
    reference = None
    for spec in specs:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        if reference is None:
            reference = z
        else:
            for key in ("panel", "positive", "valid_columns"):
                if not np.array_equal(z[key], reference[key]):
                    raise SystemExit(
                        f"{name}: '{key}' differs from the first set -- these matrices are NOT "
                        f"combinable. Rows must be the same pockets in the same order and "
                        f"columns the same candidate panel."
                    )
        out[name] = {"decoder": z["decoder"], "composition": z["composition"],
                     "n_ckpt": len(z["checkpoints"])}
    return out, reference["positive"], reference["valid_columns"]


def fuse(decoder, composition, w):
    return (1.0 - w) * decoder + w * composition


def auc_of(matrix, positive, valid):
    return per_pocket_auc(matrix, positive, valid)[0]


def ensemble_curve(decoder_members, composition_members, positive, valid, weights):
    dec = decoder_members.mean(axis=0)
    comp = composition_members.mean(axis=0)
    curve = {round(w, 2): auc_of(fuse(dec, comp, w), positive, valid) for w in weights}
    best_w = max(curve, key=curve.get)
    return {"decoder_only": curve[0.0], "composition_only": curve[1.0],
            "curve": curve, "best_w": best_w, "best_auc": curve[best_w]}


def bootstrap_pairs(matrices_a, matrices_b, positive, valid, w_a, w_b, n_boot, rng):
    """Paired bootstrap over POCKETS of (auc_a - auc_b). Same resampled rows for both."""
    dec_a, comp_a = matrices_a
    dec_b, comp_b = matrices_b
    fused_a = fuse(dec_a.mean(axis=0), comp_a.mean(axis=0), w_a)
    fused_b = fuse(dec_b.mean(axis=0), comp_b.mean(axis=0), w_b)
    n = positive.shape[0]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, n, n)
        deltas[i] = (auc_of(fused_a[rows], positive[rows], valid)
                     - auc_of(fused_b[rows], positive[rows], valid))
    return deltas


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sets", nargs="+", required=True, help="name=path.npz")
    parser.add_argument("--n_boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sets, positive, valid = load_sets(args.sets)
    weights = [i / 10 for i in range(11)]
    rng = np.random.default_rng(args.seed)

    print(f"panel: {positive.shape[0]} pockets x {positive.shape[1]} candidates, "
          f"{int(valid.sum())} valid columns -- alignment verified across "
          f"{len(sets)} set(s)\n")

    results = {}
    combos = []
    names = list(sets)
    for size in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, size))

    print(f"{'members':38s}{'ckpts':>6}{'decoder':>9}{'comp':>8}{'FUSED':>9}{'@w':>5}")
    print("-" * 75)
    for combo in combos:
        dec = np.concatenate([sets[n]["decoder"] for n in combo])
        comp = np.concatenate([sets[n]["composition"] for n in combo])
        n_ckpt = sum(sets[n]["n_ckpt"] for n in combo)
        r = ensemble_curve(dec, comp, positive, valid, weights)
        r["n_ckpt"] = n_ckpt
        r["n_members"] = len(dec)
        results["+".join(combo)] = r
        print(f"{'+'.join(combo):38s}{n_ckpt:>6}{r['decoder_only']:>9.4f}"
              f"{r['composition_only']:>8.4f}{r['best_auc']:>9.4f}{r['best_w']:>5.1f}")

    # Paired bootstrap: the best combined set against each single set.
    best = max(results, key=lambda k: results[k]["best_auc"])
    print(f"\npaired bootstrap over pockets, {args.n_boot} resamples -- '{best}' vs each set")
    print("(uses each side's own best w, so it is the optimistic comparison for both)")
    for name in names:
        if name == best:
            continue
        deltas = bootstrap_pairs(
            (np.concatenate([sets[n]["decoder"] for n in best.split("+")]),
             np.concatenate([sets[n]["composition"] for n in best.split("+")])),
            (sets[name]["decoder"], sets[name]["composition"]),
            positive, valid, results[best]["best_w"], results[name]["best_w"],
            args.n_boot, rng,
        )
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        frac = float((deltas > 0).mean())
        verdict = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
        print(f"  vs {name:22s} delta {deltas.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(>0)={frac:.2f}  {verdict}")

    if args.output:
        serialisable = {k: {kk: (vv if not isinstance(vv, dict)
                                 else {str(a): b for a, b in vv.items()})
                            for kk, vv in v.items()} for k, v in results.items()}
        with open(args.output, "w") as handle:
            json.dump(serialisable, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
