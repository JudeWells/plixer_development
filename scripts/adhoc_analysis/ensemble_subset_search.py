"""Post-hoc: which ensemble members actually combine favourably?

Reads the member matrices saved by poc2mol_ensembling.py, so no model forwards are re-run.
Every member is already column z-normalised, so averaging is unweighted and a member with a
wide dynamic range cannot dominate -- see that script's NORMALISATION note.

Reports:
  1. per-member solo AUC, so a weak member is visible before it is blended in;
  2. the member-member correlation matrix of scores, since ensembling pays only to the extent
     members are decorrelated -- highly correlated members average to one member;
  3. greedy forward selection, which is the honest way to ask "which subset combines well"
     without enumerating 2^N;
  4. the aug-collapsed view: average each model's augmentations FIRST, then treat models as
     the ensemble units, which separates "averaging noise" from "averaging models".

⚠️ Greedy selection on the same panel it is scored on OVERFITS -- the reported best subset is
optimistically biased. A held-out pocket split is the fix; the split-half number below is the
honest estimate (select on half, score on the other).

Usage: python scripts/adhoc_analysis/ensemble_subset_search.py --matrices ens2_members.npz
"""
from __future__ import annotations

import argparse, itertools, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.likelihood_eval import per_pocket_auc  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrices", required=True)
    p.add_argument("--max_subset", type=int, default=8)
    args = p.parse_args()

    d = np.load(args.matrices, allow_pickle=True)
    Z, keys = d["members"], [str(k) for k in d["member_keys"]]
    pos, valid = d["positive"], d["valid"]
    N, P, Cn = Z.shape
    print(f"{N} members, {P} pockets, {Cn} candidates\n")

    def auc(m):
        return per_pocket_auc(m, pos, valid)[0]

    solo = np.array([auc(Z[i]) for i in range(N)])
    print("--- solo AUC per member ---")
    for i in np.argsort(-solo):
        print(f"  {keys[i]:<22}{solo[i]:.4f}")
    print(f"  mean {solo.mean():.4f}  sd {solo.std():.4f}")

    # correlation between members, over the scored entries only
    flat = Z[:, :, valid].reshape(N, -1)
    corr = np.corrcoef(flat)
    off = corr[np.triu_indices(N, 1)]
    print(f"\n--- member-member score correlation ---")
    print(f"  mean {off.mean():.3f}   min {off.min():.3f}   max {off.max():.3f}")
    print("  (near 1.0 => redundant members; ensembling gains little)")

    # aug-collapsed: average each model's augmentations first
    models = sorted({k.split("_aug")[0] for k in keys})
    coll, coll_names = [], []
    for m in models:
        ix = [i for i, k in enumerate(keys) if k.startswith(m + "_aug")]
        coll.append(Z[ix].mean(axis=0)); coll_names.append(m)
    coll = np.stack(coll)
    coll_solo = np.array([auc(c) for c in coll])
    print(f"\n--- aug-collapsed ({len(models)} models, augs averaged first) ---")
    for i in np.argsort(-coll_solo):
        print(f"  {coll_names[i]:<22}{coll_solo[i]:.4f}")
    print(f"  all models averaged   {auc(coll.mean(axis=0)):.4f}")

    # greedy forward selection over the aug-collapsed models
    def greedy(pool, score_idx=None):
        chosen, best, trace = [], -1, []
        remaining = list(range(len(pool)))
        while remaining and len(chosen) < args.max_subset:
            cand = [(auc(pool[chosen + [j]].mean(axis=0)), j) for j in remaining]
            s, j = max(cand)
            if s <= best:
                break
            best = s; chosen.append(j); remaining.remove(j)
            trace.append((len(chosen), coll_names[j], s))
        return chosen, best, trace

    chosen, best, trace = greedy(coll)
    print(f"\n--- greedy forward selection (⚠️ selected AND scored on the same pockets) ---")
    for n, name, s in trace:
        print(f"  +{name:<21} n={n}  AUC={s:.4f}")
    print(f"  best subset: {[coll_names[j] for j in chosen]}  AUC={best:.4f}")

    # honest estimate: select on half the pockets, score on the other half
    rng = np.random.default_rng(0)
    perm = rng.permutation(P); h = P // 2
    halves = [(perm[:h], perm[h:]), (perm[h:], perm[:h])]
    outs = []
    for sel, sco in halves:
        def auc_sub(m, rows):
            return per_pocket_auc(m[rows], pos[rows], valid)[0]
        chosen2, remaining, best2 = [], list(range(len(coll))), -1
        while remaining and len(chosen2) < args.max_subset:
            s, j = max((auc_sub(coll[chosen2 + [k]].mean(axis=0), sel), k) for k in remaining)
            if s <= best2:
                break
            best2 = s; chosen2.append(j); remaining.remove(j)
        outs.append(auc_sub(coll[chosen2].mean(axis=0), sco))
    print(f"\n--- split-half honest estimate ---")
    print(f"  select on half, score on the other: {np.mean(outs):.4f}  (folds {outs[0]:.4f}/{outs[1]:.4f})")
    print(f"  all-models baseline on same panel : {auc(coll.mean(axis=0)):.4f}")
    print("  If greedy >> split-half, the subset choice was fitting noise; prefer averaging all.")


if __name__ == "__main__":
    main()
