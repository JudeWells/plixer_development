"""Report benchmark AUCs for the full chronological test split and its PLINDER / seq-sim subsets.

Takes the per-model matrices written by `benchmark_test_set.py`, ensembles them, fuses with the
composition readout, and reports each result on:

    chronological  943 systems  -- the whole split
    PLINDER        107          -- rows of the same matrix
    seq-sim        141          -- rows of the same matrix

**Subsets are ROW slices, so the candidate panel stays 943 for all three.** Re-scoring a subset
against its own smaller panel would make it an easier ranking problem -- with 107 candidates a
random ranker still gets 0.5, but the variance and the achievable ceiling both change, and the
three numbers would not be comparable to each other. Keeping the panel fixed means a difference
between splits is a difference in *pocket difficulty*, which is the thing being asked about.

⚠️ PLINDER and seq-sim overlap by 94 of their 107 and 141 systems, so they are not independent
samples; do not treat a difference between them as if it were.

The composition readout is rebuilt here rather than stored, from the per-pocket predicted channel
masses and the per-candidate channel counts, matching `fusion_ensemble.py`: score a candidate by
the negative L1 distance between its channel counts and the pocket's predicted channel mass,
after scaling. Both readouts are column z-normalised before averaging or blending, which is the
transform `likelihood_auc_znorm` itself applies.
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


def composition_matrix(mass, candidate_counts):
    """Pocket x candidate score from channel occupancy alone -- no decoder involved.

    Mirrors fusion_ensemble.py: each pocket's predicted per-channel mass is matched against each
    candidate's per-channel atom counts. The mass is on a voxel scale and the counts are integers,
    so the pocket's mass vector is rescaled to the counts' total before the L1 distance is taken;
    otherwise the comparison would be dominated by the units rather than the composition.
    """
    scores = np.empty((mass.shape[0], candidate_counts.shape[0]))
    totals = candidate_counts.sum(axis=1, keepdims=True).clip(min=1)
    normalised_counts = candidate_counts / totals
    for i in range(mass.shape[0]):
        vector = mass[i]
        vector = vector / max(vector.sum(), 1e-8)
        scores[i] = -np.abs(normalised_counts - vector[None, :]).sum(axis=1)
    return scores


def auc(matrix, positive, valid):
    return per_pocket_auc(matrix, positive, valid)[0]


def evaluate(decoder_z, composition_z, positive, valid, rows, weights):
    d = decoder_z[:, rows, :].mean(axis=0)
    c = composition_z[:, rows, :].mean(axis=0)
    p, v = positive[rows], valid
    curve = {round(w, 2): auc((1 - w) * d + w * c, p, v) for w in weights}
    best_w = max(curve, key=curve.get)
    return {
        "n_pockets": int(len(rows)),
        "decoder": curve[0.0],
        "composition": curve[1.0],
        "fused_best": curve[best_w],
        "best_w": best_w,
        "fused_at_0.3": curve[0.3],
        "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrices", default="results/bench/*.npz")
    parser.add_argument("--ensemble", default="A1_e2e_warm,A2_e2e_anneal,A3_frozen,A4_sft",
                        help="comma-separated tags forming the headline ensemble")
    parser.add_argument("--plinder_csv", default="data/test_set_plinder_split.csv")
    parser.add_argument("--seqsim_csv", default="data/test_set_seq_sim_split.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.matrices))
    if not paths:
        raise SystemExit(f"no matrices matched {args.matrices}")

    loaded, reference = {}, None
    for path in paths:
        z = np.load(path, allow_pickle=True)
        tag = str(z["tag"][0])
        if reference is None:
            reference = z
        else:
            for key in ("panel", "positive", "valid_columns", "system_ids"):
                if not np.array_equal(z[key], reference[key]):
                    raise SystemExit(f"{tag}: '{key}' differs -- matrices are not combinable")
        loaded[tag] = z

    system_ids = [str(s) for s in reference["system_ids"]]
    positive = reference["positive"]
    valid = reference["valid_columns"]
    counts = reference["candidate_counts"]
    weights = [i / 10 for i in range(11)]

    plinder = set(pd.read_csv(args.plinder_csv).system_id)
    seqsim = set(pd.read_csv(args.seqsim_csv).system_id)
    subsets = {
        "chronological": list(range(len(system_ids))),
        "PLINDER": [i for i, s in enumerate(system_ids) if s in plinder],
        "seq-sim": [i for i, s in enumerate(system_ids) if s in seqsim],
    }
    print(f"panel {positive.shape[1]} candidates; pockets per split: "
          + ", ".join(f"{k} {len(v)}" for k, v in subsets.items()))
    overlap = len(set(subsets["PLINDER"]) & set(subsets["seq-sim"]))
    print(f"⚠️ PLINDER n seq-sim overlap: {overlap} systems -- the two subsets are NOT independent\n")

    # Per-tag z-normed matrices, computed once.
    dec_z = {t: znormalise_columns(z["decoder"][0]) for t, z in loaded.items()}
    com_z = {t: znormalise_columns(composition_matrix(z["mass"][0], counts))
             for t, z in loaded.items()}

    results = {}
    tags = list(loaded)
    ensemble_tags = [t for t in args.ensemble.split(",") if t in loaded]

    def report(name, member_tags):
        d = np.stack([dec_z[t] for t in member_tags])
        c = np.stack([com_z[t] for t in member_tags])
        row = {}
        for split, rows in subsets.items():
            row[split] = evaluate(d, c, positive, valid, rows, weights)
        results[name] = row
        return row

    print(f"{'model / set':28s}{'split':16s}{'n':>5}{'decoder':>9}{'comp':>8}{'FUSED':>9}{'@w':>5}")
    print("-" * 80)
    for tag in tags:
        row = report(tag, [tag])
        for split in subsets:
            r = row[split]
            print(f"{tag[:28]:28s}{split:16s}{r['n_pockets']:>5}{r['decoder']:>9.4f}"
                  f"{r['composition']:>8.4f}{r['fused_best']:>9.4f}{r['best_w']:>5.1f}")
        print()

    if len(ensemble_tags) > 1:
        row = report(f"ENSEMBLE({len(ensemble_tags)})", ensemble_tags)
        print("=" * 80)
        for split in subsets:
            r = row[split]
            print(f"{'ENSEMBLE '+str(len(ensemble_tags))+' members':28s}{split:16s}"
                  f"{r['n_pockets']:>5}{r['decoder']:>9.4f}{r['composition']:>8.4f}"
                  f"{r['fused_best']:>9.4f}{r['best_w']:>5.1f}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({k: {s: {kk: vv for kk, vv in r.items() if kk != "curve"}
                           for s, r in v.items()} for k, v in results.items()},
                      handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
