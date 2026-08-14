"""Benchmark Boltz-2's affinity head on the same hit-vs-decoy ranking task as Plixer.

Plixer scores a pocket against a candidate panel with one cheap forward pass per candidate.
Boltz-2 has no such shortcut: its affinity head reads a predicted complex, so every
(pocket, candidate) pair needs its own structure prediction. Measured here at ~10 s/pair on an
H100 for a 500-residue protein, which is what forces the panel to be sized deliberately rather
than inherited from the Plixer benchmark's 943 candidates.

THE COMPARISON IS MADE FAIR BY SHRINKING PLIXER, NOT BY GROWING BOLTZ. Plixer's 943x943 matrix
is already computed, so slicing it to whatever (pockets x candidates) sub-panel Boltz-2 can
afford costs nothing and both models are then scored on identical pockets, identical candidates
and identical positives. Scoring the two on different panels would be meaningless: AUC depends
on how many candidates there are and what they are.

MSAs ARE GENERATED ONCE PER PROTEIN AND REUSED. Boltz-2's affinity module expects an MSA;
running it single-sequence (`msa: empty`) would handicap it and invite the obvious objection.
But re-querying the MSA server for all N_candidates YAMLs of the same protein would be
pointless and would hammer a public service. Phase A runs one prediction per pocket WITH
`--use_msa_server`, which yields both the pocket's own true-ligand affinity and a processed
`<name>_0.csv` MSA; phase B builds the remaining pairs referencing that file.

Subcommands:
  prepare-msa    write phase-A YAMLs (one per pocket, its true ligand, MSA from the server)
  prepare-pairs  write phase-B YAMLs for every remaining (pocket, candidate), reusing MSAs
  collect        read the affinity JSONs into a matrix and score it against Plixer
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.likelihood_eval import per_pocket_auc, znormalise_columns   # noqa: E402


def load_panel(args):
    """The pockets to score and the candidate panel, both drawn from one split file."""
    chrono = pd.read_csv(args.chrono_csv).set_index("system_id")
    subset = pd.read_csv(args.subset_csv)
    system_ids = [s for s in subset.system_id if s in chrono.index][: args.max_pockets]
    # Panel = the true ligands of the FULL subset (not just the scored pockets), so a pilot on
    # fewer pockets still ranks against the whole subset's chemistry.
    panel_ids = [s for s in subset.system_id if s in chrono.index][: args.panel_size]
    for s in system_ids:                      # every scored pocket's own ligand must be present
        if s not in panel_ids:
            panel_ids.append(s)
    panel = [chrono.loc[s].smiles for s in panel_ids]
    return chrono, system_ids, panel_ids, panel


def write_yaml(path, sequence, smiles, msa=None):
    protein = {"id": "A", "sequence": sequence}
    if msa is not None:
        protein["msa"] = msa
    doc = {"version": 1,
           "sequences": [{"protein": protein}, {"ligand": {"id": "B", "smiles": smiles}}],
           "properties": [{"affinity": {"binder": "B"}}]}
    yaml.safe_dump(doc, open(path, "w"), sort_keys=False)


def cmd_prepare_msa(args):
    chrono, system_ids, panel_ids, panel = load_panel(args)
    out = os.path.join(args.work, "phaseA")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for sid in system_ids:
        row = chrono.loc[sid]
        write_yaml(os.path.join(out, f"{sid}.yaml"), row.protein_sequence, row.smiles)
    print(f"phase A: {len(system_ids)} YAMLs -> {out}")
    print("run with --use_msa_server; this yields each pocket's true-ligand affinity AND its MSA")


def cmd_prepare_pairs(args):
    chrono, system_ids, panel_ids, panel = load_panel(args)
    # Globbed rather than a fixed path: phase A is sharded across GPUs, so boltz writes one
    # results dir per shard (boltz_results_phaseA_0, _1, ...) and the MSA could be in any.
    msa_index = {os.path.basename(p)[:-6]: p
                 for p in glob.glob(os.path.join(args.work, "**", "msa", "*_0.csv"),
                                    recursive=True)}
    out = os.path.join(args.work, "phaseB")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    written, missing_msa = 0, []
    for sid in system_ids:
        row = chrono.loc[sid]
        msa = msa_index.get(sid)
        if msa is None:
            missing_msa.append(sid)
            continue
        for j, (pid, smiles) in enumerate(zip(panel_ids, panel)):
            if pid == sid:
                continue                      # the true ligand was scored in phase A
            write_yaml(os.path.join(out, f"{sid}__{j:04d}.yaml"),
                       row.protein_sequence, smiles, msa=os.path.abspath(msa))
            written += 1
    print(f"phase B: {written} YAMLs -> {out}")
    if missing_msa:
        print(f"⚠️ {len(missing_msa)} pockets have no MSA and were skipped: {missing_msa[:5]}")
        print("   those pockets cannot be scored; re-run phase A for them before collecting")


def read_affinity(path):
    """Boltz-2 writes log(IC50) in uM: LOWER is stronger binding. Return a score where HIGHER
    is better, so it composes with per_pocket_auc like every other readout here."""
    with open(path) as handle:
        blob = json.load(handle)
    return {
        "neg_pred_value": -float(blob["affinity_pred_value"]),
        "prob_binary": float(blob["affinity_probability_binary"]),
    }


def cmd_collect(args):
    chrono, system_ids, panel_ids, panel = load_panel(args)
    index_of = {p: j for j, p in enumerate(panel_ids)}

    scores = {k: np.full((len(system_ids), len(panel_ids)), np.nan)
              for k in ("neg_pred_value", "prob_binary")}
    found = 0
    # One recursive glob covers both phases and any sharding: the prediction directory is named
    # after the YAML, which is "<system_id>" in phase A and "<system_id>__<panel_index>" in B.
    for path in glob.glob(os.path.join(args.work, "**", "predictions", "*", "affinity_*.json"),
                          recursive=True):
        name = os.path.basename(os.path.dirname(path))
        if "__" in name:
            sid, _, jj = name.partition("__")
            j = int(jj) if jj.isdigit() else None
        else:
            sid, j = name, index_of.get(name)
        if sid in system_ids and j is not None:
            i = system_ids.index(sid)
            values = read_affinity(path)
            for key in scores:
                scores[key][i, j] = values[key]
            found += 1

    total = len(system_ids) * len(panel_ids)
    print(f"collected {found} affinity predictions "
          f"({found / total:.1%} of {len(system_ids)}x{len(panel_ids)} cells)")

    positive = np.zeros((len(system_ids), len(panel_ids)), dtype=bool)
    for i, sid in enumerate(system_ids):
        target = chrono.loc[sid].smiles
        positive[i] = np.array([s == target for s in panel])

    # A pocket is scorable only if its own ligand and at least one decoy came back.
    usable = np.array([
        (~np.isnan(scores["prob_binary"][i]) & positive[i]).any()
        and (~np.isnan(scores["prob_binary"][i]) & ~positive[i]).sum() >= 5
        for i in range(len(system_ids))
    ])
    print(f"pockets scorable (own ligand + >=5 decoys present): {int(usable.sum())}"
          f"/{len(system_ids)}")
    if not usable.any():
        raise SystemExit("nothing scorable yet -- let the runs finish")

    valid = np.ones(len(panel_ids), dtype=bool)
    results = {}
    for key, matrix in scores.items():
        # ⚠️ DO NOT impute missing cells. Phase A scores every pocket's TRUE ligand and finishes
        # long before phase B has scored the decoys, so during a partial run the positive is
        # always present while most negatives are absent. Filling those with the row minimum --
        # which is safe when cells are missing at random -- pins the decoys to the floor and the
        # true ligand ranks top by construction: it read AUC 0.947 at 21% completion, versus
        # 0.696 for the same model on the complete pilot. Score only the cells that exist.
        aucs = []
        for i in range(matrix.shape[0]):
            if not usable[i]:
                continue
            have = ~np.isnan(matrix[i])
            pos_scores = matrix[i][have & positive[i]]
            neg_scores = matrix[i][have & ~positive[i]]
            if len(pos_scores) == 0 or len(neg_scores) < args.min_decoys:
                continue
            wins = (pos_scores[:, None] > neg_scores[None, :]).sum() \
                 + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).sum()
            aucs.append(float(wins / (len(pos_scores) * len(neg_scores))))
        raw = float(np.mean(aucs)) if aucs else float("nan")

        # z-norm needs a column mean over pockets; with partial data compute it from the cells
        # present, and again score only those.
        col_mean = np.nanmean(matrix, axis=0)
        col_sd = np.nanstd(matrix, axis=0)
        col_sd[~np.isfinite(col_sd) | (col_sd < 1e-12)] = 1.0
        zmat = (matrix - col_mean) / col_sd
        zaucs = []
        for i in range(zmat.shape[0]):
            if not usable[i]:
                continue
            have = ~np.isnan(zmat[i])
            ps = zmat[i][have & positive[i]]
            ns = zmat[i][have & ~positive[i]]
            if len(ps) == 0 or len(ns) < args.min_decoys:
                continue
            wins = (ps[:, None] > ns[None, :]).sum() + 0.5 * (ps[:, None] == ns[None, :]).sum()
            zaucs.append(float(wins / (len(ps) * len(ns))))
        znorm = float(np.mean(zaucs)) if zaucs else float("nan")

        results[key] = {"raw": raw, "znorm": znorm, "pockets_scored": len(zaucs),
                        "mean_decoys_per_pocket": float(np.nanmean(
                            (~np.isnan(matrix) & ~positive).sum(axis=1)[usable]))}
        print(f"  Boltz-2 {key:16s} raw AUC {raw:.4f}   znorm AUC {znorm:.4f}   "
              f"({len(zaucs)} pockets, {results[key]['mean_decoys_per_pocket']:.0f} decoys each)")

    if args.plixer_matrices:
        compare_plixer(args, system_ids, panel_ids, panel, positive, usable, results)

    if args.output:
        json.dump({"n_pockets": int(usable.sum()), "n_candidates": len(panel_ids),
                   "found_cells": found, "boltz": results},
                  open(args.output, "w"), indent=2)
        print(f"\nwrote {args.output}")


def compare_plixer(args, system_ids, panel_ids, panel, positive, usable, results):
    """Slice Plixer's precomputed 943x943 matrix to exactly this sub-panel."""
    paths = sorted(glob.glob(args.plixer_matrices))
    if not paths:
        print(f"\n(no Plixer matrices matched {args.plixer_matrices})")
        return
    members = [np.load(p, allow_pickle=True) for p in paths]
    all_ids = [str(s) for s in members[0]["system_ids"]]
    all_panel = [str(s) for s in members[0]["panel"]]

    rows = [all_ids.index(s) for s in system_ids if s in all_ids]
    cols = [all_panel.index(smiles) for smiles in panel if smiles in all_panel]
    if len(rows) != len(system_ids) or len(cols) != len(panel):
        print(f"\n⚠️ could only align {len(rows)}/{len(system_ids)} pockets and "
              f"{len(cols)}/{len(panel)} candidates -- comparison skipped")
        return

    sub_pos = positive[:, [panel.index(all_panel[c]) for c in cols]]
    stack = np.stack([znormalise_columns(m["decoder"][0])[np.ix_(rows, cols)] for m in members])
    ensemble = stack.mean(axis=0)
    print(f"\nPlixer on the SAME {int(usable.sum())} pockets x {len(cols)} candidates:")
    for label, matrix in [("single (first member)", stack[0]), (f"ensemble ({len(members)})", ensemble)]:
        auc = per_pocket_auc(matrix[usable], sub_pos[usable], np.ones(len(cols), bool))[0]
        print(f"  Plixer {label:22s} znorm AUC {auc:.4f}")
        results.setdefault("plixer", {})[label] = auc


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare-msa", "prepare-pairs", "collect"])
    parser.add_argument("--work", default="../boltz_bench/run")
    parser.add_argument("--chrono_csv", default="data/test_set_chronological_split.csv")
    parser.add_argument("--subset_csv", default="data/test_set_plinder_split.csv")
    parser.add_argument("--max_pockets", type=int, default=20)
    parser.add_argument("--panel_size", type=int, default=40)
    parser.add_argument("--plixer_matrices", default="results/bench/*.npz")
    parser.add_argument("--min_decoys", type=int, default=20,
                        help="skip a pocket unless this many decoys have been scored; "
                             "guards against a partial run flattering the positive")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.makedirs(args.work, exist_ok=True)
    {"prepare-msa": cmd_prepare_msa,
     "prepare-pairs": cmd_prepare_pairs,
     "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
