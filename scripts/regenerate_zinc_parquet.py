"""Precompute ZINC per-atom features + float32 coords, mirroring the HiQBind v2 schema.

Unlike HiQBind, nothing is lost in the current ZINC parquet -- it stores a full `mol_block`, so
the graph, atom order and coordinates are all present and consistent. This script is therefore
not a repair; it is a precomputation, and it buys two things:

1. THE NEW CHANNELS. `is_aromatic` / `n_hydrogens` / `is_acceptor` / `formal_charge` per atom, so
   the 11-channel scheme works identically on ZINC and HiQBind.
2. SPEED. The dataset currently calls `Chem.MolFromMolBlock` for EVERY sample
   (`datasets.py:293`), which CLAUDE.md §3h identifies as the stage-1 bottleneck -- ~790
   samples/s, data-bound. Doing the parse once offline removes RDKit from the hot path entirely,
   so the channel change should make the expensive stage-1 retrain FASTER, not slower.

The original `../zinc20_parquet` stays untouched and remains the archival source: the raw ZINC20
mol2 files are gone (CLAUDE.md §3), so those mol_blocks are the only copy. `mol_block` is
deliberately NOT carried into v2 -- keeping it would defeat the size and speed gain, and the
original is the backup.

Splits are preserved exactly: `index_train.csv` / `index_val.csv` assign files to splits, and
this writes new index files listing the same batches with corrected paths and actual row counts.

Usage:
    python scripts/regenerate_zinc_parquet.py --workers 48
    python scripts/regenerate_zinc_parquet.py --limit_files 4 --workers 4   # smoke test
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

# Identical definition to the HiQBind script -- the two datasets MUST use the same acceptor
# rule or the channel means differ between the ZINC and pocket halves of a combined batch.
ACCEPTOR = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]),$([O,S;H0;v2]),$([N;v3;!$(N-*=[O,N,P,S])]),$([nH0,o,s;+0])]"
)


def atom_features(mol):
    n = mol.GetNumAtoms()
    aromatic = np.zeros(n, np.uint8)
    n_hydrogens = np.zeros(n, np.uint8)
    acceptor = np.zeros(n, np.uint8)
    charge = np.zeros(n, np.int8)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        aromatic[i] = int(atom.GetIsAromatic())
        charge[i] = int(atom.GetFormalCharge())
        explicit = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
        n_hydrogens[i] = min(255, explicit + atom.GetTotalNumHs())
    for (idx,) in mol.GetSubstructMatches(ACCEPTOR):
        acceptor[idx] = 1
    return aromatic, n_hydrogens, acceptor, charge


def process_file(job):
    in_path, out_path = job
    try:
        df = pd.read_parquet(in_path, columns=["smiles", "mol_block"])
        records, failed = [], 0
        for smiles, block in zip(df["smiles"], df["mol_block"]):
            text = block.decode() if isinstance(block, (bytes, bytearray)) else block
            # removeHs=False to mirror HiQBind, whose stored atom list includes explicit H.
            # The voxeliser assigns H to no channel, so they cost storage only.
            mol = Chem.MolFromMolBlock(text, removeHs=False, sanitize=True)
            if mol is None or mol.GetNumConformers() == 0:
                failed += 1
                continue
            coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32).T  # (3, N)
            aromatic, n_h, acceptor, charge = atom_features(mol)
            records.append({
                "smiles": smiles,
                "ligand_coords": coords.reshape(-1),
                "ligand_coords_shape": list(coords.shape),
                "ligand_element_symbols": [a.GetSymbol() for a in mol.GetAtoms()],
                "ligand_is_aromatic": aromatic,
                "ligand_n_hydrogens": n_h,
                "ligand_is_acceptor": acceptor,
                "ligand_formal_charge": charge,
            })
        if records:
            pd.DataFrame(records).to_parquet(out_path, index=False)
        return os.path.basename(in_path), len(records), failed, None
    except Exception:
        return os.path.basename(in_path), 0, 0, traceback.format_exc(limit=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", default="../zinc20_parquet")
    p.add_argument("--out_dir", default="../zinc20_parquet_v2")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit_files", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    in_files = sorted(glob.glob(os.path.join(args.in_dir, "*.parquet")))
    if args.limit_files:
        in_files = in_files[: args.limit_files]
    jobs = [(f, os.path.join(args.out_dir, os.path.basename(f))) for f in in_files]
    print(f"{len(jobs)} files -> {args.out_dir}", flush=True)

    counts, total_rows, total_failed, errors = {}, 0, 0, []
    # One pool for the whole run; parallelise per FILE so each worker's RDKit import is amortised
    # over ~1,600 molecules instead of one.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, n_rows, n_failed, err) in enumerate(pool.map(process_file, jobs, chunksize=1)):
            if err:
                errors.append((name, err))
            counts[name] = n_rows
            total_rows += n_rows
            total_failed += n_failed
            if i % 200 == 0:
                print(f"  [{i+1}/{len(jobs)}] rows={total_rows:,} failed={total_failed}", flush=True)

    # Rewrite the split index files. They define train/val at FILE level (60 of 5537 held out,
    # provably disjoint -- CLAUDE.md §3h), so preserving them preserves the split exactly.
    for index_name in ("index_train.csv", "index_val.csv", "index.csv"):
        src = os.path.join(args.in_dir, index_name)
        if not os.path.exists(src):
            continue
        idx = pd.read_csv(src)
        rows = []
        for _, r in idx.iterrows():
            base = os.path.basename(r["parquet_file"])
            if base in counts and counts[base] > 0:
                rows.append({"parquet_file": os.path.join(args.out_dir, base),
                             "file_size": counts[base]})
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, index_name), index=False)
        print(f"  wrote {index_name}: {len(rows)} files, {sum(r['file_size'] for r in rows):,} molecules")

    print(f"\nfiles written : {sum(1 for v in counts.values() if v)}")
    print(f"molecules     : {total_rows:,}")
    print(f"unparseable   : {total_failed:,}")
    if errors:
        print(f"file errors   : {len(errors)}")
        for name, err in errors[:3]:
            print(f"   {name}: {err.splitlines()[-1]}")


if __name__ == "__main__":
    main()
