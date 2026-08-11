"""Regenerate the HiQBind parquet from the raw structures: float32 coords + per-atom features.

Two things are wrong with the existing parquet, both fixed here.

1. COORDINATES ARE bfloat16-QUANTISED. `create_hiqbind_dataset.py` parses through docktgrid's
   MolecularParser, whose `molparser.py:79` does `torch.tensor(coords, dtype=DTYPE)` -- and
   `DTYPE = torch.bfloat16` is our own site-packages patch (CLAUDE.md §1). Verified:
   `max |parquet - bfloat16(sdf)| = 0.000000` exactly, against 0.249 A vs true float64.
   Measured error: 0.141 A mean worst-case per ligand, up to 0.492 A; protein coords reach
   133 A magnitude where bfloat16 spacing is 0.5 A. Against a 0.75 A voxel that is up to
   two-thirds of a voxel of jitter on BOTH the model's input and its target. §3b bug 3 fixed
   the bfloat16 *arithmetic* in the voxeliser but could not recover precision already destroyed
   on disk.

2. THE BOND GRAPH WAS DISCARDED. Only coords + element symbols were kept, so aromaticity,
   H-counts and H-bond-acceptor status -- the basis of the new channel scheme -- are not
   recoverable. Canonical-SMILES order matches the stored order only 3.1% of the time, so they
   cannot be mapped back after the fact. From the raw SDF the order matches 100%.

Design: store per-atom FEATURES, not channel ids, so the channel grouping stays a runtime config
choice and changing it never requires reprocessing.

Two safeguards, neither optional:
  * PIN to the existing system_id list and carry split / cluster / *_cluster_id across unchanged,
    so every previously measured result stays comparable. Nothing is re-clustered.
  * VALIDATE by bfloat16-rounding the regenerated coordinates and requiring an exact match to the
    existing parquet. That proves we are reading the same source version end to end -- not merely
    the same tarball.

Usage:
    python scripts/regenerate_hiqbind_parquet.py --workers 32
    python scripts/regenerate_hiqbind_parquet.py --limit_files 2 --workers 4   # smoke test
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------------
# CRITICAL: force float32 BEFORE docktgrid's parser binds DTYPE. `molparser.py` does
# `from .config import DTYPE`, which copies the value at import time, so patching
# docktgrid.config afterwards would be too late -- hence the module-level rebind as well.
# Using the same parser (rather than a hand-rolled one) is deliberate: it guarantees the
# regenerated files contain exactly the same atoms in the same order as the originals.
# ---------------------------------------------------------------------------------------
import docktgrid.config as _dg_config  # noqa: E402
_dg_config.DTYPE = torch.float32
import docktgrid.molparser as _dg_molparser  # noqa: E402
_dg_molparser.DTYPE = torch.float32
from docktgrid.molparser import MolecularParser  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

# Acceptor definition. Deliberately excludes amide N (lone pair delocalised into the carbonyl,
# so not an acceptor) -- ~0.5 per ligand, 29% of all N-without-H. That exclusion is the entire
# reason this is stored rather than derived from element + H-count at load time.
ACCEPTOR = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]),$([O,S;H0;v2]),$([N;v3;!$(N-*=[O,N,P,S])]),$([nH0,o,s;+0])]"
)

PINNED = ["system_id", "smiles", "split", "cluster", "protein_cluster_id", "ligand_cluster_id"]


def ligand_atom_features(mol):
    """Per-atom features in the molecule's own atom order.

    Order matters: MolToPDBFile writes atoms in mol order and the ligand is all-HETATM, so the
    parser's output order equals this order. Verified empirically at 100% on 192 systems.
    """
    n = mol.GetNumAtoms()
    aromatic = np.zeros(n, dtype=np.uint8)
    n_hydrogens = np.zeros(n, dtype=np.uint8)
    acceptor = np.zeros(n, dtype=np.uint8)
    charge = np.zeros(n, dtype=np.int8)

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        aromatic[i] = int(atom.GetIsAromatic())
        charge[i] = int(atom.GetFormalCharge())
        # The SDF carries explicit hydrogens, so count bonded H directly and add any implicit
        # ones; GetTotalNumHs() alone reports 0 when all H are explicit.
        explicit = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
        n_hydrogens[i] = min(255, explicit + atom.GetTotalNumHs())

    for (idx,) in mol.GetSubstructMatches(ACCEPTOR):
        acceptor[idx] = 1
    return aromatic, n_hydrogens, acceptor, charge


def load_ligand(system_dir):
    """Read the ligand SDF exactly as create_hiqbind_dataset.py did, plus a sanitized copy."""
    paths = (glob.glob(os.path.join(system_dir, "*_ligand_refined.sdf"))
             or glob.glob(os.path.join(system_dir, "*_ligand.sdf"))
             or glob.glob(os.path.join(system_dir, "*_ligand.pdb")))
    if len(paths) != 1:
        return None, None
    path = paths[0]
    if path.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        raw = supplier[0] if len(supplier) else None
    else:
        raw = Chem.MolFromPDBFile(path, removeHs=False)
    if raw is None:
        return None, None
    # Aromaticity and the acceptor SMARTS both need sanitization; keep the unsanitized molecule
    # for geometry so a sanitization failure costs features, not the whole system.
    featured = Chem.Mol(raw)
    try:
        Chem.SanitizeMol(featured)
    except Exception:
        featured = None
    return raw, featured


def process_row(args):
    system_id, raw_root = args
    try:
        system_dir = os.path.join(raw_root, system_id.split("_")[0], system_id)
        protein_paths = (glob.glob(os.path.join(system_dir, "*_protein_refined.pdb"))
                         or glob.glob(os.path.join(system_dir, "*_protein.pdb")))
        if len(protein_paths) != 1:
            return system_id, None, "protein file missing"
        raw_mol, featured_mol = load_ligand(system_dir)
        if raw_mol is None:
            return system_id, None, "ligand unreadable"

        parser = MolecularParser()
        with tempfile.TemporaryDirectory() as tmp:
            ligand_pdb = os.path.join(tmp, "ligand.pdb")
            Chem.MolToPDBFile(raw_mol, ligand_pdb)
            protein = parser.parse_file(protein_paths[0], ".pdb")
            ligand = parser.parse_file(ligand_pdb, ".pdb")

        protein_coords = np.asarray(protein.coords, dtype=np.float32)
        ligand_coords = np.asarray(ligand.coords, dtype=np.float32)
        n_lig = ligand_coords.shape[1]

        if featured_mol is not None and featured_mol.GetNumAtoms() == n_lig:
            aromatic, n_h, acceptor, charge = ligand_atom_features(featured_mol)
            features_ok = True
        else:
            aromatic = np.zeros(n_lig, np.uint8); n_h = np.zeros(n_lig, np.uint8)
            acceptor = np.zeros(n_lig, np.uint8); charge = np.zeros(n_lig, np.int8)
            features_ok = False

        return system_id, {
            "protein_coords": protein_coords.reshape(-1),
            "protein_coords_shape": list(protein_coords.shape),
            "protein_element_symbols": list(protein.element_symbols),
            "ligand_coords": ligand_coords.reshape(-1),
            "ligand_coords_shape": list(ligand_coords.shape),
            "ligand_element_symbols": list(ligand.element_symbols),
            "ligand_is_aromatic": aromatic,
            "ligand_n_hydrogens": n_h,
            "ligand_is_acceptor": acceptor,
            "ligand_formal_charge": charge,
            "features_ok": features_ok,
        }, None
    except Exception:
        return args[0], None, traceback.format_exc(limit=2)


def validate(new, old, system_id):
    """Regenerated coords must equal the stored ones after bfloat16 rounding, exactly."""
    problems = []
    for key in ("ligand_coords", "protein_coords"):
        a = np.asarray(new[key], dtype=np.float32)
        b = np.asarray(old[key], dtype=np.float32)
        if a.shape != b.shape:
            problems.append(f"{key}: shape {a.shape} vs {b.shape}")
            continue
        rounded = torch.tensor(a).to(torch.bfloat16).to(torch.float32).numpy()
        if not np.array_equal(rounded, b):
            problems.append(f"{key}: max|bf16(new)-old| = {np.abs(rounded-b).max():.4f}")
    if list(new["ligand_element_symbols"]) != list(old["ligand_element_symbols"]):
        problems.append("ligand element symbols differ")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="../hiqbind/raw_data_hiq_sm")
    p.add_argument("--in_dir", default="../hiqbind/parquet")
    p.add_argument("--out_dir", default="../hiqbind/parquet_v2")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit_files", type=int, default=None)
    p.add_argument("--skip_validation", action="store_true")
    args = p.parse_args()

    totals = dict(rows=0, written=0, missing=0, feat_fail=0, validation_fail=0)
    validation_examples = []

    # ONE pool for the whole run. Creating it per input file spawns 16 workers x 983 files,
    # and every worker re-imports torch + rdkit + docktgrid -- startup then dwarfs the actual
    # work, which is only ~0.04 s/system.
    pool = ProcessPoolExecutor(max_workers=args.workers)

    for split in ("train", "val", "test"):
        in_files = sorted(glob.glob(os.path.join(args.in_dir, split, "*.parquet")))
        if args.limit_files:
            in_files = in_files[: args.limit_files]
        out_split = os.path.join(args.out_dir, split)
        os.makedirs(out_split, exist_ok=True)
        print(f"\n=== {split}: {len(in_files)} files ===", flush=True)

        for file_index, in_file in enumerate(in_files):
            df = pd.read_parquet(in_file)
            totals["rows"] += len(df)
            jobs = [(sid, args.raw) for sid in df["system_id"]]
            results = {}
            for sid, payload, err in pool.map(process_row, jobs, chunksize=4):
                if payload is None:
                    totals["missing"] += 1
                else:
                    results[sid] = payload

            records = []
            for _, row in df.iterrows():
                payload = results.get(row["system_id"])
                if payload is None:
                    continue
                if not payload.pop("features_ok"):
                    totals["feat_fail"] += 1
                if not args.skip_validation:
                    problems = validate(payload, row, row["system_id"])
                    if problems:
                        totals["validation_fail"] += 1
                        if len(validation_examples) < 5:
                            validation_examples.append((row["system_id"], problems))
                        continue
                records.append({**{k: row[k] for k in PINNED}, **payload})

            if records:
                pd.DataFrame(records).to_parquet(
                    os.path.join(out_split, os.path.basename(in_file)), index=False)
                totals["written"] += len(records)
            if file_index % 25 == 0:
                print(f"  [{file_index+1}/{len(in_files)}] written={totals['written']} "
                      f"missing={totals['missing']} validation_fail={totals['validation_fail']}",
                      flush=True)

    pool.shutdown()

    print("\n================ summary ================")
    for k, v in totals.items():
        print(f"  {k:18s} {v}")
    if validation_examples:
        print("\n  validation failures (first few):")
        for sid, problems in validation_examples:
            print(f"    {sid}: {problems}")
        print("\n  A bfloat16-rounding mismatch means the raw files are NOT the version that")
        print("  built the current parquet. Do not use the output until that is resolved.")
    elif not args.skip_validation:
        print("\n  All regenerated coordinates round-trip to the stored ones exactly.")
        print("  Same source version confirmed end to end; coords are now float32.")


if __name__ == "__main__":
    main()
