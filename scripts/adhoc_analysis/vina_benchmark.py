"""Cross-dock every candidate ligand into every pocket with AutoDock Vina, and score the ranking.

Same task as the Plixer and Boltz-2 benchmarks: for each pocket, rank a shared candidate panel
and ask how highly the true binder places. Vina supplies the classical-docking baseline that any
virtual-screening claim is expected to beat.

RECEPTOR PREPARATION, AND WHY IT IS DEFENSIBLE HERE
--------------------------------------------------
The raw structures are not on this node; `../hiqbind/parquet_v2` stores protein coordinates and
element symbols only -- no residue names, no atom names, no bonds. That sounds fatal for docking
and mostly is not, because **Vina's scoring function does not use partial charges** (that is
AutoDock4). Vina needs atom types plus hydrogen-bond donor/acceptor classification, and:

  * the parquet retains EXPLICIT HYDROGENS, so donors are identified geometrically -- a polar
    hydrogen is one within 1.25 A of an N or O;
  * Vina collapses aliphatic and aromatic carbon (C and A) to the same hydrophobic term, so the
    absence of ring perception costs nothing in the scoring function;
  * acceptors (NA/OA/SA) follow from element identity.

What is genuinely lost relative to template-based preparation: per-residue protonation decisions
already baked into the coordinates (we inherit whatever HiQBind produced, which is reasonable),
and any metal/cofactor typing. This is a fair-but-not-pristine receptor, and the result should be
read as such. If properly prepared poses exist elsewhere, prefer them.

The box is centred on the pocket's TRUE ligand centroid and sized to its extent plus a margin,
so every candidate is docked into the same site -- this is re-docking the cognate ligand and
cross-docking all the others, which is what the ranking task requires.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

POLAR_CUTOFF = 1.25          # A; H within this of N/O is polar (donor hydrogen, AD type HD)
ACCEPTOR = {"N": "NA", "O": "OA", "S": "SA"}


def receptor_pdbqt(coords, elements, out_path):
    """Minimal AutoDock receptor. Nonpolar hydrogens are dropped (Vina is united-atom for
    those); polar hydrogens become HD so donors are represented."""
    coords = np.asarray(coords, dtype=float)
    elements = [str(e).strip().capitalize() for e in elements]

    heavy = [i for i, e in enumerate(elements) if e != "H"]
    hydrogens = [i for i, e in enumerate(elements) if e == "H"]
    heavy_xyz = coords[heavy] if heavy else np.zeros((0, 3))

    polar_h = []
    if hydrogens and len(heavy):
        for h in hydrogens:
            d = np.linalg.norm(heavy_xyz - coords[h], axis=1)
            j = int(d.argmin())
            if d[j] <= POLAR_CUTOFF and elements[heavy[j]] in ("N", "O"):
                polar_h.append(h)

    lines, serial = [], 1
    for i in heavy:
        e = elements[i]
        ad = ACCEPTOR.get(e, "C" if e == "C" else e.upper()[:2])
        x, y, z = coords[i]
        # The charge field is TEN characters wide, not nine. Vina's PDBQT parser is strict
        # about it and rejects the file with 'Charge "0.000 " is not valid' -- a one-character
        # error that costs the whole run. Matched against a Meeko-written ligand PDBQT.
        lines.append(f"ATOM  {serial:5d}  {e:<3s} UNK A   1    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00{0.0:10.3f} {ad:<2s}")
        serial += 1
    for i in polar_h:
        x, y, z = coords[i]
        lines.append(f"ATOM  {serial:5d}  H   UNK A   1    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00{0.0:10.3f} HD")
        serial += 1
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(heavy), len(polar_h)


def ligand_pdbqt(smiles, out_path, seed=0):
    """3D-embed a SMILES and write a flexible-ligand PDBQT via Meeko."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    except Exception:
        pass
    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    if not setups:
        return False
    text, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return False
    with open(out_path, "w") as fh:
        fh.write(text)
    return True


def dock_one(job):
    """One (pocket, candidate) docking. Returns the best Vina score (kcal/mol, lower better)."""
    from vina import Vina
    receptor, ligand, centre, size, exhaustiveness, seed = job
    try:
        # cpu=1 is essential. Vina defaults to cpu=0 meaning "use every core", so each of the
        # N pool workers would spawn a full-machine thread pool and they would fight each other;
        # the run gets slower as workers are added. One core per docking, parallelism from the
        # pool.
        v = Vina(sf_name="vina", seed=seed, verbosity=0, cpu=1)
        v.set_receptor(receptor)
        v.set_ligand_from_file(ligand)
        v.compute_vina_maps(center=list(centre), box_size=list(size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)
        return float(v.energies(n_poses=1)[0][0])
    except Exception as error:
        return f"ERROR:{error}"


def build_inputs(args):
    chrono = pd.read_csv(args.chrono_csv).set_index("system_id")
    subset = pd.read_csv(args.subset_csv)
    ids = [s for s in subset.system_id if s in chrono.index][: args.max_pockets]
    panel_ids = [s for s in subset.system_id if s in chrono.index][: args.panel_size]
    for s in ids:
        if s not in panel_ids:
            panel_ids.append(s)

    os.makedirs(f"{args.work}/receptors", exist_ok=True)
    os.makedirs(f"{args.work}/ligands", exist_ok=True)

    wanted = set(ids)
    boxes = {}
    for path in sorted(glob.glob(os.path.join(args.parquet, "*.parquet"))):
        table = pq.read_table(path, columns=[
            "system_id", "protein_coords", "protein_coords_shape",
            "protein_element_symbols", "ligand_coords", "ligand_coords_shape"])
        frame = table.to_pandas()
        for _, row in frame[frame.system_id.isin(wanted)].iterrows():
            sid = row.system_id
            rec = f"{args.work}/receptors/{sid}.pdbqt"
            pc = np.asarray(row.protein_coords, dtype=float).reshape(row.protein_coords_shape)
            lc = np.asarray(row.ligand_coords, dtype=float).reshape(row.ligand_coords_shape)
            if pc.shape[0] == 3:
                pc = pc.T
            if lc.shape[0] == 3:
                lc = lc.T
            if not os.path.exists(rec):
                receptor_pdbqt(pc, row.protein_element_symbols, rec)
            centre = lc.mean(axis=0)
            extent = lc.max(axis=0) - lc.min(axis=0) + 2 * args.margin
            boxes[sid] = (centre.tolist(),
                          np.clip(extent, args.min_box, args.max_box).tolist())
    json.dump(boxes, open(f"{args.work}/boxes.json", "w"))

    made, failed = 0, []
    for pid in panel_ids:
        out = f"{args.work}/ligands/{pid}.pdbqt"
        if os.path.exists(out):
            made += 1
            continue
        if ligand_pdbqt(chrono.loc[pid].smiles, out, seed=args.seed):
            made += 1
        else:
            failed.append(pid)
    print(f"receptors: {len(boxes)}   ligands: {made}/{len(panel_ids)}"
          + (f"   FAILED to embed: {len(failed)}" if failed else ""))
    json.dump({"system_ids": ids, "panel_ids": panel_ids, "failed_ligands": failed},
              open(f"{args.work}/manifest.json", "w"))
    return ids, panel_ids, boxes


def cmd_prepare(args):
    build_inputs(args)


def cmd_dock(args):
    meta = json.load(open(f"{args.work}/manifest.json"))
    boxes = json.load(open(f"{args.work}/boxes.json"))
    ids, panel_ids = meta["system_ids"], meta["panel_ids"]
    failed = set(meta["failed_ligands"])

    os.makedirs(f"{args.work}/scores", exist_ok=True)
    jobs, keys = [], []
    for sid in ids:
        if sid not in boxes:
            continue
        centre, size = boxes[sid]
        for pid in panel_ids:
            if pid in failed:
                continue
            out = f"{args.work}/scores/{sid}__{pid}.json"
            if os.path.exists(out):
                continue
            jobs.append((f"{args.work}/receptors/{sid}.pdbqt",
                         f"{args.work}/ligands/{pid}.pdbqt",
                         centre, size, args.exhaustiveness, args.seed))
            keys.append((sid, pid, out))

    print(f"docking {len(jobs)} pairs on {args.workers} workers "
          f"(exhaustiveness {args.exhaustiveness})", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(dock_one, job): k for job, k in zip(jobs, keys)}
        for future in as_completed(futures):
            sid, pid, out = futures[future]
            value = future.result()
            json.dump({"system_id": sid, "candidate": pid, "score": value},
                      open(out, "w"))
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"docked {done} pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "dock"])
    parser.add_argument("--work", default="../vina_bench/plinder")
    parser.add_argument("--parquet", default="../hiqbind/parquet_v2/test")
    parser.add_argument("--chrono_csv", default="data/test_set_chronological_split.csv")
    parser.add_argument("--subset_csv", default="data/test_set_plinder_split.csv")
    parser.add_argument("--max_pockets", type=int, default=107)
    parser.add_argument("--panel_size", type=int, default=107)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--min_box", type=float, default=18.0)
    parser.add_argument("--max_box", type=float, default=30.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"prepare": cmd_prepare, "dock": cmd_dock}[args.command](args)


if __name__ == "__main__":
    main()
