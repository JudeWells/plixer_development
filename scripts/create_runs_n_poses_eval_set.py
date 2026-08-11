"""Build a drug-like, leakage-controlled evaluation set from Runs N' Poses.

Runs N' Poses (Skrinjar et al., Nat Struct Mol Biol 2026; https://github.com/plinder-org/runs-n-poses)
ships ~2,600 post-2021 PDB complexes annotated with pocket- and ligand-level similarity to
co-folding training data. Its `ligand_is_proper` flag removes ions and crystallisation artifacts
but **not** cofactors, so ~24% of "proper" ligands are HEM/FAD/NAD/ATP and friends. This script
applies an explicit drug-likeness filter on top and emits parquet in the layout
`src.data.poc2mol.datasets.ParquetDataset` expects.

Every system here postdates Plixer's structural training cutoff of 2019-12-25, so the whole set is
zero-shot. Each row additionally carries its NTAB novelty tier — the max Morgan-Tanimoto of the
ligand to the HiQBind train+val ligands — so metrics can be reported stratified by ligand novelty
rather than against a single similarity threshold.

⚠️ **Hydrogens.** HiQBind proteins are pdbfixer-protonated (49.8% of atoms are H); Runs N' Poses
receptors are deposited crystallographic coordinates and are largely unprotonated (5.7% H), and its
ligand SDFs are heavy-atom only. Under the corrected 4-channel protein / 9-channel ligand encoding
this is harmless — hydrogen maps to no channel and is dropped at voxelisation — and normalising over
heavy atoms only, the two sets agree closely (protein S/(C+O+N+S): 0.0060 here vs 0.0053 for
HiQBind test). **But it invalidates the `h5` and `hall6` protein-representation variants from
sweep 2 (§3e), which give hydrogen its own protein channel.** Do not evaluate those arms on this
set without protonating the receptors first.

Usage:
    ./venvPlixer/bin/python scripts/create_runs_n_poses_eval_set.py \
        --work-dir ../runs_n_poses --output-dir ../runs_n_poses/parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import tarfile
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ZENODO = "https://zenodo.org/api/records/18366081/files"
ANNOTATIONS_URL = f"{ZENODO}/annotations.csv/content"
GROUND_TRUTH_URL = f"{ZENODO}/ground_truth.tar.gz/content"

# Plixer's structural training cutoff, established from PDB release dates of the HiQBind splits
# (see scripts/adhoc_analysis/paper_audit_new_pdb_pool.py). Recorded for provenance only —
# every Runs N' Poses system postdates it.
TRAINING_CUTOFF = "2019-12-25"

# NTAB novelty tiers (Mattsson & Walters, bioRxiv 10.64898/2026.06.29.735309).
NOVELTY_TIERS = [(0.0, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.01)]

# Cofactors, nucleotides, energy carriers and common metabolites. `ligand_is_proper` keeps these,
# but they are not drug-like and a pocket-conditioned generator should not be scored on them.
COFACTOR_CCDS = {
    # haem and flavins
    "HEM", "HEC", "HEA", "HEB", "SRM", "FAD", "FMN", "FDA", "MQ7", "MQ8", "MQ9", "UQ1", "UQ2", "PQN",
    # nicotinamide / adenine nucleotides
    "NAD", "NAI", "NAP", "NDP", "NAX", "ADP", "ATP", "AMP", "ANP", "ACP", "AGS", "APR", "A3P", "5GP",
    "GDP", "GTP", "GNP", "GSP", "GMP", "CDP", "CTP", "CMP", "UDP", "UTP", "UMP", "UPG", "UD1",
    "TDP", "TTP", "TMP", "IDP", "ITP", "5AD", "ADN", "GUN", "URI", "THM", "CTN",
    # coenzymes and cofactors
    "COA", "ACO", "CAA", "MCA", "SAM", "SAH", "MTA", "TPP", "TDT", "H4B", "HBI", "BH4", "PLP", "PMP",
    "P5P", "B12", "COB", "B1Z", "F43", "THG", "THF", "FOL", "BTN", "LPA", "GSH", "GDS", "GTT",
    # iron-sulfur clusters, chlorophylls
    "SF4", "FES", "FE2", "FCO", "NFU", "CLA", "CHL", "BCL", "BPH", "PHO",
    # central metabolites and sugars
    "AKG", "CIT", "FUM", "SIN", "MAL", "OAA", "PEP", "G6P", "F6P", "FBP", "G3P", "R5P", "RIB",
    "GLC", "MAN", "NAG", "BMA", "GAL", "FUC", "SIA", "XYS", "NDG", "SUC", "TRE", "GLA", "A2G",
    "SGN", "IDS", "BGC", "PYR", "LAC", "ACE", "FMT", "OXL", "GAR",
    # isoprenoids, lipids, sterols, detergents, vitamins
    "AR6", "IPE", "DMA", "FPP", "GPP", "GGP", "IHP", "IPD", "MYR", "PLM", "STE", "OLA", "DAO",
    "HTG", "CHD", "CLR", "Y01", "PEE", "PGT", "PC1", "LMT", "BOG", "DDQ", "RET", "VIT", "TOC", "MEN",
    # free amino acids
    "TRP", "TYR", "PHE", "HIS", "ARG", "LYS", "GLU", "ASP", "SER", "THR", "CYS", "MET", "LEU",
    "ILE", "VAL", "PRO", "GLY", "ALA", "ASN", "GLN",
}


# --------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------

def fetch(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log.info("using cached %s", dest)
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    log.info("downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def ensure_ground_truth(work_dir: str) -> str:
    """Download and unpack ground_truth.tar.gz, returning the extracted directory."""
    gt_dir = os.path.join(work_dir, "ground_truth")
    if os.path.isdir(gt_dir) and glob.glob(os.path.join(gt_dir, "*", "receptor.cif")):
        log.info("using cached %s", gt_dir)
        return gt_dir
    tar_path = fetch(GROUND_TRUTH_URL, os.path.join(work_dir, "ground_truth.tar.gz"))
    log.info("extracting %s", tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(work_dir)
    return gt_dir


# --------------------------------------------------------------------------------------
# structure parsing
# --------------------------------------------------------------------------------------

def parse_cif_atoms(cif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an mmCIF `_atom_site` loop into (coords (N,3) float32, element symbols (N,) str).

    Runs N' Poses receptor files are whitespace-delimited with a single `_atom_site` loop, so the
    column order is read from the header rather than assumed. Alternate locations other than the
    first and models other than 1 are dropped.
    """
    fields: list[str] = []
    coords: list[tuple[float, float, float]] = []
    elements: list[str] = []
    with open(cif_path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                fields.append(line.strip().split(".", 1)[1])
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if not fields:
                raise ValueError(f"no _atom_site header found in {cif_path}")
            parts = line.split()
            if len(parts) < len(fields):
                continue
            rec = dict(zip(fields, parts))
            if rec.get("label_alt_id", ".") not in (".", "?", "A"):
                continue
            if rec.get("pdbx_PDB_model_num", "1") not in ("1", ".", "?"):
                continue
            try:
                coords.append(
                    (float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"]))
                )
            except (KeyError, ValueError):
                continue
            elements.append(rec.get("type_symbol", "C"))
    return np.asarray(coords, dtype=np.float32), np.asarray(elements, dtype=object)


def load_ligand(sdf_path: str):
    """Return (coords (N,3), elements (N,), canonical SMILES) for a Runs N' Poses ligand SDF."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:  # fall back to an unsanitised read for awkward valences
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        return None, None, None
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=np.float32)
    elements = np.asarray([a.GetSymbol() for a in mol.GetAtoms()], dtype=object)
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        smiles = Chem.MolToSmiles(mol)
    if not smiles:
        return None, None, None
    return coords, elements, smiles


# --------------------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------------------

def apply_filters(ann: pd.DataFrame, args) -> pd.DataFrame:
    """Drug-likeness filter on top of Runs N' Poses' own `ligand_is_proper` flag."""
    n0 = len(ann)
    df = ann[ann.ligand_is_proper == True].copy()  # noqa: E712 — pandas mask, not identity
    log.info("ligand_is_proper: %d -> %d rows (dropped %d ions/artifacts)", n0, len(df), n0 - len(df))

    # Each predicate is evaluated against the frame as it stands, so the reported drop counts are
    # the marginal effect of that filter rather than its overlap with the ones already applied.
    steps = [
        ("cofactor/metabolite CCD",
         lambda d: ~d.ligand_ccd_code.isin(COFACTOR_CCDS)),
        (f"MW outside [{args.min_mw}, {args.max_mw}]",
         lambda d: d.ligand_molecular_weight.between(args.min_mw, args.max_mw)),
        (f"fewer than {args.min_rings} ring(s)",
         lambda d: d.ligand_num_rings >= args.min_rings),
        (f"fewer than {args.min_heavy_atoms} heavy atoms",
         lambda d: d.ligand_num_heavy_atoms >= args.min_heavy_atoms),
    ]
    for label, predicate in steps:
        keep = predicate(df)
        dropped = int((~keep).sum())
        df = df[keep]
        log.info("  drop %-32s -%5d  -> %5d rows", label, dropped, len(df))
    log.info("drug-like ligands: %d rows / %d PDB entries / %d unique CCDs",
             len(df), df.entry_pdb_id.nunique(), df.ligand_ccd_code.nunique())
    return df


# --------------------------------------------------------------------------------------
# novelty tiering
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# structural filtering
# --------------------------------------------------------------------------------------

# A hand-curated CCD blocklist only catches enumerated codes, and nucleotide/glycan chemistry has a
# long tail of one-off codes. These patterns catch the same chemistry generically.
SMARTS = {
    # aromatic N bonded to the anomeric carbon of a furanose — the N-glycosidic bond that defines a
    # nucleoside. Catches adenosine/guanosine/cytidine/uridine and all their phosphates, and the
    # nucleotide arms of NAD/FAD/CoA regardless of CCD code.
    # The non-anomeric ring carbons must be [CX4;R], not [CH1;R]: deoxyribose has a CH2 at C2', so
    # an all-CH1 pattern silently misses every dNTP (8OG, DGT, DTP, DCP, DAT, ...).
    "nucleoside": "[n;R][CH1;R]1[OX2;R][CX4;R][CX4;R][CX4;R]1",
    "furanose": "[CX4;R]1[OX2;R][CX4;R][CX4;R][CX4;R]1",
    "pyranose": "[CX4;R]1[OX2;R][CX4;R][CX4;R][CX4;R][CX4;R]1",
    "hydroxyl": "[OX2H]",
}
_COMPILED = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}


def _sugar_rings(mol) -> int:
    return len(mol.GetSubstructMatches(_COMPILED["furanose"])) + len(
        mol.GetSubstructMatches(_COMPILED["pyranose"])
    )


def structural_verdict(smiles: str, args) -> str | None:
    """Return the name of the first structural filter this ligand fails, or None if it passes.

    Deliberately surgical. A blanket "contains phosphorus" or "contains a sugar ring" rule would
    discard legitimate inhibitors — phosphinic-acid pseudopeptides, aryl glycosides, nucleoside
    analogue antivirals are all real drugs. These rules target free sugars, oligosaccharides and
    nucleoside/nucleotide cofactors specifically.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return "unparseable"
    if mol.HasSubstructMatch(_COMPILED["nucleoside"]):
        return "nucleoside/nucleotide"
    n_sugar = _sugar_rings(mol)
    if n_sugar >= 2:
        return "oligosaccharide"
    n_oh = len(mol.GetSubstructMatches(_COMPILED["hydroxyl"]))
    has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    if n_sugar >= 1 and n_oh >= 2 and not has_aromatic:
        return "free sugar"
    if Descriptors.MolWt(mol) < args.min_structural_mw:
        return f"MW < {args.min_structural_mw:g}"
    if args.min_qed > 0 and QED.qed(mol) < args.min_qed:
        return f"QED < {args.min_qed:g}"
    return None


def apply_structural_filters(out: pd.DataFrame, args) -> pd.DataFrame:
    """Second filtering stage, on the molecule as actually stored (SMILES from the ligand SDF)."""
    n0 = len(out)
    # Hyphenated CCD codes are oligopeptide / oligosaccharide ligands. Plixer trained on HiQBind's
    # `hiq_sm` subset only, so these are a train/test category mismatch regardless of drug-likeness.
    is_peptide = out.ligand_ccd_code.astype(str).str.contains("-")
    log.info("  drop %-32s -%5d  -> %5d rows", "peptide/oligomer CCD code",
             int(is_peptide.sum()), n0 - int(is_peptide.sum()))
    out = out[~is_peptide]

    verdicts = out.smiles.map(lambda s: structural_verdict(s, args))
    for reason, n in verdicts.value_counts().items():
        log.info("  drop %-32s -%5d", reason, int(n))
    out = out[verdicts.isna()]
    log.info("structural filters: %d -> %d rows", n0, len(out))
    return out.reset_index(drop=True)


def hiqbind_reference_smiles(parquet_root: str) -> list[str]:
    smiles: set[str] = set()
    for split in ("train", "val"):
        for f in sorted(glob.glob(os.path.join(parquet_root, split, "*.parquet"))):
            smiles.update(pd.read_parquet(f, columns=["smiles"]).smiles.values)
    return sorted(smiles)


def max_similarity_to_reference(query_smiles: list[str], reference: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fps(smis):
        out = []
        for s in smis:
            m = Chem.MolFromSmiles(s) if s else None
            out.append(gen.GetFingerprint(m) if m is not None else None)
        return out

    ref = [f for f in fps(reference) if f is not None]
    return np.array(
        [max(DataStructs.BulkTanimotoSimilarity(f, ref)) if f is not None else np.nan
         for f in fps(query_smiles)],
        dtype=np.float32,
    )


def tier_label(sim: float) -> str:
    if not np.isfinite(sim):
        return "unknown"
    for lo, hi in NOVELTY_TIERS:
        if lo <= sim < hi:
            return "exact" if lo >= 1.0 else f"[{lo:.2f},{hi:.2f})"
    return "exact"


def cluster_ligands(smiles: list[str], cutoff: float = 0.3) -> dict[str, int]:
    """Butina clustering on Morgan fingerprints, matching create_hiqbind_dataset.py."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    uniq = sorted(set(s for s in smiles if s))
    fps, kept = [], []
    for s in uniq:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(gen.GetFingerprint(m))
            kept.append(s)
    if not fps:
        return {}
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    return {kept[idx]: cid for cid, members in enumerate(clusters) for idx in members}


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def build_rows(df: pd.DataFrame, gt_dir: str, crop_radius: float) -> list[dict]:
    rows, skipped = [], defaultdict(int)
    for rec in df.itertuples(index=False):
        sys_dir = os.path.join(gt_dir, rec.system_id)
        receptor = os.path.join(sys_dir, "receptor.cif")
        ligand_sdf = os.path.join(sys_dir, "ligand_files", f"{rec.ligand_instance_chain}.sdf")
        if not (os.path.exists(receptor) and os.path.exists(ligand_sdf)):
            skipped["missing files"] += 1
            continue
        try:
            lig_coords, lig_elements, smiles = load_ligand(ligand_sdf)
            if lig_coords is None:
                skipped["unparseable ligand"] += 1
                continue
            prot_coords, prot_elements = parse_cif_atoms(receptor)
            if len(prot_coords) == 0:
                skipped["empty receptor"] += 1
                continue

            # Crop the receptor around the ligand. The loader prunes to max_atom_dist=32 A anyway,
            # so a wider radius here is lossless and leaves headroom for translation augmentation.
            centre = lig_coords.mean(axis=0)
            keep = np.linalg.norm(prot_coords - centre, axis=1) <= crop_radius
            prot_coords, prot_elements = prot_coords[keep], prot_elements[keep]
            if len(prot_coords) == 0:
                skipped["no protein within crop radius"] += 1
                continue

            # ParquetDataset expects (3, N) coordinate arrays stored flattened as float16.
            rows.append({
                "system_id": f"{rec.system_id}__{rec.ligand_instance_chain}",
                "smiles": smiles,
                "protein_coords": prot_coords.T.astype(np.float16).flatten(),
                "protein_coords_shape": np.array(prot_coords.T.shape, dtype=np.int64),
                "protein_element_symbols": np.char.title(prot_elements.astype(str)).astype("U"),
                "ligand_coords": lig_coords.T.astype(np.float16).flatten(),
                "ligand_coords_shape": np.array(lig_coords.T.shape, dtype=np.int64),
                "ligand_element_symbols": np.char.title(lig_elements.astype(str)).astype("U"),
                "split": "test",
                "rnp_system_id": rec.system_id,
                "entry_pdb_id": rec.entry_pdb_id,
                "ligand_ccd_code": rec.ligand_ccd_code,
                "rnp_pocket_cluster": rec.cluster,
                "rnp_morgan_tanimoto": getattr(rec, "morgan_tanimoto", np.nan),
            })
        except Exception as exc:  # keep going; a handful of malformed entries is expected
            skipped[f"error: {type(exc).__name__}"] += 1
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        log.warning("skipped %5d systems (%s)", n, reason)
    return rows


def write_indices(split_dir: str) -> None:
    out = os.path.join(split_dir, "indices")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))
    all_samples, file_indices = [], []
    cluster_samples = defaultdict(list)
    for file_idx, path in enumerate(files):
        df = pd.read_parquet(path)
        for row_idx, row in df.iterrows():
            all_samples.append({"system_id": row["system_id"], "cluster": row["cluster"]})
            file_indices.append(file_idx)
            cluster_samples[row["cluster"]].append(
                {"file_idx": file_idx, "system_id": row["system_id"], "row_idx": int(row_idx)}
            )
    json.dump({"samples": all_samples, "file_indices": file_indices},
              open(os.path.join(out, "global_index.json"), "w"), indent=2)
    json.dump(cluster_samples, open(os.path.join(out, "cluster_index.json"), "w"), indent=2)
    json.dump({i: os.path.basename(p) for i, p in enumerate(files)},
              open(os.path.join(out, "file_mapping.json"), "w"), indent=2)
    json.dump({"total_samples": len(all_samples), "total_clusters": len(cluster_samples),
               "total_files": len(files)},
              open(os.path.join(out, "index_summary.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default="../runs_n_poses", help="download/extract cache")
    ap.add_argument("--output-dir", default="../runs_n_poses/parquet")
    ap.add_argument("--hiqbind-parquet", default="../hiqbind/parquet",
                    help="used as the reference set for ligand novelty tiering")
    ap.add_argument("--min-mw", type=float, default=150.0)
    ap.add_argument("--max-mw", type=float, default=800.0)
    ap.add_argument("--min-rings", type=int, default=1)
    ap.add_argument("--min-heavy-atoms", type=int, default=10)
    ap.add_argument("--crop-radius", type=float, default=40.0,
                    help="receptor atoms further than this from the ligand centroid are dropped")
    ap.add_argument("--entries-per-file", type=int, default=500)
    ap.add_argument("--keep-cofactors", action="store_true",
                    help="skip the cofactor filter (reproduces the raw ligand_is_proper set)")
    ap.add_argument("--min-structural-mw", type=float, default=200.0,
                    help="fragment floor applied to the parsed ligand")
    ap.add_argument("--min-qed", type=float, default=0.15,
                    help="light QED backstop; 0 disables. Kept low deliberately — QED penalises "
                         "large natural products, and macrolides etc. are legitimate drugs")
    ap.add_argument("--skip-structural", action="store_true",
                    help="skip the structural stage (reproduces the CCD-blocklist-only set)")
    ap.add_argument("--skip-novelty", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    if args.keep_cofactors:
        COFACTOR_CCDS.clear()

    ann = pd.read_csv(fetch(ANNOTATIONS_URL, os.path.join(args.work_dir, "annotations.csv")))
    log.info("annotations: %d rows / %d systems / %d PDB entries",
             len(ann), ann.system_id.nunique(), ann.entry_pdb_id.nunique())
    df = apply_filters(ann, args)

    gt_dir = ensure_ground_truth(args.work_dir)
    rows = build_rows(df, gt_dir, args.crop_radius)
    if not rows:
        raise SystemExit("no systems survived — check the ground-truth extraction path")
    out = pd.DataFrame(rows)
    log.info("built %d systems from %d PDB entries", len(out), out.entry_pdb_id.nunique())
    if not args.skip_structural:
        out = apply_structural_filters(out, args)
    log.info("final: %d systems / %d PDB entries / %d unique CCDs",
             len(out), out.entry_pdb_id.nunique(), out.ligand_ccd_code.nunique())

    # Ligand novelty against the HiQBind training ligands — the axis NTAB argues for.
    if args.skip_novelty:
        out["max_tanimoto_to_train"] = np.nan
    else:
        reference = hiqbind_reference_smiles(args.hiqbind_parquet)
        log.info("tiering against %d HiQBind train+val ligands", len(reference))
        out["max_tanimoto_to_train"] = max_similarity_to_reference(out.smiles.tolist(), reference)
    out["novelty_tier"] = [tier_label(s) for s in out.max_tanimoto_to_train]

    # Cluster ids drive ParquetDataset's sampling; keep every row and let redundancy reduction be a
    # downstream choice (one representative per cluster) rather than baking it in here.
    ligand_clusters = cluster_ligands(out.smiles.tolist())
    pocket_codes = {c: i for i, c in enumerate(sorted(out.rnp_pocket_cluster.astype(str).unique()))}
    out["protein_cluster_id"] = out.rnp_pocket_cluster.astype(str).map(pocket_codes).astype(int)
    out["ligand_cluster_id"] = out.smiles.map(lambda s: ligand_clusters.get(s, -1)).astype(int)
    out["cluster"] = "P" + out.protein_cluster_id.astype(str) + "L" + out.ligand_cluster_id.astype(str)

    split_dir = os.path.join(args.output_dir, "test")
    os.makedirs(split_dir, exist_ok=True)
    for old in glob.glob(os.path.join(split_dir, "*.parquet")):
        os.remove(old)
    for i in range(0, len(out), args.entries_per_file):
        chunk = out.iloc[i : i + args.entries_per_file].reset_index(drop=True)
        chunk.to_parquet(os.path.join(split_dir, f"rnp_{i // args.entries_per_file:04d}.parquet"),
                         index=False)
    write_indices(split_dir)

    log.info("wrote %s", split_dir)
    log.info("systems=%d  pocket_clusters=%d  ligand_clusters=%d  protein-ligand clusters=%d",
             len(out), out.protein_cluster_id.nunique(), out.ligand_cluster_id.nunique(),
             out.cluster.nunique())
    log.info("NTAB novelty tiers vs HiQBind train+val (cutoff %s):", TRAINING_CUTOFF)
    for tier, n in out.novelty_tier.value_counts().sort_index().items():
        log.info("  %-14s %5d  %5.1f%%", tier, n, 100 * n / len(out))


if __name__ == "__main__":
    main()
