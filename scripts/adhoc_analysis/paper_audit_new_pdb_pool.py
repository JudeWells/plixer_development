"""Audit the pool of post-training-cutoff PDB data available for a new Plixer test set.

Establishes (a) the true training cutoff from PDB release dates, (b) how much
protein-ligand structure has been released since, and (c) NTAB-style ligand
novelty tiers of the new ligand chemotypes against the HiQBind training ligands.

Reference for the tiering scheme:
  Mattsson & Walters, "Identifying and Addressing Systematic Data Leakage in
  Protein-Ligand Affinity Benchmarks", bioRxiv 10.64898/2026.06.29.735309.

Network-bound (RCSB search + data APIs). Run from the repo root:
    ./venvPlixer/bin/python scripts/adhoc_analysis/paper_audit_new_pdb_pool.py
"""

import argparse
import collections
import glob
import json
import urllib.request

import numpy as np
import pandas as pd

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/graphql"
PARQUET_ROOT = "../hiqbind/parquet"

# Drug-like non-polymer ligand; matches the MW window HiQBind curates over.
LIGAND_MW = {"from": 150, "to": 800}


def _post(url, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def search(nodes, return_type="entry", rows=0, start=0):
    payload = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": return_type,
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
        },
    }
    try:
        r = _post(SEARCH_URL, payload)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read()[:400]) from e
    return r.get("total_count", 0), [x["identifier"] for x in r.get("result_set", [])]


def text_node(attribute, operator, value, service="text"):
    return {
        "type": "terminal",
        "service": service,
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def date_nodes(lo, hi=None):
    n = [text_node("rcsb_accession_info.initial_release_date", "greater_or_equal", lo)]
    if hi:
        n.append(text_node("rcsb_accession_info.initial_release_date", "less", hi))
    return n


PROTEIN = text_node("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein")
DRUGLIKE = text_node("chem_comp.formula_weight", "range", LIGAND_MW)
RESOLUTION = text_node("rcsb_entry_info.resolution_combined", "less_or_equal", 2.5)


def load_splits():
    """HiQBind splits, keeping only the columns this audit needs."""
    cols = ["system_id", "smiles", "protein_cluster_id", "ligand_cluster_id"]
    out = {}
    for split in ("train", "val", "test"):
        files = sorted(glob.glob(f"{PARQUET_ROOT}/{split}/*.parquet"))
        out[split] = pd.concat(
            [pd.read_parquet(f, columns=cols) for f in files], ignore_index=True
        )
    return out


def release_dates(pdb_ids, batch=500):
    """initial_release_date for each PDB id, via the GraphQL data API."""
    dates = {}
    for i in range(0, len(pdb_ids), batch):
        ids = [p.upper() for p in pdb_ids[i : i + batch]]
        query = (
            "{entries(entry_ids:%s){rcsb_id rcsb_accession_info{initial_release_date}}}"
            % json.dumps(ids)
        )
        for e in _post(DATA_URL, {"query": query})["data"]["entries"] or []:
            if e:
                dates[e["rcsb_id"].lower()] = e["rcsb_accession_info"]["initial_release_date"][:10]
    return dates


def new_chem_comps(cutoff, page=1000, cap=20000):
    """Chemical components whose first PDB release postdates the training cutoff."""
    nodes = [
        text_node("rcsb_chem_comp_info.initial_release_date", "greater_or_equal", cutoff, "text_chem"),
        text_node("chem_comp.formula_weight", "range", LIGAND_MW, "text_chem"),
        text_node("chem_comp.type", "exact_match", "non-polymer", "text_chem"),
    ]
    ids = []
    for start in range(0, cap, page):
        try:
            _, got = search(nodes, return_type="mol_definition", rows=page, start=start)
        except RuntimeError:
            break
        if not got:
            break
        ids += got
    return sorted(set(ids))


def comp_smiles(comp_ids, batch=250):
    smiles = {}
    for i in range(0, len(comp_ids), batch):
        query = (
            "{chem_comps(comp_ids:%s){rcsb_id rcsb_chem_comp_descriptor{SMILES_stereo}}}"
            % json.dumps(comp_ids[i : i + batch])
        )
        for e in _post(DATA_URL, {"query": query})["data"]["chem_comps"] or []:
            if e and e.get("rcsb_chem_comp_descriptor"):
                smiles[e["rcsb_id"]] = e["rcsb_chem_comp_descriptor"]["SMILES_stereo"]
    return smiles


def novelty_tiers(query_smiles, reference_smiles):
    """Max Morgan-Tanimoto of each query against the reference set, binned into NTAB tiers."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smis):
        fps, kept = [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                fps.append(gen.GetFingerprint(m))
                kept.append(s)
        return fps, kept

    ref, _ = fingerprint(reference_smiles)
    qfp, qkept = fingerprint(query_smiles)
    best = np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref)) for f in qfp])
    return best, qkept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tiers", action="store_true", help="skip the RDKit novelty tiering")
    args = ap.parse_args()

    splits = load_splits()
    pdb_by_split = {
        k: sorted({s.split("_")[0].lower() for s in v.system_id}) for k, v in splits.items()
    }
    for k, v in splits.items():
        print(
            f"{k:6s} rows={len(v):6d} systems={v.system_id.nunique():6d} "
            f"smiles={v.smiles.nunique():6d} pdb={len(pdb_by_split[k]):6d} "
            f"protein_clusters={v.protein_cluster_id.nunique():5d}"
        )

    all_pdb = sorted(set().union(*pdb_by_split.values()))
    print(f"\nfetching release dates for {len(all_pdb)} PDB entries ...")
    dates = release_dates(all_pdb)

    bounds = {}
    for k, v in pdb_by_split.items():
        ds = sorted(dates[p] for p in v if p in dates)
        bounds[k] = (ds[0], ds[-1])
        years = collections.Counter(d[:4] for d in ds)
        print(f"{k:6s} n={len(ds):6d} {ds[0]} -> {ds[-1]}")
        if k == "test":
            print("       by year:", dict(sorted(years.items())))

    # Everything the model has ever seen ends here; the new pool starts the next day.
    seen_max = max(bounds["train"][1], bounds["val"][1], bounds["test"][1])
    cutoff = (pd.Timestamp(seen_max) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\ndataset compilation cutoff = {seen_max}; new pool starts {cutoff}")

    print("\npost-cutoff PDB pool")
    windows = [("all post-cutoff", cutoff, None)]
    for y in range(int(cutoff[:4]), 2027):
        windows.append((str(y), f"{y}-01-01", f"{y + 1}-01-01"))
    for label, lo, hi in windows:
        base = date_nodes(lo, hi)
        n_prot, _ = search(base + [PROTEIN])
        n_lig, _ = search(base + [PROTEIN, DRUGLIKE])
        n_res, _ = search(base + [PROTEIN, DRUGLIKE, RESOLUTION])
        print(
            f"  {label:16s} protein={n_prot:7d}  +druglike_ligand={n_lig:7d}  +res<=2.5A={n_res:7d}"
        )

    comp_ids = new_chem_comps(cutoff)
    train_ccd = {s.split("_")[1] for s in splits["train"].system_id} | {
        s.split("_")[1] for s in splits["val"].system_id
    }
    print(f"\nligand chemotypes: train+val CCDs={len(train_ccd)}  new post-cutoff CCDs={len(comp_ids)}")
    if args.skip_tiers:
        return

    smiles = comp_smiles(comp_ids)
    reference = sorted(set(splits["train"].smiles) | set(splits["val"].smiles))
    best, _ = novelty_tiers([smiles[c] for c in sorted(smiles)], reference)

    print(f"\nNTAB novelty tiers: max Tanimoto of new PDB ligands vs HiQBind train+val (n={len(best)})")
    for lo, hi in [(0.0, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        n = int(((best >= lo) & (best < hi)).sum())
        print(f"  [{lo:.2f},{hi:.2f})   {n:6d}  {100 * n / len(best):5.1f}%")
    n = int((best >= 1.0).sum())
    print(f"  ==1.00 (exact)  {n:6d}  {100 * n / len(best):5.1f}%")
    print(f"  median max-sim {np.median(best):.3f}")


if __name__ == "__main__":
    main()
