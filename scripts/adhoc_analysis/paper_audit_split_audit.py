import glob, pandas as pd, pyarrow.parquet as pq, numpy as np

ROOT = "/mnt/disk2/VoxelDiffOuter"
COLS = ["system_id", "smiles", "split", "protein_cluster_id", "ligand_cluster_id"]

frames = {}
for split in ["train", "val", "test"]:
    fs = sorted(glob.glob(f"{ROOT}/hiqbind/parquet/{split}/*.parquet"))
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in fs], ignore_index=True)
    frames[split] = d
    print(f"{split:6s} {len(fs):4d} files  {len(d):6d} rows  {d.system_id.nunique():6d} unique system_id  "
          f"{d.protein_cluster_id.nunique():5d} protein clusters")

tr, va, te = frames["train"], frames["val"], frames["test"]
S = {k: set(v.system_id) for k, v in frames.items()}

print("\n=== 1. SYSTEM-ID OVERLAP BETWEEN PARQUET SPLITS ===")
print(f"  train n val  : {len(S['train'] & S['val'])}")
print(f"  train n test : {len(S['train'] & S['test'])}")
print(f"  val   n test : {len(S['val']   & S['test'])}")

print("\n=== 2. PROTEIN-CLUSTER OVERLAP (the thing that actually controls leakage) ===")
C = {k: set(v.protein_cluster_id.dropna()) for k, v in frames.items()}
print(f"  train n val  clusters: {len(C['train'] & C['val'])}   (val was meant to be cluster-disjoint)")
print(f"  train n test clusters: {len(C['train'] & C['test'])}   (chronological split -> overlap EXPECTED)")

print("\n=== 3. EVAL TEST SPLITS vs POC2MOL/COMBINED TRAINING DATA ===")
for name, csv in [("chrono ", "test_set_chronological_split.csv"),
                  ("PLINDER", "test_set_plinder_split.csv"),
                  ("seq-sim", "test_set_seq_sim_split.csv")]:
    ev = pd.read_csv(f"{ROOT}/plixer/data/{csv}")
    ids = set(ev.system_id)
    in_tr, in_va, in_te = len(ids & S["train"]), len(ids & S["val"]), len(ids & S["test"])
    print(f"  {name}  n={len(ids):4d}   in parquet train: {in_tr:4d}   in val: {in_va:4d}   in test: {in_te:4d}")

print("\n=== 4. SMILES-LEVEL OVERLAP (same ligand seen in training, different pocket) ===")
tr_smi = set(tr.smiles)
for name, csv in [("chrono ", "test_set_chronological_split.csv"),
                  ("PLINDER", "test_set_plinder_split.csv"),
                  ("seq-sim", "test_set_seq_sim_split.csv")]:
    ev = pd.read_csv(f"{ROOT}/plixer/data/{csv}")
    n = ev.smiles.isin(tr_smi).sum()
    print(f"  {name}  {n:4d}/{len(ev):4d} ({100*n/len(ev):5.1f}%) of test ligands appear verbatim in poc2mol/combined TRAIN")

print("\n=== 5. TEST LIGANDS IN THE ZINC PRETRAINING CORPUS (vox2smiles) ===")
zf = sorted(glob.glob(f"{ROOT}/zinc20_parquet/*.parquet"))
print(f"  scanning {len(zf)} zinc parquet files for exact SMILES matches ...")
targets = {}
for name, csv in [("chrono ", "test_set_chronological_split.csv"),
                  ("PLINDER", "test_set_plinder_split.csv"),
                  ("seq-sim", "test_set_seq_sim_split.csv")]:
    targets[name] = set(pd.read_csv(f"{ROOT}/plixer/data/{csv}").smiles)
allt = set().union(*targets.values())
found = set()
for i, f in enumerate(zf):
    col = [c for c in pq.ParquetFile(f).schema_arrow.names if "smi" in c.lower()]
    if not col: continue
    s = pq.read_table(f, columns=col[:1]).to_pandas().iloc[:, 0]
    found |= (allt & set(s))
    if i % 1000 == 0 and i:
        print(f"    ...{i}/{len(zf)} files, {len(found)} matches so far")
for name, t in targets.items():
    print(f"  {name}  {len(t & found):4d}/{len(t):4d} test ligands found verbatim in ZINC20")
