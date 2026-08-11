import glob, os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/disk2/VoxelDiffOuter"; P = f"{ROOT}/plixer"
SP = "/tmp/claude-1000/-mnt-disk2-VoxelDiffOuter-plixer/b47ebbbe-b4db-429c-b7d9-fadd50311a3d/scratchpad"
CACHE = f"{SP}/zinc_hits.json"

splits = {n: pd.read_csv(f"{P}/data/{c}") for n, c in
          [("chrono", "test_set_chronological_split.csv"),
           ("PLINDER", "test_set_plinder_split.csv"),
           ("seq-sim", "test_set_seq_sim_split.csv")]}
allt = set().union(*[set(d.smiles) for d in splits.values()])

if os.path.exists(CACHE):
    zinc_hits = set(json.load(open(CACHE)))
else:
    zinc_hits = set()
    for f in sorted(glob.glob(f"{ROOT}/zinc20_parquet/*.parquet")):
        col = [c for c in pq.ParquetFile(f).schema_arrow.names if "smi" in c.lower()]
        if col:
            zinc_hits |= (allt & set(pq.read_table(f, columns=col[:1]).to_pandas().iloc[:, 0]))
    json.dump(sorted(zinc_hits), open(CACHE, "w"))
print(f"test ligands found in ZINC20: {len(zinc_hits)}")

tr_smi = set()
for f in sorted(glob.glob(f"{ROOT}/hiqbind/parquet/train/*.parquet")):
    tr_smi |= set(pq.read_table(f, columns=["smiles"]).to_pandas().smiles)
print(f"unique train SMILES: {len(tr_smi)}")

# ---------- A. does memorisation inflate the SIMILARITY ENRICHMENT (Table 1)? ----------
print("\n" + "=" * 78)
print("A. TANIMOTO SIMILARITY split by whether the true ligand was seen in training")
B = f"{P}/evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol"
for name, f in [("chrono ", f"{B}/chrono/chrono_combined_model_results_20250516_002423.csv"),
                ("PLINDER", f"{B}/plinder/plinder_combined_model_results_20250516_000646.csv")]:
    d = pd.read_csv(f, low_memory=False)
    d["seen_pocket_train"] = d.smiles.isin(tr_smi)
    d["seen_zinc"] = d.smiles.isin(zinc_hits)
    d["seen_any"] = d.seen_pocket_train | d.seen_zinc
    bh = (d.decoy_tanimoto_similarity >= 0.3).mean()
    print(f"\n  {name}  (n={len(d)}, overall EF={((d.tanimoto_similarity>=0.3).mean()/bh):.2f})")
    for lab, sub in [("ligand SEEN in train or ZINC", d[d.seen_any]),
                     ("ligand NEVER seen         ", d[~d.seen_any])]:
        if len(sub) == 0: continue
        ef = (sub.tanimoto_similarity >= 0.3).mean() / bh
        print(f"    {lab}  n={len(sub):4d}  mean Tanimoto={sub.tanimoto_similarity.mean():.3f}  "
              f"hit-rate={100*(sub.tanimoto_similarity>=0.3).mean():5.1f}%  EF={ef:.2f}")

# ---------- B. does it inflate the LIKELIHOOD AUC (Table 2)? ----------
print("\n" + "=" * 78)
print("B. LIKELIHOOD AUC split by the same criterion  (paper's July run)")
ER = f"{P}/evaluation_results"
RUNS = [("chrono ", "chrono", f"{ER}/checkpoints_model_run_2025-07-02_batched/chrono/plixer_likelihood_scores/likelihood_scores"),
        ("PLINDER", "PLINDER", f"{ER}/checkpoints_model_run_2025-07-02/plinder/plixer_likelihood_scores/likelihood_scores"),
        ("seq-sim", "seq-sim", f"{ER}/checkpoints_model_run_2025-07-02/seqsim/plixer_likelihood_scores/likelihood_scores")]
for label, key, d in RUNS:
    master = splits[key]; s = master.smiles.values
    sid = {x: i for i, x in enumerate(master.system_id.values)}; n = len(s)
    M = np.full((n, n), np.nan)
    for f in glob.glob(f"{d}/likelihood_output_*.csv"):
        nm = os.path.basename(f)[len("likelihood_output_"):-4]
        if nm not in sid: continue
        i = sid[nm]; dd = pd.read_csv(f); keep = [j for j, x in enumerate(s) if x != s[i]]
        if len(dd) != 1 + len(keep) or int(dd.iloc[0].is_hit) != 1: continue
        M[i, i] = dd.iloc[0].likelihood; M[i, keep] = dd.likelihood.values[1:]
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0)
    Z = (M - mu[None, :]) / np.where(sd > 0, sd, np.nan)[None, :]
    seen = np.array([(x in tr_smi) or (x in zinc_hits) for x in s])
    rows = {}
    for mat, tag in [(M, "raw"), (Z, "znorm")]:
        r = {True: [], False: []}
        for i in range(n):
            v = mat[i, :]; m = np.isfinite(v)
            if not m[i] or m.sum() < 10: continue
            l = np.zeros(n, int); l[i] = 1
            r[bool(seen[i])].append(roc_auc_score(l[m], v[m]))
        rows[tag] = r
    print(f"\n  {label}   seen={seen.sum()}/{n}")
    for tag in ("raw", "znorm"):
        a, b = np.array(rows[tag][True]), np.array(rows[tag][False])
        print(f"    {tag:6s}  SEEN mean AUC={a.mean():.4f} (n={len(a)})   UNSEEN mean AUC={b.mean():.4f} (n={len(b)})   diff={a.mean()-b.mean():+.4f}")
