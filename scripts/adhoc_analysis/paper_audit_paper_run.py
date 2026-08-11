import os, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/disk2/VoxelDiffOuter/plixer"
ER = f"{ROOT}/evaluation_results"

RUNS = [
    ("chrono ", "test_set_chronological_split.csv", f"{ER}/checkpoints_model_run_2025-07-02_batched/chrono/plixer_likelihood_scores/likelihood_scores", 0.67),
    ("PLINDER", "test_set_plinder_split.csv",       f"{ER}/checkpoints_model_run_2025-07-02/plinder/plixer_likelihood_scores/likelihood_scores", 0.58),
    ("seq-sim", "test_set_seq_sim_split.csv",       f"{ER}/checkpoints_model_run_2025-07-02/seqsim/plixer_likelihood_scores/likelihood_scores", 0.61),
    ("PLINDER(batched)", "test_set_plinder_split.csv", f"{ER}/checkpoints_model_run_2025-07-02_batched/plinder/plixer_likelihood_scores/likelihood_scores", 0.58),
]

def build(csv, d):
    master = pd.read_csv(f"{ROOT}/data/{csv}")
    s = master.smiles.values; sid = {x: i for i, x in enumerate(master.system_id.values)}; n = len(s)
    M = np.full((n, n), np.nan); ok = 0; skip = 0
    for f in glob.glob(f"{d}/likelihood_output_*.csv"):
        name = os.path.basename(f)[len("likelihood_output_"):-4]
        if name not in sid: skip += 1; continue
        i = sid[name]; dd = pd.read_csv(f)
        keep = [j for j, x in enumerate(s) if x != s[i]]
        if len(dd) != 1 + len(keep) or int(dd.iloc[0].is_hit) != 1: skip += 1; continue
        M[i, i] = dd.iloc[0].likelihood; M[i, keep] = dd.likelihood.values[1:]; ok += 1
    return M, ok, skip, n

def aucs(M, n):
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0)
    Z = (M - mu[None, :]) / np.where(sd > 0, sd, np.nan)[None, :]
    def go(mat):
        o = []
        for i in range(n):
            v = mat[i, :]; m = np.isfinite(v)
            if not m[i] or m.sum() < 10: continue
            l = np.zeros(n, int); l[i] = 1
            o.append(roc_auc_score(l[m], v[m]))
        return np.array(o)
    return go(M), go(Z)

print(f"{'split':18s} {'n':>5s}  {'paper':>6s}  {'raw mean':>9s}  {'raw med':>8s}   {'Z mean':>7s}  {'Z med':>7s}")
print("-" * 74)
for name, csv, d, paper in RUNS:
    if not os.path.isdir(d):
        print(f"{name:18s}  MISSING {d}"); continue
    M, ok, skip, n = build(csv, d)
    r, z = aucs(M, n)
    flag = "  <-- MATCH" if abs(r.mean() - paper) < 0.006 else ""
    print(f"{name:18s} {ok:5d}  {paper:6.2f}  {r.mean():9.4f}  {np.median(r):8.4f}   {z.mean():7.4f}  {np.median(z):7.4f}{flag}")
