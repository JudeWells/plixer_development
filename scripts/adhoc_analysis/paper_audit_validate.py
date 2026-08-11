import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SP = "/tmp/claude-1000/-mnt-disk2-VoxelDiffOuter-plixer/b47ebbbe-b4db-429c-b7d9-fadd50311a3d/scratchpad"
ROOT = "/mnt/disk2/VoxelDiffOuter/plixer"
M = np.load(f"{SP}/M.npy"); master = pd.read_csv(f"{SP}/master.csv"); n = M.shape[0]

# ---- A. cross-check my reconstructed row AUCs against the values the eval stored ----
stored = pd.read_csv(f"{ROOT}/evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/plixer_likelihood_scores/all_decoy_likelihoods.csv")
sid_to_i = {s: i for i, s in enumerate(master.system_id.values)}
mine, theirs = [], []
for _, r in stored.iterrows():
    i = sid_to_i.get(r.system_id)
    if i is None: continue
    v = M[i, :]; m = np.isfinite(v)
    if not m[i]: continue
    lab = np.zeros(n, int); lab[i] = 1
    mine.append(roc_auc_score(lab[m], v[m])); theirs.append(r.auc_roc)
mine, theirs = np.array(mine), np.array(theirs)
print("A. RECONSTRUCTION CROSS-CHECK vs values stored by the original eval")
print(f"   n={len(mine)}  max|diff| = {np.abs(mine-theirs).max():.2e}   identical: {np.allclose(mine, theirs)}")
print(f"   stored mean AUC = {theirs.mean():.4f}   stored median = {np.median(theirs):.4f}")

# ---- B. does the pocket signal survive controlling for molecular size? ----
hac = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else np.nan
                for s in master.smiles.values], dtype=float)
print(f"\nB. SIZE CONTROL   (heavy-atom count parsed for {np.isfinite(hac).sum()}/{n})")

mu_c = np.nanmean(M, axis=0); sd_c = np.nanstd(M, axis=0)
Z = (M - mu_c[None, :]) / np.where(sd_c > 0, sd_c, np.nan)[None, :]

def auc_size_matched(mat, tol):
    out = []
    for i in range(n):
        if not np.isfinite(hac[i]): continue
        v = mat[i, :]
        sel = np.isfinite(v) & np.isfinite(hac) & (np.abs(hac - hac[i]) <= tol * hac[i])
        if not sel[i] or sel.sum() < 30: continue
        lab = np.zeros(n, int); lab[i] = 1
        out.append(roc_auc_score(lab[sel], v[sel]))
    return np.array(out)

for tol, name in [(1e9, "all decoys"), (0.20, "+/-20% heavy atoms"), (0.10, "+/-10% heavy atoms")]:
    raw = auc_size_matched(M, tol); z = auc_size_matched(Z, tol)
    print(f"   {name:22s} n={len(z):4d}   raw AUC = {raw.mean():.4f}   z-norm AUC = {z.mean():.4f}")

# how much of the raw likelihood is just size?
colmean = np.nanmean(M, axis=0); ok = np.isfinite(colmean) & np.isfinite(hac)
print(f"\n   corr(ligand mean likelihood, heavy atoms) = {np.corrcoef(colmean[ok], hac[ok])[0,1]:.3f}")
print("   -> if strongly negative, the raw metric is largely a size ranking")

# ---- C. permutation null: shuffle which pocket is 'true' ----
rng = np.random.default_rng(0); perm_auc = []
for _ in range(200):
    p = rng.permutation(n)
    i = rng.integers(n); j = p[i]
    v = Z[i, :]; m = np.isfinite(v)
    if not m[j]: continue
    lab = np.zeros(n, int); lab[j] = 1
    perm_auc.append(roc_auc_score(lab[m], v[m]))
print(f"\nC. PERMUTATION NULL (z-norm, random pocket-ligand pairing): mean AUC = {np.mean(perm_auc):.4f}")
