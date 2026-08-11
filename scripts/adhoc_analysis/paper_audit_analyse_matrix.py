import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

SP = "/tmp/claude-1000/-mnt-disk2-VoxelDiffOuter-plixer/b47ebbbe-b4db-429c-b7d9-fadd50311a3d/scratchpad"
M = np.load(f"{SP}/M.npy")
master = pd.read_csv(f"{SP}/master.csv")
n = M.shape[0]

def auc_along(mat, axis):
    """axis=1 -> per pocket (row) over ligands; axis=0 -> per ligand (col) over pockets."""
    out, ranks = [], []
    for k in range(n):
        v = mat[k, :] if axis == 1 else mat[:, k]
        m = np.isfinite(v)
        if not m[k]:
            continue
        lab = np.zeros(n, dtype=int); lab[k] = 1
        if m.sum() < 10:
            continue
        out.append(roc_auc_score(lab[m], v[m]))
        # rank of the true entry, 1 = best
        order = np.argsort(-v[m])
        true_pos = np.where(np.arange(n)[m][order] == k)[0][0] + 1
        ranks.append(true_pos)
    return np.array(out), np.array(ranks)

print("=" * 74)
print("1. RAW likelihood  (what the paper reports)")
row_auc, row_rank = auc_along(M, axis=1)
print(f"   ROW view  (fix pocket, rank 943 ligands)   mean AUC = {row_auc.mean():.4f}   median = {np.median(row_auc):.4f}")
print(f"     median rank of true ligand: {np.median(row_rank):.0f} / {n}   top-1%: {(row_rank<=9.43).mean()*100:.1f}%   top-10: {(row_rank<=10).mean()*100:.1f}%")

col_auc, col_rank = auc_along(M, axis=0)
print(f"   COL view  (fix ligand, rank 943 pockets)   mean AUC = {col_auc.mean():.4f}   median = {np.median(col_auc):.4f}")
print(f"     median rank of true pocket: {np.median(col_rank):.0f} / {n}   top-1%: {(col_rank<=9.43).mean()*100:.1f}%   top-10: {(col_rank<=10).mean()*100:.1f}%")

print()
print("=" * 74)
print("2. VARIANCE DECOMPOSITION   M[i,j] = mu + pocket_i + ligand_j + interaction_ij")
mask = np.isfinite(M)
mu = np.nanmean(M)
a = np.nanmean(M, axis=1) - mu            # pocket main effect
b = np.nanmean(M, axis=0) - mu            # ligand main effect
resid = M - (mu + a[:, None] + b[None, :])
va, vb, ve = np.nanvar(a), np.nanvar(b), np.nanvar(resid[mask])
tot = va + vb + ve
print(f"   pocket effect      : {va:.5f}  ({100*va/tot:5.1f}%)   <- 'this pocket gives high likelihood to everything'")
print(f"   ligand effect      : {vb:.5f}  ({100*vb/tot:5.1f}%)   <- 'this molecule is intrinsically likely'")
print(f"   interaction (resid): {ve:.5f}  ({100*ve/tot:5.1f}%)   <- THE ONLY PART THAT CAN ENCODE SPECIFICITY")

print()
print("=" * 74)
print("3. COLUMN-NORMALISED  (z-score each ligand across all pockets -> removes ligand prior)")
mu_c = np.nanmean(M, axis=0); sd_c = np.nanstd(M, axis=0)
Z = (M - mu_c[None, :]) / np.where(sd_c > 0, sd_c, np.nan)[None, :]
zrow_auc, zrow_rank = auc_along(Z, axis=1)
print(f"   ROW view on z-scored matrix                mean AUC = {zrow_auc.mean():.4f}   median = {np.median(zrow_auc):.4f}")
print(f"     median rank of true ligand: {np.median(zrow_rank):.0f} / {n}   top-1%: {(zrow_rank<=9.43).mean()*100:.1f}%   top-10: {(zrow_rank<=10).mean()*100:.1f}%")

print()
print("=" * 74)
print("4. BOOTSTRAP 95% CI over pockets")
rng = np.random.default_rng(0)
for name, arr in [("raw row AUC", row_auc), ("z-norm row AUC", zrow_auc), ("col AUC", col_auc)]:
    bs = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"   {name:16s} {arr.mean():.4f}   95% CI [{lo:.4f}, {hi:.4f}]   frac pockets >0.5: {(arr>0.5).mean()*100:.1f}%")

print()
print("=" * 74)
print("5. IS THE LIGAND PRIOR PREDICTING THE HIT?  (leakage check)")
# If a ligand has a high mean likelihood across ALL pockets, is it more likely to be
# ranked top by its own pocket? Rank ligands by column mean alone -- a pocket-blind baseline.
blind = np.tile(b, (n, 1))                       # pocket-independent score
blind[~mask] = np.nan
bl_auc, bl_rank = auc_along(blind, axis=1)
print(f"   POCKET-BLIND baseline (score = ligand mean only)  mean AUC = {bl_auc.mean():.4f}")
print(f"     -> a model that never looks at the pocket already achieves this.")

pd.DataFrame({
    "system_id": master.system_id.values[:len(row_auc)],
    "raw_row_auc": row_auc, "znorm_row_auc": zrow_auc,
}).to_csv(f"{SP}/per_pocket_auc.csv", index=False)
