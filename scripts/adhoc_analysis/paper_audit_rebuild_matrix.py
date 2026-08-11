"""Reconstruct the pocket x ligand likelihood matrix from per-pocket CSVs.

The generating code (evaluate_combined_vox2smiles.py:352-394) wrote, for each pocket:
    row 0            -> the pocket's own ligand (is_hit=1)
    rows 1..N        -> [s for s in df.smiles.values if s not in batch['smiles']]
                        i.e. the master list in fixed order, own SMILES removed.
So ligand identity is recoverable by replaying that filter.
"""
import os, glob, sys
import numpy as np
import pandas as pd

ROOT = "/mnt/disk2/VoxelDiffOuter/plixer"
LIK_DIR = os.path.join(ROOT, "evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/plixer_likelihood_scores/likelihood_scores")
MASTER = os.path.join(ROOT, "data/test_set_chronological_split.csv")

master = pd.read_csv(MASTER)
all_smiles = master["smiles"].values
sid_list = master["system_id"].values
n = len(all_smiles)
sid_to_idx = {s: i for i, s in enumerate(sid_list)}
print(f"master list: {n} systems, {master.smiles.nunique()} unique SMILES")

files = sorted(glob.glob(os.path.join(LIK_DIR, "likelihood_output_*.csv")))
print(f"likelihood files: {len(files)}")

M = np.full((n, n), np.nan, dtype=np.float64)   # M[pocket_i, ligand_j]
ok, bad = 0, []

for f in files:
    sid = os.path.basename(f)[len("likelihood_output_"):-len(".csv")]
    if sid not in sid_to_idx:
        bad.append((sid, "system_id not in master")); continue
    i = sid_to_idx[sid]
    own = all_smiles[i]
    d = pd.read_csv(f)

    # Replay the exact filter used at generation time
    keep = [j for j, s in enumerate(all_smiles) if s != own]

    if len(d) != 1 + len(keep):
        bad.append((sid, f"row mismatch: file={len(d)} expected={1+len(keep)}")); continue
    if int(d.iloc[0]["is_hit"]) != 1 or d["is_hit"].sum() != 1:
        bad.append((sid, "hit not exactly at row 0")); continue

    M[i, i] = d.iloc[0]["likelihood"]
    M[i, keep] = d["likelihood"].values[1:]
    ok += 1

print(f"\nreconstructed OK: {ok}/{len(files)}   failed: {len(bad)}")
for sid, why in bad[:10]:
    print(f"   FAIL {sid}: {why}")

np.save("/tmp/claude-1000/-mnt-disk2-VoxelDiffOuter-plixer/b47ebbbe-b4db-429c-b7d9-fadd50311a3d/scratchpad/M.npy", M)
master.to_csv("/tmp/claude-1000/-mnt-disk2-VoxelDiffOuter-plixer/b47ebbbe-b4db-429c-b7d9-fadd50311a3d/scratchpad/master.csv", index=False)
print("saved M.npy")

filled = np.isfinite(M)
print(f"matrix fill: {filled.sum()}/{M.size} ({100*filled.sum()/M.size:.1f}%)")
print(f"rows with data: {filled.any(axis=1).sum()}  cols with data: {filled.any(axis=0).sum()}")
