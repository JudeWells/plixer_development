"""Measure whether HiQBind protein hydrogens are oriented toward the ligand (§3f).

Mechanism under test: `fix_protein.py:581` never mass-zeroes hydrogens, so during
`refineAddedAtomPositions` every H is free while all heavy atoms are frozen, and the
minimised force field contains the ligand. Rotatable polar hydrogens should therefore
relax toward ligand acceptors.

Test, using only coordinates and element symbols (all the parquet carries):

  * assign every protein H to its nearest protein heavy atom -- its parent. Bond lengths
    separate cleanly (H-O 0.97, H-N 1.01, H-C 1.09 A), so the parent element classifies
    the H as polar (N/O-bound, rotatable) or nonpolar (C-bound, pinned by its frozen parent).
  * for each H, compare the distance to the nearest ligand acceptor (N/O) from the H itself
    against that from its parent. `parent - H` distance is the projection of the bond vector
    onto the direction of the ligand: positive means the H points at the ligand.

Nonpolar C-H is the built-in null. It cannot rotate (its parent is frozen and its geometry
is fixed by the heavy-atom frame), so whatever bias it shows is the geometric baseline.
Polar H showing a materially larger bias than nonpolar H is ligand-conditioning.
"""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import torch

from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.poc2mol.datasets import ParquetDataset

POCKET_CUTOFF = 6.0     # A from any ligand heavy atom
HBOND_CUTOFF = 2.6      # H...acceptor distance for a hydrogen bond


def analyse(n_complexes=300):
    cfg = Poc2MolDataConfig(batch_size=1)
    ds = ParquetDataset(config=cfg, data_path="../hiqbind/parquet/val",
                        rotate=False, translation=0.0, use_cluster_member_zero=True)
    stats = {k: [] for k in ("polar_pocket", "nonpolar_pocket", "polar_distal", "nonpolar_distal")}
    hbonds = {"polar_pocket": [0, 0], "nonpolar_pocket": [0, 0]}

    seen = 0
    for cluster_i in range(len(ds.cluster_ids)):
        if seen >= n_complexes:
            break
        info = ds.cluster_index[ds.cluster_ids[cluster_i]][0]
        row = ds._get_dataframe(info["file_idx"]).iloc[info["row_idx"]]

        pc = np.asarray(row["protein_coords"], dtype=np.float64).reshape(row["protein_coords_shape"]).T
        lc = np.asarray(row["ligand_coords"], dtype=np.float64).reshape(row["ligand_coords_shape"]).T
        pe = np.char.title(np.asarray(row["protein_element_symbols"]).astype(str))
        le = np.char.title(np.asarray(row["ligand_element_symbols"]).astype(str))

        is_h = pe == "H"
        heavy = ~is_h
        lig_acceptor = np.isin(le, ["N", "O"])
        lig_heavy = le != "H"
        if is_h.sum() == 0 or lig_acceptor.sum() == 0:
            continue
        seen += 1

        h_xyz, heavy_xyz, heavy_el = pc[is_h], pc[heavy], pe[heavy]
        acc_xyz, lig_xyz = lc[lig_acceptor], lc[lig_heavy]

        # parent = nearest protein heavy atom
        d_ph = np.linalg.norm(h_xyz[:, None, :] - heavy_xyz[None, :, :], axis=-1)
        parent = d_ph.argmin(axis=1)
        parent_xyz, parent_el = heavy_xyz[parent], heavy_el[parent]
        polar = np.isin(parent_el, ["N", "O"])

        # distance to nearest ligand acceptor, from H and from its parent
        dH = np.linalg.norm(h_xyz[:, None, :] - acc_xyz[None, :, :], axis=-1).min(axis=1)
        dP = np.linalg.norm(parent_xyz[:, None, :] - acc_xyz[None, :, :], axis=-1).min(axis=1)
        toward = dP - dH                                   # >0 means H points at the ligand

        # pocket membership from the parent, so it does not depend on where the H points
        d_parent_lig = np.linalg.norm(parent_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1).min(axis=1)
        pocket = d_parent_lig < POCKET_CUTOFF

        for key, mask in (("polar_pocket", polar & pocket), ("nonpolar_pocket", ~polar & pocket),
                          ("polar_distal", polar & ~pocket), ("nonpolar_distal", ~polar & ~pocket)):
            if mask.any():
                stats[key].append(toward[mask])
        for key, mask in (("polar_pocket", polar & pocket), ("nonpolar_pocket", ~polar & pocket)):
            if mask.any():
                hbonds[key][0] += int((dH[mask] < HBOND_CUTOFF).sum())
                hbonds[key][1] += int(mask.sum())

    print(f"HiQBind val, {seen} complexes, pocket = parent within {POCKET_CUTOFF} A of a ligand heavy atom\n")
    print(f"{'group':18s} {'n':>9s} {'mean(dParent-dH)':>18s} {'% pointing at ligand':>21s}")
    out = {}
    for key, chunks in stats.items():
        v = np.concatenate(chunks)
        out[key] = (v.mean(), (v > 0).mean(), len(v))
        print(f"{key:18s} {len(v):9d} {v.mean():+18.4f} {(v > 0).mean():20.1%}")

    print("\nH...ligand-acceptor contacts within %.1f A (hydrogen-bond range):" % HBOND_CUTOFF)
    for key, (n, tot) in hbonds.items():
        print(f"  {key:18s} {n:6d} / {tot:6d} = {n / max(tot,1):.2%}")

    pp, np_ = out["polar_pocket"], out["nonpolar_pocket"]
    print("\n--- interpretation ---")
    print(f"  polar-vs-nonpolar excess in the pocket : {pp[0] - np_[0]:+.4f} A")
    print(f"  same excess far from the ligand        : {out['polar_distal'][0] - out['nonpolar_distal'][0]:+.4f} A")
    print("  Nonpolar C-H cannot rotate, so it is the geometric null. A pocket-specific")
    print("  polar excess that vanishes distally is the signature of ligand-conditioned")
    print("  protonation; a uniform offset would just be chemistry (polar H sit on the surface).")


if __name__ == "__main__":
    analyse()
