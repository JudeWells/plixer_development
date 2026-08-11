"""Correctness checks for the batched voxeliser.

Three things are established here:

1. **Ground truth.** A naive float64 CPU implementation of the occupancy function agrees
   with :class:`BatchedVoxelizer` to float32 precision. This is the real correctness test
   -- it does not depend on the legacy code being right.
2. **Equivalence with the legacy voxeliser.** Given the *same* (pre-centred) input,
   ``UnifiedVoxelGrid`` and ``BatchedVoxelizer`` agree. This isolates "did the refactor
   change the maths" from "did centring change the maths".
3. **The size of the centring fix.** Voxelising in absolute PDB coordinates in bfloat16,
   as the legacy path did, is measurably wrong; this quantifies by how much.

Run: venvPlixer/bin/python tests/test_batched_voxelizer.py
"""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import torch

from docktgrid.molecule import MolecularComplex
from docktgrid.molparser import MolecularData

from src.data.common.voxelization.batched import (
    BatchedVoxelizer,
    atom_record_from_complex,
    collate_voxel_inputs,
)
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.common.voxelization.molecule_utils import (
    apply_random_rotation,
    apply_random_translation,
    prune_distant_atoms,
)
from src.data.common.voxelization.voxelizer import UnifiedView, UnifiedVoxelGrid
from src.data.poc2mol.datasets import ParquetDataset

CFG = Poc2MolDataConfig(batch_size=1)
DS = ParquetDataset(config=CFG, data_path="../hiqbind/parquet/train")
VIEW = UnifiedView(CFG)
FAILURES = []


def check(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def build_complex(file_idx, row_idx, transform=False):
    row = DS._get_dataframe(file_idx).iloc[row_idx]
    dt = CFG.dtype
    prot = torch.tensor(row["protein_coords"], dtype=dt).reshape(list(row["protein_coords_shape"]))
    lig = torch.tensor(row["ligand_coords"], dtype=dt).reshape(list(row["ligand_coords_shape"]))
    complex_obj = MolecularComplex(
        MolecularData(None, prot, row["protein_element_symbols"]),
        MolecularData(None, lig, row["ligand_element_symbols"]),
    )
    if transform:
        complex_obj = apply_random_rotation(complex_obj)
        complex_obj = apply_random_translation(complex_obj, 6.0)
    return prune_distant_atoms(complex_obj, 32.0)


def voxelize(vox, batch):
    return vox(batch["atom_xyz"], batch["atom_radius"], batch["atom_slot"],
               batch["batch_size"], batch["n_channels"])


def naive_float64(record, axes_dims, vox_size, box_dims):
    """Reference occupancy, computed the slow obvious way in float64."""
    coords = record["coords"].to(torch.float64)
    radii = record["vdw_radii"].to(torch.float64)
    channels = record["channels"]

    axes = [
        torch.arange(0, int(b / vox_size), dtype=torch.float64) * vox_size - b / 2
        for b in box_dims
    ]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    points = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=0)

    dist = torch.sqrt(((coords.unsqueeze(2) - points.unsqueeze(1)) ** 2).sum(0))  # (A, P)
    occ = 1.0 - torch.exp(-((radii.unsqueeze(1) / dist) ** 12))

    out = torch.zeros(channels.shape[0], points.shape[1], dtype=torch.float64)
    for c in range(channels.shape[0]):
        mask = channels[c]
        if mask.any():
            out[c] = occ[mask].amax(dim=0)
    return out.view(channels.shape[0], *axes_dims)


def main():
    complexes = [build_complex(f, r) for f in range(4) for r in range(3)]
    records = [atom_record_from_complex(c, VIEW, box_dims=CFG.box_dims) for c in complexes]
    batch = collate_voxel_inputs(records)

    vox32 = BatchedVoxelizer(CFG, compute_dtype=torch.float32).to("cuda")
    vox16 = BatchedVoxelizer(CFG, compute_dtype=torch.bfloat16).to("cuda")
    got32 = voxelize(vox32, batch).float().cpu()
    got16 = voxelize(vox16, batch).float().cpu()

    ulp = torch.tensor(1.0, dtype=CFG.dtype)
    ulp = (torch.nextafter(ulp, torch.tensor(2.0, dtype=CFG.dtype)) - ulp).float().item()
    truth = [
        naive_float64(records[i], vox32.axes_dims, CFG.vox_size, CFG.box_dims).float()
        for i in range(3)
    ]

    # Compare before the output cast. config.dtype is bfloat16, whose spacing near 0.5 is
    # 0.0039, so a bfloat16 output can only ever agree to ~2e-3 no matter how exact the
    # arithmetic is -- that would measure the storage dtype, not the implementation.
    print("\n1a. arithmetic, with the neighbourhood cutoff disabled")
    # A cutoff that spans the whole box makes this an exact all-atoms x all-points
    # evaluation, so any disagreement is arithmetic rather than truncation.
    span = max(CFG.box_dims) / min(a.min().item() for a in [batch["atom_radius"]])
    exact = BatchedVoxelizer(CFG, compute_dtype=torch.float32, cutoff_ratio=span).to("cuda")
    exact.out_dtype = torch.float32
    exact_raw = voxelize(exact, batch).cpu()
    worst = max((exact_raw[i] - truth[i]).abs().max().item() for i in range(3))
    check("float32 arithmetic == float64 naive", worst < 1e-5, f"max|diff| = {worst:.2e}")

    print(f"\n1b. neighbourhood cutoff at {vox32.cutoff_ratio}x vdW radius")
    prod = BatchedVoxelizer(CFG, compute_dtype=torch.float32).to("cuda")
    prod.out_dtype = torch.float32
    prod_raw = voxelize(prod, batch).cpu()
    worst = max((prod_raw[i] - truth[i]).abs().max().item() for i in range(3))
    bound = 1.0 - float(np.exp(-vox32.cutoff_ratio ** -12))
    check(
        "truncation stays under the stored resolution",
        worst < ulp / 2,
        f"max|diff| = {worst:.2e}, analytic bound = {bound:.2e}, "
        f"half-ulp of {CFG.dtype} = {ulp / 2:.2e}",
    )

    stored = (got32 - prod_raw).abs().max().item()
    check(
        f"{CFG.dtype} output within half an ulp",
        stored <= ulp / 2 + 1e-9,
        f"max|diff| = {stored:.2e}, half-ulp = {ulp / 2:.2e}",
    )

    print("\n2. equivalence with the legacy voxeliser on identical (pre-centred) input")
    legacy = UnifiedVoxelGrid(CFG)
    worst = 0.0
    for i, complex_obj in enumerate(complexes[:4]):
        centred = build_complex(i // 3, i % 3)
        centre = centred.ligand_center.clone()
        centred.coords = centred.coords - centre.unsqueeze(1)
        centred.ligand_center = torch.zeros_like(centre)
        want = legacy.voxelize(centred).float().cpu()
        worst = max(worst, (got16[i] - want).abs().max().item())
    check(
        "bfloat16 batched == legacy voxeliser",
        worst < 0.06,
        f"max|diff| = {worst:.4f} (bfloat16 resolution on a 0-1 occupancy)",
    )

    print("\n3. padding and batching must not perturb a sample")
    solo = voxelize(vox32, collate_voxel_inputs([records[0]]))
    delta = (solo[0].float().cpu() - got32[0]).abs().max().item()
    check("sample alone == sample in batch", delta == 0.0, f"max|diff| = {delta:.2e}")
    finite = bool(torch.isfinite(got32).all())
    check("output is finite", finite, f"no NaN/Inf = {finite}")

    print("\n4. how much the centring fix changes the data (information, not pass/fail)")
    mags = np.array([c.ligand_center.abs().max().item() for c in complexes])
    print(f"     |ligand_center|_inf: median {np.median(mags):.1f} A, max {mags.max():.1f} A")
    legacy_uncentred = torch.stack(
        [UnifiedVoxelGrid(CFG).voxelize(c).float().cpu() for c in complexes]
    )
    diff = (got32 - legacy_uncentred).abs()
    occupied = got32 > 0.01
    print(f"     legacy (uncentred bf16) vs fixed (centred fp32):")
    print(f"       max|diff|                            = {diff.max().item():.4f}")
    print(f"       mean|diff| over occupied voxels      = {diff[occupied].mean().item():.4f}")
    print(f"       occupied voxels wrong by >0.05       = {(diff[occupied] > 0.05).float().mean().item():.2%}")
    bf_gap = (got32 - got16).abs()
    print(f"     residual bfloat16-vs-float32 gap after centring:")
    print(f"       occupied voxels differing by >0.05   = {(bf_gap[occupied] > 0.05).float().mean().item():.2%}")

    print()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
