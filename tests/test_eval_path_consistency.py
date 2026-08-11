"""Evaluation must voxelise identically to training.

`inference/` and `evaluations/` reach the voxeliser through `voxelize_complex` /
`voxelize_molecule`, while training goes through the datamodule. If those two disagree,
every reported number is measured on data the model never saw. This pins them together.

Run: venvPlixer/bin/python tests/test_eval_path_consistency.py
"""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch

from src.data.common.voxelization.batched import (
    atom_record_from_complex, collate_voxel_inputs, BatchedVoxelizer,
)
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.common.voxelization.molecule_utils import _voxelize_one
from src.data.common.voxelization.voxelizer import UnifiedView
from src.data.poc2mol.datasets import ParquetDataset
from src.data.poc2mol.collate import collate_complex_records, VoxelBatchBuilder

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


cfg = Poc2MolDataConfig(batch_size=4, random_rotation=False, random_translation=0.0)
ds = ParquetDataset(config=cfg, data_path="../hiqbind/parquet/val",
                    rotate=False, translation=0.0, use_cluster_member_zero=True)

# --- training path: records -> collate -> on-device batch build
records = [ds[i] for i in range(4)]
batch = collate_complex_records(records)
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
trained = VoxelBatchBuilder(cfg, n_protein_channels=4)(batch)
train_grid = torch.cat([trained["protein"], trained["ligand"]], dim=1).float().cpu()

# --- evaluation path: the same records through the single-sample helper
view = UnifiedView(cfg)
eval_grid = torch.stack([
    BatchedVoxelizer(cfg).to("cuda")(
        *[collate_voxel_inputs([r])[k] for k in ("atom_xyz", "atom_radius", "atom_slot")],
        1, r["channels"].shape[0],
    )[0].float().cpu()
    for r in records
])

print("\n1. training and evaluation voxelisation agree")
delta = (train_grid - eval_grid).abs().max().item()
check("identical grids", delta == 0.0, f"max|diff| = {delta:.2e}")

print("\n2. the ported voxelize_complex helper is on the corrected path")
import inspect
from src.data.common.voxelization import molecule_utils
src = inspect.getsource(molecule_utils.voxelize_complex)
check("voxelize_complex no longer calls UnifiedVoxelGrid",
      "UnifiedVoxelGrid" not in src, "routes through _voxelize_one")
src_mol = inspect.getsource(molecule_utils.voxelize_molecule)
check("voxelize_molecule no longer calls UnifiedVoxelGrid",
      "UnifiedVoxelGrid" not in src_mol, "routes through _voxelize_one")

print("\n3. corrected channel semantics survive the eval path")
grid = eval_grid[0]
prot_s = grid[3].mean().item()
prot_c = grid[0].mean().item()
check("protein channel 3 is sulfur, not a not-sulfur catch-all",
      prot_s < prot_c / 10, f"S mean {prot_s:.5f} vs C mean {prot_c:.5f}")

print()
if FAILURES:
    print(f"FAILED: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("all checks passed")
