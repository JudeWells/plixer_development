"""Collate and on-device voxelisation for the Poc2Mol pipeline.

The split of responsibilities is the point of this module:

* dataloader **workers** produce CPU atom records (see
  :func:`src.data.common.voxelization.batched.atom_record_from_complex`),
* :func:`collate_complex_records` flattens a list of them into one ragged batch,
* :class:`VoxelBatchBuilder` turns that into voxel grids on whichever device the batch has
  already been moved to, i.e. the rank's own GPU.

Nothing here touches CUDA until the last step, which is what makes ``num_workers > 0`` and
DDP possible at all.
"""

from __future__ import annotations

import torch

from src.data.common.voxelization.batched import BatchedVoxelizer, collate_voxel_inputs


def collate_complex_records(records):
    """Merge per-sample atom records into a ragged batch plus passthrough metadata."""
    if "coords" not in records[0]:
        raise TypeError(
            "Expected CPU atom records but got pre-voxelised samples. ComplexDataset (the "
            "PDB/MOL2 path used by evaluations/) has not been ported off UnifiedVoxelGrid "
            "yet -- see CLAUDE.md open item 7. ParquetDataset is the supported path."
        )
    batch = collate_voxel_inputs(records)
    batch["name"] = [r["name"] for r in records]
    batch["smiles"] = [r["smiles"] for r in records]
    if "cluster" in records[0]:
        batch["cluster"] = [r["cluster"] for r in records]
    if "load_time" in records[0]:
        batch["load_time"] = torch.tensor([r["load_time"] for r in records])
    return batch


class VoxelBatchBuilder:
    """Builds voxel grids from a collated batch, lazily binding to the batch's device.

    The voxeliser cannot be constructed against a fixed device at config time: under DDP
    each rank gets a different one, and Lightning only assigns it after the process group
    is up. Binding on first use inside ``on_after_batch_transfer`` gets it right without
    the caller having to know the rank.
    """

    def __init__(self, config, n_protein_channels: int, compute_dtype=torch.float32):
        self.config = config
        self.n_protein_channels = n_protein_channels
        self.compute_dtype = compute_dtype
        self._voxelizer = None
        self._device = None

    def _get(self, device):
        if self._voxelizer is None or self._device != device:
            self._voxelizer = BatchedVoxelizer(
                self.config,
                compute_dtype=self.compute_dtype,
                cutoff_ratio=self.config.get("voxel_cutoff_ratio", 2.0),
            aggregation=self.config.get("voxel_aggregation", "max"),
            radius_scale=self.config.get("voxel_radius_scale", 1.0),
            ).to(device)
            self._device = device
        return self._voxelizer

    def __call__(self, batch):
        """Replace the ragged atom fields with ``protein`` and ``ligand`` voxel grids."""
        if "atom_xyz" not in batch:
            return batch  # already voxelised, or a batch that carries no atoms

        voxelizer = self._get(batch["atom_xyz"].device)
        grid = voxelizer(
            batch["atom_xyz"],
            batch["atom_radius"],
            batch["atom_slot"],
            batch["batch_size"],
            batch["n_channels"],
        )

        out = {k: v for k, v in batch.items() if not k.startswith("atom_")}
        out.pop("n_channels", None)
        out.pop("batch_size", None)

        if self.n_protein_channels:
            out["protein"] = grid[:, : self.n_protein_channels]
            out["ligand"] = grid[:, self.n_protein_channels :]
        else:
            out["protein"] = None
            out["ligand"] = grid
        return out
