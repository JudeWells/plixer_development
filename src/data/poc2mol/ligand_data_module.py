"""Ligand-only voxel batches, for unconditionally pretraining the generative Poc2Mol.

HiQBind has 9,872 clusters. At effective batch 1536 that is ~6 optimiser steps per epoch,
which is thin for a generative model that has to learn what a plausible ligand density
looks like *at all* before it can learn which one a pocket implies. ZINC20 supplies
8,963,333 molecules with no pocket, and the same trick the Vox2Smiles decoder already uses
applies here: train on ligand grids with the protein channels present but identically zero,
which is exactly the unconditional branch classifier-free guidance needs. The pretrained
weights then transfer to the pocket-conditioned stage without a single shape change --
provided ``n_protein_channels`` matches the fine-tuning data config, which is why it is
explicit here rather than inferred.

The dataset itself is reused unchanged from the Vox2Smiles pipeline
(:class:`~src.data.vox2smiles.datasets.ParquetVox2SmilesDataset`): identical voxelisation,
identical channel map, identical augmentation. Only the collation differs -- the SMILES
tokens it emits are ignored here, and the batch is reshaped into the
``{'protein', 'ligand'}`` form the Poc2Mol models consume.
"""

from __future__ import annotations

from typing import Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.common.voxelization.batched import BatchedVoxelizer, collate_voxel_inputs


def collate_ligand_records(records):
    """Merge per-sample atom records into a ragged batch, dropping the SMILES tokens."""
    if "coords" not in records[0]:
        raise TypeError(
            "Expected CPU atom records; this datamodule builds grids on the rank's device "
            "in on_after_batch_transfer."
        )
    batch = collate_voxel_inputs(records)
    batch["smiles"] = [r.get("smiles_str", r.get("smiles", "")) for r in records]
    batch["name"] = [r.get("name", f"mol_{i}") for i, r in enumerate(records)]
    return batch


class LigandVoxelBatchBuilder:
    """Build ligand grids on the batch's device and pad the protein slot with zeros.

    Bound lazily to the device for the same reason as
    :class:`src.data.poc2mol.collate.VoxelBatchBuilder`: under DDP each rank has a
    different one and Lightning assigns it only after the process group is up.
    """

    def __init__(self, config, n_protein_channels: int = 4, compute_dtype=torch.float32):
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
        if "atom_xyz" not in batch:
            return batch

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
        out["ligand"] = grid
        # Zeros, not None: the pretrained weights must have the same input width as the
        # pocket-conditioned stage, and an all-zero pocket is precisely the unconditional
        # branch the model is trained to handle via condition dropout.
        out["protein"] = (
            grid.new_zeros(grid.shape[0], self.n_protein_channels, *grid.shape[2:])
            if self.n_protein_channels
            else None
        )
        return out


class LigandOnlyDataModule(LightningDataModule):
    """Datamodule over ligand-only voxel grids (ZINC20), for generative pretraining."""

    def __init__(
        self,
        config,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        num_workers: int = 8,
        n_protein_channels: int = 4,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 4,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.val_batch_size = val_batch_size or min(
            8, getattr(config, "batch_size", 8)
        )
        self.batch_builder = LigandVoxelBatchBuilder(config, n_protein_channels)

    def setup(self, stage: Optional[str] = None):
        # Datasets are always supplied by config; nothing to construct here.
        return

    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        return self.batch_builder(batch)

    def _loader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_ligand_records,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, self.val_batch_size, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return self._loader(self.test_dataset, self.val_batch_size, shuffle=False)
