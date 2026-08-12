"""Mixed pocket + ligand-only batches for stage B of the generative Poc2Mol.

Fine-tuning the ZINC-pretrained flow model on HiQBind alone risks two things: catastrophic
forgetting of the density prior that 8.86M molecules bought, and overfitting to 9,872
clusters. Keeping ZINC in the mix at 50% addresses both, and costs nothing extra to
implement because the ZINC rows are exactly the unconditional branch classifier-free
guidance already trains on.

How the two sources share a batch
---------------------------------
A HiQBind record carries protein + ligand atoms (4 + 11 channels); a ZINC record carries
ligand atoms only. They are made shape-compatible by giving the ZINC dataset a voxel config
with ``has_protein: true``: :class:`StoredLigandComplex` reports ``n_atoms_protein = 0``, so
``UnifiedView.get_protein_channels`` masks every atom out and the 4 protein slots are emitted
EMPTY. Same channel count, same layout, no padding logic, and the all-zero pocket is exactly
what the model's unconditional branch expects. This mirrors what the Vox2Smiles decoder
already does for its combined stage.

Validation is deliberately NOT mixed: it is two separate dataloaders, so ``val/hiqbind/*``
and ``val/zinc/*`` are clean, independently-sized metrics rather than a blend whose value
depends on the sampling ratio. Checkpoint selection still uses the pocket metric.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.common.voxelization.batched import BatchedVoxelizer, collate_voxel_inputs


class MixedSourceDataset(Dataset):
    """Draw from a pocket dataset or a ligand-only dataset, tagging which one.

    ``samples_per_epoch`` defaults to twice the pocket dataset's length so that an epoch
    contains about as many POCKET samples as a pocket-only epoch would. That keeps epoch
    numbers, validation cadence and the ``val/sample/dice`` curve comparable with the
    from-scratch control arm, which is the whole point of running a control.
    """

    def __init__(
        self,
        pocket_dataset: Dataset,
        ligand_dataset: Dataset,
        prob_pocket: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        seed: int = 0,
    ):
        if not 0.0 <= prob_pocket <= 1.0:
            raise ValueError(f"prob_pocket must be in [0, 1], got {prob_pocket}")
        self.pocket_dataset = pocket_dataset
        self.ligand_dataset = ligand_dataset
        self.prob_pocket = prob_pocket
        self.samples_per_epoch = samples_per_epoch or 2 * len(pocket_dataset)
        self.seed = seed

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Seeded per index rather than from global RNG state: dataloader workers each get
        # their own torch/numpy seed, and an unseeded draw here would make the source mix
        # depend on worker count.
        rng = np.random.default_rng(self.seed + idx)
        if rng.random() < self.prob_pocket:
            record = dict(self.pocket_dataset[int(rng.integers(len(self.pocket_dataset)))])
            record["has_pocket"] = True
        else:
            record = dict(self.ligand_dataset[int(rng.integers(len(self.ligand_dataset)))])
            record["has_pocket"] = False
        return record


def collate_mixed_records(records):
    """Flatten mixed atom records into one ragged batch, carrying the source tag."""
    if "coords" not in records[0]:
        raise TypeError("expected CPU atom records; grids are built on the rank's device")
    channel_counts = {int(r["channels"].shape[0]) for r in records}
    if len(channel_counts) != 1:
        raise ValueError(
            f"records disagree on channel count {sorted(channel_counts)}. The ligand-only "
            "dataset must be configured with has_protein: true so its (empty) protein "
            "slots are emitted and both sources share a layout."
        )
    batch = collate_voxel_inputs(records)
    batch["has_pocket"] = torch.tensor(
        [bool(r.get("has_pocket", True)) for r in records], dtype=torch.bool
    )
    batch["name"] = [r.get("name", r.get("smiles_str", ""))[:64] for r in records]
    batch["smiles"] = [r.get("smiles", r.get("smiles_str", "")) for r in records]
    return batch


class MixedVoxelBatchBuilder:
    """Voxelise on the batch's device and split protein/ligand, zeroing absent pockets."""

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
        if "atom_xyz" not in batch:
            return batch

        voxelizer = self._get(batch["atom_xyz"].device)
        grid = voxelizer(
            batch["atom_xyz"], batch["atom_radius"], batch["atom_slot"],
            batch["batch_size"], batch["n_channels"],
        )
        out = {k: v for k, v in batch.items() if not k.startswith("atom_")}
        out.pop("n_channels", None)
        out.pop("batch_size", None)

        protein = grid[:, : self.n_protein_channels]
        out["ligand"] = grid[:, self.n_protein_channels :]
        # Ligand-only rows already voxelise to an empty pocket; this makes that a guarantee
        # rather than a property of the channel masking two modules away.
        if "has_pocket" in out:
            mask = out["has_pocket"].to(protein.device).view(-1, *([1] * (protein.dim() - 1)))
            protein = torch.where(mask, protein, torch.zeros_like(protein))
        out["protein"] = protein
        return out


class MixedFlowDataModule(LightningDataModule):
    """Train on a pocket/ligand-only mixture; validate on each source separately."""

    def __init__(
        self,
        config,
        pocket_train_dataset: Dataset,
        ligand_train_dataset: Dataset,
        pocket_val_dataset: Dataset,
        ligand_val_dataset: Optional[Dataset] = None,
        prob_pocket: float = 0.5,
        samples_per_epoch: Optional[int] = None,
        num_workers: int = 12,
        n_protein_channels: int = 4,
        val_batch_size: Optional[int] = None,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 4,
        seed: int = 0,
        ligand_config=None,
    ):
        # `ligand_config` is accepted and ignored. It lives under `data:` in the YAML purely
        # as the interpolation source for the ZINC dataset's voxel config, and Hydra hands
        # every key under that node to this constructor.
        super().__init__()
        self.config = config
        self.prob_pocket = prob_pocket
        self.num_workers = num_workers
        self.n_protein_channels = n_protein_channels
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.val_batch_size = val_batch_size or min(4, config.batch_size)

        self.train_dataset = MixedSourceDataset(
            pocket_train_dataset, ligand_train_dataset,
            prob_pocket=prob_pocket, samples_per_epoch=samples_per_epoch, seed=seed,
        )
        # Order fixes the metric names, so it is part of the contract: index 0 is the
        # pocket set, and that is the one checkpoints are selected on.
        self.val_datasets = {"hiqbind": pocket_val_dataset}
        if ligand_val_dataset is not None:
            self.val_datasets["zinc"] = ligand_val_dataset

        self.batch_builder = MixedVoxelBatchBuilder(config, n_protein_channels)

    @property
    def val_dataset_kinds(self):
        """Source name per validation dataloader, in dataloader order."""
        return list(self.val_datasets)

    def setup(self, stage: Optional[str] = None):
        return

    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        return self.batch_builder(batch)

    def _loader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_mixed_records,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        return [self._loader(d, self.val_batch_size, shuffle=False)
                for d in self.val_datasets.values()]
