from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

import torch

from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.data.common.voxelization.batched import BatchedVoxelizer
from src.data.vox2smiles.datasets import Vox2SmilesDataset, get_collate_function
from src.data.vox2smiles.poc2mol_inference import Poc2MolInferenceBuilder
from src.data.common.protein_channels import assemble_decoder_input


class LigandVoxelBuilder:
    """Turn collated atom records into ``pixel_values`` on the rank's own device.

    Bound lazily to the batch's device: under DDP each rank has a different one and
    Lightning only assigns it after the process group is up, so the device is not known
    when the datamodule is constructed.
    """

    def __init__(self, config, compute_dtype=torch.float32, inject_protein=False,
                 n_protein_channels=0, protein_mask_probability=0.0):
        self.config = config
        self.compute_dtype = compute_dtype
        self.inject_protein = inject_protein
        self.n_protein_channels = n_protein_channels
        self.protein_mask_probability = protein_mask_probability
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

    def __call__(self, batch, training: bool = True):
        if "atom_xyz" not in batch:
            return batch  # e.g. Poc2MolOutputDataset, which supplies pixel_values directly

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
        needs = out.pop("needs_poc2mol", None)

        if self.n_protein_channels:
            protein = grid[:, : self.n_protein_channels]
            ligand = grid[:, self.n_protein_channels :]
        else:
            protein, ligand = None, grid

        # Ligand-only pretraining: the protein channels exist but are always empty, and the
        # flag plane tells the decoder so. That is what lets the same weights be fine-tuned
        # on pocket data later.
        out["pixel_values"] = assemble_decoder_input(
            ligand, protein, needs, self.inject_protein,
            self.protein_mask_probability, training,
        )
        return out


class Vox2SmilesDataModule(LightningDataModule):
    """
    Lightning data module for voxelized molecules with SMILES strings.
    """
    def __init__(
        self,
        config: Vox2SmilesDataConfig,
        data_path: str,
        val_split: float = 0.1,
        test_split: float = 0.1,
        train_dataset = None,
        val_datasets = None,  # New unified mapping of validation datasets
        test_dataset = None,
        num_workers = 0,
        poc2mol_model = None,
        poc2mol_ckpt_path: str = None,
        max_poc2mol_loss: float = None,
        quality_filter: str = "none",
        n_protein_channels: int = 4,
        inject_protein: bool = False,
        protein_mask_probability: float = 0.0,
        predicted_ligand_probability: float = 1.0,
        predicted_ramp_start_step: int = 0,
        predicted_ramp_end_step: int = 0,
    ):
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset_provided = train_dataset
        self.val_datasets_provided = {}

        if val_datasets is not None:
            self.val_datasets_provided.update(val_datasets)

        self.test_dataset_provided = test_dataset
        self.tokenizer = build_smiles_tokenizer()
        self.collate_fn = get_collate_function(self.tokenizer)
        self.num_workers = num_workers
        # The frozen Poc2Mol used to live inside Poc2MolOutputDataset, pinned to cuda:0.
        # It now runs here, batched, on whichever device the batch landed on.
        if poc2mol_model is not None:
            self.voxel_builder = Poc2MolInferenceBuilder(
                config,
                poc2mol_model=poc2mol_model,
                ckpt_path=poc2mol_ckpt_path,
                n_protein_channels=n_protein_channels if config.has_protein else 0,
                max_poc2mol_loss=max_poc2mol_loss,
                quality_filter=quality_filter,
                pad_token_id=self.tokenizer.pad_token_id,
                inject_protein=inject_protein,
                protein_mask_probability=protein_mask_probability,
                predicted_ligand_probability=predicted_ligand_probability,
                predicted_ramp_start_step=predicted_ramp_start_step,
                predicted_ramp_end_step=predicted_ramp_end_step,
            )
        else:
            self.voxel_builder = LigandVoxelBuilder(
                config,
                inject_protein=inject_protein,
                n_protein_channels=n_protein_channels if config.has_protein else 0,
                protein_mask_probability=protein_mask_probability,
            )

    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        """Voxelise, and run the frozen Poc2Mol, on the batch's own device."""
        if isinstance(self.voxel_builder, Poc2MolInferenceBuilder):
            # The quality filter drops poor Poc2Mol reconstructions from the training
            # loss; validation must see everything or the metric is measured on a
            # self-selected subset.
            training = getattr(self.trainer, "training", True) if self.trainer else True
            step = int(getattr(self.trainer, "global_step", 0)) if self.trainer else 0
            return self.voxel_builder(batch, apply_quality_filter=training,
                                      training=training, global_step=step)
        training = getattr(self.trainer, "training", True) if self.trainer else True
        return self.voxel_builder(batch, training=training)

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            # Use provided datasets if available, otherwise create default ones
            if self.train_dataset_provided is not None:
                self.train_dataset = self.train_dataset_provided
            else:
                self.train_dataset = Vox2SmilesDataset(
                    data_path=f"{self.data_path}/train",

                    config=self.config,
                    random_rotation=self.config.random_rotation,
                    random_translation=self.config.random_translation
                )
            
            # ---------------- Validation datasets ----------------
            if self.val_datasets_provided:
                # User supplied mapping of datasets; use directly
                self.val_datasets = self.val_datasets_provided
            else:
                # Legacy behaviour – single default validation dataset
                default_val_ds = Vox2SmilesDataset(
                    data_path=f"{self.data_path}/val_5k",
                    config=self.config,
                    random_rotation=False,
                    random_translation=0.0,
                )
                self.val_datasets = {"default_val": default_val_ds}
        
        if stage == 'test' or stage is None:
            if self.test_dataset_provided is not None:
                self.test_dataset = self.test_dataset_provided
            else:
                self.test_dataset = Vox2SmilesDataset(
                    data_path=f"{self.data_path}/test",
                    config=self.config,
                    random_rotation=False,  # No rotation for testing
                    random_translation=0.0   # No translation for testing
                )

    def train_dataloader(self):
        """Get the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    @property
    def val_dataset_names(self):
        """Validation dataset keys in dataloader order."""
        return list(getattr(self, "val_datasets", {}) or {})

    @property
    def val_dataset_kinds(self):
        """Reporting split per validation dataloader, in dataloader order.

        The model logs `val/<kind>/<metric>` from this, so the labelling follows the actual
        dataset type rather than a positional assumption -- adding or reordering a val
        dataset cannot silently relabel a curve.

        Classified by dataset class, not by name: a Poc2MolOutputDataset is conditioned on
        the pocket (and, at validation, on Poc2Mol's prediction), anything else is
        ligand-only ZINC-style data.
        """
        kinds = []
        for dataset in (getattr(self, "val_datasets", {}) or {}).values():
            is_poc2mol = type(dataset).__name__ == "Poc2MolOutputDataset" or hasattr(
                dataset, "complex_dataset"
            )
            kinds.append("poc2mol" if is_poc2mol else "zinc")
        return kinds

    def val_dataloader(self):
        """Get the validation data loader."""
        loaders = []
        for name, dataset in self.val_datasets.items():
            # Determine batch size – prefer dataset.config.val_batch_size if present
            batch_size = getattr(self.config, 'val_batch_size', getattr(self.config, 'batch_size', 32))
            # Allow dataset to override via attribute `batch_size`
            if hasattr(dataset, 'config') and hasattr(dataset.config, 'val_batch_size'):
                batch_size = dataset.config.val_batch_size
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                    pin_memory=False,
                    persistent_workers=True if self.num_workers > 0 else False,
                )
            )
        return loaders

    def test_dataloader(self):
        """Get the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        ) 