from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.poc2mol.collate import VoxelBatchBuilder, collate_complex_records
from src.data.poc2mol.datasets import ComplexDataset, DockstringTestDataset


class ComplexDataModule(LightningDataModule):
    """
    Lightning data module for protein-ligand complexes.
    """
    def __init__(
        self,
        config: Poc2MolDataConfig,
        pdb_dir: str,
        val_pdb_dir: str,
        test_pdb_dir: Optional[str] = None,
        num_workers: int = 0,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = 4,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        # Historically hardcoded to min(4, batch_size), which is fine for a metric that is
        # a plain average over the split but starves any metric evaluated on a fixed budget
        # of batches -- the generative model's sampled Dice sees 4 pockets per batch, and
        # selecting a checkpoint on 8 pockets is selecting on noise. None keeps the old
        # behaviour exactly.
        self.val_batch_size = val_batch_size
        self.pdb_dir = pdb_dir
        self.val_pdb_dir = val_pdb_dir
        self.test_pdb_dir = test_pdb_dir or val_pdb_dir  # Use val_pdb_dir as default for test
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        # Datasets now emit CPU atom records; the grids are built on the rank's own device
        # in on_after_batch_transfer. See src/data/poc2mol/collate.py for why.
        self.batch_builder = VoxelBatchBuilder(
            config,
            n_protein_channels=len(config.protein_channels) if config.has_protein else 0,
        )

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                self.train_dataset = ComplexDataset(
                    self.config,
                    pdb_dir=self.pdb_dir,
                    translation=self.config.random_translation,
                    rotate=self.config.random_rotation,
                )

            if self.val_dataset is None:
                self.val_dataset = ComplexDataset(
                    self.config,
                    pdb_dir=self.val_pdb_dir,
                    translation=0.0,  # No translation for validation
                    rotate=False,     # No rotation for validation
                )

        if stage == 'test' or stage is None:
            if self.test_dataset is None:
                self.test_dataset = ComplexDataset(
                    self.config,
                    pdb_dir=self.test_pdb_dir,
                    translation=0.0,  # No translation for testing
                    rotate=False,     # No rotation for testing
                )

    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        """Voxelise on the device the batch has just been moved to."""
        return self.batch_builder(batch)

    def _loader(self, dataset, batch_size, shuffle, num_workers=None):
        num_workers = self.num_workers if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_complex_records,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=self.prefetch_factor if num_workers > 0 else None,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        """Get the training data loader."""
        return self._loader(self.train_dataset, self.config.batch_size, shuffle=True)

    def _eval_batch_size(self):
        return self.val_batch_size or min(4, self.config.batch_size)

    def val_dataloader(self):
        """Get the validation data loader."""
        return self._loader(self.val_dataset, self._eval_batch_size(), shuffle=False)

    def test_dataloader(self):
        """Get the test data loader."""
        return self._loader(self.test_dataset, self._eval_batch_size(), shuffle=False)
