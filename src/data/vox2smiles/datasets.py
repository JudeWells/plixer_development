import os
import glob
import pickle
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.voxelization.batched import atom_record_from_complex, collate_voxel_inputs
from src.data.common.voxelization.voxelizer import (
    UnifiedView,
    RDkitMolecularComplex,
    StoredLigandComplex,
)
from src.data.common.voxelization.molecule_utils import (
    load_mol_from_pickle,
    prepare_rdkit_molecule,
    apply_random_rotation,
    apply_random_translation,
    prune_distant_atoms,
    voxelize_molecule
)

# Per-atom feature columns written by scripts/regenerate_zinc_parquet.py, and the marker
# that a row is parquet_v2 at all.
_V2_FEATURE_KEYS = ("is_aromatic", "n_hydrogens", "is_acceptor", "formal_charge")


def molecule_atom_record(mol, voxel_config, view: UnifiedView) -> dict:
    """CPU atom record for a standalone RDKit molecule (the ligand-only path)."""
    complex_obj = prepare_rdkit_molecule(mol, voxel_config)
    return atom_record_from_complex(
        complex_obj,
        view,
        box_dims=voxel_config.box_dims,
        cutoff_ratio=voxel_config.get("voxel_cutoff_ratio", 2.0),
    )


def stored_molecule_atom_record(row, voxel_config, view: UnifiedView) -> dict:
    """CPU atom record from parquet_v2's stored arrays -- the RDKit-free ligand path.

    Mirrors ``prepare_rdkit_molecule`` -> ``atom_record_from_complex`` exactly, including the
    order of augmentations, so v2 records are drop-in interchangeable with v1 ones. The only
    behavioural differences come from the data: float32 coordinates instead of
    bfloat16-quantised ones, and per-atom features so the 11-channel scheme resolves.

    Hydrogens are left in place rather than honouring ``include_hydrogens``. They are stored
    (regeneration used ``removeHs=False``, mirroring HiQBind) but belong to no channel -- the
    catch-all enumerates H and is inverted -- so ``atom_record_from_complex`` drops them via
    its ``keep`` mask. Stripping them here would change nothing except the work done.
    """
    # np.array (a copy), not np.asarray: parquet-backed arrays are read-only, and a tensor
    # aliasing one would be a silent in-place-write hazard for any future transform.
    coords = torch.from_numpy(
        np.array(row["ligand_coords"], dtype=np.float32)
    ).reshape(list(row["ligand_coords_shape"]))

    features = {
        key: np.asarray(row[f"ligand_{key}"])
        for key in _V2_FEATURE_KEYS
        if f"ligand_{key}" in row
    }
    complex_obj = StoredLigandComplex(coords, row["ligand_element_symbols"], features)

    if voxel_config.random_rotation:
        complex_obj = apply_random_rotation(complex_obj)
    if voxel_config.random_translation > 0:
        complex_obj = apply_random_translation(complex_obj, voxel_config.random_translation)
    if voxel_config.max_atom_dist is not None and voxel_config.max_atom_dist > 0:
        complex_obj = prune_distant_atoms(
            complex_obj, voxel_config.max_atom_dist, voxel_config.has_protein
        )

    return atom_record_from_complex(
        complex_obj,
        view,
        box_dims=voxel_config.box_dims,
        cutoff_ratio=voxel_config.get("voxel_cutoff_ratio", 2.0),
    )


def get_collate_function(tokenizer):
    """
    Create a collate function for the Vox2Smiles dataset.

    Samples arrive as CPU atom records rather than voxel grids -- the grids are built on
    the rank's own device in ``Vox2SmilesDataModule.on_after_batch_transfer``. See
    ``src/data/common/voxelization/batched.py`` for why.
    """

    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch, pad_token_id=pad_token_id):
        """Merge a list of dataset samples into a batch and trim trailing padding.
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])        # (B, L)
        attention_mask = torch.stack([item["attention_mask"] for item in batch])  # (B, L)

        poc2mol_loss = None
        if "poc2mol_loss" in batch[0]:
            poc2mol_loss = torch.tensor([item["poc2mol_loss"] for item in batch])
        all_pad_positions = (input_ids == pad_token_id).all(dim=0)  # (L,)
        if torch.any(all_pad_positions):
            trim_len = torch.where(all_pad_positions)[0][0].item()
            input_ids = input_ids[:, :trim_len]
            attention_mask = attention_mask[:, :trim_len]

        smiles_str = [item["smiles_str"] for item in batch]
        # Optional decoy SMILES list (same for all items typically)
        has_candidates = "candidate_tokens" in batch[0]

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "smiles_str": smiles_str,
            "poc2mol_loss": poc2mol_loss,
        }
        batch_dict.update(collate_voxel_inputs(batch))
        batch_dict["needs_poc2mol"] = torch.tensor(
            [bool(item.get("needs_poc2mol", False)) for item in batch], dtype=torch.bool
        )

        # Samples that carry protein atoms as well as ligand atoms tag them separately, so
        # the two grids can be built and combined independently downstream.
        if "protein_coords" in batch[0]:
            batch_dict.update(collate_voxel_inputs(batch, key_prefix="protein_"))

        if has_candidates:
            batch_dict["candidate_tokens"] = batch[0]["candidate_tokens"]
            batch_dict["binder_indices"] = torch.tensor([item["binder_index"] for item in batch], dtype=torch.long)
        return batch_dict

    return collate_fn


class Vox2SmilesDataset(Dataset):
    """
    Dataset for voxelized molecules with SMILES strings.
    Loads RDKit molecules from pickle files, voxelizes them, and pairs them with SMILES strings.
    """
    def __init__(
        self,
        data_path: str,
        config: Vox2SmilesDataConfig,
        random_rotation: bool = True,
        random_translation: float = 6.0
    ):
        self.data_path = data_path
        self.data = glob.glob(f"{data_path}/*.pickle")
        self.tokenizer = build_smiles_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.max_smiles_len = config.max_smiles_len
        
        # Override config values if provided
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        
        # Store the config
        self.config = config
        
        self.voxel_config = Vox2SmilesDataConfig(
            vox_size=self.config.vox_size,
            box_dims=self.config.box_dims,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
            # Honour the caller's setting rather than forcing ligand-only. With
            # has_protein=True a standalone molecule still yields the protein channel
            # slots, empty -- RDkitMolecularComplex reports n_atoms_protein = 0, so
            # UnifiedView.get_protein_channels masks every atom out. That keeps ZINC and
            # HiQBind records the same shape so they can share a batch, and it is exactly
            # the masked-protein input the decoder needs for ligand-only pretraining.
            has_protein=self.config.has_protein,
            ligand_channel_names=self.config.ligand_channel_names,
            protein_channel_names=self.config.protein_channel_names,
            protein_channels=self.config.protein_channels,
            ligand_channels=self.config.ligand_channels,
            max_atom_dist=self.config.max_atom_dist,
            dtype=self.config.dtype
        )
        self.view = UnifiedView(self.voxel_config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load a pickled RDKit molecule, voxelize it, and pair it with its SMILES string.
        """
        # Load the molecule
        path = self.data[idx]
        with open(path, "rb") as f:
            mol_data = pickle.load(f)
        
        # If there are multiple conformers, randomly select one
        if "conformers" in mol_data:
            conformer_idx = np.random.randint(0, len(mol_data["conformers"]))
            mol = mol_data["conformers"][conformer_idx]["rd_mol"]
        else:
            mol = mol_data["rd_mol"]
        if not self.config.include_hydrogens:
            mol = Chem.RemoveHs(mol)
        
        # Extract a CPU atom record; the grid is built on the rank's device later
        record = molecule_atom_record(mol, self.voxel_config, self.view)

        # Get the SMILES string
        smiles_str = self.tokenizer.bos_token + Chem.MolToSmiles(mol) + self.tokenizer.eos_token

        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt"
        )
        record.update({
            "input_ids": smiles["input_ids"].squeeze(),
            "attention_mask": smiles["attention_mask"].squeeze(),
            "smiles_str": smiles_str,
        })
        return record


import os
import glob
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset

from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.voxelization.molecule_utils import voxelize_molecule


class ParquetVox2SmilesDataset(Dataset):
    """
    Dataset for voxelized molecules with SMILES strings loaded from parquet files.
    Each parquet file contains multiple molecules, allowing efficient storage and loading.
    """
    def __init__(
        self,
        data_path: str,
        config: Vox2SmilesDataConfig,
        index_file: str = "index.csv",
        random_rotation: bool = None,
        random_translation: float = None,
        cache_size: int = 10,
    ):
        self.data_path = data_path
        self.tokenizer = build_smiles_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.max_smiles_len = config.max_smiles_len
        
        # Override config values if provided
        self.random_rotation = random_rotation if random_rotation is not None else config.random_rotation
        self.random_translation = random_translation if random_translation is not None else config.random_translation
        
        # Store the config
        self.config = config
        
        # Load the index file which lists all parquet files
        index_path = os.path.join(data_path, index_file)
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            self.file_list = df['parquet_file'].tolist()
            self.file_sizes = df['file_size'].tolist()
            self.total_molecules = sum(self.file_sizes)
        else:
            print("Indexing molecule files")
            self.file_list = glob.glob(os.path.join(data_path, "**.parquet"))
            print(f"Found {len(self.file_list)} parquet files")
            self.file_sizes = []
            self.total_molecules = 0
            for file_path in self.file_list:
                # Just read the metadata to get row count (faster than loading data)
                df = pd.read_parquet(file_path, columns=['source_file'])
                file_size = len(df)
                self.file_sizes.append(file_size)
                self.total_molecules += file_size
            # Atomic write: under DDP all ranks build this at once and a partially
            # written CSV would be readable by another rank.
            from src.data.poc2mol.datasets import _atomic_write
            _atomic_write(index_path, lambda f: pd.DataFrame(
                {"parquet_file": self.file_list, "file_size": self.file_sizes}
            ).to_csv(f, index=False))
        
        self.molecule_map = []
        for file_idx, size in enumerate(self.file_sizes):
            for row_idx in range(size):
                self.molecule_map.append((file_idx, row_idx))
        print(f"Molecule file index created with {len(self.molecule_map)} molecules")
        self.cache = {}
        self.cache_size = cache_size

        self.voxel_config =Vox2SmilesDataConfig(
            vox_size=self.config.vox_size,
            box_dims=self.config.box_dims,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
            # Honour the caller's setting rather than forcing ligand-only. With
            # has_protein=True a standalone molecule still yields the protein channel
            # slots, empty -- RDkitMolecularComplex reports n_atoms_protein = 0, so
            # UnifiedView.get_protein_channels masks every atom out. That keeps ZINC and
            # HiQBind records the same shape so they can share a batch, and it is exactly
            # the masked-protein input the decoder needs for ligand-only pretraining.
            has_protein=self.config.has_protein,
            ligand_channel_names=self.config.ligand_channel_names,
            protein_channel_names=self.config.protein_channel_names,
            protein_channels=self.config.protein_channels,
            ligand_channels=self.config.ligand_channels,
            max_atom_dist=self.config.max_atom_dist,
            dtype=self.config.dtype
        )
        self.view = UnifiedView(self.voxel_config)

    def __len__(self):
        return self.total_molecules

    def __getitem__(self, idx):
        """
        Load a molecule from a parquet file, voxelize it, and pair it with its SMILES string.
        """
        file_idx, row_idx = self.molecule_map[idx]
        file_path = self.file_list[file_idx]
        
        # Try to get the file from cache
        if file_path not in self.cache:
            # If cache is full, remove the oldest entry
            if len(self.cache) >= self.cache_size:
                # Get the least recently used file
                oldest_file = next(iter(self.cache))
                del self.cache[oldest_file]
            
            # Load the parquet file
            self.cache[file_path] = pd.read_parquet(file_path)
        
        # Get the molecule data
        mol_data = self.cache[file_path].iloc[row_idx]

        # parquet_v2: coordinates and per-atom features are stored directly, so no RDKit
        # parse is needed and none is wanted -- MolFromMolBlock per sample is the stage-1
        # bottleneck (CLAUDE.md §3h). v2 deliberately drops `mol_block`, so its presence is
        # what distinguishes the two layouts.
        if 'mol_block' not in mol_data:
            record = stored_molecule_atom_record(mol_data, self.voxel_config, self.view)
            # The stored SMILES, not MolToSmiles of a reparsed block. It is the same string
            # the v1 path produced (the regeneration carried the column across unchanged),
            # and re-deriving it would put RDKit straight back in the hot path.
            smiles_str = self.tokenizer.bos_token + mol_data['smiles'] + self.tokenizer.eos_token
            smiles = self.tokenizer(
                smiles_str,
                padding='max_length',
                max_length=self.max_smiles_len,
                truncation=True,
                return_tensors="pt",
            )
            record.update({
                "input_ids": smiles["input_ids"].squeeze(),
                "attention_mask": smiles["attention_mask"].squeeze(),
                "smiles_str": smiles_str,
            })
            return record

        # Reconstruct the RDKit molecule from the mol block
        mol_block = mol_data['mol_block']
        mol = Chem.MolFromMolBlock(mol_block.decode() if isinstance(mol_block, bytes) else mol_block)
        if not self.config.include_hydrogens:
            mol = Chem.RemoveHs(mol)
        if mol is None:
            # If we can't parse the mol block, try to create from SMILES
            smiles = mol_data['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self.__getitem__(np.random.randint(0, len(self)))

        
        # Add hydrogens and generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            if self.config.include_hydrogens:
                mol = Chem.AddHs(mol)
            # Use a standard RDKit conformer generation
            try:
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            except:
                # If conformer generation fails, we'll skip this molecule
                # and provide a fallback
                mol = Chem.MolFromSmiles("c1ccccc1")
                if self.config.include_hydrogens:
                    mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        # Extract a CPU atom record; the grid is built on the rank's device later
        record = molecule_atom_record(mol, self.voxel_config, self.view)

        # Get the SMILES string
        smiles_str = self.tokenizer.bos_token + Chem.MolToSmiles(mol) + self.tokenizer.eos_token

        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt"
        )
        if self.tokenizer.unk_token_id in smiles.input_ids:
            print(f"UNK token in SMILES string: {smiles_str}")
        record.update({
            "input_ids": smiles["input_ids"].squeeze(),
            "attention_mask": smiles["attention_mask"].squeeze(),
            "smiles_str": smiles_str,
        })
        return record


class Poc2MolOutputDataset(Dataset):
    """Protein-ligand complexes destined to be turned into Poc2Mol outputs.

    Historically this class *owned* a Poc2Mol model, put it on ``cuda`` in ``__init__``,
    and ran it one sample at a time inside ``__getitem__``. That made DDP impossible --
    every rank and every worker would have loaded the model onto ``cuda:0`` -- and forced
    ``num_workers: 0``, which is why the pipeline ran at 25 samples/s.

    The model now lives in the datamodule and runs batched, under ``no_grad``, on the
    rank's own device (see ``Poc2MolInferenceBuilder``). What is left here is bookkeeping:
    hand back the complex's atom record plus the tokenised ground-truth SMILES, tagged so
    the batch builder knows which samples need a Poc2Mol forward pass.
    """

    def __init__(
        self,
        complex_dataset,
        max_smiles_len=200,
        decoy_smiles_list: list = None,
        include_decoys: bool = True,
        poc2mol_model=None,
        ckpt_path: str = None,
    ):
        if poc2mol_model is not None or ckpt_path is not None:
            raise TypeError(
                "Poc2MolOutputDataset no longer holds a Poc2Mol model. Configure the "
                "checkpoint on the datamodule instead (data.poc2mol_ckpt_path), so the "
                "model is loaded once per rank and run batched on the rank's own device."
            )
        self.complex_dataset = complex_dataset
        self.tokenizer = build_smiles_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.max_smiles_len = max_smiles_len

        self.include_decoys = include_decoys
        if self.include_decoys:
            assert decoy_smiles_list is not None, "decoy_smiles_list must be provided if include_decoys is True"
            self.decoy_smiles_list = decoy_smiles_list
            self.tokenize_decoys()
        else:
            self.decoy_smiles_list = []

    def __len__(self):
        return len(self.complex_dataset)

    def __getitem__(self, idx):
        record = self.complex_dataset[idx]
        smiles_str = record["smiles"]

        binder_idx = None
        if self.include_decoys:
            # Find index of binder in global decoy list
            binder_idx = self.decoy_smiles_list.index(smiles_str)

        smiles_str = self.tokenizer.bos_token + smiles_str + self.tokenizer.eos_token

        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt",
        )

        record = dict(record)
        record.update({
            "input_ids": smiles["input_ids"].squeeze(),
            "attention_mask": smiles["attention_mask"].squeeze(),
            "smiles_str": smiles_str,
            # Marks this sample for a frozen Poc2Mol forward pass in the batch builder.
            "needs_poc2mol": True,
        })

        if self.include_decoys:
            record.update({
                "candidate_tokens": self.tokenized_decoy_smiles,
                "binder_index": binder_idx,
            })

        return record

    def tokenize_decoys(self):
        """Tokenize the global decoy SMILES list **once** and store stacked tensors.

        The result is a dictionary with keys (input_ids, attention_mask, token_type_ids)
        each mapping to a tensor of shape (N_decoys, L).
        """
        smiles_with_tokens = [
            self.tokenizer.bos_token + smi + self.tokenizer.eos_token
            for smi in self.decoy_smiles_list
        ]

        max_len = max(
            len(self.tokenizer.tokenize(s, padding=False, truncation=False))
            for s in smiles_with_tokens
        )

        tokenized = self.tokenizer(
            smiles_with_tokens,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors='pt',
        )

        # Ensure tensors are on CPU to avoid unnecessary GPU memory duplication
        self.tokenized_decoy_smiles = {k: v for k, v in tokenized.items()}


class CombinedDataset(Dataset):
    """
    Dataset that combines Poc2Mol outputs and original Vox2Smiles data.
    This is used for fine-tuning Vox2Smiles on a mix of Poc2Mol outputs and original data.

    Both sources yield atom records with the same channel layout, so they can share a
    batch: the ligand-only source leaves the protein channel slots empty (see
    ``has_protein`` in the voxel config).

    The ``max_poc2mol_loss`` quality filter cannot run here any more -- it needs the
    Poc2Mol loss, which is only known once the model has run, and the model now runs
    batched in the datamodule. The threshold is applied there instead, by masking rejected
    samples out of the language-modelling loss. See ``Poc2MolInferenceBuilder``.
    """
    def __init__(
        self,
        poc2mol_output_dataset,
        vox2smiles_dataset,
        prob_poc2mol=0.5, # probability of poc2mol
        max_poc2mol_loss=1.2, # worst loss tolerated to train on poc2mol
    ):
        self.poc2mol_output_dataset = poc2mol_output_dataset
        self.vox2smiles_dataset = vox2smiles_dataset
        self.prob_poc2mol = prob_poc2mol
        self.max_poc2mol_loss = max_poc2mol_loss
        # Calculate the number of samples from each dataset
        self.n_poc2mol = len(poc2mol_output_dataset)
        self.n_vox2smiles = len(vox2smiles_dataset)

        # Calculate the total number of samples
        self.n_total = self.n_poc2mol + self.n_vox2smiles
        print(f"Total number of samples: {self.n_total}")
        print(f"Number of Poc2Mol samples: {self.n_poc2mol}")
        print(f"Number of Vox2Smiles samples: {self.n_vox2smiles}")

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        """
        Get a sample from either the Poc2Mol output dataset or the Vox2Smiles dataset.
        """
        if np.random.random() < self.prob_poc2mol:
            return self.poc2mol_output_dataset[idx % self.n_poc2mol]

        result = self.vox2smiles_dataset[idx % self.n_vox2smiles]
        result["needs_poc2mol"] = False
        return result
