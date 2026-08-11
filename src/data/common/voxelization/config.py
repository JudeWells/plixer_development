from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import torch

class ConfigClass:
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)


def resolve_dtype(value: Any) -> torch.dtype:
    """Coerce a config dtype to a real ``torch.dtype``.

    The Hydra configs express it as ``${oc.select:torch.bfloat16,torch.bfloat16}``, which
    resolves to the *string* ``"torch.bfloat16"`` rather than the dtype object. The old
    code worked around this with scattered ``eval(...)`` calls at every use site; this is
    the single place that conversion happens.
    """
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.rsplit(".", 1)[-1]
        resolved = getattr(torch, name, None)
        if isinstance(resolved, torch.dtype):
            return resolved
    raise ValueError(f"Cannot interpret {value!r} as a torch dtype")

# ---------------------------------------------------------------------------------------
# The 11-channel ligand scheme (parquet_v2). Splits on the two things a pocket most directly
# constrains -- hybridisation and donor/acceptor identity -- while keeping enough element
# identity for the decoder to name atoms.
#
# Measured on 3,840 HiQBind ligands: every channel is populated (worst is `other` at 9.8% of
# ligands, P(dead in a 128-batch) = 1.7e-06) against the old scheme where iodine was dead in
# 29% of batches. Information share spans 1.6-16.9% instead of 0.4-26%.
#
# ORDER MATTERS: the catch-all must be LAST, because `ligand_last_channel_is_catch_all`
# inverts `ligand_chs[-1]`. Cl/Br/I are merged because they are sterically similar, all
# halogen-bond donors and routinely interchanged in SAR -- merged they reach 18.7% of ligands
# versus 0.96% for iodine alone. F is kept separate: 1.47 A, no halogen bonding, a different
# chemical role. An H-bond DONOR channel was considered and dropped as ~100% redundant with
# N_withH + O_withH (2.36 vs 2.35 atoms/ligand).
# ---------------------------------------------------------------------------------------
LIGAND_CHANNELS_V2 = {
    0: ["C_aliphatic"],
    1: ["C_aromatic"],
    2: ["N_withH"],
    3: ["N_noH"],
    4: ["O_withH"],
    5: ["O_noH"],
    6: ["S"],
    7: ["Cl", "Br", "I"],
    8: ["F"],
    9: ["HBA"],
    # catch-all: inverted at match time, so this becomes "not any common element", i.e.
    # P/Se/B/metals, and drops hydrogen. Must remain the last entry.
    10: ["C", "H", "O", "N", "S", "Cl", "F", "I", "Br"],
}

LIGAND_CHANNEL_NAMES_V2 = [
    "carbon_aliphatic", "carbon_aromatic", "nitrogen_with_h", "nitrogen_no_h",
    "oxygen_with_h", "oxygen_no_h", "sulfur", "halogen", "fluorine",
    "hbond_acceptor", "other",
]


@dataclass
class VoxelizationConfig(ConfigClass):
    """
    Unified configuration for voxelization of molecules and protein-ligand complexes.
    This ensures consistent voxelization parameters across different models.
    """
    # Basic voxelization parameters
    vox_size: float = 0.75
    box_dims: List[float] = field(default_factory=lambda: [24.0, 24.0, 24.0])
    
    # Rotation and translation parameters
    random_rotation: bool = True
    random_translation: float = 6.0
    
    # Channel configuration
    has_protein: bool = True
    
    # Channel names for better interpretability
    ligand_channel_names: List[str] = field(default_factory=lambda: [
        "carbon", "oxygen", "nitrogen", "sulfur", 
        "chlorine", "fluorine", "iodine", "bromine", "other"
    ])
    
    protein_channel_names: List[str] = field(default_factory=lambda: [
        "carbon", "oxygen", "nitrogen", "sulfur"
    ])
    
    # Channel mappings (which elements go into which channel)
    protein_channels: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["C"],
        1: ["O"],
        2: ["N"],
        3: ["S"],
    })

    # Whether the *last* entry of each channel map is a catch-all, i.e. "any element
    # NOT listed here". The ligand map's last entry enumerates the common elements so
    # inverting it yields the exotics (and drops H); the protein map's last entry is a
    # single element, so inverting it would yield "everything except that element".
    # See the note in UnifiedView.get_molecular_complex_channels.
    ligand_last_channel_is_catch_all: bool = True
    protein_last_channel_is_catch_all: bool = False
    
    ligand_channels: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["C"],
        1: ["O"],
        2: ["N"],
        3: ["S"],
        4: ["Cl"],
        5: ["F"],
        6: ["I"],
        7: ["Br"],
        8: ["C", "H", "O", "N", "S", "Cl", "F", "I", "Br"]
    })
    
    # Maximum atom distance from ligand center (for pruning distant atoms)
    max_atom_dist: Optional[float] = 32.0
    
    # How overlapping atoms combine in a voxel. "max" is the historical behaviour and is
    # non-injective -- max(1,1)=1, so two coincident atoms are indistinguishable from one and
    # atom positions are destroyed before the grid is sampled (no grid resolution recovers
    # them). "sum" accumulates, so the field counts overlapping atoms and positions survive.
    voxel_aggregation: str = "max"
    # Scales the effective vdW radius. At 1.0 the kernel is nearly a hard sphere of radius
    # 1.7 A against 1.44 A bonds, so atoms merge. Narrowing it separates them.
    voxel_radius_scale: float = 1.0

    # Data type for tensors
    dtype: torch.dtype = torch.bfloat16
    remove_hydrogens: bool = True


@dataclass
class Poc2MolDataConfig(VoxelizationConfig):
    """
    Configuration specific to the Poc2Mol data pipeline.
    Extends the base VoxelizationConfig with Poc2Mol-specific parameters.
    """
    batch_size: int = 32
    target_samples_per_batch: int = 128
    has_protein: bool = True
    # Indices of channels to use for ligand and protein
    # These are indices into the voxelized output, not the channel mappings above
    ligand_channel_indices: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8, 9, 10, 11, 12])
    protein_channel_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    fnames: Optional[List[str]] = None
    system_ids: Optional[List[str]] = None


@dataclass
class Vox2SmilesDataConfig(VoxelizationConfig):
    """
    Configuration specific to the Vox2Smiles data pipeline.
    Extends the base VoxelizationConfig with Vox2Smiles-specific parameters.
    """
    batch_size: int = 24
    val_batch_size: int = 100
    secondary_val_batch_size: int = 10
    max_smiles_len: int = 200

    # Samples per optimiser step. src/train.py derives accumulate_grad_batches from this
    # and the world size, and OVERWRITES whatever trainer.accumulate_grad_batches was set
    # to. Because this field did not previously exist on the Vox2Smiles config, it fell
    # back to batch_size, i.e. accumulation 1 -- so the `accumulate_grad_batches: 16` in
    # configs/experiment/train_vox2smiles_combined_hiqbind.yaml never took effect. Set it
    # explicitly; it is the knob that keeps the effective batch matched across arms and
    # world sizes.
    target_samples_per_batch: int = 24
    
    # For Vox2Smiles, we typically don't need protein channels
    has_protein: bool = False 
    include_hydrogens: bool = True