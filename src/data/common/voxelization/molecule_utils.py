import os
import pickle
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from docktgrid.transforms import RandomRotation
from docktgrid.molecule import MolecularComplex
import pathlib
from src.data.common.voxelization.voxelizer import RDkitMolecularComplex, UnifiedVoxelGrid
from src.data.common.voxelization.config import VoxelizationConfig


def load_mol_from_pickle(path):
    """Load an RDKit molecule from a pickle file."""
    with open(path, "rb") as f:
        mol_data = pickle.load(f)
    
    # If the pickle contains multiple conformers, use the first one
    if "conformers" in mol_data:
        mol = mol_data["conformers"][0]["rd_mol"]
    else:
        mol = mol_data["rd_mol"]
    
    return mol


def load_complex_from_files(protein_path, ligand_path, parser=None):
    """Load a protein-ligand complex from PDB and MOL2 files."""
    from src.data.docktgrid_mods import MolecularParserWrapper
    
    if parser is None:
        parser = MolecularParserWrapper()
    
    return MolecularComplex(protein_path, ligand_path, molparser=parser)

    

def apply_random_rotation(molecular_complex):
    """Apply a random rotation about the ligand centre, in the coordinates' own dtype.

    docktgrid's RandomRotation builds its matrix in `docktgrid.config.DTYPE`, which our
    site-packages patch sets to bfloat16 (CLAUDE.md §1). Now that coordinates are loaded in
    float32 -- so that parquet_v2's precision actually reaches the grid -- that matmul raises
    "expected m1 and m2 to have the same dtype". Building the matrix here in the coords' dtype
    fixes it AND removes this path's dependence on the docktgrid patch, which does not survive
    a venv rebuild.

    Q from a QR of a Gaussian matrix, sign-corrected, is uniform over O(3); flipping a column
    when the determinant is negative restricts it to SO(3), i.e. rotations without reflections.
    """
    coords = molecular_complex.coords
    n_atoms_ligand = molecular_complex.ligand_data.coords.shape[1]

    gaussian = torch.randn(3, 3, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    rotation_matrix = q.to(coords.dtype)

    centre = molecular_complex.ligand_center.to(coords.dtype).reshape(3, 1)
    molecular_complex.coords = rotation_matrix @ (coords - centre) + centre
    coords = molecular_complex.coords
    molecular_complex.ligand_data.coords = molecular_complex.coords[:, -n_atoms_ligand:]
    molecular_complex.protein_data.coords = molecular_complex.coords[:,:-n_atoms_ligand]
    molecular_complex.ligand_center = torch.mean(
        molecular_complex.ligand_data.coords, 1
        ).to(molecular_complex.ligand_center.dtype)
    return molecular_complex


def apply_random_translation(molecular_complex, max_translation):
    """Apply a random translation to a molecular complex."""
    if max_translation <= 0:
        return molecular_complex
    
    translation_vector_length = np.random.uniform(0, max_translation)
    translation_vector = torch.tensor(
        np.random.uniform(-1, 1, 3) * translation_vector_length,
        dtype=torch.float16
    )
    molecular_complex.ligand_center += translation_vector
    return molecular_complex


def prune_distant_atoms(complex_obj, max_atom_dist, has_protein=True):
    """Remove atoms that are too far from the ligand center."""
    if max_atom_dist is None or max_atom_dist <= 0:
        return complex_obj
    
    ligand_center = complex_obj.ligand_center
    
    # Prune atoms in the entire complex
    dists = torch.linalg.vector_norm(
        complex_obj.coords.T - ligand_center, dim=1
    )
    mask = dists < max_atom_dist
    assert mask.max(), "atoms were pruned to zero"
    complex_obj.coords = complex_obj.coords[:, mask]
    complex_obj.vdw_radii = complex_obj.vdw_radii[mask]
    complex_obj.element_symbols = complex_obj.element_symbols[mask]
    complex_obj.n_atoms = complex_obj.coords.shape[1]

    # Prune ligand atoms
    lig_dists = torch.linalg.vector_norm(
        complex_obj.ligand_data.coords.T - ligand_center, dim=1
    )
    lig_mask = lig_dists < max_atom_dist
    assert lig_mask.max(), "Ligand atoms were pruned to zero"
    complex_obj.ligand_data.coords = complex_obj.ligand_data.coords[:, lig_mask]
    
    # Handle element symbols differently based on type
    if isinstance(complex_obj.ligand_data.element_symbols, (np.ndarray, pd.Series)):
        complex_obj.ligand_data.element_symbols = complex_obj.ligand_data.element_symbols[lig_mask.numpy()]
    else:
        complex_obj.ligand_data.element_symbols = complex_obj.ligand_data.element_symbols[lig_mask]
    
    complex_obj.n_atoms_ligand = complex_obj.ligand_data.coords.shape[1]

    # If there are protein atoms, prune them too
    if has_protein and complex_obj.n_atoms_protein > 0:
        prot_dists = torch.linalg.vector_norm(
            complex_obj.protein_data.coords.T - ligand_center, dim=1
        )
        prot_mask = prot_dists < max_atom_dist
        complex_obj.protein_data.coords = complex_obj.protein_data.coords[:, prot_mask]
        complex_obj.protein_data.element_symbols = complex_obj.protein_data.element_symbols[prot_mask]
        complex_obj.n_atoms_protein = complex_obj.protein_data.coords.shape[1]
        assert complex_obj.n_atoms_protein > 0, "Protein atoms were pruned to zero"
    assert complex_obj.n_atoms_ligand > 0, "Ligand atoms were pruned to zero"
    return complex_obj


def prepare_rdkit_molecule(mol, config):
    """Prepare an RDKit molecule for voxelization."""
    # Convert to our molecular complex format
    molecular_complex = RDkitMolecularComplex(mol)
    
    # Apply transformations
    if config.random_rotation:
        molecular_complex = apply_random_rotation(molecular_complex)
    
    if config.random_translation > 0:
        molecular_complex = apply_random_translation(molecular_complex, config.random_translation)
    
    if config.max_atom_dist is not None and config.max_atom_dist > 0:
        molecular_complex = prune_distant_atoms(
            molecular_complex, 
            config.max_atom_dist,
            config.has_protein
        )
    
    return molecular_complex


def prepare_protein_ligand_complex(protein, ligand, config):
    """Prepare a protein-ligand complex for voxelization."""
    # Load the complex
    if isinstance(protein, (str, pathlib.Path)):
        complex_obj = load_complex_from_files(protein, ligand)
    else:
        complex_obj = MolecularComplex(protein, ligand)
    # Apply transformations
    if config.random_rotation:
        complex_obj = apply_random_rotation(complex_obj)
    
    if config.random_translation > 0:
        complex_obj = apply_random_translation(complex_obj, config.random_translation)
    
    if config.max_atom_dist is not None and config.max_atom_dist > 0:
        complex_obj = prune_distant_atoms(complex_obj, config.max_atom_dist)
    
    return complex_obj


def _voxelize_one(complex_obj, config):
    """Voxelise a single prepared complex through the batched voxeliser.

    These two entry points are what `inference/` and `evaluations/` call, so routing them
    here is what keeps evaluation numerically consistent with training. The legacy
    `UnifiedVoxelGrid` path voxelised in absolute PDB coordinates in bfloat16, which put
    3.67% of occupied voxels more than 0.05 out (see CLAUDE.md §3b) -- evaluating a model
    on data the model was never trained on.

    Imported lazily: `batched` imports `voxelizer`, which imports this module.
    """
    from src.data.common.voxelization.batched import atom_record_from_complex, voxelize_records
    from src.data.common.voxelization.voxelizer import UnifiedView

    view = UnifiedView(config)
    record = atom_record_from_complex(
        complex_obj,
        view,
        box_dims=config.box_dims,
        cutoff_ratio=config.get("voxel_cutoff_ratio", 2.0),
    )
    return voxelize_records([record], config)[0]


def voxelize_molecule(mol, config):
    """Voxelize an RDKit molecule using the unified voxelizer."""
    molecular_complex = prepare_rdkit_molecule(mol, config)
    return _voxelize_one(molecular_complex, config)


def voxelize_complex(protein, ligand, config):
    """Voxelize a protein-ligand complex using the unified voxelizer."""
    complex_obj = prepare_protein_ligand_complex(protein, ligand, config)
    voxel = _voxelize_one(complex_obj, config)

    # Extract protein and ligand channels based on config
    if config.has_protein:
        protein_channels = len(config.protein_channels)
        protein_voxel = voxel[:protein_channels]
        ligand_voxel = voxel[protein_channels:]
    else:
        protein_voxel = None
        ligand_voxel = voxel

    return protein_voxel, ligand_voxel, complex_obj 