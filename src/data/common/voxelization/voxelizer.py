import torch
import numpy as np
from typing import List, Dict, Optional, Union, Any

from docktgrid.grid import Grid3D
from docktgrid.view import View
from docktgrid.molecule import MolecularComplex
from docktgrid.config import DEVICE, DTYPE
from docktgrid.periodictable import ptable

from src.data.common.voxelization.config import VoxelizationConfig


# Channel tokens that consult per-atom features rather than element symbols alone.
_FEATURE_TOKENS = {
    "C_aromatic", "C_aliphatic", "N_withH", "N_noH", "O_withH", "O_noH", "HBA",
}


class UnifiedView(View):
    """
    A unified view class that can handle both protein-ligand complexes and standalone ligands.
    """
    def __init__(
        self,
        config: VoxelizationConfig,
    ):
        super().__init__()
        self.channels = config.ligand_channels
        self.ch_names = config.ligand_channel_names
        self.has_protein = config.has_protein
        self.protein_channels = config.protein_channels
        self.protein_ch_names = config.protein_channel_names if config.has_protein else []
        self.ligand_catch_all = config.get('ligand_last_channel_is_catch_all', True)
        self.protein_catch_all = config.get('protein_last_channel_is_catch_all', False)

    def get_num_channels(self):
        """Get the total number of channels."""
        protein_channels = len(self.protein_channels) if self.has_protein else 0
        ligand_channels = len(self.channels)
        return protein_channels + ligand_channels

    def get_channels_names(self):
        """Get the names of all channels."""
        protein_names = [f"{ch}_protein" for ch in self.protein_ch_names] if self.has_protein else []
        ligand_names = [f"{ch}_ligand" for ch in self.ch_names]
        return protein_names + ligand_names

    def get_molecular_complex_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for all atoms."""
        # Get element symbols from the molecular complex.
        #
        # Normalise case before matching. The two data sources disagree: HiQBind parquet
        # carries PDB-style uppercase ("CL", "BR", "SE") while RDKit hands back title case
        # ("Cl", "Br", "Se"). Matching raw symbols against the title-case channel maps meant
        # every HiQBind chlorine and bromine fell through to the ligand catch-all, so the
        # chlorine and bromine channels were permanently empty for HiQBind while ZINC
        # populated them normally -- a train/serve split straight through the middle of the
        # combined model. docktgrid already normalises this way for its vdW lookup
        # (``ptable[symbol.title()]``), so title case is the established convention here.
        symbs = np.char.title(np.asarray(molecular_complex.element_symbols).astype(str))

        parent_cache = {}

        def hydrogen_parents():
            """Element of the heavy atom each hydrogen is bonded to.

            Bond lengths separate cleanly (H-C 1.09, H-N 1.01, H-O 0.97 A), so nearest
            heavy neighbour identifies the parent reliably. Used to split hydrogens into
            the rotatable polar ones and the geometrically pinned nonpolar ones -- see the
            leakage diagnostic in §3f of CLAUDE.md.
            """
            if "v" not in parent_cache:
                from scipy.spatial import cKDTree

                is_h = symbs == "H"
                heavy = ~is_h
                out = np.full(len(symbs), "", dtype=object)
                if is_h.any() and heavy.any():
                    coords = molecular_complex.coords
                    coords = coords.float().cpu().numpy() if hasattr(coords, "float") else np.asarray(coords)
                    tree = cKDTree(coords[:, heavy].T)
                    _, nearest = tree.query(coords[:, is_h].T, k=1)
                    out[is_h] = symbs[heavy][nearest]
                parent_cache["v"] = out
            return parent_cache["v"]

        def match(elements):
            """Which atoms belong in a channel whose element list is `elements`.

            Beyond plain element symbols, three tokens are understood:
              "*"           every atom -- a generic total-density channel, without having
                            to enumerate the periodic table.
              "H_polar"     hydrogens bonded to N/O/S. These are the rotatable ones, and
                            the ones HiQBind's minimiser relaxes toward ligand acceptors.
              "H_nonpolar"  hydrogens bonded to C. Pinned by their frozen parent heavy
                            atom, so they carry shape information without the ligand
                            conditioning -- the control arm for the §3f diagnostic.
            """
            if "*" in elements:
                return np.ones(len(symbs), dtype=bool)
            if "H_polar" in elements:
                return (symbs == "H") & np.isin(hydrogen_parents(), ["N", "O", "S"])
            if "H_nonpolar" in elements:
                return (symbs == "H") & (hydrogen_parents() == "C")

            # Feature-derived tokens (the 11-channel scheme). These need per-atom properties
            # that element symbols cannot supply -- aromaticity, H count, acceptor status --
            # which is why parquet_v2 stores them. They are the two things the pocket most
            # directly constrains: hybridisation (planar aromatic slots vs 3D aliphatic
            # pockets) and donor/acceptor identity.
            token = next((e for e in elements if e in _FEATURE_TOKENS), None)
            if token is not None:
                features = getattr(molecular_complex, "atom_features", None)
                if not features:
                    raise ValueError(
                        f"Channel token {token!r} needs per-atom features, but this "
                        "MolecularComplex carries none. Feature-based channels require the "
                        "regenerated parquet (parquet_v2 / zinc20_parquet_v2), which stores "
                        "is_aromatic / n_hydrogens / is_acceptor per atom."
                    )
                # Features are stored LIGAND-ONLY and right-aligned here. The complex is
                # protein-then-ligand, and `prune_distant_atoms` drops distant PROTEIN atoms
                # between construction and channel assignment -- so any protein-side offset
                # computed at attach time goes stale ("operands could not be broadcast
                # together"). Ligand atoms are always the trailing block and are never pruned,
                # so aligning from the end is correct however much protein was removed.
                def right_aligned(values, dtype):
                    values = np.asarray(values)
                    out = np.zeros(len(symbs), dtype=dtype)
                    if len(values):
                        out[len(symbs) - len(values):] = values
                    return out

                aromatic = right_aligned(features["is_aromatic"], bool)
                n_hydrogens = right_aligned(features["n_hydrogens"], np.int16)
                acceptor = right_aligned(features["is_acceptor"], bool)
                if token == "C_aromatic":
                    return (symbs == "C") & aromatic
                if token == "C_aliphatic":
                    return (symbs == "C") & ~aromatic
                if token == "N_withH":
                    return (symbs == "N") & (n_hydrogens > 0)
                if token == "N_noH":
                    return (symbs == "N") & (n_hydrogens == 0)
                if token == "O_withH":
                    return (symbs == "O") & (n_hydrogens > 0)
                if token == "O_noH":
                    return (symbs == "O") & (n_hydrogens == 0)
                if token == "HBA":
                    # Not derivable from element + H count: the acceptor definition excludes
                    # amide N (lone pair delocalised into the carbonyl), ~0.5 per ligand and
                    # 29% of all N-without-H. That exclusion is why it is stored.
                    return acceptor

            return np.isin(symbs, elements)

        # Initialize channels for ligand
        ligand_chs = np.asarray([match(self.channels[c]) for c in range(len(self.channels))])

        # The last ligand channel lists every common element, so inverting it leaves the
        # exotic elements only -- and, because H is in that list, drops hydrogens.
        if self.ligand_catch_all:
            np.invert(ligand_chs[-1], out=ligand_chs[-1])

        # If we have protein channels, process them too
        if self.has_protein:
            protein_chs = np.asarray([match(self.protein_channels[c]) for c in range(len(self.protein_channels))])
            # The protein map's last entry is a single element (S), NOT a list of common
            # elements, so inverting it would produce an "everything except S" channel that
            # duplicates C/O/N, admits hydrogens, and leaves protein S unrepresented.
            # That was the behaviour up to 2026-08; it is off by default now.
            if self.protein_catch_all:
                np.invert(protein_chs[-1], out=protein_chs[-1])
            return torch.from_numpy(np.vstack([protein_chs, ligand_chs]))
        
        return torch.from_numpy(ligand_chs)

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for ligand atoms."""
        chs = self.get_molecular_complex_channels(molecular_complex)
        
        # If we have protein channels, they come first, so we need to slice accordingly
        if self.has_protein:
            ligand_chs = chs[len(self.protein_channels):]
        else:
            ligand_chs = chs
            
        # Exclude protein atoms from ligand channels
        ligand_chs[..., : -molecular_complex.n_atoms_ligand] = False
        return ligand_chs

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        if not self.has_protein:
            return torch.tensor([], dtype=torch.bool)
            
        chs = self.get_molecular_complex_channels(molecular_complex)
        protein_chs = chs[:len(self.protein_channels)]
        
        # Exclude ligand atoms from protein channels
        protein_chs[..., -molecular_complex.n_atoms_ligand:] = False
        return protein_chs

    def __call__(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Concatenate all channels in a single tensor."""
        if self.has_protein:
            protein = self.get_protein_channels(molecular_complex)
        else:
            protein = None
            
        ligand = self.get_ligand_channels(molecular_complex)
        
        return torch.cat(
            (
                protein if protein is not None else torch.tensor([], dtype=torch.bool),
                ligand if ligand is not None else torch.tensor([], dtype=torch.bool),
            ),
        )


class UnifiedVoxelGrid:
    """
    A unified voxel grid class that can handle both protein-ligand complexes and standalone ligands.
    This is based on the existing VoxelGrid and RDkitVoxelGrid classes but with a unified interface.
    """
    def __init__(
        self,
        config: VoxelizationConfig,
    ):
        """Initialize the unified voxel grid."""
        self.config = config
        self.occupancy_func = self._voxelize_vdw  # Currently only supporting vdw occupancy
        self.grid = Grid3D(config.vox_size, config.box_dims)
        self.view = UnifiedView(config)

    @property
    def num_channels(self):
        """Get total number of channels for the chosen view configuration."""
        return self.view.get_num_channels()

    @property
    def shape(self):
        """Get voxel grid shape with channels first (n_channels, dim1, dim2, dim3)."""
        n_channels = self.num_channels
        dim1, dim2, dim3 = self.grid.axes_dims
        return (n_channels, dim1, dim2, dim3)

    def get_channels_mask(self, molecule):
        """Build channels mask for each atom."""
        return self.view(molecule)

    def voxelize(self, molecule, out=None, channels=None, requires_grad=False):
        """Voxelize molecule and return voxel grid."""
        if out is None:
            out = torch.zeros(
                self.shape, dtype=self.config.dtype, device=DEVICE, requires_grad=requires_grad
            )
        else:
            if out.shape != self.shape:
                raise ValueError(
                    f"`out` shape must be == {self.shape}, currently it is {out.shape}"
                )
            out = torch.as_tensor(out, self.config.dtype, DEVICE, requires_grad=requires_grad)

        if channels is None:
            channels = self.get_channels_mask(molecule)
        else:
            cshape = (self.num_channels, molecule.n_atoms)
            if channels.shape != cshape:
                raise ValueError(
                    f"`channels` shape must be == {cshape}, currently it is {channels.shape}"
                )
            channels = torch.as_tensor(channels, dtype=self.config.dtype, device=DEVICE)

        # Create voxel based on occupancy option
        self._voxelize_vdw(molecule, out, channels)

        return out.view(self.shape)

    @torch.no_grad()
    def _voxelize_vdw(self, molecule, out, channels) -> None:
        """Voxelize using van der Waals radii."""
        points = self.grid.points
        center = molecule.ligand_center
        # Translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # Reshape to n_channels, n_points
        out = out.view(channels.shape[0], grid[x].shape[0])

        self._calc_vdw_occupancies(
            out,
            channels,
            molecule.coords[x].to(DEVICE),
            molecule.coords[y].to(DEVICE),
            molecule.coords[z].to(DEVICE),
            grid[x].to(DEVICE),
            grid[y].to(DEVICE),
            grid[z].to(DEVICE),
            molecule.vdw_radii.to(DEVICE),
        )

    @staticmethod
    @torch.jit.script
    def _calc_vdw_occupancies(
        out: torch.Tensor,  # output tensor, shape (n_channels, n_points)
        channels: torch.Tensor,  # bool mask of channels, shape (n_channels, n_atoms)
        ax: torch.Tensor,  # x coords of atoms, shape (n_atoms,)
        ay: torch.Tensor,  # y coords of atoms, shape (n_atoms,)
        az: torch.Tensor,  # z coords of atoms, shape (n_atoms,)
        px: torch.Tensor,  # x coords of grid points, shape (n_points, 1)
        py: torch.Tensor,  # y coords of grid points, shape (n_points, 1)
        pz: torch.Tensor,  # z coords of grid points, shape (n_points, 1)
        vdws: torch.Tensor,  # vdw radii of atoms, shape (n_atoms,)
    ):
        """Calculate voxel occupancies using van der Waals radii."""
        dist = torch.sqrt(
            torch.pow(ax - px, 2) + torch.pow(ay - py, 2) + torch.pow(az - pz, 2)
        )
        occs = 1 - torch.exp(-1 * torch.pow(vdws / dist, 12))  # voxel occupancies
        
        # Convert occs to the same dtype as out to avoid dtype mismatch
        occs = occs.to(dtype=out.dtype)

        for i, mask in enumerate(channels):
            if torch.any(mask):
                torch.amax(occs[:, mask], dim=1, out=out[i])


class RDkitMolecularComplex(MolecularComplex):
    """
    A wrapper for RDKit molecules to make them compatible with the MolecularComplex interface.
    This allows us to use the same voxelization code for both protein-ligand complexes and standalone ligands.
    """
    def __init__(self, mol):
        """Initialize from an RDKit molecule."""
        self.mol = mol
        self.coords = self._get_coords()
        self.element_symbols = self._get_element_symbols()
        self.ligand_center = self._get_ligand_center()
        self.n_atoms = self.mol.GetNumAtoms()
        self.n_atoms_ligand = self.n_atoms
        self.n_atoms_protein = 0
        self.vdw_radii = self._get_vdw_radii()
        
        # These are placeholders to maintain compatibility with MolecularComplex
        class DummyData:
            def __init__(self, parent):
                self.coords = parent.coords
                self.element_symbols = parent.element_symbols
                self.vdw_radii = parent.vdw_radii
        
        self.ligand_data = DummyData(self)
        self.protein_data = DummyData(self)

    def _get_coords(self):
        """Get atom coordinates from the RDKit molecule."""
        conf = self.mol.GetConformer()
        coords = np.array(conf.GetPositions(), dtype=np.float32).T
        return torch.tensor(coords, dtype=DTYPE)

    def _get_element_symbols(self):
        """Get element symbols from the RDKit molecule."""
        symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        return np.array(symbols)

    def _get_ligand_center(self):
        """Calculate the center of the molecule."""
        return torch.mean(self.coords, 1).to(dtype=DTYPE)

    def _get_vdw_radii(self):
        """Get van der Waals radii for each atom."""
        return torch.tensor(
            [ptable[a.title()]["vdw"] for a in self.element_symbols],
            dtype=DTYPE,
        )


class StoredLigandComplex(MolecularComplex):
    """Ligand-only complex built from parquet_v2's stored arrays -- no RDKit involved.

    Same interface as :class:`RDkitMolecularComplex`, with three differences that are the
    whole point of it:

    * **Coordinates stay float32.** ``RDkitMolecularComplex`` casts to docktgrid's ``DTYPE``,
      which our site-packages patch sets to bfloat16 (CLAUDE.md §1) -- precisely the
      quantisation parquet_v2 was regenerated to remove (§15c, up to 0.49 A on a 0.75 A
      voxel). Casting here would discard that precision again before it ever reached the
      grid, silently undoing the regeneration.
    * **Per-atom features are carried**, so the 11-channel scheme (`C_aromatic`, `N_withH`,
      `HBA`, ...) resolves on ZINC exactly as it does on HiQBind.
    * **No RDKit parse.** ``Chem.MolFromMolBlock`` per sample is the stage-1 bottleneck
      (§3h, ~790 samples/s, data-bound); v2 drops ``mol_block`` and this path with it.

    ``n_atoms_protein = 0`` mirrors ``RDkitMolecularComplex``, so with ``has_protein=True``
    the protein channel slots are still emitted but masked empty -- which is what lets a
    ligand-only ZINC record share a batch with a HiQBind complex.
    """

    def __init__(self, coords, element_symbols, features=None):
        self.mol = None
        coords = coords if torch.is_tensor(coords) else torch.as_tensor(coords)
        self.coords = coords.to(torch.float32)
        self.element_symbols = np.asarray(element_symbols).astype(str)
        self.ligand_center = torch.mean(self.coords, 1)
        self.n_atoms = self.coords.shape[1]
        self.n_atoms_ligand = self.n_atoms
        self.n_atoms_protein = 0
        self.vdw_radii = torch.tensor(
            [ptable[a.title()]["vdw"] for a in self.element_symbols],
            dtype=torch.float32,
        )
        if features:
            self.atom_features = features

        class DummyData:
            def __init__(self, parent):
                self.coords = parent.coords
                self.element_symbols = parent.element_symbols
                self.vdw_radii = parent.vdw_radii

        self.ligand_data = DummyData(self)
        self.protein_data = DummyData(self) 