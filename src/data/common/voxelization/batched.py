"""Batched vdW voxelisation, run on the training process's own device.

Why this exists
---------------
The original pipeline voxelised inside ``Dataset.__getitem__`` via
:class:`~src.data.common.voxelization.voxelizer.UnifiedVoxelGrid`, which allocates on
``docktgrid.config.DEVICE``. That constant resolves to a bare ``torch.device("cuda")``,
i.e. *the current device*, so every dataloader worker voxelised on ``cuda:0`` regardless
of rank. Two consequences:

* ``num_workers`` had to stay at 0 (CUDA tensors cannot cross a fork), which capped the
  pipeline at ~25 samples/s against a model that trains at ~870.
* Under DDP every rank would have piled onto ``cuda:0``.

The fix is to split the work: dataloader workers do CPU-only preparation (parquet ->
coordinates, vdW radii, per-channel atom masks) and the voxel grid itself is built here,
batched, on the rank's own device. CPU voxelisation is not an option -- measured at
2249 ms/sample against 3.6 ms on an H100.

The occupancy function is unchanged from the reference implementation::

    occ(atom, point) = 1 - exp(-(vdw_radius / distance) ** 12)
    channel value    = max over the atoms assigned to that channel

Two representation choices make this fast enough not to bottleneck training:

*Ragged, not padded.* Atom counts vary by more than 2x across a batch, so padding to the
batch maximum wastes most of the work. Atoms are flattened into one list tagged with the
``(sample, channel)`` slot they write to.

*Local neighbourhoods.* The occupancy of an atom decays as ``d**-12``; beyond
``cutoff_ratio`` times its vdW radius the contribution is far below the resolution of the
stored dtype. Each atom therefore touches a small cube of voxels rather than all 32768.

``tests/test_batched_voxelizer.py`` checks both against a float64 reference.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from src.data.common.voxelization.config import VoxelizationConfig, resolve_dtype
from src.data.common.voxelization.voxelizer import UnifiedView


def atom_record_from_complex(
    complex_obj,
    view: UnifiedView,
    box_dims: Optional[Sequence[float]] = None,
    cutoff_ratio: float = 2.0,
) -> dict:
    """Extract the CPU-side atom record a dataloader worker should hand to the collate fn.

    Everything here is cheap array work; no voxel grid is built and no CUDA is touched,
    which is what allows ``num_workers > 0``.

    Coordinates are **re-expressed relative to the grid centre** here rather than in the
    voxeliser. The reference implementation built grid points as ``points + center`` and
    then took ``atom - grid_point``, all in bfloat16, using absolute PDB coordinates that
    run to 127 A. bfloat16 spacing at 127 A is 1.0 A, i.e. coarser than the 0.75 A voxel,
    so distant pockets were voxelised on a visibly quantised grid. Since
    ``atom - (points + center) == (atom - center) - points`` exactly, subtracting the
    centre in float32 up front removes the problem at no cost.

    Two classes of atom are dropped, both provably unable to affect the output:

    * Atoms belonging to no channel. After the protein channel map was corrected to
      C/O/N/S these are almost all hydrogens -- roughly half of every complex.
    * Atoms that cannot reach the box. Callers prune to ``max_atom_dist`` (32 A) but the
      box is only 24 A across, so an atom further than ``half_extent + cutoff`` from the
      centre along any axis lies outside every voxel's neighbourhood. This is an exact
      filter given the same ``cutoff_ratio`` the voxeliser uses, not an approximation.
    """
    channels = view(complex_obj).cpu()
    keep = channels.any(dim=0)

    center = complex_obj.ligand_center.to(torch.float32).cpu()
    coords = complex_obj.coords.to(torch.float32).cpu() - center.unsqueeze(1)
    radii = complex_obj.vdw_radii.to(torch.float32).cpu()

    if box_dims is not None:
        half_extent = torch.tensor(
            [float(d) / 2.0 for d in box_dims], dtype=torch.float32
        ).unsqueeze(1)
        reach = half_extent + cutoff_ratio * radii.unsqueeze(0)
        keep &= (coords.abs() <= reach).all(dim=0)

    return {
        "coords": coords[:, keep],
        "vdw_radii": radii[keep],
        "channels": channels[:, keep],
        "center": center,
    }


def collate_voxel_inputs(items: Sequence[dict], key_prefix: str = "") -> dict:
    """Flatten per-sample atom records into one ragged batch.

    Emits one entry per ``(atom, channel)`` membership. With the corrected channel maps
    every atom belongs to at most one channel, so this is normally just the atom list, but
    the representation stays correct for overlapping channel definitions.
    """
    coords, radii, targets = [], [], []
    n_channels = items[0][f"{key_prefix}channels"].shape[0]

    for sample_idx, item in enumerate(items):
        channel_idx, atom_idx = item[f"{key_prefix}channels"].nonzero(as_tuple=True)
        coords.append(item[f"{key_prefix}coords"][:, atom_idx].T)
        radii.append(item[f"{key_prefix}vdw_radii"][atom_idx])
        targets.append(sample_idx * n_channels + channel_idx)

    return {
        f"{key_prefix}atom_xyz": torch.cat(coords, dim=0),
        f"{key_prefix}atom_radius": torch.cat(radii, dim=0),
        f"{key_prefix}atom_slot": torch.cat(targets, dim=0),
        f"{key_prefix}center": torch.stack([it[f"{key_prefix}center"] for it in items]),
        f"{key_prefix}n_channels": n_channels,
        f"{key_prefix}batch_size": len(items),
    }


def voxelize_records(records, config, device=None, compute_dtype: torch.dtype = torch.float32):
    """One-shot ``records -> (B, C, X, Y, Z)``, for inference and evaluation.

    Training goes through the datamodule, which keeps a voxeliser bound to the rank's
    device across batches. This is the convenience path for code that voxelises once and
    does not care about the setup cost.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = collate_voxel_inputs(records)
    voxelizer = BatchedVoxelizer(
        config,
        compute_dtype=compute_dtype,
        cutoff_ratio=config.get("voxel_cutoff_ratio", 2.0),
    ).to(device)
    return voxelizer(
        batch["atom_xyz"],
        batch["atom_radius"],
        batch["atom_slot"],
        batch["batch_size"],
        batch["n_channels"],
    )


class BatchedVoxelizer:
    """Build voxel grids for a whole batch on one device.

    Args:
        config: supplies ``vox_size``, ``box_dims`` and the output ``dtype``.
        compute_dtype: precision for the distance/occupancy arithmetic. The reference
            implementation used bfloat16 throughout, which quantises a 30 A coordinate to
            roughly 0.12 A -- an appreciable fraction of the 0.75 A voxel. Output is cast
            to ``config.dtype`` regardless, so this only controls intermediates.
        cutoff_ratio: an atom is evaluated out to ``cutoff_ratio * vdw_radius``. At 2.0 the
            largest neglected occupancy is ``1 - exp(-2**-12)`` = 2.4e-4, more than an
            order of magnitude below the 3.9e-3 resolution of a bfloat16 output.
        max_chunk_elements: cap on ``atoms x neighbourhood`` per chunk.
    """

    def __init__(
        self,
        config: VoxelizationConfig,
        compute_dtype: torch.dtype = torch.float32,
        cutoff_ratio: float = 2.0,
        max_chunk_elements: int = 1 << 26,
        aggregation: str = "max",
        radius_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.compute_dtype = compute_dtype
        self.cutoff_ratio = cutoff_ratio
        # How overlapping atoms combine.
        #   "max" -- the historical behaviour. Idempotent under overlap: where two atoms
        #            both saturate a voxel, max(1,1) = 1, indistinguishable from one atom.
        #            Bonded heavy atoms sit 1.9 voxels apart with 2.3-voxel vdW radii, so
        #            their spheres merge and ~34% of occupied voxels saturate. Measured
        #            separability (connected blobs / atoms) is 0.07 -- the carbon skeleton
        #            collapses into a single object, and atom positions are destroyed
        #            BEFORE sampling, so finer voxels cannot recover them.
        #   "sum" -- overlapping density accumulates, so a voxel covered by n atoms reads
        #            ~n and atom count/position survive. Values are no longer bounded by 1.
        if aggregation not in {"max", "sum"}:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.aggregation = aggregation
        # Shrinks the effective vdW radius. At 1.0 the kernel is essentially a hard sphere
        # of radius 1.7 A against 1.44 A bonds, so neighbouring atoms merge regardless of
        # aggregation or grid resolution. Narrowing it is what makes individual atoms
        # separable; summing is what makes the overlap countable. Both are needed.
        self.radius_scale = radius_scale
        self.max_chunk_elements = max_chunk_elements
        self.out_dtype = resolve_dtype(config.dtype)

        self.vox_size = float(config.vox_size)
        self.box_dims = [float(d) for d in config.box_dims]
        self.axes_dims = tuple(int(d / self.vox_size) for d in self.box_dims)
        self.half_extent = torch.tensor([d / 2.0 for d in self.box_dims], dtype=torch.float64)

        self.n_points = int(self.axes_dims[0] * self.axes_dims[1] * self.axes_dims[2])
        self._dims = torch.tensor(self.axes_dims, dtype=torch.long)
        # strides for x-major, z-minor flattening -- must match Grid3D's point ordering
        self._strides = torch.tensor(
            [self.axes_dims[1] * self.axes_dims[2], self.axes_dims[2], 1], dtype=torch.long
        )
        self._offsets_cache: dict = {}
        self.device = torch.device("cpu")
        if device is not None:
            self.to(device)

    def to(self, device) -> "BatchedVoxelizer":
        self.device = torch.device(device)
        self.half_extent = self.half_extent.to(self.device)
        self._dims = self._dims.to(self.device)
        self._strides = self._strides.to(self.device)
        self._offsets_cache = {}
        return self

    def _offsets(self, radius_voxels: int) -> torch.Tensor:
        """Integer voxel offsets covering a cube of half-width ``radius_voxels``."""
        if radius_voxels not in self._offsets_cache:
            span = torch.arange(
                -radius_voxels, radius_voxels + 1, dtype=torch.long, device=self.device
            )
            gx, gy, gz = torch.meshgrid(span, span, span, indexing="ij")
            self._offsets_cache[radius_voxels] = torch.stack(
                [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1
            )
        return self._offsets_cache[radius_voxels]

    @torch.no_grad()
    def __call__(
        self,
        atom_xyz: torch.Tensor,
        atom_radius: torch.Tensor,
        atom_slot: torch.Tensor,
        batch_size: int,
        n_channels: int,
    ) -> torch.Tensor:
        """Voxelise one ragged batch.

        Args:
            atom_xyz: ``(N, 3)`` coordinates relative to each sample's grid centre.
            atom_radius: ``(N,)`` van der Waals radii.
            atom_slot: ``(N,)`` destination ``sample_idx * n_channels + channel_idx``.
            batch_size: number of samples in the batch.
            n_channels: channels per sample.

        Returns:
            ``(B, C, X, Y, Z)`` occupancies in ``config.dtype``.
        """
        device = self.device
        atom_xyz = atom_xyz.to(device, non_blocking=True).to(self.compute_dtype)
        atom_radius = atom_radius.to(device, non_blocking=True).to(self.compute_dtype)
        atom_slot = atom_slot.to(device, non_blocking=True)

        n_slots = batch_size * n_channels
        # One extra trailing element absorbs writes from neighbourhood points that fall
        # outside the box; it is discarded below. Without it, out-of-range entries would
        # have to be clamped onto a real voxel and would corrupt it.
        flat = torch.zeros(n_slots * self.n_points + 1, dtype=self.compute_dtype, device=device)

        if atom_xyz.numel():
            radius_voxels = int(
                torch.ceil(atom_radius.max().float() * self.radius_scale * self.cutoff_ratio / self.vox_size).item()
            )
            offsets = self._offsets(radius_voxels)
            per_atom = offsets.shape[0]
            chunk = max(1, self.max_chunk_elements // per_atom)
            for start in range(0, atom_xyz.shape[0], chunk):
                stop = min(start + chunk, atom_xyz.shape[0])
                self._accumulate(
                    atom_xyz[start:stop],
                    atom_radius[start:stop],
                    atom_slot[start:stop],
                    offsets,
                    flat,
                )

        return (
            flat[:-1]
            .view(batch_size, n_channels, *self.axes_dims)
            .to(self.out_dtype)
        )

    def _accumulate(self, xyz, radius, slot, offsets, flat) -> None:
        half = self.half_extent.to(self.compute_dtype)

        # fractional voxel coordinate of each atom, then the integer cube around it
        voxel_pos = (xyz + half) / self.vox_size                     # (n, 3)
        base = torch.floor(voxel_pos).long()                         # (n, 3)
        index = base.unsqueeze(1) + offsets.unsqueeze(0)             # (n, M, 3)

        inside = ((index >= 0) & (index < self._dims)).all(dim=-1)   # (n, M)

        point = index.to(self.compute_dtype) * self.vox_size - half  # (n, M, 3)
        delta = point - xyz.unsqueeze(1)
        sq_dist = delta.mul_(delta).sum(dim=-1)                      # (n, M)

        occupancy = sq_dist
        occupancy.sqrt_()
        occupancy.reciprocal_()
        occupancy.mul_(radius.unsqueeze(1) * self.radius_scale)
        occupancy.pow_(12)
        occupancy.neg_().exp_()
        occupancy.neg_().add_(1.0)
        # An atom sitting exactly on a grid point gives distance 0 -> inf -> occupancy 1,
        # which is the intended limit of the occupancy function.

        destination = slot.unsqueeze(1) * self.n_points + (index * self._strides).sum(dim=-1)
        destination = torch.where(inside, destination, flat.shape[0] - 1)

        # Chunking is safe for both: max of maxes == max, sum of sums == sum.
        flat.scatter_reduce_(
            0, destination.reshape(-1), occupancy.reshape(-1),
            reduce="amax" if self.aggregation == "max" else "sum", include_self=True,
        )
