"""Batch transform for END-TO-END training, where Poc2Mol is *not* run in the datamodule.

``Poc2MolInferenceBuilder`` runs a frozen Poc2Mol here, under ``torch.no_grad()``, from
``Vox2SmilesDataModule.on_after_batch_transfer`` -- a datamodule hook, outside the autograd
graph the LightningModule builds. That is fine while the upstream is frozen and fatal once
we want the language-modelling loss to reach it: there are three separate severances (the
hook, the ``no_grad``, and ``_bind``'s ``requires_grad_(False)``), and deleting any one of
them alone does nothing.

So this builder does the voxelisation and nothing else. It hands back the two grids
separately

    ``protein_voxels``  (B, P, X, Y, Z) -- Poc2Mol's input. All-zero for ligand-only rows,
                        which carry no protein atoms at all.
    ``ligand_voxels``   (B, L, X, Y, Z) -- the TRUE ligand density. Two jobs: the voxel-level
                        supervision target for pocket rows, and the decoder input for
                        ligand-only (ZINC) rows.
    ``has_pocket``      (B,) bool -- which rows carry a pocket, i.e. which rows Poc2Mol runs
                        on and which ones can produce a gradient for it.

and leaves ``pixel_values`` for ``EndToEndPoc2Smiles`` to assemble, inside ``training_step``,
where the graph is live. See ``src/models/end_to_end.py``.
"""

from __future__ import annotations

import torch

from src.data.common.voxelization.batched import BatchedVoxelizer


class EndToEndVoxelBuilder:
    """Voxelise a mixed batch and hand the protein and ligand grids back separately.

    Bound lazily to the batch's device for the same reason ``LigandVoxelBuilder`` is: under
    DDP each rank has a different one and Lightning only assigns it after the process group
    is up.
    """

    def __init__(
        self,
        voxel_config,
        n_protein_channels: int = 4,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.voxel_config = voxel_config
        self.n_protein_channels = n_protein_channels
        self.compute_dtype = compute_dtype
        self._voxelizer = None
        self._device = None

    def _get(self, device):
        if self._voxelizer is None or self._device != device:
            self._voxelizer = BatchedVoxelizer(
                self.voxel_config,
                compute_dtype=self.compute_dtype,
                cutoff_ratio=self.voxel_config.get("voxel_cutoff_ratio", 2.0),
                aggregation=self.voxel_config.get("voxel_aggregation", "max"),
                radius_scale=self.voxel_config.get("voxel_radius_scale", 1.0),
            ).to(device)
            self._device = device
        return self._voxelizer

    def __call__(self, batch, training: bool = True):
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
        needs = out.pop("needs_poc2mol", None)

        if needs is None:
            needs = torch.zeros(grid.shape[0], dtype=torch.bool, device=grid.device)

        out["protein_voxels"] = grid[:, : self.n_protein_channels]
        out["ligand_voxels"] = grid[:, self.n_protein_channels :]
        # Zero the protein for rows that have none, so a ligand-only row cannot pick up
        # whatever the protein slots happen to contain. They are already empty -- the
        # unified view masks every protein atom out when there are none -- so this is
        # belt-and-braces, and it keeps the guarantee explicit rather than inherited.
        gate = needs.view(-1, *([1] * (out["protein_voxels"].dim() - 1)))
        out["protein_voxels"] = out["protein_voxels"] * gate.to(out["protein_voxels"].dtype)
        out["has_pocket"] = needs
        return out
