"""Assemble the SMILES decoder's input, optionally carrying protein context.

Experiment 1: give Vox2Smiles the pocket's protein density alongside the ligand density it
already sees, instead of only Poc2Mol's predicted ligand. The decoder then has direct
access to the pocket rather than only what survived the Poc2Mol bottleneck.

Layout of the assembled tensor, in channel order:

    [0 : L]        ligand density -- ground truth (ZINC) or Poc2Mol's prediction (HiQBind)
    [L : L+P]      protein density, zeroed when absent or masked
    [L+P]          a constant plane: 1 where the protein channels are meaningful, else 0

The flag plane matters more than it looks. Empty protein channels are not
self-identifying: a genuinely sparse pocket and a masked one both read as near-zero, and
the model would have to infer which from the density statistics. Because the first thing
the ViT does is a Conv3d over the patch, a constant input plane contributes a constant
vector to every patch embedding -- so this is exactly equivalent to a learned
"protein absent" embedding, at the cost of one channel and no special-casing.

Masking during combined training is what keeps ligand-only pretraining transferable and
stops the decoder becoming dependent on a signal that is unavailable in the ligand-only
regime. It also buys a free diagnostic: evaluate the trained model with the protein masked
and see how much of the gain disappears.
"""

from __future__ import annotations

from typing import Optional

import torch


def assemble_decoder_input(
    ligand: torch.Tensor,
    protein: Optional[torch.Tensor],
    has_protein: Optional[torch.Tensor],
    inject_protein: bool,
    mask_probability: float = 0.0,
    training: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build the decoder's ``pixel_values``.

    Args:
        ligand: ``(B, L, X, Y, Z)`` ligand density.
        protein: ``(B, P, X, Y, Z)`` protein density, or None.
        has_protein: ``(B,)`` bool, which samples actually carry protein density. ZINC
            samples are False, so they are masked regardless of ``mask_probability``.
        inject_protein: when False, returns ``ligand`` untouched -- this is the baseline
            arm, and it must stay bit-identical to not-having-the-feature.
        mask_probability: chance of hiding the protein from a sample that has it.
        training: masking is applied during training only. Validation always shows the
            protein so the metric measures the deployed condition; the masked condition is
            measured deliberately, as a separate evaluation.
        generator: optional RNG, for reproducible masking.

    Returns:
        ``(B, L, ...)`` if not injecting, else ``(B, L + P + 1, ...)``.
    """
    if not inject_protein:
        return ligand

    if protein is None:
        raise ValueError(
            "inject_protein=True but no protein density was supplied. The voxel config "
            "needs has_protein=True so the protein channel slots are produced."
        )

    batch_size = ligand.shape[0]
    device = ligand.device

    if has_protein is None:
        keep = torch.ones(batch_size, dtype=torch.bool, device=device)
    else:
        keep = has_protein.to(device=device, dtype=torch.bool)

    if training and mask_probability > 0:
        draw = torch.rand(batch_size, device=device, generator=generator)
        keep = keep & (draw >= mask_probability)

    gate = keep.view(-1, *([1] * (ligand.dim() - 1)))
    masked_protein = protein * gate.to(protein.dtype)
    flag = gate.to(ligand.dtype).expand(-1, 1, *ligand.shape[2:])

    return torch.cat([ligand, masked_protein, flag], dim=1)


def decoder_input_channels(n_ligand_channels: int, n_protein_channels: int, inject_protein: bool) -> int:
    """Channel count the ViT must be configured with. Keep configs in sync with this."""
    if not inject_protein:
        return n_ligand_channels
    return n_ligand_channels + n_protein_channels + 1
