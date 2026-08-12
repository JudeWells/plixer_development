"""Time-conditioned 3D residual U-Net -- the network behind the flow-matching Poc2Mol.

Why a separate module rather than a flag on ``ResidualUNetSE3D``
---------------------------------------------------------------
A flow model has to read the timestep at *every* resolution, and the standard way to do
that is FiLM/AdaGN: a per-channel ``(scale, shift)`` derived from an embedding of ``t``,
applied inside each residual block after its last convolution and before the residual add.
The vendored blocks in ``pytorch3dunet_lib`` take ``forward(x)`` only, and threading a
second argument through them would change a library the regression Poc2Mol still depends
on. The encoder/decoder scaffolding is therefore duplicated here -- it is short -- while
the actual convolution stacks (``SingleConv``, ``TransposeConvUpsampling``, the SE gates)
are imported unchanged, so the two models stay comparable block for block and a parameter
count difference can only come from the FiLM heads and the wider input conv.

Two deliberate departures from the vendored code:

* Only the ResNet decoder path is implemented (transposed-conv upsampling, summation
  joining). That is the branch ``upsample='default'`` selects for ``ResNetBlockSE``, which
  is the only basic module Poc2Mol has ever used.
* The FiLM heads and the final 1x1 convolution are **zero-initialised**. At step zero the
  network therefore predicts a zero velocity field and every block is exactly its
  time-independent counterpart, which keeps the first few hundred steps from being spent
  unlearning a random velocity.

Conditioning on the pocket is by input concatenation: ``forward`` takes the noisy ligand
grid and the protein grid and stacks them on the channel axis. Dropping the condition (for
classifier-free guidance) is the caller's job -- pass zeros.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.models.pytorch3dunet_lib.unet3d.buildingblocks import (
    ResNetBlockSE,
    TransposeConvUpsampling,
)


def number_of_features_per_level(init_channel_number: int, num_levels: int) -> List[int]:
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer/DDPM sinusoidal features for a continuous ``t`` in [0, 1].

    ``t`` is multiplied by ``max_period ** 0`` .. i.e. scaled by 1000 first, which is the
    convention every diffusion codebase uses and keeps the low-frequency components from
    being nearly constant across the interval.
    """

    def __init__(self, dim: int, max_period: float = 10000.0, scale: float = 1000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().reshape(-1, 1) * self.scale * freqs.reshape(1, -1)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeResNetBlockSE(ResNetBlockSE):
    """``ResNetBlockSE`` with a FiLM modulation driven by the time embedding.

    Placement follows the usual diffusion U-Net: after the block's second convolution
    (which, with a ``'gcr'``-style order, is groupnorm -> conv with the non-linearity
    stripped) and before the residual addition. Nothing downstream re-normalises it away
    within the block, so the modulation reaches the residual stream intact.
    """

    def __init__(self, in_channels: int, out_channels: int, temb_dim: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.film = nn.Linear(temb_dim, 2 * out_channels)
        # Identity at initialisation: scale 0 -> (1 + scale) == 1, shift 0.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = self.conv1(x)
        out = self.conv2(residual)
        out = self.conv3(out)

        scale, shift = self.film(temb.to(out.dtype)).chunk(2, dim=1)
        spatial = (1,) * (out.dim() - 2)
        out = out * (1.0 + scale.view(*scale.shape, *spatial)) + shift.view(
            *shift.shape, *spatial
        )

        out = out + residual
        out = self.non_linearity(out)
        return self.se_module(out)


class TimeEncoder(nn.Module):
    """One encoder stage: optional max-pool, then a time-conditioned residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        conv_kernel_size: int = 3,
        apply_pooling: bool = True,
        pool_kernel_size: int = 2,
        conv_layer_order: str = "gcr",
        num_groups: int = 8,
    ):
        super().__init__()
        self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size) if apply_pooling else None
        self.basic_module = TimeResNetBlockSE(
            in_channels,
            out_channels,
            temb_dim=temb_dim,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            is3d=True,
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x, temb)


class TimeDecoder(nn.Module):
    """One decoder stage: transposed-conv upsample, summation join, conditioned block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        conv_kernel_size: int = 3,
        scale_factor: int = 2,
        conv_layer_order: str = "gcr",
        num_groups: int = 8,
    ):
        super().__init__()
        self.upsampling = TransposeConvUpsampling(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            scale_factor=scale_factor,
            is3d=True,
        )
        # Summation joining leaves the channel count at out_channels, matching the
        # `adapt_channels` branch the vendored Decoder takes for ResNet blocks.
        self.basic_module = TimeResNetBlockSE(
            out_channels,
            out_channels,
            temb_dim=temb_dim,
            kernel_size=conv_kernel_size,
            order=conv_layer_order,
            num_groups=num_groups,
            is3d=True,
        )

    def forward(
        self, encoder_features: torch.Tensor, x: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = encoder_features + x
        return self.basic_module(x, temb)


class TimeConditionedResidualUNetSE3D(nn.Module):
    """Residual SE U-Net that maps ``(x_t, t, condition) -> velocity``.

    Args:
        in_channels: channels of the *noisy* field, i.e. the ligand channel count.
        cond_channels: channels of the conditioning field concatenated at the input
            (the protein grid). Zero for an unconditional model.
        out_channels: velocity channels; equals ``in_channels`` for flow matching.
        f_maps: width of the first level, doubled per level (or an explicit list).
        temb_dim: width of the time embedding MLP. Defaults to ``4 * f_maps[0]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        f_maps: Union[int, Sequence[int]] = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 5,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        temb_dim: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        f_maps = list(f_maps)
        if len(f_maps) < 2:
            raise ValueError("Required at least 2 levels in the U-Net")

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.f_maps = f_maps

        temb_dim = temb_dim or 4 * f_maps[0]
        self.temb_dim = temb_dim
        # Kept separate from the MLP so the sinusoidal features can be computed in float32
        # (their precision matters, and they are free) while the projection runs in
        # whatever dtype the weights are -- consumers freeze this model in bfloat16.
        self.time_embed = SinusoidalTimeEmbedding(f_maps[0])
        self.time_mlp = nn.Sequential(
            nn.Linear(f_maps[0], temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            encoders.append(
                TimeEncoder(
                    in_channels + cond_channels if i == 0 else f_maps[i - 1],
                    out_feature_num,
                    temb_dim=temb_dim,
                    apply_pooling=i != 0,
                    pool_kernel_size=pool_kernel_size,
                    conv_kernel_size=conv_kernel_size,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            )
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            decoders.append(
                TimeDecoder(
                    reversed_f_maps[i],
                    reversed_f_maps[i + 1],
                    temb_dim=temb_dim,
                    conv_kernel_size=conv_kernel_size,
                    scale_factor=pool_kernel_size,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            )
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        # Zero output at init: the ODE starts as the identity map rather than pushing
        # samples along a random field.
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x``: (B, C_lig, X, Y, Z). ``t``: (B,) in [0, 1]. ``cond``: (B, C_prot, ...)."""
        if self.cond_channels:
            if cond is None:
                raise ValueError(
                    f"model was built with cond_channels={self.cond_channels} but no "
                    "condition was passed; supply zeros for the unconditional branch"
                )
            x = torch.cat([x, cond.to(x.dtype)], dim=1)
        elif cond is not None:
            raise ValueError("model was built unconditional but a condition was passed")

        temb = self.time_mlp(self.time_embed(t).to(self.time_mlp[0].weight.dtype))

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x, temb)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x, temb)

        return self.final_conv(x)
