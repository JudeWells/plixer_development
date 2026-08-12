"""Generative Poc2Mol: conditional flow matching from a pocket to a ligand density.

What changes and why
--------------------
The regression Poc2Mol minimises BCE+Dice against the single true ligand grid, so its
optimum is the *conditional mean* of ligand density given the pocket. Every pathology
measured on the predecessor branch follows from that: the prediction is equivalent to a
~27 deg rigid rotation of the truth, every channel over-emits (1.33x-10.4x), only 58% of
predicted carbon mass lands on real density, and past 60 deg the prediction overlaps a
WRONG pose better than the true one (report.md 12g, 18b).

This module replaces the objective. The model learns a velocity field that transports
Gaussian noise to the ligand density, so a *sample* is a draw from p(ligand | pocket)
rather than its mean -- there is no averaging over poses to blur.

The formulation is rectified flow / conditional-OT flow matching (Lipman et al. 2023,
Liu et al. 2023), which is what modern systems (SD3, Flux) converged on and is a strict
simplification of the score-matching setup VoxMol/VoxBind used: a straight probability
path, a single MSE objective and no noise schedule to tune.

    x0 ~ N(0, I),  x1 = ligand density (rescaled),  t ~ p(t) on [0, 1]
    x_t = (1 - (1 - sigma_min) t) x0 + t x1
    target u = x1 - (1 - sigma_min) x0
    loss = || v_theta(x_t, t, pocket) - u ||^2

Three refinements over the 2023 recipe, all measured wins in the image literature and all
configurable here:

* **Logit-normal timestep sampling** (SD3, Esser et al. 2024). Uniform ``t`` wastes
  capacity on the trivially easy ends of the path; a logit-normal density concentrates
  training where the velocity is hardest to predict.
* **Timestep shift** ``t <- s t / (1 + (s - 1) t)``, applied to the training density *and*
  the sampling grid so the two agree. Higher-dimensional data needs more of the path spent
  near noise. NOTE the convention: here ``t = 0`` is noise and ``t = 1`` is data, the
  reverse of SD3, so "more time near noise" is ``s < 1`` in this codebase, NOT ``s > 1``.
* **Classifier-free guidance.** The pocket is dropped (zeroed) on a fraction of training
  samples, which makes the same weights an unconditional model, and at sampling time
  ``v = v_uncond + w (v_cond - v_uncond)`` sharpens the dependence on the pocket. This is
  the direct knob for "commit to a pose" and has no analogue in the regression model.

Metrics
-------
``val/loss`` here is the flow-matching MSE. It is *not* comparable to the regression
model's ``val/loss`` (different objective entirely) and, like it, it is not the quantity
anyone cares about -- a model can predict velocities well and still sample poor densities.
Model selection should use ``val/sample/dice``, which samples the ODE and scores the result
against the true ligand with exactly the pooled soft Dice used by
``scripts/adhoc_analysis/poc2mol_rotation_control.py``. The yardstick to beat is **0.5027**
-- the regression model on the FULL HiQBind v2 val split (n=1019, sem 0.0033). Quote that,
not a subsample: the first 128 pockets give 0.4976 and the first 256 give 0.5080, and
comparing a flow number taken on one subset against a baseline taken on another understated
the gap by 0.028 on 2026-08-11.
report.md 12g's 0.596 is a different checkpoint on 104 PLINDER-panel pockets and is NOT
comparable to anything scored here, though its ~27 deg rotation calibration stands for that
setting.

Note that the Dice floor problem (report.md 3c) does not exist here: MSE over velocity has
no per-channel averaging and an empty target channel still receives gradient, because its
target velocity ``-x0`` is non-zero noise. The model is penalised for emitting mass into a
channel the ligand leaves empty, which is what produced the constant fluorine smear under
Dice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from transformers.optimization import get_scheduler

from src.models.flow_unet3d import TimeConditionedResidualUNetSE3D


class FlowUnetConfig:
    """Architecture of the velocity network.

    ``ligand_channels`` and ``protein_channels`` are named for what they carry rather than
    following the regression config's ``in_channels``/``out_channels``, because the U-Net's
    input is the concatenation of both and getting that arithmetic wrong in a YAML file is
    exactly the kind of silent channel mismatch that has cost this project runs before.
    """

    def __init__(
        self,
        ligand_channels: int = 11,
        protein_channels: int = 4,
        f_maps: int = 64,
        layer_order: str = "gcr",
        num_groups: int = 8,
        num_levels: int = 5,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        temb_dim: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):
        # `out_channels` is the regression config's name for the same quantity, and the
        # shared stage-3 data config sets it. Accepted so a flow model can be dropped into
        # that pipeline, but checked rather than ignored -- a silent disagreement between
        # the two names is exactly the channel mismatch this class was named to prevent.
        if out_channels is not None and out_channels != ligand_channels:
            raise ValueError(
                f"out_channels={out_channels} contradicts ligand_channels={ligand_channels}; "
                "for a flow model the output width IS the ligand channel count"
            )
        self.ligand_channels = ligand_channels
        self.protein_channels = protein_channels
        self.f_maps = f_maps
        self.layer_order = layer_order
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.temb_dim = temb_dim


def pooled_soft_dice(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample soft Dice pooled over channels -- identical to the yardstick script.

    Pooled, not per-channel-averaged: averaging lets the many empty channels dominate (an
    all-zero target channel contributes a fixed 1.0 regardless of the prediction), which is
    the floor that made the regression model's ``val/loss`` unusable (report.md 3c).
    """
    dims = tuple(range(1, a.dim()))
    num = 2.0 * (a * b).sum(dim=dims)
    den = (a * a).sum(dim=dims) + (b * b).sum(dim=dims)
    return num / (den + eps)


def apply_time_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """SD3's resolution shift: ``s t / (1 + (s - 1) t)``. ``shift == 1`` is the identity."""
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


class Poc2MolFlow(LightningModule):
    """Flow-matching Poc2Mol. Drop-in for the regression model at the batch interface.

    Consumes the same batches as :class:`src.models.poc2mol.Poc2Mol` (``batch['protein']``,
    ``batch['ligand']``) and exposes the same ``forward(prot_vox, labels=None) ->
    {'predicted_ligand_voxels': ...}`` contract, except that the prediction is produced by
    integrating the ODE rather than by a single forward pass. Set ``protein_channels = 0``
    on the config for unconditional (ZINC) pretraining.
    """

    is_generative = True

    def __init__(
        self,
        config: FlowUnetConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[Dict[str, Any]] = None,
        # ---- flow-matching objective -------------------------------------------------
        sigma_min: float = 0.0,
        # "mse" is flow matching proper. "dice" and "hybrid" score the implied x1 with the
        # pooled soft Dice the project measures on, trading the transport guarantee for
        # agreement with the metric; see the note in flow_loss.
        loss_type: str = "mse",
        dice_weight: float = 1.0,
        time_sampling: str = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        time_shift: float = 1.0,
        cond_dropout_prob: float = 0.1,
        # ---- data representation -----------------------------------------------------
        occupancy_shift: float = 0.5,
        occupancy_scale: float = 0.5,
        # ---- sampling ----------------------------------------------------------------
        sample_steps: int = 50,
        sampler: str = "heun",
        guidance_scale: float = 1.0,
        sample_clamp: Optional[float] = None,
        # ---- validation --------------------------------------------------------------
        val_sample_every_n_epochs: int = 1,
        n_val_sample_pockets: int = 128,
        val_sample_steps: int = 25,
        val_guidance_scale: Optional[float] = None,
        val_noise_seed: int = 1234,
        n_val_t_grid: int = 32,
        n_t_bins: int = 8,
        val_restore_t: Optional[float] = 0.5,
        # Also score the CONDITIONAL-MEAN readout during validation. The sampled readout
        # understates this model class badly (0.247 vs 0.458 on the same checkpoint,
        # 2026-08-12), so selecting checkpoints on it selects on the wrong quantity --
        # exactly the failure report.md 18h documents for the regression model. Both are
        # logged: val/sample/dice stays comparable with earlier runs, val/mean/dice is the
        # one worth selecting on.
        val_mean_readout: bool = True,
        val_mean_draws: int = 8,
        val_mean_guidance: float = 3.0,
        # ---- EMA ---------------------------------------------------------------------
        ema_decay: float = 0.999,
        ema_warmup_steps: int = 1000,
        # ---- housekeeping ------------------------------------------------------------
        matmul_precision: str = "high",
        override_optimizer_on_load: bool = False,
        img_save_dir: Optional[str] = None,
        visualise_val: bool = False,
        n_samples_for_visualisation: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if time_sampling not in {"logit_normal", "uniform"}:
            raise ValueError(f"unknown time_sampling {time_sampling!r}")
        if sampler not in {"euler", "heun"}:
            raise ValueError(f"unknown sampler {sampler!r}")
        if occupancy_scale <= 0:
            raise ValueError("occupancy_scale must be positive")

        self.model = TimeConditionedResidualUNetSE3D(
            in_channels=config.ligand_channels,
            out_channels=config.ligand_channels,
            cond_channels=config.protein_channels,
            f_maps=config.f_maps,
            layer_order=config.layer_order,
            num_groups=config.num_groups,
            num_levels=config.num_levels,
            conv_kernel_size=config.conv_kernel_size,
            pool_kernel_size=config.pool_kernel_size,
            temb_dim=config.temb_dim,
        )
        self.flow_config = config
        self.n_ligand_channels = config.ligand_channels
        self.n_protein_channels = config.protein_channels

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler or {}

        self.sigma_min = sigma_min
        if loss_type not in {"mse", "dice", "hybrid"}:
            raise ValueError(f"unknown loss_type {loss_type!r}")
        self.loss_type = loss_type
        self.dice_weight = dice_weight
        self.time_sampling = time_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.time_shift = time_shift
        self.cond_dropout_prob = cond_dropout_prob

        self.occupancy_shift = occupancy_shift
        self.occupancy_scale = occupancy_scale

        self.sample_steps = sample_steps
        self.sampler = sampler
        self.guidance_scale = guidance_scale
        # Bound on |x| in model space, applied after every ODE step. This is the standard
        # remedy for a trajectory that runs away, but it is OFF by default on purpose: with
        # it on, a diverging model still returns a plausible-looking saturated grid and the
        # diagnostics understate the problem. Diagnose first, then enable it.
        self.sample_clamp = sample_clamp

        self.val_sample_every_n_epochs = val_sample_every_n_epochs
        self.n_val_sample_pockets = n_val_sample_pockets
        self.val_sample_steps = val_sample_steps
        self.val_guidance_scale = (
            guidance_scale if val_guidance_scale is None else val_guidance_scale
        )
        self.val_noise_seed = val_noise_seed
        self.n_val_t_grid = n_val_t_grid
        self.n_t_bins = n_t_bins
        self.val_restore_t = val_restore_t
        self.val_mean_readout = val_mean_readout
        self.val_mean_draws = val_mean_draws
        self.val_mean_guidance = val_mean_guidance
        # Per-source counters, reset in on_validation_epoch_start.
        self._val_pockets_sampled = {}

        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self._ema_shadow: Optional[Dict[str, torch.Tensor]] = None
        self._ema_backup: Optional[Dict[str, torch.Tensor]] = None
        self._pending_raw_state: Optional[Dict[str, torch.Tensor]] = None

        self.override_optimizer_on_load = override_optimizer_on_load
        self.img_save_dir = img_save_dir
        self.visualise_val = visualise_val
        self.n_samples_for_visualisation = n_samples_for_visualisation

        torch.set_float32_matmul_precision(matmul_precision)

    # ------------------------------------------------------------------ representation

    def to_model_space(self, occupancy: torch.Tensor) -> torch.Tensor:
        """Occupancy in [0, 1] -> the space the flow runs in (default [-1, 1])."""
        return (occupancy.float() - self.occupancy_shift) / self.occupancy_scale

    def to_occupancy(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """Inverse of :meth:`to_model_space`, optionally clamped back into [0, 1]."""
        occ = x.float() * self.occupancy_scale + self.occupancy_shift
        return occ.clamp(0.0, 1.0) if clamp else occ

    def _condition(self, protein: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
        """The conditioning tensor, or ``None`` for an unconditional model.

        A model built with ``protein_channels = 0`` never sees a pocket. Otherwise a
        missing/absent protein grid (ZINC rows in a mixed batch) becomes zeros, which is
        exactly the unconditional branch classifier-free guidance is trained on.
        """
        if not self.n_protein_channels:
            return None
        if protein is None:
            return reference.new_zeros(
                reference.shape[0], self.n_protein_channels, *reference.shape[2:]
            )
        if protein.shape[1] != self.n_protein_channels:
            raise ValueError(
                f"batch['protein'] has {protein.shape[1]} channels but the model was built "
                f"for {self.n_protein_channels}; a mismatch here would concatenate silently "
                "and shift every channel the U-Net reads"
            )
        return protein.float()

    # ---------------------------------------------------------------------- objective

    def _sample_time(self, n: int, device, generator=None) -> torch.Tensor:
        if self.time_sampling == "uniform":
            t = torch.rand(n, device=device, generator=generator)
        else:
            normal = torch.randn(n, device=device, generator=generator)
            t = torch.sigmoid(self.logit_normal_mean + self.logit_normal_std * normal)
        return apply_time_shift(t, self.time_shift)

    def flow_loss(
        self,
        ligand: torch.Tensor,
        protein: Optional[torch.Tensor],
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator=None,
        drop_condition: bool = True,
        bucket_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """One flow-matching step.

        Returns ``{"loss": ...}``, plus per-timestep-bucket losses when
        ``bucket_diagnostics`` is set. Those keys are deliberately absent by default: a
        bucket can be empty for a given batch, and a metric that some ranks log and others
        do not is a DDP deadlock waiting to happen at epoch-level aggregation.
        """
        if ligand.shape[1] != self.n_ligand_channels:
            raise ValueError(
                f"batch['ligand'] has {ligand.shape[1]} channels but the model was built "
                f"for {self.n_ligand_channels}; check the data config's ligand_channels "
                "against model.config.ligand_channels"
            )
        x1 = self.to_model_space(ligand)
        cond = self._condition(protein, x1)

        if noise is None:
            noise = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        x0 = noise
        if t is None:
            t = self._sample_time(x1.shape[0], x1.device, generator=generator)

        # Classifier-free guidance needs an unconditional branch, so a fraction of samples
        # are trained with the pocket zeroed. Per-sample, not per-batch: the two branches
        # then share every batch and the unconditional one is never starved.
        if cond is not None and drop_condition and self.cond_dropout_prob > 0:
            keep = (
                torch.rand(cond.shape[0], device=cond.device, generator=generator)
                >= self.cond_dropout_prob
            )
            cond = cond * keep.view(-1, *([1] * (cond.dim() - 1))).to(cond.dtype)

        t_broadcast = t.view(-1, *([1] * (x1.dim() - 1))).to(x1.dtype)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_broadcast) * x0 + t_broadcast * x1
        target = x1 - (1.0 - self.sigma_min) * x0

        velocity = self.model(x_t, t, cond)
        spatial = tuple(range(1, x1.dim()))
        per_sample = (velocity.float() - target).pow(2).mean(dim=spatial)

        # ---- optional Dice term, scored in x1-space -------------------------------------
        # Flow matching's guarantee rests on MSE: the conditional expectation is the
        # minimiser of squared error, which is why regressing the per-sample target
        # (x1 - x0) recovers the MARGINAL velocity field and its ODE transports noise to
        # data. A Dice term forfeits that -- the ODE no longer targets p(data).
        #
        # It is offered anyway because the ODE turned out to be worse than not using it:
        # every readout that integrates LESS scores better, and the one we actually deploy
        # is the single-step x0 + v(x0, 0, c). If the sampler is discarded, the flow
        # property is a guarantee being paid for and thrown away, and the loss becomes a
        # free empirical choice.
        #
        # The parameterisation inverts exactly -- x_t = (1-t) x0 + t x1 and v = x1 - x0 give
        # x1 = x_t + (1-t) v -- so Dice is well defined at every t, and the (1-t) factor
        # makes the gradient to v vanish as t -> 1. That AUTOMATICALLY concentrates the term
        # near t = 0, which is exactly where the deployed readout lives, with no hand-tuned
        # weighting. The flip side is almost no signal at high t, which is why `hybrid`
        # (MSE + weight * Dice) is the sane default rather than pure `dice`.
        if self.loss_type != "mse":
            t_b = t.view(-1, *([1] * (x1.dim() - 1))).to(x1.dtype)
            x1_hat = x_t + (1.0 - t_b) * velocity.float()
            # Dice is defined on occupancies; clamping is what keeps an over-shooting
            # prediction from producing a negative "intersection".
            occ_hat = self.to_occupancy(x1_hat, clamp=True)
            occ_true = self.to_occupancy(x1, clamp=True)
            dice_per_sample = 1.0 - pooled_soft_dice(occ_hat, occ_true)
            if self.loss_type == "dice":
                per_sample = dice_per_sample
            else:  # hybrid
                per_sample = per_sample + self.dice_weight * dice_per_sample
            out_dice = dice_per_sample.mean().detach()
        else:
            out_dice = None

        out = {"loss": per_sample.mean(), "per_sample": per_sample.detach()}
        if out_dice is not None:
            out["dice_term"] = out_dice
        # Where along the path is the model weak? Bucketed rather than binned per-step so
        # the curve is readable at any batch size.
        if bucket_diagnostics:
            with torch.no_grad():
                bucket = (t * self.n_t_bins).long().clamp(0, self.n_t_bins - 1)
                for b in range(self.n_t_bins):
                    mask = bucket == b
                    if mask.any():
                        out[f"t_bucket_{b}"] = per_sample[mask].mean().detach()
        return out

    # ------------------------------------------------------------------------ sampling

    @torch.no_grad()
    def sample(
        self,
        protein: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        grid_shape: Optional[tuple] = None,
        n_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        sampler: Optional[str] = None,
        generator=None,
        x0: Optional[torch.Tensor] = None,
        return_occupancy: bool = True,
        t_start: float = 0.0,
        return_stats: bool = False,
    ):
        """Integrate the probability-flow ODE from noise (t=0) to data (t=1).

        Args:
            protein: conditioning grid ``(B, C_prot, X, Y, Z)``. Required unless the model
                is unconditional or ``x0``/``batch_size`` + ``grid_shape`` are given.
            n_steps: ODE steps. Heun costs two network evaluations per step.
            guidance_scale: classifier-free guidance weight. 1.0 disables it (and halves
                the cost, since the unconditional branch is then not evaluated).
            x0: initial state at ``t_start``; pass it to compare settings on one draw.
            t_start: where on the path to start. 0 is ordinary sampling; a value in (0, 1)
                with ``x0`` set to a partially-noised TRUE grid is the restoration probe --
                see ``_log_sample_metrics``.
            return_stats: also return trajectory statistics. The ODE state has a known
                healthy scale (both endpoints have RMS ~1 in model space, and so does every
                point between), so RMS growth is a direct, threshold-free readout of the
                sampler leaving the data manifold.

        Returns occupancies in [0, 1] by default, or raw model-space values; a
        ``(tensor, stats)`` pair when ``return_stats`` is set.
        """
        n_steps = n_steps or self.sample_steps
        sampler = sampler or self.sampler
        w = self.guidance_scale if guidance_scale is None else guidance_scale

        device = self.device
        if x0 is None:
            if protein is not None:
                shape = (protein.shape[0], self.n_ligand_channels, *protein.shape[2:])
            else:
                if batch_size is None or grid_shape is None:
                    raise ValueError(
                        "sample() needs either `protein`, or `x0`, or "
                        "`batch_size` + `grid_shape`"
                    )
                shape = (batch_size, self.n_ligand_channels, *grid_shape)
            x0 = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)

        x = x0.float()
        cond = self._condition(protein.float() if protein is not None else None, x)
        uncond = torch.zeros_like(cond) if (cond is not None and w != 1.0) else None

        # Same shift as training, so the steps are dense where the model was trained dense.
        grid = apply_time_shift(
            torch.linspace(0.0, 1.0, n_steps + 1, device=device), self.time_shift
        )
        if t_start > 0.0:
            # Keep the shifted schedule rather than re-spacing the sub-interval, so a
            # restoration probe traverses exactly the steps ordinary sampling would.
            grid = torch.cat([torch.tensor([t_start], device=device), grid[grid > t_start]])
            n_steps = grid.numel() - 1

        # The ODE state is integrated in float32 whatever the weights are, but the network
        # has to be fed its own dtype: consumers cast the frozen model to bfloat16 (see
        # Poc2MolInferenceBuilder), and a float32 input against bfloat16 weights is a hard
        # error rather than a silent upcast.
        param_dtype = next(self.model.parameters()).dtype

        def velocity(x_in: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
            t_vec = t_scalar.expand(x_in.shape[0])
            x_in = x_in.to(param_dtype)
            if uncond is None:
                return self.model(x_in, t_vec, cond.to(param_dtype) if cond is not None else None).float()
            # Both branches in one forward pass: same kernels, half the launches.
            doubled = torch.cat([x_in, x_in], dim=0)
            v = self.model(
                doubled,
                torch.cat([t_vec, t_vec], dim=0),
                torch.cat([cond, uncond], dim=0).to(param_dtype),
            ).float()
            v_cond, v_uncond = v.chunk(2, dim=0)
            return v_uncond + w * (v_cond - v_uncond)

        n_elements = x[0].numel()
        traj_rms_max = 0.0

        for i in range(n_steps):
            t0, t1 = grid[i], grid[i + 1]
            dt = t1 - t0
            v0 = velocity(x, t0)
            if sampler == "euler":
                x = x + dt * v0
            else:  # Heun: one corrector evaluation, second-order accurate
                x_euler = x + dt * v0
                v1 = velocity(x_euler, t1)
                x = x + dt * 0.5 * (v0 + v1)

            if self.sample_clamp is not None:
                x = x.clamp(-self.sample_clamp, self.sample_clamp)
            if return_stats:
                traj_rms_max = max(
                    traj_rms_max,
                    float(x.pow(2).sum(dim=tuple(range(1, x.dim()))).div(n_elements).sqrt().max()),
                )

        result = self.to_occupancy(x) if return_occupancy else x
        if not return_stats:
            return result

        occ_raw = self.to_occupancy(x, clamp=False)
        stats = {
            # Both endpoints of the probability path have RMS ~1 in model space, and so
            # does every point between, so this is ~1 for a healthy trajectory whatever the
            # model has learned. It is the metric that separates "not trained yet" from
            # "the sampler ran away", which sample Dice alone cannot do.
            "traj_rms_max": traj_rms_max,
            "final_rms": float(x.pow(2).mean().sqrt()),
            # Occupancy BEFORE the clamp. The clamp in to_occupancy would otherwise hide a
            # divergence completely: x = 1e6 and x = 1.5 produce the same clamped grid.
            "out_of_range": float(((occ_raw < -0.1) | (occ_raw > 1.1)).float().mean()),
            "max_abs_occ": float(occ_raw.abs().max()),
        }
        return result, stats

    @torch.no_grad()
    def predict_expected(
        self,
        protein: Optional[torch.Tensor] = None,
        n_draws: int = 1,
        mode: str = "one_step",
        n_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: float = 1.0,
        generator=None,
        return_occupancy: bool = True,
    ) -> torch.Tensor:
        """Estimate E[ligand | pocket] -- the DICE-OPTIMAL readout of a generative model.

        Why this exists: ``dice(pred, true)`` is maximised in expectation by the conditional
        MEAN, not by a sample. That is the whole reason the regression Poc2Mol scores 0.5027
        while a single draw from this model scores ~0.23 -- the regression net is trained to
        output exactly the estimator the metric rewards. Comparing a sample against it on
        Dice is a category error; comparing this against it is not.

        Three ways to get the mean out, cheapest first:

        ``one_step``
            At ``t = 0`` the path's velocity is ``x1 - x0``, so a single network evaluation
            gives ``x0 + v(x0, 0, c) = E[x1 | x0, c]``. One forward pass, versus 100 for a
            Heun trajectory. Averaging over ``n_draws`` values of ``x0`` marginalises the
            residual dependence on the noise draw.
        ``mean_of_k``
            Average ``n_draws`` complete ODE samples. Converges to the same quantity from
            the other direction, at ``n_draws`` x the cost, but each member is a genuine
            sample so the average is an honest Monte-Carlo estimate of the mean.
        ``one_step_mid``
            ``one_step`` evaluated at ``t = 0.5`` on a half-noised draw and extrapolated.
            Sometimes better conditioned than t = 0, where the input carries no signal.

        ``temperature`` < 1 shrinks the initial noise, which narrows the distribution being
        averaged over. At 0 with ``one_step`` the estimate becomes fully deterministic.
        """
        if mode not in {"one_step", "mean_of_k", "one_step_mid"}:
            raise ValueError(f"unknown mode {mode!r}")

        device = self.device
        if protein is None:
            raise ValueError("predict_expected needs a pocket to condition on")
        protein = protein.float()
        shape = (protein.shape[0], self.n_ligand_channels, *protein.shape[2:])
        param_dtype = next(self.model.parameters()).dtype
        w = self.guidance_scale if guidance_scale is None else guidance_scale
        cond = self._condition(protein, torch.empty(shape, device=device))
        uncond = torch.zeros_like(cond) if (cond is not None and w != 1.0) else None

        def velocity(x_in, t_value):
            t_vec = torch.full((x_in.shape[0],), float(t_value), device=device)
            x_in = x_in.to(param_dtype)
            if uncond is None:
                return self.model(x_in, t_vec, cond.to(param_dtype)).float()
            v = self.model(
                torch.cat([x_in, x_in], dim=0),
                torch.cat([t_vec, t_vec], dim=0),
                torch.cat([cond, uncond], dim=0).to(param_dtype),
            ).float()
            v_cond, v_uncond = v.chunk(2, dim=0)
            return v_uncond + w * (v_cond - v_uncond)

        total = torch.zeros(shape, device=device, dtype=torch.float32)
        for _ in range(max(1, n_draws)):
            x0 = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
            x0 = x0 * temperature
            if mode == "mean_of_k":
                total += self.sample(protein=protein, x0=x0, n_steps=n_steps,
                                     guidance_scale=w, return_occupancy=False).float()
            elif mode == "one_step":
                total += x0 + velocity(x0, 0.0)
            else:  # one_step_mid: x_t = (1-t) x0 + t x1  ->  x1 = x_t + (1-t) v
                t_mid = 0.5
                # No access to x1, so the midpoint state is built from noise alone; this is
                # a probe of whether t=0 is a badly conditioned place to ask the question.
                x_mid = x0
                total += x_mid + (1.0 - t_mid) * velocity(x_mid, t_mid)
        expected = total / max(1, n_draws)
        return self.to_occupancy(expected) if return_occupancy else expected

    # ------------------------------------------------------------------- lightning API

    def forward(self, prot_vox, labels=None, **sample_kwargs):
        """Compatibility surface with :class:`src.models.poc2mol.Poc2Mol`.

        Returns ``predicted_ligand_voxels`` obtained by sampling. There is no
        ``predicted_ligand_logits`` -- the model has no logit parameterisation -- so
        consumers that reach for it will fail loudly rather than read a wrong tensor.
        """
        predicted = self.sample(protein=prot_vox, **sample_kwargs)
        out = {"predicted_ligand_voxels": predicted}
        if labels is not None:
            out.update(self.flow_loss(labels, prot_vox, drop_condition=False))
        return out

    def training_step(self, batch, batch_idx):
        if "load_time" in batch:
            self.log("train/load_time", batch["load_time"].mean(), on_step=True,
                     on_epoch=False, prog_bar=True)

        outputs = self.flow_loss(
            batch["ligand"], batch.get("protein"), bucket_diagnostics=True
        )
        self.log("train/batch_loss", outputs["loss"], on_step=True, on_epoch=False,
                 prog_bar=True)
        self.log("train/loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True)

        # Mixed-source training (stage B): the two sources are different problems -- one is
        # conditional, one is not -- so a single averaged curve cannot show ZINC being
        # forgotten while the pocket loss improves, which is the specific risk of mixing.
        # Step-only and rank-zero-only: a batch can contain zero rows of a source, and a
        # metric some ranks log and others do not deadlocks DDP at epoch-level reduction.
        if "has_pocket" in batch:
            per_sample = outputs["per_sample"]
            has_pocket = batch["has_pocket"].to(per_sample.device)
            for name, mask in (("pocket", has_pocket), ("ligand_only", ~has_pocket)):
                if bool(mask.any()):
                    self.log(f"train/loss_{name}", per_sample[mask].mean(),
                             on_step=True, on_epoch=False, rank_zero_only=True)
            self.log("train/frac_pocket", has_pocket.float().mean(),
                     on_step=True, on_epoch=False, rank_zero_only=True)

        for key, value in outputs.items():
            if key in {"loss", "per_sample"}:
                continue
            # Step-only and rank-zero-only: these keys come and go with the batch's
            # timestep draw, so they must never participate in cross-rank reduction.
            self.log(f"train/{key}", value, on_step=True, on_epoch=False,
                     rank_zero_only=True)
        return outputs["loss"]

    def _global_val_indices(self, batch_idx: int, n: int, device) -> torch.Tensor:
        """Position of each sample in the validation set, assuming full batches.

        Everything deterministic about validation is keyed off this rather than off
        ``batch_idx``, so the noise a sample gets and the timestep it is scored at do not
        change when the validation batch size does. Without that, ``val/loss`` would be a
        different quantity at every batch size and no two runs would be comparable.
        """
        return batch_idx * n + torch.arange(n, device=device)

    def _deterministic_noise(self, indices, shape, device, offset: int = 0):
        """One independent, reproducible noise draw per validation sample."""
        generator = torch.Generator(device=device)
        draws = []
        for index in indices.tolist():
            generator.manual_seed(self.val_noise_seed + offset + int(index))
            draws.append(
                torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
            )
        return torch.stack(draws)

    def _val_kind(self, dataloader_idx: int) -> Optional[str]:
        """Source name for this validation dataloader, or None if there is only one.

        Read from the datamodule rather than assumed positionally, so adding or reordering a
        validation set cannot silently relabel a curve.
        """
        datamodule = getattr(self.trainer, "datamodule", None) if self.trainer else None
        kinds = getattr(datamodule, "val_dataset_kinds", None)
        if not kinds or len(kinds) < 2:
            return None
        return kinds[dataloader_idx] if dataloader_idx < len(kinds) else str(dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # Deterministic by construction: each sample gets its own reproducible noise draw
        # and sits at a fixed point of a stratified timestep grid, so val/loss is the same
        # quantity at every epoch and across runs. A stochastic val loss would be a moving
        # target for model selection -- the mistake report.md 3e documents.
        ligand = batch["ligand"]
        n = ligand.shape[0]
        indices = self._global_val_indices(batch_idx, n, ligand.device)
        noise = self._deterministic_noise(
            indices, (self.n_ligand_channels, *ligand.shape[2:]), ligand.device
        )
        t = apply_time_shift(
            ((indices % self.n_val_t_grid).float() + 0.5) / self.n_val_t_grid,
            self.time_shift,
        )

        outputs = self.flow_loss(
            ligand, batch.get("protein"), t=t, noise=noise, drop_condition=False
        )

        # With a mixed training set, validation is two SEPARATE dataloaders rather than a
        # blend, so each source gets an independently-sized metric whose value does not move
        # when the training mixture does. `add_dataloader_idx=False` because the name
        # already carries the source; letting Lightning append /dataloader_idx_N would
        # rename the metric a checkpoint is selected on.
        kind = self._val_kind(dataloader_idx)
        prefix = f"val/{kind}" if kind else "val"
        for key, value in outputs.items():
            if key == "per_sample":
                continue
            self.log(f"{prefix}/{key}", value, on_step=False, on_epoch=True,
                     prog_bar=(key == "loss" and not kind), sync_dist=True,
                     add_dataloader_idx=False)
        # The unprefixed name stays pinned to the POCKET set, so `val/loss` means the same
        # thing here as in the from-scratch control arm and the two remain comparable.
        if kind == "hiqbind":
            self.log("val/loss", outputs["loss"], on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True, add_dataloader_idx=False)

        # The metric that actually matters: sample the ODE and score the density. Capped by
        # a POCKET budget rather than a batch count, because it costs val_sample_steps
        # forward passes per batch and a batch-count cap would silently change the size of
        # the metric's sample whenever the batch size changed. The budget is per rank.
        if self._should_sample(kind):
            self._log_sample_metrics(batch, batch_idx, kind)

        return outputs["loss"]

    def on_validation_epoch_start(self):
        self._val_pockets_sampled = {}
        if self.ema_decay > 0:
            self._ema_swap_in()

    def _should_sample(self, kind: Optional[str] = None) -> bool:
        if self.val_sample_every_n_epochs <= 0:
            return False
        # Budget is PER SOURCE. A shared counter would let the first dataloader spend the
        # whole allowance and leave the second with no sampling metrics at all.
        if self._val_pockets_sampled.get(kind or "", 0) >= self.n_val_sample_pockets:
            return False
        return self.current_epoch % self.val_sample_every_n_epochs == 0

    def _log_sample_metrics(self, batch, batch_idx: int, kind: Optional[str] = None) -> None:
        ligand = batch["ligand"].float()
        n = ligand.shape[0]
        key = kind or ""
        self._val_pockets_sampled[key] = self._val_pockets_sampled.get(key, 0) + n
        # Same rule as the loss metrics: the name carries the source, and the unprefixed
        # names stay pinned to the pocket set so they mean what they meant before mixing.
        prefix = f"val/{kind}" if kind else "val"
        pinned = kind in (None, "hiqbind")
        indices = self._global_val_indices(batch_idx, n, ligand.device)
        # Offset the seed so a sample's starting noise is not the same draw its flow loss
        # was scored against.
        x0 = self._deterministic_noise(
            indices, (self.n_ligand_channels, *ligand.shape[2:]), ligand.device,
            offset=10_000,
        )

        was_training = self.training
        self.eval()
        predicted, stats = self.sample(
            protein=batch.get("protein"),
            x0=x0,
            n_steps=self.val_sample_steps,
            guidance_scale=self.val_guidance_scale,
            return_stats=True,
        )

        # ------------------------------------------------------------ divergence probes
        # The failure this guards against: training loss falls nicely, then sampling walks
        # off the data manifold and returns a nonsensical grid. It happens because the
        # network only ever sees x_t ON the true path during training, so once integration
        # error pushes the state off it the velocities are extrapolations, and the error
        # compounds. Sample Dice alone cannot diagnose it -- an untrained model and a
        # diverged one both score ~0.
        #
        # `restore` starts from the TRUE path at t = val_restore_t and integrates the rest
        # of the way. It uses the same weights and the same sampler, so:
        #
        #   restore high, sample low  -> the velocity field is right, the TRAJECTORY is
        #                                diverging. Fix with more steps, a stronger sampler,
        #                                lower guidance, or model.sample_clamp.
        #   both low                  -> the model has not learned the field yet. Keep
        #                                training; the sampler is not the problem.
        #   restore low, sample high  -> should not happen; suspect a metric bug.
        restore = None
        if self.val_restore_t is not None:
            t_r = float(self.val_restore_t)
            x1_true = self.to_model_space(ligand)
            x_partial = (1.0 - t_r) * x0 + t_r * x1_true
            restored = self.sample(
                protein=batch.get("protein"),
                x0=x_partial,
                t_start=t_r,
                n_steps=self.val_sample_steps,
                guidance_scale=self.val_guidance_scale,
            )
            restore = pooled_soft_dice(restored, ligand).mean()
        if was_training:
            self.train()

        # The tuned readout, alongside the sampled one. 8 draws x CFG = 16 network evals,
        # against 50 for the 25-step Heun sample, so this is the cheaper of the two.
        if self.val_mean_readout:
            mean_pred = self.predict_expected(
                protein=batch.get("protein"), mode="one_step",
                n_draws=self.val_mean_draws, guidance_scale=self.val_mean_guidance,
            )
            mean_dice = pooled_soft_dice(mean_pred, ligand).mean()

        dice = pooled_soft_dice(predicted, ligand).mean()
        restore_name = f"restore/dice_t{int(100 * self.val_restore_t):02d}" if restore is not None else None

        def log_metric(name, value):
            self.log(f"{prefix}/{name}", value, on_step=False, on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)
            if prefix != "val" and pinned:
                self.log(f"val/{name}", value, on_step=False, on_epoch=True,
                         sync_dist=True, add_dataloader_idx=False)

        self.log(f"{prefix}/sample/dice", dice, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        if prefix != "val" and pinned:
            # The checkpoint monitor and the control arm both reference this exact name.
            self.log("val/sample/dice", dice, on_step=False, on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)
        if restore is not None:
            log_metric(restore_name, restore)
        if self.val_mean_readout:
            log_metric("mean/dice", mean_dice)

        # Scale diagnostics. Every point of the probability path has RMS ~1 in model space,
        # so traj_rms_max is ~1 when the trajectory is healthy REGARDLESS of how well the
        # model has learned -- it is a threshold-free divergence alarm. rms_ratio compares
        # the endpoint against the true data's own scale.
        true_rms = self.to_model_space(ligand).pow(2).mean().sqrt().clamp(min=1e-6)
        log_metric("sample/traj_rms_max", stats["traj_rms_max"])
        log_metric("sample/rms_ratio", stats["final_rms"] / float(true_rms))
        # Fraction of voxels outside a plausible occupancy range BEFORE clamping, and the
        # single worst voxel. Real grids live in [0, 1]; these stay near 0 and near 1.
        log_metric("sample/out_of_range", stats["out_of_range"])
        log_metric("sample/max_abs_occ", stats["max_abs_occ"])
        # Sparsity, against the data's own value. A diverged sample saturates everywhere.
        log_metric("sample/occupied_frac", (predicted > 0.5).float().mean())
        log_metric("data/occupied_frac", (ligand > 0.5).float().mean())

        # Same three emission diagnostics as the regression model, so the two are directly
        # comparable: total mass ratio, share of mass on real density, share of mass in
        # channels the ligand leaves empty.
        total = predicted.sum().clamp(min=1e-6)
        occupied = ligand > 0.05
        empty_channel = ligand.sum(dim=(2, 3, 4)) <= 0
        log_metric("sample/emission_ratio", total / ligand.sum().clamp(min=1e-6))
        log_metric("sample/on_target", (predicted * occupied).sum() / total)
        log_metric("sample/empty_frac",
                   (predicted.sum(dim=(2, 3, 4)) * empty_channel).sum() / total)

        if self.visualise_val and batch_idx == 0:
            try:
                from src.evaluation.visual import visualise_batch

                k = self.n_samples_for_visualisation
                save_dir = f"{self.img_save_dir}/val" if self.img_save_dir else None
                visualise_batch(
                    batch["ligand"][:k],
                    predicted[:k].detach().cpu().numpy(),
                    batch["name"][:k],
                    save_dir=save_dir,
                    batch=str(batch_idx),
                )
            except Exception as exc:  # visualisation must never kill a run
                print(f"Error visualising batch {batch_idx}: {exc}")

    def test_step(self, batch, batch_idx):
        outputs = self.flow_loss(batch["ligand"], batch.get("protein"), drop_condition=False)
        return outputs["loss"]

    # ------------------------------------------------------------------------ optimiser

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        scheduler_config = dict(self.scheduler_config)
        if not scheduler_config:
            return {"optimizer": optimizer}

        scheduler_type = scheduler_config.get("type", "step")
        if scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 100),
                gamma=scheduler_config.get("gamma", 0.997),
            )
        else:
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
            if isinstance(num_warmup_steps, float) and 0 <= num_warmup_steps < 1:
                num_warmup_steps = int(num_training_steps * num_warmup_steps)

            scheduler_specific_kwargs = {}
            if scheduler_type == "cosine_with_restarts":
                scheduler_specific_kwargs["num_cycles"] = scheduler_config.get("num_cycles", 1)
            elif scheduler_type == "cosine_with_min_lr":
                scheduler_specific_kwargs["min_lr_rate"] = scheduler_config.get("min_lr_rate", 0.1)
            elif scheduler_type == "warmup_stable_decay":
                # Why this exists: a cosine sized to max_steps anneals the LR to its floor
                # whether or not the model has converged. The 2026-08-11 shift sweep hit
                # exactly that -- LR was at a quarter of peak by epoch 449 and at the 0.1
                # floor by 599, while val/loss was still falling monotonically. Every arm
                # was therefore stopped by its schedule rather than by convergence, which
                # confounds "which shift is better" with "which tolerates a decaying LR".
                #
                # Stable-then-decay holds the LR flat for most of the budget, so the run is
                # limited by the data and the model rather than by the schedule, and the
                # decay is a deliberate final anneal instead of a slow throttle.
                # Fractions of the total, so the same config works at any max_epochs.
                total = num_training_steps
                decay = scheduler_config.get("num_decay_steps")
                if decay is None:
                    decay = int(total * scheduler_config.get("decay_fraction", 0.2))
                stable = scheduler_config.get("num_stable_steps")
                if stable is None:
                    stable = max(0, total - num_warmup_steps - decay)
                scheduler_specific_kwargs.update(
                    num_stable_steps=int(stable), num_decay_steps=int(decay),
                    min_lr_ratio=scheduler_config.get("min_lr_ratio", 0.1),
                )

            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_config.get("interval", "step"),
                "frequency": scheduler_config.get("frequency", 1),
            },
        }

    # ----------------------------------------------------------------------------- EMA
    #
    # Generative models are routinely several points better when evaluated on an
    # exponential moving average of the weights than on the online ones, so the EMA is not
    # a nicety here -- it is part of the model.
    #
    # It lives on the LightningModule rather than in a Callback on purpose: Lightning skips
    # `_call_callbacks_on_save_checkpoint` entirely when `save_weights_only=True`
    # (checkpoint_connector.dump_checkpoint), which is the setting every Poc2Mol experiment
    # config uses -- a callback would have silently written online weights instead.
    #
    # Contract:
    #   * validation runs on the EMA weights, and
    #   * `checkpoint["state_dict"]` holds the EMA weights,
    # so the metric a checkpoint is selected on is measured on the weights that checkpoint
    # contains. The online weights ride along in `checkpoint["raw_state_dict"]` whenever the
    # checkpoint is a full one, which makes a resume exact.

    def _ema_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.is_floating_point()]

    @torch.no_grad()
    def _ema_update(self) -> None:
        params = dict(self.named_parameters())
        if self._ema_shadow is None:
            self._ema_shadow = {
                name: params[name].detach().clone().float()
                for name in self._ema_parameter_names()
            }
            return
        step = int(self.global_step)
        # Ramp the decay in: at step 0 an EMA with decay 0.999 is 99.9% random init, and it
        # would take ~10k steps to forget that.
        decay = self.ema_decay
        if self.ema_warmup_steps > 0:
            decay = min(decay, (1.0 + step) / (float(self.ema_warmup_steps) + step))
        for name, shadow in self._ema_shadow.items():
            shadow.lerp_(params[name].detach().float(), 1.0 - decay)

    def _ema_swap_in(self) -> None:
        if self._ema_shadow is None:
            return
        params = dict(self.named_parameters())
        self._ema_backup = {
            name: params[name].detach().clone() for name in self._ema_shadow
        }
        with torch.no_grad():
            for name, shadow in self._ema_shadow.items():
                params[name].copy_(shadow.to(params[name].dtype))

    def _ema_swap_out(self) -> None:
        if self._ema_backup is None:
            return
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name, backup in self._ema_backup.items():
                params[name].copy_(backup)
        self._ema_backup = None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_decay > 0:
            self._ema_update()

    # NOTE: the EMA swap-in lives in `on_validation_epoch_start` above, next to the
    # sampling budget reset, so there is exactly one definition of that hook.

    def on_validation_epoch_end(self):
        self._ema_swap_out()

    def on_train_start(self):
        # Resume path: Lightning has already loaded the EMA weights (they are the saved
        # state_dict), so the shadow is seeded from them and the online weights are put
        # back where they belong.
        if self.ema_decay > 0 and self._ema_shadow is None:
            params = dict(self.named_parameters())
            self._ema_shadow = {
                name: params[name].detach().clone().float()
                for name in self._ema_parameter_names()
            }
        if self._pending_raw_state is not None:
            missing, unexpected = self.load_state_dict(self._pending_raw_state, strict=False)
            if missing or unexpected:
                print(f"raw_state_dict restore: missing={missing[:4]} unexpected={unexpected[:4]}")
            self._pending_raw_state = None

    def on_save_checkpoint(self, checkpoint):
        if self.ema_decay <= 0 or self._ema_shadow is None:
            return
        state = checkpoint.get("state_dict")
        if state is None:
            return
        # A full checkpoint carries the optimiser, so it is meant for resuming: keep the
        # online weights alongside. A weights-only checkpoint is meant for consumption, and
        # doubling its size for a resume it cannot do anyway would be waste.
        if "optimizer_states" in checkpoint:
            checkpoint["raw_state_dict"] = {k: v for k, v in state.items()}
        for name, shadow in self._ema_shadow.items():
            if name in state:
                state[name] = shadow.detach().to(state[name].dtype).cpu()

    def on_load_checkpoint(self, checkpoint):
        """Optionally discard optimiser/scheduler state, as the regression model does."""
        if self.override_optimizer_on_load:
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
        raw = checkpoint.pop("raw_state_dict", None)
        if raw is not None and not self.override_optimizer_on_load:
            self._pending_raw_state = raw
