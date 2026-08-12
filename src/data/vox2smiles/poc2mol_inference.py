"""Run the frozen Poc2Mol as a batch transform, on the rank's own device.

This used to live inside ``Poc2MolOutputDataset.__getitem__``: a Poc2Mol model was placed
on ``cuda`` in the dataset constructor and invoked one sample at a time. Under DDP that
puts a copy of the model on ``cuda:0`` for every rank, and it forces ``num_workers: 0``
because CUDA tensors cannot be produced in a forked worker. Running it here instead --
from ``on_after_batch_transfer``, batched, under ``no_grad`` -- fixes both, and turns 64
separate forward passes into one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.common.voxelization.batched import BatchedVoxelizer
from src.data.common.voxelization.config import resolve_dtype
from src.data.common.protein_channels import assemble_decoder_input


def poc2mol_loss_floor(target, alpha=1.0, beta=1.0, epsilon=1e-8):
    """The loss a *perfect* prediction of ``target`` would score.

    The Dice term averages over all 9 ligand channels, but a typical ligand occupies only
    ~3.6 of them, and an all-zero target channel gives ``dice = 2*0/(sum(pred^2)+0) = 0``
    however good the prediction is. So the loss carries a large additive constant that
    depends on the ligand's element composition, not on model quality: measured floor is
    0.6256 on average but ranges 0.473-0.802 per sample (CLAUDE.md §3c).

    Subtracting this makes the loss comparable across ligands. Without it, an absolute
    quality threshold rejects halogen-free ligands however well reconstructed, and accepts
    chemically rich ones reconstructed badly.
    """
    perfect = torch.where(
        target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0)
    )
    return poc2mol_loss_per_sample(perfect, target, alpha=alpha, beta=beta, epsilon=epsilon)


def poc2mol_loss_per_sample(logits, target, alpha=1.0, beta=1.0, epsilon=1e-8):
    """Per-sample BCEDice loss.

    The training criterion (``compute_per_channel_dice``) flattens to ``(C, N*spatial)``,
    pooling the Dice term across the batch. Evaluated one sample at a time -- which is what
    the old per-item code did, and what the ``max_poc2mol_loss`` threshold was calibrated
    against -- that reduces to per-(sample, channel) Dice averaged over channels. This
    reproduces that, vectorised over the batch.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    ).mean(dim=(1, 2, 3, 4))

    probs = torch.sigmoid(logits.float())
    truth = target.float()
    spatial = (2, 3, 4)
    intersect = (probs * truth).sum(spatial)
    denominator = (probs * probs).sum(spatial) + (truth * truth).sum(spatial)
    dice_coefficient = 2 * (intersect / denominator.clamp(min=epsilon))
    dice = 1.0 - dice_coefficient.mean(dim=1)

    return alpha * bce + beta * dice


class Poc2MolInferenceBuilder:
    """Voxelise a mixed batch, then replace ligand voxels with Poc2Mol predictions.

    Samples flagged ``needs_poc2mol`` carry protein atoms; their ligand channels are
    replaced by what Poc2Mol predicts from the protein. Ligand-only samples (ZINC) keep
    their ground-truth ligand voxels. Both arrive with the same channel layout so they can
    share a batch.
    """

    def __init__(
        self,
        voxel_config,
        poc2mol_model=None,
        ckpt_path: str = None,
        n_protein_channels: int = 4,
        max_poc2mol_loss: float = None,
        quality_filter: str = "none",
        loss_alpha: float = 1.0,
        loss_beta: float = 1.0,
        pad_token_id: int = 0,
        inject_protein: bool = False,
        protein_mask_probability: float = 0.0,
        predicted_ligand_probability: float = 1.0,
        predicted_ramp_start_step: int = 0,
        predicted_ramp_end_step: int = 0,
        compute_dtype: torch.dtype = torch.float32,
        generative_readout: str = "one_step",
        generative_draws: int = 4,
        generative_guidance: float = 3.0,
        generative_steps: int = 50,
    ):
        self.voxel_config = voxel_config
        self.pad_token_id = pad_token_id
        self.poc2mol_model = poc2mol_model
        self.ckpt_path = ckpt_path
        self.n_protein_channels = n_protein_channels
        self.max_poc2mol_loss = max_poc2mol_loss
        # "none"     -- train on every Poc2Mol output. At inference the decoder must handle
        #               whatever Poc2Mol produces, so filtering training down to the easy
        #               reconstructions is a train/serve mismatch.
        # "relative" -- threshold the EXCESS over each sample's own loss floor, which is
        #               what "reconstruction quality" actually means here.
        # "absolute" -- the legacy behaviour; gates on ligand composition rather than
        #               quality (CLAUDE.md §3c). Kept only for reproducing old runs.
        if quality_filter not in {"none", "relative", "absolute"}:
            raise ValueError(f"unknown quality_filter {quality_filter!r}")
        self.quality_filter = quality_filter
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.inject_protein = inject_protein
        self.protein_mask_probability = protein_mask_probability
        # Three-stage curriculum. A complex sample can supply either its TRUE ligand voxels
        # or Poc2Mol's prediction; this is the probability of the latter.
        #   stage 2: 0.0  -- learn to read the protein channels against clean ligand density
        #   stage 3: ramp 0 -> 1 over [ramp_start_step, ramp_end_step], so the decoder meets
        #            Poc2Mol's error distribution gradually rather than as a step change.
        # Separating the two means "can it use the protein?" is learned before "can it cope
        # with a noisy upstream?", instead of confounding them.
        self.predicted_ligand_probability = predicted_ligand_probability
        self.predicted_ramp_start_step = predicted_ramp_start_step
        self.predicted_ramp_end_step = predicted_ramp_end_step
        # How a GENERATIVE Poc2Mol is read out. Ignored for the regression model.
        #
        #   one_step  E[ligand | pocket] from a single network evaluation per draw. This is
        #             the quantity the regression model is trained to emit, it scores far
        #             better on Dice than a sample (0.372 vs 0.231 measured 2026-08-11), and
        #             it costs `generative_draws` forwards instead of ~100 for a Heun
        #             trajectory -- which is what makes it affordable INSIDE a training loop.
        #   sample    a genuine draw from p(ligand | pocket). Use for the multi-hypothesis
        #             evaluation, where several decoded draws are aggregated per pocket.
        if generative_readout not in {"one_step", "sample", "mean_of_k"}:
            raise ValueError(f"unknown generative_readout {generative_readout!r}")
        self.generative_readout = generative_readout
        self.generative_draws = generative_draws
        self.generative_guidance = generative_guidance
        self.generative_steps = generative_steps
        self.compute_dtype = compute_dtype

        self._voxelizer = None
        self._device = None
        self._weights_loaded = False

    # ------------------------------------------------------------------ setup

    def _bind(self, device):
        """Move the voxeliser and the frozen model onto the batch's device, once."""
        if self._device == device:
            return
        self._voxelizer = BatchedVoxelizer(
            self.voxel_config,
            compute_dtype=self.compute_dtype,
            cutoff_ratio=self.voxel_config.get("voxel_cutoff_ratio", 2.0),
            aggregation=self.voxel_config.get("voxel_aggregation", "max"),
            radius_scale=self.voxel_config.get("voxel_radius_scale", 1.0),
        ).to(device)

        if self.poc2mol_model is not None:
            if not self._weights_loaded and self.ckpt_path is not None:
                self._load_weights()
            self.poc2mol_model = self.poc2mol_model.to(resolve_dtype(self.voxel_config.dtype)).to(device)
            self.poc2mol_model.eval()
            for parameter in self.poc2mol_model.parameters():
                parameter.requires_grad_(False)

        self._device = device

    def _load_weights(self):
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.poc2mol_model.model.load_state_dict(state_dict)
        else:
            self.poc2mol_model.load_state_dict(checkpoint)
        self._weights_loaded = True

    # ------------------------------------------------------------- transform

    def predicted_fraction(self, global_step: int) -> float:
        """Probability of using Poc2Mol's prediction instead of the true ligand voxels."""
        target = self.predicted_ligand_probability
        if self.predicted_ramp_end_step <= self.predicted_ramp_start_step:
            return target
        span = self.predicted_ramp_end_step - self.predicted_ramp_start_step
        progress = (global_step - self.predicted_ramp_start_step) / span
        return float(target * min(max(progress, 0.0), 1.0))

    def __call__(self, batch, apply_quality_filter: bool = True, training: bool = True,
                 global_step: int = 0):
        if "atom_xyz" not in batch:
            return batch

        self._bind(batch["atom_xyz"].device)

        grid = self._voxelizer(
            batch["atom_xyz"],
            batch["atom_radius"],
            batch["atom_slot"],
            batch["batch_size"],
            batch["n_channels"],
        )
        protein = grid[:, : self.n_protein_channels]
        ligand = grid[:, self.n_protein_channels :]

        out = {k: v for k, v in batch.items() if not k.startswith("atom_")}
        out.pop("n_channels", None)
        out.pop("batch_size", None)

        needs = out.pop("needs_poc2mol", None)
        if needs is None or not bool(needs.any()) or self.poc2mol_model is None:
            out["pixel_values"] = assemble_decoder_input(
                ligand, protein, needs, self.inject_protein,
                self.protein_mask_probability, training,
            )
            out["protein_voxels"] = protein
            out["poc2mol_loss"] = None
            # Per-sample "this row carries a pocket". Kept on the batch so training_step can
            # split the language-modelling loss by source; without it the only available
            # split is poc2mol_loss > 0, which cannot separate a ligand-only row from a
            # complex row that happens to be using its TRUE ligand voxels.
            out["has_pocket"] = needs
            return out

        # One forward pass for the whole batch. Running it on every sample and discarding
        # the ligand-only rows is cheaper than the gather/scatter, and keeps shapes static.
        with torch.no_grad():
            if getattr(self.poc2mol_model, "is_generative", False):
                # Flow-matching Poc2Mol: there is no logit map, so the density is produced
                # by the configured readout and pushed back through a logit to be scored on
                # the BCEDice scale the quality filter and `poc2mol_loss` are calibrated on.
                # The clamp keeps a saturated voxel from an infinite BCE term.
                if self.generative_readout == "sample":
                    predicted = self.poc2mol_model.sample(
                        protein=protein, n_steps=self.generative_steps,
                        guidance_scale=self.generative_guidance)
                else:
                    predicted = self.poc2mol_model.predict_expected(
                        protein=protein, mode=self.generative_readout,
                        n_draws=self.generative_draws,
                        guidance_scale=self.generative_guidance,
                        n_steps=self.generative_steps)
                predicted = predicted.to(ligand.dtype)
                predicted_logits = torch.logit(
                    predicted.float().clamp(1e-4, 1.0 - 1e-4)
                )
            else:
                predicted_logits = self.poc2mol_model.model(x=protein)
                predicted = torch.sigmoid(predicted_logits)
            losses = poc2mol_loss_per_sample(
                predicted_logits, ligand, alpha=self.loss_alpha, beta=self.loss_beta
            )

        # Which complex samples get the prediction rather than the ground-truth ligand.
        # Validation always uses the prediction: that is the deployed condition, and a
        # curriculum-weighted validation metric would drift as the ramp advanced.
        fraction = self.predicted_fraction(global_step) if training else 1.0
        use_pred = needs
        if fraction < 1.0:
            draw = torch.rand(len(needs), device=needs.device)
            use_pred = needs & (draw < fraction)

        select = use_pred.view(-1, *([1] * (ligand.dim() - 1)))
        decoder_ligand = torch.where(select, predicted, ligand)
        # Zero the pocket only for rows that HAVE no pocket (ZINC), i.e. mask on `needs`.
        # This previously masked on `select` -- the rows using Poc2Mol's prediction -- which
        # in stage 2 (predicted_ligand_probability = 0) is no rows at all, so the protein
        # channels were zeroed for every training sample of the stage whose whole purpose is
        # learning to read them. Validation uses fraction = 1.0 and so still showed the
        # pocket, making it a train/serve mismatch that the val metric could not reveal.
        has_pocket = needs.view(-1, *([1] * (protein.dim() - 1)))
        protein = torch.where(has_pocket, protein, torch.zeros_like(protein))
        # Experiment 1: the decoder sees the pocket alongside the predicted ligand. Only
        # samples that actually carry protein density (`needs`) can contribute it.
        out["pixel_values"] = assemble_decoder_input(
            decoder_ligand, protein, needs, self.inject_protein,
            self.protein_mask_probability, training,
        )
        out["protein_voxels"] = protein
        out["has_pocket"] = needs
        # -1 marks "not a Poc2Mol sample", matching the convention the logging in
        # VoxToSmilesModel.training_step already expects.
        out["poc2mol_loss"] = torch.where(use_pred, losses, torch.full_like(losses, -1.0))
        out["predicted_fraction"] = torch.tensor(float(fraction))

        if (
            apply_quality_filter
            and self.quality_filter != "none"
            and self.max_poc2mol_loss is not None
        ):
            if self.quality_filter == "relative":
                floor = poc2mol_loss_floor(
                    ligand, alpha=self.loss_alpha, beta=self.loss_beta
                )
                excess = losses - floor
                out["poc2mol_excess_loss"] = torch.where(
                    needs, excess, torch.full_like(excess, -1.0)
                )
                rejected = use_pred & (excess > self.max_poc2mol_loss)
            else:
                rejected = use_pred & (losses > self.max_poc2mol_loss)
            out["poc2mol_rejected"] = rejected
            if bool(rejected.any()):
                # Previously a rejected sample was swapped for a ZINC one at __getitem__
                # time, which is impossible now that the loss is only known post-batching.
                # Masking the labels to -100 instead drops those samples from the
                # cross-entropy while leaving HuggingFace's token-count normalisation over
                # the surviving samples intact.
                if bool(rejected.all()):
                    # Never hand back an all-masked batch: the loss would be 0/0.
                    rejected = rejected & (
                        torch.arange(len(rejected), device=rejected.device) != int(losses.argmin())
                    )
                    out["poc2mol_rejected"] = rejected
                # Blank the labels with the *pad* token, not -100. VoxToSmilesModel.forward
                # derives decoder_attention_mask from the raw labels and HuggingFace shifts
                # them right into decoder_input_ids, so a literal -100 would reach the
                # embedding lookup. Pad flows through that path correctly and is mapped to
                # -100 immediately afterwards, so these rows contribute no loss and do not
                # count towards its token normalisation.
                labels = out["input_ids"].clone()
                labels[rejected] = self.pad_token_id
                out["input_ids"] = labels
        else:
            out["poc2mol_rejected"] = torch.zeros_like(needs)

        return out
