"""End-to-end Poc2Mol -> Vox2Smiles, with supervision at BOTH the voxel and the token layer.

The three-stage pipeline trains Poc2Mol on a voxel reconstruction loss, freezes it, and
trains the decoder on whatever density it emits. Poc2Mol therefore optimises for the
conditional mean of the ligand density, which is what maximises Dice -- not for whatever
representation the decoder can actually read. This module closes that loop: the
language-modelling loss backpropagates through the sigmoid, into the 3D U-Net.

    protein voxels --[Poc2Mol U-Net]--> ligand density --[ViT+GPT2]--> SMILES
                                             |                            |
                                          BCE+Dice                   cross-entropy
                                             \\__________ + w * __________/

Why keep the voxel loss at all
------------------------------
With only an LM loss there is nothing anchoring the intermediate representation. The U-Net
is free to stop being a density model and emit whatever pattern the decoder finds
convenient -- a private code, not a ligand. That would still be a working generator, but it
throws away the interpretable middle layer the whole project is built on, it makes
`density_diagnostics.py` and every Dice number meaningless, and it is the failure mode most
likely to look like progress on the training curve while the model quietly overfits 9,872
HiQBind clusters. The BCE+Dice term is the leash; ``voxel_loss_weight`` is its length, and
it is the single most important knob here.

Subclassing VoxToSmilesModel rather than composing
--------------------------------------------------
``val/likelihood_auc_znorm`` is the metric this whole branch is judged on, and it is
computed by a fair amount of machinery -- a pocket x candidate matrix accumulated across
the epoch, all-gathered across ranks, z-normalised by column. Reimplementing it would put
the new number at risk of not being comparable to the 0.7522 baseline. Inheriting it means
the metric is computed by literally the same code; the only difference is where
``pixel_values`` came from.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.data.common.protein_channels import assemble_decoder_input
from src.data.vox2smiles.poc2mol_inference import poc2mol_loss_per_sample
from src.models.vox2smiles import VoxToSmilesModel


def load_poc2mol_weights(poc2mol_model, ckpt_path: str) -> None:
    """Load a standalone Poc2Mol checkpoint into ``poc2mol_model``.

    Poc2Mol checkpoints are Lightning ones whose state_dict keys are ``model.<...>`` (the
    LightningModule wraps the U-Net as ``self.model``), so the prefix comes off before the
    tensors go into the U-Net. Strict, deliberately: a silent mismatch here is a wrong
    channel count, which is exactly the failure the 11ch/9ch confusion produces and which
    otherwise shows up only as an unexplained drop in Dice.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    stripped = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    poc2mol_model.model.load_state_dict(stripped, strict=True)


class EndToEndPoc2Smiles(VoxToSmilesModel):
    """Poc2Mol and Vox2Smiles as a single differentiable model.

    Args:
        config: the decoder config, exactly as ``VoxToSmilesModel`` takes it.
        poc2mol_model: an instantiated ``Poc2Mol``. Owned by this module, so its parameters
            are in ``self.parameters()`` and its weights ride in the checkpoint.
        poc2mol_ckpt_path: standalone Poc2Mol checkpoint to initialise from. Loaded here
            rather than through ``init_weights_from`` because that path is already carrying
            the decoder's stage-1 checkpoint, and the two upstreams live in different files.
        voxel_loss_weight: multiplier on the BCE+Dice term. 1.0 reproduces Poc2Mol's own
            training objective exactly, so the LM loss is a pure addition to it; smaller
            values let the density drift towards whatever the decoder wants.
        poc2mol_lr: learning rate for the upstream. Defaults well below the decoder's:
            Poc2Mol arrives pretrained and 576 epochs deep, and the LM gradient reaching it
            is a signal it has never seen before.
        lm_grad_to_poc2mol: set False to sever the LM gradient at the Poc2Mol boundary while
            keeping everything else identical. This is the control arm -- it isolates "does
            the end-to-end gradient help?" from "does merely un-freezing Poc2Mol help?".
        poc2mol_warmup_steps: keep Poc2Mol frozen for this many optimiser steps. The decoder
            is being asked to read a density whose distribution is about to start moving; a
            brief hold lets it settle first.
    """

    def __init__(
        self,
        config,
        poc2mol_model,
        poc2mol_ckpt_path: Optional[str] = None,
        voxel_loss_weight: float = 1.0,
        poc2mol_lr: float = 1e-5,
        poc2mol_weight_decay: float = 0.05,
        lm_grad_to_poc2mol: bool = True,
        poc2mol_warmup_steps: int = 0,
        poc2mol_val_chunk: int = 32,
        inject_protein: bool = False,
        override_optimizer_on_load: bool = False,
        visualise_val: bool = True,
        n_samples_for_validity_testing: int = 30,
    ) -> None:
        super().__init__(
            config,
            override_optimizer_on_load=override_optimizer_on_load,
            visualise_val=visualise_val,
            n_samples_for_validity_testing=n_samples_for_validity_testing,
        )
        # The parent's save_hyperparameters inspects THIS frame, not its own, so it stored
        # `poc2mol_model` -- which would pickle a 117M-parameter module into every
        # checkpoint's hyper_parameters, alongside the state_dict it is already in. A second
        # save_hyperparameters(ignore=...) does not undo that: it merges into the existing
        # dict rather than replacing it, so the key survives. Remove it explicitly.
        self.save_hyperparameters(ignore=["poc2mol_model"], logger=False)
        self.hparams.pop("poc2mol_model", None)

        self.poc2mol = poc2mol_model
        if poc2mol_ckpt_path:
            load_poc2mol_weights(self.poc2mol, poc2mol_ckpt_path)

        self.voxel_loss_weight = float(voxel_loss_weight)
        self.poc2mol_lr = float(poc2mol_lr)
        self.poc2mol_weight_decay = float(poc2mol_weight_decay)
        self.lm_grad_to_poc2mol = bool(lm_grad_to_poc2mol)
        self.poc2mol_warmup_steps = int(poc2mol_warmup_steps)
        self.poc2mol_val_chunk = int(poc2mol_val_chunk)
        self.inject_protein = bool(inject_protein)

        # `_bind` in the frozen path hard-casts Poc2Mol to bfloat16, which is fine for
        # inference and wrong here: the master weights must stay fp32 or the optimiser
        # update is quantised away at these learning rates. Compute still runs in bf16 --
        # the trainer is on `bf16-mixed`, so autocast handles the forward.
        self.poc2mol.float()

    # ------------------------------------------------------------------ forward

    @property
    def _poc2mol_active(self) -> bool:
        """Whether Poc2Mol takes gradient this step."""
        return self.global_step >= self.poc2mol_warmup_steps

    def build_pixel_values(self, batch, training: bool):
        """Run Poc2Mol and assemble the decoder's input, keeping the graph intact.

        Returns ``(pixel_values, info)``. ``info`` carries the voxel loss and the per-sample
        diagnostics the training/validation logging needs.
        """
        protein = batch["protein_voxels"]
        true_ligand = batch["ligand_voxels"]
        has_pocket = batch["has_pocket"].to(protein.device)

        index = has_pocket.nonzero(as_tuple=True)[0]
        # A batch with no pocket rows would leave every Poc2Mol parameter out of the graph,
        # which plain DDP rejects outright ("expected to have finished reduction..."). At
        # batch 64 and prob_poc2mol 0.5 that is a ~1e-19 event, but the failure mode is a
        # hard crash hours in, so it is handled rather than assumed away: run one row and
        # weight its contribution to zero, which keeps the parameters in the graph.
        degenerate = index.numel() == 0
        if degenerate:
            index = torch.zeros(1, dtype=torch.long, device=protein.device)

        # Dropout only belongs on if the upstream is actually learning. At poc2mol_lr 0 this
        # module reproduces the frozen stage-3 pipeline, which ran Poc2Mol under .eval();
        # leaving dropout active there would inject noise into the density the decoder sees
        # and make the baseline arm quietly not-the-baseline.
        self.poc2mol.train(training and self.poc2mol_lr > 0)
        with torch.set_grad_enabled(training):
            if training:
                logits = self.poc2mol.model(x=protein[index])
            else:
                # Validation batches are far larger than training ones (val_batch_size is
                # 200 against a training batch of 32), and the U-Net's activations scale
                # with the row count even under no_grad. Chunking bounds the peak so a
                # validation pass cannot OOM a run that trains perfectly well.
                logits = torch.cat([
                    self.poc2mol.model(x=protein[index[start:start + self.poc2mol_val_chunk]])
                    for start in range(0, index.numel(), self.poc2mol_val_chunk)
                ])
            predicted = torch.sigmoid(logits)
            # Pooled BCE+Dice, i.e. the criterion Poc2Mol was actually trained with
            # (`compute_per_channel_dice` flattens to (C, N*spatial), so the Dice term is
            # pooled across the batch). Using a per-sample variant here would be a
            # different objective from the one that produced the 0.5027 baseline.
            components = self.poc2mol.loss(logits, true_ligand[index].to(logits.dtype))
            voxel_loss = sum(components.values())
            # Zero the TERM, never the graph. Detaching Poc2Mol during warmup (or on a
            # degenerate all-ZINC batch) would leave its parameters out of the backward
            # pass entirely, and plain DDP treats an unreduced parameter as a hard error --
            # a crash that would only appear once, hours in. Multiplying by zero keeps
            # every parameter in the graph, gives it a zero gradient, and so freezes it
            # exactly as intended.
            if degenerate or not self._poc2mol_active:
                voxel_loss = voxel_loss * 0.0

        with torch.no_grad():
            per_sample = poc2mol_loss_per_sample(logits.detach(), true_ligand[index])

        decoder_ligand = true_ligand.to(predicted.dtype)
        if not degenerate:
            lm_reaches_upstream = self.lm_grad_to_poc2mol and self._poc2mol_active
            upstream = predicted if lm_reaches_upstream else predicted.detach()
            decoder_ligand = decoder_ligand.index_copy(0, index, upstream)

        pixel_values = assemble_decoder_input(
            decoder_ligand, protein, has_pocket, self.inject_protein,
            mask_probability=0.0, training=training,
        )
        # The decoder's weights are bf16 (VoxToSmilesModel casts them in __init__), so hand
        # it bf16 activations. The cast is differentiable; the gradient returns to Poc2Mol's
        # fp32 parameters through it.
        pixel_values = pixel_values.to(next(self.model.parameters()).dtype)

        # Same convention as Poc2MolInferenceBuilder: -1 marks "not a Poc2Mol row", which is
        # what the inherited logging in VoxToSmilesModel.training_step expects.
        loss_column = torch.full(
            (protein.shape[0],), -1.0, device=protein.device, dtype=per_sample.dtype
        )
        if not degenerate:
            loss_column = loss_column.index_copy(0, index, per_sample)

        info = {
            "voxel_loss": voxel_loss,
            "voxel_components": components,
            "per_sample_voxel_loss": loss_column,
            "n_pocket_rows": 0 if degenerate else int(index.numel()),
            "predicted": predicted,
            "true_ligand": true_ligand[index],
            "degenerate": degenerate,
        }
        return pixel_values, info

    def _prepare(self, batch, training: bool):
        """Turn a raw voxel batch into one the inherited step methods understand."""
        pixel_values, info = self.build_pixel_values(batch, training=training)
        prepared = dict(batch)
        prepared["pixel_values"] = pixel_values
        prepared["poc2mol_loss"] = info["per_sample_voxel_loss"]
        return prepared, info

    # ------------------------------------------------------------------ training

    def training_step(self, batch, batch_idx):
        prepared, info = self._prepare(batch, training=True)
        # The parent computes the LM loss and logs train/loss, train/accuracy,
        # train/poc2mol_voxel_loss and the per-source LM breakdown, all unchanged.
        lm_loss = super().training_step(prepared, batch_idx)

        voxel_loss = info["voxel_loss"]
        total = lm_loss + self.voxel_loss_weight * voxel_loss

        n = info["n_pocket_rows"] or 1
        self.log("train/lm_loss", lm_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/e2e_voxel_loss", voxel_loss, on_step=True, on_epoch=False,
                 prog_bar=True, batch_size=n)
        self.log("train/total_loss", total, on_step=True, on_epoch=False)
        for name, value in info["voxel_components"].items():
            self.log(f"train/poc2mol/{name}", value, on_step=True, on_epoch=False,
                     batch_size=n)
        self.log("train/poc2mol/active", float(self._poc2mol_active),
                 on_step=True, on_epoch=False)
        return total

    # ---------------------------------------------------------------- validation

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prepared, info = self._prepare(batch, training=False)

        # Voxel-level quality, on the pocket rows only. This is the tripwire for the density
        # collapsing while the LM loss keeps improving -- CLAUDE.md's warning that with only
        # an LM term Poc2Mol stops being a density model. Reported as soft Dice on the
        # pooled grid, i.e. the 0.5027 yardstick's definition.
        if not info["degenerate"] and self._val_split(dataloader_idx) == "poc2mol":
            with torch.no_grad():
                predicted = info["predicted"].float()
                truth = info["true_ligand"].float()
                # Per-sample soft Dice pooled over channels AND space -- byte-for-byte the
                # definition in density_diagnostics.pooled_soft_dice, so this number is
                # directly comparable to the 0.5027 yardstick and to anything that script
                # prints. The definition is NOT interchangeable with the obvious
                # alternatives: on one untouched batch of this very checkpoint it reads
                #     0.467  this one (per-sample, pooled over channels+space)
                #     0.311  per-channel Dice pooled across the batch, averaged
                #     0.194  per (sample, channel), averaged
                # The last two are dragged down by the ~5 of 11 channels a typical ligand
                # leaves empty, each of which contributes an unavoidable zero (§3c).
                dims = tuple(range(1, predicted.dim()))
                intersect = 2.0 * (predicted * truth).sum(dims)
                denominator = (predicted * predicted).sum(dims) + (truth * truth).sum(dims)
                dice = (intersect / denominator.clamp(min=1e-8)).mean()
                n = info["n_pocket_rows"]
                self.log("val/poc2mol/dice", dice, on_step=False, on_epoch=True,
                         sync_dist=True, batch_size=n)
                self.log("val/poc2mol/voxel_loss", info["voxel_loss"], on_step=False,
                         on_epoch=True, sync_dist=True, prog_bar=True, batch_size=n)
                total_mass = predicted.sum().clamp(min=1e-6)
                self.log("val/poc2mol/emission_ratio",
                         total_mass / truth.sum().clamp(min=1e-6),
                         on_step=False, on_epoch=True, sync_dist=True, batch_size=n)
                self.log("val/poc2mol/on_target",
                         (predicted * (truth > 0.05)).sum() / total_mass,
                         on_step=False, on_epoch=True, sync_dist=True, batch_size=n)

        return super().validation_step(prepared, batch_idx, dataloader_idx)

    # ---------------------------------------------------------------- optimisers

    def configure_optimizers(self):
        """Two parameter groups: the pretrained upstream gets the smaller learning rate.

        The scheduler multiplies both groups by the same factor, so the ratio between them
        is fixed by construction rather than drifting over the run.
        """
        decoder_wd = float(getattr(self.hparams.config, "weight_decay", 0.01))
        poc2mol_parameters = list(self.poc2mol.parameters())
        poc2mol_ids = {id(p) for p in poc2mol_parameters}
        decoder_parameters = [p for p in self.parameters() if id(p) not in poc2mol_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": decoder_parameters, "lr": self.hparams.config.lr,
                 "weight_decay": decoder_wd, "name": "decoder"},
                {"params": poc2mol_parameters, "lr": self.poc2mol_lr,
                 "weight_decay": self.poc2mol_weight_decay, "name": "poc2mol"},
            ],
            lr=self.hparams.config.lr,
            weight_decay=decoder_wd,
        )

        # Scheduler construction is identical to the parent's; only the optimizer differs,
        # so it is built here from the same config block rather than duplicated.
        return self._attach_scheduler(optimizer)

    def _attach_scheduler(self, optimizer):
        from transformers.optimization import get_scheduler

        scheduler_config = getattr(self.hparams.config, "scheduler", {})
        scheduler_type = scheduler_config.get("type", "step")

        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 100),
                gamma=scheduler_config.get("gamma", 0.997),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        num_training_steps = scheduler_config.get(
            "num_training_steps", self.trainer.estimated_stepping_batches
        )
        num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
        if isinstance(num_warmup_steps, float) and 0 <= num_warmup_steps < 1:
            num_warmup_steps = int(num_training_steps * num_warmup_steps)

        scheduler_specific_kwargs = {}
        if scheduler_type == "cosine_with_restarts":
            scheduler_specific_kwargs["num_cycles"] = scheduler_config.get("num_cycles", 1)
        elif scheduler_type == "cosine_with_min_lr":
            scheduler_specific_kwargs["min_lr_rate"] = scheduler_config.get("min_lr_rate", 0.1)
        elif scheduler_type == "warmup_stable_decay":
            scheduler_specific_kwargs["num_stable_steps"] = scheduler_config["num_stable_steps"]
            scheduler_specific_kwargs["num_decay_steps"] = scheduler_config["num_decay_steps"]
            scheduler_specific_kwargs["min_lr_ratio"] = scheduler_config.get("min_lr_ratio", 0.1)

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
