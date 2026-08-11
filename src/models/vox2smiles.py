import torch
import numpy as np
import copy
from lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import (VisionEncoderDecoderModel,
                          VisionEncoderDecoderConfig,
                          ViTConfig,
                          GPT2Config,
                          get_scheduler,
                          SchedulerType)
import wandb
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image, ImageDraw, ImageFont
from src.models.modeling_vit_3d import ViTModel3D
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.utils.metrics import (
    accuracy_from_outputs,
    calculate_validity,
    calculate_novelty,
    calculate_uniqueness,
    calculate_exact_match,
    calculate_paired_similarity,
    _blocked_rdkit_logs,
)
from src.utils.likelihood_eval import evaluate_likelihood_ranking


class VoxToSmilesModel(LightningModule):
    # "combined" is the pool of the other two, not a third dataset.
    VAL_SPLITS = ("zinc", "poc2mol", "combined")
    # loss/accuracy are teacher-forced; validity/exact_match/tanimoto come from free-running
    # generation and are the ones that actually reflect deployed behaviour.
    VAL_METRICS = ("loss", "accuracy", "validity", "exact_match", "tanimoto")

    def __init__(
        self,
        config,
        override_optimizer_on_load: bool = False,
        visualise_val: bool = True,
        n_samples_for_validity_testing: int = 30,
    ) -> None:
        super().__init__()
        if "torch_dtype" not in config:
            config.torch_dtype = torch.bfloat16
        self.save_hyperparameters(logger=False)

        self.tokenizer = build_smiles_tokenizer()

        vit_config = ViTConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            qkv_bias=config.qkv_bias,
            encoder_stride=config.encoder_stride,
            torch_dtype=config.torch_dtype,
        )

        # These must be the tokens the model is actually TRAINED on. They previously pointed
        # at [CLS]=1 and [SEP]=4, while training uses [BOS]=2 and [EOS]=3 -- so `generate`
        # waited for a [SEP] the model never emits, ran to max_length=200 every time, and
        # appended garbage after a perfectly good SMILES. Truncating at the real [EOS]
        # recovered a valid molecule in every sample checked, so this depressed validity,
        # uniqueness and novelty across training, inference and evaluation alike.
        # The loss path masks labels itself and never reads these, so training was unaffected
        # and no retraining is needed: the config is rebuilt from the tokenizer on load, which
        # repairs existing checkpoints retroactively.
        gpt2_config = GPT2Config(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=len(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id
        )

        # add_pooling_layer=False: VisionEncoderDecoderModel consumes the encoder's
        # last_hidden_state and never touches pooler_output, so a pooler's weights would
        # receive no gradient. Under DDP that is a hard error ("expected to have finished
        # reduction ... marking parameters ready only once"), avoidable otherwise only by
        # find_unused_parameters=True, which costs a full extra parameter sweep per step.
        # The pooler was dead weight before, so dropping it changes no computation.
        encoder = ViTModel3D(vit_config, add_pooling_layer=False)
        
        # pad_token_id was [EOS]=3 here, which made `generate` treat the real end token as
        # padding. It also feeds shift_tokens_right, which fills -100 label positions with
        # pad -- harmless either way since decoder_attention_mask already zeroes them, so
        # correcting it does not perturb training.
        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=vit_config,
            decoder_config=gpt2_config,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.model = VisionEncoderDecoderModel(config=encoder_decoder_config, encoder=encoder)
        if vit_config.torch_dtype is not None:
            if not isinstance(vit_config.torch_dtype, torch.dtype):
                raise ValueError(f"Unsupported torch_dtype: {vit_config.torch_dtype}")
            else:
                self.model = self.model.to(vit_config.torch_dtype)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_acc = MeanMetric()

        # Validation is reported three ways so the two failure modes of the combined stage
        # are separable:
        #   zinc     -- ligand-only, ground-truth voxels. Detects catastrophic forgetting of
        #               the pretraining task while the model adapts to pockets.
        #   poc2mol  -- complexes conditioned on Poc2Mol's PREDICTED density. The deployed
        #               condition, and the thing the whole pipeline is judged on.
        #   combined -- the two pooled, i.e. the training distribution.
        # A single averaged number hides the trade-off entirely: the model can buy pocket
        # performance with ZINC ability and the combined metric barely moves.
        self.val_metrics = torch.nn.ModuleDict({
            f"{split}__{metric}": MeanMetric()
            for split in self.VAL_SPLITS
            for metric in self.VAL_METRICS
        })

        self.override_optimizer_on_load = override_optimizer_on_load
        self.train_sample_counter = 0
        self.train_poc2mol_sample_counter = 0
        # Accumulates the pocket x candidate likelihood matrix across the validation epoch.
        # It cannot be computed per batch: z-normalising a ligand's score needs that ligand
        # scored against MANY pockets, so the whole matrix has to exist first.
        self._likelihood_rows = []
        self.visualise_val = visualise_val
        self.n_samples_for_validity_testing = n_samples_for_validity_testing

    def forward(self, pixel_values, labels=None):
        if labels is None:
            raise ValueError("Labels are required for Vox2Smiles forward method.")

        decoder_attention_mask = (labels != self.tokenizer.pad_token_id).long()

        masked_labels = labels.clone()
        masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100

        return self.model(
            pixel_values=pixel_values,
            labels=masked_labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        outputs.logits = outputs.logits.detach()
        del outputs.encoder_last_hidden_state
        del outputs.past_key_values
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_sample_counter += len(batch['pixel_values'])
        self.log("train/n_samples", self.train_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
        if 'poc2mol_loss' in batch and batch['poc2mol_loss'] is not None:
            elements_with_loss_mask = batch['poc2mol_loss'] > 0
            self.train_poc2mol_sample_counter += elements_with_loss_mask.int().sum()
            self.log("train/proportion_from_poc2mol", self.train_poc2mol_sample_counter / self.train_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            masked_labels = labels.clone()
            masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100
            with torch.no_grad():
                accuracy = accuracy_from_outputs(outputs, masked_labels, start_ix=1, ignore_index=-100)
            if elements_with_loss_mask.int().sum() == 0:
                self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(labels))
            elif elements_with_loss_mask.int().sum() == len(labels):
                self.log("train/poc2mol_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(labels))
            self.log("train/n_poc2mol_samples", self.train_poc2mol_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            if elements_with_loss_mask.any():
                # VOXEL reconstruction quality of the frozen upstream (BCEDice between its
                # predicted ligand density and the true grid) -- NOT a language-modelling
                # loss. Named explicitly because `train/poc2mol_loss` sitting beside
                # `val/poc2mol/loss`, which IS a cross-entropy, was thoroughly confusing.
                # Scale check: this lives at ~0.80 against Poc2Mol's own 0.8039 and its
                # 0.6403 floor, while the decoder's CE is ~0.05.
                self.log("train/poc2mol_voxel_loss", batch['poc2mol_loss'][elements_with_loss_mask].mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=elements_with_loss_mask.int().sum())

        self._log_train_lm_loss_by_source(batch, outputs, labels)
        return loss

    def _log_train_lm_loss_by_source(self, batch, outputs, labels):
        """Language-modelling loss on the TRAINING data, split by where the sample came from.

        `val/poc2mol/loss` rising while `val/zinc/loss` falls is consistent with two very
        different stories: the decoder overfitting the small complex set, or it simply failing
        to learn the pocket task at all. Those are distinguished by the TRAINING loss on the
        same distribution -- if train/poc2mol_pred falls while val/poc2mol rises, it is
        memorisation; if both rise, the task itself is regressing.

        Three groups, because during the ramp a complex row may supply either ligand density:
          zinc         -- no pocket at all
          poc2mol_true -- pocket, ground-truth ligand voxels
          poc2mol_pred -- pocket, Poc2Mol's predicted density. Directly comparable to
                          val/poc2mol/loss, which always uses the prediction.
        """
        has_pocket = batch.get("has_pocket")
        if has_pocket is None:
            return

        with torch.no_grad():
            masked = labels.clone()
            masked[masked == self.tokenizer.pad_token_id] = -100
            # Same alignment as everywhere else: HF shifts decoder inputs internally, so
            # logits[:, i] predicts labels[:, i]; slice both from 1 to skip the BOS position.
            shift_logits = outputs.logits[:, 1:, :].float()
            shift_labels = masked[:, 1:]
            per_token = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.shape)
            keep = shift_labels != -100
            per_sample = (per_token * keep).sum(1) / keep.sum(1).clamp(min=1)

            has_pocket = has_pocket.to(per_sample.device).bool()
            poc2mol_loss = batch.get("poc2mol_loss")
            used_prediction = (
                (poc2mol_loss > 0).to(per_sample.device)
                if poc2mol_loss is not None
                else torch.zeros_like(has_pocket)
            )
            groups = {
                "zinc": ~has_pocket,
                "poc2mol_true": has_pocket & ~used_prediction,
                "poc2mol_pred": has_pocket & used_prediction,
            }
            for name, mask in groups.items():
                n = int(mask.sum())
                if n:
                    self.log(f"train/{name}/lm_loss", per_sample[mask].mean(),
                             on_step=True, on_epoch=False, batch_size=n)

    # ------------------------------------------------------------------ validation

    def _val_split(self, dataloader_idx: int) -> str:
        """Which reporting split this validation dataloader belongs to.

        Read off the datamodule rather than inferred from dataloader_idx, so reordering or
        adding a val dataset cannot silently relabel the metrics.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        kinds = getattr(datamodule, "val_dataset_kinds", None)
        if kinds and dataloader_idx < len(kinds):
            return kinds[dataloader_idx]
        return "poc2mol"

    def _update_val(self, metric: str, split: str, value, weight: int = 1):
        """Update the split's metric and the pooled 'combined' one from a single value."""
        if value is None:
            return
        if isinstance(value, float) and np.isnan(value):
            return
        for key in (split, "combined"):
            name = f"{key}__{metric}"
            if name in self.val_metrics:
                self.val_metrics[name].update(value, weight=weight)

    def on_validation_epoch_start(self):
        self._likelihood_rows = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        split = self._val_split(dataloader_idx)
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        masked_labels = labels.clone()
        masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100

        outputs = self(pixel_values, labels=masked_labels)
        loss = outputs.loss
        with torch.no_grad():
            accuracy = accuracy_from_outputs(outputs, masked_labels, start_ix=1, ignore_index=-100)

        n = pixel_values.size(0)
        self._update_val("loss", split, loss.detach(), weight=n)
        self._update_val("accuracy", split, accuracy, weight=n)

        # ---- free-running generation metrics -------------------------------------
        # Teacher-forced loss/accuracy saturate and say little about deployed behaviour;
        # these three come from actually sampling the molecule. All are computed from ONE
        # generate call, so the cost is a single decode of n_samples_for_validity_testing.
        if batch_idx == 0 or batch_idx * n < self.n_samples_for_validity_testing:
            generated_smiles = self.generate_smiles(
                pixel_values[:self.n_samples_for_validity_testing], max_attempts=1
            )
            if len(generated_smiles) > 0:
                reference_smiles = [
                    sm.replace(" ", "")
                    for sm in self.tokenizer.batch_decode(
                        labels[:len(generated_smiles)], skip_special_tokens=True
                    )
                ]
                m = len(generated_smiles)
                self._update_val("validity", split, calculate_validity(generated_smiles), weight=m)
                self._update_val("exact_match", split,
                                 calculate_exact_match(generated_smiles, reference_smiles), weight=m)
                # Mean Morgan-Tanimoto of the generated molecule to the TRUE ligand. Unlike
                # exact match this degrades gracefully -- it still moves when the model is
                # close but not identical, which is the regime stage 3 will live in.
                self._update_val("tanimoto", split,
                                 calculate_paired_similarity(generated_smiles, reference_smiles),
                                 weight=m)

        if batch_idx < 3 and self.visualise_val:
            try:
                sample_str = "" if split == "zinc" else "poc2mol_output "
                self.visualize_smiles(batch, sample_str=sample_str)
            except Exception as e:
                print("Error visualizing smiles: ", e)

        # ---- likelihood ranking against a shared decoy panel ----------------------
        # Only accumulated here; the AUC needs the whole pocket x candidate matrix, which
        # does not exist until the epoch ends (see src/utils/likelihood_eval.py).
        if "candidate_tokens" in batch and "binder_indices" in batch:
            self._accumulate_likelihood_rows(batch, pixel_values)

        return loss

    def _accumulate_likelihood_rows(self, batch, pixel_values):
        """Score every candidate under every pocket in this batch; stash the rows."""
        cand_ids = batch["candidate_tokens"]["input_ids"].to(pixel_values.device)
        binder_indices = batch["binder_indices"]
        pad_id = self.tokenizer.pad_token_id
        n_candidates = cand_ids.size(0)

        masked = cand_ids.clone()
        masked[masked == pad_id] = -100
        gather_idx = masked.clone()
        gather_idx[gather_idx == -100] = 0
        valid = masked != -100
        seq_len = valid.sum(dim=1).clamp(min=1)

        with torch.no_grad():
            for i in range(pixel_values.size(0)):
                repeated = pixel_values[i: i + 1].repeat(
                    n_candidates, *([1] * (pixel_values.dim() - 1))
                )
                logits = self(repeated, labels=cand_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                token_lp = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1) * valid
                # Mean per-token, matching the published convention. This is exactly the
                # quantity that carries the ligand-size nuisance the z-norm removes.
                row = (token_lp.sum(dim=1) / seq_len).detach().cpu().numpy()
                self._likelihood_rows.append((row, int(binder_indices[i].item())))

    def on_validation_epoch_end(self):
        # Log every accumulated metric here rather than inside validation_step: the
        # "combined" pool is fed from several dataloaders, and logging one key from more
        # than one dataloader_idx inside the step is what Lightning rejects.
        for split in self.VAL_SPLITS:
            for metric in self.VAL_METRICS:
                meter = self.val_metrics[f"{split}__{metric}"]
                if meter.update_count == 0:
                    continue
                value = meter.compute()
                self.log(f"val/{split}/{metric}", value, prog_bar=(metric in ("loss", "exact_match")))
                # val/loss is what ModelCheckpoint and early stopping monitor. Point it at
                # the pooled figure so selection reflects the training distribution.
                if split == "combined" and metric == "loss":
                    self.log("val/loss", value, prog_bar=True)
                meter.reset()

        self._log_likelihood_metrics()

    def _log_likelihood_metrics(self):
        rows = self._gather_likelihood_rows()
        self._likelihood_rows = []
        if not rows:
            return

        scores = np.stack([r for r, _ in rows])
        smiles = self._candidate_smiles()
        positive = np.zeros_like(scores, dtype=bool)
        for row_ix, (_, binder_ix) in enumerate(rows):
            if smiles is not None:
                # Match positives by SMILES identity, not index: duplicate SMILES in the
                # panel would otherwise be scored as misses (CLAUDE.md 5.1).
                target = smiles[binder_ix] if binder_ix < len(smiles) else None
                if target is not None:
                    positive[row_ix] = np.array([s == target for s in smiles])
                    continue
            positive[row_ix, binder_ix] = True

        valid_columns = None
        if smiles is not None:
            with _blocked_rdkit_logs():
                valid_columns = np.array([Chem.MolFromSmiles(s) is not None for s in smiles])

        for name, value in evaluate_likelihood_ranking(scores, positive, valid_columns).items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                self.log(f"val/{name}", value, prog_bar=(name == "likelihood_auc_znorm"),
                         rank_zero_only=True)

    def _gather_likelihood_rows(self):
        """Collect every rank's rows. Each rank validates a different shard of pockets, so
        without this the matrix would be a quarter of its size and the z-norm correspondingly
        noisier."""
        rows = self._likelihood_rows
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return rows
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, rows)
        return [item for shard in gathered if shard for item in shard]

    def _candidate_smiles(self):
        """The shared decoy panel, in candidate order, if any val dataset carries one."""
        datamodule = getattr(self.trainer, "datamodule", None)
        for dataset in (getattr(datamodule, "val_datasets", {}) or {}).values():
            panel = getattr(dataset, "decoy_smiles_list", None)
            if panel:
                return list(panel)
        return None

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        masked_labels = labels.clone()
        masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100
        with torch.no_grad():
            accuracy = accuracy_from_outputs(outputs, masked_labels, start_ix=1, ignore_index=-100)
        self.test_acc(accuracy)
        self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # weight_decay was never passed, so every run so far has used AdamW's 0.01 default --
        # i.e. it was never a deliberate choice. With 172.7M parameters over 9,872 HiQBind
        # clusters (~17,500 params per cluster) it is one of the few regularisation knobs that
        # costs nothing to turn, so it is now explicit and sweepable.
        weight_decay = float(getattr(self.hparams.config, "weight_decay", 0.01))
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.config.lr, weight_decay=weight_decay
        )
        
        # Get scheduler configuration from config
        scheduler_config = getattr(self.hparams.config, "scheduler", {})
        scheduler_type = scheduler_config.get("type", "step")
        
        if scheduler_type == "step":
            # Default StepLR scheduler
            step_size = scheduler_config.get("step_size", 100)
            gamma = scheduler_config.get("gamma", 0.997)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            # Use transformers' get_scheduler for other scheduler types
            num_training_steps = scheduler_config.get("num_training_steps", self.trainer.estimated_stepping_batches)
            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
            
            if isinstance(num_warmup_steps, float) and 0 <= num_warmup_steps < 1:
                # If warmup_steps is a fraction, calculate the absolute number
                num_warmup_steps = int(num_training_steps * num_warmup_steps)
            
            scheduler_specific_kwargs = {}
            
            # Handle specific scheduler parameters
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
                scheduler_specific_kwargs=scheduler_specific_kwargs
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": scheduler_config.get("interval", "step"),
                    "frequency": scheduler_config.get("frequency", 1),
                },
            }
    
    
    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading, optionally overriding optimizer and scheduler states.

        If override_optimizer_on_load is True, we'll remove the optimizer and
        lr_scheduler states from the checkpoint, forcing Lightning to create new ones
        based on the current config hyperparameters.
        """
        if self.override_optimizer_on_load:
            if "optimizer_states" in checkpoint:
                print(
                    "Overriding optimizer state from checkpoint with current config values"
                )
                del checkpoint["optimizer_states"]

            if "lr_schedulers" in checkpoint:
                print(
                    "Overriding lr scheduler state from checkpoint with current config values"
                )
                del checkpoint["lr_schedulers"]

            # Set a flag to tell Lightning not to expect optimizer states
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
    

    def is_valid_smiles(self, smiles):
        # Called once per generated sample inside generate_smiles. Without the log block an
        # untrained decoder emits a multi-line RDKit parse error for every sample of every
        # validation, which buries anything real in the training log.
        try:
            with _blocked_rdkit_logs():
                mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return True
            else:
                return False
        except:
            return False
    
    def repetition_count(self, smiles):
        max_reps = 0
        current_reps = 0
        for sm in smiles:
            sm = sm.replace(' ', '')
            for i in range(len(sm)):
                for j in range(i+1, len(sm)):
                    if sm[i] == sm[j]:
                        current_reps += 1
                        max_reps = max(max_reps, current_reps)
                    else:
                        current_reps = 0
        return max_reps
    
    @torch.inference_mode()
    def generate_smiles(
            self, 
            pixel_values, 
            max_length=200, 
            max_attempts=6, 
            max_token_repeats=10, 
            do_sample=False, 
            temperature=1.0
        ):
        with torch.no_grad():
            assert max_attempts > 0, "max_attempts must be greater than 0"
            batch_size = pixel_values.shape[0]
            results = [None] * batch_size
            best_results = [None] * batch_size
            min_repetition_scores = [float('inf')] * batch_size
            need_generation = [True] * batch_size
            attempts = [0] * batch_size
            
            while any(need_generation) and max(attempts) < max_attempts:
                indices_to_generate = [i for i, need_gen in enumerate(need_generation) if need_gen]
                if len(indices_to_generate) < batch_size:
                    current_pixel_values = pixel_values[indices_to_generate]
                else:
                    current_pixel_values = pixel_values
                
                current_do_sample = do_sample or max(attempts) > 0  # Use sampling after first attempt
                tokens = self.model.generate(current_pixel_values, max_length=max_length, do_sample=current_do_sample, temperature=temperature)
                tokens = tokens.detach().cpu()
                predicted_smiles = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                predicted_smiles = [sm.replace(' ', '') for sm in predicted_smiles]
                del tokens
                for idx, gen_idx in enumerate(indices_to_generate):
                    attempts[gen_idx] += 1
                    current_smiles = predicted_smiles[idx]
                    
                    # Track this attempt if it's a valid SMILES string
                    if len(current_smiles) > 0 and self.is_valid_smiles(current_smiles):
                        # Check repetition score
                        rep_count = self.repetition_count([current_smiles])
                        
                        # Update best result if this has a lower repetition score
                        if rep_count < min_repetition_scores[gen_idx]:
                            best_results[gen_idx] = current_smiles
                            min_repetition_scores[gen_idx] = rep_count
                        
                        # If under threshold, consider this a success
                        if rep_count <= max_token_repeats:
                            results[gen_idx] = current_smiles
                            need_generation[gen_idx] = False
                        

                    else:
                        if best_results[gen_idx] is None:
                            best_results[gen_idx] = current_smiles
        
            for i in range(batch_size):
                if results[i] is None:
                    results[i] = best_results[i]
        torch.cuda.empty_cache()
        return results

    def visualize_smiles(self, batch, sample_str=""):
        actual_smiles = batch["smiles_str"]
        actual_smiles = [sm.replace("[BOS]", '').replace("[EOS]", "") for sm in actual_smiles]
        predicted_smiles = self.generate_smiles(batch["pixel_values"])

        images_to_log = []

        for i, (pred, actual) in enumerate(zip(predicted_smiles, actual_smiles)):
            pred_img = self.smiles_to_image(pred, f"Predicted: {pred}")
            if pred_img:
                images_to_log.append(wandb.Image(pred_img, caption=f"Sample {i} {sample_str}Predicted"))

            actual_img = self.smiles_to_image(actual, f"Actual: {actual}")
            if actual_img:
                images_to_log.append(wandb.Image(actual_img, caption=f"Sample {i} {sample_str}Actual"))

        # Only rank zero has a live wandb run; the other ranks would raise on wandb.log.
        if images_to_log and self.trainer.is_global_zero:
            wandb.log({"SMILES Comparison": images_to_log})
            for img in images_to_log:
                if hasattr(img, "image") and hasattr(img.image, "close"):
                    img.image.close()
            images_to_log.clear()

    def smiles_to_image(self, smiles, label):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(300, 300))

                # Add SMILES string as text to the image
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((10, 0), label, font=font, fill=(0, 0, 0))

                return img  # Return PIL Image object directly
            else:
                return None
        except:
            return None