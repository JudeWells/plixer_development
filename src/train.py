import os
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict
import math

from src.utils import rich_utils

# Dataloader workers used to build voxel grids on the GPU, which meant CUDA tensors had to
# be created in the worker and forced the 'spawn' start method (and a slow re-import of the
# whole stack per worker). Workers are now CUDA-free -- voxelisation moved to
# on_after_batch_transfer -- so 'fork' is both safe and far cheaper to start, which matters
# at 16 workers x 8 ranks. Override with PLIXER_MP_START_METHOD=spawn if a worker ever
# needs CUDA again.
if __name__ == "__main__":
    multiprocessing.set_start_method(
        os.environ.get("PLIXER_MP_START_METHOD", "fork"), force=True
    )

from src.utils import (
    ProvenanceCallback,
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    write_provenance,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

log = RankedLogger(__name__, rank_zero_only=True)


def _resolve_world_size(cfg: DictConfig) -> int:
    """Number of processes the trainer will launch, from the trainer config alone.

    Needed before the Trainer exists, so it has to mirror Lightning's own interpretation of
    `devices`: an int is a count, -1 or "auto" means every visible device, and a list is an
    explicit selection.
    """
    devices = cfg.trainer.get("devices", 1)
    num_nodes = int(cfg.trainer.get("num_nodes", 1) or 1)

    if isinstance(devices, str):
        devices = -1 if devices == "auto" else int(devices)
    if isinstance(devices, int):
        n_devices = torch.cuda.device_count() if devices == -1 else devices
    else:  # list / ListConfig of device indices
        n_devices = len(devices)

    return max(1, int(n_devices)) * max(1, num_nodes)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Record git state, resolved config and parent checkpoints before anything runs.
    # `write_provenance` is rank-zero-only, so other ranks get None; only rank zero
    # writes checkpoints, so an empty record elsewhere is harmless.
    provenance_record = write_provenance(cfg, cfg.paths.output_dir) or {}

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    if cfg.get("logger") and "wandb" in cfg.get("logger"):
        with open_dict(cfg):
            if cfg.logger.wandb.get("name") is None:
                num_params = sum(p.numel() for p in model.parameters())
                if num_params >= 1_000_000_000:
                    params_str = f"{num_params / 1_000_000_000:.3g}B"
                else:
                    params_str = f"{num_params / 1_000_000:.3g}M"
                tags_str = ""
                if cfg.get("tags"):
                    tags_str = "||".join(cfg.tags)

                if tags_str:
                    cfg.logger.wandb.name = f"{tags_str}-{params_str}"
                else:
                    cfg.logger.wandb.name = params_str

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Embeds the provenance record into every checkpoint this run saves, so a stray
    # .ckpt file is enough to identify the code and config that produced it.
    callbacks.append(ProvenanceCallback(provenance_record))

    batch_size = cfg.data.config.batch_size
    target_samples_per_batch = cfg.data.config.get("target_samples_per_batch", batch_size)

    # Under DDP every rank contributes a full batch to each optimiser step, so the world
    # size has to divide out here. Without it, moving from 1 to 8 GPUs would silently
    # multiply the effective batch by 8 and the "matched budget" comparison between the
    # baseline and the protein-channel arm would not be matched at all.
    world_size = _resolve_world_size(cfg)
    samples_per_step = batch_size * world_size
    accumulate_grad_batches = max(1, round(target_samples_per_batch / samples_per_step))

    # Set accumulate_grad_batches in trainer config
    cfg.trainer.accumulate_grad_batches = accumulate_grad_batches

    effective = samples_per_step * accumulate_grad_batches
    log.info(
        f"Effective batch: {batch_size} per rank x {world_size} ranks x "
        f"{accumulate_grad_batches} accumulation = {effective} samples/step "
        f"(target {target_samples_per_batch})"
    )
    # `val_check_interval` counts TRAINING MICRO-BATCHES, so accumulation silently rescales
    # it: the same 250 that means "every 250 optimiser steps" at accumulate=1 means "every
    # 62" at accumulate=4. That in turn rescales EarlyStopping's patience, which counts
    # validation checks. On 2026-08-12 this stopped an end-to-end arm at step 1312 with a
    # nominal patience of 3000, and no arm would have reached its LR anneal. Nothing here
    # is wrong per se, so this warns rather than raises -- but it should never again be
    # discovered by reading a truncated run.
    val_check_interval = cfg.trainer.get("val_check_interval", None)
    if accumulate_grad_batches > 1 and isinstance(val_check_interval, int):
        steps_between = val_check_interval / accumulate_grad_batches
        patience = None
        if cfg.get("callbacks") and cfg.callbacks.get("early_stopping"):
            patience = cfg.callbacks.early_stopping.get("patience")
        log.warning(
            f"val_check_interval={val_check_interval} counts MICRO-BATCHES; with "
            f"accumulate_grad_batches={accumulate_grad_batches} that is a validation every "
            f"{steps_between:g} OPTIMISER STEPS"
            + (f", so early-stopping patience {patience} = {patience * steps_between:g} steps"
               if patience else "")
            + ". Multiply by accumulate_grad_batches if you meant optimiser steps."
        )

    if effective != target_samples_per_batch:
        log.warning(
            f"Effective batch {effective} != target {target_samples_per_batch}. "
            f"{target_samples_per_batch} is not divisible by batch_size x world_size "
            f"({samples_per_step}); runs at different world sizes will NOT be comparable. "
            f"Pick a batch_size that divides it evenly."
        )

    # Weights-only initialisation, distinct from `ckpt_path`. `ckpt_path` goes to
    # trainer.fit and restores EVERYTHING -- global_step, optimizer and scheduler state --
    # which is right for resuming an interrupted run but wrong for starting a new curriculum
    # stage: stage 2 would begin at stage 1's ~100k steps, landing past its own warmup and
    # potentially past max_steps. This loads the tensors and nothing else, so the new stage
    # starts at step 0 with its own schedule.
    init_from = cfg.get("init_weights_from")
    if init_from:
        log.info(f"Initialising model weights from {init_from} (weights only, step resets to 0)")
        checkpoint = torch.load(init_from, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        # Metric buffers legitimately come and go as metrics are added, so those are
        # expected; anything else means the architectures differ and the run is not what
        # it claims to be -- most likely a channel-count mismatch between arms, since the
        # patch-embedding conv width differs between the 9ch and 14ch decoders.
        unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(("val_", "train_", "test_"))]
        missing = [k for k in incompatible.missing_keys if not k.startswith(("val_", "train_", "test_"))]
        if unexpected or missing:
            log.warning(f"state_dict mismatch on init: missing={missing[:8]} unexpected={unexpected[:8]}")
        else:
            log.info("state_dict loaded cleanly (metric buffers aside)")
        # Same shape as the entries write_provenance emits; logging_utils reads .get on these.
        provenance_record.setdefault("parent_checkpoints", []).append(
            {"config_key": "init_weights_from", "path": str(init_from)}
        )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "provenance": provenance_record,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        try:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        except Exception as e:
            log.info(f"Error during training: {e}")
            save_dir = cfg.callbacks.model_checkpoint.dirpath
            os.makedirs(save_dir, exist_ok=True)
            ckpt_save_path = os.path.join(
                save_dir,
                "interrupted.ckpt"
            )
            trainer.save_checkpoint(ckpt_save_path)
            log.info(f"Saved checkpoint to {ckpt_save_path}")
            raise e
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == "__main__":
    main()