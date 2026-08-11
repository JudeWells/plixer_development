#!/usr/bin/env bash
# Experiment 1 stage 1 -- learning-rate sweep, FROM SCRATCH.
#
# The earlier resume-based sweep was confounded: the frozen checkpoint was weights-only, so
# Adam restarted with zero moment estimates and every arm spiked (0.195 -> 0.38/0.47/0.52,
# magnitude ordered by LR) before recovering. Warmup rescales the step but cannot reconstruct
# a missing second moment, so that design could not answer "which LR is best".
#
# Here every arm starts from random init with an identical config -- same data, effective
# batch, warmup, seed -- so learning rate is the only variable. This is also the question that
# actually matters, since the production run will train from scratch.
#
# 8 arms, one GPU each, spanning 1e-4 (the incumbent) to 2e-3 so the sweep brackets the point
# of instability rather than just ranking survivors.
#
# Per-GPU batch 128 with accumulation 2 = 256 effective, matching the earlier 4-GPU runs so
# their curves remain a usable reference. Batch 256/GPU OOMs once real SMILES lengths and the
# voxeliser are accounted for.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/exp1_lr_scratch/${STAMP}"; mkdir -p "${DIR}"
echo "${DIR}" > /tmp/exp1_lrscratch_dir.txt
echo "from-scratch LR sweep: ${DIR}"

gpu=0
for lr in 1e-4 2e-4 3e-4 5e-4 7e-4 1e-3 1.5e-3 2e-3; do
  CUDA_VISIBLE_DEVICES="${gpu}" nohup venvPlixer/bin/python src/train.py \
    experiment=exp1_zinc_protein \
    task_name="exp1_lrs_${lr}" \
    ckpt_path=null \
    model.config.lr="${lr}" \
    trainer=default trainer.devices=1 \
    data.num_workers=14 \
    data.config.batch_size=128 \
    data.config.target_samples_per_batch=256 \
    trainer.max_epochs=50 \
    trainer.val_check_interval=1000 \
    +trainer.num_sanity_val_steps=0 \
    callbacks.model_checkpoint.save_weights_only=True \
    callbacks.model_checkpoint.save_top_k=1 \
    logger.wandb.group="exp1_lr_scratch_${STAMP}" \
    logger.wandb.name="scratch_lr${lr}" \
    > "${DIR}/lr${lr}.log" 2>&1 &
  echo $! > "${DIR}/lr${lr}.pid"
  echo "  gpu${gpu}  lr=${lr}"
  gpu=$((gpu+1)); sleep 15
done
