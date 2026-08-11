#!/usr/bin/env bash
# Experiment 1 stage 1 -- learning-rate sweep to speed up convergence.
#
# Resumes all arms from ONE frozen checkpoint (checkpoints/exp1_stage1_base/), with
# override_optimizer_on_load=true so the new LR actually takes effect rather than the
# restored optimiser state overriding it. The incumbent s1_protein continues at lr 1e-4 on
# GPUs 4-7 as the control.
#
# Per-GPU batch 64 -> 128, with accumulation 2, so the effective batch stays 256 -- identical
# to the incumbent's 64 x 4 ranks. That isolates the learning rate as the only variable.
#
# A synthetic probe suggested 256/GPU would fit (51.6 GB of 80), but that used fixed 64-token
# labels; real batches pad to the longest SMILES in the batch (up to ~104 tokens) and the
# voxeliser plus validation-time generation allocate on the same device. 256 OOM'd in
# practice, so 128 it is.
#
# Worth knowing: one GPU at batch 128 sustains ~745 samples/s, about what all four incumbent
# GPUs manage at batch 64 -- the old setting was badly GPU-underutilised.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
# Variable SMILES lengths fragment the allocator badly; expandable segments avoids the
# "tried to allocate 318 MiB" failures seen at batch 256.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="checkpoints/exp1_stage1_base/s1_protein_lr_sweep_start.ckpt"
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/exp1_lr_sweep/${STAMP}"; mkdir -p "${DIR}"
echo "${DIR}" > /tmp/exp1_lr_dir.txt
echo "lr sweep dir: ${DIR}  (control: s1_protein at lr 1e-4)"

gpu=0
for lr in 2e-4 3e-4 5e-4 1e-3; do
  name="lr${lr}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup venvPlixer/bin/python src/train.py \
    experiment=exp1_zinc_protein \
    task_name="exp1_s1_lr_${lr}" \
    ckpt_path="${CKPT}" \
    model.override_optimizer_on_load=true \
    model.config.lr="${lr}" \
    model.config.scheduler.num_warmup_steps=500 \
    trainer=default trainer.devices=1 \
    data.num_workers=18 \
    data.config.batch_size=128 \
    data.config.target_samples_per_batch=256 \
    trainer.max_epochs=50 \
    trainer.val_check_interval=1000 \
    +trainer.num_sanity_val_steps=0 \
    callbacks.model_checkpoint.save_weights_only=True \
    logger.wandb.group="exp1_lr_sweep_${STAMP}" \
    logger.wandb.name="s1_lr${lr}_bs256" \
    > "${DIR}/${name}.log" 2>&1 &
  echo $! > "${DIR}/${name}.pid"
  echo "  gpu${gpu}  lr=${lr}  batch=128 x accum 2 = 256 effective"
  gpu=$((gpu+1)); sleep 20
done
