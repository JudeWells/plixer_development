#!/usr/bin/env bash
# Experiment 1, stage 1 of 3: ligand-only ZINC pretraining, BOTH arms in lockstep.
#
#   baseline  9 channels  (ligand only)
#   protein  14 channels  (9 ligand + 4 protein + 1 protein-present flag; protein empty and
#                          flag 0 throughout this stage, since ZINC molecules have no pocket)
#
# The arms are given identical data, effective batch, LR schedule and seed. Only the channel
# count differs, so any downstream difference is attributable. 4 GPUs each rather than 8 for
# one arm at a time: a matched budget matters more than finishing one arm sooner.
#
# Curriculum (stages 2 and 3 follow, resuming from these weights):
#   2. mix in protein-ligand complexes with TRUE ligand voxels -> learn to read the protein
#   3. ramp Poc2Mol predictions in place of the true ligand -> learn to cope with upstream error
set -uo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/exp1_stage1_zinc/${STAMP}"; mkdir -p "${DIR}"
echo "${DIR}" > /tmp/exp1_stage1_dir.txt
echo "stage-1 sweep dir: ${DIR}"

launch () {  # name  experiment  gpus
  local name="$1" exp="$2" gpus="$3"
  CUDA_VISIBLE_DEVICES="${gpus}" nohup venvPlixer/bin/python src/train.py \
    experiment="${exp}" \
    task_name="exp1_s1_${name}" \
    trainer=ddp trainer.devices=4 \
    data.num_workers=14 \
    data.config.batch_size=64 \
    data.config.target_samples_per_batch=256 \
    trainer.max_epochs=50 \
    trainer.val_check_interval=4000 \
    +trainer.num_sanity_val_steps=0 \
    callbacks.model_checkpoint.save_weights_only=True \
    callbacks.model_checkpoint.save_top_k=3 \
    callbacks.model_checkpoint.save_last=True \
    logger.wandb.group="exp1_stage1_zinc_${STAMP}" \
    logger.wandb.name="s1_${name}" \
    > "${DIR}/${name}.log" 2>&1 &
  echo $! > "${DIR}/${name}.pid"
  echo "  ${name}: gpus ${gpus}"
}

launch baseline exp1_zinc_baseline 0,1,2,3
sleep 30
launch protein  exp1_zinc_protein  4,5,6,7
