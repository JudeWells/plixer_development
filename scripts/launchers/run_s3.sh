#!/usr/bin/env bash
# Experiment 1, decoder fine-tune on a ZINC + Poc2Mol-output mixture (nebius2).
#
# Starts from the stage-1 maxagg decoder (14ch) and ramps Poc2Mol's PREDICTED ligand density
# in over steps 2k-15k, so early training is the stage-2 regime (true voxels + real pocket,
# learning to read the protein channels) and late training is the deployed regime. That keeps
# the curriculum's separation without needing two launches.
#
# `max` aggregation throughout -- the frozen Poc2Mol was trained on max-aggregated grids and
# its sigmoid output is [0,1]; `sum` would be out of distribution at both its input and its
# output (CLAUDE.md 10b), which is why the sumagg stage-1 arm cannot come here yet.
#
# Validation always uses the prediction (fraction=1.0 when not training), so val/poc2mol/* is
# the deployed metric from step 0 and does not drift as the ramp advances.
set -uo pipefail
cd /home/judewells/plixer_outer/plixer || exit 1
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/exp1_s3_prod/${STAMP}"; mkdir -p "${DIR}"
echo "${DIR}" > /tmp/exp1_s3_dir.txt
echo "stage-3 fine-tune: ${DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup venvPlixer/bin/python src/train.py \
  experiment=exp1_s3_protein \
  task_name="exp1_s3_maxagg" \
  init_weights_from=checkpoints/s1_maxagg_last.ckpt \
  trainer.devices=4 \
  data.num_workers=12 \
  data.config.batch_size=64 \
  data.config.target_samples_per_batch=256 \
  data.config.voxel_aggregation=max \
  data.train_dataset.prob_poc2mol=0.5 \
  data.predicted_ramp_start_step=2000 \
  data.predicted_ramp_end_step=15000 \
  model.config.lr=5e-5 \
  model.n_samples_for_validity_testing=100 \
  model.config.scheduler.num_warmup_steps=1000 \
  model.config.scheduler.num_stable_steps=39000 \
  model.config.scheduler.num_decay_steps=20000 \
  model.config.scheduler.min_lr_ratio=0.03 \
  +trainer.max_steps=60000 \
  trainer.max_epochs=10000 \
  trainer.val_check_interval=2000 \
  +trainer.num_sanity_val_steps=0 \
  logger.wandb.group="exp1_s3_${STAMP}" \
  logger.wandb.name="s3_maxagg_${STAMP}" \
  > "${DIR}/s3_maxagg.log" 2>&1 &
PID=$!
echo $PID > "${DIR}/s3_maxagg.pid"
echo "  launched pid ${PID} on GPUs 0-3"

echo "waiting 240s for steady state..."
sleep 240
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "FAILED to start. Log tail:"; tail -25 "${DIR}/s3_maxagg.log"; exit 1
fi
echo "alive. wandb:"; grep -ho "https://wandb.ai/[^ ]*" "${DIR}/s3_maxagg.log" | sort -u
tr '\r' '\n' < "${DIR}/s3_maxagg.log" | grep -o "Epoch [0-9]*:.*it/s.*" | tail -1
