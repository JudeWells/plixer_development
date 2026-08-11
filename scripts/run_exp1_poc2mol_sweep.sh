#!/usr/bin/env bash
# Experiment 1 -- hyperparameter sweep for the SHARED Poc2Mol.
#
# Eight configs, one per GPU, in parallel. This is a better use of the node than one 8-way
# DDP run: at effective batch 128 a DDP rank gets only 16 samples, which starves an H100
# (measured 61 samples/s/rank against 869 at batch 64) and buys just 1.3x over a single
# GPU. Same wall clock, 8x the information.
#
# Grid: one factor at a time around the inherited config (slot 1), plus one strong-
# regularisation combo. HiQBind is only 9,872 clusters against a 117M-parameter UNet, so
# the regularisation axes are the ones most likely to matter.
#
# Each run trains to 600 epochs -- well past the expected val/loss minimum -- so the
# overfitting turn is visible rather than inferred. Best checkpoint is kept by val/loss.
#
# Usage:  bash scripts/run_exp1_poc2mol_sweep.sh [max_epochs]
set -uo pipefail

cd "$(dirname "$0")/.."
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
# Each run gets its own dataloader workers; 8 runs x 12 workers = 96 of 128 cores, leaving
# headroom for the eight main processes.
WORKERS=12
MAX_EPOCHS="${1:-600}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/exp1_poc2mol_sweep/${STAMP}"
mkdir -p "${SWEEP_DIR}"

#        name              lr      weight_decay  dropout
CONFIGS=(
  "base            1e-4    0.2   0.1"
  "lr_low          3e-5    0.2   0.1"
  "lr_high         3e-4    0.2   0.1"
  "wd_low          1e-4    0.05  0.1"
  "wd_high         1e-4    0.5   0.1"
  "drop_none       1e-4    0.2   0.0"
  "drop_high       1e-4    0.2   0.2"
  "reg_combo       3e-4    0.5   0.2"
)

echo "Launching ${#CONFIGS[@]} runs, ${MAX_EPOCHS} epochs each, logs in ${SWEEP_DIR}"
printf '%-6s %-14s %-8s %-6s %-6s\n' GPU NAME LR WD DROP

gpu=0
for cfg in "${CONFIGS[@]}"; do
  read -r name lr wd drop <<< "${cfg}"
  printf '%-6s %-14s %-8s %-6s %-6s\n' "${gpu}" "${name}" "${lr}" "${wd}" "${drop}"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup venvPlixer/bin/python src/train.py \
    experiment=exp1_poc2mol \
    task_name="exp1_poc2mol_${name}" \
    model.lr="${lr}" \
    model.weight_decay="${wd}" \
    model.config.dropout_prob="${drop}" \
    data.num_workers="${WORKERS}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    logger.wandb.group="exp1_poc2mol_${STAMP}" \
    logger.wandb.name="poc2mol_${name}_lr${lr}_wd${wd}_do${drop}" \
    > "${SWEEP_DIR}/${name}.log" 2>&1 &

  echo $! > "${SWEEP_DIR}/${name}.pid"
  gpu=$((gpu + 1))
  # Stagger: eight simultaneous cold starts contend badly on parquet index reads and on
  # W&B run creation.
  sleep 20
done

echo
echo "All launched. Monitor with:"
echo "  bash scripts/watch_exp1_poc2mol_sweep.sh ${SWEEP_DIR}"
echo "Stop all with:"
echo "  cat ${SWEEP_DIR}/*.pid | xargs -r kill"
wait
