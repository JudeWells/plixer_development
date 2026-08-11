#!/usr/bin/env bash
# Experiment 1 -- shared Poc2Mol, protein-representation sweep.
#
# Four protein encodings x {dropout off, dropout on} = 8 runs, one per GPU.
#
#   cons4  C/O/N/S                  -- control, the corrected encoding
#   h5     C/O/N/S + H              -- does an explicit hydrogen channel help?
#   all5   C/O/N/S + ALL            -- generic total density ("*" matches every atom)
#   hall6  C/O/N/S + H + ALL        -- both
#
# Why representation rather than more hyperparameters: the previous sweep
# (20260806_135055) found lr, weight decay and dropout all within 0.003 of each other,
# while the pre-fix encoding -- which carried protein hydrogens in the mislabelled channel
# 3 -- reached a lower val/loss than any corrected 4-channel run. That points at the input
# representation, not the optimiser.
#
# Dropout is a real variable here for the first time. nn.Dropout is only inserted when
# layer_order contains 'd', and every previous config used 'gcr', so `dropout_prob` was a
# silent no-op and three slots of the last sweep were unwitting duplicates.
#
# Validation is now deterministic (no rotation, no translation, fixed cluster member), so
# val/loss is a stable benchmark -- but absolute values are NOT comparable with runs from
# before 2026-08-06.
#
# Usage:  bash scripts/run_exp1_poc2mol_protein_repr.sh [max_epochs]
set -uo pipefail

cd "$(dirname "$0")/.."
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
WORKERS=12
MAX_EPOCHS="${1:-600}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/exp1_poc2mol_protein_repr/${STAMP}"
mkdir -p "${SWEEP_DIR}"

#        name          experiment-config      layer_order  dropout_prob
CONFIGS=(
  "cons4_nodrop  exp1_poc2mol_cons4  gcr   0.0"
  "cons4_drop    exp1_poc2mol_cons4  gcrd  0.1"
  "h5_nodrop     exp1_poc2mol_h5     gcr   0.0"
  "h5_drop       exp1_poc2mol_h5     gcrd  0.1"
  "all5_nodrop   exp1_poc2mol_all5   gcr   0.0"
  "all5_drop     exp1_poc2mol_all5   gcrd  0.1"
  "hall6_nodrop  exp1_poc2mol_hall6  gcr   0.0"
  "hall6_drop    exp1_poc2mol_hall6  gcrd  0.1"
)

echo "Launching ${#CONFIGS[@]} runs, ${MAX_EPOCHS} epochs each, logs in ${SWEEP_DIR}"
printf '%-5s %-14s %-20s %-6s %s\n' GPU NAME CONFIG ORDER DROPOUT

gpu=0
for cfg in "${CONFIGS[@]}"; do
  read -r name experiment order drop <<< "${cfg}"
  printf '%-5s %-14s %-20s %-6s %s\n' "${gpu}" "${name}" "${experiment}" "${order}" "${drop}"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup venvPlixer/bin/python src/train.py \
    experiment="${experiment}" \
    task_name="exp1_prot_${name}" \
    model.config.layer_order="${order}" \
    model.config.dropout_prob="${drop}" \
    data.num_workers="${WORKERS}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    logger.wandb.group="exp1_poc2mol_protein_repr_${STAMP}" \
    logger.wandb.name="poc2mol_${name}" \
    > "${SWEEP_DIR}/${name}.log" 2>&1 &

  echo $! > "${SWEEP_DIR}/${name}.pid"
  gpu=$((gpu + 1))
  sleep 20
done

echo
echo "Monitor with:  bash scripts/watch_exp1_poc2mol_sweep.sh ${SWEEP_DIR}"
echo "Stop all with: cat ${SWEEP_DIR}/*.pid | xargs -r kill"
wait
