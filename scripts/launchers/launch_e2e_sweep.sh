#!/bin/bash
# End-to-end Poc2Mol -> Vox2Smiles: four arms, 2 GPUs each, all 8 H100s.
#
# The arms form a ladder, each rung adding exactly one thing to the one below:
#   Z (gpu 0,1)  frozen upstream                      -- baseline: stage 3, this val regime
#   D (gpu 2,3)  + upstream trains on its voxel loss  -- effect of merely unfreezing
#   B (gpu 4,5)  + the LM gradient reaches it         -- effect of the end-to-end gradient
#   C (gpu 6,7)  + density anchor loosened to 0.1     -- effect of the loss balance
# B minus D is the measurement this branch exists to make. Z is what everything is read
# against -- NOT the published 0.7522, which came from a stochastic-validation run.
#
# ⚠️ This is a FILE, deliberately. CLAUDE.md §6: `pgrep -f "task_name=X"` matches the shell
# running the command too, so a later `pkill` on that pattern kills the invoking shell
# part-way through its own work. Keeping the launch in a script keeps the pattern off the
# invoking command line. This has already killed two runs.
set -u

cd /home/judewells/plixer_outer/plixer || exit 1

LOG_DIR="logs/e2e_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/pids.txt"
: > "$PID_FILE"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Distinct rendezvous port per arm. Four concurrent DDP jobs on one node will otherwise
# race for the same default MASTER_PORT, and the loser fails at process-group init.
launch () {
  local arm=$1 gpus=$2 port=$3
  CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$port" \
    ./venvPlixer/bin/python src/train.py \
      experiment="$arm" \
      trainer.devices=2 \
    > "$LOG_DIR/$arm.log" 2>&1 &
  echo "$!  $arm  gpus=$gpus" >> "$PID_FILE"
  echo "launched $arm on GPUs $gpus (pid $!)"
}

launch e2e_z_frozen      0,1 29511
launch e2e_d_control     2,3 29512
launch e2e_b_balanced    4,5 29513
launch e2e_c_lm_dominant 6,7 29514

echo
echo "logs: $LOG_DIR"
echo "waiting 420s before the health check..."
sleep 420

# A health check must assert a W&B RUN URL APPEARED, not merely that the process lives.
# CLAUDE.md §6: a run started with WANDB_MODE=offline looks identical in the progress bar
# and cannot be switched online afterwards; and a launcher that only checked backgrounding
# once reported 8 idle GPUs as "running" for hours after a composition error had killed
# everything in under a second.
echo
echo "================ HEALTH CHECK ================"
FAILED=0
while read -r pid arm gpus; do
  if kill -0 "$pid" 2>/dev/null; then
    alive="ALIVE"
  else
    alive="DEAD"; FAILED=1
  fi
  url=$(grep -o 'https://wandb.ai/[^ ]*' "$LOG_DIR/$arm.log" 2>/dev/null | tail -1)
  if [ -z "$url" ]; then url="NO W&B URL"; FAILED=1; fi
  steps=$(grep -c "it/s" "$LOG_DIR/$arm.log" 2>/dev/null)
  printf '%-22s pid=%-8s %-6s %s\n' "$arm" "$pid" "$alive" "$url"
  if [ "$alive" = "DEAD" ]; then
    echo "---- last 25 lines of $LOG_DIR/$arm.log ----"
    tail -25 "$LOG_DIR/$arm.log"
    echo "--------------------------------------------"
  fi
done < "$PID_FILE"

echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo
if [ "$FAILED" -eq 0 ]; then
  echo "HEALTH CHECK PASSED -- all 4 arms alive with W&B runs"
else
  echo "HEALTH CHECK FAILED"
fi
exit "$FAILED"
