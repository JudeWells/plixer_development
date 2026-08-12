#!/bin/bash
# Launch ONE end-to-end arm on a given GPU pair, then health-check it.
#
#   scripts/launchers/launch_e2e_arm.sh <experiment> <gpus> <port> <round_tag> [overrides...]
#   scripts/launchers/launch_e2e_arm.sh e2e_b_balanced 4,5 29521 r3
#
# Exists because arms finish at different times and the node should not sit idle waiting for
# a whole round to complete. Arms are independent runs at identical step budgets, so
# staggering them in wall-clock costs nothing in comparability.
#
# ⚠️ A FILE, deliberately -- CLAUDE.md §6: a later `pkill -f <pattern>` matches the shell
# that ran the command if the pattern appears on its command line. Keep launches in scripts.
set -u

[ $# -lt 4 ] && { echo "usage: $0 <experiment> <gpus> <port> <round_tag> [overrides...]"; exit 2; }

ARM=$1; GPUS=$2; PORT=$3; ROUND=$4; shift 4

cd /home/judewells/plixer_outer/plixer || exit 1

LOG_DIR="logs/e2e_${ROUND}"
mkdir -p "$LOG_DIR"
TASK="${ARM}_${ROUND}"
LOG="$LOG_DIR/${TASK}.log"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PORT" \
  ./venvPlixer/bin/python src/train.py \
    experiment="$ARM" \
    task_name="$TASK" \
    trainer.devices=2 \
    logger.wandb.group="e2e_${ROUND}" \
    "$@" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID  $TASK  gpus=$GPUS" >> "$LOG_DIR/pids.txt"
echo "launched $TASK on GPUs $GPUS (pid $PID), log $LOG"

# Health check: a live PID is NOT enough. An offline W&B run looks identical in the progress
# bar and cannot be switched online later (CLAUDE.md §6), and a composition error can kill
# the process in under a second while the launcher reports success.
sleep 300
if ! kill -0 "$PID" 2>/dev/null; then
  echo "HEALTH CHECK FAILED -- $TASK died. Last 25 lines:"
  tail -25 "$LOG"
  exit 1
fi
URL=$(grep -ao 'https://wandb.ai/[^ ]*' "$LOG" | tail -1)
if [ -z "$URL" ]; then
  echo "HEALTH CHECK FAILED -- $TASK alive but no W&B run URL. Last 25 lines:"
  tail -25 "$LOG"
  exit 1
fi
echo "HEALTH CHECK PASSED -- $TASK  $URL"
