#!/bin/bash
# End-to-end from a WARM-STARTED decoder: four arms, 2 GPUs each, all 8 H100s on nebius1.
#
# The companion sweep (scripts/launchers/launch_e2e_sweep.sh, running on nebius2) starts the
# decoder from stage 1 -- ZINC only, meeting Poc2Mol's density for the first time at step 0.
# This one starts it from checkpoints/s3_v2_11ch/step_0003000_auc_0.7576.ckpt, a decoder that
# has already been through stage 3 against this exact Poc2Mol. The premise is that the
# end-to-end gradient is a refinement and deserves a competent decoder to refine.
#
#   W0 (gpu 0,1)  frozen upstream, decoder 5e-5   -- the warm baseline; also the overfit test
#   W1 (gpu 2,3)  + LM gradient, poc2mol_lr 1e-5  -- gentle upstream rate (untested anywhere)
#   W2 (gpu 4,5)  + LM gradient, poc2mol_lr 1e-4  -- arm B's settings, warm decoder
#   W3 (gpu 6,7)  W2 with the decoder at 2e-5     -- guards the warm decoder against overfit
#
# W2 - W0 is the end-to-end gradient's value on a competent decoder. W2 - B (other node) is
# whether warm-starting is worth anything; see e2e_w0_warm_frozen.yaml for the ancestry
# confound that reading carries.
#
# ⚠️ This is a FILE, deliberately. CLAUDE.md §6: `pgrep -f "task_name=X"` matches the shell
# running the command too, so a later `pkill` on that pattern kills the invoking shell
# part-way through its own work. Keeping the launch in a script keeps the pattern off the
# invoking command line. This has already killed two runs.
set -u

cd /home/judewells/plixer_outer/plixer || exit 1

LOG_DIR="logs/e2e_warm_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/pids.txt"
: > "$PID_FILE"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Distinct rendezvous port per arm. Four concurrent DDP jobs on one node will otherwise race
# for the same default MASTER_PORT and the loser fails at process-group init. These are also
# offset from the companion sweep's 2951x block in case both ever run on one host.
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

launch e2e_w0_warm_frozen    0,1 29611
launch e2e_w1_warm_lr1e5     2,3 29612
launch e2e_w2_warm_lr1e4     4,5 29613
launch e2e_w3_warm_declr2e5  6,7 29614

echo
echo "logs: $LOG_DIR"
echo "waiting 420s before the health check..."
sleep 420

# A health check must assert a W&B RUN URL APPEARED, not merely that the process lives.
# CLAUDE.md §6: a run started with WANDB_MODE=offline looks identical in the progress bar and
# cannot be switched online afterwards; and a launcher that only checked backgrounding once
# reported 8 idle GPUs as "running" for hours after a composition error had killed everything
# in under a second.
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
  printf '%-24s pid=%-8s %-6s %s\n' "$arm" "$pid" "$alive" "$url"
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
