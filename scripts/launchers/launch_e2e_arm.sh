#!/bin/bash
# Launch one end-to-end arm onto named GPUs and register it with the running idle-watcher.
#
#   scripts/launchers/launch_e2e_arm.sh <experiment> <gpu_ids> <master_port> [devices]
#
# Used to refill GPUs as arms early-stop, so the node never idles. The PID is appended to the
# sweep's pids.txt, which watch_gpu_idle.sh re-reads every cycle -- so a newly launched arm is
# tracked without restarting the watcher.
#
# ⚠️ A FILE, deliberately. CLAUDE.md §6: `pgrep -f "task_name=X"` matches the shell running
# the command too, so a later `pkill` on that pattern kills the invoking shell part-way
# through. Keeping the launch in a script keeps the pattern off the invoking command line.
set -u

cd /home/judewells/plixer_outer/plixer || exit 1

ARM=$1
GPUS=$2
PORT=$3
DEVICES=${4:-1}

LOG_DIR=$(ls -dt logs/e2e_warm_sweep_* 2>/dev/null | head -1)
[ -z "$LOG_DIR" ] && LOG_DIR="logs/e2e_warm_sweep_manual" && mkdir -p "$LOG_DIR"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PORT" \
  ./venvPlixer/bin/python src/train.py \
    experiment="$ARM" \
    trainer.devices="$DEVICES" \
  > "$LOG_DIR/$ARM.log" 2>&1 &

echo "$!  $ARM  gpus=$GPUS" >> "$LOG_DIR/pids.txt"
echo "launched $ARM on GPUs $GPUS (pid $!), log $LOG_DIR/$ARM.log"
