#!/usr/bin/env bash
# Generative Poc2Mol -- launch stage A (ZINC pretrain, 6 GPUs) alongside the from-scratch
# HiQBind control arm (2 GPUs), on nebius2.
#
# The two arms answer different questions and must run together:
#   stage A     -- learn what a ligand density IS, from 8.86M ZINC molecules, no pocket.
#   scratch     -- the control for "did pretraining help". Identical to stage B in every
#                  respect except `init_weights_from`, so the comparison is clean.
#
# Effective batch is matched by target_samples_per_batch, which src/train.py divides by
# batch_size x world_size -- 384 = 64 x 6 for stage A, 512 = 64 x 2 x 4 accumulation for the
# control. Do NOT set trainer.accumulate_grad_batches; it is overwritten.
#
# Health check at the end: a launcher that only backgrounds a process has, in this project,
# reported 8 idle GPUs as "running" for hours after a composition error killed the job in
# under a second.
set -u
cd ~/plixer_outer/plixer || exit 1

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=/tmp/flow_launch_${STAMP}
mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

# --------------------------------------------------------------- stage A: ZINC pretrain
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 WANDB_MODE=online nohup ./venvPlixer/bin/python src/train.py \
    experiment=flow_zinc_pretrain \
    task_name=flow_zinc_pretrain \
    trainer.devices=6 \
    data.config.batch_size=64 \
    data.config.target_samples_per_batch=384 \
    data.num_workers=12 \
    > "$LOGDIR/stage_a.log" 2>&1 &
PID_A=$!
echo "stage A  pid $PID_A  (GPUs 2-7)"

sleep 20

# ------------------------------------------------- control: HiQBind flow, no pretraining
# check_val_every_n_epoch=2 because validation samples the ODE (128 pockets x 25 steps x 2
# Heun evaluations) and an epoch here is only ~19 optimiser steps.
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online nohup ./venvPlixer/bin/python src/train.py \
    experiment=flow_poc2mol_hiqbind \
    task_name=flow_hiqbind_scratch \
    trainer.devices=2 \
    trainer.max_epochs=1200 \
    trainer.check_val_every_n_epoch=2 \
    data.config.batch_size=64 \
    data.config.target_samples_per_batch=512 \
    data.num_workers=12 \
    > "$LOGDIR/hiqbind_scratch.log" 2>&1 &
PID_B=$!
echo "control  pid $PID_B  (GPUs 0-1)"

# ------------------------------------------------------------------------- health check
echo "waiting 300s before the health check..."
sleep 300

FAILED=0
for entry in "stage_a:$PID_A" "hiqbind_scratch:$PID_B"; do
  name=${entry%%:*}; pid=${entry##*:}
  if kill -0 "$pid" 2>/dev/null; then
    echo "OK   $name (pid $pid) alive"
    tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE "train/batch_loss=[0-9.]+" | tail -1
    # Print the run URL, and say so loudly if there isn't one. These arms were once
    # launched with WANDB_MODE=offline and nobody noticed for half an hour; a live run
    # cannot be flipped to online afterwards, so it has to be caught in the first minutes.
    url=$(tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE "https://wandb.ai/[^ ]*runs/[A-Za-z0-9]+" | tail -1)
    if [ -n "$url" ]; then echo "     W&B: $url"; else
      echo "     W&B: NO URL -- run is offline or the logger failed"; FAILED=1
    fi
  else
    echo "DEAD $name (pid $pid) -- last 25 lines:"
    tr '\r' '\n' < "$LOGDIR/$name.log" | grep -vE "^\s*$" | tail -25
    FAILED=1
  fi
done

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
[ "$FAILED" -eq 0 ] && echo "=== BOTH ARMS RUNNING ($STAMP) ===" || echo "=== LAUNCH FAILED ==="
exit $FAILED
