#!/usr/bin/env bash
# Overnight GPU queue. Each watcher waits for a specific run to exit, then starts its
# successor on the GPUs that run frees. Deterministic pairing rather than dynamic GPU
# discovery, so two watchers can never claim the same card.
#
#   GPUs 2-3  flow_stage_b_pocket_only finishes at epoch 600 (still rising at +0.007
#             dice/100ep when this was written) -> continue it for another 600 epochs from
#             its own best checkpoint, with a fresh cosine schedule.
#   GPUs 0-1  s3_flow_11ch finishes (max_steps 4000 or early stopping) -> run the
#             MULTI-HYPOTHESIS evaluation, which is the thing only a generative model can
#             do: decode several independent voxel draws per pocket and aggregate.
#
# A status line for every arm is appended to the log every 20 minutes so the morning has a
# trail even if nothing else survives.
set -u
cd ~/plixer_outer/plixer || exit 1
LOG=/tmp/overnight_queue.log
say () { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
say "queue started"

# ------------------------------------------------------------------ status logger
(
  while true; do
    sleep 1200
    {
      echo "----- $(date -u +%H:%M:%S)"
      for n in s3_flow_11ch s3_regression_11ch flow_stage_b_pocket_only \
               flow_stage_b_mixed_shift033 flow_stage_b_pocket_only_x2; do
        c=$(pgrep -cf "task_name=$n" 2>/dev/null || echo 0)
        [ "$c" -gt 0 ] && echo "  $n: $c procs"
      done
      nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
    } >> "$LOG"
  done
) &

# ------------------------------------------- GPUs 2-3: continue the best flow arm
(
  while pgrep -f "task_name=flow_stage_b_pocket_only " > /dev/null 2>&1; do sleep 180; done
  sleep 60
  CK=$(ls -t logs/flow_stage_b_pocket_only/runs/*/checkpoints/epoch_*dice*.ckpt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then say "no pocket_only checkpoint; skipping continuation"; exit 0; fi
  CK=$(readlink -f "$CK")
  say "pocket_only finished; continuing from $(basename "$CK") on GPUs 2-3"
  CUDA_VISIBLE_DEVICES=2,3 WANDB_MODE=online nohup ./venvPlixer/bin/python src/train.py \
      experiment=flow_poc2mol_hiqbind task_name=flow_stage_b_pocket_only_x2 \
      init_weights_from="$CK" trainer.devices=2 trainer.max_epochs=600 \
      trainer.check_val_every_n_epoch=2 data.config.batch_size=64 \
      data.config.target_samples_per_batch=512 data.num_workers=10 \
      model.lr=3e-5 model.scheduler.num_warmup_steps=500 \
      > /tmp/pocket_only_x2.log 2>&1 &
  sleep 420
  if pgrep -f "task_name=flow_stage_b_pocket_only_x2" > /dev/null; then
    say "pocket_only_x2 running: $(tr '\r' '\n' < /tmp/pocket_only_x2.log | grep -oE 'https://wandb.ai/[^ ]*runs/[A-Za-z0-9]+' | tail -1)"
  else
    say "pocket_only_x2 FAILED: $(tr '\r' '\n' < /tmp/pocket_only_x2.log | grep -vE '^\s*$' | tail -3)"
  fi
) &

# ------------------------- GPUs 0-1: multi-hypothesis eval once the flow decoder is done
(
  while pgrep -f "task_name=s3_flow_11ch" > /dev/null 2>&1; do sleep 180; done
  sleep 60
  DEC=$(ls -t logs/s3_flow_11ch/runs/*/checkpoints/step_*auc*.ckpt 2>/dev/null | head -1)
  FLOW=$(ls -t logs/flow_stage_b_pocket_only/runs/*/checkpoints/epoch_*dice*.ckpt 2>/dev/null | head -1)
  if [ -z "$DEC" ]; then say "s3_flow produced no scored checkpoint; skipping multi-hypothesis eval"; exit 0; fi
  say "s3_flow finished; multi-hypothesis eval with $(basename "$DEC")"
  CUDA_VISIBLE_DEVICES=0,1 nohup ./venvPlixer/bin/python \
      scripts/adhoc_analysis/flow_multi_hypothesis_eval.py \
      --decoder_ckpt "$DEC" --flow_ckpt "$(readlink -f "$FLOW")" \
      --n_hypotheses 8 --n_batches 8 --output /tmp/multi_hypothesis.json \
      > /tmp/multi_hypothesis.log 2>&1 &
  sleep 600
  say "multi-hypothesis eval: $(tail -3 /tmp/multi_hypothesis.log | tr '\n' ' ')"
) &

wait
