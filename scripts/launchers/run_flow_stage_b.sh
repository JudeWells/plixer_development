#!/usr/bin/env bash
# Generative Poc2Mol -- stage B. Waits for the ZINC pretrain to finish, takes its best
# checkpoint, and starts three arms on the GPUs it frees (2-7). GPUs 0-1 stay with the
# from-scratch control, which is still running.
#
# The four arms form an ablation ladder, each differing from its neighbour by ONE thing:
#
#   control      (already running, GPUs 0-1)  no pretraining, pocket-only
#   pocket_only  = control + stage-A weights            -> the value of PRETRAINING
#   mixed        = pocket_only + 50% ZINC in training   -> the value of KEEPING ZINC
#   mixed_shift3 = mixed + time_shift 3.0               -> the timestep-schedule sweep
#
# So pocket_only carries the mixed arms' LR and warmup (5e-5, 1500 steps) rather than its
# own config's, or the "value of mixing" comparison would confound the mixture with the
# schedule.
#
# Effective batch is 512 everywhere (64 per rank x 2 ranks x 4 accumulation), and one epoch
# contains ~9,872 POCKET samples in every arm -- a mixed epoch is twice the samples but half
# of them are ZINC. So the arms are comparable at matched EPOCH numbers, which is what the
# val/sample/dice curves should be read against.
set -u
cd ~/plixer_outer/plixer || exit 1

echo "waiting for the ZINC pretrain to finish..."
while pgrep -f "experiment=flow_zinc_pretrain" > /dev/null 2>&1; do sleep 120; done
echo "stage A finished at $(date -u +%Y-%m-%d\ %H:%M:%S)"
sleep 30

CKPT_DIR=$(ls -td logs/flow_zinc_pretrain/runs/*/checkpoints 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then echo "ERROR: no stage A checkpoint directory"; exit 1; fi
# Best = lowest val/loss, which is in the filename (step_XXXXXX_valloss_X.XXXXX.ckpt).
BEST=$(ls "$CKPT_DIR"/step_*valloss_*.ckpt 2>/dev/null \
       | sed 's/.*valloss_//; s/\.ckpt$//' | sort -n | head -1)
CKPT=$(ls "$CKPT_DIR"/step_*valloss_"$BEST".ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "ERROR: no scored checkpoint in $CKPT_DIR"; ls -l "$CKPT_DIR"; exit 1; fi
CKPT=$(readlink -f "$CKPT")
echo "stage A best checkpoint: $CKPT  (val/loss $BEST)"

# The stage-A weights are the whole point of this stage; refuse to launch without them
# rather than silently training four from-scratch arms overnight.
if ! ./venvPlixer/bin/python -c "
import sys, torch
sd = torch.load('$CKPT', map_location='cpu').get('state_dict', {})
n = sum(v.numel() for v in sd.values())
print(f'  checkpoint holds {len(sd)} tensors, {n/1e6:.1f}M parameters')
sys.exit(0 if n > 1e6 else 1)
"; then echo "ERROR: stage A checkpoint looks empty"; exit 1; fi

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=/tmp/flow_stage_b_${STAMP}
mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

launch () {  # name  gpus  experiment  extra-overrides...
  local name=$1; local gpus=$2; local experiment=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpus WANDB_MODE=online nohup ./venvPlixer/bin/python src/train.py \
      experiment="$experiment" \
      task_name=flow_stage_b_${name} \
      init_weights_from="$CKPT" \
      trainer.devices=2 \
      trainer.max_epochs=600 \
      trainer.check_val_every_n_epoch=2 \
      data.config.batch_size=64 \
      data.config.target_samples_per_batch=512 \
      data.num_workers=10 \
      "$@" \
      > "$LOGDIR/${name}.log" 2>&1 &
  eval "PID_${name}=$!"
  echo "$name pid $(eval echo \$PID_${name}) (GPUs $gpus, $experiment)"
}

# The ablation: pretrained but pocket-only, carrying the mixed arms' schedule.
launch pocket_only 2,3 flow_poc2mol_hiqbind \
    model.lr=5e-5 model.scheduler.num_warmup_steps=1500
sleep 25
# The main run.
launch mixed 4,5 flow_stage_b_mixed
sleep 25
# Timestep-schedule sweep on top of the main run.
launch mixed_shift3 6,7 flow_stage_b_mixed model.time_shift=3.0

echo "waiting 420s before the health check..."
sleep 420

FAILED=0
for name in pocket_only mixed mixed_shift3; do
  pid=$(eval echo \$PID_${name})
  if kill -0 "$pid" 2>/dev/null; then
    echo "OK   $name (pid $pid) $(tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE 'train/batch_loss=[0-9.]+' | tail -1)"
    url=$(tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE "https://wandb.ai/[^ ]*runs/[A-Za-z0-9]+" | tail -1)
    if [ -n "$url" ]; then echo "     W&B: $url"; else
      echo "     W&B: NO URL -- run is offline or the logger failed"; FAILED=1
    fi
    # A silent failure mode worth catching early: init_weights_from loading nothing would
    # leave the arm training from scratch, which looks identical in the progress bar.
    grep -q "state_dict loaded cleanly" "$LOGDIR/$name.log" \
      && echo "     stage-A weights loaded cleanly" \
      || echo "     WARNING: no clean state_dict load line -- check $LOGDIR/$name.log"
  else
    echo "DEAD $name (pid $pid) -- last 25 lines:"
    tr '\r' '\n' < "$LOGDIR/$name.log" | grep -vE "^\s*$" | tail -25
    FAILED=1
  fi
done

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
if [ "$FAILED" -eq 0 ]; then
  echo "=== STAGE B RUNNING ($STAMP) ==="
  echo "compare on val/sample/dice (pinned to the POCKET set) at matched EPOCH numbers."
  echo "per-source curves: val/hiqbind/loss vs val/zinc/loss, train/loss_pocket vs train/loss_ligand_only."
  echo "yardstick 0.5027 (regression, full 1019-pocket val split). Then sweep guidance:"
  echo "  ./venvPlixer/bin/python scripts/adhoc_analysis/poc2mol_flow_eval.py \\"
  echo "      --ckpt <best.ckpt> --n_batches 16 --guidance 1.0 1.5 2.0 3.0 --steps 25 50 --n_samples 4"
else
  echo "=== STAGE B LAUNCH FAILED ==="
fi
exit $FAILED
