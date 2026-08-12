#!/usr/bin/env bash
# Timestep-shift sweep BELOW 0.33, on a flat learning rate.
#
# 0.33 was the best value tried on 2026-08-11 and sat at the EDGE of the explored range
# (3.0 / 1.0 / 0.33), so the optimum may well be lower. Lower shift = more training mass at
# the noisy end of the path: frac(t < 0.25) is 50.5% at 0.33, 69.6% at 0.2, 88.6% at 0.1,
# 97.1% at 0.05.
#
# Flat LR (constant_with_warmup), because the previous arms annealed to their floor while
# val/loss was still falling -- they were stopped by the schedule, not by convergence.
# Early stopping on val/mean/dice with patience 40 ends each arm on an observed plateau.
#
# 0.33 is re-run under the new schedule so all four are internally comparable; the old
# 0.33 run is NOT a valid reference here because its LR decayed.
set -u
cd ~/plixer_outer/plixer || exit 1

CKA=$(ls -t logs/flow_zinc_pretrain/runs/*/checkpoints/step_*valloss*.ckpt 2>/dev/null | head -1)
[ -z "$CKA" ] && { echo "ERROR: no stage-A checkpoint"; exit 1; }
CKA=$(readlink -f "$CKA")
echo "stage-A weights: $CKA"

STAMP=$(date -u +%Y%m%d_%H%M%S); LOGDIR=/tmp/shift_sweep_${STAMP}; mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

i=0
for SHIFT in 0.33 0.2 0.1 0.05; do
  GPUS="$((i*2)),$((i*2+1))"
  NAME="flow_shift_${SHIFT/./p}"
  CUDA_VISIBLE_DEVICES=$GPUS WANDB_MODE=online nohup ./venvPlixer/bin/python src/train.py \
      experiment=flow_shift_sweep task_name="$NAME" \
      model.time_shift="$SHIFT" init_weights_from="$CKA" \
      trainer.devices=2 data.config.batch_size=64 \
      data.config.target_samples_per_batch=512 data.num_workers=10 \
      > "$LOGDIR/${NAME}.log" 2>&1 &
  eval "PID_${i}=$!"; eval "NAME_${i}=$NAME"
  echo "$NAME pid $(eval echo \$PID_${i}) (GPUs $GPUS, shift $SHIFT)"
  i=$((i+1)); sleep 25
done

echo "health check in 480s..."
sleep 480
FAILED=0
for j in 0 1 2 3; do
  pid=$(eval echo \$PID_${j}); name=$(eval echo \$NAME_${j})
  if kill -0 "$pid" 2>/dev/null; then
    echo "OK   $name"
    tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE "https://wandb.ai/[^ ]*runs/[A-Za-z0-9]+" | tail -1 | sed 's/^/     W&B: /'
    grep -q "state_dict loaded cleanly" "$LOGDIR/$name.log" && echo "     stage-A weights loaded" \
      || { echo "     WARNING: no clean state_dict load"; FAILED=1; }
  else
    echo "DEAD $name:"; tr '\r' '\n' < "$LOGDIR/$name.log" | grep -vE "^\s*$" | tail -20; FAILED=1
  fi
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
[ "$FAILED" -eq 0 ] && echo "=== SHIFT SWEEP RUNNING ($STAMP) ===" || echo "=== LAUNCH FAILED ==="
exit $FAILED
