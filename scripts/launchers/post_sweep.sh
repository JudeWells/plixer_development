#!/usr/bin/env bash
# After the BCE sweep finishes, evaluate every arm's BEST checkpoint on the two diagnostics
# that are NOT floor-bound. §18h showed val/loss can move opposite to discrimination, so the
# arms must be ranked on these, not on the training curve.
cd ~/plixer_outer/plixer || exit 1
LOGDIR=logs/poc2mol_bce_sweep/20260811_000434
echo "waiting for sweep arms to finish..."
while pgrep -f "venvPlixer/bin/python src/train.py" > /dev/null 2>&1; do sleep 60; done
echo "all arms finished at $(date -u +%H:%M:%S)"
sleep 30

i=0
for arm in a1_ctrl a10 a30 sched; do
  d=$(ls -d logs/poc2mol_sweep_${arm}/runs/*/checkpoints 2>/dev/null | head -1)
  rd=$(dirname "$d")
  # best = lowest val/loss in the filename. Recorded for reference only; §18h is the reason
  # we do not TRUST it, but it is still the checkpoint the pipeline would have shipped.
  best=$(ls "$d"/epoch_*valloss*.ckpt 2>/dev/null | sed 's/.*valloss_//' | sort -n | head -1)
  ck=$(ls "$d"/epoch_*valloss_${best} 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "$arm: NO CHECKPOINT"; continue; }
  cp "$ck" /tmp/sw_${arm}.ckpt
  echo "$arm -> $(basename $ck)"
  CUDA_VISIBLE_DEVICES=$i nohup ./venvPlixer/bin/python scripts/adhoc_analysis/poc2mol_scheme_discrimination.py \
      --run_dir "$rd" --checkpoint /tmp/sw_${arm}.ckpt --output /tmp/sw_disc_${arm}.json \
      > /tmp/sw_disc_${arm}.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$((i+1)) nohup ./venvPlixer/bin/python scripts/adhoc_analysis/poc2mol_channel_emission.py \
      --run_dir "$rd" --checkpoint /tmp/sw_${arm}.ckpt --output /tmp/sw_emit_${arm}.json \
      > /tmp/sw_emit_${arm}.log 2>&1 &
  i=$((i+2)); [ $i -ge 8 ] && { wait; i=0; }
done
wait
echo "=== ANALYSIS COMPLETE $(date -u +%H:%M:%S) ==="
for arm in a1_ctrl a10 a30 sched; do
  echo "########## $arm"
  grep -A 3 "=== discrimination" /tmp/sw_disc_${arm}.log 2>/dev/null | tail -3
  grep -A 14 "^channel" /tmp/sw_emit_${arm}.log 2>/dev/null | head -15
done
