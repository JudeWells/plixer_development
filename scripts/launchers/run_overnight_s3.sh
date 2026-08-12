#!/usr/bin/env bash
# Overnight allocation, 2026-08-11 22:30 -> 06:30. All 8 GPUs.
#
# Reprioritised toward the FULL PIPELINE, because reconstruction Dice cannot settle this
# branch: it is maximised by the conditional mean, which is exactly what the regression
# Poc2Mol is trained to emit. The question that matters is whether the generative density
# decodes into better SMILES -- Tanimoto, and likelihood ranking of true binders.
#
#   GPUs 0-1  s3_flow_11ch        decoder fine-tuned on GENERATIVE Poc2Mol density
#   GPUs 2-3  flow_stage_b_pocket_only   (KEPT -- still rising, +0.007 dice/100ep)
#   GPUs 4-5  s3_regression_11ch  the matched control: same everything, regression density
#   GPUs 6-7  flow_stage_b_mixed_shift033 (KEPT -- corrected timestep direction, promising)
#
# STOPPED: flow_hiqbind_scratch (flat at 0.211 for ~300 epochs; its job as the
# no-pretraining control is done) and flow_stage_b_mixed (rising but the least
# distinguished of the three flow arms). Both keep their W&B history.
#
# The two s3 arms differ ONLY in which Poc2Mol supplies the ligand density, so any
# difference in val/likelihood_auc_znorm or val/poc2mol/tanimoto is attributable to that.
set -u
cd ~/plixer_outer/plixer || exit 1

FLOW_CKPT=$(ls -t logs/flow_stage_b_pocket_only/runs/*/checkpoints/epoch_*dice*.ckpt 2>/dev/null | head -1)
[ -z "$FLOW_CKPT" ] && { echo "ERROR: no stage-B flow checkpoint"; exit 1; }
FLOW_CKPT=$(readlink -f "$FLOW_CKPT")
echo "generative density from: $FLOW_CKPT"
echo "regression density from: checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt"

STAMP=$(date -u +%Y%m%d_%H%M%S)
LOGDIR=/tmp/overnight_${STAMP}; mkdir -p "$LOGDIR"; echo "logs -> $LOGDIR"

# --- free GPUs 0-1 and 4-5 ------------------------------------------------------------
for name in flow_hiqbind_scratch flow_stage_b_mixed; do
  pgrep -f "task_name=${name}\b" > /tmp/_stop_$$.txt 2>/dev/null
  # \b so flow_stage_b_mixed does not match flow_stage_b_mixed_shift033
  pgrep -f "task_name=${name} " >> /tmp/_stop_$$.txt 2>/dev/null
  sort -u /tmp/_stop_$$.txt | xargs -r kill -9 2>/dev/null
  rm -f /tmp/_stop_$$.txt
  echo "stopped $name"
done
sleep 25
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

launch () {  # name gpus experiment extra...
  local name=$1 gpus=$2 experiment=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpus WANDB_MODE=online nohup ./venvPlixer/bin/python src/train.py \
      experiment="$experiment" task_name="$name" trainer.devices=2 \
      data.num_workers=10 "$@" > "$LOGDIR/${name}.log" 2>&1 &
  eval "PID_${name}=$!"
  echo "$name pid $(eval echo \$PID_${name}) (GPUs $gpus)"
}

# generative_draws=2, not 4: each draw is a full UNet evaluation and CFG doubles it, so
# k=4 would put 8 Poc2Mol forwards in front of every decoder step and halve the number of
# steps this can reach overnight. k=2 costs 4 and gives up little (Dice 0.354 at k=4 vs
# 0.372 at k=16, both far above a single sample's 0.231).
launch s3_flow_11ch 0,1 s3_flow_11ch \
    data.poc2mol_ckpt_path="$FLOW_CKPT" data.generative_draws=2 data.generative_guidance=3.0
sleep 30
launch s3_regression_11ch 4,5 s3_regression_11ch
sleep 30

echo "health check in 480s..."
sleep 480
FAILED=0
for name in s3_flow_11ch s3_regression_11ch; do
  pid=$(eval echo \$PID_${name})
  if kill -0 "$pid" 2>/dev/null; then
    echo "OK   $name (pid $pid)"
    tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE "https://wandb.ai/[^ ]*runs/[A-Za-z0-9]+" | tail -1 | sed 's/^/     W&B: /'
    tr '\r' '\n' < "$LOGDIR/$name.log" | grep -oE "train/loss=[0-9.]+" | tail -1 | sed 's/^/     /'
  else
    echo "DEAD $name -- last 25 lines:"; tr '\r' '\n' < "$LOGDIR/$name.log" | grep -vE "^\s*$" | tail -25
    FAILED=1
  fi
done
for name in flow_stage_b_pocket_only flow_stage_b_mixed_shift033; do
  echo "$name still alive: $(pgrep -cf "task_name=$name") procs"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
[ "$FAILED" -eq 0 ] && echo "=== OVERNIGHT SET RUNNING ($STAMP) ===" || echo "=== LAUNCH FAILED ==="
exit $FAILED
