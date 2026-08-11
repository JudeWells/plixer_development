#!/usr/bin/env bash
# Stage 1 (ZINC ligand-only pretraining) on parquet_v2 -- both channel schemes, 4 GPUs each.
#
# Mirrors the Poc2Mol v2 A/B running on nebius2 (poc2mol_hiqbind_v2_{9ch,11ch}) so that
# whichever ligand-channel scheme wins upstream has a matching stage-1 decoder ready. The
# encoder's patch-embedding conv width is set by the channel count, so a stage-1 checkpoint
# CANNOT be reused across schemes -- running only one arm would mean a ~87 h wait after the
# Poc2Mol result lands.
#
# The 9ch arm doubles as the control for coordinate precision: it differs from the published
# stage-1 setup only in float32 coords (CLAUDE.md §15c), so it separates what the precision
# fix alone buys from what the channel scheme adds.
#
# Schedule is copied verbatim from the §9c production runs (val/loss 0.019, 63% exact match),
# so this changes the data against a known-good recipe rather than two things at once.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=./venvPlixer/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/exp1_stage1_v2/${STAMP}"
mkdir -p "$LOGDIR"
GROUP="exp1_zinc_v2_${STAMP}"

launch () {                       # $1 experiment  $2 gpus  $3 short name
  # trainer/ddp.yaml asks for devices: 8, but each arm is masked to 4 GPUs, and Lightning
  # errors rather than clamping ("You requested gpu: [0..7] but your machine only has
  # [0..3]"). Must be set explicitly per arm.
  CUDA_VISIBLE_DEVICES="$2" WANDB_MODE=online nohup $PY src/train.py \
      experiment="$1" \
      trainer.devices=4 \
      logger.wandb.group="${GROUP}" \
      logger.wandb.name="$3" \
      > "${LOGDIR}/$3.log" 2>&1 &
  echo $!
}

PID_9=$(launch exp1_zinc_v2_9ch  0,1,2,3 v2_9ch)
PID_11=$(launch exp1_zinc_v2_11ch 4,5,6,7 v2_11ch)
echo "launched  v2_9ch pid=$PID_9   v2_11ch pid=$PID_11"
echo "group     ${GROUP}"
echo "logs      ${LOGDIR}"

# The first attempt at the §9c production runs died within ONE SECOND on a Hydra composition
# error and was reported as running, because the launcher only backgrounded the process and
# never looked back. All 8 GPUs sat idle. Never launch without confirming the PIDs survive.
echo "waiting 240 s to confirm both survive startup..."
sleep 240
fail=0
for pair in "$PID_9:v2_9ch" "$PID_11:v2_11ch"; do
  pid="${pair%%:*}"; name="${pair##*:}"
  if kill -0 "$pid" 2>/dev/null; then
    echo "OK   $name (pid $pid) alive"
  else
    echo "DEAD $name (pid $pid) -- log tail:"
    tail -25 "${LOGDIR}/${name}.log"
    fail=1
  fi
done
exit $fail
