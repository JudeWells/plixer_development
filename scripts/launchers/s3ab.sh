#!/usr/bin/env bash
# Stage-3 A/B: matched 9ch and 11ch pipelines (Poc2Mol + decoder), 4 GPUs each.
# Decides the channel scheme on the DOWNSTREAM metric, since §18a showed val/loss is not
# comparable across schemes and the pose-free composition readout ties them.
set -uo pipefail
cd ~/plixer_outer/plixer || exit 1
PY=./venvPlixer/bin/python
STAMP=$(date +%Y%m%d_%H%M%S); LOGDIR="logs/exp1_s3_v2/${STAMP}"; mkdir -p "$LOGDIR"
GROUP="exp1_s3_v2b_${STAMP}"
launch(){ local s="$1" g="$2"
  CUDA_VISIBLE_DEVICES="$g" WANDB_MODE=online nohup $PY src/train.py \
    experiment=exp1_s3_v2_${s}ch trainer.devices=4 \
    logger.wandb.group="${GROUP}" logger.wandb.name="v2_${s}ch" \
    > "${LOGDIR}/${s}ch.log" 2>&1 & echo $!; }
P9=$(launch 9 0,1,2,3); P11=$(launch 11 4,5,6,7)
echo "group ${GROUP}"; echo "logs ${LOGDIR}"; echo "9ch pid=$P9  11ch pid=$P11"
sleep 420; fail=0
for pr in "$P9:9ch" "$P11:11ch"; do p="${pr%%:*}"; n="${pr##*:}"
  kill -0 "$p" 2>/dev/null && echo "OK   $n" || { echo "DEAD $n"; tail -n 25 "${LOGDIR}/${n}.log"; fail=1; }
done
echo "S3AB_LAUNCH_DONE fail=$fail group=${GROUP}"
