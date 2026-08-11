#!/usr/bin/env bash
# Does a better-CALIBRATED Poc2Mol help the decoder? The claim §18k made and §18l left untested.
#
# All four arms share the SAME 11-channel decoder init (s1_v2_11ch.ckpt) and the same schedule.
# The ONLY variable is which Poc2Mol checkpoint supplies the density, differing in the BCE
# weight it was trained with. §18j: alpha takes per-channel over-emission from 1.5-6.5x down to
# ~1.0-1.9x, at the cost of monotonically worse Dice (+1.2% real error at a10, +11.6% at a100).
# The hypothesis is that stage 1 pretrains the decoder on TRUE grids (ratio 1.0), so density
# carrying 2-6x that mass is a distribution shift the decoder must absorb.
#
# Schedule fixed vs the previous A/B: decay now starts at step 2000, on the observed optimum
# (~1500-3000). Previously it started at 6000 and early stopping fired first, so no arm ever
# saw the anneal.
set -uo pipefail
cd ~/plixer_outer/plixer || exit 1
PY=./venvPlixer/bin/python
STAMP=$(date +%Y%m%d_%H%M%S); LOGDIR="logs/exp1_s3_bce/${STAMP}"; mkdir -p "$LOGDIR"
GROUP="exp1_s3_bce_${STAMP}"
launch(){ local a="$1" g="$2"
  CUDA_VISIBLE_DEVICES="$g" WANDB_MODE=online nohup $PY src/train.py \
    experiment=exp1_s3_v2_11ch trainer.devices=2 \
    data.poc2mol_ckpt_path=checkpoints/poc2mol_bce/poc2mol_11ch_${a}.ckpt \
    data.config.target_samples_per_batch=256 \
    task_name="s3_bce_${a}" \
    logger.wandb.group="${GROUP}" logger.wandb.name="${a}" \
    > "${LOGDIR}/${a}.log" 2>&1 & echo $!; }
declare -A P
P[a1_ctrl]=$(launch a1_ctrl 0,1); P[a10]=$(launch a10 2,3)
P[a30]=$(launch a30 4,5);         P[a100]=$(launch a100 6,7)
echo "group ${GROUP}"; echo "logs ${LOGDIR}"
for k in "${!P[@]}"; do echo "  $k pid=${P[$k]}"; done
sleep 420; fail=0
for k in "${!P[@]}"; do kill -0 "${P[$k]}" 2>/dev/null && echo "OK   $k" || { echo "DEAD $k"; tail -n 25 "${LOGDIR}/$k.log"; fail=1; }; done
echo "S3BCE_LAUNCH_DONE fail=$fail group=${GROUP}"
