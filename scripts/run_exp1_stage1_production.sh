#!/usr/bin/env bash
# Experiment 1, stage 1 PRODUCTION -- two arms differing only in voxel aggregation.
#
#   maxagg : voxel_aggregation=max  (current default)
#   sumagg : voxel_aggregation=sum  (additive)
#
# Why the aggregation matters: `max` is non-injective under overlap. Bonded heavy atoms sit
# 1.9 voxels apart with 2.3-voxel vdW radii, so their spheres merge and max(1,1)=1 makes two
# atoms indistinguishable from one. ~34% of occupied voxels saturate and the carbon skeleton
# collapses to a single connected blob (0.07 blobs per atom). That information is destroyed
# BEFORE the grid is sampled, so finer voxels cannot recover it -- measured: halving
# vox_size to 0.375 A leaves both saturation and separability unchanged.
# `sum` accumulates instead, giving a count field that reaches ~5.6, from which atom
# positions are recoverable. Narrowing the kernel as well (voxel_radius_scale ~0.3) would
# make atoms directly separable (0.91 blobs/atom), but that is a second variable and is
# deliberately NOT changed here.
#
# Schedule (both arms identical): lr 3e-4, warmup-stable-decay, 500k steps total --
# 2k warmup, 298k stable, 200k decay to 0.03x peak (i.e. down to 9e-6). accumulation 1,
# batch 128/GPU, 4 GPUs => effective batch 512.
#
# Why 0.03 and not 0.1: run ciqwntcy accidentally measured the LR/loss relationship by
# ramping 1e-5 -> 1e-4 under a warmup. val/loss rose monotonically with LR (0.1758 at
# 2.4e-5 -> 0.198 at 1e-4), i.e. the model was sitting at the optimiser noise floor, whose
# excess loss is ~linear in LR. Fitting L* + c*lr to those two points extrapolates to
# L* ~ 0.169 at zero LR, and the observed minimum was still at the lowest LR sampled --
# so there was no sign of a floor being reached by 3e-5. 0.03x lands at 9e-6, below the
# best LR that run actually sampled.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/exp1_stage1_prod/${STAMP}"; mkdir -p "${DIR}"
echo "${DIR}" > /tmp/exp1_prod_dir.txt
echo "stage-1 production: ${DIR}"

launch () {
  local name="$1" agg="$2" gpus="$3"
  CUDA_VISIBLE_DEVICES="${gpus}" nohup venvPlixer/bin/python src/train.py \
    experiment=exp1_zinc_protein \
    task_name="exp1_s1_prod_${name}" \
    trainer=ddp trainer.devices=4 \
    data.num_workers=14 \
    data.config.batch_size=128 \
    data.config.target_samples_per_batch=512 \
    data.config.voxel_aggregation="${agg}" \
    model.config.lr=3e-4 \
    model.config.scheduler.num_warmup_steps=2000 \
    model.config.scheduler.num_stable_steps=298000 \
    model.config.scheduler.num_decay_steps=200000 \
    model.config.scheduler.min_lr_ratio=0.03 \
    +trainer.max_steps=500000 \
    trainer.max_epochs=1000 \
    trainer.val_check_interval=5000 \
    +trainer.num_sanity_val_steps=0 \
    callbacks.model_checkpoint.save_weights_only=False \
    callbacks.model_checkpoint.save_top_k=2 \
    callbacks.model_checkpoint.save_last=True \
    logger.wandb.group="exp1_stage1_prod_${STAMP}" \
    logger.wandb.name="s1_prod_${name}" \
    > "${DIR}/${name}.log" 2>&1 &
  echo $! > "${DIR}/${name}.pid"
  echo "  ${name}: aggregation=${agg}  gpus=${gpus}"
}
# Verify each arm actually survived startup. The first attempt at these runs died in Hydra
# config composition within a second of launch and left 8 idle GPUs; because the launcher
# only backgrounded the process and never checked it, nothing reported the failure.
check () {
  local name="$1" pid; pid="$(cat "${DIR}/${name}.pid")"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "  FAILED ${name} (pid ${pid} is gone). Last lines:"
    tail -5 "${DIR}/${name}.log" | sed 's/^/    /'
    return 1
  fi
  echo "  ok ${name} (pid ${pid} alive)"
}

launch maxagg max 0,1,2,3
sleep 30
launch sumagg sum 4,5,6,7

echo "waiting 180s for both arms to reach steady state..."
sleep 180
FAILED=0
check maxagg || FAILED=1
check sumagg || FAILED=1
if [ "${FAILED}" -ne 0 ]; then
  echo "AT LEAST ONE ARM FAILED TO START -- see logs in ${DIR}"
  exit 1
fi
echo "both arms running; logs in ${DIR}"
