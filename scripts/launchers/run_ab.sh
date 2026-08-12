#!/usr/bin/env bash
# Experiment 1 A/B: do protein channels help the decoder?
#
# Both arms start from the SAME stage-1 model. The baseline's 9-channel checkpoint is the
# 14-channel one with the patch-embedding conv sliced to its first 9 (ligand) input channels --
# verified bit-identical in function to the 14ch model with the protein zeroed. Using the old
# genuine 9ch stage-1 checkpoint instead would have compared stage-1 quality (0.19 vs 0.019),
# not protein channels.
#
# Schedule is sized from the DATA, not inherited. The previous run's val/poc2mol/loss bottomed at
# ~768k complex samples seen. At effective batch 300 with prob_poc2mol 0.5 that is 150 complex
# samples/step, i.e. ~5,100 steps -- so 8k total with the decay covering 4k-8k puts the annealed
# region on top of the expected optimum instead of 50k steps past it.
#
# Monitor is val/poc2mol/loss, NOT val/loss: val/loss pools in ~104k ZINC samples that keep
# improving while the pocket task degrades, so it can drift away from the deployed metric.
# val_check_interval 250 (was 2000) so the optimum is not missed; limit_val_batches keeps that
# affordable.
set -uo pipefail
cd /home/judewells/plixer_outer/plixer || exit 1
export PROJECT_ROOT="$(pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/exp1_ab/${STAMP}"; mkdir -p "${DIR}"
echo "${DIR}" > /tmp/exp1_ab_dir.txt
echo "exp1 A/B: ${DIR}"

common_overrides () {
  echo "trainer.devices=2 \
    data.num_workers=12 \
    data.config.batch_size=150 \
    data.config.target_samples_per_batch=300 \
    data.config.voxel_aggregation=max \
    data.train_dataset.prob_poc2mol=0.5 \
    data.predicted_ramp_start_step=200 \
    data.predicted_ramp_end_step=2000 \
    model.config.lr=5e-5 \
    model.n_samples_for_validity_testing=100 \
    model.config.scheduler.num_warmup_steps=300 \
    model.config.scheduler.num_stable_steps=3700 \
    model.config.scheduler.num_decay_steps=4000 \
    model.config.scheduler.min_lr_ratio=0.03 \
    +trainer.max_steps=8000 \
    trainer.max_epochs=10000 \
    trainer.val_check_interval=250 \
    +trainer.limit_val_batches=10 \
    +trainer.num_sanity_val_steps=0 \
    callbacks.model_checkpoint.save_top_k=5 \
    seed=42"
}

launch () {
  local name="$1" experiment="$2" ckpt="$3" gpus="$4"
  local ov="experiment=${experiment} task_name=exp1_ab_${name} init_weights_from=${ckpt} $(common_overrides)"
  if ! venvPlixer/bin/python src/train.py --cfg job --resolve ${ov} \
       logger.wandb.group=dry logger.wandb.name=dry > "${DIR}/${name}.dry" 2>&1; then
    echo "DRY RUN FAILED for ${name}:"; tail -6 "${DIR}/${name}.dry"; return 1
  fi
  echo "  ${name}: dry run ok (num_channels=$(grep -oE 'num_channels: [0-9]+' ${DIR}/${name}.dry | head -1 | grep -oE '[0-9]+'))"
  CUDA_VISIBLE_DEVICES="${gpus}" nohup venvPlixer/bin/python src/train.py ${ov} \
    logger.wandb.group="exp1_ab_${STAMP}" \
    logger.wandb.name="ab_${name}_${STAMP}" \
    > "${DIR}/${name}.log" 2>&1 &
  echo $! > "${DIR}/${name}.pid"
  echo "  ${name}: launched pid $(cat ${DIR}/${name}.pid) on GPUs ${gpus}"
}

launch protein  exp1_s3_protein  checkpoints/s1_maxagg_last.ckpt      0,1 || exit 1
sleep 20
launch baseline exp1_s3_baseline checkpoints/s1_maxagg_last_9ch.ckpt  2,3 || exit 1

echo "waiting 420s (covers startup and the first validation at step 250)..."
sleep 420
FAIL=0
for n in protein baseline; do
  p=$(cat "${DIR}/${n}.pid")
  if kill -0 "$p" 2>/dev/null; then
    echo "  ok ${n} (pid $p)"
    grep -ho "https://wandb.ai/[^ ]*runs/[^ ]*" "${DIR}/${n}.log" | sort -u | sed 's/^/     /'
    tr '\r' '\n' < "${DIR}/${n}.log" | grep -o "Epoch [0-9]*:.*it/s.*" | tail -1 | cut -c1-120 | sed 's/^/     /'
  else
    echo "  FAILED ${n}"; tail -20 "${DIR}/${n}.log"; FAIL=1
  fi
done
echo "--- checkpoints written so far ---"
ls logs/exp1_ab_*/runs/*/checkpoints/ 2>/dev/null | head
exit $FAIL
