#!/usr/bin/env bash
# Sweep 2: extend the alpha range, AND get an error bar.
# Sweep 1 showed over-emission falling monotonically in alpha with on_target RISING, but the
# discrimination gain (a30 +0.014 over control) is only ~1.5-2 SE. Two of these four arms are
# therefore SEED REPLICATES of sweep 1's endpoints -- without them "alpha 30 helps" is not
# separable from seed noise, and the whole sweep stays suggestive.
cd ~/plixer_outer/plixer || exit 1
STAMP=$(date +%Y%m%d_%H%M%S); LOGDIR="logs/poc2mol_bce_sweep2/${STAMP}"; mkdir -p "$LOGDIR"
GROUP="poc2mol_bce_sweep2_${STAMP}"
PY=./venvPlixer/bin/python
COMMON=( experiment=exp1_poc2mol_cons4 data=poc2mol_hiqbind_v2_11ch
  model.config.out_channels=11 model.config.layer_order=gcrd model.config.dropout_prob=0.1
  trainer=ddp trainer.devices=2 data.num_workers=12
  data.config.batch_size=384 data.config.target_samples_per_batch=1536
  trainer.max_epochs=450 +trainer.num_sanity_val_steps=0
  callbacks.model_checkpoint.save_top_k=2 callbacks.model_checkpoint.save_last=True )
launch () { local n="$1" g="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$g" WANDB_MODE=online nohup $PY src/train.py "${COMMON[@]}" "$@" \
    task_name="poc2mol_sw2_${n}" logger.wandb.group="${GROUP}" logger.wandb.name="${n}" \
    > "${LOGDIR}/${n}.log" 2>&1 & echo $!; }
declare -A P
P[a100]=$(launch a100 0,1 model.loss.alpha=100.0 seed=42)
P[a300]=$(launch a300 2,3 model.loss.alpha=300.0 seed=42)
P[a30_s1]=$(launch a30_s1 4,5 model.loss.alpha=30.0 seed=1)      # replicate of sweep-1 a30
P[a1_s1]=$(launch a1_s1  6,7 model.loss.alpha=1.0  seed=1)       # replicate of sweep-1 control
echo "group ${GROUP}"; echo "logs ${LOGDIR}"
for k in "${!P[@]}"; do echo "  $k pid=${P[$k]}"; done
sleep 300; fail=0
for k in "${!P[@]}"; do
  kill -0 "${P[$k]}" 2>/dev/null && echo "OK   $k" || { echo "DEAD $k"; tail -n 20 "${LOGDIR}/$k.log"; fail=1; }
done
echo "SWEEP2_LAUNCH_DONE fail=$fail"
