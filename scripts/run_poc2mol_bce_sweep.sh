#!/usr/bin/env bash
# Poc2Mol sweep: BCE weight, plus one LR-schedule arm. Waits for the in-flight v2 runs to
# finish first, then takes all 8 GPUs.
#
# WHY BCE WEIGHT. §18 measured every ligand channel over-emitting (1.3x carbon, up to 10x
# iodine), with a near-constant smear: fluorine emits 59 units of mass when the ligand has NO
# fluorine and 75 when it does. The cause is structural -- an all-zero target channel makes
# Dice's numerator identically zero, so Dice supplies NO gradient there (§3c). BCE is the only
# term that charges the model for that mass, and at alpha 1.0 it contributes ~0.011 of the
# loss against Dice's ~0.62, i.e. ~56x smaller. alpha 10 and 30 bring it to the same order.
#
# WHAT TO READ. NOT val/loss -- ~76% of it is the Dice floor, a constant no model can improve,
# and the floor also shifts with alpha so cross-arm comparison of it is meaningless. Read the
# new val/emission/* metrics:
#   empty_frac  share of predicted mass in channels the ligand leaves EMPTY. The target.
#   on_target   share of predicted mass landing on real ligand density. Guards against the
#               degenerate win of simply emitting less everywhere, including where it should.
#   ratio       total predicted / total true mass.
# A good result is empty_frac DOWN and on_target UP. empty_frac down with on_target flat or
# down means the model just got quieter, which is not the same as getting sharper.
#
# THE SCHEDULE ARM. The v2 runs warm up over 500 steps, but HiQBind is only 9,872 clusters so
# at effective batch 1536 an epoch is ~6 optimiser steps -- 500 steps is 83 EPOCHS of warmup
# out of a 600-epoch run, after which cosine decays across the entire remainder to 0.5x peak.
# There is no stable phase at all. `sched` tests a short warmup (50 steps ~ 8 epochs) and a
# deeper floor (0.1x).
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=./venvPlixer/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/poc2mol_bce_sweep/${STAMP}"
mkdir -p "$LOGDIR"
GROUP="poc2mol_bce_sweep_${STAMP}"

# ---- wait for the in-flight v2 runs to finish -----------------------------------------
echo "waiting for in-flight poc2mol runs to reach max_epochs..."
while pgrep -f "venvPlixer/bin/python src/train.py" > /dev/null 2>&1; do sleep 60; done
echo "GPUs free at $(date -u +%H:%M:%S)"
sleep 30   # let CUDA contexts tear down before allocating

COMMON=(
  experiment=exp1_poc2mol_cons4
  data=poc2mol_hiqbind_v2_11ch
  model.config.out_channels=11
  model.config.layer_order=gcrd
  model.config.dropout_prob=0.1
  trainer=ddp
  trainer.devices=2
  data.num_workers=12
  data.config.batch_size=384
  data.config.target_samples_per_batch=1536   # -> accumulation 2 on 2 GPUs, effective 1536,
                                              # matched to the 4-GPU v2 runs
  trainer.max_epochs=450                      # past the observed optimum (426 for ch11)
  +trainer.num_sanity_val_steps=0
  callbacks.model_checkpoint.save_top_k=2
  callbacks.model_checkpoint.save_last=True
  seed=42
)

launch () {   # $1 name  $2 gpus  $3.. extra overrides
  local name="$1" gpus="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpus" WANDB_MODE=online nohup $PY src/train.py \
      "${COMMON[@]}" "$@" \
      task_name="poc2mol_sweep_${name}" \
      logger.wandb.group="${GROUP}" logger.wandb.name="${name}" \
      > "${LOGDIR}/${name}.log" 2>&1 &
  echo $!
}

declare -A PIDS
PIDS[a1_ctrl]=$(launch a1_ctrl 0,1 model.loss.alpha=1.0)
PIDS[a10]=$(launch a10     2,3 model.loss.alpha=10.0)
PIDS[a30]=$(launch a30     4,5 model.loss.alpha=30.0)
PIDS[sched]=$(launch sched 6,7 model.loss.alpha=1.0 \
                 model.scheduler.num_warmup_steps=50 model.scheduler.min_lr_rate=0.1)

echo "group ${GROUP}"; echo "logs  ${LOGDIR}"
for k in "${!PIDS[@]}"; do echo "  $k pid=${PIDS[$k]}"; done

# §9c: a launcher that only backgrounds a process and never looks back reported 8 idle GPUs
# as "running" for hours. Always confirm survival.
echo "confirming survival in 300 s..."
sleep 300
fail=0
for k in "${!PIDS[@]}"; do
  if kill -0 "${PIDS[$k]}" 2>/dev/null; then echo "OK   $k"; else
    echo "DEAD $k -- log tail:"; tail -n 25 "${LOGDIR}/${k}.log"; fail=1
  fi
done
exit $fail
