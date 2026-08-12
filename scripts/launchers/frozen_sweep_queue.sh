#!/bin/bash
# Hyperparameter sweep on the FROZEN-upstream baseline (arm Z), 2 GPUs per arm.
#
#   S1  schedule compressed to 1500 steps  -- anneal onto the observed optimum, not past it
#   S2  decoder lr 5e-5 -> 2e-5            -- the other reading of "peaks early then declines"
#   S3  weight_decay 0.1 + dropout 0.1     -- actually regularise 172.7M params on 9,872 clusters
#   S4  prob_poc2mol 0.5 -> 0.8            -- train on pockets rather than half ZINC
#
# Everything else is held at arm Z, which has two seeds at 0.7601/0.7618 (mean 0.7610), so
# each arm is one change away from a well-measured reference.
#
# Powered to detect ~0.012 at one seed per arm (per-run seed sigma ~0.0045, pooled over four
# replicate pairs). Anything smaller than that needs a replicate before it means anything --
# §14d's 24-hyperparameter sweep "all within noise" was run at a much coarser noise level, so
# the point here is not to re-run it but to look at a resolution it could not reach.
set -u
cd /home/judewells/plixer_outer/plixer || exit 1

QUEUE=(sweep_s1_short_schedule sweep_s2_lower_lr sweep_s3_regularised sweep_s4_more_pockets)
PORT=29540

for arm in "${QUEUE[@]}"; do
  # -F', *' is load-bearing: --format=csv,noheader,nounits still emits "0, 512", so awk's
  # default splitting makes $1 the string "0," and yields CUDA_VISIBLE_DEVICES="0,,1,".
  while true; do
    mapfile -t free < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                        | awk -F', *' '$2 < 2000 {print $1}')
    [ "${#free[@]}" -ge 2 ] && break
    sleep 60
  done

  gpus="${free[0]},${free[1]}"
  echo "SWEEP: launching $arm on GPUs $gpus"
  ./scripts/launchers/launch_e2e_arm.sh "$arm" "$gpus" "$PORT" hp 2>&1 | tail -2
  PORT=$((PORT + 1))
done

echo "SWEEP QUEUE DRAINED -- all four frozen-baseline arms launched"
