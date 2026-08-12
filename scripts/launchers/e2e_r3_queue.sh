#!/bin/bash
# Drain the round-3 arm queue onto GPU pairs as they free up.
#
# Round-2 arms finish at different times (early stopping fires wherever each arm's noise
# spike landed), so waiting for a whole round before starting the next leaves GPUs idle for
# up to an hour. Arms are independent runs at identical step budgets -- staggering them in
# wall-clock costs nothing in comparability -- so the queue just fills pairs as they appear.
#
# Arm B is already running (launched by hand on GPUs 4,5); this covers the rest of the
# matched set. Round 3 differs from round 2 ONLY in that early stopping is disabled, so
# every arm runs the full 4000 steps and completes the LR anneal.
set -u
cd /home/judewells/plixer_outer/plixer || exit 1

QUEUE=(e2e_z_frozen e2e_d_control e2e_a_anchored_hard)
PORT=29522

for arm in "${QUEUE[@]}"; do
  # Wait for a free pair. Idle is judged on MEMORY: a live job holds ~30 GB throughout but
  # drops to ~0% utilisation during validation, so a utilisation test would see false idles.
  # -F', *' matters: `--format=csv,noheader,nounits` still emits "0, 512", so with awk's
  # default whitespace splitting $1 is the string "0," -- comma included. That produced
  # CUDA_VISIBLE_DEVICES="0,,1," and killed the run at Trainer instantiation.
  while true; do
    mapfile -t free < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                        | awk -F', *' '$2 < 2000 {print $1}')
    [ "${#free[@]}" -ge 2 ] && break
    sleep 60
  done

  gpus="${free[0]},${free[1]}"
  echo "QUEUE: launching $arm on GPUs $gpus"
  # The launcher blocks ~300 s on its own health check, which also stops this loop from
  # re-detecting the same GPUs as free before the new job has allocated them.
  ./scripts/launchers/launch_e2e_arm.sh "$arm" "$gpus" "$PORT" r3 2>&1 | tail -3
  PORT=$((PORT + 1))
done

echo "QUEUE DRAINED -- all round-3 arms launched"
