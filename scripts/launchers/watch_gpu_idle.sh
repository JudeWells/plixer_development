#!/bin/bash
# Emit an event whenever the node's occupancy CHANGES -- a GPU frees up, or a sweep arm exits.
#
# Purpose: keep all 8 H100s busy without polling by hand. Every stdout line becomes a
# notification, so this deliberately prints only on a state TRANSITION; a steady state of
# "4 arms training, 0 GPUs free" is silent.
#
# Idleness is judged by MEMORY, not utilisation. A rank sitting in validation, in an
# all_gather, or between accumulation steps reads 0% utilisation while holding ~14 GB and
# being very much alive -- the health check in launch_e2e_warm_sweep.sh caught exactly that
# on GPUs 0/2/4/6. Memory below the threshold is the only reliable "nothing is loaded here".
set -u

cd /home/judewells/plixer_outer/plixer || exit 1

FREE_MIB=${FREE_MIB:-2000}       # below this, a GPU holds no model
INTERVAL=${INTERVAL:-60}
# An end-to-end checkpoint is 2.4 GB (both models + AdamW state for 289M parameters). Six
# concurrent arms at save_top_k 2 + save_last filled a 193 GB disk in 45 minutes, which
# killed four runs with ENOSPC and took down the two long-running ones when their checkpoint
# writes failed. 30 GB is roughly one more checkpoint per arm plus headroom -- enough warning
# to trim before anything dies.
DISK_WARN_GB=${DISK_WARN_GB:-30}

# Newest sweep by default; pass a pids.txt to pin one.
PID_FILE=${1:-$(ls -dt logs/e2e_warm_sweep_*/pids.txt 2>/dev/null | head -1)}

previous=""
while true; do
  free_list=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
              | awk -v t="$FREE_MIB" -F', *' '$2 < t {printf "%s ", $1}')
  free_count=$(wc -w <<< "$free_list")

  dead=""
  alive_count=0
  if [ -n "$PID_FILE" ] && [ -f "$PID_FILE" ]; then
    while read -r pid arm _; do
      [ -z "${pid:-}" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        alive_count=$((alive_count + 1))
      else
        dead="$dead $arm"
      fi
    done < "$PID_FILE"
  fi

  disk_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
  low_disk=no
  [ "${disk_gb:-999}" -lt "$DISK_WARN_GB" ] && low_disk=yes

  state="free=$free_count alive=$alive_count dead=$dead lowdisk=$low_disk"
  if [ "$state" != "$previous" ]; then
    stamp=$(date +%H:%M:%S)
    if [ "$low_disk" = yes ]; then
      echo "$stamp  ⚠️ DISK LOW: ${disk_gb}G free (< ${DISK_WARN_GB}G) -- trim checkpoints before a run dies with ENOSPC"
    fi
    if [ "$free_count" -gt 0 ]; then
      echo "$stamp  GPUS FREE: $free_count (ids: ${free_list:-none}) -- arms alive: $alive_count, finished:${dead:- none}"
    else
      echo "$stamp  all GPUs busy -- arms alive: $alive_count, finished:${dead:- none}"
    fi
    previous="$state"
  fi

  sleep "$INTERVAL"
done
