#!/bin/bash
# Keep every GPU busy: whenever a pair frees up, launch the next arm from the queue.
#
# The idle-watcher only NOTIFIES; between turns nobody is listening, so an arm finishing at
# 03:00 would leave two H100s idle until someone looked. This closes that loop.
#
# QUEUE: scripts/launchers/queue.txt, one experiment name per line, '#' comments ignored. A
# launched line is rewritten with a '# launched <time>' prefix rather than deleted, so the file
# doubles as a record of what went out and in what order. Append to it to extend the run plan.
#
# ── TWO FAILURE MODES THIS HAS ALREADY HIT, both of which drained the queue into dead runs ──
#
# 1. PORT REUSE. The rendezvous port must be picked by AVAILABILITY, never by an in-process
#    counter: a counter resets when this script restarts and reissues ports that still-running
#    jobs hold, and torch then dies at init_process_group with "Address already in use". That
#    failure is invisible to the launcher -- the process backgrounds fine and dies seconds
#    later -- so the GPUs go back to looking idle and the next arm is launched onto them too.
#
# 2. STARTUP BLINDNESS. A new run spends ~2 minutes indexing ZINC before it allocates any GPU
#    memory, so nvidia-smi reports its GPUs as free that whole time. An in-process cooldown
#    does not survive a restart of this script, and restarting it mid-startup double-books the
#    pair. Hence the CLAIMS FILE: every launch records (gpu, pid, timestamp) on disk, and a GPU
#    counts as busy if it is claimed by a live pid within STARTUP_GRACE even when nvidia-smi
#    says otherwise. Persisting to disk is the point -- it is what makes this restart-safe.
#
# Idleness is otherwise judged by MEMORY, not utilisation: a rank in validation or between
# accumulation steps reads 0% while holding ~30 GB (CLAUDE.md §0).
set -u

cd /home/judewells/plixer_outer/plixer || exit 1

QUEUE=${QUEUE:-scripts/launchers/queue.txt}
CLAIMS=${CLAIMS:-logs/.gpu_claims}
FREE_MIB=${FREE_MIB:-2000}
INTERVAL=${INTERVAL:-60}
STARTUP_GRACE=${STARTUP_GRACE:-600}
MIN_DISK_GB=${MIN_DISK_GB:-25}
PORT_BASE=${PORT_BASE:-29800}

touch "$CLAIMS"

free_port () {
  local p=$PORT_BASE
  while ss -Hltn 2>/dev/null | grep -q ":${p}[[:space:]]"; do p=$((p + 1)); done
  echo "$p"
}

# GPUs claimed by a launch that is still inside its startup window and whose process lives.
claimed_gpus () {
  local now; now=$(date +%s)
  local out=""
  while read -r gpu pid ts; do
    [ -z "${ts:-}" ] && continue
    if [ $((now - ts)) -lt "$STARTUP_GRACE" ] && kill -0 "$pid" 2>/dev/null; then
      out="$out $gpu"
    fi
  done < "$CLAIMS"
  echo "$out"
}

prune_claims () {
  local now tmp
  now=$(date +%s)
  tmp=$(mktemp)
  while read -r gpu pid ts; do
    [ -z "${ts:-}" ] && continue
    if [ $((now - ts)) -lt "$STARTUP_GRACE" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$gpu $pid $ts" >> "$tmp"
    fi
  done < "$CLAIMS"
  mv "$tmp" "$CLAIMS"
}

while true; do
  prune_claims

  if ! grep -qvE '^\s*(#|$)' "$QUEUE" 2>/dev/null; then
    echo "$(date +%H:%M:%S)  queue empty -- autofill idle (append names to $QUEUE to resume)"
    sleep 300
    continue
  fi

  busy_claimed=" $(claimed_gpus) "
  free_ids=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
             | awk -v t="$FREE_MIB" -F', *' '$2 < t {print $1}')
  # Drop anything inside another launch's startup window.
  available=""
  for g in $free_ids; do
    case "$busy_claimed" in *" $g "*) ;; *) available="$available $g" ;; esac
  done
  n_free=$(wc -w <<< "$available")
  disk_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')

  if [ "$n_free" -ge 2 ]; then
    if [ "${disk_gb:-999}" -lt "$MIN_DISK_GB" ]; then
      # Refusing to launch is the right failure: ENOSPC mid-run killed six arms once already,
      # and a queued arm loses nothing by waiting.
      echo "$(date +%H:%M:%S)  ⚠️ ${n_free} GPUs free but only ${disk_gb}G disk (< ${MIN_DISK_GB}G) -- NOT launching. Trim checkpoints."
      sleep "$INTERVAL"
      continue
    fi

    pending=($available)
    while [ "${#pending[@]}" -ge 2 ]; do
      next=$(grep -vE '^\s*(#|$)' "$QUEUE" 2>/dev/null | head -1)
      [ -z "${next:-}" ] && break
      g1="${pending[0]}"; g2="${pending[1]}"
      pending=("${pending[@]:2}")
      port=$(free_port)
      bash scripts/launchers/launch_e2e_arm.sh "$next" "$g1,$g2" "$port" 2 >/dev/null 2>&1
      pid=$(tail -1 logs/e2e_warm_sweep_*/pids.txt 2>/dev/null | awk '{print $1}')
      now=$(date +%s)
      echo "$g1 $pid $now" >> "$CLAIMS"
      echo "$g2 $pid $now" >> "$CLAIMS"
      stamp=$(date +%H:%M:%S)
      sed -i "0,\|^${next}\s*$|s||# launched ${stamp} ${next}|" "$QUEUE"
      echo "$stamp  LAUNCHED $next on GPUs $g1,$g2 port $port (${disk_gb}G disk, $(grep -cvE '^\s*(#|$)' "$QUEUE") left in queue)"
      sleep 20   # stagger: several DDP jobs racing through ZINC indexing at once thrashes the FS
    done
  fi
  sleep "$INTERVAL"
done
