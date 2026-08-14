#!/bin/bash
# Keep a long Boltz run inside its disk budget.
#
# ⚠️ THE ORIGINAL VERSION OF THIS SCRIPT DID NOT WORK, and the reason is worth recording.
# Boltz runs structure prediction for EVERY input first, then affinity as a second pass, so
# affinity_*.json only appears at the very end of a shard. A janitor gated on "has an affinity
# json" therefore never fires while the structure phase is filling the disk -- 8 shards x ~1400
# structures took the free space from 28 GB to 10 GB before anything was cleaned.
#
# Two classes of file, two rules:
#   pae_*.npz / pde_*.npz  pure diagnostics, ~2.3 MB per prediction, used by neither the
#                          affinity pass nor the benchmark. Deleted on sight, no gating.
#   *.pdb / pre_affinity   INPUT to the affinity pass. Deleted only once affinity_*.json
#                          exists in that directory, i.e. after it has been consumed.
#
# processed/msa is deliberately left alone: Boltz writes one converted MSA copy per input
# (~940 MB per shard) and the affinity pass re-reads from it.
set -u

ROOT=${1:-/home/judewells/plixer_outer/boltz_bench/full}
INTERVAL=${INTERVAL:-90}
# Report every REPORT_EVERY cycles, not every cycle. At 90s the janitor reclaims 100-170 MB
# each pass for hours; one notification per pass is ~200 messages of pure noise, and a monitor
# that floods gets throttled and stops being useful for the one message that matters.
REPORT_EVERY=${REPORT_EVERY:-20}
freed_total=0
cycle=0

while true; do
  before=$(df -B1 --output=avail /home | tail -1 | tr -dc '0-9')

  # Unconditional: diagnostics we never read.
  find "$ROOT" \( -name "pae_*.npz" -o -name "pde_*.npz" \) -delete 2>/dev/null

  # Gated: only after the affinity pass has consumed the structure.
  while IFS= read -r dir; do
    ls "$dir"/affinity_*.json >/dev/null 2>&1 || continue
    find "$dir" -maxdepth 1 -type f \( -name "*.pdb" -o -name "pre_affinity_*.npz" \
         -o -name "plddt_*.npz" \) -delete 2>/dev/null
  done < <(find "$ROOT" -type d -path "*/predictions/*" 2>/dev/null)

  after=$(df -B1 --output=avail /home | tail -1 | tr -dc '0-9')
  freed=$((after - before))
  [ "$freed" -gt 0 ] && freed_total=$((freed_total + freed))
  cycle=$((cycle + 1))

  # Only two things are worth interrupting for: a periodic heartbeat with progress, and disk
  # actually running out.
  if [ "$after" -lt 8589934592 ]; then
    echo "$(date +%H:%M:%S)  ⚠️ DISK CRITICAL: $((after/1073741824))G free -- janitor cannot reclaim more"
  elif [ $((cycle % REPORT_EVERY)) -eq 0 ]; then
    done_n=$(find "$ROOT" -name "confidence_*.json" 2>/dev/null | wc -l)
    aff_n=$(find "$ROOT" -name "affinity_*.json" 2>/dev/null | wc -l)
    echo "$(date +%H:%M:%S)  structures $done_n, affinities $aff_n, $((after/1073741824))G free (janitor reclaimed $((freed_total/1073741824))G total)"
  fi
  sleep "$INTERVAL"
done
