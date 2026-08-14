#!/bin/bash
# Wait for phase A, then prepare and launch phase B across 8 GPUs.
# Split out as a file so the long wait survives independently of any one shell.
set -u
cd /home/judewells/plixer_outer || exit 1
W=boltz_bench/full

while pgrep -f "boltz predict $W/phaseA" >/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S)  phase A finished: $(ls $W/outA_*/boltz_results_phaseA_*/predictions/*/affinity_*.json 2>/dev/null | wc -l)/107"

cd plixer && ./venvPlixer/bin/python scripts/adhoc_analysis/boltz_benchmark.py prepare-pairs \
  --work ../boltz_bench/full --max_pockets 107 --panel_size 107 2>&1 | grep -vE "Warning|import pkg"
cd /home/judewells/plixer_outer

N=$(ls $W/phaseB/*.yaml 2>/dev/null | wc -l)
[ "$N" -eq 0 ] && { echo "no phase B yamls -- aborting"; exit 1; }
for i in 0 1 2 3 4 5 6 7; do rm -rf $W/phaseB_$i; mkdir -p $W/phaseB_$i; done
i=0
for f in $W/phaseB/*.yaml; do mv "$f" $W/phaseB_$((i%8))/; i=$((i+1)); done
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup ./venvBoltz/bin/boltz predict $W/phaseB_$i --out_dir $W/outB_$i \
    --model boltz2 --output_format pdb \
    > /tmp/claude-1001/-home-judewells-plixer-outer-plixer/454addf4-3a75-41da-91d7-7f19d0c04e18/scratchpad/fullB_$i.log 2>&1 &
done
echo "$(date +%H:%M:%S)  phase B launched: $N pairs across 8 shards (~$((N/8)) each)"
