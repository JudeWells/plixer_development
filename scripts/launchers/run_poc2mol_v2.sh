#!/usr/bin/env bash
# Poc2Mol on parquet_v2 (float32 coords), OLD 9-channel vs NEW 11-channel ligand map.
# Everything else identical, so the channel scheme is the single variable.
set -uo pipefail
cd /home/judewells/plixer_outer/plixer || exit 1
export PROJECT_ROOT="$(pwd)" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/poc2mol_v2/${STAMP}"; mkdir -p "$DIR"; echo "$DIR" > /tmp/poc2mol_v2_dir.txt
echo "poc2mol v2: $DIR"

launch () {
  local name="$1" data="$2" out="$3" gpus="$4"
  local ov="experiment=exp1_poc2mol_cons4 task_name=poc2mol_v2_${name} \
    data=${data} model.config.out_channels=${out} \
    model.config.layer_order=gcrd model.config.dropout_prob=0.1 \
    trainer=ddp trainer.devices=4 data.num_workers=12 \
    data.config.batch_size=384 data.config.target_samples_per_batch=1536 \
    trainer.max_epochs=600 +trainer.num_sanity_val_steps=0 \
    callbacks.model_checkpoint.save_top_k=3 callbacks.model_checkpoint.save_last=True \
    seed=42"
  if ! venvPlixer/bin/python src/train.py --cfg job --resolve ${ov} \
        logger.wandb.group=dry logger.wandb.name=dry > "$DIR/${name}.dry" 2>&1; then
    echo "  DRY FAIL ${name}"; tail -5 "$DIR/${name}.dry"; return 1; fi
  echo "  ${name}: dry ok (out_channels=$(grep -oE 'out_channels: [0-9]+' $DIR/${name}.dry|head -1|grep -oE '[0-9]+'))"
  CUDA_VISIBLE_DEVICES="$gpus" nohup venvPlixer/bin/python src/train.py ${ov} \
    logger.wandb.group="poc2mol_v2_${STAMP}" logger.wandb.name="${name}" \
    > "$DIR/${name}.log" 2>&1 &
  echo $! > "$DIR/${name}.pid"; echo "  ${name}: pid $(cat $DIR/${name}.pid) gpus $gpus"
}

launch ch9  poc2mol_hiqbind_v2_9ch  9  0,1,2,3 || exit 1
sleep 20
launch ch11 poc2mol_hiqbind_v2_11ch 11 4,5,6,7 || exit 1

echo "waiting 300s..."; sleep 300
for n in ch9 ch11; do
  p=$(cat "$DIR/${n}.pid")
  if kill -0 "$p" 2>/dev/null; then
    echo "  ok ${n}"; grep -ho "https://wandb.ai/[^ ]*runs/[^ ]*" "$DIR/${n}.log" | sort -u | sed 's/^/     /'
    tr '\r' '\n' < "$DIR/${n}.log" | grep -o "Epoch [0-9]*:.*it/s.*" | tail -1 | cut -c1-100 | sed 's/^/     /'
  else echo "  FAILED ${n}"; tail -15 "$DIR/${n}.log"; fi
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
