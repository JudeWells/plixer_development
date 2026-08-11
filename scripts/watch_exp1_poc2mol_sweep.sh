#!/usr/bin/env bash
# Summarise the state of an exp1 Poc2Mol sweep: current epoch, best val/loss so far, and
# how many epochs have passed since that best -- which is the direct read on whether a run
# has turned the corner into overfitting.
#
# Usage: bash scripts/watch_exp1_poc2mol_sweep.sh logs/exp1_poc2mol_sweep/<stamp>
set -uo pipefail
DIR="${1:?usage: watch_exp1_poc2mol_sweep.sh <sweep_dir>}"

printf '%-12s %-8s %-6s %-9s %-9s %-9s %-8s %-10s %s\n' RUN STATE EPOCH TRAIN VAL BEST_VAL SINCE GAP NOTE
for log in "${DIR}"/*.log; do
  [[ -e "${log}" ]] || continue
  name="$(basename "${log}" .log)"
  pidfile="${DIR}/${name}.pid"

  state="dead"
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    state="running"
  fi

  # Lightning rewrites the progress bar with \r, so one epoch produces hundreds of lines
  # carrying the same val/loss. Split on \r, then keep the LAST val/loss seen for each
  # epoch number -- that is the value after that epoch's validation. Without the dedupe,
  # "epochs since best" counts bar repaints instead of epochs.
  series="$(tr '\r' '\n' < "${log}" \
    | grep -oE 'Epoch [0-9]+:.*val/loss=[0-9.]+' \
    | sed -E 's/^Epoch ([0-9]+):.*val\/loss=([0-9.]+).*/\1 \2/' \
    | awk '{last[$1]=$2} END {for (e in last) print e, last[e]}' \
    | sort -n)"

  # Same dedupe for train/loss, so the train-vs-val gap can be read off. Overfitting is
  # train still falling while val stops falling or rises -- val flattening alone can just
  # mean the learning rate schedule has taken over.
  train_series="$(tr '\r' '\n' < "${log}" \
    | grep -oE 'Epoch [0-9]+:.*train/loss=[0-9.]+' \
    | sed -E 's/^Epoch ([0-9]+):.*train\/loss=([0-9.]+).*/\1 \2/' \
    | awk '{last[$1]=$2} END {for (e in last) print e, last[e]}' \
    | sort -n)"
  train="$(echo "${train_series}" | awk 'NF{v=$2} END{print v}')"

  epoch="$(tr '\r' '\n' < "${log}" | grep -oE '^Epoch [0-9]+' | tail -1 | grep -oE '[0-9]+')"
  last="$(echo "${series}" | awk 'NF{v=$2} END{print v}')"
  best="$(echo "${series}" | awk 'NF{if (b=="" || $2+0 < b+0) {b=$2; e=$1}} END{print b}')"
  best_ep="$(echo "${series}" | awk 'NF{if (b=="" || $2+0 < b+0) {b=$2; e=$1}} END{print e}')"
  last_ep="$(echo "${series}" | awk 'NF{e=$1} END{print e}')"

  since=""
  if [[ -n "${best_ep}" && -n "${last_ep}" ]]; then
    since=$((last_ep - best_ep))
  fi

  gap=""
  if [[ -n "${train}" && -n "${last}" ]]; then
    gap="$(awk -v v="${last}" -v t="${train}" 'BEGIN{printf "%+.4f", v-t}')"
  fi

  note=""
  if grep -qE "Traceback|CUDA out of memory|Error executing job" "${log}"; then
    note="ERROR - see ${log}"
  elif [[ "${state}" == "dead" && -n "${epoch}" ]]; then
    note="finished at epoch ${epoch}"
  elif [[ -n "${since}" && "${since}" -ge 50 ]]; then
    note="overfitting (best was epoch ${best_ep})"
  fi

  printf '%-12s %-8s %-6s %-9s %-9s %-9s %-8s %-10s %s\n' \
    "${name}" "${state}" "${epoch:--}" "${train:--}" "${last:--}" "${best:--}" \
    "${since:--}" "${gap:--}" "${note}"
done

echo
echo "SINCE  = epochs since the best val/loss; large and growing means past the minimum."
echo "GAP    = val - train. Overfitting is GAP widening while SINCE grows: the model keeps"
echo "         fitting the training set while validation stops improving. Val flattening on"
echo "         its own can just be the LR schedule."
