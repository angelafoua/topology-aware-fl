#!/usr/bin/env bash
# Per-task LR search at strong communication (p=0.5), TAD-LoRA T=1, seed=0.
# Paper's LR grid.
set -euo pipefail
TASKS=${TASKS:-"sst2 qnli qqp mnli"}
LRS=${LRS:-"2e-4 5e-4 1e-3 2e-3 5e-3"}
for task in $TASKS; do
  for lr in $LRS; do
    OUT="runs/lrsearch_${task}_lr${lr}"
    if [[ -f "$OUT/result.json" ]]; then
      echo "[skip] $OUT"
      continue
    fi
    bash scripts/run_single.sh "$task" "tad" 0.5 1 "$lr" 0
    mv "runs/${task}_tad_p0.5_T1_lr${lr}_s0" "$OUT"
  done
done
