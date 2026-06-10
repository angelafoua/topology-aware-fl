#!/usr/bin/env bash
# Full paper grid (sequential). On 8xA100 each run uses all GPUs, so we serialize.
# Expect this to take many GPU-days for the complete grid -- use scripts/run_lr_search.sh
# first to pick LR per (task, method) at p=0.5, then this script for the headline grid.
#
# Usage:
#   bash scripts/run_grid.sh                  # default subset
#   TASKS="mnli" METHODS="tad" bash scripts/run_grid.sh
set -euo pipefail
TASKS=${TASKS:-"sst2 qnli qqp mnli"}
METHODS=${METHODS:-"lora ffa rolora tad"}
PS=${PS:-"0.5 0.2 0.1 0.05 0.02 0.01"}
# Switching intervals: divisors of 150 from the paper's preliminary set.
TS=${TS:-"1 2 3 5 10 15"}
SEEDS=${SEEDS:-"0 1 2"}
# Per-method LR after lr-search; override via env if desired.
LR_DEFAULT=${LR_DEFAULT:-5e-4}

for task in $TASKS; do
  for method in $METHODS; do
    # Only TAD-LoRA actually sweeps T; the rest fix T (paper protocol).
    if [[ "$method" == "tad" ]]; then
      MTS="$TS"
    elif [[ "$method" == "rolora" ]]; then
      MTS="1"
    else
      MTS="1"
    fi
    for p in $PS; do
      for T in $MTS; do
        for seed in $SEEDS; do
          OUT="runs/${task}_${method}_p${p}_T${T}_s${seed}"
          if [[ -f "$OUT/result.json" ]]; then
            echo "[skip] $OUT exists"
            continue
          fi
          bash scripts/run_single.sh "$task" "$method" "$p" "$T" "$LR_DEFAULT" "$seed"
        done
      done
    done
  done
done
