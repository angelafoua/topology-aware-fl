#!/usr/bin/env bash
# Single-config training run.
# Usage: bash scripts/run_single.sh <task> <method> <p> <T> <lr> <seed>
set -euo pipefail
TASK=${1:-sst2}
METHOD=${2:-tad}
P=${3:-0.1}
T=${4:-3}
LR=${5:-5e-4}
SEED=${6:-0}
OUT="runs/${TASK}_${METHOD}_p${P}_T${T}_lr${LR}_s${SEED}"
mkdir -p "$OUT"
python -u main.py \
  --task "$TASK" --method "$METHOD" --p "$P" --T "$T" --lr "$LR" --seed "$SEED" \
  --num_clients 100 --rounds 150 --local_steps 20 \
  --batch_size 32 --max_len 128 --world_size 8 \
  --output_dir "$OUT" 2>&1 | tee "$OUT/train.log"
