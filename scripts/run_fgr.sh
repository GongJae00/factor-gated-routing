#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

export FGR_OUTPUT_DIR="${FGR_OUTPUT_DIR:-output}"
export DSPRITES_PATH="${DSPRITES_PATH:-/tmp/dsprites/dsprites.npz}"

MODEL="${1:-ROSTFRG}"
DATASET="${2:-dsprites}"
STEPS="${3:-400000}"
SEED="${4:-42}"

OUT="$FGR_OUTPUT_DIR/${DATASET}_${MODEL}_seed${SEED}"
mkdir -p "$OUT"
echo "Training $MODEL on $DATASET (seed=$SEED, steps=$STEPS) → $OUT"

.venv/bin/python3 -m src.train \
  --model "$MODEL" --dataset "$DATASET" \
  --output-dir "$OUT" --steps "$STEPS" \
  --batch-size 128 --seed "$SEED" \
  --num-workers 0 --log-every 10000 --ckpt-every 100000
