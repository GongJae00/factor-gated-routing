#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

export FGR_OUTPUT_DIR="${FGR_OUTPUT_DIR:-output}"
export DSPRITES_PATH="${DSPRITES_PATH:-/tmp/dsprites/dsprites.npz}"

MODELS=("FGR" "SDiT" "EncDiff" "MMDiT-k" "CoInD" "CF-DiT")
DATASET="${1:-dsprites}"
STEPS="${2:-400000}"
SEED="${3:-42}"

echo "=== FGR Full Experiment Pipeline ==="
echo "Dataset: $DATASET | Steps: $STEPS | Seed: $SEED"
echo "Models: ${MODELS[*]}"
echo "Output: $FGR_OUTPUT_DIR"
echo "===================================="

for model in "${MODELS[@]}"; do
  OUT="$FGR_OUTPUT_DIR/${DATASET}_${model}_seed${SEED}"
  mkdir -p "$OUT"
  echo ""
  echo ">>> Training $model on $DATASET (seed=$SEED) <<<"
  .venv/bin/python3 -m src.train \
    --model "$model" --dataset "$DATASET" \
    --output-dir "$OUT" --steps "$STEPS" \
    --batch-size 128 --seed "$SEED" \
    --num-workers 0 --log-every 10000 --ckpt-every 100000
  echo ">>> $model complete <<<"
done

echo ""
echo "=== All training complete ==="
