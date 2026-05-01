#!/bin/bash
# Sequential driver for table 1.1 standard classification.
# Default: mnist + buildings, all three set-models. Override via DATASETS / MODELS.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-single_buildings single_mnist}"
MODELS="${MODELS:-deepset settransformer-sab settransformer-isab}"
EPOCHS="${EPOCHS:-80}"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age

for ds in $DATASETS; do
  for model in $MODELS; do
    tag="${model//-/_}"
    out="model_edges/results/${tag}_${ds}"
    echo "================================================================"
    echo "[run] dataset=$ds model=$model out=$out"
    echo "================================================================"
    python -u model_edges/test.py \
      --dataset "$ds" \
      --set-model "$model" \
      --epochs "$EPOCHS" \
      --output-dir "$out"
  done
done

echo
echo "all runs complete"
