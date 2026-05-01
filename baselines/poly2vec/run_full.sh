#!/usr/bin/env bash
#
# Full Poly2Vec sweep on the ArcSet shape-classification benchmarks.
# Runs 80 epochs per dataset.  Skip with $SKIP="..." to drop a dataset.
#
# Example:
#   SKIP="single_quickdraw" bash run_full.sh
#
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/Users/alexxu/miniforge3/envs/age/bin/python}
EPOCHS=${EPOCHS:-80}
BS=${BS:-64}
SEED=${SEED:-42}
SKIP=${SKIP:-}

ALL=("single_buildings" "single_mnist" "single_omniglot" "single_quickdraw")

for ds in "${ALL[@]}"; do
    if [[ ",${SKIP}," == *",${ds},"* ]]; then
        echo "[skip] $ds"; continue
    fi
    case "$ds" in
        single_buildings|single_mnist) BS_DS=$BS ;;
        single_omniglot) BS_DS=32 ;;     # 1623 classes, smaller batch is gentler on memory
        single_quickdraw) BS_DS=64 ;;
    esac
    echo "==============================================="
    echo "[run] $ds  epochs=$EPOCHS  batch_size=$BS_DS"
    echo "==============================================="
    "$PY" run_arcset_dataset.py \
        --dataset "$ds" \
        --epochs "$EPOCHS" \
        --batch-size "$BS_DS" \
        --seed "$SEED" \
        --output-dir "results/${ds}_full"
done
