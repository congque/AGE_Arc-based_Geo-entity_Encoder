#!/bin/bash
# Run all 4 polygon baselines on single_mnist, 80 epochs each.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

MNIST_GPKG="$ROOT/data/single_mnist/mnist_scaled_normalized.gpkg"
EPOCHS=${EPOCHS:-80}

run() {
    local name=$1; shift
    local logdir="$ROOT/baselines/$name/_runs"
    mkdir -p "$logdir"
    local log="$logdir/single_mnist.full.log"
    echo "================================================================"
    echo "[$name] start single_mnist full run, $EPOCHS epochs"
    echo "================================================================"
    "$@" 2>&1 | tee "$log"
}

# 1. NUFT — CPU only
run nuft \
    python -u "$ROOT/baselines/nuft/run_arcset_dataset.py" \
        --gpkg "$MNIST_GPKG" --label_col label --num_vert 64 \
        --num_epoch "$EPOCHS" --device cpu

# 2. PolygonGNN — uses pre-built single_mnist.pkl (download if missing)
( cd "$ROOT/baselines/polygongnn" && \
  run polygongnn \
      python -u train_new.py --dataset smnist --nepoch "$EPOCHS" \
          --train_batch 64 --test_batch 128 --log False )

# 3. Poly2Vec
run poly2vec \
    python -u "$ROOT/baselines/poly2vec/run_arcset_dataset.py" \
        --dataset single_mnist --epochs "$EPOCHS" --batch-size 64 \
        --encoder-device cpu --head-device mps

# 4. Geo2Vec
run geo2vec \
    python -u "$ROOT/baselines/geo2vec/run_arcset_dataset.py" \
        --dataset single_mnist --device mps --sdf-epochs "$EPOCHS" \
        --cls-epochs "$EPOCHS"

echo "all 4 baseline runs on single_mnist done"
