#!/bin/bash
# Run all 4 polygon baselines on single_buildings, 80 epochs each.
# Saves logs to baselines/<name>/_runs/single_buildings.full.log
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

BUILDINGS_GPKG="$ROOT/data/single_buildings/ShapeClassification.gpkg"
EPOCHS=${EPOCHS:-80}

run() {
    local name=$1; shift
    local logdir="$ROOT/baselines/$name/_runs"
    mkdir -p "$logdir"
    local log="$logdir/single_buildings.full.log"
    echo "================================================================"
    echo "[$name] start single_buildings full run, $EPOCHS epochs"
    echo "[$name] log: $log"
    echo "================================================================"
    "$@" 2>&1 | tee "$log"
    echo "[$name] done"
}

# 1. NUFT — CPU only (DDSL needs float64)
run nuft \
    python -u "$ROOT/baselines/nuft/run_arcset_dataset.py" \
        --gpkg "$BUILDINGS_GPKG" --label_col label --num_vert 64 \
        --num_epoch "$EPOCHS" --device cpu

# 2. PolygonGNN — uses pre-built single_building.pkl
( cd "$ROOT/baselines/polygongnn" && \
  run polygongnn \
      python -u train_new.py --dataset sbuilding --nepoch "$EPOCHS" \
          --train_batch 32 --test_batch 64 --log False )

# 3. Poly2Vec
run poly2vec \
    python -u "$ROOT/baselines/poly2vec/run_arcset_dataset.py" \
        --dataset single_buildings --epochs "$EPOCHS" --batch-size 64 \
        --encoder-device cpu --head-device mps

# 4. Geo2Vec
run geo2vec \
    python -u "$ROOT/baselines/geo2vec/run_arcset_dataset.py" \
        --dataset single_buildings --device mps --sdf-epochs "$EPOCHS" \
        --cls-epochs "$EPOCHS"

echo "all 4 baseline runs on single_buildings done"
