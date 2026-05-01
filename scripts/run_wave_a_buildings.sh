#!/bin/bash
# Wave A: 3 baselines on single_buildings, 80 epochs each.
# NUFT already done separately on CPU.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

EPOCHS=${EPOCHS:-80}

echo "================================================================"
echo "[wave A] PolygonGNN sbuilding ($EPOCHS epoch)"
echo "================================================================"
( cd "$ROOT/baselines/polygongnn" && \
  python -u train_new.py --dataset sbuilding --nepoch "$EPOCHS" \
      --train_batch 32 --test_batch 64 --log False 2>&1 | \
  tee "$ROOT/baselines/polygongnn/_runs/single_buildings.full.log" )

mkdir -p "$ROOT/baselines/poly2vec/_runs"
echo "================================================================"
echo "[wave A] Poly2Vec single_buildings ($EPOCHS epoch)"
echo "================================================================"
python -u "$ROOT/baselines/poly2vec/run_arcset_dataset.py" \
    --dataset single_buildings --epochs "$EPOCHS" --batch-size 64 \
    --encoder-device cpu --head-device mps 2>&1 | \
  tee "$ROOT/baselines/poly2vec/_runs/single_buildings.full.log"

mkdir -p "$ROOT/baselines/geo2vec/_runs"
echo "================================================================"
echo "[wave A] Geo2Vec single_buildings ($EPOCHS epoch)"
echo "================================================================"
python -u "$ROOT/baselines/geo2vec/run_arcset_dataset.py" \
    --dataset single_buildings --device mps --sdf-epochs "$EPOCHS" \
    --cls-epochs "$EPOCHS" 2>&1 | \
  tee "$ROOT/baselines/geo2vec/_runs/single_buildings.full.log"

echo "[wave A] all done"
