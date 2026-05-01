#!/bin/bash
# Wave B: 3 baselines on single_mnist + PolyMP on buildings/mnist.
# NUFT mnist is already done separately (96.63%).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

EPOCHS=${EPOCHS:-80}

mkdir -p "$ROOT/baselines/polygongnn/_runs" \
         "$ROOT/baselines/poly2vec/_runs" \
         "$ROOT/baselines/geo2vec/_runs" \
         "$ROOT/baselines/polymp/_runs"

echo "================================================================"
echo "[wave B] PolygonGNN smnist ($EPOCHS epoch)"
echo "================================================================"
( cd "$ROOT/baselines/polygongnn" && \
  python -u train_new.py --dataset smnist --nepoch "$EPOCHS" \
      --train_batch 64 --test_batch 128 --log False 2>&1 | \
  tee "$ROOT/baselines/polygongnn/_runs/single_mnist.full.log" )

echo "================================================================"
echo "[wave B] Poly2Vec single_mnist ($EPOCHS epoch)"
echo "================================================================"
python -u "$ROOT/baselines/poly2vec/run_arcset_dataset.py" \
    --dataset single_mnist --epochs "$EPOCHS" --batch-size 64 \
    --encoder-device cpu --head-device mps 2>&1 | \
  tee "$ROOT/baselines/poly2vec/_runs/single_mnist.full.log"

echo "================================================================"
echo "[wave B] Geo2Vec single_mnist ($EPOCHS epoch)"
echo "================================================================"
python -u "$ROOT/baselines/geo2vec/run_arcset_dataset.py" \
    --dataset single_mnist --device mps --sdf-epochs "$EPOCHS" \
    --cls-epochs "$EPOCHS" 2>&1 | \
  tee "$ROOT/baselines/geo2vec/_runs/single_mnist.full.log"

echo "================================================================"
echo "[wave B] PolyMP buildings + mnist ($EPOCHS epoch each)"
echo "================================================================"
for ds in single_buildings single_mnist; do
  for model in polymp dsc_polymp; do
    python -u "$ROOT/baselines/polymp/run_arcset_dataset.py" \
        --dataset "$ds" --model "$model" --epochs "$EPOCHS" --device mps 2>&1 | \
      tee "$ROOT/baselines/polymp/_runs/${ds}_${model}.full.log"
  done
done

echo "[wave B] all done"
