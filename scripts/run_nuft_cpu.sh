#!/bin/bash
# CPU-only NUFT pipeline: buildings + mnist, sequential, 80 epochs each.
# Safe to run alongside MPS jobs (won't contend).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE

EPOCHS=${EPOCHS:-80}
mkdir -p "$ROOT/baselines/nuft/_runs"

for gpkg_path in \
    "data/single_buildings/ShapeClassification.gpkg" \
    "data/single_mnist/mnist_scaled_normalized.gpkg"; do
  ds_name=$(basename "$gpkg_path" .gpkg)
  log="$ROOT/baselines/nuft/_runs/${ds_name}.full.log"
  echo "================================================================"
  echo "[nuft] $gpkg_path | log $log"
  echo "================================================================"
  python -u "$ROOT/baselines/nuft/run_arcset_dataset.py" \
      --gpkg "$ROOT/$gpkg_path" --label_col label --num_vert 64 \
      --num_epoch "$EPOCHS" --device cpu 2>&1 | tee "$log"
done

echo "[nuft] all 2 datasets done"
