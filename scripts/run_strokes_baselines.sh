#!/bin/bash
# Stroke datasets (omniglot, quickdraw) x baselines that support polylines.
# Geo2Vec: SDF works for any geometry.
# Poly2Vec: polyline FT path supports MultiLineString.
# NUFT and PolygonGNN are polygon-only -> skip.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

EPOCHS=${EPOCHS:-80}

run() {
    local name=$1; shift
    local ds=$1; shift
    local logdir="$ROOT/baselines/$name/_runs"
    mkdir -p "$logdir"
    local log="$logdir/$ds.full.log"
    echo "================================================================"
    echo "[$name] start $ds full run, $EPOCHS epochs"
    echo "================================================================"
    "$@" 2>&1 | tee "$log"
}

for ds in single_omniglot single_quickdraw; do
  # Geo2Vec
  run geo2vec "$ds" \
      python -u "$ROOT/baselines/geo2vec/run_arcset_dataset.py" \
          --dataset "$ds" --device mps --sdf-epochs "$EPOCHS" --cls-epochs "$EPOCHS"

  # Poly2Vec
  run poly2vec "$ds" \
      python -u "$ROOT/baselines/poly2vec/run_arcset_dataset.py" \
          --dataset "$ds" --epochs "$EPOCHS" --batch-size 64 \
          --encoder-device cpu --head-device mps
done

echo "all 4 stroke-baseline runs done"
