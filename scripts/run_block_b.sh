#!/bin/bash
# Block B — Wave B remainder. Sequential MPS to avoid contention deaths.
# Skip jobs whose summary already exists (cheap resume).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

EPOCHS=${EPOCHS:-80}

mkdir -p "$ROOT/baselines/poly2vec/_runs" \
         "$ROOT/baselines/polymp/_runs" \
         "$ROOT/baselines/geo2vec/_runs"

run_step() {
    local name=$1 desc=$2 sentinel=$3
    shift 3
    if [ -f "$sentinel" ]; then
        echo "[block B] SKIP $name ($desc) — sentinel $sentinel exists"
        return 0
    fi
    echo "================================================================"
    echo "[block B] START $name — $desc"
    echo "================================================================"
    "$@"
    echo "[block B] DONE  $name"
}

# 1. Poly2Vec single_mnist (encoder CPU, head MPS, 80 epoch). The previous
#    Wave B Poly2Vec hung on load; running it solo avoids that.
run_step poly2vec_mnist "Poly2Vec on single_mnist (80 epoch)" \
    "$ROOT/baselines/poly2vec/results/single_mnist/summary.json" \
    bash -c "python -u $ROOT/baselines/poly2vec/run_arcset_dataset.py \
        --dataset single_mnist --epochs $EPOCHS --batch-size 64 \
        --encoder-device cpu --head-device mps 2>&1 | \
        tee $ROOT/baselines/poly2vec/_runs/single_mnist.full.log"

# 2-5. PolyMP {polymp, dsc_polymp} × {single_buildings, single_mnist} on MPS
for ds in single_buildings single_mnist; do
  for model in polymp dsc_polymp; do
    run_step "polymp_${model}_${ds}" \
      "PolyMP $model on $ds (80 epoch)" \
      "$ROOT/baselines/polymp/_runs/${ds}_${model}/summary.json" \
      bash -c "python -u $ROOT/baselines/polymp/run_arcset_dataset.py \
        --dataset $ds --model $model --epochs $EPOCHS --device mps 2>&1 | \
        tee $ROOT/baselines/polymp/_runs/${ds}_${model}.full.log"
  done
done

# 6. Geo2Vec single_mnist on CPU (MPS produced NaN on buildings; retry on CPU
#    once buildings rerun confirms it). Slow — ~2 h.
run_step geo2vec_mnist "Geo2Vec single_mnist on CPU (80 epoch)" \
    "$ROOT/baselines/geo2vec/results/single_mnist/summary.json" \
    bash -c "python -u $ROOT/baselines/geo2vec/run_arcset_dataset.py \
        --dataset single_mnist --device cpu --sdf-epochs $EPOCHS \
        --cls-epochs $EPOCHS 2>&1 | \
        tee $ROOT/baselines/geo2vec/_runs/single_mnist.cpu.full.log"

echo "[block B] all done"
