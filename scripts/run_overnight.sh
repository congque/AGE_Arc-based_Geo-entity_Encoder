#!/bin/bash
# Master overnight driver. Runs Block B → Block C → Block D sequentially.
# Each block has sentinel skip (won't redo finished jobs) and continues past
# individual failures so a single bad run doesn't block the whole night.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

EPOCHS=${EPOCHS:-80}
LOG_ROOT="$ROOT/_overnight_logs"
mkdir -p "$LOG_ROOT"

now() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(now)] overnight pipeline starting"

run() {
    local label=$1; shift
    local sentinel=$1; shift
    local log="$LOG_ROOT/${label}.log"
    if [ -n "$sentinel" ] && [ -f "$sentinel" ]; then
        echo "[$(now)] SKIP $label (sentinel $sentinel exists)"
        return 0
    fi
    echo "[$(now)] BEGIN $label -> $log"
    "$@" > "$log" 2>&1 || echo "[$(now)] FAIL  $label (continuing)"
    echo "[$(now)] END   $label"
}

PYTHON="$ROOT"
PY=/Users/alexxu/miniforge3/envs/age/bin/python

# ============================================================
# BLOCK B (Wave B remainder, sequential MPS)
# ============================================================
echo "[$(now)] === BLOCK B === Wave B remainder"

run polygongnn_smnist_redo "" \
    bash -c "cd $ROOT/baselines/polygongnn && $PY -u train_new.py --dataset smnist --nepoch $EPOCHS --train_batch 64 --test_batch 128 --log False"
# polygongnn already done; sentinel skip via summary file would be safer but
# their save format differs. Manually skip if log file already exists.
if [ -s "$ROOT/baselines/polygongnn/_runs/single_mnist.full.log" ]; then
  echo "[$(now)] (polygongnn smnist already has full log)"
fi

run poly2vec_smnist "$ROOT/baselines/poly2vec/results/single_mnist/summary.json" \
    bash -c "$PY -u $ROOT/baselines/poly2vec/run_arcset_dataset.py --dataset single_mnist --epochs $EPOCHS --batch-size 64 --encoder-device cpu --head-device mps"

for ds in single_buildings single_mnist; do
  for model in polymp dsc_polymp; do
    run "polymp_${model}_${ds}" \
        "$ROOT/baselines/polymp/_runs/${ds}_${model}/summary.json" \
        bash -c "$PY -u $ROOT/baselines/polymp/run_arcset_dataset.py --dataset $ds --model $model --epochs $EPOCHS --device mps"
  done
done

run geo2vec_smnist_cpu "$ROOT/baselines/geo2vec/results/single_mnist/summary.json" \
    bash -c "$PY -u $ROOT/baselines/geo2vec/run_arcset_dataset.py --dataset single_mnist --device cpu --sdf-epochs $EPOCHS --cls-epochs $EPOCHS"

# ============================================================
# BLOCK C (few-shot omniglot, fixed config: cosine + learnable temp + freq=6)
# ============================================================
echo "[$(now)] === BLOCK C === few-shot Omniglot 12 runs"

# Priority: SAB and ISAB (proven capacity) before DeepSet.
for model in settransformer-sab settransformer-isab deepset; do
  for nway in 5 20; do
    for kshot in 1 5; do
      tag="${model//-/_}"
      out="$ROOT/model_edges/results/fs_${tag}_single_omniglot_${nway}w_${kshot}s"
      sentinel="$out/summary.json"
      run "fs_${tag}_${nway}w_${kshot}s" "$sentinel" \
          bash -c "$PY -u $ROOT/model_edges/testfs.py \
              --dataset single_omniglot --set-model $model \
              --epochs $EPOCHS --train-episodes 200 \
              --val-episodes 200 --test-episodes 1000 \
              --n-way $nway --k-shot $kshot --n-query 15 \
              --xy-num-freqs 6 --proto-distance cosine --proto-init-temp 10.0 \
              --output-dir $out"
    done
  done
done

# ============================================================
# BLOCK D (stroke baselines on omniglot)
# ============================================================
echo "[$(now)] === BLOCK D === stroke baselines on Omniglot"

run geo2vec_omniglot_cpu "$ROOT/baselines/geo2vec/results/single_omniglot/summary.json" \
    bash -c "$PY -u $ROOT/baselines/geo2vec/run_arcset_dataset.py --dataset single_omniglot --device cpu --sdf-epochs $EPOCHS --cls-epochs $EPOCHS"

run poly2vec_omniglot "$ROOT/baselines/poly2vec/results/single_omniglot/summary.json" \
    bash -c "$PY -u $ROOT/baselines/poly2vec/run_arcset_dataset.py --dataset single_omniglot --epochs $EPOCHS --batch-size 64 --encoder-device cpu --head-device mps"

echo "[$(now)] === overnight pipeline DONE ==="
