#!/bin/bash
#SBATCH --job-name=sketch_fs
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sx2490/arcset/_overnight_logs/%x-%j.out
#SBATCH --error=/scratch/sx2490/arcset/_overnight_logs/%x-%j.err

# Parametrised by env vars (passed via sbatch --export or --partition/--account flags):
#   NAME      = sketchembednet | sketchformer
#   DATASET   = single_omniglot | single_quickdraw
#   NWAY      = 5 | 20
#   KSHOT     = 1 | 5
#   EPOCHS    = (default 80; smoke uses 2)
#   LIMIT     = optional dataset cap
#   TAG       = optional results subdir suffix
# Partition / account are passed at submit time:
#   sbatch -p h200_tandon -A torch_pr_196_tandon_advanced ...
#   sbatch -p l40s_public -A torch_pr_196_general ...

set -euo pipefail

: "${NAME:?need NAME=sketchembednet|sketchformer}"
: "${DATASET:?need DATASET=single_omniglot|single_quickdraw}"
: "${NWAY:=5}"
: "${KSHOT:=1}"
: "${EPOCHS:=80}"
: "${TRAIN_EPISODES:=200}"
: "${VAL_EPISODES:=200}"
: "${TEST_EPISODES:=1000}"
: "${LIMIT:=}"
: "${TAG:=}"

source /scratch/sx2490/arcset/env/conda/etc/profile.d/conda.sh
conda activate sketch_tf

export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2

cd /scratch/sx2490/arcset

ADAPTER="baselines/${NAME}/run_arcset_dataset.py"
if [ ! -f "$ADAPTER" ]; then
    echo "ERROR: missing adapter $ADAPTER" >&2
    exit 1
fi

OUTDIR="baselines/${NAME}/results/fs_${NAME}_${DATASET}_${NWAY}w_${KSHOT}s${TAG}"
mkdir -p "$OUTDIR"

EXTRA=()
if [ -n "$LIMIT" ]; then EXTRA+=(--limit "$LIMIT"); fi

echo "[start] $(date) NAME=$NAME DATASET=$DATASET NWAY=$NWAY KSHOT=$KSHOT EPOCHS=$EPOCHS"
nvidia-smi || true

python "$ADAPTER" \
    --dataset "$DATASET" \
    --n-way "$NWAY" \
    --k-shot "$KSHOT" \
    --epochs "$EPOCHS" \
    --train-episodes "$TRAIN_EPISODES" \
    --val-episodes "$VAL_EPISODES" \
    --test-episodes "$TEST_EPISODES" \
    --output-dir "$OUTDIR" \
    "${EXTRA[@]}"

echo "[done] $(date)"
