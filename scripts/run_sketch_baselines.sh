#!/usr/bin/env bash
# Submit the SketchEmbedNet + Sketchformer few-shot sweep on Omniglot.
#
# Usage:
#   bash scripts/run_sketch_baselines.sh smoke                # 1 smoke per baseline
#   bash scripts/run_sketch_baselines.sh full                 # 4 settings × 2 baselines
#   bash scripts/run_sketch_baselines.sh full single_quickdraw  # quickdraw sweep
#
# Optional env vars:
#   PARTITION=h200_tandon|l40s_public  (default h200_tandon for full, l40s_public for smoke)
#   ACCOUNT=...                        (paired with PARTITION; default chosen accordingly)
set -euo pipefail

MODE=${1:-smoke}
DATASET=${2:-single_omniglot}
ROOT=/scratch/sx2490/arcset

cd "$ROOT"

if [ "$MODE" = smoke ]; then
    PARTITION_DEFAULT=l40s_public
    ACCOUNT_DEFAULT=torch_pr_196_general
else
    PARTITION_DEFAULT=h200_tandon
    ACCOUNT_DEFAULT=torch_pr_196_tandon_advanced
fi

PARTITION=${PARTITION:-$PARTITION_DEFAULT}
ACCOUNT=${ACCOUNT:-$ACCOUNT_DEFAULT}

submit() {
    local name=$1 nway=$2 kshot=$3 epochs=$4 tag=$5
    local job_name="${name}_${DATASET#single_}_${nway}w${kshot}s${tag}"
    sbatch \
        -p "$PARTITION" \
        -A "$ACCOUNT" \
        --job-name="$job_name" \
        --export=ALL,NAME="$name",DATASET="$DATASET",NWAY="$nway",KSHOT="$kshot",EPOCHS="$epochs",TAG="$tag" \
        scripts/train_sketch.slurm
}

if [ "$MODE" = smoke ]; then
    submit sketchembednet 5 1 2 _smoke
    submit sketchformer   5 1 2 _smoke
elif [ "$MODE" = full ]; then
    for name in sketchembednet sketchformer; do
        for cfg in "5 1" "5 5" "20 1" "20 5"; do
            read -r nway kshot <<<"$cfg"
            submit "$name" "$nway" "$kshot" 80 ""
        done
    done
else
    echo "unknown mode: $MODE (use smoke|full)" >&2
    exit 2
fi
