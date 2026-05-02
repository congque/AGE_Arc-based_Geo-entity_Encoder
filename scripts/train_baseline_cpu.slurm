#!/bin/bash
# CPU-bound baselines (poly2vec encoder, geo2vec SDF sampling). No GPU waste.
#SBATCH --account=torch_pr_196_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sx2490/arcset/_logs/blcpu_%x_%j.out
#SBATCH --error=/scratch/sx2490/arcset/_logs/blcpu_%x_%j.err

set -e
NAME=${NAME:?missing NAME}
DATASET=${DATASET:?missing DATASET}
EPOCHS=${EPOCHS:-80}

cd /scratch/sx2490/arcset
source env/conda/etc/profile.d/conda.sh
conda activate arcset
export KMP_DUPLICATE_LIB_OK=TRUE

case "$NAME" in
  poly2vec)
    python -u baselines/poly2vec/run_arcset_dataset.py \
      --dataset $DATASET --epochs $EPOCHS --batch-size 64 \
      --encoder-device cpu --head-device cpu
    ;;
  geo2vec)
    python -u baselines/geo2vec/run_arcset_dataset.py \
      --dataset $DATASET --device cpu --sdf-epochs $EPOCHS --cls-epochs $EPOCHS
    ;;
  nuft)
    GPKG=data/${DATASET}/$(ls data/${DATASET}/ | head -1)
    python -u baselines/nuft/run_arcset_dataset.py \
      --gpkg "$GPKG" --label_col label --num_vert 64 \
      --num_epoch $EPOCHS --device cpu
    ;;
esac
