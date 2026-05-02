#!/bin/bash
# Generic baseline runner. Pass NAME=<nuft|polygongnn|poly2vec|geo2vec|polymp>
# DATASET=<single_buildings|single_mnist|single_omniglot|single_quickdraw>
# Optional: PMODEL=<polymp|dsc_polymp> for polymp variant.
#SBATCH --account=torch_pr_196_general
#SBATCH --partition=l40s_public
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sx2490/arcset/_logs/bl_%x_%j.out
#SBATCH --error=/scratch/sx2490/arcset/_logs/bl_%x_%j.err

set -e
NAME=${NAME:?missing NAME}
DATASET=${DATASET:?missing DATASET}
EPOCHS=${EPOCHS:-80}

cd /scratch/sx2490/arcset
source env/conda/etc/profile.d/conda.sh
conda activate arcset
export KMP_DUPLICATE_LIB_OK=TRUE

case "$NAME" in
  nuft)
    GPKG=data/${DATASET}/$(ls data/${DATASET}/ | head -1)
    python -u baselines/nuft/run_arcset_dataset.py \
      --gpkg "$GPKG" --label_col label --num_vert 64 \
      --num_epoch $EPOCHS --device cpu
    ;;
  polygongnn)
    case "$DATASET" in
      single_buildings) PG_DS=sbuilding ;;
      single_mnist) PG_DS=smnist ;;
      *) echo "PolygonGNN only supports buildings/mnist"; exit 1 ;;
    esac
    cd baselines/polygongnn
    python -u train_new.py --dataset $PG_DS --nepoch $EPOCHS \
      --train_batch 64 --test_batch 128 --log False
    ;;
  poly2vec)
    python -u baselines/poly2vec/run_arcset_dataset.py \
      --dataset $DATASET --epochs $EPOCHS --batch-size 64 \
      --encoder-device cpu --head-device cuda
    ;;
  geo2vec)
    python -u baselines/geo2vec/run_arcset_dataset.py \
      --dataset $DATASET --device cuda --sdf-epochs $EPOCHS --cls-epochs $EPOCHS
    ;;
  polymp)
    PMODEL=${PMODEL:-polymp}
    python -u baselines/polymp/run_arcset_dataset.py \
      --dataset $DATASET --model $PMODEL --epochs $EPOCHS --device cuda
    ;;
  *)
    echo "Unknown NAME=$NAME"; exit 1
    ;;
esac
