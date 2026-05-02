#!/bin/bash
#SBATCH --job-name=arc-std
#SBATCH --account=torch_pr_196_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/sx2490/arcset/_logs/std_%x_%j.out
#SBATCH --error=/scratch/sx2490/arcset/_logs/std_%x_%j.err

# Usage: sbatch --export=ALL,DATASET=single_mnist,MODEL=settransformer-sab \
#               --job-name=std_sab_mnist scripts/train_arcset_standard.slurm
set -e
DATASET=${DATASET:?missing DATASET}
MODEL=${MODEL:?missing MODEL}
EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-128}
EXTRA=${EXTRA:-}

cd /scratch/sx2490/arcset
source env/conda/etc/profile.d/conda.sh
conda activate arcset

OUT=model_edges/results/${MODEL//-/_}_${DATASET}
mkdir -p "$OUT"

python -u model_edges/test.py \
    --dataset "$DATASET" --set-model "$MODEL" \
    --epochs "$EPOCHS" --batch-size "$BATCH" \
    --device cuda --output-dir "$OUT" $EXTRA
