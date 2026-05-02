#!/bin/bash
#SBATCH --job-name=arc-fs
#SBATCH --account=torch_pr_196_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sx2490/arcset/_logs/fs_%x_%j.out
#SBATCH --error=/scratch/sx2490/arcset/_logs/fs_%x_%j.err

# Usage: sbatch --export=ALL,DATASET=single_omniglot,MODEL=settransformer-sab,NWAY=5,KSHOT=1 \
#               --job-name=fs_sab_5w1s scripts/train_arcset_fewshot.slurm
set -e
DATASET=${DATASET:-single_omniglot}
MODEL=${MODEL:?missing MODEL}
NWAY=${NWAY:-5}
KSHOT=${KSHOT:-1}
NQUERY=${NQUERY:-15}
EPOCHS=${EPOCHS:-80}
TRAIN_EP=${TRAIN_EP:-200}
VAL_EP=${VAL_EP:-200}
TEST_EP=${TEST_EP:-1000}
XYFREQ=${XYFREQ:-6}
EXTRA=${EXTRA:-}

cd /scratch/sx2490/arcset
source env/conda/etc/profile.d/conda.sh
conda activate arcset

TAG=${MODEL//-/_}
OUT=model_edges/results/fs_${TAG}_${DATASET}_${NWAY}w_${KSHOT}s
mkdir -p "$OUT"

python -u model_edges/testfs.py \
    --dataset "$DATASET" --set-model "$MODEL" \
    --epochs "$EPOCHS" --train-episodes "$TRAIN_EP" \
    --val-episodes "$VAL_EP" --test-episodes "$TEST_EP" \
    --n-way "$NWAY" --k-shot "$KSHOT" --n-query "$NQUERY" \
    --xy-num-freqs "$XYFREQ" \
    --proto-distance cosine --proto-init-temp 10.0 \
    --device cuda --output-dir "$OUT" $EXTRA
