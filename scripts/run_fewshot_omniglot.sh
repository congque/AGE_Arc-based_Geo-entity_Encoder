#!/bin/bash
# Few-shot Omniglot: 4 settings (5w-1s, 5w-5s, 20w-1s, 20w-5s) x 3 ArcSet models.
# Each: 80 epochs of episodic training with Prototypical Network decoder.
set -uo pipefail
cd "$(dirname "$0")/.."

source /Users/alexxu/miniforge3/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

EPOCHS=${EPOCHS:-80}
TRAIN_EP=${TRAIN_EP:-200}
VAL_EP=${VAL_EP:-200}
TEST_EP=${TEST_EP:-600}

for nway in 5 20; do
  for kshot in 1 5; do
    for model in deepset settransformer-sab settransformer-isab; do
      tag="${model//-/_}"
      out="model_edges/results/fs_${tag}_single_omniglot_${nway}w_${kshot}s"
      log="${out}/run.log"
      mkdir -p "$out"
      echo "================================================================"
      echo "[fs] $model $nway w $kshot s -> $out"
      echo "================================================================"
      python -u model_edges/testfs.py \
        --dataset single_omniglot --set-model "$model" \
        --epochs "$EPOCHS" --train-episodes "$TRAIN_EP" \
        --val-episodes "$VAL_EP" --test-episodes "$TEST_EP" \
        --n-way "$nway" --k-shot "$kshot" --n-query 15 \
        --output-dir "$out" 2>&1 | tee "$log"
    done
  done
done

echo "all 12 few-shot omniglot runs done"
