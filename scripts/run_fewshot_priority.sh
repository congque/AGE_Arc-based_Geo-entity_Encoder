#!/bin/bash
# Few-shot omniglot priority pipeline:
# SAB and ISAB first (proven to learn), then DeepSet as baseline reference.
# 4 settings (5w-1s, 5w-5s, 20w-1s, 20w-5s) x 3 models = 12 runs total.
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

# Priority order: SAB + ISAB first (capacity proven sufficient), then DeepSet
for model in settransformer-sab settransformer-isab deepset; do
  for nway in 5 20; do
    for kshot in 1 5; do
      tag="${model//-/_}"
      out="model_edges/results/fs_${tag}_single_omniglot_${nway}w_${kshot}s"
      log="${out}/run.log"
      # Skip if summary already exists (resume support)
      if [ -f "$out/summary.json" ]; then
        echo "[fs] SKIP ${out} (summary exists)"
        continue
      fi
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
