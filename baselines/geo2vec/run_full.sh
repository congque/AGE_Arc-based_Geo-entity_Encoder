#!/usr/bin/env bash
# Geo2Vec baseline -- full 80-epoch sweep across all four ArcSet shape-
# classification datasets.
#
# DO NOT RUN AS PART OF THE SMOKE TEST. SDF sampling for omniglot/quickdraw
# (multilinestring entities) is several hours on Apple M-series hardware
# even with multiprocessing (the SDF computation itself is shapely / pure-CPU
# and is the dominant cost; only the SDF MLP and the classifier head run on
# MPS).
#
# Usage (from the repo root):
#   bash baselines/geo2vec/run_full.sh
#
# Override the python interpreter via:
#   PY=/path/to/python bash baselines/geo2vec/run_full.sh

set -euo pipefail

PY=${PY:-/Users/alexxu/miniforge3/envs/age/bin/python}
DEVICE=${DEVICE:-mps}
SDF_EPOCHS=${SDF_EPOCHS:-80}
CLS_EPOCHS=${CLS_EPOCHS:-80}
NUM_PROCESS=${NUM_PROCESS:-8}
OUT_DIR=${OUT_DIR:-baselines/geo2vec/runs}
mkdir -p "${OUT_DIR}"

run_one () {
    local DATASET=$1
    local LOG="${OUT_DIR}/${DATASET}.log"
    local EMB="${OUT_DIR}/${DATASET}_emb.npy"
    echo "=================================================="
    echo "Geo2Vec full run -> ${DATASET}"
    echo "log: ${LOG}"
    echo "=================================================="
    "${PY}" baselines/geo2vec/run_arcset_dataset.py \
        --dataset "${DATASET}" \
        --device "${DEVICE}" \
        --sdf-epochs "${SDF_EPOCHS}" \
        --cls-epochs "${CLS_EPOCHS}" \
        --num-process "${NUM_PROCESS}" \
        --save-emb "${EMB}" \
        2>&1 | tee "${LOG}"
}

run_one single_buildings
run_one single_mnist
run_one single_quickdraw
run_one single_omniglot   # heaviest; run last
