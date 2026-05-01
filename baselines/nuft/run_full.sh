#!/usr/bin/env bash
# Full 80-epoch NUFT shape-classification runs on the two ArcSet datasets that
# contain (Multi)Polygons. Do NOT run this for the smoke test.
#
# Notes
# -----
# * Uses the same conda env as the smoke test (`age`, see README in this folder).
# * NUFT requires polygons -> we explicitly skip the open-curve datasets
#   (`single_omniglot`, `single_quickdraw`); see run_arcset_dataset.py.
# * Device: prefer cuda if available, else cpu. MPS is forced off because
#   DDSL needs float64 (not supported on Apple GPUs).
set -euo pipefail
cd "$(dirname "$0")"

ENV_PYTHON=${ENV_PYTHON:-/Users/alexxu/miniforge3/envs/age/bin/python}
DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-80}
BATCH=${BATCH:-64}
LR=${LR:-1e-3}
NUM_VERT=${NUM_VERT:-64}
FREQ=${FREQ:-16}
EMBED=${EMBED:-64}
ENCODER=${ENCODER:-nuft_ddsl}   # nuft_ddsl | nuftifft_ddsl | nuft_specpool

run_one () {
    local NAME=$1
    local GPKG=$2
    local LABEL=$3
    echo "=========================================================="
    echo "Dataset: ${NAME}"
    echo "GPKG:    ${GPKG}"
    echo "=========================================================="
    "${ENV_PYTHON}" run_arcset_dataset.py \
        --gpkg "${GPKG}" \
        --label_col "${LABEL}" \
        --num_vert "${NUM_VERT}" \
        --num_epoch "${EPOCHS}" \
        --batch_size "${BATCH}" \
        --lr "${LR}" \
        --pgon_enc "${ENCODER}" \
        --pgon_embed_dim "${EMBED}" \
        --nuft_freqXY "${FREQ}" "${FREQ}" \
        --device "${DEVICE}" \
        --out_dir "_runs/${NAME}" 2>&1 | tee "_runs/${NAME}.log"
}

mkdir -p _runs

run_one single_buildings ../../data/single_buildings/ShapeClassification.gpkg label
run_one single_mnist     ../../data/single_mnist/mnist_scaled_normalized.gpkg label
