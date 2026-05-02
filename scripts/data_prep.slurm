#!/bin/bash
#SBATCH --job-name=arcset-data-prep
#SBATCH --account=torch_pr_196_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sx2490/arcset/_logs/data-prep-%j.out
#SBATCH --error=/scratch/sx2490/arcset/_logs/data-prep-%j.err

set -e
mkdir -p /scratch/sx2490/arcset/_logs /scratch/sx2490/arcset/data_raw
cd /scratch/sx2490/arcset
source /scratch/sx2490/arcset/env/conda/etc/profile.d/conda.sh
conda activate arcset

# (single_buildings.gpkg already pushed via rsync; single_mnist.gpkg via Git LFS — fetch it)

# ============= 1. PolygonGNN HF dataset (small + topo pairs) =============
echo "=== [1/4] PolygonGNN HF download ==="
python <<'PY'
from huggingface_hub import hf_hub_download
import os
os.makedirs("data_raw/polygongnn", exist_ok=True)
for f in ["building_with_index.pkl", "mp_building.pkl",
          "multi_mnist_sparse.pkl", "single_building.pkl", "single_mnist.pkl"]:
    p = hf_hub_download("Dzy6/PolygonGNN", f, repo_type="dataset", local_dir="data_raw/polygongnn")
    print(f"downloaded: {p}")
PY

# topo_polygongnn/pairs.pkl already rsynced; if missing regenerate
if [ ! -f data/topo_polygongnn/pairs.pkl ]; then
    python scripts/prepare_polygongnn.py
fi

# ============= 2. Single MNIST polygon (Git LFS) =============
echo "=== [2/4] check single_mnist gpkg ==="
ls -la data/single_mnist/ 2>&1
if [ ! -s data/single_mnist/mnist_scaled_normalized.gpkg ] || \
   [ "$(stat -c%s data/single_mnist/mnist_scaled_normalized.gpkg 2>/dev/null)" -lt 1000000 ]; then
    echo "single_mnist gpkg missing or LFS pointer -- need to copy from another source"
    echo "(SKIP for now; ArcSet 1.1 retrain on mnist will fail until this is provided)"
else
    echo "single_mnist OK"
fi

# ============= 3. Omniglot strokes =============
echo "=== [3/4] Omniglot ==="
mkdir -p data_raw/omniglot
cd data_raw/omniglot
if [ ! -f strokes_background.zip ]; then
    wget --no-check-certificate -q https://github.com/brendenlake/omniglot/raw/master/python/strokes_background.zip
fi
if [ ! -f strokes_evaluation.zip ]; then
    wget --no-check-certificate -q https://github.com/brendenlake/omniglot/raw/master/python/strokes_evaluation.zip
fi
[ ! -d strokes_background ] && unzip -q strokes_background.zip
[ ! -d strokes_evaluation ] && unzip -q strokes_evaluation.zip
cd /scratch/sx2490/arcset
python scripts/prepare_omniglot.py

# ============= 4. QuickDraw (100 cls × 10k) =============
echo "=== [4/4] QuickDraw 100x10k ==="
if [ ! -s data/single_quickdraw/quickdraw.gpkg ] || \
   [ "$(stat -c%s data/single_quickdraw/quickdraw.gpkg 2>/dev/null)" -lt 100000000 ]; then
    python scripts/prepare_quickdraw.py --num-classes 100 --samples-per-class 10000
else
    echo "quickdraw.gpkg already present"
fi

echo "=== data prep done ==="
ls -lh data/*/*.gpkg data/*/*.pkl 2>&1
