# AGE: Arc-based Geo-entity Encoder

Code, scripts and partial results for the ArcSet experiments on single-entity
shape classification, low-shot, few-shot, and pair-wise relation tasks.

## Included

### Models (`model_edges/`)
- `entitydeepset.py` — ArcSet-DeepSet (φ → sum/sum_mean → ρ → head)
- `entitysettransformer_sab.py` — ArcSet-SetTransformer with full self-attention (SAB)
- `entitysettransformer_isab.py` — ArcSet-SetTransformer with induced-set attention (ISAB)
- `load_entities.py` — geometry → arc-feature edge sets, with dynamic Fourier-frequency selection
- `test.py` — standard supervised classification driver
- `testfs.py` — episodic few-shot with Prototypical Network decoder
- `test_topo.py` — siamese pair-wise relation classifier

### Drivers (`scripts/`)
- `prepare_omniglot.py` — convert Omniglot stroke txt files → MultiLineString gpkg (1623 classes)
- `prepare_quickdraw.py` — stream Google QuickDraw simplified ndjson → MultiLineString gpkg
- `prepare_polygongnn.py` — convert PolygonGNN HeteroData pkl → simple `(poly_a, poly_b, label)` pickle
- `run_standard_classification.sh` — 1.1 standard matrix on buildings + mnist × {deepset, sab, isab}
- `run_baselines_buildings.sh` / `run_baselines_mnist.sh` — full driver sequencing all polygon baselines
- `run_wave_a_buildings.sh` / `run_wave_b_mnist.sh` — staged baseline runs (per-MPS-availability)
- `run_nuft_cpu.sh` — CPU-only NUFT pipeline (DDSL needs float64)
- `run_fewshot_priority.sh` — episodic few-shot omniglot, 4 settings × 3 ArcSet models

### Datasets (`data/`)
- `single_buildings/ShapeClassification.gpkg` — 10-class polygon, ~5 k samples (in-repo)
- `single_mnist/mnist_scaled_normalized.gpkg` — 10-class polygon, ~50 k (Git-LFS)
- `single_omniglot/omniglot.gpkg` — 1623-class MultiLineString stroke, ~32 k (regenerable, gitignored)
- `single_quickdraw/quickdraw.gpkg` — 100-class MultiLineString stroke, ~890 k (regenerable, gitignored)
- `topo_polygongnn/pairs.pkl` — 5000 polygon pairs, 100-class shape-pair labels (small, tracked)

## Environment

Tested with osx-arm64 conda (`miniforge3`):

```bash
conda create -n age -c conda-forge python=3.10 numpy 'pytorch>=2.10' \
    geopandas shapely tqdm requests 'huggingface_hub>=0.23' -y
conda activate age
pip install torch_geometric tensorboard fiona pyarrow osmnx triangle
```

Apple Silicon: PyTorch automatically uses MPS where supported; fall back to
CPU when needed (`PYTORCH_ENABLE_MPS_FALLBACK=1`). For DDSL / Geo2Vec set
`KMP_DUPLICATE_LIB_OK=TRUE`.

## Reproducing the data

```bash
git lfs pull                                          # single_mnist
python scripts/prepare_omniglot.py                    # → data/single_omniglot/
python scripts/prepare_quickdraw.py --num-classes 100 \
    --samples-per-class 10000                          # → data/single_quickdraw/
python scripts/prepare_polygongnn.py                  # → data/topo_polygongnn/
```

## Standard classification (table 1.1)

```bash
EPOCHS=80 bash scripts/run_standard_classification.sh
```

Single-run usage:
```bash
python -u model_edges/test.py \
    --dataset single_buildings \
    --set-model settransformer-sab \
    --epochs 80 \
    --output-dir model_edges/results/settransformer_sab_single_buildings
```

`--set-model` choices: `deepset`, `settransformer-sab`, `settransformer-isab`.

`--xy-num-freqs auto` (default) picks `clip(ceil(log2(avg_arcs)) + 3, 6, 9)`
based on the loaded gpkg, so polygon datasets get 7 and dense stroke datasets
saturate to 9. Pass an integer to override.

## Few-shot Omniglot (Prototypical Network)

```bash
python -u model_edges/testfs.py \
    --dataset single_omniglot \
    --set-model settransformer-sab \
    --n-way 5 --k-shot 1 --n-query 15 \
    --epochs 80 --train-episodes 200 --val-episodes 200 --test-episodes 600
```

Episodic training only — no pretraining; the encoder is updated against the
ProtoNet loss directly. Class splits use the Omniglot `background`/`evaluation`
columns when present.

## Pair-wise relation classification (siamese)

```bash
python -u model_edges/test_topo.py \
    --input data/topo_polygongnn/pairs.pkl \
    --set-model settransformer-sab \
    --epochs 80
```

The two polygons are encoded **separately** by a shared ArcSet encoder; the
4× combination feature `[z_a, z_b, |z_a − z_b|, z_a · z_b]` feeds an MLP head.
Joint encoding (treating the two entities as one set) is intentionally avoided
even though ArcSet supports it — that is a different setting.

## Baselines

Cloned upstream into `baselines/<name>/` (gitignored), patched to run under
torch 2.11 / no torch_sparse. See [`baselines/PATCHES.md`](baselines/PATCHES.md)
for the per-baseline patch list and adapter instructions.

| baseline | venue | adapter | drivers |
|---|---|---|---|
| NUFT | GeoInformatica 2023 | `baselines/nuft/run_arcset_dataset.py` | `scripts/run_nuft_cpu.sh` |
| PolygonGNN | KDD 2024 | uses upstream `train_new.py --dataset sbuilding/smnist` | wave A/B drivers |
| Poly2Vec | ICML 2025 | `baselines/poly2vec/run_arcset_dataset.py` | wave A/B drivers |
| Geo2Vec | AAAI 2026 oral | `baselines/geo2vec/run_arcset_dataset.py` | wave A/B drivers |
| PolyMP | GeoInformatica 2025 | `baselines/polymp/run_arcset_dataset.py` | wave B driver |

## Partial results

Best test accuracy after 80 epochs (single-seed; SAB is our headline number):

| method | single_buildings | single_mnist |
|---|---|---|
| **ArcSet-SAB (ours)** | **0.9827** | **0.9854** |
| PolygonGNN | 0.973 | (running) |
| ArcSet-ISAB | 0.967 | 0.982 |
| ArcSet-DeepSet | 0.932 | 0.981 |
| NUFT | 0.850 | 0.966 |
| Poly2Vec | 0.809 | (running) |
| Geo2Vec | 0.086 ⚠ misconfigured first run | (running) |
| PolyMP / DSC | (queued) | (queued) |
