# Baseline Reproduction Notes

We compare ArcSet against four open-source polygon / geo-entity encoders.
Each upstream repo is cloned into `baselines/<name>/` and is **gitignored** —
the collaborator must clone them locally, then apply the patches listed below.

| name | upstream repo | venue / year |
|---|---|---|
| nuft | https://github.com/gengchenmai/polygon_encoder | Mai et al., GeoInformatica 2023 |
| polygongnn | https://github.com/dyu62/PolyGNN | Yu et al., KDD 2024 |
| poly2vec | https://github.com/USC-InfoLab/poly2vec | Siampou et al., ICML 2025 |
| geo2vec | https://github.com/chuchen2017/GeoNeuralRepresentation | Chu et al., AAAI 2026 oral |
| polymp | https://github.com/zexhuang/PolyMP | Huang et al., GeoInformatica 2025 |

For each, `baselines/<name>/run_arcset_dataset.py` is our adapter that:
- loads our gpkg via geopandas,
- converts each shape to the upstream input format (rasterized indicator,
  HeteroData, vertex tensor, etc.),
- runs upstream `Trainer` / training loop with their published hyperparameters,
- prints a final `test_acc` and `macro_f1`.

`baselines/<name>/run_full.sh` is the 80-epoch driver. The umbrella driver
`scripts/run_wave_a_buildings.sh` and `scripts/run_wave_b_mnist.sh` orchestrate
all three baselines on each polygon dataset.

## Setup
```bash
cd baselines
git clone https://github.com/gengchenmai/polygon_encoder nuft
git clone https://github.com/dyu62/PolyGNN polygongnn
git clone https://github.com/USC-InfoLab/poly2vec poly2vec
git clone https://github.com/chuchen2017/GeoNeuralRepresentation geo2vec
git clone https://github.com/zexhuang/PolyMP polymp
```

Then copy `run_arcset_dataset.py` and `run_full.sh` into each respective
directory (or fetch them from this branch via `git checkout Alex --
baselines/<name>/run_arcset_dataset.py`).

## Per-baseline patches

These are required to make the upstream code run under our env (Python 3.10,
torch 2.11, torch_geometric 2.7, no torch_sparse / torch_scatter compiled
extensions, MPS device).

### nuft
- `polygoncode/polygonembed/trainer_helper.py`: in `eval_pgon_cla_model`,
  fall back to the in-memory model when the saved checkpoint is missing;
  add a terminal-checkpoint write so `load_model` finds something to read;
  use `weights_only=False` for torch 2.11 namespace deserialisation.
- `run_arcset_dataset.py:238`: skip the wrapper's redundant post-train
  `run_eval()` because `Trainer.run_train` already evaluated.
- DDSL casts to float64 inside the encoder, which MPS does not support — the
  adapter forces `--device cpu`.

### polygongnn
- `model_new.py`: replaced `from torch_scatter import scatter` with
  `from torch_geometric.utils import scatter` (drop-in compatible).
- `util.py`: rewrote the `triplets()` function (originally used
  `torch_sparse.SparseTensor`) using only torch built-ins (cumsum, argsort,
  scatter_add_) — verified equivalent on 50 random graphs.

### poly2vec
- `models/fourier_encoder.py:preprocess_polygon`: when shapely's `buffer(0)`
  cleanup returns a `MultiPolygon` (self-intersecting input), collapse to the
  largest piece so the downstream `.exterior` access works.
- Adapter requires `pip install triangle` for constrained Delaunay
  triangulation (no GPU/MPS path; encoder forced to CPU).

### geo2vec
- `runners/list2embedding.py:88` (and around 273): initialise
  `shape_embedding` / `entity_embedding` so the function does not raise
  `UnboundLocalError` when only the shape branch runs.
- Adapter exposes upstream defaults (`num_layers=8`, `z_size=256`,
  `samples_perUnit_shape=100`, `point_sample_shape=20`,
  `code_reg_weight_shape=1.0`). The first round of smoke runs accidentally
  used downscaled values and the model never converged — see `summary.json`
  in `_runs/single_buildings/` for a 8.6% chance-level outcome from that
  bad config; the proper run is pending.

### polymp
- `data/dataset.py`: prepended `DDSL/` to `sys.path` so the bundled DDSL
  package imports cleanly; added optional `compute_spec=False` and support
  for both `(N, 2)` and `(2, N)` coord layouts.
- `train/trainer.py`: made `torchinfo` optional, fixed cfg type handling,
  and dropped the `verbose=` kwarg from `CosineAnnealingWarmRestarts` (not
  supported in our PyTorch build).

## Running a baseline
```bash
# example: NUFT on single_buildings, 80 epochs
cd /path/to/AGE_Arc-based_Geo-entity_Encoder
source <conda>/etc/profile.d/conda.sh
conda activate age
export KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1
bash scripts/run_wave_a_buildings.sh   # PolygonGNN + Poly2Vec + Geo2Vec
bash scripts/run_wave_b_mnist.sh       # same on mnist + PolyMP both datasets
bash scripts/run_nuft_cpu.sh           # NUFT (CPU-only) buildings + mnist
```
