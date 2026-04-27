# AGE: Arc-based Geo-entity Encoder

Minimal code and data for reproducing our single-entity edge-set experiments.

## Included

- `model_edges/entitydeepset.py`
- `model_edges/entitysettransformer.py`
- `model_edges/load_entities.py`
- `model_edges/test.py`
- `data/single_buildings/ShapeClassification.gpkg`
- `data/single_mnist/mnist_scaled_normalized.gpkg`

## Environment

Tested with:

- Python 3.10
- PyTorch
- NumPy
- GeoPandas
- Shapely

## Run

The unified entry is:

```bash
python -u model_edges/test.py --dataset <dataset> --set-model <deepset|settransformer>
```

Datasets:

- `single_buildings`
- `single_mnist`

Default edge features already use the current best setting:

- `xy_num_freqs = 8`
- `length_fourier = True`
- `second_harmonic = True`
- `use_endpoints = False`

### DeepSet on single_buildings

```bash
python -u model_edges/test.py \
  --dataset single_buildings \
  --set-model deepset \
  --epochs 100 \
  --output-dir model_edges/results/deepset_single_buildings
```

### SetTransformer on single_buildings

```bash
python -u model_edges/test.py \
  --dataset single_buildings \
  --set-model settransformer \
  --epochs 100 \
  --output-dir model_edges/results/settransformer_single_buildings
```

### DeepSet on single_mnist

```bash
python -u model_edges/test.py \
  --dataset single_mnist \
  --set-model deepset \
  --epochs 100 \
  --output-dir model_edges/results/deepset_single_mnist
```

### SetTransformer on single_mnist

```bash
python -u model_edges/test.py \
  --dataset single_mnist \
  --set-model settransformer \
  --epochs 100 \
  --output-dir model_edges/results/settransformer_single_mnist
```

## Notes

- Both models use the same edge features for fair comparison.
- If needed, you can override defaults with:
  - `--no-length-fourier`
  - `--no-second-harmonic`
  - `--xy-num-freqs`
  - `--length-num-freqs`
