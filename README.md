# AGE: Arc-based Geo-entity Encoder

Minimal code and data for reproducing our single-entity edge-set experiments.

## Included

- `model_edges/entitydeepset.py`
- `model_edges/entitysettransformer.py`
- `model_edges/load_entities.py`
- `model_edges/test_entitydeepset_shapeclassification.py`
- `model_edges/test_entitysettransformer_shapeclassification.py`
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

### DeepSet on single_buildings

```bash
python -u model_edges/test_entitydeepset_shapeclassification.py \
  --input data/single_buildings/ShapeClassification.gpkg \
  --epochs 100 \
  --length-fourier \
  --second-harmonic \
  --output-dir model_edges/results/entitydeepset_single_buildings
```

### SetTransformer on single_buildings

```bash
python -u model_edges/test_entitysettransformer_shapeclassification.py \
  --input data/single_buildings/ShapeClassification.gpkg \
  --epochs 100 \
  --length-fourier \
  --second-harmonic \
  --output-dir model_edges/results/entitysettransformer_single_buildings
```

### DeepSet on single_mnist

```bash
python -u model_edges/test_entitydeepset_shapeclassification.py \
  --input data/single_mnist/mnist_scaled_normalized.gpkg \
  --epochs 100 \
  --length-fourier \
  --second-harmonic \
  --output-dir model_edges/results/entitydeepset_single_mnist
```

### SetTransformer on single_mnist

```bash
python -u model_edges/test_entitysettransformer_shapeclassification.py \
  --input data/single_mnist/mnist_scaled_normalized.gpkg \
  --epochs 100 \
  --length-fourier \
  --second-harmonic \
  --output-dir model_edges/results/entitysettransformer_single_mnist
```

## Notes

- Both models use the same edge features for fair comparison.
- The current best-performing setting in our experiments uses:
  - `xy_num_freqs = 8`
  - `length_fourier = True`
  - `second_harmonic = True`
