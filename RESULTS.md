# ArcSet experiment results

All numbers are single-seed test accuracy (best-val checkpoint), 80 epochs.
Generated on NYU Torch HPC (`/scratch/sx2490/arcset/`); env at
`/scratch/sx2490/arcset/env/conda/envs/arcset/`.

## Table 1 — Polygon shape classification (single-entity supervised)

| method | venue | input | single_buildings | single_mnist |
|---|---|---|---|---|
| DeepSet (ours) | — | arc set | 0.9269 | 0.9803 |
| **SetTransformer-SAB (ours)** | — | arc set | **0.9774** | **0.9868** |
| SetTransformer-ISAB (ours) | — | arc set | 0.9601 | 0.9837 |
| NUFT | Mai et al. GeoInformatica 2023 | rasterised polygon | 0.8523 | 0.9663 |
| PolygonGNN | Yu et al. KDD 2024 | visibility graph | 0.973 | 0.907 |
| Poly2Vec | Siampou et al. ICML 2025 | 2D Fourier | 0.8005 | _pending_ |
| Geo2Vec | Chu et al. AAAI 2026 | SDF + adaptive PE | 0.9721 | _pending_ |
| PolyMP | Huang et al. GeoInformatica 2025 | graph MP | 0.8843 | 0.9730 |
| PolyMP-DSC | Huang et al. GeoInformatica 2025 | graph MP + DSC | 0.8790 | 0.9754 |

`single_mnist` results for Poly2Vec/Geo2Vec are queued on the cached/GPU
pipeline (refactored to precompute features once and train head on GPU).

## Table 2 — Few-shot Omniglot (1623-class stroke, Lake background/evaluation split)

Episodic ProtoNet evaluation: 80 epochs, 200 episodes/epoch, eval over
1000 episodes per setting.

| method | input | 5w-1s | 5w-5s | 20w-1s | 20w-5s |
|---|---|---|---|---|---|
| DeepSet (ours) | arc set | 0.9186 | 0.9749 | 0.8567 | 0.9558 |
| **SetTransformer-SAB (ours)** | arc set | 0.9111 | 0.9759 | **0.8893** | 0.9613 |
| SetTransformer-ISAB (ours) | arc set | 0.5891¹ | 0.8935 | 0.8443 | 0.9476 |
| **SketchEmbedNet** | image + stroke | **0.9513** | **0.9857** | 0.8667 | **0.9587** |
| Sketchformer | stroke seq | 0.7819 | 0.8592 | 0.6315 | 0.8223 |

¹ ISAB at k_shot=1 plateaus due to inducing-point bottleneck under single-support
prototypes; the other ISAB cells train normally.

**Headline**: SketchEmbedNet (image input + stroke decoder pretraining) wins
three of four settings; ArcSet-SAB beats it on the hardest 20w-1s while
operating on raw vector input with no rasterisation or pretraining.
Sketchformer underperforms ArcSet across the board.

## Caveats

- **Split convention**: Lake et al. background (964 classes train) /
  evaluation (659 classes test). Not the Vinyals 2016 split most ProtoNet
  image-Omniglot tables use; flag this in the Setup section.
- **No QuickDraw few-shot** here; only the standard supervised classification
  is benchmarked on the 100-class quickdraw subset.
- Reproducibility: see `baselines/PATCHES.md` for the patches we applied to
  each upstream repo so they run under torch 2.11 / no torch_sparse.
- ArcSet ProtoNet decoder uses cosine similarity with a learnable temperature
  (default `--proto-distance cosine --proto-init-temp 10.0`). Encoder uses
  masked mean pooling for SAB/ISAB to bypass the single-seed PMA collapse on
  long arc sets (default `--sab-pooling mean`).

## Runs / artifacts

- Per-run summaries: `model_edges/results/<run>/summary.json`
- Baseline summaries: `baselines/<name>/results/<dataset>/summary.json` (or
  `_runs/<dataset>/summary_<model>.json` for PolyMP variants)
- Slurm templates under `scripts/` (CPU and GPU partitioned)
- HPC project root: `/scratch/sx2490/arcset/`

## Open work

- 6 cached GPU jobs queued on `h200_tandon`: Poly2Vec/Geo2Vec on mnist /
  omniglot / quickdraw. With the precompute caching refactor each finishes
  in minutes; just waiting on the per-user QOS to clear.
- Stretch: stroke baselines on QuickDraw — sketch sweep already trains in
  ~10 min on h200; can rerun once the queue frees.
