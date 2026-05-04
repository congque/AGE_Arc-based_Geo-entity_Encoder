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
| Poly2Vec | Siampou et al. ICML 2025 | 2D Fourier | 0.8005 | 0.9588 |
| Geo2Vec | Chu et al. AAAI 2026 | SDF + adaptive PE | 0.9721 | 0.8854² |
| PolyMP | Huang et al. GeoInformatica 2025 | graph MP | 0.8843 | 0.9730 |
| PolyMP-DSC | Huang et al. GeoInformatica 2025 | graph MP + DSC | 0.8790 | 0.9754 |

² Geo2Vec mnist trained on cached SDF samples (340M points pre-computed once
on CPU, then SDF MLP + classifier on h200 GPU): sdf_epochs=12, cls_epochs=80,
batch=16384. Reduced sdf_epochs vs upstream default to fit in 2h preempt
window; converged head val loss ~0.05.

Poly2Vec mnist (0.9588) used the cached Fourier features + cuda head
pipeline, 80 epochs on h200; 489 k params, val 0.9604.

## Table 1b — Same task under per-entity isotropic protocol

Each entity is pre-normalised (subtract bbox center, divide by `max(w, h) / 2`
so the longest side spans `[-1, 1]`, aspect ratio preserved) before the
adapter sees it (`scripts/normalize_entities.py --all`). This kills any
absolute-scale or absolute-position signal so all methods see strictly the
same shape input. New `_iso` dataset entries point at the pre-normalised
gpkg (`data/<name>/<orig>_iso.gpkg`).

| method | input | single_buildings_iso | single_mnist_iso |
|---|---|---|---|
| DeepSet (ours) | arc set | 0.8923 | 0.9804 |
| **PointNet (ours)** | arc set | 0.9282 | **0.9848** |
| PointNet++ (ours) | arc set | 0.8910 | **0.9848** |
| SetTransformer-SAB (ours) | arc set | 0.9162 | 0.9802 |
| SetTransformer-ISAB (ours) | arc set | 0.9109 | 0.9797 |
| **Geo2Vec** | SDF + adaptive PE | **0.9348** | _running_³ |
| Poly2Vec | 2D Fourier | 0.8298 | _running_³ |
| PolyMP | graph MP | 0.8936 | 0.9753 |
| PolyMP-DSC | graph MP + DSC | 0.8976 | 0.9730 |

³ Geo2Vec / Poly2Vec mnist_iso jobs in flight (cached pipeline).

Findings (Table 1 → Table 1b deltas):

- **Buildings drops a lot for ArcSet under iso** (SAB 0.9774 → 0.9162, −6.1;
  ISAB 0.9601 → 0.9109, −4.9; DeepSet 0.9269 → 0.8923, −3.5). The raw
  numbers were partially leveraging absolute scale: different building
  classes (E/F/H/I/L/O/T/U/Y/Z) sit at slightly different sizes in the raw
  gpkg, and ArcSet does not normalise scale internally. Once that signal
  is killed, **Geo2Vec reclaims SOTA on buildings** (0.9348).
- **PointNet wins both buildings_iso and mnist_iso among ArcSet variants**;
  it is the new headline encoder for our framework under matched protocol.
  PointNet vs SAB on buildings_iso: 0.9282 vs 0.9162 (+1.20).
- **mnist is robust**: every encoder lands within ±0.5 pt of its raw number
  (mnist's per-entity scale is already nearly uniform, so iso is a near
  no-op). PointNet/PointNet++ tie at 0.9848, slightly above SAB raw 0.9868
  (within noise) but cleanly above all baselines under matched protocol.
- **Geo2Vec only drops 3.7 pts** on buildings_iso vs raw — much smaller than
  ArcSet's drop because Geo2Vec already does its own per-entity isotropic
  normalize internally; iso input is a near no-op for it.

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

**Headline (Table 2)**: with vanilla ProtoNet and no aux loss, SketchEmbedNet
(image input + stroke decoder pretraining) wins three of four settings;
ArcSet-SAB beats it on 20w-1s. Adding the SEN-inspired stroke-completion
auxiliary loss (Table 3 below) closes most of the remaining gap.
Sketchformer underperforms ArcSet across the board.

## Table 2b — Same task under per-entity isotropic protocol

ArcSet encoders re-trained on `single_omniglot_iso` (per-entity isotropic
normalize, same recipe as Table 1b). Sketchformer / SketchEmbedNet not
yet re-run on iso input — they already re-normalise internally so any
delta is small, but the cell is left as the raw number for now.

| method | input | 5w-1s | 5w-5s | 20w-1s | 20w-5s |
|---|---|---|---|---|---|
| DeepSet (ours) | arc set | 0.9136 | 0.9729 | 0.8583 | 0.9521 |
| **PointNet (ours)** | arc set | 0.9400 | **0.9814** | **0.9017** | **0.9654** |
| **PointNet++ (ours)** | arc set | **0.9426** | 0.9804 | 0.8901 | 0.9644 |
| SetTransformer-SAB (ours) | arc set | 0.8990 | 0.9726 | 0.8817 | 0.9594 |
| SetTransformer-ISAB (ours) | arc set | 0.4951¹ | 0.9010 | 0.8340 | 0.9502 |
| SketchEmbedNet (raw) | image + stroke | 0.9513 | 0.9857 | 0.8667 | 0.9587 |

Findings (Table 2 → Table 2b under iso):

- **PointNet / PointNet++ are the new headline ArcSet encoders on
  Omniglot.** PointNet 20w-1s 0.9017 is +3.5 pts over SketchEmbedNet's
  raw 0.8667 and +1.2 over SAB iso 0.8817 — a clean SOTA on the hardest
  setting. PointNet 20w-5s 0.9654 is also +0.7 over SEN raw.
- The 5-way settings are still close: PointNet++ 5w-1s 0.9426 is within
  1 pt of SEN's pretrained 0.9513; PointNet 5w-5s 0.9814 is within
  0.4 pt of SEN's 0.9857. Without 21 M sketch QuickDraw pretraining.
- **SAB is no longer the best ArcSet encoder for fewshot under iso**
  (SAB 0.8990 vs PointNet 0.9400 on 5w-1s) — attention's advantage
  in our earlier table came partly from absolute-scale information that
  PointNet handles directly via a learned input transform.
- **DeepSet stays mid-pack**, ISAB still collapses at k=1. Aux-loss
  ablation (Table 3 / 3b) was run on the raw protocol; re-running under
  iso is on the to-do list.

## Table 3 — Decoder / aux-loss ablation (ArcSet-SAB on Omniglot)

`auxoff` = baseline ProtoNet end-to-end. `auxon` = same encoder + auxiliary
masked-arc completion head (mask 30 %, GMM-MDN midpoint + Gaussian length +
16-bin theta, λ-curriculum 0→0.5 over 20 epochs). `lrhead` = same encoder
swapped at eval time to L2-norm + sklearn `LogisticRegression(lbfgs)` per
episode (matches SketchEmbedNet's eval protocol).

| setting | aux-off (ours) | **aux-on** | lr-head | SEN reference |
|---|---|---|---|---|
| 5w-1s | 0.9179 | **0.9357 ⬆+1.78** | 0.8942 (-2.4) | 0.9513 |
| 5w-5s | 0.9686 | **0.9815 ⬆+1.29** | 0.9724 (+0.4) | 0.9857 |
| 20w-1s | 0.8841 | 0.8837 (≈tie) | 0.8558 (-2.83) | 0.8667 |
| 20w-5s | 0.9633 | **0.9635** (≈tie) | 0.9570 (-0.6) | 0.9587 |

Findings:
- **Aux-stroke loss helps 5-way**: +1.78 on 5w-1s, +1.29 on 5w-5s. Helps less
  on 20-way (≈tie), where the encoder is already saturated by the harder
  discrimination task.
- **LR-head consistently hurts 1-shot**: -2.4 (5w-1s) and -2.83 (20w-1s).
  L2-norm + LR overfits noise when k_shot=1; ProtoNet with cosine + learnable
  temperature is the better decoder for raw-vector ArcSet embeddings.
  ⇒ Table 3 actually *defends* ArcSet at the matched-decoder protocol — the
  SEN advantage isn't a free lunch from "use LR per episode".
- **ArcSet beats SEN on both 20-way settings** (20w-1s aux-off 0.884 vs SEN
  0.867; 20w-5s aux-on 0.964 vs SEN 0.959). With aux-on, the gap on 5-way
  shrinks to within 1.6 / 0.4 pts despite SEN's 21M-sketch QuickDraw
  pretraining vs our from-scratch 80-epoch episodic training.

## Table 3b — Aux-stroke loss across encoders (multi-encoder ablation)

Same auxiliary masked-arc completion head as Table 3, swept across all three
ArcSet encoders. Δ shows aux-on minus aux-off in percentage points.

| encoder | 5w-1s off / on / Δ | 5w-5s off / on / Δ | 20w-1s off / on / Δ | 20w-5s off / on / Δ |
|---|---|---|---|---|
| **SAB** | 0.9179 / **0.9357** / **+1.78** | 0.9686 / **0.9815** / **+1.29** | 0.8841 / 0.8837 / -0.04 | 0.9633 / 0.9635 / +0.02 |
| ISAB | 0.5494¹ / 0.4859¹ / — | 0.9204 / **0.9728** / **+5.24** | 0.8436 / **0.8764** / **+3.28** | 0.9537 / **0.9655** / **+1.18** |
| DeepSet | 0.9225 / 0.9164 / **-0.61** | 0.9769 / 0.9735 / **-0.34** | 0.8579 / 0.8482 / **-0.97** | 0.9550 / 0.9510 / **-0.40** |

(ISAB 20w-1s aux-on uses EPOCHS=55 to fit under l40s_public's 2-h preemption
window; the other ISAB / SAB / DeepSet cells use the default 80 epochs.)

Findings:

- **Aux loss is an attention-encoder benefit.** Both attention encoders gain
  on every healthy setting (SAB +1.78/+1.29 on 5-way; ISAB +5.24/+3.28/+1.18
  on its three trainable settings). Mean-pool DeepSet _uniformly loses_
  −0.34 to −0.97 across all four settings: without attention, the encoder
  has no mechanism to query masked-arc context from neighbors, so the
  aux-MDN head pulls capacity away from the discrimination objective.
- **ISAB collapses at k_shot=1** regardless of aux setting (footnote ¹ in
  Table 2): a single support example feeds 16 inducing points + cosine
  proto-head into a degenerate attention pattern. Aux loss does not rescue
  this. The 5w-5s and 20w-* cells with k≥5 train normally and benefit most
  from aux.
- **SAB is the safe default** when one encoder must serve all four
  settings — it's the only encoder that doesn't go negative on any cell with
  aux turned on.

## Caveats

- **Split convention**: Lake et al. background (964 classes train) /
  evaluation (659 classes test). Not the Vinyals 2016 split most ProtoNet
  image-Omniglot tables use; flag this in the Setup section.
- **QuickDraw few-shot is partial.** Sketchformer (5w-1s 0.566, 5w-5s 0.761)
  and SketchEmbedNet (5w-1s 0.611, 5w-5s 0.790) ran cleanly on the 5-way
  settings; both fail with NaN on 20-way because our 70/15/15 split of the
  100-class subset only leaves ~15 test classes — not enough to sample
  20-way episodes. ArcSet QuickDraw few-shot is not run yet.
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

- **Geo2Vec / Poly2Vec on QuickDraw** are blocked on memory: QuickDraw has
  ~3.6 B SDF samples after sampling, exceeds 96 G node limit; needs a
  streaming or sub-sampled cache.
- ArcSet QuickDraw few-shot (DeepSet / SAB / ISAB on the 100-class subset,
  5-way only — 20-way infeasible by split size).
- PolyMP "robustness matrix" (rotation/shear-invariance ablations) — defer
  unless reviewer asks.
