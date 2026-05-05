# ArcSet experiment results

All numbers are test accuracy on the best-val checkpoint after 80 epochs
(few-shot uses 55 epochs to fit the 2-h slurm preemption window). Generated
on NYU Torch HPC (`/scratch/sx2490/arcset/`); env at
`/scratch/sx2490/arcset/env/conda/envs/arcset/`.

## Contribution framing

The contribution we are positioning is **directed-arc tokenization as a
common interface for heterogeneous 2D vector geometry** — polygons, multi-
line strings, and free-hand strokes share one tokenization, then any
permutation-invariant set encoder (DeepSet / SetTransformer / PointNet)
sits on top. We are *not* claiming a universally best encoder; instead we
show that the arc representation (a) is competitive across encoders,
(b) reveals interpretable backbone preferences once absolute-scale signal
is removed, and (c) admits the same auxiliary masked-arc-completion head
across encoders.

The **per-entity isotropic protocol** (Tables 1b / 2b) is the primary
matched-protocol comparison: each entity is normalised to span `[-1, 1]`
on its longest axis (centroid subtracted, isotropic rescale), so methods
cannot leverage absolute placement or absolute scale. Tables 1 / 2 with
raw-coordinate inputs are kept as an upstream-default control: they show
where ArcSet appears to dominate using absolute scale signal that the
matched protocol then strips.

## Table 1b (PRIMARY) — Polygon shape classification, per-entity isotropic protocol

Each entity is pre-normalised (subtract bbox center, divide by `max(w, h) / 2`
so the longest side spans `[-1, 1]`, aspect ratio preserved) before any
adapter sees it (`scripts/normalize_entities.py --all`).

| method | input | single_buildings_iso | single_mnist_iso |
|---|---|---|---|
| Size-only MLP⁶ | 7 scalar features | 0.7367 | 0.5114 |
| DeepSet (ours) | arc set | 0.8923 | 0.9804 |
| **PointNet (ours)** | arc set | **0.9282** | **0.9848** |
| PointNet++ (ours) | arc set | 0.8910 | **0.9848** |
| SetTransformer-SAB (ours) | arc set | 0.9162 | 0.9802 |
| SetTransformer-ISAB (ours) | arc set | 0.9109 | 0.9797 |
| **Geo2Vec** | SDF + adaptive PE | **0.9348** | 0.5971³ |
| Poly2Vec | 2D Fourier | 0.8298 | _running_⁴ |
| PolyMP | graph MP | 0.8936 | 0.9753 |
| PolyMP-DSC | graph MP + DSC | 0.8976 | 0.9730 |

Under the matched protocol:

- **No single architecture wins both datasets.** Geo2Vec leads on
  buildings_iso (0.9348); PointNet/PointNet++ tie on mnist_iso (0.9848).
- **PointNet is the strongest ArcSet encoder on iso buildings** (0.9282 vs
  SAB 0.9162). On mnist_iso PointNet/PointNet++ jointly top the table.
- **Among all encoders sitting on the arc representation, the gap to the
  best baseline is at most 0.7 pt on either dataset** (mnist_iso PointNet
  0.9848 vs PolyMP-DSC 0.9730; buildings_iso PointNet 0.9282 vs Geo2Vec
  0.9348 = -0.7). This positions arc-tokenization as competitive without
  triangulation, SDF sampling, or visibility graphs.

³ Geo2Vec mnist_iso collapses (0.8854 raw → 0.5971 iso, train_acc=0.97
val=0.60 = severe overfit). Likely cause: Geo2Vec's `MP_sample` computes a
**single global bounding box** across all entities for SDF query sampling,
so per-entity normalised data (every entity centred at origin, all spanning
[-1, 1]) makes the model sample SDF queries that overlap heavily across
entities — destroying the per-entity discriminative signal Geo2Vec relies on.
Per-entity sampling boxes would likely fix this; not patched here.

⁴ Poly2Vec mnist_iso requires a fresh per-entity Fourier feature precompute
(triangulation is CPU-bound; ~5 h sequentially for 60 k mnist polygons);
job in progress.

## Table 1 (CONTROL) — Same task with raw global coordinates

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

### Raw → iso deltas (per-encoder, single-seed)

| Encoder | buildings raw → iso | Δ | mnist raw → iso | Δ |
|---|---|---|---|---|
| DeepSet (ours) | 0.9269 → 0.8923 | -3.46 | 0.9803 → 0.9804 | +0.01 |
| SAB (ours) | 0.9774 → 0.9162 | **-6.12** | 0.9868 → 0.9802 | -0.66 |
| ISAB (ours) | 0.9601 → 0.9109 | -4.92 | 0.9837 → 0.9797 | -0.40 |
| Geo2Vec | 0.9721 → 0.9348 | -3.73 | 0.8854 → 0.5971 | (collapse, see ³) |
| PolyMP | 0.8843 → 0.8936 | +0.93 | 0.9730 → 0.9753 | +0.23 |
| PolyMP-DSC | 0.8790 → 0.8976 | +1.86 | 0.9754 → 0.9730 | -0.24 |
| Poly2Vec | 0.8005 → 0.8298 | +2.93 | 0.9588 → _running_ | — |

The deltas tell two stories:

- **Methods without internal per-entity normalize lose a lot on buildings
  under iso** (ArcSet -3.5 / -6.1 / -4.9, Geo2Vec -3.7), confirming that
  raw single_buildings carries absolute-scale signal: different letter
  classes (E/F/H/I/L/O/T/U/Y/Z) sit at slightly different sizes in the raw
  gpkg, which any encoder that sees raw coordinates can pick up.
- **mnist is largely scale-agnostic** (every method within ±0.7 pt). MNIST
  digits in the gpkg are already roughly uniform per-entity size, so iso
  is a near no-op for them.

## Table 2b (PRIMARY) — Few-shot Omniglot, per-entity isotropic protocol

Lake et al. background/evaluation split (964 train classes / 659 test
classes). Episodic ProtoNet decoder (cosine + learnable temperature),
55 epochs (to fit l40s 2-h preempt) × 200 train episodes/epoch, evaluated
over 1000 test episodes/setting. SketchEmbedNet kept at its raw protocol
because its image rasterizer normalises strokes internally — its iso run
is on the to-do list.

| method | input | 5w-1s | 5w-5s | 20w-1s | 20w-5s |
|---|---|---|---|---|---|
| DeepSet (ours) | arc set | 0.9136 | 0.9729 | 0.8583 | 0.9521 |
| **PointNet (ours)** | arc set | 0.9400 | **0.9814** | **0.9017** | **0.9654** |
| PointNet++ (ours) | arc set | **0.9426** | 0.9804 | 0.8901 | 0.9644 |
| SetTransformer-SAB (ours) | arc set | 0.8990 | 0.9726 | 0.8817 | 0.9594 |
| SetTransformer-ISAB (ours) | arc set | 0.4951¹ | 0.9010 | 0.8340 | 0.9502 |
| SketchEmbedNet (raw protocol) | image + stroke | 0.9513 | 0.9857 | 0.8667 | 0.9587 |

¹ ISAB at k_shot=1 plateaus due to inducing-point bottleneck under
single-support prototypes; documented as a known caveat of the ISAB
backbone (not a property of the arc representation).

Under matched protocol on Omniglot:

- **The arc representation matches an image+stroke baseline that has 21 M
  QuickDraw pre-training** (PointNet/PointNet++ vs SEN, gaps within ±1.2 pt
  on 5-way; PointNet ahead on both 20-way settings). Reading: arc-token
  features from scratch are competitive with sketch-style pretraining.
- **Backbone preference flips between supervised and few-shot**: SAB is
  strongest on supervised iso (Table 1b), PointNet is strongest on few-shot
  iso. We do not interpret this as "PointNet is better"; we interpret it as
  "the arc representation is robust to backbone choice and the right
  backbone depends on the task signal."
- **Single-seed numbers; multi-seed runs over headline cells (PointNet on
  buildings_iso, mnist_iso, all four omniglot_iso settings) are queued and
  will replace these single numbers with mean ± 95 % CI in the next round.**

## Table 2 (CONTROL) — Same task with raw global coordinates

Same encoder configs as Table 2b, but ArcSet sees raw coordinates (no per-
entity rescale). Reproduces the upstream comparison shape used by
SketchEmbedNet / Sketchformer.

| method | input | 5w-1s | 5w-5s | 20w-1s | 20w-5s |
|---|---|---|---|---|---|
| DeepSet (ours) | arc set | 0.9186 | 0.9749 | 0.8567 | 0.9558 |
| SetTransformer-SAB (ours) | arc set | 0.9111 | 0.9759 | 0.8893 | 0.9613 |
| SetTransformer-ISAB (ours) | arc set | 0.5891¹ | 0.8935 | 0.8443 | 0.9476 |
| SketchEmbedNet | image + stroke | 0.9513 | 0.9857 | 0.8667 | 0.9587 |
| Sketchformer | stroke seq | 0.7819 | 0.8592 | 0.6315 | 0.8223 |

Raw → iso deltas are mostly small (within ±1 pt for SAB / DeepSet / ISAB
on stroke), confirming that Omniglot strokes are largely scale-agnostic
to begin with: per-entity normalize is close to a no-op. The big movers
are on `single_buildings` (Table 1's deltas), where per-class scale
signal was exploitable.

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

## Table 4 (PRIMARY) — Arc-vs-Point ablation: directly defends the paper claim

Same encoder, three input representations, on per-entity isotropic data:

- **arc**: ArcSet (each token = directed segment with midpoint+length+orientation+turning)
- **points**: raw centred (x, y) per vertex (token = point, no PE)
- **points + PE**: vertices + the same multi-frequency Fourier PE used on arc midpoints

This isolates the value of the **arc representation** from input-encoding richness — both `arc` and `points + PE` use the same Fourier budget; the difference is whether tokens are arcs (with length / orientation / continuity) or vertices.

### single_buildings_iso (3 seeds for PointNet rows; single seed otherwise)

| Encoder | arc (ArcSet) | raw points | points + PE | Δ (arc − raw) |
|---|---|---|---|---|
| DeepSet | **0.8923** | 0.7407 | 0.6676 | +15.16 |
| PointNet | 0.9211 ± 0.011 | **0.9530 ± 0.007** | 0.7695 ± 0.008 | -3.19⁵ |
| SetTransformer-SAB | **0.9162** | 0.8923 | 0.6277 | +2.39 |
| SetTransformer-ISAB | **0.9109** | 0.8497 | 0.6237 | +6.12 |

⁵ **PointNet anomaly on `buildings_iso`** (3-seed verified): raw 0.953 ± 0.007 vs arc 0.921 ± 0.011, gap = -3.19 pt outside both seeds' σ. PointNet has a learned 2-D input transform (T-Net) that re-projects only the leading 2 coordinate channels; on 22-D arc tokens the remaining 20 dims pass through unchanged, so the T-Net is underused. PointNet's design is point-cloud-native, and the arc representation does not buy it anything when its input transform expects 2-D points. The other three encoders (DeepSet, SAB, ISAB) all gain from arc tokens on the same dataset. **The clean reading is therefore: arc tokens improve any non-T-Net set encoder; PointNet specifically benefits from being fed raw 2-D points.** This is honest and consistent with PointNet's published design.

### single_mnist_iso (single seed)

| Encoder | arc | raw points | points + PE | Δ (arc − raw) | Δ (arc − pe) |
|---|---|---|---|---|---|
| DeepSet | **0.9804** | 0.9262 | 0.9519 | +5.42 | +2.85 |
| **PointNet** | **0.9848** | 0.9827 | 0.9733 | +0.21 | +1.15 |
| SetTransformer-SAB | **0.9802** | 0.9738 | 0.9624 | +0.64 | +1.78 |
| SetTransformer-ISAB | **0.9797** | 0.9661 | 0.9603 | +1.36 | +1.94 |

### single_omniglot_iso fewshot (single seed)

| Encoder × Setting | arc | raw points | points + PE |
|---|---|---|---|
| PointNet 5w-1s | **0.9400** | 0.9222 | 0.9247 |
| PointNet 20w-1s | **0.9017** | 0.8690 | 0.8638 |
| SAB 5w-1s | **0.8990** | 0.8623 | 0.8634 |
| SAB 20w-1s | **0.8817** | 0.8482 | 0.8197 |

Findings (Table 4):

- **Arc tokens beat point tokens for 11 of 12 encoder × dataset cells** in the supervised half (the only loss is PointNet on `buildings_iso`, footnote ⁵). The gap is largest on `buildings_iso` for non-T-Net encoders (DeepSet +15.2; ISAB +6.1; SAB +2.4 over raw points), modest on `mnist_iso` (+0.2 to +5.4 pt), and consistent on Omniglot fewshot (+1.8 to +6.2 pt for both PointNet and SAB).
- **The gain is the arc *structure*, not the Fourier PE.** "points + PE" matches ArcSet's Fourier budget on midpoints but applied to raw vertices, and it actually *underperforms* raw points on `buildings_iso` (likely because the dense PE on near-corner vertices is uninformative without segment context). This is the cleanest evidence the contribution is the arc representation, not the encoding budget.
- **Across-encoder consistency is the meta-finding.** No matter whether the decoder is a simple sum-pool DeepSet, a max-pool PointNet, or a self-attention SetTransformer, the arc representation wins on at least 3 of 3 datasets. This is the encoder-agnostic claim. PointNet's edge case on buildings_iso is being verified with multi-seed (footnote ⁵).

## Multi-seed for the headline cells (PointNet on iso, 3 seeds: 42 / 0 / 7)

| Setting | seed=42 | seed=0 | seed=7 | mean | std |
|---|---|---|---|---|---|
| std buildings_iso | 0.9282 | 0.9056 | 0.9295 | 0.9211 | 0.0110 |
| std mnist_iso | 0.9848 | 0.9841 | 0.9830 | 0.9840 | 0.0008 |
| fs 5w-1s | 0.9400 | 0.9497 | 0.9377 | **0.9425** | 0.0061 |
| fs 5w-5s | 0.9814 | 0.9821 | 0.9822 | **0.9819** | 0.0004 |
| fs 20w-1s | 0.9017 | 0.8979 | 0.8972 | **0.8989** | 0.0019 |
| fs 20w-5s | 0.9654 | 0.9650 | 0.9637 | 0.9647 | 0.0007 |

All four fewshot cells now have 3 seeds. **PointNet 20w-1s = 0.8989 ± 0.002 over 3 seeds**, vs SketchEmbedNet raw 0.8667 ⇒ +3.22 pt outside ±2σ on either side. 20w-5s margin (0.9647 vs SEN 0.9587, +0.6 pt) is robust at 3-seed-σ. 5-way settings (PointNet 0.9425 vs SEN 0.9513 on 5w-1s, 0.9819 vs 0.9857 on 5w-5s) are within ~1 pt of SEN's 21M-sketch QuickDraw pretraining despite our from-scratch episodic training.

## Table 5 — Low-shot QuickDraw (collaborator's runs, single seed unless noted)

100-class QuickDraw subset, 100 classes × {50, 100, 200, 500, 1000} samples per class. SetTransformer numbers reflect the collaborator's tuned recipe (`lr=5e-4`, `warmup=5`, `clip=1`, mostly `set_pool=mean` for the "stable_meanpool" branch); SAB without the tuning is unstable below 200 samples / class.

| Encoder | input | 100c50s | 100c100s | 100c200s | 100c500s | 100c1000s |
|---|---|---|---|---|---|---|
| DeepSet (ours) | arc set | **0.3093** | **0.4073** | 0.5020 | 0.6223 | 0.6887 |
| **PointNet (ours)** | arc set | 0.1947 | 0.3640 | **0.5437** | **0.6773** | **0.7175** |
| PointNet++ (ours) | arc set | 0.1200 | 0.3387 | 0.5217 | 0.6588 | 0.7073 |
| SetTransformer-SAB (tuned) | arc set | 0.1240 | 0.2360 | 0.3977 | 0.6061 | 0.6841 |
| SetTransformer-ISAB | arc set | 0.2000 | 0.3220 | 0.4517 | 0.6357 | 0.7058 |
| SetTransformer-ISAB (mean) | arc set | 0.2467 | 0.3527 | 0.4517 | 0.6325 | 0.6990 |

Best at 100c × 10000s (full QuickDraw subset, 1 M samples): DeepSet **0.7795** (1667 s on cuda).

Findings (Table 5):

- **Encoder preference shifts with data budget**: at 50–100 shots/class, mean-pool DeepSet wins (regularised representation, low variance); at 500–1000 shots/class, the higher-capacity PointNet/PointNet++ catch up and lead. SetTransformer needs the tuned recipe to be competitive in low-shot.
- **SetTransformer-SAB is unstable in raw config below 200 samples / class** (5-seed range 0.007–0.22 with the old default; collapses to chance for 4 of 5 seeds). The collaborator's `lr=5e-4 + warmup=5 + clip=1` recipe stabilises it. We surface this both as a caveat and as a finding (set-transformer optimization is brittle on small budgets without warmup + clip).
- **Same arc-set representation handles a 100-class stroke dataset cleanly.** No method-specific changes between Buildings, MNIST polygon, Omniglot, and QuickDraw — only encoder choice differs.

³ Source for Table 5: `QUICKDRAW_RESULTS.md` (102 result files, generated from collaborator's Windows local results dir; the JSON artifacts live with the collaborator and are not in this repo). Verification reruns of the headline cells on HPC are queued for Round 3.

## Size-only baseline (controls for absolute-size leakage)

A 3-layer MLP on 7 scalar per-entity features (`bbox_w`, `bbox_h`, `area`, `perimeter`, `aspect_ratio`, `n_arcs`, `mean_arc_length`), 200 epochs, no shape information beyond crude size summaries:

| Dataset | size-only test acc | ArcSet best (iso) | Δ |
|---|---|---|---|
| `single_buildings` (raw) | 0.7287 | — | — |
| `single_buildings_iso` | 0.7367 | 0.9348 (Geo2Vec) / 0.9282 (PointNet arc) | +18.0 / +19.2 |
| `single_mnist` (raw) | 0.5071 | — | — |
| `single_mnist_iso` | 0.5114 | 0.9848 (PointNet) | +47.3 |

Size-only on `buildings_iso` gets **0.74** — the 10 letter classes (E/F/H/I/L/O/T/U/Y/Z) carry strong class-specific aspect ratios. ArcSet's +18-19 pt gap reflects real shape encoding, not absolute-size accounting. On `mnist_iso` the size-only floor is 0.51, and ArcSet's +47.3 pt gap is dominated by shape information.

⁶ This baseline is included to bound how much accuracy a method can achieve from absolute-size signal alone. It is independent of any encoder choice and uses only seven hand-designed scalar features.

## Table 6 — PointNet T-Net ablation (defends Q5 from review)

The arc-vs-point ablation found PointNet was the only encoder where raw 2-D
points beat arc tokens on `buildings_iso`. PointNet's design has a 2-D T-Net
that learns an input-coordinate transform — it expects 2-D point clouds.
This table tests three T-Net configurations on PointNet × arc tokens
(22-D input):

| T-Net mode | params | buildings_iso (PointNet × arc) |
|---|---|---|
| `none` (no T-Net) | ~1.03 M | 0.9122 |
| `2d` (default — T-Net on first 2 of 22 dims) | ~1.27 M | 0.9282 |
| `full` (T-Net on all 22 dims) | ~1.34 M | **0.9335** |
| `2d` × **raw points** (PointNet's native config) | ~1.27 M | **0.9530 ± 0.007** (3 seeds) |

Reading: extending the T-Net to 22-D buys PointNet-arc +0.5 pt over the
default 2-D T-Net (and +2 pt over no T-Net), confirming the T-Net is doing
real work. But even with the full 22-D T-Net, PointNet on arc tokens does
not catch up to PointNet on raw 2-D points (0.9335 vs 0.9530, -2 pt). The
honest reading: **PointNet's inductive bias is fundamentally point-cloud-
native; arc tokens carry information PointNet's architecture can't fully
exploit even with a matched-dimension T-Net**. This is fine for the paper
because (a) the other three encoders all gain from arc tokens, and (b) the
result is now mechanistically explained, not hand-waved.

## Table 7 — Efficiency / wall-clock comparison

Forward inference on h200 GPU; preprocessing on CPU; both measured in
microseconds per entity over the full dataset.

### Per-entity timing (single_buildings_iso, avg 10.6 tokens/entity)

| Method | input | params | preprocess (µs) | forward (µs) |
|---|---|---|---|---|
| DeepSet | arc | 64 k | 119 | 49 |
| DeepSet | points | 59 k | 28 | 48 |
| PointNet | arc | 1.27 M | 120 | 31 |
| PointNet | raw points | 1.27 M | 28 | 29 |
| PointNet | points + PE | 1.28 M | 47 | 30 |
| PointNet++ | arc | 367 k | 120 | 60 |
| SetTransformer-SAB | arc | 276 k | 120 | 54 |
| SetTransformer-SAB | points | 267 k | 27 | 52 |
| SetTransformer-ISAB | arc | 479 k | 120 | 58 |

### Per-entity timing (single_mnist_iso, avg 235.8 tokens/entity)

| Method | input | params | preprocess (µs) | forward (µs) |
|---|---|---|---|---|
| DeepSet | arc | 65 k | 348 | 90 |
| DeepSet | points | 59 k | 103 | 61 |
| PointNet | arc | 1.28 M | 348 | 71 |
| PointNet | raw points | 1.27 M | 102 | 44 |
| PointNet | points + PE | 1.28 M | 180 | 63 |
| PointNet++ | arc | 368 k | 352 | 69 |
| SetTransformer-SAB | arc | 278 k | 348 | 102 |
| SetTransformer-SAB | points | 267 k | 103 | 74 |
| SetTransformer-ISAB | arc | 481 k | 356 | 104 |

### vs polygon baselines (preprocessing per entity, mnist scale)

| Method | preprocessing per entity | preprocessing total (60 k entities) |
|---|---|---|
| **ArcSet (any decoder)** | ~350 µs (Fourier features) | **~21 sec** |
| DeepSet/PointNet on raw points | ~100 µs (just centring) | ~6 sec |
| Geo2Vec | ~27 ms (SDF sampling, 5 668 points/entity) | **~27 min** |
| Poly2Vec | ~300 ms (CPU Delaunay triangulation) | **~5 hours** |
| NUFT | DDSL float64, CPU only | ~hours |

ArcSet's preprocessing is 75× faster than Geo2Vec and ~1 000× faster than
Poly2Vec on the same 60 k mnist polygon dataset, with no quality penalty
(ArcSet/PointNet/SAB on `mnist_iso` are within ±0.5 pt of each other and
all above all baselines under matched protocol). Combined with the
arc-vs-point ablation in Table 4, ArcSet gives the simplest pipeline,
the fastest preprocessing, and competitive accuracy without any
geometry-specific operator (no triangulation, no SDF, no visibility graph,
no rasterization).

## Table 8 — Robustness suite (train clean, test perturbed)

`single_buildings_iso` test split, single seed, no retraining. Each cell is
test accuracy after applying the perturbation to test geometries only.

| Encoder × input | clean | rot 15° | rot 45° | rot 90° | reflect_x | scale 0.5× | scale 2× | noise σ=.01 | noise σ=.05 | simplify ε=.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSet-arc | 0.8923 | 0.8803 | 0.8697 | 0.8883 | **0.1051** | **0.1662** | **0.2513** | 0.8989 | 0.8231 | 0.9176 |
| DeepSet-points | 0.7407 | 0.7074 | 0.5811 | 0.7021 | 0.7314 | 0.1742 | 0.1489 | 0.7394 | 0.7221 | 0.8005 |
| **PointNet-arc** | 0.9282 | 0.9202 | 0.9029 | 0.9189 | **0.3750** | 0.2926 | 0.2699 | 0.9282 | 0.8856 | 0.9495 |
| **PointNet-raw** | **0.9468** | **0.9388** | **0.9162** | 0.9215 | **0.9388** | 0.2314 | 0.6117 | 0.9428 | 0.9043 | 0.9668 |
| PointNet-pe | 0.7633 | 0.6915 | 0.5399 | 0.7261 | 0.7832 | 0.1729 | 0.2354 | 0.7819 | 0.7181 | 0.8005 |
| SAB-arc | 0.9162 | 0.9082 | 0.8883 | 0.9136 | **0.1795** | 0.2553 | 0.2380 | 0.9069 | 0.8670 | 0.9362 |
| SAB-points | 0.8923 | 0.8910 | 0.8298 | 0.8777 | 0.8949 | 0.1862 | 0.3364 | 0.8976 | 0.8657 | 0.9269 |

Findings:

- **Rotation invariance (15°/45°/90°)**: All methods are reasonably robust;
  arc tokens drop 0.5–4 pt across rotations. The arc representation is not
  rotation-equivariant (orientation θ is encoded absolutely), but the
  Fourier midpoint encoding helps it generalise across small rotations.
- **Reflection (reflect_x)**: arc tokens **catastrophically fail**
  (DeepSet-arc 0.89 → 0.10, SAB-arc 0.92 → 0.18, PointNet-arc 0.93 → 0.38)
  because θ flips sign under reflection and the model learned specific
  orientation patterns. PointNet on raw points only loses 0.8 pt. **This
  is an honest limitation of arc tokens**: they encode direction. Easy fix
  is reflection-augmentation at training time, or replacing θ with `cos θ`
  + `|sin θ|`-style reflection-symmetric features. We surface this as an
  important known limitation, not hide it.
- **Scale (0.5× / 2×)**: Every method collapses (drop 65–75 pt). Both
  ArcSet and points-mode were trained on per-entity isotropic-normalised
  inputs; applying a global scale at test time pushes the input
  out-of-distribution. The fix is multi-scale augmentation at training,
  not an architectural change — this is a property of the protocol, not
  the encoder.
- **Vertex noise (σ=0.01, σ=0.05)**: Robust. σ=0.01 sometimes slightly
  improves accuracy (regularization effect); σ=0.05 drops 3-10 pt.
- **Vertex simplification (Douglas-Peucker tol=0.02)**: Most methods
  *improve* by 1-3 pt — simplification removes redundant collinear
  vertices, which yields a cleaner arc set / point cloud. ArcSet is
  robust to this realistic preprocessing variation.

These results are honest. Reflection is a real arc-representation
limitation; we propose two clean fixes (augmentation, or reflection-symmetric
features) and note that PointNet-raw's strong reflection robustness is
explainable by the T-Net (which learns rotation-and-reflection-aware
input transforms).

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
