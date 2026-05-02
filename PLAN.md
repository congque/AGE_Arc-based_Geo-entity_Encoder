# ArcSet experiments — running plan

Last sync: 2026-04-30 evening (M5 Pro / `age` conda env / osx-arm64).

## ✅ Completed

### ArcSet 1.1 standard classification (full 80 epochs)

| set encoder | single_buildings | single_mnist |
|---|---|---|
| **SetTransformer-SAB** | **0.9827** | **0.9854** |
| SetTransformer-ISAB | 0.9668 | 0.9824 |
| DeepSet | 0.9322 | 0.9807 |

Smoke checks: `single_quickdraw` (1 epoch ISAB ⇒ 0.4681), `single_omniglot` few-shot smoke (DeepSet 5w-1s ⇒ 0.2734), `topo_polygongnn` siamese smoke (DeepSet 2 epoch ⇒ 0.0507).

### Polygon baselines on single_buildings (full 80 epoch)

| baseline | single_buildings | notes |
|---|---|---|
| **NUFT** (Mai 2023) | **0.8503** | DDSL float64 → CPU only |
| **PolygonGNN** (KDD 2024) | **0.973** | torch_sparse-free patch by Codex |
| **Poly2Vec** (ICML 2025) | **0.8085** | encoder CPU + MPS head |
| Geo2Vec (AAAI 2026 oral) | **0.9721** | CPU rerun with upstream defaults — matches paper claim ~0.97 |
| PolyMP / DSC | (queued) | — |

### Polygon baselines on single_mnist

| baseline | single_mnist |
|---|---|
| **NUFT** | **0.9663** |
| **PolygonGNN** | **0.907** |
| Poly2Vec | (queued — Wave B Poly2Vec died on MPS load) |
| Geo2Vec | (queued — depends on buildings rerun fix) |
| PolyMP | (queued) |

### Few-shot Omniglot
| setting | DeepSet | SAB | ISAB |
|---|---|---|---|
| **5-way 1-shot** | 0.2734 (chance 0.20) | (stuck at val 0.34, killed) | — |
| 5-way 5-shot | — | — | — |
| 20-way 1-shot | — | — | — |
| 20-way 5-shot | — | — | — |

⚠️ The current `testfs.py` plateaus at very low accuracy on Omniglot stroke
data. Train loss is stuck at chance value; only ~10% margin above chance.
This is an open issue — see "Open ideas" below.

---

## ⏳ Running now

| pid | task | device | progress |
|---|---|---|---|
| 79622 | Geo2Vec buildings rerun | CPU | SDF epoch ~33/80 (loss 12 → 11.7) |

Everything else stopped (Wave B Poly2Vec mnist hung at load; few-shot SAB
5w-1s was killed mid-run when the priority pipeline got SIGINT'd earlier).

---

## 📋 Queued tasks (in execution order)

Each block runs sequentially to avoid the MPS contention that killed earlier
parallel attempts.

### Block A — finish Geo2Vec buildings (CPU, in progress)
- Wait for current run to finish. Verify `test_acc` is sensibly above chance.

### Block B — Wave B remainder (mnist polygon baselines)

These all touch MPS. Run **one at a time** to avoid hangs.

1. Poly2Vec × single_mnist (80 epoch, encoder CPU + head MPS, ~30 min)
2. PolyMP `polymp` × single_buildings (80 epoch, MPS, ~10 min)
3. PolyMP `dsc_polymp` × single_buildings (~10 min)
4. PolyMP `polymp` × single_mnist (~30 min)
5. PolyMP `dsc_polymp` × single_mnist (~30 min)
6. Geo2Vec × single_mnist on **CPU** if buildings CPU rerun confirms MPS is the issue (~1 h)

Total Block B time estimate: ~3–4 h sequential.

### Block C — few-shot Omniglot (Prototypical Network)

12 runs total: 4 settings × 3 models. **Sequential**, MPS heavy.

- 5w-1s: SAB → ISAB → DeepSet (DeepSet was already 0.27)
- 5w-5s: SAB → ISAB → DeepSet
- 20w-1s: SAB → ISAB → DeepSet
- 20w-5s: SAB → ISAB → DeepSet

Each run ~30–60 min ⇒ Block C ~6–12 h. SAB at ~34% val on 5w-1s suggests we
can land at ~40–50% test on this hard config — reasonable for 1623-class
stroke data given the encoder size.

### Block D — stroke baselines on omniglot / quickdraw

Geo2Vec and Poly2Vec support polylines (NUFT and PolygonGNN are
polygon-only). **T2Vec and T-JEPA** are trajectory encoders — originally
GPS road-network designs but adaptable to (x, y) stroke sequences. Run on
the smaller dataset first:

1. Geo2Vec × single_omniglot (CPU SDF) — ~2–3 h
2. Poly2Vec × single_omniglot — ~1 h
3. **T2Vec** × single_omniglot — adapt repo `boathit/t2vec`
   (Julia + PyTorch 0.4) to read our gpkg as flat (x, y) sequence;
   skip cell-tokenization step (no road network).
4. **T-JEPA** × single_omniglot — adapt `arxiv 2406.12913` ref impl;
   masked context-target prediction in latent space, no reconstruction.
5. Geo2Vec / Poly2Vec / T2Vec / T-JEPA × single_quickdraw — likely too
   long locally; defer until HPC access is restored.

QuickDraw with 890 k entities × 80 epoch is many hours per baseline
locally. Defer the QuickDraw runs.

### Block E — PolyMP "robustness" matrix
Optional. Their paper reports rotation/shear-invariance ablations with
varying transform fractions (0, 0.2, …, 0.8). Skip unless reviewer asks.

---

## 💡 Open ideas to explore

### Few-shot debug — **RESOLVED**

Real bug: **single-seed PMA in SAB / ISAB became input-independent on long
arc sets** (Omniglot avg 168 arcs/entity). Codex probed embedding diversity:

- 200 random Omniglot embeds had mean pairwise L2 = **0.00181** (collapsed)
- per-dim std = **7.7e-05** (essentially zero)
- variance died after PMA: `0.0678 → 0.0242 → 0.00233 → 0.000205`

**Patch** (`entitysettransformer_sab.py` + `entitysettransformer_isab.py`):
add `set_pooling="mean"` option that bypasses PMA and does masked mean
pooling. **Default for SAB/ISAB few-shot is now `mean`.** PMA still
selectable via `--sab-pooling pma`.

**Verified**: 10-epoch SAB 5w-1s (mean pooling, freq=6, cosine+temp=10):
val **0.8485**, test **0.7688 ± 0.0226**. Trajectory matches the 0.8634
target. Will run full 80-epoch matrix next.

Other diagnostic findings:
- `xy_num_freqs=6` matters more than 9 for stroke (~+5 pts).
- Cosine + learnable temp helps once PMA collapse is fixed.

### Geo2Vec issue
- MPS NaN looks fundamental (autodecoder + log_sampling Fourier produces
  extreme values that overflow f16). On CPU it converges (current run loss
  12 → 11.7 over 30 epochs).
- For mnist (50k entities × CPU SDF) the run will be slow; budget ~2 h.

### Topology / pair-wise
- Deferred per user direction. PolygonGNN HF datasets are not strict DE-9IM;
  current `test_topo.py` siamese smoke gets 5% / 100 classes.
- If we revisit: synthesize a small DE-9IM dataset from buildings/mnist
  polygons (sample pairs, use shapely to label DC/EC/PO/EQ/TPP/NTPP/
  TPPi/NTPPi).

### Stroke-specific baselines
- Sketch-RNN / Sketchformer / Sketchformer++ / ViSketch-GPT / HiT-JEPA
  reported in lit-search. Add only if Block D lands and we want stronger
  competitors on stroke. **Lower priority** — ArcSet already differentiated
  on polygon datasets.

### Repro paper writing
- Once Block A–C finish, the paper has clean polygon SOTA on buildings
  (98.27% vs PolygonGNN 97.3%) and a competitive mnist number (98.54% vs
  NUFT 96.6%, PolygonGNN 90.7%).
- Topology is deferred. Few-shot is the next chapter — needs the SAB
  numbers from Block C.

---

## 🎯 Immediate next move

1. Wait for Geo2Vec CPU buildings rerun (Block A, ~15 min remaining).
2. Launch Block B sequentially (single bash driver, no MPS parallelism).
3. Launch Block C (few-shot) only after Block B completes — they all want
   the GPU.
4. Block D after Block C.

Each block has a dedicated driver under `scripts/` so you can resume from
any checkpoint by re-running the block driver.
