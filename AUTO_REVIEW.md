# ArcSet — Auto Review Loop

External reviewer: Codex `gpt-5.4` (model_reasoning_effort=xhigh).
Target venue: NeurIPS (solid acceptance, not borderline).
Loop started: 2026-05-04.

## Round 1 (2026-05-04)

### Assessment (Summary)

- Score: 4/10
- Verdict: not ready
- Key criticisms:
  - Unmatched evaluation protocol (raw vs iso, ArcSet ProtoNet cosine vs SEN LR-per-episode) — rejectable on fairness
  - Raw Table 1 win partly absolute-scale leakage (SAB 6.1 pt drop under iso on buildings)
  - Single seed throughout; +3.5 pt headline can't be defended without CI
  - Benchmark breadth (2 polygon toys + Omniglot) below NeurIPS bar
  - Thesis unstable: many encoder variants, no clean message about what is responsible

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 4/10

Verdict: not ready

Weakness 1: Unmatched evaluation protocol — fatal — The comparisons are not reviewer-clean. You mix raw global coordinates vs per-entity normalization, ProtoNet vs LR episodic heads, and incomplete `iso` reruns for key baselines. A skeptical reviewer can reject on fairness before engaging the method. Minimum fix: make a matched-protocol table the primary result. Re-run ArcSet and the main baselines under the same normalization and the same few-shot head, or explicitly narrow claims and remove any "SOTA" language where protocols differ.

Weakness 2: Headline polygon win is partly a scale-leakage win — fatal — Your own `raw -> iso` control materially weakens the original story: SAB drops 6.1 points on `buildings`, and the winner changes. That means raw Table 1 is not clean evidence of better shape encoding. Minimum fix: make `*_iso` the default polygon protocol, add a raw-vs-iso delta table for all methods, and include a trivial size-only baseline (`bbox w/h/area`, mean arc length) to show how much of raw performance is explainable by absolute scale alone.

Weakness 3: No statistical reliability — major — Single-seed, best-validation reporting is not enough for margins this small, especially in few-shot. The claimed `+3.5` on Omniglot `20w-1s` is not solid without variance. Minimum fix: run 3-5 seeds for the headline models and closest baselines, report mean ± std or 95% CI, and at minimum cover `buildings_iso`, `mnist_iso`, Omniglot `20w-1s`, and `20w-5s`.

Weakness 4: Benchmark breadth is below NeurIPS bar — major — Two toy polygon datasets plus Omniglot do not establish a general geo-entity encoder. The polygon side especially looks narrow and confounded, and QuickDraw few-shot is unfinished. Minimum fix: add one real-world vector-geometry benchmark on actual map entities or retrieval, plus one completed large-scale stroke result; otherwise stop selling this as a broad geo-entity encoder and scope it as a vector-shape representation paper.

Weakness 5: The thesis is unstable — major — The paper does not yet isolate what is actually responsible for gains. Under `iso`, PointNet/PointNet++ become strongest; aux helps attention encoders but hurts DeepSet; PMA collapses; ISAB has pathologies. Right now the contribution reads as "something in this stack helps sometimes," which is not a stable top-venue message. Minimum fix: re-center the claim on the arc representation, demote broken variants from the main story, and add one compact `iso` ablation suite over feature subsets, encoder family, and rotation/reflection/scale perturbations.

Strongest framing: Stronger scientifically, weaker rhetorically. The `iso` results remove the easiest dismissal of the paper, namely that ArcSet wins because the datasets leak absolute scale through global placement, and they expose the more defensible contribution: directed-arc tokenization is a useful common interface for heterogeneous vector geometries, with different set backbones exploiting it differently depending on whether absolute metric cues are present. That is a narrower claim than "ArcSet-SAB is universally best," but it is much more reviewer-proof because it isolates the representation's value instead of over-claiming architecture superiority.

Smallest delta vs Geo2Vec/Poly2Vec evaluation:
- Add one primary matched-protocol benchmark table: `raw` and `iso`, same preprocessing assumptions, same few-shot head where applicable, same reporting convention for all methods.
- Add one real-world polygon benchmark beyond letters/digits, ideally a genuine geo-shape classification or retrieval task.
- Add a standard robustness suite: rotation, reflection, uniform scale, mild shear, and vertex simplification/noise.
- Add multi-seed statistics for headline numbers and any claimed wins.
- Add efficiency/scalability measurements: preprocessing cost, train/inference time, and memory vs number of arcs.
- Add one clean representation ablation: midpoint only, `+length`, `+orientation`, `+relative angles`, `+endpoints`, reported under `iso`.

If I were calibrating this as a NeurIPS reviewer, the current manuscript is rejectable on protocol alone. The idea is promising, but the paper is not yet reviewer-solid.

</details>

### Plan for Round 2 (implementation in this round)

Cheap-and-high-impact fixes (will implement before re-review):

1. **Reframe RESULTS.md primary**: iso table primary, raw as supplementary delta. Reword "SOTA" claims to be protocol-specific.
2. **Size-only baseline**: features = (bbox w, h, area, n_arcs, mean_arc_len, perimeter) → MLP head; 80 epochs on each dataset + iso variant. Bounds how much absolute size explains.
3. **Multi-seed (3 seeds: 42, 0, 7)** for headline cells: PointNet on buildings_iso, mnist_iso, omniglot_iso × {5w-1s, 5w-5s, 20w-1s, 20w-5s}; SAB raw on omniglot 20w-1s + 20w-5s; SEN raw 20w-1s. ~24 jobs.
4. **Representation ablation**: SAB on buildings_iso, mnist_iso, omniglot_iso 5w-1s with feature sets {midpoint_only, +length, +orientation, +rel_angles, +endpoints}. ~15 jobs.
5. **Robustness eval**: rotate {15°, 45°, 90°}, reflect-x, uniform scale {0.5, 2.0}, shear, vertex-simplification at test time without retraining; report on PointNet + SAB iso checkpoints. ~30 evaluations (cheap).

Deferred (too expensive for this loop):

- Real-world polygon benchmark (curating + integration ≥ 1 week)
- Geo2Vec mnist_iso patch (modify upstream MP_sample to per-entity bbox)
- ArcSet QuickDraw 5-way fewshot (could do but low-priority for the headline)

Status: implementing.
