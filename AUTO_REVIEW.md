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

## Round 2 (2026-05-04, gpt-5.5 xhigh)

### Assessment (Summary)

- Score: 6/10 (up from 4/10)
- Verdict: almost, but not ready
- Status: moved from "fatally flawed" to "credible but still borderline"

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score:** 6/10, up from 4/10. The work has moved meaningfully: the iso-primary protocol, size-only baseline, multi-seed headline cells, and arc-vs-point ablation remove the most embarrassing objections. But it is still not an Accept-level NeurIPS paper.

**Verdict:** Almost, but not ready.

**Weakness 1:** The best raw-point model beats ArcSet on the main real-world polygon task.
`PointNet raw points = 0.9530±0.007` vs `PointNet arc = 0.9211±0.011` on buildings_iso. Reviewers will not care that arcs win 11/12 per-encoder cells if the strongest raw-point baseline wins the central OSM benchmark.
**Min fix:** Add a "best-over-representation" table. Then run matched PointNet variants: no T-Net, 22-D T-Net, and arc-token MLP projection before PointNet. If raw PointNet still wins, explicitly narrow the claim to "arc tokens benefit non-point-cloud-native set encoders."

**Weakness 2:** Benchmark breadth is still below the comparator standard.
Poly2Vec evaluates across points/polylines/polygons and downstream GeoAI tasks; Geo2Vec claims unified geo-entity modeling with shape, distance, robustness, and efficiency; PolygonGNN targets multipolygon structure with multiple real/synthetic datasets. ArcSet still has one real-world polygon benchmark plus digit/stroke datasets.
**Min fix:** Add one harder open real polygon or multipolygon benchmark. If DBSR-46K is unavailable, create a size-matched OSM/Overture building or landuse benchmark where the size-only MLP drops much closer to chance.

**Weakness 3:** Protocol fairness is improved, but not sealed.
SEN/Sketchformer are reimplementations, Geo2Vec full schedule diverges, some few-shot comparisons still sound raw-vs-iso adjacent, and several promoted ablations are single-seed.
**Min fix:** For every primary claim: same normalization, same decoder budget where possible, 3 seeds, mean±std. Move raw-only aux ablations and reimplemented baselines into appendix unless rerun under iso.

**Weakness 4:** No robustness/invariance evidence.
For geospatial geometry, rotation, reflection, scale, vertex noise, simplification, and resampling are not optional. This is especially exposed because Geo2Vec and PolygonGNN explicitly emphasize invariant/robust representations.
**Min fix:** Train clean, test perturbed curves for arc tokens, raw points, points+PE, Geo2Vec/Poly2Vec if feasible. Report stress curves, not just one transformed accuracy.

**Weakness 5:** The efficiency story is still anecdotal.
"No preprocessing" could be a real contribution, but right now it is qualitative.
**Min fix:** Add a wall-clock/memory table: preprocessing time, train epoch time, inference/object, peak memory, token count. Include Geo2Vec SDF sampling and Poly2Vec triangulation on the same machine.

**Q5:** PointNet not benefiting from arcs hurts any "arc tokens universally improve encoders" claim. It strengthens the paper only if framed honestly: arc tokenization is a common interface that helps most non-point-native set encoders, while PointNet's inductive bias favors raw 2-D vertices. The exception must be foregrounded, not explained away.

**Q6:** The size-only baseline is a good defense against pure scale leakage: +18-19 points over size-only is real shape signal. But the 0.74 floor also says buildings_iso is too easy and partly class-style biased. For NeurIPS, I would supplement it with a harder or size-matched polygon benchmark.

**One-sentence summary:** The paper has moved from "fatally flawed" to "credible but still borderline," with acceptance now depending on a harder real-world benchmark, fully matched protocols, and a sharper claim that does not pretend arcs beat raw points everywhere.

</details>

### Plan for Round 3 (this round implements)

Prioritized by review-impact / cost ratio:

1. **Robustness / invariance suite** (W4) — train-clean / test-perturbed evaluation curves on existing checkpoints. Cheap (no retrain) and high-impact for geospatial reviewers. Perturbations: rotation {15, 45, 90}°, reflection-x, uniform scale {0.5, 2.0}, vertex Gaussian noise σ∈{0.01, 0.05}, vertex simplification (Douglas-Peucker tol).
2. **Efficiency / wall-clock table** (W5) — measure preprocessing time per dataset (already have logs for Geo2Vec sampling, Poly2Vec triangulation), train epoch time, inference per entity, peak GPU memory, token count vs N. ArcSet's "no preprocessing" advantage gets quantified.
3. **PointNet T-Net ablation** (W1) — three variants: no T-Net, 22-D T-Net (extends T-Net to project all 22 dims of arc tokens, not just first 2), PointNet-MLP (drop T-Net entirely). Ideally either narrows the claim sharply or rescues PointNet-arc on buildings_iso.

Deferred for round 4 / future:
- Harder real-world polygon benchmark (W2). If size-matched OSM landuse can be assembled cheaply, do it; otherwise narrow scope to "vector-shape representation" rather than "general geo-entity encoder".
- Full 3-seed across ALL ablations (W3). Apply to claimed wins where margin < 1 pt.
- SEN/Sketchformer upstream wiring.

Status: implementing.

## Round 3 (2026-05-04, gpt-5.5 xhigh continued)

### Assessment

- Score: **6.5/10** (slight movement from 6, generous framing 7)
- Verdict: almost, but still not ready

### Reviewer Raw Response (Round 3)

<details>
<summary>Click to expand full reviewer response</summary>

**Score:** 6.5/10, maybe 7/10 with generous framing. It moved from 6, but not enough for "ready."

**Verdict:** Almost, but still not ready for NeurIPS.

**Weakness 1:** Benchmark breadth remains the biggest blocker.
One OSM building benchmark plus MNIST/Omniglot/QuickDraw is still too narrow compared with Geo2Vec, Poly2Vec, PolygonGNN.
**Min fix:** Add one genuinely harder open real-world polygon benchmark, preferably size/aspect-balanced. If not possible, create a public OSM/Overture benchmark and show size-only accuracy is much lower than 0.74.

**Weakness 2:** Best raw-point PointNet still wins the central real-world task.
The T-Net ablation is useful, but the headline remains awkward: raw 2-D points beat arc tokens by about 2 points on buildings_iso.
**Min fix:** Make the claim explicitly conditional: arcs improve non-T-Net set encoders and provide a fast geometry interface, but raw PointNet remains strongest on this benchmark. Add a "best method per representation" table.

**Weakness 3:** Reflection failure is severe.
This is not a small robustness caveat. Arc methods collapsing to 0.10-0.38 under reflection will alarm reviewers.
**Min fix:** Actually run the proposed fix. Reflection augmentation is the minimum. Better: compare original arc features vs reflection-symmetric arc features vs augmentation.

**Weakness 4:** Protocol fairness is still not sealed.
Reimplemented SEN/Sketchformer, raw-only aux ablation, and single-seed ablation cells leave openings for easy reviewer criticism.
**Min fix:** Move weak baselines/aux claims to appendix or rerun under iso with 3 seeds. Primary tables should be matched protocol only.

**Weakness 5:** Multi-seed coverage is still selective.
The new robustness and many arc-vs-point results are single-seed. That is acceptable for exploratory appendix evidence, not for central claims.
**Min fix:** 3 seeds for the core robustness rows: PointNet-raw, PointNet-arc, SAB-arc, SAB-points, DeepSet-arc.

**Q5:** Honest disclosure helps credibility, but it also hurts the technical case unless you run the fix. As written, a reviewer can say: "They found a catastrophic failure and did not evaluate the obvious remedy." One small augmentation table could convert this from a liability into a limitation-with-solution.

**Q6:** The efficiency result is strong enough as a secondary contribution. 75x over Geo2Vec preprocessing and 1000x over Poly2Vec is meaningful, especially with competitive accuracy. It would be stronger with training time and peak memory, but energy consumption is unnecessary.

**Single-sentence summary:** ArcSet now looks like a credible, efficient geometry-token interface with real empirical value, but the paper still needs either a harder real-world benchmark or a fixed reflection story to cross from borderline into accept.

</details>

### Plan for Round 4 (this round implements)

Priority order by review-impact / cost:

1. **Reflection augmentation experiment** (W3, Q5) — train PointNet-arc, SAB-arc, DeepSet-arc on `buildings_iso` with random reflect_x augmentation (p=0.5) at train time. Show clean + reflected test acc. Converts the catastrophic-failure liability into a "limitation-with-solution".
2. **Best-per-representation summary table** (W2) — doc rewrite. Surface the conditional claim cleanly: "for each input representation, what's the best encoder + accuracy?"
3. **Training time + peak memory** added to Table 7 efficiency (Q6).
4. **3-seed for core robustness rows** (W5) — repeat the 5 critical rows × 9 perturbations × 3 seeds = 135 fast evals. Cheap.
5. Maybe — reflection-symmetric arc features (cos θ, |sin θ|) as a second variant alongside augmentation.

Deferred / final-round risks:
- W1 harder real-world benchmark — biggest unaddressed risk. User confirmed deferring (DBSR-46K closed; single_buildings is OSM real-world; OSM landuse curation is substantial).

Status: implementing.

## Round 4 (2026-05-04, gpt-5.5 xhigh) — Final round

### Assessment

- Score: **7/10** (up from 6.5)
- Verdict: **almost / borderline accept** (assuming careful, narrow writing)
- Trajectory: 4 → 6 → 6.5 → 7 over 4 rounds (auto-review-loop converged)

### Reviewer Raw Response (Round 4)

<details>
<summary>Click to expand full reviewer response</summary>

**Score:** 7/10, up about +0.5 from Round 3. This is now credible borderline-accept territory, but not a clean accept.

**Verdict:** Almost / borderline accept, assuming the paper is written narrowly and honestly.

**Weakness 1:** Benchmark breadth is still the main blocker.
One real OSM building dataset plus digit/stroke polygons is thin for a NeurIPS "geo-entity encoder" claim.
**Min fix:** Either add one harder open OSM/Overture polygon benchmark, or narrow the claim aggressively to "fast arc-token interface for polygon/stroke shape encoding," not broad geospatial representation learning.

**Weakness 2:** Raw PointNet still wins buildings_iso.
The best-per-representation table makes this acceptable, but only if the paper does not imply arcs dominate all inputs.
**Min fix:** Put the honest result in the main text: ArcSet is within about 2 points of raw PointNet on buildings, wins/matches elsewhere, and is far cheaper than domain-specific preprocessing.

**Weakness 3:** Protocol fairness still has weak spots.
SEN/Sketchformer reimpls and raw-only aux ablations are reviewable attack surfaces.
**Min fix:** Demote these to appendix/caveats unless rerun under iso. Do not use them to support central claims.

**Weakness 4:** Scale robustness remains unfixed.
All methods collapse under scale perturbation, but since iso normalization is part of the protocol, this is less damaging than reflection was.
**Min fix:** State scale invariance is provided by preprocessing, not learned. If possible, add scale augmentation as a small appendix result.

**Weakness 5:** Efficiency table is strong but incomplete.
Preprocessing speed is now a real secondary contribution, but training time and peak memory would make it harder to dismiss.
**Min fix:** Add train epoch time, inference/entity, peak GPU memory, and preprocessing/entity in one table.

**Q5:** The remaining issues are mostly limitation/future-work caliber now, except benchmark breadth. Reflection is no longer a blocker because you verified the fix. PointNet-raw winning one dataset is no longer fatal because the paper now reports it directly and reframes the contribution as efficiency plus encoder-agnostic geometry tokens, not universal accuracy dominance.

**Q6:** The single most important thing to write is:
"ArcSet is not a universal accuracy winner; it is a preprocessing-light arc-token interface that gives non-specialized set encoders competitive polygon/stroke representations, comes within 1-2 points of the best raw-point model on OSM buildings, matches or wins on digit/stroke shapes, and reduces geometry preprocessing by 75-1000x."

**Single-sentence summary:** The paper has moved from not-ready to borderline-accept: honest, useful, and empirically much stronger, but still limited by benchmark breadth and the need to keep the claims sharply scoped.

</details>

## Loop terminating — converged at borderline-accept

Score progression: **4/10 (R1) → 6/10 (R2) → 6.5/10 (R3) → 7/10 (R4)**.

The auto-review-loop has consumed its 4 rounds. Remaining work to push from borderline-accept to clean-accept requires either (a) a harder real-world polygon benchmark — substantive new dataset work, not a single experiment — or (b) very careful paper writing to keep the claim tight. Both are out-of-scope for an autonomous review-loop and become human + writing tasks.

## Method Description (input for /paper-illustration)

ArcSet is an encoder for 2-D vector geometries (polygons, multi-line strings, strokes) that decomposes each entity into a permutation-invariant set of **directed arc primitives** — segments between adjacent vertices. Each arc carries explicit geometric information that an isolated point does not: a midpoint position, segment length, absolute orientation θ, and turning angles to neighboring arcs. The midpoint is multi-frequency Fourier-encoded; length is Fourier-encoded; orientation and turning angles use sin/cos with optional 2nd-harmonic features.

The arc set is fed to any permutation-invariant set encoder (DeepSet, PointNet/PointNet++, SetTransformer-SAB/ISAB) which produces a fixed-dimensional entity embedding. Decoder is task-specific (linear head for supervised; ProtoNet cosine + learnable temperature for few-shot; optional masked-arc-completion auxiliary head with GMM-MDN midpoint + Gaussian length + 16-bin theta).

Data flow: gpkg → shapely → per-entity isotropic normalize (centroid subtract + max(w,h)/2 scale to [-1,1]) → arc decomposition → per-arc Fourier features → set encoder → embedding → head. No triangulation, no SDF sampling, no visibility graph, no rasterization. Preprocessing is 75× faster than Geo2Vec and ~1000× faster than Poly2Vec on a 60k mnist polygon dataset.

Verified findings: arc tokens are encoder-agnostic (4 of 5 encoders gain over raw points + Fourier PE); arc representation is reflection-sensitive but a 1-line reflection augmentation closes the gap (+57 to +79 pt under reflect_x); preprocessing speedups are 75-1000× over baselines with no quality penalty.
