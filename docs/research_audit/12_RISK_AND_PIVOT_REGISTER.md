# 12 — Risk and Pivot Register

Every identified risk for the Factor-Gated Routing project, with early-warning signals, detection methods, mitigations, and pre-planned pivot actions. This register is living — update probabilities and statuses as evidence accumulates.

---

## Risk Register

### RISK-001: Nearest Literature Collision Invalidates Novelty

| Field | Detail |
|-------|--------|
| **Description** | A published or preprint paper implements the same core idea (factor-specific diffusion paths with graph-structured routing). Novelty claim collapses. |
| **Probability** | Medium |
| **Impact** | High |
| **Early signal** | Literature search finds title/abstract within Levenshtein distance < 3 of our core claim. |
| **Detection** | Continuous Semantic Scholar alert for: "factor" + "diffusion" + "routing" + "graph" + "path", "disentangled diffusion transformer", "causal diffusion generation". Check arXiv and OpenReview weekly. |
| **Mitigation** | Publish preprint early (arXiv) with timestamp. Differentiate via graph-surgical intervention + paired NoiseTrace evaluation + Path Non-Interference Theorem (three-pronged specificity). |
| **Pivot action** | If collision is exact: differentiate on evaluation protocol (paired leakage matrix), graph misspecification experiments (our unique contribution), or theory tightness. If entirely pre-empted: pivot to Pivot A (auditability) or Pivot C (identifiable decomposition). |
| **Kill criterion** | Published identical architecture + identical evaluation + same datasets. |

### RISK-002: Shared x_t Leakage Makes Path Isolation Semantically Meaningless

| Field | Detail |
|-------|--------|
| **Description** | Even with output gate=0, shared x_t provides a bypass: other streams can infer the cut factor's value from x_t structure. Path Non-Interference Theorem holds computationally, but the *semantic* isolation claim ("factor i is removed from the image") is false. |
| **Probability** | High |
| **Impact** | High |
| **Early signal** | Full source cut invariance test (E4-05) passes mathematically, but off-target leakage (E4-01) shows no FGR advantage over DiT. |
| **Detection** | Compare leakage matrix at different noise levels (t): early denoising steps show massive x_t leakage, late steps show path structure. If leakage doesn't decrease with path cut, x_t dominates. |
| **Mitigation** | Frame Path Non-Interference Theorem as *computational-path* isolation, NOT semantic disentanglement. Acknowledge x_t as the inherent leakage channel in all denoisers. |
| **Pivot action** | Narrow claim scope: "computational path isolation" rather than "factor isolation." Add per-timestep leakage analysis to show when and how path structure matters. |
| **Kill criterion** | Off-target leakage identical between FGR (with path cut) and DiT (monolithic) — path structure provides zero empirical benefit. |

### RISK-003: Branch Decomposition Non-Identifiability (Gauge Freedom)

| Field | Detail |
|-------|--------|
| **Description** | The additive aggregation ε̂ = ε_base + Σ_i ε_i has a gauge freedom: any δ_i satisfying Σ_i δ_i = 0 can be shifted between branches without changing the total output. Branch semantics are not uniquely identified by reconstruction loss alone. |
| **Probability** | High |
| **Impact** | Medium |
| **Early signal** | Factor-specific projections P_i show no specialization — different branches produce visually similar patterns regardless of which factor they receive. |
| **Detection** | Compute cross-branch cosine similarity of ε_i outputs. If cos(ε_i, ε_j) → 1 for i≠j, branches are interchangeable. Measure factor-editing specificity per branch under gate sweep. |
| **Mitigation** | Add centering regularization (E_{F_i}[ε_i] ≈ 0). Use factor-specific initial embeddings that constrain the input manifold. Report identifiability diagnostics in paper as a limitation. |
| **Pivot action** | If branches are non-identifiable: pivot to Pivot C (identifiable score decomposition). Develop ANOVA/hierarchical-centering theory proving identifiability under centering + orthogonality constraints. |
| **Kill criterion** | No branch shows factor-specificity after centering regularization + input embedding constraints. Architecture claims unsupported. |

### RISK-004: Base Stream Collapses Factor Streams

| Field | Detail |
|-------|--------|
| **Description** | The base stream (nuisance predictor) learns to generate all image content, while factor streams contribute near-zero. Factor editing produces no measurable change. |
| **Probability** | Medium |
| **Impact** | High |
| **Early signal** | ||ε_base|| / ||ε_total|| > 0.9 in early training. Factor stream norms monotonically decreasing. |
| **Detection** | Monitor ||ε_i|| / ||ε_total|| for each stream during training. Gate sweep should show factor-stream contribution near zero at all gate values. |
| **Mitigation** | Capacity-limit the base stream (fewer parameters, smaller hidden dim). Add factor-stream norm preservation loss. Initialize base stream output smaller. |
| **Pivot action** | Remove base stream; use only factor streams + explicit nuisance-factor streams (treat position as a factor). If that still collapses: pivot to Pivot A — focus on auditability of factor paths even if base stream dominates. |
| **Kill criterion** | After capacity limiting + norm preservation, base stream still contributes > 90% at convergence. Factor paths are vestigial. |

### RISK-005: Gate Training Causes Branch Compensation/Leakage

| Field | Detail |
|-------|--------|
| **Description** | Training with stochastic gates leads branches to compensate: when stream i is gated off, other streams learn to encode factor i's information. Gate=0 no longer isolates factor i semantically. |
| **Probability** | Medium |
| **Impact** | Medium |
| **Early signal** | Leakage matrix off-diagonal entries INCREASE with gate training vs inference-only gating. |
| **Detection** | Compare E4-06: gate-trained model leakage matrix vs inference-only-gated model leakage matrix. If off-diagonal entries are higher in gate-trained model, compensation is occurring. |
| **Mitigation** | Use path-invariance loss: L = ||ε(f_i; output_gate[i]=0) - ε(f_i'; output_gate[i]=0)||². Explicitly penalize branches for encoding information about factors whose gate is zero. |
| **Pivot action** | Drop gate training. Use only inference-time gating (constant gate=1 during training, sweep gate during evaluation). Add gate-training as a negative result in paper. |
| **Kill criterion** | Path-invariance loss prevents convergence or doesn't reduce compensation. Gate training harmful in all regimes. |

### RISK-006: Wrong DAG Shows No Different from Correct DAG

| Field | Detail |
|-------|--------|
| **Description** | On Causal3DIdent, empty/complete/reversed/random DAG produces statistically indistinguishable leakage matrix from the correct SCM-derived DAG. Graph alignment doesn't matter empirically. |
| **Probability** | Medium |
| **Impact** | High |
| **Early signal** | E5-04: correct DAG not ranked best. Overlapping CIs across all graph variants. |
| **Detection** | ANOVA over graph variants × factor pairs. If graph variant explains < 1% of variance in leakage, graph structure is irrelevant. |
| **Mitigation** | Verify Causal3DIdent edges really encode detectable dependencies (check pairwise mutual information between factor values). If factors are near-independent, graph structure cannot matter regardless of architecture. |
| **Pivot action** | Drop graph-structured contribution. Pivot to Pivot A (auditability): emphasize path non-interference + leakage audit, not graph correctness. |
| **Kill criterion** | No graph variant outperforms empty graph. Graph edges never contribute positively. |

### RISK-007: Graph Edges Never Actually Used (Attention Weights Uniform/Zero)

| Field | Detail |
|-------|--------|
| **Description** | Cross-stream attention edges are implemented but the model learns to ignore them — attention weights are uniform or all near-zero. The graph structure is decorative, not functional. |
| **Probability** | Medium |
| **Impact** | Medium |
| **Early signal** | Attention weight entropy → log(K) (maximum, uniform) or attention weights → 0 (residual dominates). |
| **Detection** | Extract per-layer per-edge attention weight statistics during evaluation. Compute distribution of weights, entropy, and sparsity. If weight magnitude < 0.01 for all edges at convergence, edges are dead. |
| **Mitigation** | Initialize edge attention with positive bias. Add edge utilization loss: L_util = -Σ log(w_{j→i} + ε). |
| **Pivot action** | If edges are unused after utilization loss: simplify architecture to IndependentStreamDiT (no edges) and report graph structure as negative ablation result. |
| **Kill criterion** | Edge utilization loss prevents training or edges still near-zero after 5K steps with explicit incentives. |

### RISK-008: O(K) Cost Makes Scaling Infeasible

| Field | Detail |
|-------|--------|
| **Description** | Per-stream DiT blocks mean parameter count and FLOPs scale linearly with K. At K=6 (3DShapes), training is 3-4× slower than canonical DiT. At K=50 (real-world factors), infeasible on RTX 5080 16GB. |
| **Probability** | High |
| **Impact** | Medium |
| **Early signal** | Memory OOM at K=6 with reasonable batch size. Single epoch > 10× wall time of DiT. |
| **Detection** | Profile per-stream forward/backward time and memory. Compare total training time vs DiT at matched parameter count (adjust DiT width to match). |
| **Mitigation** | Parameter-share stream blocks (shared weights across streams). Factor-specific conditioning via FiLM/adapter layers rather than full separate blocks. Trade some isolation for scalability. |
| **Pivot action** | Acknowledge O(K) as intrinsic limitation. Frame contribution as: "we show proof-of-concept at K≤6; scaling is orthogonal engineering problem." Compare FLOPs-matched rather than parameter-matched. If RTX 5080 can't handle K=6: use K=4 (dSprites K=3) only and state scaling limitation clearly. |
| **Kill criterion** | Cannot run main experiments at target K within RTX 5080 16GB. |

### RISK-009: Oracle Bias Inflates/Deflates Leakage Measurements

| Field | Detail |
|-------|--------|
| **Description** | The oracle classifier has its own errors (misclassification, calibration drift on generated images). Leakage matrix entries conflate true factor change with oracle noise. |
| **Probability** | High |
| **Impact** | Medium |
| **Early signal** | Oracle accuracy on ground-truth < 95% per factor. Oracle accuracy on generated images significantly lower than on ground-truth. |
| **Detection** | Report oracle confusion matrix and calibration per factor. Measure oracle accuracy ON GENERATED IMAGES (not just ground-truth). If generated accuracy < 90%, leakage measurements are oracle-noise-dominated. |
| **Mitigation** | Only use factors where oracle accuracy > 95% on both ground-truth and generated images. Report per-factor oracle accuracy alongside leakage. Add oracle ensemble (multiple architectures) for agreement check. |
| **Pivot action** | If oracle accuracy insufficient: use simpler factors (binary shape, coarse scale bins), train better oracle (ConvNeXt-Base), or use ground-truth latent values (available in Causal3DIdent SCM) instead of oracle. |
| **Kill criterion** | Oracle accuracy < 80% on generated images for ≥2 factors. Leakage matrix uninterpretable. |

### RISK-010: Statistical Power Insufficient at Pilot Scale

| Field | Detail |
|-------|--------|
| **Description** | 2 seeds × 1000 samples at pilot stage cannot detect small-to-medium effect sizes. Pilot results inconclusive, confirmatory decision ambiguous. |
| **Probability** | Medium |
| **Impact** | Medium |
| **Early signal** | CIs overlap for all model comparisons. p-values 0.05-0.20 range. |
| **Detection** | Power analysis: given observed effect size and variance at pilot, how many seeds/samples needed for 80% power at α=0.05? |
| **Mitigation** | Use 3 seeds for pilot if budget allows. Pre-register minimum detectable effect size (Cohen's d > 0.3). Accept that pilot is directional only — gate to confirmatory is "direction matches hypothesis" not "p < 0.05." |
| **Pivot action** | Run confirmatory with more seeds (8+) if pilot is directionally promising. If inconclusive after confirmatory: report as negative result with full power analysis. |
| **Kill criterion** | Confirmatory with 5 seeds × 1000 samples still yields wide CIs. Effect too small to measure at feasible scale. |

### RISK-011: adaLN-Zero Implementation Reveals Existing DiTBlock Bugs

| Field | Detail |
|-------|--------|
| **Description** | Fixing DiTBlock to use adaLN-Zero (residual gating α initialized to 0) reveals that existing DiTBlock implementation has bugs. All baselines and FGR need fixing. Timeline explodes. |
| **Probability** | Medium |
| **Impact** | Low |
| **Early signal** | adaLN-Zero fix causes NaN or loss plateau in any model. |
| **Detection** | Test adaLN-Zero on canonical DiT first. If DiT doesn't train with adaLN-Zero, existing code is the problem. |
| **Mitigation** | Implement adaLN-Zero as a configurable option (not default). Test on canonical DiT first, then apply to FGR. Keep both code paths (old and new) for debugging. |
| **Pivot action** | If adaLN-Zero breaks everything: postpone to WP-11, use current DiTBlock as-is, note in paper limitations. adaLN-Zero is a training stabilization technique, not a core contribution. |
| **Kill criterion** | N/A — this is a fixable engineering issue, not a fundamental risk. |

### RISK-012: Paired Evaluation Shows Zero Benefit Over Unpaired

| Field | Detail |
|-------|--------|
| **Description** | Paired NoiseTrace evaluation (our methodological contribution) produces same CIs and conclusions as independent-sample evaluation. Our evaluation innovation is unnecessary complexity. |
| **Probability** | Low |
| **Impact** | Medium |
| **Early signal** | E6-08: CI widths identical between paired and unpaired. NC2 (different NoiseTrace) produces same metric distribution as independent samples. |
| **Detection** | Compare CI widths, minimum detectable effect sizes, and conclusion agreement between paired and unpaired protocols. |
| **Mitigation** | Frame paired evaluation as correctness guarantee, not statistical-power innovation: "paired ensures identical exogenous randomness, which is required for valid counterfactual comparison per Pearl's definition." Even if CIs are similar, paired is methodologically correct. |
| **Pivot action** | Report equivalence of paired and unpaired as a finding. Keep paired for correctness but don't claim statistical-power benefit. |
| **Kill criterion** | N/A — paired is always correct for counterfactual comparison. |

### RISK-013: Base Stream Absorbs All Semantic Information

| Field | Detail |
|-------|--------|
| **Description** | Stronger version of RISK-004. Not only does the base stream dominate, but it learns all factor-specific information. Factor streams contribute pure noise. Editing has zero effect. |
| **Probability** | Low |
| **Impact** | High |
| **Early signal** | Factor stream output norms → 0. Factor_edit target accuracy = chance. Leakage matrix = random (no structure). |
| **Detection** | Ablation: remove factor streams entirely → generated images unchanged (within LPIPS < 0.01). |
| **Mitigation** | Add explicit factor-stream output norm regularization. Initialize factor streams with non-zero contribution bias. Remove base stream entirely for first training phase. |
| **Pivot action** | Pivot to IndependentStreamDiT (no base stream). If that also fails: fundamental architectural insight is wrong — shared x_t provides all needed information about all factors. |
| **Kill criterion** | Factor streams contribute < 1% of output after all mitigations. |

### RISK-014: dSprites/3DShapes Too Simple — All Models Saturate

| Field | Detail |
|-------|--------|
| **Description** | dSprites and 3DShapes are so simple that all models (FGR, DiT, baselines) achieve near-perfect conditional generation and factor editing. No signal to differentiate architectures. |
| **Probability** | Medium |
| **Impact** | Medium |
| **Early signal** | Target accuracy > 98% for ALL models on dSprites factor_edit. Off-target leakage < 2% for ALL models. |
| **Detection** | Check for ceiling effects: is any metric within 1% of maximum for all models? |
| **Mitigation** | Use harder split (S3 systematic compositional) to create difficulty. Reduce training data to create scarcity. Use more challenging metrics (LPIPS distance rather than accuracy). Add noise/distortion to test images. |
| **Pivot action** | Add a harder dataset: Falcor3D, MPI3D, or a custom synthetic with more factors and correlations. Frame saturation as "all models learn simple factors; differences emerge on compositional generalization." |
| **Kill criterion** | Cannot create difficulty gap with available datasets and splits. |

### RISK-015: Causal3DIdent Has No Real Factor Graph Structure

| Field | Detail |
|-------|--------|
| **Description** | Causal3DIdent's SCM may have such weak causal edges (near-zero structural coefficients) that factor values are effectively independent. Graph misspecification experiments are meaningless because correct=empty=random. |
| **Probability** | Medium |
| **Impact** | High |
| **Early signal** | Pairwise mutual information between Causal3DIdent factors near zero. Causal effect estimates from SCM near zero for all edges. |
| **Detection** | Compute MI(x_i, x_j) for all factor pairs from the SCM. Fit a linear model predicting child from parent — if R² < 0.1 for all edges, causal structure is weak. |
| **Mitigation** | Self-generate a Causal3DIdent variant with stronger causal edges (larger structural coefficients). Or use a graph-structured synthetic dataset with known strong edges. |
| **Pivot action** | Drop Causal3DIdent. Construct synthetic dataset with known, strong, measurable factor dependencies. Or pivot to Pivot B (IIT alignment): align internal representations to SCM rather than just input graph. |
| **Kill criterion** | Strongest causal edge in Causal3DIdent has effect size (Cohen's d) < 0.1. Graph irrelevancy is baked into the benchmark. |

### RISK-016: Reviewer Rejects Causal Terminology Even After Pivot

| Field | Detail |
|-------|--------|
| **Description** | Despite our strict terminology policy (no do-operator, no causal claims), reviewers still read "graph surgery" and "factor intervention" as causal claims and reject on grounds of insufficient causal justification. |
| **Probability** | Medium |
| **Impact** | Medium |
| **Early signal** | Internal review by colleague unfamiliar with project: "isn't this just a causal model?" |
| **Detection** | Pre-submission: ask 3 colleagues to read abstract and identify any causal claims. If > 0 flag "graph surgery" as causal, terminology is still leaking. |
| **Mitigation** | Replace "graph surgery" → "edge-aware routing." Replace "intervention" → "routing mode." Add explicit terminology section in paper intro disclaiming causal interpretation. |
| **Pivot action** | Full terminology reset: all language replaced with computational/architectural terms. "Stream routing" instead of "intervention." "Path cut" instead of "ablation." "Edge mask" instead of "graph surgery." |
| **Kill criterion** | N/A — terminology is a presentational fix, not a fundamental issue. |

### RISK-017: Path Non-Interference Theorem Seen as Trivial by Reviewers

| Field | Detail |
|-------|--------|
| **Description** | The Path Non-Interference Theorem states "if you multiply by zero, the gradient is zero." Reviewers dismiss as obvious architectural tautology, not a theoretical contribution. |
| **Probability** | Medium |
| **Impact** | Low |
| **Early signal** | Colleague feedback: "isn't this just chain rule?" |
| **Detection** | Test: can a first-year PhD student derive the theorem in < 5 minutes? If yes, it's trivial. |
| **Mitigation** | Frame as architectural guarantee, not mathematical profundity. "Unlike prior work where factor isolation is aspirational, FGR provides a formal computational non-interference guarantee." Emphasize the trajectory-level corollary (induction over timesteps) as the non-trivial part. |
| **Pivot action** | Move theorem to architecture section (as property, not theory). Replace theory section with identifiability analysis (branch decomposition uniqueness) or gate sensitivity bound (Lipschitz). |
| **Kill criterion** | N/A — theorem remains correct and useful even if elementary. |

### RISK-018: Implementation Complexity Blows Up Timeline

| Field | Detail |
|-------|--------|
| **Description** | The 50+ tasks in WP-00 through WP-18 take significantly longer than estimated. Synchronous routing, intervention API, graph validation, per-stream projections, base stream, NoiseTrace, paired evaluation, 5 baselines, oracle training, statistics — total effort exceeds available time. |
| **Probability** | High |
| **Impact** | Medium |
| **Early signal** | WP-00 takes > 1 week. WP-03 takes > 2 weeks. |
| **Detection** | Track actual-vs-estimated per WP. If cumulative burn rate > 1.5× estimate by WP-05, trigger rescoping. |
| **Mitigation** | Prioritize ruthlessly: P0 tasks only. Cut baselines to 2 (canonical DiT + IndependentStreamDiT). Cut Causal3DIdent from initial scope. Use deterministic DDIM only (drop DDPM + trace for now). Automate evaluation scripts early. |
| **Pivot action** | Descope to minimum viable paper: dSprites only, 2 baselines, 3 seeds, factor_edit protocol only. Submit to workshop. Expand to full paper with remaining experiments in revision cycle. |
| **Kill criterion** | P0 tasks not complete after 6 weeks full-time effort. |

### RISK-019: Correct Graph Routing Requires Known Graph — Not Available for Real Datasets

| Field | Detail |
|-------|--------|
| **Description** | FGR's graph-surgical intervention requires knowing which factors causally influence which others. For real-world images (CelebA, ImageNet), no such graph exists. Contribution is locked to synthetic benchmarks with known SCMs. |
| **Probability** | High (certain) |
| **Impact** | Medium |
| **Early signal** | Every attempt to define factor graph for a real dataset (e.g., "smile → mouth_open" on CelebA) runs into ambiguity about direction, completeness, and confounding. |
| **Detection** | Attempt to specify factor graph for 1 real dataset. Survey 3 colleagues for agreement on edges. If agreement < 70%, graph is subjective. |
| **Mitigation** | Acknowledge synthetic-only as limitation. Propose graph-learning extension (learn edges from data) as future work. Emphasize that evaluation on known-SCM benchmarks is rigorous because graph is given, not assumed. |
| **Pivot action** | Frame contribution as "method for benchmarking and auditing factor-level generation with known structure." Target: methodology paper ("how to evaluate factor editing properly") rather than applied paper ("editing faces with graphs"). |
| **Kill criterion** | Venue requires real-dataset results. |

### RISK-020: Top-Tier Venue Requires ImageNet-Scale Results

| Field | Detail |
|-------|--------|
| **Description** | Top-tier venues (NeurIPS, ICML, ICLR) increasingly expect large-scale experiments. dSprites (737K images, 64×64) and 3DShapes (480K images, 64×64) are considered toy benchmarks. |
| **Probability** | Medium |
| **Impact** | Medium |
| **Early signal** | Recent accepted papers at target venue: all have ≥ 1 ImageNet-scale or real-dataset experiment. |
| **Detection** | Survey last 2 years of target venue. Count papers using only synthetic benchmarks. If < 10%, adjust venue target. |
| **Mitigation** | Add a proof-of-concept on a real dataset (CelebA with 5-10 attributes as factors, or ImageNet subcategory editing). Frame synthetic benchmarks as "controlled evaluation with known ground truth — essential for measuring leakage." |
| **Pivot action** | Target SCI Q1 journal instead of top-tier ML conference. Or add lightweight ImageNet experiment: pre-trained VAE + DiT, factor streams on class labels, minimal training. |
| **Kill criterion** | Venue desk-rejects based on benchmark scale. |

### RISK-021: Gradient Flow Through Gate Variables Is Zero During Training

| Field | Detail |
|-------|--------|
| **Description** | If gates are applied as post-hoc multipliers (output * gate), but gate variables have no gradient path to the loss, gate training via path-invariance loss cannot work. |
| **Probability** | Low |
| **Impact** | Medium |
| **Early signal** | Gate parameters have zero gradient norm throughout training. |
| **Detection** | Monitor grad_norm of gate parameters. If zero after first backward pass, check computational graph connection. |
| **Mitigation** | Gates must be differentiable (sigmoid output of learnable parameters) and the loss must explicitly depend on the gates. Use gated ε̂ = Σ_i gate_i · ε_i in training and pass gradient through. |
| **Pivot action** | If gate gradient is structurally zero: make gates multiplicative directly in the forward path (not post-hoc). Or drop gate training entirely. |
| **Kill criterion** | Gates cannot receive gradient by architectural design. |

### RISK-022: Deterministic DDIM η=0 Image Quality Is Poor

| Field | Detail |
|-------|--------|
| **Description** | DDIM η=0 (deterministic sampling) produces lower-quality images than stochastic DDPM. Evaluation metrics (FID, oracle accuracy) are worse than they would be with stochastic sampling, weakening results. |
| **Probability** | Medium |
| **Impact** | Low |
| **Early signal** | DDIM-generated images show visible artifacts (grid patterns, blur, color shift). FID significantly worse than DDPM-generated. |
| **Detection** | Compare DDIM η=0 vs DDPM (50 vs 1000 steps) on image quality metrics. If DDIM FID > DDPM FID + 20, DDIM quality is a problem. |
| **Mitigation** | Use more DDIM steps (100-250). Train with DDIM-compatible objectives. Or use DDPM with shared counter-seeds (NoiseTrace-B) for paired evaluation — same logical guarantee, better quality. |
| **Pivot action** | Switch to NoiseTrace-B (DDPM with counter seeds) as primary evaluation sampler. DDIM remains for quick iteration. |
| **Kill criterion** | DDPM with counter seeds also produces poor quality. |

### RISK-023: Position x/y as Nuisance Cannot Be Learnt by Base Stream

| Field | Detail |
|-------|--------|
| **Description** | Position x/y in dSprites are not well-captured by the base stream (capacity too low, or x_t already encodes position implicitly). Base stream + factor streams together produce blurry or misplaced objects. |
| **Probability** | Low |
| **Impact** | Medium |
| **Early signal** | Generated shapes appear at wrong positions. Position oracle accuracy low on generated images. |
| **Detection** | Train a position-only oracle and test on generated images. If position accuracy < 80%, position is not being generated correctly. |
| **Mitigation** | Add position as explicit factor streams (streams 3 and 4). Report both configurations (with and without position streams) as an ablation. |
| **Pivot action** | Treat position as factors. Acknowledge that "nuisance" and "factor" distinction is dataset-dependent. Base stream becomes purely x_t residual prediction. |
| **Kill criterion** | Even with position as factor streams, position accuracy < 80%. |

---

## Pivot Descriptions

### Pivot A: Mechanistic Auditability Paper

**Trigger**: RISK-006 confirmed (graph doesn't matter), RISK-002 confirmed (shared x_t dominates), or RISK-007 confirmed (edges never used).

**Core contribution shift**:

| From | To |
|------|-----|
| Graph-structured factor routing improves editing | Factor path decomposition enables auditing of what each factor contributes |
| Correct > wrong graph | Any factor path > monolithic (for interpretability, not performance) |
| Graph-surgical intervention | Path non-interference + leakage audit |

**Paper narrative**: "Current diffusion models are black boxes: it's unclear how latent factors contribute to generation. We introduce Factor-Path Diffusion, an architecture that factorizes the denoiser into explicit per-factor computational paths with provable non-interference. While this does not improve generation quality over monolithic models, it enables (1) precise measurement of factor-level contributions via per-path norm analysis, (2) detection of unwanted factor leakage through the shared input tensor, and (3) verification that architectural claims about factor isolation are empirically grounded. Our leakage audit protocol reveals that even with provable path non-interference, the shared x_t channel carries significant factor information — a finding with implications for all interpretable generation research."

**Key changes**:
- Remove graph edges entirely (IndependentStreamDiT is sufficient).
- Remove graph misspecification experiments.
- Add: per-timestep contribution analysis, per-stream visualization, factor-deletion effect quantification.
- Add: "surprising finding" — shared x_t leaks factor information even with path cut.
- Primary metric becomes interpretability/auditability, not editing performance.
- Target venue: interpretability or XAI track, not core ML.

**Evidence needed**:
- Leakage audit pipeline (already defined in Protocol 2).
- Per-stream contribution analysis (norm curves, visual patterns).
- x_t leakage quantification (cut a path → measure residual factor information in output).

**Risk**: May be seen as "negative result + analysis" — hard to place. Need compelling insight about x_t leakage to carry the paper.

---

### Pivot B: Graph-Surgical Causal Abstraction Paper

**Trigger**: RISK-015 confirmed (Causal3DIdent causal structure is weak/genuine) but we believe causal alignment is the right contribution. OR reviewer feedback demands causal justification.

**Core contribution shift**:

| From | To |
|------|-----|
| Path non-interference in conditional generation | Neural-SCM alignment via interventional training |
| Factor editing as evaluation | Causal abstraction (IIT) as evaluation |
| 6 intervention modes for architecture | 6 intervention modes for SCM alignment testing |

**Paper narrative**: "We propose Graph-Surgical Diffusion (GSD), a framework for training diffusion models whose internal factor representations respect a known causal graph. Our key insight: by constraining the denoiser's computational graph to match the factor-level SCM and training with interventional data, we achieve a form of mechanistic interpretability — the model's internal interventions align with SCM-level interventions. We evaluate alignment using interchange intervention training (IIT) adapted to diffusion: intervening on a factor embedding should produce the same effect as intervening on the SCM's structural equation."

**Key changes**:
- Add IIT training objective: alignment loss between internal intervention effect and SCM intervention effect.
- Add interventional data generation: train with (x, condition, intervention_spec) tuples where intervention_spec specifies which SCM equations are modified.
- Replace factor_edit evaluation with IIT alignment metrics (interchange accuracy, intervention agreement).
- Add formal definition: "a diffusion model (I,I')-aligns with an SCM if internal intervention on node i under input I produces outputs matching SCM intervention under input I'."
- Keep graph-surgical intervention modes as architectural support for IIT.

**Evidence needed**:
- IIT alignment scores across nodes.
- Causal abstraction diagrams showing internal→SCM mapping.
- Comparison with non-graph baselines on IIT metrics.
- Known-SCM dataset with testable causal structure.

**Risk**: Requires implementing IIT training loop, which is complex. Literature on causal abstraction in diffusion is thin — novelty is high but acceptance risk is also high. Need a strong known-SCM benchmark.

---

### Pivot C: Identifiable Score Decomposition Paper

**Trigger**: RISK-003 confirmed (branch non-identifiability is severe and cannot be papered over). OR we want to make the theoretical contribution stronger than "chain rule applied to gating."

**Core contribution shift**:

| From | To |
|------|-----|
| Architecture paper with empirical evaluation | Theory paper with identifiability guarantees |
| Factor paths as architectural choice | Factor paths as the unique solution under constraints |
| Leakage matrix as evaluation | Branch decomposition uniqueness as theoretical contribution |
| Graph-surgical routing | Hierarchical ANOVA decomposition of the score |

**Paper narrative**: "Additive decomposition of diffusion score functions into factor-specific components is ubiquitous but under-theorized: when is ε(x,t,f) = Σ_i ε_i(x,t,f_i) meaningful? We prove that without constraints, the decomposition is non-identifiable — infinitely many equivalent decompositions exist. We characterize the gauge group and show that centering constraints (E[ε_i] = 0 per factor) + orthogonality (⟨ε_i, ε_j⟩ = 0 for i≠j) yield unique decompositions under mild distributional assumptions. We implement this in Factor-Gated Diffusion with per-stream centering regularization and demonstrate that the identified branches correspond to genuine factor-specific computational contributions."

**Key changes**:
- Theory section becomes the core contribution.
- Add: identifiability theorem, gauge group characterization, uniqueness proof under centering+orthogonality.
- Architecture becomes an existence proof, not the main contribution.
- Replace leakage matrix with branch identification quality metrics (cross-branch correlation, per-branch factor specificity after centering).
- Add: functional ANOVA decomposition connection (score function as Sobol-Hoeffding decomposition).
- Empirical section becomes validation of identifiability guarantees, not benchmarking.

**Evidence needed**:
- Mathematical proof of identifiability under constraints.
- Empirical measurement of uniqueness: do different initializations converge to the same (up to permutation) branch decomposition?
- Cross-branch correlation before/after centering regularization.
- Factor specificity curves for identified vs unidentified branches.

**Risk**: Theory-first paper requires tight proofs. Reviewers may demand more complex function classes than additive decomposition. May need to connect to functional ANOVA or operator-valued reproducing kernels. Venue: theory track (NeurIPS theory, ICML theory, or a statistics journal).

---

## Risk Status Summary

| Risk ID | Description | Probability | Impact | Status |
|---------|-------------|-------------|--------|--------|
| RISK-001 | Literature collision | Medium | High | Monitoring |
| RISK-002 | Shared x_t leakage | High | High | To test (Stage 4) |
| RISK-003 | Branch non-identifiability | High | Medium | To test (Stage 4) |
| RISK-004 | Base stream collapse | Medium | High | To test (Stage 3) |
| RISK-005 | Gate training compensation | Medium | Medium | To test (Stage 4) |
| RISK-006 | Wrong DAG = correct DAG | Medium | High | To test (Stage 5) |
| RISK-007 | Edges never used | Medium | Medium | To test (Stage 5) |
| RISK-008 | O(K) scaling | High | Medium | To profile (Stage 4) |
| RISK-009 | Oracle bias | High | Medium | To measure (Stage 4) |
| RISK-010 | Statistical power | Medium | Medium | Power analysis pre-Stage 4 |
| RISK-011 | adaLN-Zero bugs | Medium | Low | To fix (WP-11) |
| RISK-012 | Paired = unpaired | Low | Medium | To test (Stage 6) |
| RISK-013 | Base absorbs all semantics | Low | High | To test (Stage 4) |
| RISK-014 | Benchmarks too simple | Medium | Medium | Monitor (Stage 4) |
| RISK-015 | Causal3DIdent no structure | Medium | High | To measure (Stage 5) |
| RISK-016 | Reviewer rejects terminology | Medium | Medium | Mitigation in paper |
| RISK-017 | Theorem seen as trivial | Medium | Low | Mitigation in paper |
| RISK-018 | Implementation timeline | High | Medium | Track per-WP |
| RISK-019 | Need known graph | High | Medium | Acknowledge as limitation |
| RISK-020 | Need ImageNet-scale | Medium | Medium | Venue selection |
| RISK-021 | Gate gradient zero | Low | Medium | To test (Stage 0) |
| RISK-022 | DDIM quality poor | Medium | Low | To test (Stage 1) |
| RISK-023 | Position nuisance fails | Low | Medium | To test (Stage 4) |

**Total risks**: 23. High probability + high impact: RISK-002. High probability + medium impact: RISK-003, RISK-008, RISK-009, RISK-018, RISK-019.

**Next review**: After Stage 3 (Stochastic Micro-Overfit) completion.
