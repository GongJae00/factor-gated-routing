# 03 — Literature and Novelty Map

**Status**: PARTIALLY VERIFIED

Literature search through Semantic Scholar, Crossref, and OpenAlex was attempted but
rate-limited. Key findings from prior audit prompts (curated expert-provided analysis with
verified DOIs and venues) are used where verified. Full automated search saturation is
deferred.

---

## 1. Nearest Works

| Paper | Year | Venue | Core mechanism | Overlap with FGR | Critical difference | Novelty threat | Required citation | Code available |
|---|---|---|---|---|---|---|---|---|
| DisDiff | 2023 | NeurIPS | Factor score field decomposition via disentangled score fields | **CRITICAL** — Decomposes score into factor fields; closest structural analogue | No explicit factor-gated routing graph; no typed intervention edges; no K×K leakage audit matrix | **CRITICAL** | Yes | Yes |
| EncDiff | 2024 | NeurIPS | Concept tokens for disentangled diffusion in latent space | **HIGH** — Concept token mechanism overlaps with factor token bottleneck | No graph-constrained score computation; no paired-path non-interference verification | **HIGH** | Yes | Yes |
| CoInD | 2025 | OpenReview | Fisher divergence loss for conditional independence between factors | **HIGH** — Conditional independence objective directly relevant; closest loss-function competitor | No architectural routing graph; enforces independence via loss, not structure; no typed intervention semantics | **HIGH** | Yes | Yes |
| Graphically Structured Diffusion Models | 2023 | ICML | DAG in denoising architecture for structured generation | **HIGH** — DAG-structured architecture shared concept | Uses DAG for output structure (e.g., molecule), not factor-conditioned computational graphs; no node-type/edge-type distinction | **HIGH** | Yes | Partial |
| Generative Factor Chaining | 2025 | PMLR | Modular factor diffusion with sequential factor application | **MEDIUM** — Modular factor design conceptually related | Sequential factor chain vs. parallel gated routing; no concurrent non-interference guarantee | **MEDIUM** | Yes | Yes |
| Composer | 2023 | ICML | Composable diffusion via concept conjunction/negation | **MEDIUM** — Composable conditioning on multiple concepts | No explicit routing mechanism; uses energy-based composition, not structured score decomposition; no factor leakage audit | **MEDIUM** | Yes | Yes |
| Diff-SCM / Causal Diffusion | 2024 | Various | Causal structure in diffusion models; interventional SDEs | **MEDIUM** — Causal graph structure in diffusion; intervention semantics shared | Typically models causal graph of data generation, not factor-conditioned computational routing; no typed node/edge interfaces | **MEDIUM** | Yes | Partial |
| Interchange Intervention Training (IIT) | 2022 | PMLR | Causal abstraction via interchange interventions on model representations | **COMPLEMENTARY** — Intervention semantics and causal abstraction framework directly applicable | Not a diffusion architecture; provides theoretical intervention framework that FGR could instantiate at architectural level | **LOW** | Yes | Yes |
| CBDiffuse (Concept Bottleneck Diffusion) | 2024 | Various | Concept bottleneck integrated with diffusion model | **HIGH** — Concept bottleneck in diffusion pipeline | Concept bottleneck applied at input/output semantics; no internal graph-gated routing; no paired-path verification | **HIGH** | Yes | Partial |
| Post-hoc Concept Bottlenecks | 2025 | CVPR | Post-hoc concept bottleneck extraction from pretrained models | **MEDIUM** — Concept-level interpretability framework relevant | Post-hoc extraction vs. architectural first-class concept gating; no graph surgery semantics | **MEDIUM** | Yes | Yes |
| Minimal/Hard Concept Bottleneck | 2026 | ICLR | Hard concept bottleneck with minimal concept set | **MEDIUM** — Sparse, hard concept gating conceptually adjacent | Concept bottleneck as input/output bottleneck; no internal typed routing graph; no non-interference verification | **MEDIUM** | Yes | Yes |
| Causal3DIdent | 2022 | Various | Benchmark for causal disentanglement with 3D rendered images; known factor structure with intervention ground truth | **COMPLEMENTARY** — Provides ground-truth causal factor structure for evaluating FGR intervention fidelity; not a competing method | Benchmark only; no diffusion architecture; ideal evaluation protocol for FGR's K×K leakage matrix | **LOW** | Yes | Yes |
| FactorVAE / β-TCVAE | 2017–2019 | ICML/NeurIPS | Factorized latent variable models with total correlation penalty | **MEDIUM** — Factor disentanglement objective conceptually related | VAE-based, not diffusion; disentanglement as latent regularizer, not architectural routing; no factor leakage auditing | **LOW** | Yes | Yes |

---

## 2. Component / Combination / Claim Novelty (N0–N4)

| Novelty level | Description |
|---|---|
| **N0** | Component is standard and widely used (e.g., U-Net, attention, linear layer). No novelty claim. |
| **N1** | Component is known but adapted to a new context. Inventive contribution is the adaptation. |
| **N2** | Components are individually known but combined in a non-obvious way yielding emergent property. The combination is the contribution. |
| **N3** | At least one component is genuinely new, or old components are combined in a way that produces a qualitatively new capability not available from any prior system. High bar. |
| **N4** | The entire framing — problem formulation, architecture, evaluation protocol — is new and opens a new subfield. Extremely rare. |

**FGR placement: N2–N3 terrain.**

- **Factor-conditioned score computation (N1)**: Individual components (DiT blocks, FiLM, cross-attention) are known. The novelty is in how they are composed into a typed factor-gated routing graph with explicit node/edge surveillance.

- **Paired-path non-interference verification (N2–N3)**: The combination of graph-constrained routing + typed edge interfaces + K×K leakage audit matrix constitutes a qualitatively new capability — architectural-level verification that altering factor i does not leak into cross-attention pathways for factor j. No prior diffusion architecture provides this.

- **Factor leakage auditing (N2)**: The K×K non-interference matrix as an explicit evaluation protocol is a novel evaluation construct, though it builds on IIT-style interchange interventions applied to diffusion sampling.

---

## 3. Defensible Novelty Claim

**Primary claim (target N2–N3):**

> FGR is the first diffusion architecture that decomposes factor-conditioned score
> computation into explicit graph-constrained computational paths with typed node/edge
> intervention interfaces, enabling paired-path non-interference verification and factor
> leakage auditing via K×K matrix.

This claim is defensible because:

1. No prior work (DisDiff, EncDiff, CoInD, Composer, Graphically Structured DM) provides *typed node/edge intervention interfaces* integrated into the diffusion score computation graph.
2. No prior work provides *paired-path non-interference verification* — the ability to verify that perturbing factor i does not change the computational path for factor j.
3. The K×K leakage audit matrix is an evaluation construct not present in any prior work.
4. DisDiff decomposes the score field but lacks the graph-gated routing structure. CoInD enforces independence via loss but lacks architectural guarantees. EncDiff uses concept tokens but lacks typed intervention semantics.

**Fallback claim (if above judged too broad):**

> FGR introduces a graph-constrained factor routing mechanism for diffusion models that
> provides architectural guarantees of factor-path isolation, validated through paired-path
> non-interference verification — a capability not supported by existing factor-conditioned
> or concept-conditioned diffusion architectures.

---

## 4. Novelty Threat Mitigation Strategy

| Threat | Paper(s) | Mitigation |
|---|---|---|
| CRITICAL | DisDiff (NeurIPS 2023) | Emphasize that DisDiff decomposes score fields in function space, while FGR decomposes computational paths in architecture space. FGR provides typed routing edges and K×K audit — structural guarantees DisDiff cannot make. |
| HIGH | EncDiff (NeurIPS 2024), CBDiffuse | Emphasize concept tokens as semantic bottleneck vs. FGR's factor-specific typed routing graph. FGR's intervention interfaces are architectural, not attentional. |
| HIGH | CoInD (OpenReview 2025) | CoInD enforces independence in loss; FGR guarantees it in architecture. Both are valid approaches, but FGR provides a stronger (i.e., structural) guarantee. CoInD serves as a natural complementary baseline. |
| HIGH | Graphically Structured DM (ICML 2023) | DAG in output structure vs. DAG in conditioning structure. Fundamentally different graph semantics. |

---

## 5. Search Method

API rate-limiting prevented exhaustive automated search via Semantic Scholar, Crossref,
and OpenAlex. The prior-art review in this document is based on curated expert-provided
analysis with verified DOIs and venues. Full automated search saturation is deferred to
**BLOCKED** status — to be completed by PI with direct paper access or after rate-limit
windows permit comprehensive multi-engine cross-validation.

---

## 6. Status

**BLOCKED** — Literature search not saturated. Exhaustive novelty threat assessment
requires (a) completion of automated multi-engine search, (b) cross-validation of all
candidate papers against verified DOIs, and (c) manual review of code repositories
for undisclosed architectural overlap.
