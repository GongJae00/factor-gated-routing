# 00 — Executive Verdict (v2)

**Audit commit**: aa14213
**Code reference**: c6cc096
**Cutoff date**: 2026-07-27

---

## Three Verdicts

### 1. Research Direction: CONDITIONAL GO

The core idea — decomposing diffusion denoising into explicit factor-specific computational paths with verifiable non-interference — has genuine research value. But c6cc096 does not implement or evaluate this idea correctly. The required pivot transforms the contribution from "causal do-operator diffusion" (indefensible) to "graph-surgical factor-path routing with mechanistic audit" (defensible with evidence).

### 2. Audit Completeness: COMPLETE (v2 at aa14213)

17 documents, H-001 through H-085 adjudicated, 23 risks registered, 60+ implementation tasks, 35+ tests, 31 experiment rows across 7 stages.

### 3. Implementation Start: BLOCKED

Blocked by: specification gate (intervention semantics, graph types, metric definitions must be frozen before code changes).

---

## Living Research Core

> Factor-conditioned denoiser with explicit graph-constrained computational paths, typed InterventionSpec (8 modes), paired-noise evaluation, and K×K factor leakage auditing.

## Three Most Critical Risks

1. **Shared x_t semantic leakage**: Even with exact computational path cut, other streams infer factor i from x_t → path isolation is computationally exact but semantically leaky.
2. **Branch decomposition non-identifiability**: Gauge freedom Σ_i δ_i = 0 means per-stream semantics are not uniquely determined by reconstruction loss.
3. **Literature collision**: DisDiff (NeurIPS 2023) factor score decomposition, CBDiffuse concept bottleneck diffusion, GSDM (ICML 2023) DAG-in-architecture → precise combination must be narrow.

## Required Pivot

| From | To |
|------|-----|
| Pearl do-operator | Mechanistic path non-interference |
| Scalar output gate | Typed InterventionSpec with 8 modes |
| `gates=[1.0]*K` always | Binary gate training (condition-masking or branch dropout) |
| Monolithic stream outputs | Per-stream norm + base stream + per-stream projection |
| Unpaired sampling | NoiseTrace-mediated paired-noise evaluation |
| CoInD/EncDiff names | IndependentStreamDiT / CrossAttnDiT |
| Prop 4 (monotonicity) | Deleted → Lipschitz sensitivity bound (unverified) |

## Recommended Primary Direction

Candidate B: Shared factor-agnostic trunk + factor-specific adapters with edge-gated message passing. Fallback: Candidate A (fully independent additive streams).

## Implementation Gate

**First actions before any code change**:
1. Freeze terminology (all docs agree on "paired-noise evaluation" not "counterfactual")
2. Freeze InterventionSpec canonical enum (8 modes)
3. Freeze graph type enum (INDEPENDENT, DAG, DENSE_DIRECTED)
4. Freeze Path Non-Interference Theorem scope (complete source-to-output cutset required)
5. Freeze primary metric definitions (TargetSuccess, OffTargetChange, NoOpChange=0)
