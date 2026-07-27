# 00 — Executive Verdict

**Date**: 2026-07-27
**Commit**: c6cc096
**Status**: CONDITIONAL GO — PIVOT REQUIRED BEFORE GPU EXPERIMENTS

---

## Final Verdict

The core idea — decomposing diffusion denoising into explicit factor-specific computational paths with verifiable non-interference — has genuine research value. But c6cc096 does not implement or evaluate this idea correctly.

**12 P0 defects** must be resolved before any GPU training budget is spent. The most critical are:
1. Unpaired sampling invalidates all intervention metrics (src/evaluate.py:81-98)
2. Core DAG routing disabled in configs (use_cross_attn=false, dag_edges=[])
3. Pearl do-operator claim is mathematically wrong (MATH_NOTES.md:19)
4. Proposition 4 is proven false by counterexample (MATH_NOTES.md:54-67)
5. 4 baseline names misrepresent their implementations

## Required Pivot

| From | To |
|------|-----|
| Pearl do-operator | Mechanistic path non-interference |
| Scalar output gate | Incoming/outgoing edge surgery |
| Monolithic stream outputs | Per-stream norm + per-stream projection |
| Random split | Held-out compositional split |
| Unpaired sampling | Shared NoiseTrace (paired counterfactual) |

## Scorecard

| Dimension | c6cc096 | After Pivot (potential) |
|-----------|---------|------------------------|
| Novelty | 3/10 | 7/10 |
| Theory | 2/10 | 7/10 |
| Implementation | 3/10 | 8/10 |
| Evaluation | 1/10 | 8/10 |
| Top-tier readiness | 2/10 | 6/10 |
| SCI Q1 readiness | 3/10 | 8/10 |

## Go / No-Go

**CONDITIONAL GO**: Proceed only after P0 defects resolved AND micro-overfit verification passes.

Kill criterion: If path non-interference is not exact (within numerical tolerance), abandon architecture claims and pivot to empirical-only contribution.
