# 02 — Claim-Evidence Matrix

Each claim from README.md and MATH_NOTES.md is traced to code and assessed.

**Confidence legend**: V=VERIFIED, P=PARTIALLY, R=REFUTED, U=UNVERIFIED, B=BLOCKED

| Claim ID | Document Source | Claim | Type | Code Trace | Status | Confidence |
|----------|----------------|-------|------|------------|--------|------------|
| C-ARCH-001 | README:8 | Per-factor architectural streams | implementation fact | src/model.py:67-76 | V | 1.00 |
| C-ARCH-002 | README:8-9 | DAG cross-stream attention | architectural property | src/model.py:111-125 (but dag_edges=[] always) | P | 0.95 — code exists but NEVER activated |
| C-INT-001 | MATH_NOTES:13 | ∂ε/∂e_i = 0 when g_i=0, no children | mathematical theorem | src/model.py:30 (gate mult at output) | P | 0.90 — correct if no cross-attention bypass; but shared x_t means semantic info leaks |
| C-CAUSAL-001 | README:10, MATH_NOTES:19 | Gate as Pearl do-operator analog | causal claim | src/model.py:30 (x = x * gate) | **R** | 1.00 — output ablation ≠ do-operator |
| C-THEORY-001 | MATH_NOTES:13 | Gradient zero proposition | mathematical theorem | Verified in model forward | P | 0.80 — correct within architectural computational graph, but child edge propagation described incorrectly |
| C-THEORY-002 | MATH_NOTES:38 | Reduced sample complexity | theoretical claim | No proof, no code | **R** | 0.95 — no assumptions, no proof, counterexamples exist |
| C-THEORY-003 | MATH_NOTES:46-51 | DAG routing correctness | architectural claim | src/model.py:119-122 | P | 0.70 — dag exists but topological sort missing, child before parent = silent edge drop |
| C-THEORY-004 | MATH_NOTES:54-67 | Gate monotonicity | mathematical theorem | — | **R** | 1.00 — explicit counterexample with rotation ODE |
| C-EVAL-001 | README:151 | Oracle change measures factor presence | empirical claim | src/evaluate.py:108-118 | **R** | 1.00 — unpaired sampling invalidates |
| C-EVAL-002 | README:153 | Non-intervention stability | empirical claim | src/evaluate.py:102-106 | **R** | 0.95 — pixel threshold, not factor-level |
| C-EVAL-003 | README:152 | Gate sweep monotonicity | empirical claim | src/evaluate.py:120-132 | **R** | 1.00 — unpaired, gate-untrained |
| C-FAIR-001 | README:176-183 | Parameter-matched baselines | benchmark claim | src/baselines.py all models | P | 0.85 — params ~matched but depth/FLOPs differ |
| C-FAIR-002 | README:180 | CoInD baseline | benchmark claim | src/baselines.py:185-224 | **R** | 1.00 — Fisher divergence NOT implemented |
| C-FAIR-003 | README:179 | EncDiff baseline | benchmark claim | src/baselines.py:74-103 | **R** | 0.98 — concept token selection NOT implemented |
| C-FAIR-004 | README:182 | CF-DiT baseline | benchmark claim | src/baselines.py:227-237 | P | 0.90 — null token collides with class 0 |

## Required Claim Changes

All causal do-operator references MUST be replaced with "mechanistic path non-interference."

Proposition 4 MUST be removed and replaced with a Lipschitz sensitivity bound.

"CoInD" MUST be renamed to "IndependentStreamDiT" or actual CoInD loss implemented.
"EncDiff" MUST be renamed to "CrossAttnDiT" or actual concept-token conditioning implemented.
