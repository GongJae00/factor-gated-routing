# 05 — Architecture Options

Comparative evaluation of four candidate architectures for factor-gated routing (FGR).
Each is assessed across ten dimensions; a primary recommendation and fallback are
provided.

---

## 1. Architecture Candidates

### A. Fully Independent Additive Score Experts

**Description**: K independent denoisers, no shared parameters. Each factor k receives its
own full εθ(x_t, k, t) model. A base/unconditional stream εθ(x_t, t) provides the shared
baseline. Factor-specific outputs are added: ε(x_t, {k_j}, t) = ε_base(x_t, t) + Σ_j ε_j(x_t, k_j, t).
Each expert sees only its own factor condition.

- **Strongest path isolation** — no shared computation across factors
- **Trivial non-interference verification** — by construction, factor i cannot leak into factor j's denoiser
- **O(K) parameter and compute cost** — infeasible for K > ~5 at realistic scale

### B. Shared Factor-Agnostic Trunk + Factor Adapters

**Description**: A shared backbone processes the noisy image x_t. Factor-specific adapter
modules (LoRA-style low-rank injections or lightweight FiLM blocks) inject per-factor
residuals at designated layers. The adapter for factor k is activated only when factor k
is in the conditioning set. Routing is implicit via adapter selection.

- **Better efficiency** — shared trunk amortizes computation
- **Weaker path isolation** — adapters interact through shared representations
- **Mitigation**: Path isolation verified post-hoc via interchange interventions
  at adapter boundaries

### C. Factor Token Bottleneck + Graph-Gated Transformer

**Description**: Factor-specific concept tokens (one or few per factor) attend to image patch
tokens through a graph-gated attention mechanism. The graph gate for factor k controls
which patches receive factor-k information. Aligns with the concept bottleneck literature
(CBDiffuse, Post-hoc Concept Bottlenecks, EncDiff).

- **Conceptually well-motivated** — direct alignment with concept bottleneck literature
- **Collision risk** — close to EncDiff + graph attention; novelty claim may be challenged
- **Attention-based gating** — softer isolation guarantees than architectural path separation

### D. Hierarchical ANOVA Score Decomposition

**Description**: Decompose the score into ANOVA-like components: base effect μ(x_t, t),
main factor effects α_k(x_t, k, t), pairwise interaction effects β_{ij}(x_{ti}, x_{tj}, t),
and higher-order DAG-constrained interactions. Impose centering constraints
(e.g., E[α_k | x_t] = 0) for mathematical identifiability.

- **Mathematical identifiability** — ANOVA decomposition is theoretically grounded
- **High implementation complexity** — centering constraints, higher-order interaction terms
- **Scalability concern** — O(K^2) interaction terms even for pairwise

---

## 2. Comparison Table

| Dimension | A: Independent Experts | B: Shared Trunk + Adapters | C: Token Bottleneck + Graph-Gate | D: Hierarchical ANOVA |
|---|---|---|---|---|
| **Strict path isolation** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ |
| **Semantic identifiability** | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **Graph surgery semantics** | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ |
| **Computational efficiency** | ★☆☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| **Parameter scaling** | O(K) — poor | O(1 + K·r) — good | O(1 + K·d) — moderate | O(K^2) — poor |
| **Novelty vs literature** | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★★★ |
| **Literature collision risk** | LOW | LOW | HIGH (EncDiff, CBDiffuse) | LOW |
| **Implementation risk** | LOW | LOW | MEDIUM | HIGH |
| **Top-tier potential** | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| **Realistic-scale feasibility** | ★☆☆☆☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |

---

## 3. Recommendation

### PRIMARY: Architecture B — Shared Factor-Agnostic Trunk + Factor Adapters

**Rationale**: Architecture B provides the best balance across all dimensions. It is
computationally feasible at realistic scale (RTX 5080 16 GB), offers strong novelty
with low literature collision risk, and has manageable implementation complexity.
The weaker path isolation (relative to A) is mitigated by the K×K non-interference
audit matrix — FGR's key contribution is not perfect isolation by construction, but
*verifiable* isolation. Architecture B makes this verification meaningful rather than
trivial.

**Implementation strategy**:
- Shared DiT trunk (or U-Net trunk) as backbone
- Per-factor LoRA adapters at each transformer block
- Factor gating: adapter k is active iff factor k ∈ conditioning set
- Non-interference verified via interchange interventions at adapter boundaries
- Base stream always active

### FALLBACK: Architecture A — Fully Independent Additive Score Experts

**Rationale**: If Architecture B fails to demonstrate sufficient path isolation (i.e., K×K
leakage matrix shows significant off-diagonal leakage that cannot be reduced through
adapter regularization), fall back to A. Architecture A provides perfect isolation by
construction, at the cost of O(K) compute. This is acceptable for small K (K ≤ 5) and
serves as a clean proof-of-concept that a fully isolated factor-conditioned diffusion
architecture is possible.

**Conditions for fallback**:
- K×K leakage matrix for Architecture B shows >10% cross-path leakage on any off-diagonal
- Adapter regularization (orthogonality constraints, gradient isolation) fails to reduce leakage
- Reviewer feedback demands stricter isolation guarantees

---

## 4. Rejected Options

### Rejected: Architecture C

Collision risk with EncDiff + Graph Attention is too high. The novelty space around
concept tokens + graph gating is crowded. While it could be a competent system, it
does not provide a clear novelty margin. May be reconsidered as a secondary baseline
if EncDiff is implemented faithfully.

### Rejected: Architecture D (as primary)

Implementation risk is unacceptable for a single-RTX-5080 project timeline. The
mathematical identifiability is attractive but requires extensive validation of centering
constraints and higher-order interaction estimation. If Architecture B succeeds and
reviewers request theoretical grounding, D can be developed as the theoretical framework
*analysis* paper rather than the primary implementation. Architecture D packages could
also serve as a future extension (FGR-v2).
