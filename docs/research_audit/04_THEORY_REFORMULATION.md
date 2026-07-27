# 04 — Theory Reformulation

## Proposition 1 → Path Non-Interference Theorem

**Original claim** (MATH_NOTES:11-16): ∂ε/∂e_i = 0 when g_i=0. Partially correct but child-stream description is inconsistent.

**Reformulated theorem**:

> **Path Non-Interference Theorem.** Let ε_θ be an FGR denoiser where factor embedding e_i enters only through stream i's initial transform and any cross-stream edges where stream i appears as parent. If all directed paths from e_i to the denoiser output pass through a multiplicative cut variable c_i, then for c_i = 0 and any e_i, e_i':
>
> ε_θ(x_t, t, e_i, e_−i; c_i=0) = ε_θ(x_t, t, e'_i, e_−i; c_i=0)
>
> Hence ∂ε_θ/∂e_i = 0 wherever differentiable.

**Trajectory-level corollary** (induction proof):

> If X_T = X'_T and per-step noise traces are identical, and at every timestep the denoiser satisfies the above invariance, then the entire reverse process trajectory is identical.

This holds for:
- Deterministic DDIM/ODE: trivially (no noise)
- Stochastic DDPM with shared noise trace: by induction
- Stochastic DDPM with independent noise: does NOT hold

**What this theorem does NOT guarantee**:
- That factor i disappears from generated images (other streams can infer it from x_t)
- That no off-target leakage occurs (shared x_t is a bypass path)
- Causal intervention equivalence
- Semantic disentanglement

## Proposition 2 → Downgraded to Hypothesis

**Original**: Sample complexity advantage claim. No proof, no assumptions.

**Downgrade**: "FGR imposes a structural inductive bias that may improve factor-specific learning when the true score admits a sparse graph-aligned additive decomposition."

Required to prove: explicit function class assumptions, correct graph, bounded approximation error, capacity-matched hypothesis classes. Beyond scope of this work.

## Proposition 3 → Replaced by Graph Validation Requirement

**Original**: DAG prevents "peeking" and supergraph edges are auto-ignored.

**Correction**: 
- In conditional diffusion, all factor labels are observed simultaneously — there IS no "peeking" in the autoregressive sense.
- Supergraph spurious edges are NOT auto-ignored without edge regularization or sparsity penalty.
- Correct graph vs empty/complete/random/reversed DAG must be tested empirically.

## Proposition 4 → Deleted + Replaced

**Original**: Lipschitz-based gate monotonicity. **PROVEN FALSE** by ODE counterexample (2D rotation).

**Counterexample**: ẋ = gAx with A = 2π[[0,-1],[1,0]] gives Δ(g) = 2-2cos(2πg) which is NOT monotonic.

**Replacement — Gate Sensitivity Bound**:

Using Grönwall inequality: if the reverse ODE drift f is L-Lipschitz in state and the factor stream norm is bounded by M:

||X_0(g) - X_0(g')|| ≤ |g-g'| · M · T · e^{LT}

This provides a Lipschitz bound on gate response (NOT monotonicity). Gate monotonicity becomes an empirical diagnostic only.

## Branch Identifiability Analysis

The current aggregation ε̂ = W·LN(Σ_i g_i h_i) has a gauge freedom: any δ_i with Σ_i δ_i = 0 can be added to h_i without changing the output. This means:
1. Branch semantics are NOT uniquely identified by the reconstruction loss alone.
2. Factor-specific input restriction and separate factor embeddings provide only weak identifiability.
3. A per-stream output projection P_i (summed in noise space) would help but not fully solve.
4. Centering constraints (E_{F_i}[ε_i] = 0) or functional ANOVA decomposition could provide stronger guarantees.

## Causal Terminology Policy

| Term | Allowed? | Condition |
|------|----------|-----------|
| conditioning | YES | Factor label given as input |
| factor edit | YES | Changing factor condition value |
| path ablation | YES | Cutting computational path |
| node ablation | YES | Zeroing neural node output |
| edge intervention | YES | Blocking neural message edge |
| graph surgery | YES | Explicit incoming/outgoing edge manipulation |
| causal intervention | NO | Unless SCM equivalence proven |
| counterfactual | YES (limited) | Only if same exogenous randomness fixed |
| do-operator | **NO** | Not justified by current architecture |
