# 14 — Definition of Done

## Phase Gate 0: Code Correctness (BEFORE ANY GPU)

- [ ] All 28 tests pass (see test plan)
- [ ] Path invariance verified (max error < 1e-5 for denoiser, < 1e-3 for trajectory)
- [ ] Same NoiseTrace identity verified (max error < 1e-6)
- [ ] Checkpoint roundtrip verified
- [ ] Config always stored in checkpoint
- [ ] Evaluation reconstructs from checkpoint, not CLI args
- [ ] No do-operator language in any file
- [ ] No baseline name misrepresentations
- [ ] `.gitignore` cleaned
- [ ] README paths correct
- [ ] `research-check .` passes

## Phase Gate 1: Micro-Overfit (GPU-short, ~1h)

- [ ] Model can overfit 16 images (loss → near 0)
- [ ] Path cut invariance is exact on overfit model
- [ ] Graph surgery semantics verified on overfit model
- [ ] Base stream doesn't collapse (contribution > 0 but < total)

## Phase Gate 2: dSprites Pilot (GPU-medium, ~12h)

- [ ] FGR trains stably (no NaN, loss decreases)
- [ ] Leakage matrix shows structure (diagonal ≠ off-diagonal)
- [ ] Factor edit: target accuracy > baseline
- [ ] Off-target preservation > baseline
- [ ] Gate training produces smooth gate response (not necessarily monotonic)
- [ ] Held-out combinations: FGR ≥ monolithic DiT

## Phase Gate 3: Full Experiment (GPU-full, ~70h)

- [ ] All 5 seeds × 1K samples
- [ ] Paired bootstrap CIs
- [ ] All baselines faithfully implemented
- [ ] Causal3DIdent results: correct > wrong graph
- [ ] Ablations complete

## Phase Gate 4: Paper Submission

- [ ] All figures generated
- [ ] All tables auto-generated from eval JSON
- [ ] No hallucinated citations
- [ ] Reproducibility manifest complete
- [ ] Oracle checkpoint released
- [ ] Exact commit hash in paper

## Kill Criteria

1. Path invariance NOT exact (error > 1e-5) → architecture has bypass path → abandon L1 claim
2. Base stream absorbs > 90% of output → factor streams don't learn → reconsider architecture
3. O(K) cost makes training infeasible at RTX 5080 scale → need shared trunk (Architecture B)
4. Gate training doesn't improve over inference-only gating → drop gate training
5. Correct graph = wrong graph on Causal3DIdent → graph alignment doesn't matter → drop graph contribution, become auditability-only paper
