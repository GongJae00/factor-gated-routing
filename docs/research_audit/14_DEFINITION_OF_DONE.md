# 14 — Definition of Done (v2)

## Gate 0: Audit Completeness

- [ ] All 17 docs exist with README, traceability, manifest
- [ ] 85 hypotheses fully adjudicated (H-001 through H-085)
- [ ] 59 AUDIT-CORR items resolved or BLOCKED with reason
- [ ] Cross-document terminology frozen (InterventionSpec, graph types, metrics)
- [ ] Literature search saturated OR BLOCKED with documented reason

## Gate 1: Specification Freeze

- [ ] Canonical InterventionSpec enum frozen (8 modes, single source of truth)
- [ ] Graph type enum frozen (INDEPENDENT, DAG, DENSE_DIRECTED, CUSTOM)
- [ ] Metric definitions frozen (TargetSuccess, OffTargetChange, NoOpChange=0)
- [ ] Architecture primary + fallback selected
- [ ] Baseline rename/implement decisions made
- [ ] No unresolved P0 semantic contradictions across docs

## Gate 2: CPU Property Tests (NO GPU)

- [ ] All 38 tests pass (see test plan v2)
- [ ] Graph validation: cycle/invalid/duplicate/self-loop all catch errors
- [ ] full_source_cut invariance: denoiser level < 1e-5 (fp32)
- [ ] direct_output_ablation: o_i=0 → contribution measured as zero
- [ ] outgoing preserve: cut incoming, preserve outgoing → child receives message
- [ ] NoiseTrace identity: same inputs → exact output match
- [ ] No do-operator or "counterfactual" language in code or docs
- [ ] Config stored in checkpoint; evaluation reconstructs from checkpoint

## Gate 3: Deterministic Micro-Overfit (~10 min GPU)

- [ ] Pre-generate N=16 fixed (x0, t, noise, condition) tuples
- [ ] Model memorizes tuples (loss → near 0)
- [ ] full_source_cut invariance verified on overfit model
- [ ] neural_graph_surgery semantics verified (incoming cut + v' inject + outgoing preserve)

## Gate 4: Stochastic Sanity (~1h GPU)

- [ ] Model trains without NaN on small batch for 1000 steps
- [ ] Loss decreases monotonically over training window
- [ ] Base stream contribution stable (measured by ablation loss increase when base removed)

## Gate 5: dSprites Pilot (~12h GPU)

- [ ] FGR vs canonical DiT vs IndependentStreamDiT on IID split + held-out pair split
- [ ] 2 training seeds
- [ ] Leakage matrix shows structure (diagonal behavior ≠ off-diagonal)
- [ ] Off-target preservation comparable or better than baselines
- [ ] Held-out combinations: FGR generalization ≥ monolithic DiT

## Gate 6: 3DShapes + Graph Correctness (~30h GPU)

- [ ] 3DShapes results consistent with dSprites findings
- [ ] Causal3DIdent: correct graph vs empty/complete/reversed
- [ ] 3 seeds minimum
- [ ] Graph fidelity metrics computed

## Gate 7: Confirmatory Experiment (~70h GPU, submission-ready)

- [ ] All faithful baselines compared
- [ ] 5 training seeds × 1K eval samples
- [ ] Paired bootstrap 95% CIs
- [ ] All ablations complete
- [ ] All tables auto-generated from eval JSON

## Gate 8: Submission

- [ ] Reproducibility manifest complete (exact commit, environment lock, seeds)
- [ ] Oracle checkpoint released
- [ ] No hallucinated citations
- [ ] All claims ladder verified against evidence

## Kill Criteria (re-evaluated v2)

1. **Path invariance NOT exact** at CPU test gate (error > tolerance) → investigate bypass paths before pivoting; implementation bug is more likely than fundamental flaw.
2. **Base stream contribution measured by branch ablation**: remove base stream, measure reconstruction quality drop. If drop < 10% → base stream adds little (acceptable). If drop > 30% → base stream absorbing too much signal → reduce capacity.
3. **O(K) cost**: Measure actual GPU time at K=3 and K=6, extrapolate. If K=10 requires >48h on RTX 5080, switch to shared trunk architecture (Candidate B).
4. **Correct graph = wrong graph**: If no significant difference on known-SCM benchmark → graph alignment doesn't help → drop graph contribution, become auditability-only paper.
5. **Gate training effectiveness**: If binary gate training doesn't improve semantic steering over untrained gates → drop continuous gate claim, keep binary intervention only.
6. **Literature saturation**: If nearest neighbor work invalidates core novelty → pivot to auditability-only or evaluation-framework paper.
