# 13 — Paper Positioning

## Recommended Title (pending literature verification)

**Factor-Path Diffusion: Graph-Surgical Routing for Auditable Conditional Generation**

Rationale: Drops "Verifiable" (vague), drops "Controllable" (too broad), adds "Graph-Surgical" (precise novelty claim pending literature), "Auditable" (distinct contribution). Title is provisional — final determination after literature saturation.

## Three-Line Central Claim (safe version)

> We factorize condition-to-score computation into explicit factor-specific paths and constrain inter-path communication with a known graph. Node and edge interventions provide exact computational-path non-interference, while paired-noise evaluation with shared random numbers quantifies target efficacy and off-target leakage via a K×K factor matrix.

**Note**: "paired counterfactual" replaced with "paired-noise evaluation with shared random numbers" — SCM causal semantics not claimed.

## Three Contributions (hypothesized, not empirically verified)

1. **Architecture**: Factor-path denoiser with typed intervention interface (8 modes: observational, factor_edit, direct_output_ablation, full_source_cut, node_deletion, edge_ablation, neural_graph_surgery, condition_mask).

2. **Theory**: Path Non-Interference Theorem — if all directed paths from factor embedding e_i to denoiser output pass through a complete cutset, then output is functionally invariant to e_i when cut. Trajectory-level induction for shared-noise sampling. Grönwall-based sensitivity bound (unverified proof sketch).

3. **Evaluation protocol**: NoiseTrace-mediated paired evaluation with common random numbers, K×K OffTargetChange matrix, held-out compositional splits, and staged experiments across dSprites, 3DShapes, and known-SCM benchmarks.

## Claims Ladder

| Level | Claim | Evidence Required | Status |
|-------|-------|-------------------|--------|
| L0 | Separate parameter paths exist per factor | Code inspection | Available in c6cc096 |
| L1 | Full source cut → exact functional invariance | Theorem + finite-diff test + trajectory induction | Requires architecture redesign |
| L2 | FGR off-target leakage ≤ monolithic baseline | Paired evaluation + OffTargetChange matrix | Hypothesis — not tested |
| L3 | Correct graph routing reduces leakage vs wrong graph | Causal3DIdent (known-SCM) benchmark | Hypothesis — not tested |
| L4 | Neural graph surgery ≈ causal intervention | IIT alignment or SCM proof | **NOT CLAIMED** — future work |

**All L2-L3 claims are hypotheses pending experiment.** Current document uses "we hypothesize" and "we will test" language.

## Paper Structure

1. **Introduction** (1.5 pages): Problem — monolithic denoisers obscure factor-specific pathways. Solution — explicit computational paths + typed interventions. No causal overclaim.

2. **Related Work** (1 page): Concept bottlenecks, disentangled diffusion, graph-structured diffusion, causal generation. Position FGR as graph-surgical factor-path routing (not causal intervention, not concept bottleneck).

3. **Problem Formulation** (0.5 pages): Distinguish conditioning from path ablation from graph surgery. Define NoiseTrace-mediated paired evaluation.

4. **Architecture** (1.5 pages): Factor streams, base stream, synchronous routing, per-stream projections, InterventionSpec API.

5. **Intervention Semantics** (1 page): 6 intervention modes, typed gates, what each mode guarantees and doesn't guarantee.

6. **Theory** (1 page): Path Non-Interference Theorem + trajectory corollary. NO monotonicity claim. NO sample complexity claim.

7. **Evaluation Protocol** (0.5 pages): NoiseTrace, paired sampling, leakage matrix, held-out splits.

8. **Experiments** (2 pages): dSprites (edit + leakage), 3DShapes (scale), Causal3DIdent (graph correctness).

9. **Ablations** (1 page): base stream, per-stream vs shared LN, gate-trained vs not, correct vs wrong DAG.

10. **Limitations** (0.5 pages): Shared x_t leakage, O(K) cost, no causal guarantee, synthetic benchmarks only.

## Figures

1. Architecture diagram with base stream + factor streams + edge gates
2. 3 intervention modes side-by-side (edit, ablation, surgery)
3. Leakage matrix heatmap (K×K)
4. Correct vs reversed DAG results
5. Gate response curves (not monotonicity, but AUC/response shapes)

## Tables

1. Main results: leakage matrix diagonals + off-diagonals for FGR vs baselines
2. Graph misspecification: correct/empty/complete/reversed DAG comparison
3. Ablation: per-stream vs shared LN, base stream contribution
4. Parameter/FLOP/depth comparison
