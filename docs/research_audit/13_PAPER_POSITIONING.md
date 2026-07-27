# 13 — Paper Positioning

## Recommended Title

**Factor-Path Diffusion: Graph-Surgical Routing for Auditable Conditional Generation**

Rationale: Drops "Verifiable" (weak), drops "Controllable" (too broad), adds "Graph-Surgical" (precise novelty), "Auditable" (distinct contribution).

## Three-Line Central Claim

> We factorize condition-to-score computation into explicit factor-specific paths and constrain inter-path communication with a known graph. Node and edge interventions provide exact computational-path non-interference, while paired counterfactual evaluation quantifies target efficacy and off-target leakage.

## Three Contributions

1. **Architecture**: Factor-path denoiser with typed graph-surgical intervention interface (6 modes: observational, factor_edit, path_ablation, node_deletion, edge_ablation, graph_surgery).

2. **Theory**: Path Non-Interference Theorem — if all paths from factor embedding e_i to output pass through multiplicative cut, then output is functionally invariant to e_i when the cut is zero. Trajectory-level corollary for shared-noise sampling.

3. **Evaluation**: Paired NoiseTrace protocol + K×K factor leakage matrix on dSprites, 3DShapes, and Causal3DIdent (known-SCM benchmark), demonstrating that correct factor graph routing reduces off-target leakage vs. monolithic and wrong-graph baselines.

## Claims Ladder

| Level | Claim | Evidence Required |
|-------|-------|-------------------|
| L0 | Separate parameter paths exist per factor | Code inspection |
| L1 | Path cut → exact functional invariance | Theorem + finite-difference test |
| L2 | FGR reduces off-target leakage | Paired evaluation + leakage matrix |
| L3 | Correct graph > wrong graph for intervention | Causal3DIdent benchmark |
| L4 | Neural intervention ≈ SCM do-intervention | **NOT CLAIMED in this version** |

L4 is deferred to future work (requires IIT alignment or SCM equivalence proof).

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
