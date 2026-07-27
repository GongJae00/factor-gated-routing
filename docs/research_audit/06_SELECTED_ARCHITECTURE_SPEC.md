# 06 — Selected Architecture Specification

## Primary Direction: ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)

Selected over alternatives because it cleanly separates factor-path computation while preserving path auditability, at manageable implementation complexity. The shared trunk processes the noisy image in a factor-agnostic way; per-factor adapter branches read from the trunk (no write-back) and produce factor-specific score contributions.

## Architecture Formula

```
# Trunk: shared factor-agnostic DiT backbone (T layers)
# Read-only — branches can read trunk activations but never modify them
z = SharedTrunk(x_t, t)             # [B, N, d_trunk] patch tokens + pos embed + t embed

# Base stream for nuisance variation (position, texture, background)
ε_base = BaseHead(z, t)             # Lightweight predictor reading from trunk

# For each factor i, an adapter branch reads from trunk:
for factor i in 1..K:
    # Adapter_i injects factor embedding into trunk representation
    # via LoRA-style injection or FiLM modulation at selected trunk layers.
    # Read-only: Adapter_i reads z but does NOT write back to the trunk.
    h_i = FactorAdapter_i(z, Embed_i(f_i), t)

    # Per-branch output head (no shared LayerNorm across branches)
    ε_i = OutputHead_i(h_i)

# Additive score decomposition
ε̂ = ε_base + Σ_i ε_i
```

**ROST-FRG key property**: Branches are read-only on the trunk. No branch writes back to trunk state. This means the trunk can be observed and audited independently of any single branch. Path non-interference is verified by measuring whether a cut branch changes other branches' trunk readings.

## Graph Type Enum

```python
class GraphType(enum.Enum):
    INDEPENDENT = "independent"      # No edges
    DAG = "dag"                      # Directed acyclic, validated at construction
    DENSE_DIRECTED = "dense_directed" # All j≠i (K>1 has 2-cycles, NOT a DAG)
    CUSTOM = "custom"                # Explicit edge list, may contain cycles (user beware)
```

**Key distinction**: DENSE_DIRECTED is NOT a DAG when K>1. It contains the 2-cycle (0→1,1→0). Graph validation at construction raises ValueError for cycles only in DAG mode. DENSE_DIRECTED mode runs synchronous layerwise updates (all states from layer-(l-1) snapshot), so node index order is irrelevant.

## Gate Taxonomy (typed InterventionSpec — canonical 8 modes)

Full canonical specification: see `spec/INTERVENTION_SPEC.md`. The 8-mode intervention table:

| Mode | factor | o_i | r_{*→i} | r_{i→*} | Meaning |
|------|--------|-----|---------|---------|---------|
| observational | original | 1 | 1 | 1 | Normal conditional generation |
| factor_edit | v→v' | 1 | 1 | 1 | Change factor value, all paths open |
| direct_output_ablation | original | 0 | 1 | 1 | Silence direct output ε_i; messages to children UNCHANGED |
| factor_source_cut | irrelevant | 0 | 0 | 0 | Cut ALL e_i→output paths; output invariant to f_i (Path Non-Interference Theorem guarantees full invariance) |
| node_deletion | irrelevant | 0 | 0 | 0 | Delete node i entirely |
| edge_ablation | original | 1 | selected 0 | 1 | Cut specific parent→child edge |
| neural_graph_surgery | v' | 1 | 0 (incoming) | 1 (outgoing) | Intervene on node i: cut parent→i edges, inject v', keep i→child edges |
| condition_mask | hidden | 1 | 1 | 1 | Replace f_i with null/masked token |

**Critical distinction**: `direct_output_ablation` (o_i=0) does NOT guarantee Path Non-Interference if other pathways from factor i to output exist (e.g., via shared trunk propagation). Only `factor_source_cut` provides the complete cutset needed for the theorem.

**Terminology**: `neural_graph_surgery` replaces the former `graph_surgery` (which was imprecisely named). It is NOT a causal do-operator. It cuts incoming neural edges, injects a new factor value, and preserves outgoing neural edges. No SCM equivalence is claimed.

## Key Design Decisions

1. **Read-only trunk access**: Adapter branches read from shared trunk but never write back. Trunk state is factor-agnostic and independently auditable.
2. **Per-branch OutputHead_i**: Each factor branch has its own output projection. Enables additive contribution interpretation: ε̂ = ε_base + Σ_i ε_i.
3. **Base stream**: Handles nuisance (dSprites position, texture). Capacity-limited to prevent collapse absorbing factor signal.
4. **Factor embedding injection**: Adapters inject factor embeddings via LoRA-style low-rank modulation or FiLM conditioning at selected trunk layers. No cross-branch message passing needed.
5. **Graph type param**: Graph configuration stored in model_config for audit. On DAG datasets (Causal3DIdent), the adapter routing policy may respect the parent-child edge set. On independent-factor datasets (dSprites, 3DShapes), all adapters operate independently.
6. **Config in checkpoint**: model_config stored alongside state_dict. Evaluation reconstructs from checkpoint, not from CLI args.

## Parameter Scaling (ROST-FRG)

| Component | Scaling | K=3 (dSprites) | K=6 (3DShapes) |
|-----------|---------|----------------|-----------------|
| Shared Trunk (DiT-S/2) | O(1) | ~12M | ~12M |
| Per-branch adapters (LoRA rank r) | O(K·r) | +r·K | +r·K |
| Base stream head | O(1) | ~2.1M | ~2.1M |
| **Total** | **O(1 + K·r)** | **~15-16M** | **~16-17M** |

Trunk cost is fixed (not O(K)), good for scalability. Adapter cost grows linearly with K at low adapter rank r (r=16 or r=32). Exact figures pending implementation — initial training run produces final count.
