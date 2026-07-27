# 06 — Selected Architecture Specification

## Primary Direction: Candidate B (Graph-Surgical Factor Routing)

Selected over alternatives because it cleanly separates graph intervention semantics while preserving path isolation, at manageable implementation complexity.

## Architecture Formula

```
z  = SharedPatchEmbed(x_t) + pos_embed + t_embed

h_i^(0) = z + FactorEmbed_i(f_i)

for layer l in 1..L:
    # Synchronous: snapshot ALL layer-(l-1) states first
    for all i:
        m_{j→i}^(l) = r_{j→i}^(l) · M_{j→i}^(l)(h_j^(l-1))  for j in Pa(i)
        h_i^(l) = StreamBlock_i^(l)(h_i^(l-1), Σ m_{j→i}^(l), t + FactorEmbed_i(f_i))

# Per-stream projection (no shared LN!)
ε_i = o_i · P_i · Norm_i(h_i^(L))

# Base stream for nuisance variation (position, texture, background)
ε_base = BasePredictor(z, t)

# Final noise prediction
ε̂ = ε_base + Σ_i ε_i
```

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

```python
@dataclass
class InterventionSpec:
    mode: str  # one of the 8 canonical modes below
    edited_factors: dict[int, int] | None       # factor_idx → new_value
    output_gates: list[float] | None             # per-stream o_i ∈ [0,1]
    incoming_cut: set[int] | None                # nodes whose incoming edges are cut
    outgoing_preserve: set[int] | None           # nodes whose outgoing edges are preserved (when incoming is cut)
    edge_mask: dict[tuple[int,int], float] | None  # (parent,child) → r value (per-edge, per-layer if layered)
```

| Mode | factor | o_i | r_{*→i} | r_{i→*} | Meaning |
|------|--------|-----|---------|---------|---------|
| observational | original | 1 | 1 | 1 | Normal conditional generation |
| factor_edit | v→v' | 1 | 1 | 1 | Change factor value, all paths open |
| direct_output_ablation | original | 0 | 1 | 1 | Silence direct output ε_i; messages to children UNCHANGED |
| full_source_cut | irrelevant | 0 | 0 | 0 | Cut ALL e_i→output paths; output invariant to f_i (Path Non-Interference Theorem applies) |
| node_deletion | irrelevant | 0 | 0 | 0 | Delete node i entirely |
| edge_ablation | original | 1 | selected 0 | 1 | Cut specific parent→child edge |
| neural_graph_surgery | v' | 1 | 0 (incoming) | 1 (outgoing) | Intervene on node i: cut parent→i edges, inject v', keep i→child edges |
| condition_mask | hidden | 1 | 1 | 1 | Replace f_i with null/masked token |

**Critical distinction**: `direct_output_ablation` (o_i=0) does NOT guarantee Path Non-Interference if outgoing messages exist. Only `full_source_cut` provides the complete cutset needed for the theorem.

**Terminology**: `neural_graph_surgery` replaces the former `graph_surgery` (which was imprecisely named). It is NOT a causal do-operator. It cuts incoming neural edges, injects a new factor value, and preserves outgoing neural edges. No SCM equivalence is claimed.

## Key Design Decisions

1. **Per-stream Norm_i + P_i**: Replaces shared LayerNorm. Enables additive contribution interpretation.
2. **Base stream**: Handles nuisance (dSprites position, texture). Capacity-limited to prevent collapse.
3. **Synchronous layerwise routing**: Layer-l states ALL computed from layer-(l-1) snapshots. No sequential dependency on stream index order.
4. **Graph validation at construction**: Topological sort, cycle detection, node range check. Invalid edges = ValueError.
5. **Config in checkpoint**: model_config stored alongside state_dict. Evaluation reconstructs from checkpoint, not from CLI args.

## Parameter Scaling

| Config | K=3 (dSprites) | K=6 (3DShapes) |
|--------|----------------|-----------------|
| FGR no-CA | 13.2M | 25.9M |
| FGR with DAG CA | 16.4M | 32.2M |
| Base stream addition | +2.1M | +2.1M |

Base stream cost is fixed (not O(K)), good for scalability.
