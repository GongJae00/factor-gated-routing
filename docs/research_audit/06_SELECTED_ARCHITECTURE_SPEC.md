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

## Gate Taxonomy (typed InterventionSpec)

| Gate | Symbol | Semantics |
|------|--------|-----------|
| output gate | o_i ∈ [0,1] | Controls direct output contribution of stream i to ε̂ |
| edge gate | r_{j→i}^(l) ∈ [0,1] | Controls message from parent j to child i at layer l |
| node cut | c_i^{in}=0 ∀ j∈Pa(i) | Cuts ALL incoming edges to node i |
| outgoing preserve | r_{i→k}=1 ∀ k∈Ch(i) | Keeps outgoing edges when incoming is cut |

## Intervention Modes

```python
@dataclass
class InterventionSpec:
    mode: str  # "observational", "factor_edit", "path_ablation", "graph_surgery"
    edited_factors: dict[int, int] | None  # factor_idx → new_value
    output_gates: list[float] | None        # per-stream o_i
    incoming_cut: set[int] | None           # nodes whose incoming edges are cut
    edge_gates: dict[(int,int), float] | None  # (parent,child) → r value
```

| Mode | factor | o_i | r_{*→i} | r_{i→*} | Meaning |
|------|--------|-----|---------|---------|---------|
| observational | original | 1 | 1 | 1 | Normal conditional generation |
| factor_edit | v→v' | 1 | 1 | 1 | Change factor value |
| path_ablation | original | 0 | 1 | 1 | Silence stream output |
| node_deletion | irrelevant | 0 | 0 | 0 | Remove stream entirely |
| edge_ablation | original | 1 | selected=0 | 1 | Cut specific edge |
| graph_surgery | v' | 1 | 0 (incoming) | 1 (outgoing) | do-like operation |

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
