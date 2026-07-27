# Architecture: ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)

## Data Flow

```
z^(0) = PatchEmbed(x_t) + PosEmbed + TimeEmbed(t)

for l = 1..L:
    z^(l) = SharedBlock_l(z^(l-1), t)
    # z is read-only to factor branches; no branch writes back to z

a_i^(0) = FactorInit_i(z^(0), Enc_i(f_i), t)

for l = 1..L:
    snapshot = {a_j^(l-1)} for all j

    m_{j→i}^l = edge_gate_{j→i}^l * Message_l(snapshot[j], edge=(j,i))

    a_i^l = FactorAdapter_i^l(
        trunk=z^l,
        previous=a_i^(l-1),
        factor_source=source_gate_i * Enc_i(f_i),
        parent_message=Aggregate({m_{j→i}^l}),
        time=t
    )

epsilon_base = BaseHead(Norm_base(z^L))
epsilon_i = FactorHead_i(Norm_i(a_i^L))
epsilon_hat = epsilon_base + sum_i output_gate_i * epsilon_i
```

## Constraints

- **trunk-to-branch**: read-only access to `z^(l)`. Factor branches may read trunk activations but must never modify them.
- **branch-to-trunk**: WRITE FORBIDDEN. No branch activation, gradient, or state may flow back into the shared trunk.
- **branch-to-branch**: only explicit edge messages via `parent_message`. No implicit cross-talk through the trunk or shared buffers.
- **layerwise synchronous**: all branch states at layer `l` are computed from a frozen snapshot of layer `l-1` branch states. No sequential/autoregressive intra-layer updates.
- **factor source path auditable** via `PathCertificate`. Every intervention on a factor source gate must be statically verifiable as reaching or not reaching a given output head.

## Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trunk_dim` | 512 (default) | |
| `branch_dim` | 256 (default) | must be ≤ `trunk_dim` |
| `adapter_depth_per_layer` | 2 | MLP depth inside each `FactorAdapter` |
| `L_trunk_blocks` | 12 (dSprites), 24 (3DShapes) | |
| `L_branch_layers` | 4 per stream | parallel; one per trunk block layer if aligned |
| `branch_core_params_shared` | false | NOT shared across factors |
| `edge_message_params_shared` | true | single `Message` module per layer, edge_key as conditioning |
| `parent_aggregation` | degree-normalized sum (default) | alternative: mean, max, attention-weighted |

## Factor Encoder API

```python
FactorSpec = CategoricalFactorSpec | ContinuousFactorSpec

CategoricalFactorSpec:
    name: str
    cardinality: int
    null_index: int                    # equals cardinality (one extra embedding for null/masked)
    encoder: nn.Embedding

ContinuousFactorSpec:
    name: str
    dim: int
    encoder: MLP | FourierFeatures
```

### Encoder Factory

```python
def build_factor_encoder(spec: FactorSpec, d_model: int) -> nn.Module:
    if isinstance(spec, CategoricalFactorSpec):
        return nn.Embedding(spec.cardinality + 1, d_model)
    elif isinstance(spec, ContinuousFactorSpec):
        return MLP(spec.dim, d_model, hidden_dim=d_model*4, depth=2)
```

## Parameter Scaling (approximate, fp32)

| K | trunk_dim | branch_dim | Total Params | Notes |
|---|-----------|------------|-------------|-------|
| 3 | 512 | 256 | ~13M | |
| 6 | 512 | 256 | ~15M | branch params dominate at small K; trunk fixed cost |
| 10 | 512 | 256 | ~18M | |

### FLOPs

```
O(L_trunk * trunk_dim^2 + K * L_branch * branch_dim^2 + |E| * branch_dim^2)
```

Where:
- `L_trunk * trunk_dim^2`: shared trunk cost
- `K * L_branch * branch_dim^2`: per-factor adapter cost
- `|E| * branch_dim^2`: inter-branch message passing cost

## Initialization

| Component | Scheme | Rationale |
|-----------|--------|-----------|
| trunk | standard DiT init (xavier uniform) | proven convergence |
| factor adapters | small init (std=0.02) | prevent early dominance over trunk |
| factor heads | zero-init | initial `epsilon_i = 0`, model starts as pure trunk |
| base head | standard init | normal denoising head |
| factor encoders | standard embedding/MLP init | |
| edge gates | initialized to 0.5 (default) | neutral message flow at start |

## Graph-Free Mode

- `graph_type = INDEPENDENT`: all `edge_gate` values set to zero, no `parent_message` computation.
- Factor branches still access trunk (`z^l`) and own factor source (`Enc_i(f_i)`).
- Equivalent to K independent adapters reading a shared trunk.
- Reduces to Fallback Architecture (Fully Independent Additive Score Experts) when edge gating is removed.

## Permutation Semantics

The architecture is permutation-equivariant over factor nodes:

- Renumbering factor nodes (bijection `π: [K] → [K]`) combined with:
  - Permuting factor embeddings: `Enc_i → Enc_{π(i)}`
  - Permuting the edge list: `(j,i) → (π(j), π(i))`
- …yields **identical** `epsilon_hat` output.

This holds because all operations are either factor-wise (encoders, adapters, heads) or defined over unordered sets (edge aggregation is sum-based and commutative).

## Forward Execution Order (Implementation)

```
1. PatchEmbed + PosEmbed + TimeEmbed → z^(0)
2. For each factor i: FactorInit → a_i^(0)
3. For l = 1 to L:
   a. z^(l) = SharedBlock_l(z^(l-1), t)           # trunk step
   b. snapshot = {a_1^(l-1), ..., a_K^(l-1)}      # freeze
   c. For each edge (j,i):                         # message passing
      m_{j→i}^l = edge_gate_{j→i}^l * Message_l(snapshot[j], edge=(j,i))
   d. For each factor i:                           # branch update
      a_i^l = FactorAdapter_i^l(trunk=z^l, previous=a_i^(l-1),
              factor_source=source_gate_i * Enc_i(f_i),
              parent_message=Aggregate({m_{j→i}^l}), time=t)
4. epsilon_base = BaseHead(Norm_base(z^L))
5. epsilon_i = FactorHead_i(Norm_i(a_i^L)) for i=1..K
6. epsilon_hat = epsilon_base + sum_i output_gate_i * epsilon_i
```

## Fallback Architecture (Candidate A)

Fully Independent Additive Score Experts:
- No shared trunk. Each factor has a complete independent denoising head.
- Scores aggregated additively: `epsilon_hat = sum_i w_i * epsilon_i(f_i)`.
- Serves as the lower-bound baseline for factor-specific modeling without shared representation.
- Recoverable from ROST-FRG by setting trunk_dim=0 and removing trunk blocks.

## PathCertificate (Static Auditing)

```python
@dataclass
class PathCertificate:
    factor_idx: int
    source_gate: bool                  # True if source path exists to encoder
    trunk_access: bool                 # True if trunk read path exists (always True in ROST-FRG)
    edge_paths: list[tuple[int, int]]  # list of (parent, child) reachable from this factor
    output_paths: list[int]            # list of output heads reachable from this factor
    is_fully_cut: bool                 # True iff no path from factor source to any output

    def verify(self, intervention: CompiledIntervention) -> bool:
        """Verify certificate against an actual compiled intervention."""
        ...
```

---

*This document is the canonical architecture specification. All implementation must conform to constraints, data flow, and initialization schemes described herein. Deviations require an architecture change proposal (ACP) approved via the project ledger.*
