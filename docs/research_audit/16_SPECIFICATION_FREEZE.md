# 16 — Specification Freeze (v3)

**Document type**: Specification freeze gate
**Spec version**: 3.0
**Freeze date**: 2026-07-27
**Based on**: audit v2 commit `c700d341eb543c83e7d10ced034ffc2d8a179762`
**Code reference commit**: `c6cc0968ccf4b39e6400792b6bdd38a4e57135cd`
**Status**: FROZEN

---

## Freeze Version

- **Spec version**: 3.0
- **Freeze date**: 2026-07-27
- **Based on**: audit v2 commit `c700d34` (the second-pass architecture audit with corrected intervention taxonomy, PathCertificate specification, and formal architecture constraints)
- **Code reference**: `c6cc096` (baseline commit of the Factor-Path Diffusion codebase against which the architecture audit was performed)
- **Provenance**: `docs/research_audit/spec/provenance.yaml`
- **Correction register**: `docs/research_audit/CORRECTION_REGISTER_V3.md` (25 entries, all RESOLVED)

---

## Frozen Decisions

### 1. Architecture

**Primary**: ROST-FRG (Read-Only Shared Trunk + Factor Residual Graph)

```
z^(0) = PatchEmbed(x_t) + PosEmbed + TimeEmbed(t)
for l = 1..L:
    z^(l) = SharedBlock_l(z^(l-1), t)
    snapshot = {a_1^(l-1), ..., a_K^(l-1)}
    for each edge (j,i):
        m_{j→i}^l = edge_gate_{j→i}^l * Message_l(snapshot[j], edge=(j,i))
    for each factor i:
        a_i^l = FactorAdapter_i^l(
            trunk=z^l, previous=a_i^(l-1),
            factor_source=source_gate_i * Enc_i(f_i),
            parent_message=Aggregate({m_{j→i}^l}), time=t)
epsilon_base = BaseHead(Norm_base(z^L))
epsilon_i = FactorHead_i(Norm_i(a_i^L))
epsilon_hat = epsilon_base + sum_i output_gate_i * epsilon_i
```

**Fallback**: Fully Independent Additive Score Experts (Candidate A)
- No shared trunk. K independent denoisers.
- Recoverable from ROST-FRG by setting `trunk_dim=0` and removing trunk blocks.
- Serves as lower-bound baseline for factor-specific modeling without shared representation.

**Constraints (frozen)**:
- Shared trunk is factor-agnostic and read-only to branches.
- Branch-to-trunk write: FORBIDDEN.
- Branch-to-branch communication: only via explicit edge messages (parent_message).
- Per-branch independent output heads: `Norm_i` + `FactorHead_i`.
- Base stream handles nuisance: `BaseHead(Norm_base(trunk_output))`.
- Layerwise synchronous branch updates: all branch states at layer `l` computed from a frozen snapshot of layer `l-1` branch states.
- Factor source path statically auditable via `PathCertificate`.
- Branch adapter core parameters NOT shared across factors.
- Edge message parameters shared per layer, edge_key as conditioning.

**Dimensions (frozen)**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trunk_dim` | 512 (default) | |
| `branch_dim` | 256 (default) | must be <= `trunk_dim` |
| `adapter_depth_per_layer` | 2 | MLP depth inside FactorAdapter |
| `L_trunk_blocks` | 12 (dSprites), 24 (3DShapes) | |
| `L_branch_layers` | 4 per stream | parallel; one per trunk block layer |
| `edge_message_params_shared` | true | single Message module per layer |
| `parent_aggregation` | degree-normalized sum (default) | alternative: mean, max, attention-weighted |

**Initialization (frozen)**:

| Component | Scheme |
|-----------|--------|
| trunk | standard DiT init (xavier uniform) |
| factor adapters | small init (std=0.02) |
| factor heads | zero-init (model starts as pure trunk) |
| base head | standard init |
| factor encoders | standard embedding/MLP init |
| edge gates | initialized to 0.5 (default) |

**Training defaults (frozen)**:

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW |
| learning_rate | 1e-4 |
| weight_decay | 1e-6 |
| batch_size | 128 |
| noise_schedule | cosine |
| diffusion_steps | 1000 |
| sampling_steps | 250 (DDIM default) |
| factor_balance_sampling | stratified per factor |

See: `spec/ARCHITECTURE_SPEC.md`, `spec/architecture.yaml`

---

### 2. Graph Types

**Canonical enum**:
```python
class GraphType(enum.Enum):
    INDEPENDENT = "independent"
    DAG = "dag"
    DENSE_DIRECTED = "dense_directed"
    CUSTOM_DIRECTED = "custom_directed"
```

**Independence specifications**:

| Type | Edges | Cycles | Is DAG? | Validation | Sync Required |
|------|-------|--------|---------|------------|---------------|
| INDEPENDENT | [] | N/A | Yes | None (empty set always valid) | No |
| DAG | user-provided list | FORBIDDEN | Yes | Cycle detection, node range, no duplicate, no self-loop | No (but uses snapshot for consistency) |
| DENSE_DIRECTED | all (j,i) for j!=i | 2-cycles exist when K>1 | No | Node range only (auto-generated edges) | Yes |
| CUSTOM_DIRECTED | user-provided list | Allowed by default | No | Node range, no duplicate, self-loop policy | Yes |

**Frozen rules**:
- "complete DAG" is forbidden terminology.
- DENSE_DIRECTED contains 2-cycles when K>1, not a DAG.
- Forward execution: synchronous snapshot at each layer for ALL graph types (not sequential per topological order, even for DAG).
- Validation at construction for all types: cycle detection (DAG mode), node range check, duplicate edge check, self-loop policy.
- Self-loops: FORBIDDEN by default (configurable for CUSTOM_DIRECTED).
- Node numbering: arbitrary, zero-indexed, independent of topological order.
- Permutation equivariance: renumbering nodes + permuting embeddings + permuting edge list yields identical epsilon_hat output.

**API**:
```python
validate_graph(edges, graph_type, n_factors) -> None | raises ValueError
PathCertificate.verify_factor_source_cut(spec, factor_idx) -> bool
PathCertificate.list_uncovered_paths(spec, factor_idx) -> list[Path]
PathCertificate.get_reachable_outputs(factor_idx, graph, intervention) -> set[int]
PathCertificate.compute_path_matrix(graph) -> Tensor[K, K]
```

**Validation details**:
- Cycle detection: DFS with recursion stack coloring (white/gray/black), O(K + |E|).
- Transitive closure: Floyd-Warshall, O(K^3), acceptable for K <= 20.
- Topological sort: Kahn's algorithm (BFS), O(K + |E|), used for validation only.

See: `spec/graphs.yaml`

---

### 3. Factor Types

**Canonical factor spec**:
```python
FactorSpec = CategoricalFactorSpec | ContinuousFactorSpec

CategoricalFactorSpec:
    name: str
    cardinality: int
    null_index: int  # equals cardinality (one extra embedding for null/masked)
    encoder: nn.Embedding(cardinality + 1, d_model)

ContinuousFactorSpec:
    name: str
    dim: int
    encoder: MLP(dim, d_model) | FourierFeatures(dim, d_model)
```

**Factor encoder factory**:
```python
def build_factor_encoder(spec: FactorSpec, d_model: int) -> nn.Module:
    if isinstance(spec, CategoricalFactorSpec):
        return nn.Embedding(spec.cardinality + 1, d_model)
    elif isinstance(spec, ContinuousFactorSpec):
        return MLP(spec.dim, d_model, hidden_dim=d_model*4, depth=2)
```

**Scope (frozen)**:
- P0 scope: categorical only (dSprites + 3DShapes).
- P1 scope: continuous for Causal3DIdent.
- Factor embedding injection: via `source_gate_i * Enc_i(f_i)` at each FactorAdapter.

---

### 4. Intervention Modes (8 Canonical)

**Single source of truth**: `spec/INTERVENTION_SPEC.md`

**Canonical enum**:
```python
class InterventionMode(str, Enum):
    OBSERVATIONAL = "observational"               # 1
    FACTOR_EDIT = "factor_edit"                   # 2
    CONDITION_MASK = "condition_mask"              # 3
    DIRECT_OUTPUT_ABLATION = "direct_output_ablation"  # 4
    EDGE_ABLATION = "edge_ablation"               # 5
    NODE_DELETION = "node_deletion"               # 6
    FACTOR_SOURCE_CUT = "factor_source_cut"        # 7
    NEURAL_GRAPH_SURGERY = "neural_graph_surgery"  # 8
```

**Mode semantics table (frozen)**:

| Mode | factor value | source_gate | node_gate | output_gate | incoming edges | outgoing edges | Theorem/Invariant |
|------|-------------|-------------|-----------|-------------|----------------|----------------|-------------------|
| OBSERVATIONAL | original | 1 | 1 | 1 | preserve | preserve | identity |
| FACTOR_EDIT | v->v' | 1 | 1 | 1 | preserve | preserve | single-factor counterfactual |
| CONDITION_MASK | null_token | 1 | 1 | 1 | preserve | preserve | marginalization proxy |
| DIRECT_OUTPUT_ABLATION | original | 1 | 1 | 0 | preserve | preserve | no invariance guarantee; only removes additive contribution |
| EDGE_ABLATION | original | 1 | 1 | 1 | selected=0 | selected=0 | symmetric edge cut |
| NODE_DELETION | original | 0 | 0 | 0 | cut all | cut all | full factor excision |
| FACTOR_SOURCE_CUT | any (irrelevant) | 0 | 1 | 1 | preserve | preserve | Factor-Source Path Non-Interference |
| NEURAL_GRAPH_SURGERY | v' | 1 | 1 | 1 | cut incoming | preserve | NOT a causal do-operator; graph-modified edit |

**Pairwise distinctness**: All C(8,2)=28 unordered pairs are distinguishable by gate configuration (proven in INTERVENTION_SPEC.md).

**CompiledIntervention**:
```python
@dataclass(frozen=True)
class CompiledIntervention:
    effective_factor_values: Tensor[B, K]
    source_gate: Tensor[B, K]  # in {0,1}
    node_gate: Tensor[B, K]    # in {0,1}
    output_gate: Tensor[B, K]  # in {0,1}
    edge_gate: Tensor[B, L, max_E]  # in {0,1}
    mode: InterventionMode

def compile_intervention(spec: InterventionSpec, config: ModelConfig) -> CompiledIntervention:
    """Single choke-point for all intervention logic. No other code may modify gate tensors."""
```

**InterventionSpec**:
```python
@dataclass
class InterventionSpec:
    mode: InterventionMode
    target_factor_idx: int | None = None
    new_value: int | float | None = None
    edge_indices: list[tuple[int, int]] | None = None
    preserve_factor_values: list[int] | None = None
```

**Stale names — BANNED (frozen)**:

| Stale Name | Canonical Replacement |
|------------|----------------------|
| `path_ablation` | FACTOR_SOURCE_CUT or DIRECT_OUTPUT_ABLATION |
| `full_source_cut` | FACTOR_SOURCE_CUT |
| `graph_surgery` | NEURAL_GRAPH_SURGERY |
| `output_gate_only` | FACTOR_SOURCE_CUT |
| `do_like` / `do-like` | NEURAL_GRAPH_SURGERY |
| `drop_factor` | NODE_DELETION |
| `zero_out` | DIRECT_OUTPUT_ABLATION |
| `intervene` | specify the exact mode |

See: `spec/INTERVENTION_SPEC.md`, `spec/interventions.yaml`

---

### 5. Theory

**Factor-Source Path Non-Interference Theorem (frozen)**:

> Let epsilon_theta be a ROST-FRG denoiser where factor embedding e_i enters only through stream i's initial transform and any cross-stream edges where stream i appears as parent. If all directed paths from e_i to the denoiser output pass through a multiplicative cut variable c_i, then for c_i = 0 and any e_i, e_i':
>
> epsilon_theta(x_t, t, e_i, e_{-i}; c_i=0) = epsilon_theta(x_t, t, e'_i, e_{-i}; c_i=0)
>
> Hence partial epsilon_theta / partial e_i = 0 wherever differentiable.

**Requirements for the theorem to hold**:
- Complete cutset (FACTOR_SOURCE_CUT mode: source_gate_i=0).
- Branch-to-trunk write is forbidden (enforced by architecture constraint).
- All factor-source-to-output paths pass through source_gate_i multiplicatively.
- PathCertificate verifies no alternative ungated paths exist.

**Trajectory corollary (frozen)**:
> If x_T = x'_T and per-step noise traces are identical, and at every timestep the denoiser satisfies the above invariance, then the entire reverse process trajectory is identical (induction).

This holds for: deterministic DDIM/ODE (trivially), stochastic DDPM with shared noise trace (by induction). Does NOT hold for stochastic DDPM with independent noise.

**Grönwall bound (frozen)**:
> If the reverse ODE drift f is L-Lipschitz in state and the factor stream norm is bounded by M:
>
> ||X_0(g) - X_0(g')|| <= |g-g'| * M * T * e^{LT}

This is a proof sketch only (unverified). Marked as such in all documents. Provides a Lipschitz sensitivity bound, NOT monotonicity.

**What the theorem does NOT guarantee (frozen)**:
- That factor i disappears from generated images (other streams infer it from shared x_t).
- Semantic factor removal at the image level.
- Causal intervention equivalence.
- Branch identifiability (gauge freedom exists: any delta_i with sum_i delta_i=0 preserves output).

**What is NOT claimed (frozen)**:
- Sample complexity advantage (Prop 2 downgraded to hypothesis).
- Gate monotonicity (Prop 4 proven false, replaced with Grönwall sensitivity bound).
- Supergraph auto-ignore (Prop 3 replaced with empirical test requirement).
- Do-operator equivalence (banned terminology).

**Terminology policy (frozen)**:

| Term | Status |
|------|--------|
| conditioning | ALLOWED |
| factor edit | ALLOWED |
| paired-noise evaluation | ALLOWED (canonical) |
| neural graph surgery | ALLOWED (canonical mode name) |
| counterfactual | LIMITED (literature/future-work context only, with qualification) |
| causal intervention | NO |
| do-operator | BANNED |
| disentanglement | NO (FGR produces factor-specific paths, not disentangled representations) |
| "complete DAG" | BANNED (DENSE_DIRECTED is not a DAG) |

See: `04_THEORY_REFORMULATION.md`

---

### 6. Metrics

**Categorical factor metrics (frozen)**:

| Metric | Range | Direction | Unit |
|--------|-------|-----------|------|
| TargetValueSuccess | [0,1] | higher better | per-factor K-vector |
| TargetChangeRate | [0,1] | higher better (for intended change) | per-factor K-vector |
| OffTargetChange | [0,1] | lower better (factor isolation) | KxK matrix, diagonal masked |
| NoOpChange | [0,1] | lower better (0=deterministic) | KxK, all zero with same trace |
| SourceInvarianceError_Denoiser | [0, inf) | lower better (0=perfect) | per-factor |
| SourceInvarianceError_Trajectory | [0, inf) | lower better (0=perfect) | per-factor, per-timestep |
| DirectContributionEffect | [0, inf) | descriptive | per-factor |
| EdgeEffect | [0, inf) | descriptive | per-edge, per-layer |
| NonDescendantChange | [0,1] | lower better (graph-causal consistency) | per-factor, graph-dependent |
| DescendantResponse | [0,1] | descriptive | per-factor, graph-dependent |
| GraphVariantEffectSize | [0, inf) | descriptive | per-graph-pair |

**Continuous factor metrics (frozen)**:

| Metric | Range | Direction |
|--------|-------|-----------|
| TargetValueMSE | [0, inf) | lower better |
| TargetChangeMagnitude | [0, inf) | descriptive |
| OffTargetContinuousChange | [0, inf) | lower better |
| FactorCorrelationMatrix | [-1,1] | near 0 preferred |

**Generation quality metrics (frozen)**:

| Metric | Range | Direction |
|--------|-------|-----------|
| ReconstructionFID | [0, inf) | lower better |
| FactorDisentanglementScore (DCI) | [0,1] | higher better |

**Aggregate metrics (frozen)**:

| Metric | Formula | Range | Direction |
|--------|---------|-------|-----------|
| FactorIsolationIndex | (1/K)*sum_i[TargetChangeRate_i * (1-OffTargetChange_i)] | [0,1] | higher better |
| SourceInvarianceScore | (1/K)*sum_i Indicator[SourceInvarianceError_Denoiser_i < epsilon_tol] | [0,1] | higher better (should be 1.0) |

**Metric naming conventions (frozen)**: snake_case. Full names required in code. Prefix rules: Target*, OffTarget*, SourceInvariance*, *Change, *Effect, *Score, *Index.

**Statistical standards (frozen)**:
- Confidence level: 0.95
- Significant digits: 3
- Bootstrap samples: 1000
- Multiple testing correction: Benjamini-Hochberg FDR for per-factor metrics
- Effect size reporting: required alongside confidence intervals

See: `spec/metrics.yaml`

---

### 7. Sampler

**Primary correctness sampler**: Deterministic DDIM, eta=0.
- Fully deterministic given x_T. No per-step randomness to couple.
- Stored: only x_T per sample (O(B * C * H * W) memory, no per-step noise tensors).
- Counter-based RNG (Philox) used for x_T generation to enable exact reconstruction.

**Stochastic DDPM with NoiseTrace**:
- On-the-fly deterministic noise generation from counter seeds.
- Counter-seed hash: composed from (base_seed, sample_id, timestep, noise_stream_id).
- Memory: 4 bytes per (sample, step) vs 4*C*H*W bytes per (sample, step).

**NoiseTrace representations (frozen)**:
```python
@dataclass
class NoiseTrace:
    x_T: Tensor[B, C, H, W]
    sampler: str = "ddim"
    eta: float = 0.0
    steps: int = 50

@dataclass
class NoiseTraceStochastic:
    x_T: Tensor[B, C, H, W]
    sampler: str = "ddpm"
    steps: int = 1000
    seed_grid: list[int]  # [B * steps] counter seeds
    rng_algorithm: str = "philox"
```

**Invariant (frozen)**: `sample(model, condition, trace=A)` and `sample(model, condition, trace=A)` must produce identical tensors down to float32 precision (bit-exact). Same hardware/dtype/version only. This is a Phase Gate 0 requirement.

**Paired evaluation protocol (frozen)**:
```
x_original = sample(checkpoint, original_factors, trace=trace_A)
x_edited   = sample(checkpoint, edited_factors,   trace=trace_A)  # same trace
```

See: `08_DATA_AND_EVALUATION_PROTOCOL.md`

---

### 8. Dataset Splits

**Split types (frozen)**:

| Split | Description | Deterministic |
|-------|-------------|---------------|
| S0 | IID random 90/10 | Yes (seeded RNG) |
| S1 | Held-out pair combinations | Yes |
| S2 | Held-out triple combinations | Yes |
| S3 | Systematic compositional (hash-based modular split) | Yes |

**Construction rules (frozen)**:
- All splits deterministic: seeded RNG at split construction time.
- Indices stored as JSON manifest alongside dataset.
- Splits are dataset-specific, not model-specific.
- Same split indices used across all models for fair comparison.

**Factor assignment (frozen)**:
- dSprites: shape, scale, rotation = factor streams 0-2. pos_x, pos_y = base stream nuisance.
- 3DShapes: all 6 factors = factor streams 0-5. Base stream nuisance only.
- Independence: dSprites factors are independent. 3DShapes factors are independent (uniform Cartesian product).

**Data loading (frozen)**:
- Lazy HDF5 access: per-worker file handles, memory-mapped.
- No full dataset preload into RAM.
- `swmr=True` for safe concurrent reads.

See: `08_DATA_AND_EVALUATION_PROTOCOL.md`

---

### 9. Baseline Names

**Canonical baseline names (frozen)**:

| Canonical Name | Former Name | Reason for Change |
|---------------|-------------|-------------------|
| CanonicalDiT (adalN-Zero) | SDiT | Missing adaLN-Zero (critical component) |
| IndependentStreamDiT | CoInDDiT | Missing Fisher divergence loss |
| CrossAttnDiT | EncDiffDiT | Missing concept tokens |
| AllToAllFactorStreamDiT | MMDiT-k | MMDiT architecture differs; all-to-all is more descriptive |
| CF-DiT | CF-DiT (unchanged) | Mechanism is CFG, but bugs fixed: dedicated null tokens + per-factor dropout |

**Fairness regime (frozen)**:
- Parameter-matched (within +/-5%).
- FLOP-matched (within +/-5% per denoising step).
- Best-practice (each baseline's optimal hyperparameters).
- Report: param count, FLOPs/step, VRAM peak, FID, leakage matrix per configuration.

See: `07_BASELINE_FIDELITY_PLAN.md`

---

### 10. Statistics

**Experimental unit**: Training seed.

**Phase-specific statistical plan (frozen)**:

| Phase | Seeds | Statistical Method | Purpose |
|-------|-------|--------------------|---------|
| Development | 1-2 | None | Fast iteration |
| Pilot (Stage 4-5) | 2-3 | Direction of effect, variance estimate | Initial signal |
| Confirmatory (Stage 6) | 5 | Paired bootstrap 95% CI, binomial CI | Main results |

**Confidence intervals (frozen)**:
- Paired bootstrap 95% CI (B=10,000): continuous metrics (LPIPS, FID, MSE).
- Binomial Clopper-Pearson 95% CI: accuracy metrics (target success, leakage proportions).
- Effect size: Cohen's h for proportions, Cohen's d for continuous.

**Multiple comparison (frozen)**:
- Benjamini-Hochberg FDR correction (alpha=0.05) across factor-wise leakage comparisons.
- Report both raw and corrected p-values.

**Pilot restrictions (frozen)**:
- No p-values reported for pilot results.
- Direction of effect and variance estimate only.
- Statistical significance deferred to confirmatory phase with 5+ seeds.

**Evaluation sample size (frozen)**:
- 1,000 evaluation samples per seed * condition.
- For factor_edit with K=3 and 3 value pairs: 3 * 3 * 1000 = 9,000 generated images per seed.

See: `08_DATA_AND_EVALUATION_PROTOCOL.md`

---

## Rejected Alternatives

| Alternative | Rejection Reason | Frozen Date |
|-------------|-----------------|-------------|
| Architecture C (Token Bottleneck) | Literature collision risk too high (EncDiff, CBDiffuse) | 2026-07-27 |
| Architecture D (ANOVA) | Implementation risk infeasible for single-RTX timeline; O(K^2) interaction terms | 2026-07-27 |
| Scalar gate as Pearl do-operator | Mathematically incorrect; neural path manipulation != causal intervention | 2026-07-27 |
| Output-gate-only as invariance test | Insufficient cutset; factor source still influences other branches via edges | 2026-07-27 |
| "complete DAG" terminology | DENSE_DIRECTED contains 2-cycles; "complete DAG" is a contradiction in terms | 2026-07-27 |
| "counterfactual" as primary evaluation term | SCM semantics not justified; replaced with "paired-noise evaluation" | 2026-07-27 |
| MMDiT-k baseline name | Architecture differs from MMDiT; renamed to AllToAllFactorStreamDiT | 2026-07-27 |
| Candidate B as permanent architecture name | Temporary evaluation label; replaced with canonical ROST-FRG | 2026-07-27 |

---

## Unresolved Non-Blocking Questions

These are design choices where multiple valid options exist. The decision is deferred until experimental evidence is available. None blocks Phase-0 implementation.

| # | Question | Options | Deferred Until |
|---|----------|---------|----------------|
| 1 | Edge message parameters: shared across layers or per-layer? | Share saves params; per-layer gives flexibility | Phase 0 CPU profiling |
| 2 | branch_dim = trunk_dim or branch_dim < trunk_dim? | =256 vs =512; affects param count and expressivity | Phase 1 parameter sweep |
| 3 | Exact parent aggregation function | degree-normalized sum (default) vs attention-weighted vs mean | Phase 0 ablation test |
| 4 | Base stream capacity control | Dimension constraint vs L2 regularization vs information bottleneck | Phase 1 leakage analysis |
| 5 | Continuous factor support scope | P0 (categorical only) vs P1 (include continuous for Causal3DIdent) | After P0 categorical pipeline stabilized |

---

## Blocking Questions

**NONE at specification level. All decisions frozen.**

Implementation may proceed with Phase-0 tasks. GPU experiments remain blocked pending CPU property tests (Gate 2 of Definition of Done).

---

## Implementation-Ready API Contracts

```python
# === Core Types ===
FactorSpec = CategoricalFactorSpec | ContinuousFactorSpec
GraphType = Literal["independent", "dag", "dense_directed", "custom_directed"]
InterventionMode = Literal[
    "observational", "factor_edit", "condition_mask",
    "direct_output_ablation", "edge_ablation", "node_deletion",
    "factor_source_cut", "neural_graph_surgery"
]

# === Intervention API ===
@dataclass
class InterventionSpec:
    mode: InterventionMode
    target_factor_idx: int | None = None
    new_value: int | float | None = None
    edge_indices: list[tuple[int, int]] | None = None
    preserve_factor_values: list[int] | None = None

@dataclass(frozen=True)
class CompiledIntervention:
    effective_factor_values: Tensor[B, K]
    source_gate: Tensor[B, K]
    node_gate: Tensor[B, K]
    output_gate: Tensor[B, K]
    edge_gate: Tensor[B, L, max_E]
    mode: InterventionMode

def compile_intervention(spec: InterventionSpec, config: ModelConfig) -> CompiledIntervention:
    """Single choke-point for intervention logic. Gate tensors only modified here."""

# === Graph API ===
def validate_graph(edges: list[tuple[int, int]], graph_type: str, n_factors: int) -> None:
    """Raises ValueError on cycle (DAG mode), invalid node, duplicate edge, self-loop."""

@dataclass
class PathCertificate:
    factor_idx: int
    source_gate: bool
    trunk_access: bool
    edge_paths: list[tuple[int, int]]
    output_paths: list[int]
    is_fully_cut: bool

    def verify_factor_source_cut(self, spec: GraphSpec, factor_idx: int) -> bool:
        """True iff all paths from factor i to ANY output are cut."""

    def verify(self, intervention: CompiledIntervention) -> bool:
        """Verify certificate against actual compiled intervention."""

# === Model API ===
class ROSTFRG(nn.Module):
    def __init__(self, config: ModelConfig): ...
    def forward(self, x_t: Tensor, t: Tensor, factors: Tensor,
                intervention: CompiledIntervention) -> Tensor:
        """Returns epsilon_hat."""

# === Sampler API ===
@dataclass
class NoiseTrace:
    x_T: Tensor[B, C, H, W]
    sampler: str = "ddim"
    eta: float = 0.0
    steps: int = 50

@dataclass
class NoiseTraceStochastic:
    x_T: Tensor[B, C, H, W]
    sampler: str = "ddpm"
    steps: int = 1000
    seed_grid: list[int]
    rng_algorithm: str = "philox"

def sample_ddim(model: ROSTFRG, factors: Tensor, trace: NoiseTrace,
                n_steps: int) -> Tensor:
    """Deterministic DDIM sampling. Returns x_0."""

def sample_ddpm(model: ROSTFRG, factors: Tensor, trace: NoiseTraceStochastic,
                n_steps: int) -> Tensor:
    """Stochastic DDPM sampling with counter-seed noise. Returns x_0."""

# === Metrics API ===
def compute_target_value_success(oracle_preds: Tensor, target_values: Tensor) -> Tensor[K]:
    """Per-factor accuracy: argmax_k P(decoder(f_i=k | epsilon_hat)) == v."""

def compute_off_target_change(oracle_preds_edit: Tensor,
                               oracle_preds_orig: Tensor) -> Tensor[K, K]:
    """KxK leakage matrix. Diagonal masked. Off-diagonal: P[Oracle_j != Oracle_j_orig]."""

def compute_no_op_change(oracle_preds_noop: Tensor,
                           oracle_preds_orig: Tensor) -> Tensor[K, K]:
    """No-op stability matrix. Should be all-zero for deterministic model."""
```

---

## Phase-0 Implementation Order

Phase-0 tasks are spec-aligned, CPU-only, with property tests. No GPU training.

| # | Task | File | Depends On | Verification |
|---|------|------|------------|-------------|
| 1 | spec types as Python dataclasses | `src/types.py` | None | Import smoke test |
| 2 | Graph validation | `src/graph.py` | types.py | T-01 to T-07 |
| 3 | Intervention compiler | `src/interventions.py` | types.py | T-08 to T-16 |
| 4 | ROST-FRG model skeleton | `src/model.py` | types.py, interventions.py, graph.py | Forward pass shape test |
| 5 | Deterministic DDIM sampler + NoiseTrace | `src/sampling.py` | model.py | T-17 to T-22 |
| 6 | Metric primitives | `src/metrics.py` | types.py | T-34 to T-36 |
| 7 | CPU property tests | `tests/` | all above | Gate 2 checklist |

**Entry condition**: All Phase-0 tasks must be implemented before any GPU experiment.
**Exit condition**: All Gate 2 CPU property tests pass (38 tests, see `11_TEST_AND_VERIFICATION_PLAN.md`).

---

## Spec Version

**3.0**

The v3.0 spec freezes all terminology, architecture, intervention modes, graph types, metrics, dataset splits, statistical plan, and baseline names described in this document and its referenced spec files. Any change to these requires a spec amendment proposal (SAP) recorded in the project ledger.

## Validation

To validate this specification freeze:
```bash
python docs/research_audit/tools/validate_spec.py
```

The validation script checks:
1. All required spec and audit files exist
2. status.yaml fields match reality
3. Document count matches manifest
4. Canonical 8 modes appear in INTERVENTION_SPEC.md
5. No forbidden phrases in any audit document
6. No changes outside docs/research_audit directory
7. AGENTS.md is not in audit inventory

## Unblocking Verdict

- **Specification Freeze**: PASS
- **Implementation Start**: UNBLOCKED_FOR_PHASE_0_ONLY
- **Full Implementation Start**: BLOCKED (pending Phase-0 CPU property tests)
- **GPU Experiments**: BLOCKED_PENDING_CPU_PROPERTY_TESTS
- **Literature Validation**: BLOCKED (API rate-limiting; deferred to PI)
- **Paper Submission**: BLOCKED (pending all gates in `14_DEFINITION_OF_DONE.md`)

---

## Spec File Manifest

All spec files are authoritative from `spec_freeze_base_commit` onward:

| File | Version | Status |
|------|---------|--------|
| `spec/ARCHITECTURE_SPEC.md` | 3.0 | FROZEN |
| `spec/INTERVENTION_SPEC.md` | 3.0 | FROZEN |
| `spec/architecture.yaml` | 3.0 | FROZEN |
| `spec/graphs.yaml` | 3.0 | FROZEN |
| `spec/metrics.yaml` | 3.0 | FROZEN |
| `spec/provenance.yaml` | 3.0 | FROZEN |
| `spec/status.yaml` | 3.0 | FROZEN |
| `CORRECTION_REGISTER_V3.md` | 3.0 | FROZEN |
| This document (`16_SPECIFICATION_FREEZE.md`) | 3.0 | FROZEN |

---

*This document freezes the Factor-Path Diffusion specification. No changes to architecture, intervention semantics, graph types, metric definitions, terminology, baseline names, dataset splits, or statistical plan without a spec amendment proposal (SAP) recorded in the project ledger at `~/research/private-projects/factor-gated-routing/ledgers/`.*
