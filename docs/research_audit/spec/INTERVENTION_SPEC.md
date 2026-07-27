# Intervention Specification

## Source of Truth

This file defines the canonical `InterventionMode` enum and intervention semantics. All other audit documents, code, test suites, and paper drafts **must** reference this file and **must not** redefine `InterventionMode` or its semantics independently.

## Canonical 8 Modes

```python
from enum import Enum

class InterventionMode(str, Enum):
    OBSERVATIONAL = "observational"         # baseline: no intervention
    FACTOR_EDIT = "factor_edit"             # change one factor value
    CONDITION_MASK = "condition_mask"       # mask a factor to null
    DIRECT_OUTPUT_ABLATION = "direct_output_ablation"  # zero out a factor's output head
    EDGE_ABLATION = "edge_ablation"         # zero specific edges
    NODE_DELETION = "node_deletion"         # fully remove a factor node
    FACTOR_SOURCE_CUT = "factor_source_cut"  # cut source_gate; node still computes
    NEURAL_GRAPH_SURGERY = "neural_graph_surgery"  # replace factor value + cut incoming edges
```

## Mode Semantics Table

| Mode | factor value | source_gate | node_gate | output_gate | incoming edges | outgoing edges | Theorem / Invariant |
|------|-------------|-------------|-----------|-------------|----------------|----------------|---------------------|
| OBSERVATIONAL | original | 1 | 1 | 1 | preserve | preserve | identity |
| FACTOR_EDIT | v→v' | 1 | 1 | 1 | preserve | preserve | single-factor counterfactual |
| CONDITION_MASK | null_token | 1 | 1 | 1 | preserve | preserve | marginalization proxy |
| DIRECT_OUTPUT_ABLATION | original | 1 | 1 | 0 | preserve | preserve | **no invariance guarantee**; only removes additive contribution |
| EDGE_ABLATION | original | 1 | 1 | 1 | selected=0 | selected=0 | symmetric edge cut |
| NODE_DELETION | original | 0 | 0 | 0 | cut all | cut all | full factor excision |
| FACTOR_SOURCE_CUT | any (irrelevant) | 0 | 1 | 1 | preserve | preserve | **Factor-Source Path Non-Interference** |
| NEURAL_GRAPH_SURGERY | v' | 1 | 1 | 1 | cut incoming | preserve | **NOT** a causal do-operator; graph-modified edit |

### Gate Definitions

- **source_gate**: controls whether factor value enters the branch via the encoder. 0 = no factor-specific information enters the branch. The branch still computes with trunk info and edge messages.
- **node_gate**: controls whether the branch module *itself* is active. 0 = the entire branch is skipped; `a_i^l = a_i^(l-1)` (identity pass-through) and no output contribution.
- **output_gate**: controls whether the factor's output head contributes to `epsilon_hat`. 0 = `epsilon_i` is set to zero vector regardless of branch state.
- **edge_gate**: per-edge scalar `∈ [0,1]`. 0 = that specific edge message is zeroed; no information flows along that directed edge.

## CompiledIntervention

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CompiledIntervention:
    """Fully materialized intervention spec, ready for forward pass injection."""

    effective_factor_values: torch.Tensor  # [B, K] — actual values fed to encoders
    source_gate: torch.Tensor              # [B, K] — ∈ {0,1}
    node_gate: torch.Tensor                # [B, K] — ∈ {0,1}
    output_gate: torch.Tensor              # [B, K] — ∈ {0,1}
    edge_gate: torch.Tensor                # [B, L, max_E] — ∈ {0,1}
    mode: InterventionMode
```

### Compiler Function

```python
def compile_intervention(
    spec: InterventionSpec,
    model_config: ModelConfig,
) -> CompiledIntervention:
    """
    Convert a high-level InterventionSpec into a fully materialized
    CompiledIntervention tensor dict.

    The compiler is the single choke-point for all intervention logic.
    No other code path may modify gate tensors during the forward pass.
    """
    ...
```

### InterventionSpec

```python
@dataclass
class InterventionSpec:
    mode: InterventionMode
    target_factor_idx: int | None = None   # K-index of the intervened factor
    new_value: int | float | None = None    # new factor value (for edits)
    edge_indices: list[tuple[int, int]] | None = None  # edges to ablate
    preserve_factor_values: list[int] | None = None     # factors to keep at original
```

## Pairwise Distinctness

All 8 modes produce distinct `CompiledIntervention` tuples. Proof by counterexample enumeration:

| Pair | Distinctive Field |
|------|-------------------|
| OBSERVATIONAL ≠ FACTOR_EDIT | `effective_factor_values` differs (v' ≠ v) |
| OBSERVATIONAL ≠ CONDITION_MASK | `effective_factor_values` differs (null ≠ v) |
| OBSERVATIONAL ≠ DIRECT_OUTPUT_ABLATION | `output_gate` differs (0 ≠ 1) |
| OBSERVATIONAL ≠ FACTOR_SOURCE_CUT | `source_gate` differs (0 ≠ 1) |
| OBSERVATIONAL ≠ NODE_DELETION | `node_gate` differs (0 ≠ 1) |
| OBSERVATIONAL ≠ EDGE_ABLATION | `edge_gate` differs (some entries 0 ≠ 1) |
| OBSERVATIONAL ≠ NEURAL_GRAPH_SURGERY | `effective_factor_values` differs AND `edge_gate` differs |
| FACTOR_EDIT ≠ CONDITION_MASK | `effective_factor_values` differs (v' ≠ null) |
| FACTOR_EDIT ≠ DIRECT_OUTPUT_ABLATION | `output_gate` differs |
| FACTOR_EDIT ≠ FACTOR_SOURCE_CUT | `source_gate` differs |
| FACTOR_EDIT ≠ NODE_DELETION | `node_gate` AND `source_gate` differ |
| FACTOR_EDIT ≠ NEURAL_GRAPH_SURGERY | `edge_gate` differs (incoming cut) |
| CONDITION_MASK ≠ DIRECT_OUTPUT_ABLATION | `output_gate` differs |
| CONDITION_MASK ≠ FACTOR_SOURCE_CUT | `source_gate` differs |
| CONDITION_MASK ≠ NODE_DELETION | `node_gate` AND `source_gate` differ |
| CONDITION_MASK ≠ NEURAL_GRAPH_SURGERY | `edge_gate` differs (incoming cut) |
| DIRECT_OUTPUT_ABLATION ≠ FACTOR_SOURCE_CUT | `source_gate` vs `output_gate` differ (different gates) |
| DIRECT_OUTPUT_ABLATION ≠ NODE_DELETION | `node_gate` AND `source_gate` differ additionally |
| DIRECT_OUTPUT_ABLATION ≠ EDGE_ABLATION | `output_gate` vs `edge_gate` differ (different gates) |
| DIRECT_OUTPUT_ABLATION ≠ NEURAL_GRAPH_SURGERY | `effective_factor_values` AND `edge_gate` differ |
| FACTOR_SOURCE_CUT ≠ NODE_DELETION | `node_gate` differs (source_cut keeps node active) |
| FACTOR_SOURCE_CUT ≠ DIRECT_OUTPUT_ABLATION | `source_gate` vs `output_gate` differ |
| FACTOR_SOURCE_CUT ≠ EDGE_ABLATION | `source_gate` vs `edge_gate` differ |
| FACTOR_SOURCE_CUT ≠ NEURAL_GRAPH_SURGERY | `edge_gate` differs (source_cut preserves edges) |
| NODE_DELETION ≠ EDGE_ABLATION | `node_gate` AND `source_gate` differ from `edge_gate` |
| NODE_DELETION ≠ NEURAL_GRAPH_SURGERY | `node_gate` differs (neural_graph_surgery keeps node active) |
| EDGE_ABLATION ≠ NEURAL_GRAPH_SURGERY | `effective_factor_values` differs (neural_graph_surgery changes value) |

**Conclusion**: All `C(8,2) = 28` unordered pairs are distinct. The 8-mode set is pairwise distinguishable.

## Stale Names

The following names are **deprecated** and must not appear in new code, documentation, or paper drafts:

| Stale Name | Canonical Replacement | Reason |
|------------|----------------------|--------|
| `path_ablation` | `FACTOR_SOURCE_CUT` or `DIRECT_OUTPUT_ABLATION` | ambiguous; conflates source cut with output removal |
| `full_source_cut` | `FACTOR_SOURCE_CUT` | redundant qualifier; source cut is always full |
| `graph_surgery` | `NEURAL_GRAPH_SURGERY` | missing "neural" qualifier; can be confused with graph edits |
| `output_gate_only` | `FACTOR_SOURCE_CUT` | wrong gate for invariance testing |
| `do_like` / `do-like` | `NEURAL_GRAPH_SURGERY` | no causal claim; "do" terminology is misleading |
| `drop_factor` | `NODE_DELETION` | ambiguous; "drop" can mean output gate or node gate |
| `zero_out` | `DIRECT_OUTPUT_ABLATION` | ambiguous; "zero out" can mean many things |
| `intervene` | specify the exact mode | too generic |

## Invariance Theorem: Factor-Source Path Non-Interference

**Statement**: For any factor `i`, setting `source_gate_i = 0` while keeping all other gates at their OBSERVATIONAL values guarantees that the model output `epsilon_hat` is independent of `f_i` (the true factor value). Formally:

```
forall f_i, f_i': epsilon_hat(source_gate_i=0, f_i) = epsilon_hat(source_gate_i=0, f_i')
```

**Proof Sketch**: With `source_gate_i = 0`, the encoded factor value `Enc_i(f_i)` is multiplied by zero and never enters the computation graph. Since branch-to-trunk write is forbidden and the branch output is only a function of (trunk, previous state, factor_source, parent_message), the factor source's value cannot influence `a_i^l` or `epsilon_i` through any path. The PathCertificate guarantees no alternative paths exist.

**Scope**: This theorem holds for `FACTOR_SOURCE_CUT` mode. It does NOT hold for `DIRECT_OUTPUT_ABLATION` (the factor source still influences the branch state and edge messages to other factors, even if its output head is zeroed).

---

*This document is the single source of truth for all intervention semantics. Any discrepancy between this document and code, tests, or paper must be resolved in favor of this document (and the code must be fixed).*
