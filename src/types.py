"""
Canonical types for Factor-Path Diffusion (ROST-FRG).

Single source of truth for InterventionMode, GraphType, FactorSpec,
InterventionSpec, CompiledIntervention, and NoiseTrace.

Matched to docs/research_audit/spec/ at spec v3.0.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Sequence, Union


class GraphType(str, enum.Enum):
    INDEPENDENT = "independent"
    DAG = "dag"
    DENSE_DIRECTED = "dense_directed"
    CUSTOM_DIRECTED = "custom_directed"


class InterventionMode(str, enum.Enum):
    OBSERVATIONAL = "observational"
    FACTOR_EDIT = "factor_edit"
    CONDITION_MASK = "condition_mask"
    DIRECT_OUTPUT_ABLATION = "direct_output_ablation"
    EDGE_ABLATION = "edge_ablation"
    NODE_DELETION = "node_deletion"
    FACTOR_SOURCE_CUT = "factor_source_cut"
    NEURAL_GRAPH_SURGERY = "neural_graph_surgery"


@dataclass
class CategoricalFactorSpec:
    name: str
    cardinality: int
    null_index: int | None = None

    def __post_init__(self):
        if self.null_index is None:
            object.__setattr__(self, "null_index", self.cardinality)


@dataclass
class ContinuousFactorSpec:
    name: str
    dim: int = 1


FactorSpec = Union[CategoricalFactorSpec, ContinuousFactorSpec]


@dataclass(frozen=True)
class GraphSpec:
    graph_type: GraphType
    num_nodes: int
    edges: tuple[tuple[int, int], ...] = ()
    allow_cycles: bool = False
    allow_self_loops: bool = False


@dataclass(frozen=True)
class InterventionSpec:
    mode: InterventionMode
    factor_edits: dict[int, int] = field(default_factory=dict)
    masked_factors: frozenset[int] = frozenset()
    ablated_outputs: frozenset[int] = frozenset()
    ablated_edges: frozenset[tuple[int, int, int]] = frozenset()
    deleted_nodes: frozenset[int] = frozenset()
    source_cut_factors: frozenset[int] = frozenset()
    surgery_nodes: dict[int, int] = field(default_factory=dict)


def compile_intervention(
    spec: InterventionSpec,
    graph_spec: GraphSpec,
    factor_specs: Sequence[FactorSpec],
    *,
    factor_values: Sequence[int],
    batch_size: int,
    num_layers: int,
    device: Any = None,
):
    """Compile a high-level InterventionSpec into low-level CompiledIntervention.

    Full implementation in src/interventions.py.
    """
    raise NotImplementedError(
        "compile_intervention not yet implemented; see src/interventions.py"
    )


@dataclass(frozen=True)
class CompiledIntervention:
    mode: InterventionMode
    effective_factor_values: Any = None
    source_gate: Any = None
    node_gate: Any = None
    output_gate: Any = None
    edge_gate: Any = None


@dataclass(frozen=True)
class DeterministicNoiseTrace:
    x_T: object  # torch.Tensor [B, C, H, W]
    timesteps: tuple[int, ...]
    sampler_config_hash: str


@dataclass(frozen=True)
class CounterNoiseTrace:
    base_seed: int
    sample_ids: tuple[int, ...]
    timesteps: tuple[int, ...]
    rng_algorithm: str = "philox"
    framework_version: str = ""
    device_backend: str = ""


__all__ = [
    "InterventionMode",
    "GraphType",
    "CategoricalFactorSpec",
    "ContinuousFactorSpec",
    "FactorSpec",
    "GraphSpec",
    "InterventionSpec",
    "CompiledIntervention",
    "DeterministicNoiseTrace",
    "CounterNoiseTrace",
    "compile_intervention",
]
