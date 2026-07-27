"""
Intervention compiler for Factor-Path Diffusion.

Translates high-level InterventionSpec into low-level CompiledIntervention
with materialized gate tensors for direct injection into forward pass.

All 8 canonical InterventionMode values from spec v3.0 are supported.
"""

from __future__ import annotations

from typing import Sequence

from src.types import (
    InterventionMode,
    InterventionSpec,
    GraphSpec,
    FactorSpec,
    CategoricalFactorSpec,
    CompiledIntervention,
)


def compile_intervention(
    spec: InterventionSpec,
    graph_spec: GraphSpec,
    factor_specs: Sequence[FactorSpec],
    *,
    factor_values: Sequence[int],
    batch_size: int,
    num_layers: int,
    device: str = "cpu",
) -> CompiledIntervention:
    """Compile high-level InterventionSpec → low-level CompiledIntervention."""

    import torch

    K = graph_spec.num_nodes

    effective = _make_effective_values(spec, factor_values, factor_specs, batch_size, K, device)
    src = torch.ones(batch_size, K, device=device)
    node = torch.ones(batch_size, K, device=device)
    out = torch.ones(batch_size, K, device=device)
    edge = _make_edge_gate(spec, graph_spec, batch_size, num_layers, device)

    mode = spec.mode

    if mode == InterventionMode.OBSERVATIONAL:
        pass

    elif mode == InterventionMode.FACTOR_EDIT:
        pass

    elif mode == InterventionMode.CONDITION_MASK:
        for fi in spec.masked_factors:
            null_idx = _null_index(factor_specs, fi)
            effective[:, fi] = null_idx

    elif mode == InterventionMode.DIRECT_OUTPUT_ABLATION:
        for fi in spec.ablated_outputs:
            out[:, fi] = 0.0

    elif mode == InterventionMode.EDGE_ABLATION:
        for layer, parent, child in spec.ablated_edges:
            if layer < num_layers:
                edge[:, layer, parent, child] = 0.0

    elif mode == InterventionMode.NODE_DELETION:
        for fi in spec.deleted_nodes:
            src[:, fi] = 0.0
            node[:, fi] = 0.0
            out[:, fi] = 0.0
            for layer in range(num_layers):
                for p in range(K):
                    if p != fi:
                        edge[:, layer, p, fi] = 0.0
                        edge[:, layer, fi, p] = 0.0

    elif mode == InterventionMode.FACTOR_SOURCE_CUT:
        for fi in spec.source_cut_factors:
            src[:, fi] = 0.0

    elif mode == InterventionMode.NEURAL_GRAPH_SURGERY:
        for fi, new_val in spec.surgery_nodes.items():
            effective[:, fi] = new_val
            for layer in range(num_layers):
                for p in range(K):
                    if p != fi:
                        edge[:, layer, p, fi] = 0.0

    return CompiledIntervention(
        mode=mode,
        effective_factor_values=effective,
        source_gate=src,
        node_gate=node,
        output_gate=out,
        edge_gate=edge,
    )


def _make_effective_values(
    spec: InterventionSpec,
    factor_values: Sequence[int],
    factor_specs: Sequence[FactorSpec],
    batch_size: int,
    K: int,
    device: str,
):
    import torch
    t = torch.full((batch_size, K), 0, dtype=torch.long, device=device)
    for i in range(K):
        t[:, i] = factor_values[i] if i < len(factor_values) else 0

    if spec.mode == InterventionMode.FACTOR_EDIT:
        for fi, val in spec.factor_edits.items():
            t[:, fi] = val
    elif spec.mode == InterventionMode.NEURAL_GRAPH_SURGERY:
        for fi, val in spec.surgery_nodes.items():
            t[:, fi] = val
    elif spec.mode == InterventionMode.CONDITION_MASK:
        for fi in spec.masked_factors:
            null_idx = _null_index(factor_specs, fi)
            t[:, fi] = null_idx

    return t


def _make_edge_gate(
    spec: InterventionSpec,
    graph_spec: GraphSpec,
    batch_size: int,
    num_layers: int,
    device: str,
):
    import torch
    K = graph_spec.num_nodes
    edge = torch.ones(batch_size, num_layers, K, K, device=device)

    for layer, parent, child in spec.ablated_edges:
        if layer < num_layers:
            edge[:, layer, parent, child] = 0.0

    if spec.mode == InterventionMode.NODE_DELETION:
        for fi in spec.deleted_nodes:
            edge[:, :, :, fi] = 0.0
            edge[:, :, fi, :] = 0.0

    elif spec.mode == InterventionMode.NEURAL_GRAPH_SURGERY:
        for fi in spec.surgery_nodes:
            edge[:, :, :, fi] = 0.0

    return edge


def _null_index(factor_specs: Sequence[FactorSpec], fi: int) -> int:
    fs = factor_specs[fi]
    if isinstance(fs, CategoricalFactorSpec):
        return fs.null_index
    return 0


def make_observational(
    factor_values: Sequence[int],
    graph_spec: GraphSpec,
    factor_specs: Sequence[FactorSpec],
    batch_size: int,
    num_layers: int,
    device: str = "cpu",
) -> CompiledIntervention:
    """Convenience: compile OBSERVATIONAL mode."""
    return compile_intervention(
        InterventionSpec(mode=InterventionMode.OBSERVATIONAL),
        graph_spec,
        factor_specs,
        factor_values=factor_values,
        batch_size=batch_size,
        num_layers=num_layers,
        device=device,
    )


def make_factor_edit(
    factor_idx: int,
    new_value: int,
    factor_values: Sequence[int],
    graph_spec: GraphSpec,
    factor_specs: Sequence[FactorSpec],
    batch_size: int,
    num_layers: int,
    device: str = "cpu",
) -> CompiledIntervention:
    """Convenience: compile FACTOR_EDIT mode for one factor."""
    return compile_intervention(
        InterventionSpec(
            mode=InterventionMode.FACTOR_EDIT,
            factor_edits={factor_idx: new_value},
        ),
        graph_spec,
        factor_specs,
        factor_values=factor_values,
        batch_size=batch_size,
        num_layers=num_layers,
        device=device,
    )


def make_factor_source_cut(
    factor_idx: int,
    factor_values: Sequence[int],
    graph_spec: GraphSpec,
    factor_specs: Sequence[FactorSpec],
    batch_size: int,
    num_layers: int,
    device: str = "cpu",
) -> CompiledIntervention:
    """Convenience: compile FACTOR_SOURCE_CUT mode for one factor."""
    return compile_intervention(
        InterventionSpec(
            mode=InterventionMode.FACTOR_SOURCE_CUT,
            source_cut_factors=frozenset([factor_idx]),
        ),
        graph_spec,
        factor_specs,
        factor_values=factor_values,
        batch_size=batch_size,
        num_layers=num_layers,
        device=device,
    )


__all__ = [
    "compile_intervention",
    "make_observational",
    "make_factor_edit",
    "make_factor_source_cut",
]
