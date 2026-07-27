"""
Tests for Factor-Path Diffusion core modules.

Covers: types, graph, interventions, model, sampling, metrics, baselines.
"""
import pytest
import torch
import numpy as np

from src.types import (
    InterventionMode, GraphType, GraphSpec, InterventionSpec,
    CategoricalFactorSpec, CompiledIntervention,
)
from src.graph import (
    validate_graph, build_graph_spec, topological_sort,
    transitive_closure, PathCertificate, GraphValidationError,
    make_dense_edges,
)
from src.interventions import (
    compile_intervention, make_observational, make_factor_edit,
    make_factor_source_cut,
)
from src.metrics import (
    compute_target_value_success, compute_target_change_rate,
    compute_off_target_change_matrix, compute_no_op_change,
    compute_source_invariance_error_trajectory,
    compute_oracle_conditional_accuracy,
)
from src.sampling import (
    get_alpha_bars, make_trace_ddim, generate_noise_trace,
)


# ── Types ──

def test_eight_intervention_modes():
    assert len(InterventionMode) == 8

def test_four_graph_types():
    assert len(GraphType) == 4

def test_factor_spec_null_index():
    fs = CategoricalFactorSpec("test", 5)
    assert fs.null_index == 5

def test_intervention_spec_defaults():
    spec = InterventionSpec(mode=InterventionMode.OBSERVATIONAL)
    assert spec.factor_edits == {}


# ── Graph ──

def test_independent_graph():
    gs = build_graph_spec("independent", 3)
    assert len(gs.edges) == 0

def test_dag_valid():
    gs = build_graph_spec("dag", 3, [(0, 1), (1, 2)])
    assert len(gs.edges) == 2

def test_dag_cycle_rejected():
    with pytest.raises(GraphValidationError, match="Cycle"):
        build_graph_spec("dag", 3, [(0, 1), (1, 2), (2, 0)])

def test_dag_invalid_node():
    with pytest.raises(GraphValidationError):
        build_graph_spec("dag", 3, [(0, 5)])

def test_dag_self_loop():
    with pytest.raises(GraphValidationError, match="Self-loop"):
        build_graph_spec("dag", 3, [(0, 0)])

def test_dag_duplicate_edge():
    with pytest.raises(GraphValidationError, match="Duplicate"):
        build_graph_spec("dag", 3, [(0, 1), (0, 1)])

def test_dense_directed():
    gs = build_graph_spec("dense_directed", 3)
    assert len(gs.edges) == 6  # 3*2
    assert (0, 1) in gs.edges
    assert (1, 0) in gs.edges

def test_topological_sort():
    order = topological_sort(((0, 1), (1, 2)), 3)
    assert order == [0, 1, 2]

def test_topological_sort_cycle():
    with pytest.raises(GraphValidationError):
        topological_sort(((1, 0), (0, 1)), 2)

def test_transitive_closure():
    closure = transitive_closure(((0, 1), (1, 2)), 3)
    assert closure[0] == [True, True, True]
    assert closure[2] == [False, False, True]
    assert closure[1] == [False, True, True]

def test_path_certificate():
    pc = PathCertificate(GraphSpec(GraphType.DAG, 3, ((0, 1), (1, 2),)))
    assert pc.get_reachable_outputs(0) == frozenset({0, 1, 2})
    assert pc.get_reachable_outputs(2) == frozenset({2})


# ── Interventions ──

def test_compile_observational():
    specs = [CategoricalFactorSpec(f"f{i}", s) for i, s in enumerate([3, 6, 40])]
    gs = GraphSpec(GraphType.INDEPENDENT, 3)
    c = make_observational([0, 2, 15], gs, specs, 4, 4)
    assert c.source_gate.sum() == 4 * 3
    assert c.output_gate.sum() == 4 * 3

def test_compile_factor_edit():
    specs = [CategoricalFactorSpec("f0", 3)]
    gs = GraphSpec(GraphType.INDEPENDENT, 1)
    c = make_factor_edit(0, 2, [0], gs, specs, 4, 4)
    assert (c.effective_factor_values[:, 0] == 2).all()

def test_compile_factor_source_cut():
    specs = [CategoricalFactorSpec(f"f{i}", s) for i, s in enumerate([3, 6, 40])]
    gs = GraphSpec(GraphType.INDEPENDENT, 3)
    c = make_factor_source_cut(0, [0, 2, 15], gs, specs, 4, 4)
    assert c.source_gate[:, 0].sum() == 0
    assert c.output_gate[:, 0].sum() == 4  # output gate still 1

def test_eight_modes_pairwise_distinct():
    specs = [CategoricalFactorSpec(f"f{i}", 4) for i in range(2)]
    gs = GraphSpec(GraphType.CUSTOM_DIRECTED, 2, ((0, 1),), allow_cycles=True)
    results = {}
    for mode in InterventionMode:
        spec = InterventionSpec(mode=mode)
        if mode == InterventionMode.FACTOR_SOURCE_CUT:
            spec = InterventionSpec(mode=mode, source_cut_factors=frozenset({0}))
        elif mode == InterventionMode.FACTOR_EDIT:
            spec = InterventionSpec(mode=mode, factor_edits={0: 1})
        elif mode == InterventionMode.CONDITION_MASK:
            spec = InterventionSpec(mode=mode, masked_factors=frozenset({0}))
        elif mode == InterventionMode.DIRECT_OUTPUT_ABLATION:
            spec = InterventionSpec(mode=mode, ablated_outputs=frozenset({0}))
        elif mode == InterventionMode.EDGE_ABLATION:
            spec = InterventionSpec(mode=mode, ablated_edges=frozenset({(0, 0, 1)}))
        elif mode == InterventionMode.NODE_DELETION:
            spec = InterventionSpec(mode=mode, deleted_nodes=frozenset({0}))
        elif mode == InterventionMode.NEURAL_GRAPH_SURGERY:
            spec = InterventionSpec(mode=mode, surgery_nodes={0: 1})
        c = compile_intervention(
            spec, gs, specs, factor_values=[0, 0], batch_size=2, num_layers=2,
        )
        # Use all gate tensors + effective values for full distinctness check
        sig = (
            c.source_gate.sum().item(),
            c.output_gate.sum().item(),
            c.node_gate.sum().item(),
            c.edge_gate.sum().item(),
            int(c.effective_factor_values[0, 0].item()),
        )
        results[mode] = sig
    distinct = len(set(results.values()))
    assert distinct == len(results), f"Only {distinct}/{len(results)} distinct mode signatures"


# ── Sampling ──

def test_linear_alpha_bars():
    ab = get_alpha_bars("linear")
    assert ab.shape == (1000,)
    assert ab[0] > 0.999
    assert ab[-1] < 0.001

def test_cosine_alpha_bars():
    ab = get_alpha_bars("cosine")
    assert ab.shape == (1000,)
    assert ab[0] > 0.99
    assert ab[-1] < 0.01

def test_trace_ddim():
    trace = make_trace_ddim(4, 1, 64, 250, device="cpu", seed=42)
    assert trace.x_T.shape == (4, 1, 64, 64)
    assert len(trace.timesteps) == 250  # T=1000, dt=4 → 250 steps

def test_noise_trace_ddpm():
    trace = generate_noise_trace(4, 1, 64, 250, seed=42)
    assert len(trace) == 249  # 250 steps, noise added on 249 transitions


# ── Metrics ──

def test_target_value_success():
    preds = [torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])]
    targets = torch.tensor([[0], [1]])
    tvs = compute_target_value_success(preds, targets)
    assert tvs[0] == 1.0

def test_target_change_rate():
    preds_edit = [torch.tensor([[0.0, 10.0, 0.0]])]
    preds_orig = [torch.tensor([[10.0, 0.0, 0.0]])]
    tcr = compute_target_change_rate(preds_edit, preds_orig)
    assert tcr[0] == 1.0

def test_off_target_change_matrix():
    # K=2: edit factor 0 and factor 1
    all_edits = [
        [torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 10.0]])],  # edit 0
        [torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 10.0]])],  # edit 1
    ]
    orig = [torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 10.0]])]
    otc = compute_off_target_change_matrix(all_edits, orig)
    assert torch.isnan(otc[0, 0])
    assert torch.isnan(otc[1, 1])
    assert otc[0, 1] == 0.0  # f1 didn't change when f0 was edited

def test_no_op_change():
    preds = [torch.tensor([[10.0, 0.0]]), torch.tensor([[10.0, 0.0]])]
    noc = compute_no_op_change(preds, preds)
    assert (noc == 0).all()

def test_source_invariance():
    x1 = torch.ones(2, 3, 8, 8)
    x2 = x1 + 1e-10
    err = compute_source_invariance_error_trajectory(x1, x2)
    assert err["rmse"] < 1e-9

def test_oracle_accuracy():
    preds = [torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([[0.0, 10.0, 0.0]])]
    factors = torch.tensor([[0, 1]])
    per_factor, overall = compute_oracle_conditional_accuracy(preds, factors)
    assert per_factor[0] == 1.0
    assert overall == 1.0


# ── Model (smoke tests) ──

def test_rostfrg_forward():
    from src.model import ROSTFRG
    torch.manual_seed(0)
    model = ROSTFRG(n_factors=3, factor_sizes=(3, 6, 40),
                    trunk_dim=256, branch_dim=128,
                    n_trunk_blocks=4, n_branch_layers=2)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    f = torch.tensor([[0, 2, 15], [1, 3, 30]])
    y = model(x, t, f)
    assert y.shape == (2, 1, 64, 64)

def test_rostfrg_with_intervention():
    from src.model import ROSTFRG
    from src.interventions import make_factor_source_cut
    specs = [CategoricalFactorSpec(f"f{i}", s) for i, s in enumerate([3, 6, 40])]
    gs = GraphSpec(GraphType.INDEPENDENT, 3)
    torch.manual_seed(0)
    model = ROSTFRG(n_factors=3, factor_sizes=(3, 6, 40),
                    trunk_dim=256, branch_dim=128,
                    n_trunk_blocks=4, n_branch_layers=2)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    f = torch.tensor([[0, 2, 15], [1, 3, 30]])
    c = make_factor_source_cut(0, [0, 2, 15], gs, specs, 2, 2)
    y = model(x, t, f, intervention=c)
    assert y.shape == (2, 1, 64, 64)

def test_rostfrg_graph_variants():
    from src.model import ROSTFRG
    torch.manual_seed(0)
    models = [
        ROSTFRG(n_factors=3, factor_sizes=(3, 6, 40), trunk_dim=256, branch_dim=128,
                n_trunk_blocks=4, n_branch_layers=2, graph_type="independent"),
        ROSTFRG(n_factors=3, factor_sizes=(3, 6, 40), trunk_dim=256, branch_dim=128,
                n_trunk_blocks=4, n_branch_layers=2, graph_type="dag", dag_edges=[(0, 1), (1, 2)]),
        ROSTFRG(n_factors=3, factor_sizes=(3, 6, 40), trunk_dim=256, branch_dim=128,
                n_trunk_blocks=4, n_branch_layers=2, graph_type="dense_directed"),
    ]
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    f = torch.tensor([[0, 2, 15], [1, 3, 30]])
    for model in models:
        y = model(x, t, f)
        assert y.shape == (2, 1, 64, 64)

def test_rostfrg_no_base():
    from src.model import ROSTFRG
    model = ROSTFRG(n_factors=3, factor_sizes=(3, 6, 40),
                    trunk_dim=256, branch_dim=128,
                    n_trunk_blocks=4, n_branch_layers=2, use_base=False)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    f = torch.tensor([[0, 2, 15], [1, 3, 30]])
    y = model(x, t, f)
    assert y.shape == (2, 1, 64, 64)


# ── Baselines ──

def test_all_baselines_forward():
    from src.baselines import (
        CanonicalDiT, IndependentStreamDiT, CrossAttnDiT, CFDiT,
    )
    from src.config import ModelConfig
    torch.manual_seed(0)
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40),
                      trunk_dim=256, branch_dim=128,
                      n_trunk_blocks=8, n_branch_layers=2)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    f = torch.tensor([[0, 2, 15], [1, 3, 30]])

    for model_cls in [CanonicalDiT, IndependentStreamDiT, CrossAttnDiT, CFDiT]:
        model = model_cls(cfg)
        model.train()
        y = model(x, t, f)
        assert y.shape == (2, 1, 64, 64)

def test_cfdit_null_token():
    from src.baselines import CFDiT
    from src.config import ModelConfig
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = CFDiT(cfg, p_uncond=0.5)
    null_indices = model.backbone.factor_embed.null_indices
    for i, s in enumerate(cfg.factor_sizes):
        assert null_indices[i] == s
        assert null_indices[i] not in range(s)


# ── Diffusion ──

def test_alpha_bars_invalid():
    with pytest.raises(ValueError):
        get_alpha_bars("invalid")


# ── Oracle ──

def test_oracle_classifier_forward():
    from src.oracle import OracleClassifier
    oracle = OracleClassifier((3, 6, 40), in_channels=1)
    x = torch.randn(2, 1, 64, 64)
    preds = oracle(x)
    assert len(preds) == 3
    assert preds[0].shape == (2, 3)
    assert preds[1].shape == (2, 6)
    assert preds[2].shape == (2, 40)

def test_oracle_gradients():
    from src.oracle import OracleClassifier
    oracle = OracleClassifier((3, 6, 40), in_channels=1)
    oracle.train()
    x = torch.randn(2, 1, 64, 64)
    preds = oracle(x)
    loss = sum(p.sum() for p in preds)
    loss.backward()
    for name, param in oracle.named_parameters():
        assert param.grad is not None, f"{name} no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} non-finite gradient"
