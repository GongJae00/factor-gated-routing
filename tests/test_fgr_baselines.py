import torch
import pytest
from src.config import ModelConfig
from src.baselines import build_baseline, SingleStreamDiT, EncDiffDiT, MMDiTk, CoInDDiT, CFDiT


def test_sdit_forward():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = SingleStreamDiT(cfg)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    factors = torch.tensor([[0, 2, 15], [1, 3, 30]])
    out = model(x, t, factors)
    assert out.shape == (2, 1, 64, 64)


def test_encdiff_forward():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = EncDiffDiT(cfg)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    factors = torch.tensor([[0, 2, 15], [1, 3, 30]])
    out = model(x, t, factors)
    assert out.shape == (2, 1, 64, 64)


def test_mmditk_forward():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = MMDiTk(cfg)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    factors = torch.tensor([[0, 2, 15], [1, 3, 30]])
    out = model(x, t, factors)
    assert out.shape == (2, 1, 64, 64)


def test_coind_forward():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = CoInDDiT(cfg)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    factors = torch.tensor([[0, 2, 15], [1, 3, 30]])
    out = model(x, t, factors)
    assert out.shape == (2, 1, 64, 64)


def test_cfdit_forward():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = CFDiT(cfg, p_uncond=0.1)
    model.train()
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    factors = torch.tensor([[0, 2, 15], [1, 3, 30]])
    out = model(x, t, factors)
    assert out.shape == (2, 1, 64, 64)


def test_baseline_param_counts_differ():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    models = {
        "SDiT": SingleStreamDiT(cfg),
        "EncDiff": EncDiffDiT(cfg),
        "MMDiT-k": MMDiTk(cfg),
        "CoInD": CoInDDiT(cfg),
    }
    params = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    assert params["SDiT"] == pytest.approx(params["CoInD"], rel=0.01)
    assert params["MMDiT-k"] > params["SDiT"]


def test_build_baseline_registry():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    for name in ["SDiT", "EncDiff", "MMDiT-k", "CoInD", "CF-DiT"]:
        model = build_baseline(name, cfg)
        assert model is not None
    with pytest.raises(ValueError):
        build_baseline("UnknownModel", cfg)
