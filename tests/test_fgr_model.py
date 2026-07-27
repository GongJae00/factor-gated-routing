import torch
import pytest
from fgr.model import FGRDiT, FGRStream
from fgr.config import ModelConfig


def test_fgr_forward_shape():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = FGRDiT(cfg)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([100, 500])
    factors = torch.tensor([[0, 2, 15], [1, 3, 30]])
    out = model(x, t, factors)
    assert out.shape == (2, 1, 64, 64)


def test_fgr_gating_zero():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = FGRDiT(cfg)
    x = torch.randn(1, 1, 64, 64)
    t = torch.tensor([500])
    factors = torch.tensor([[0, 2, 15]])
    out_all = model(x, t, factors, gates=[1.0, 1.0, 1.0])
    out_zero = model(x, t, factors, gates=[0.0, 0.0, 0.0])
    assert out_all.shape == out_zero.shape
    assert not torch.allclose(out_all, out_zero)


def test_fgr_set_ca_mode():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = FGRDiT(cfg)
    model.set_ca_mode("full")
    assert model._ca_mode == "full"
    model.set_ca_mode("dag")
    assert model._ca_mode == "dag"
    with pytest.raises(ValueError):
        model.set_ca_mode("invalid")


def test_fgr_stream_block_device():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = FGRDiT(cfg)
    model.set_inter_stream_ca(True)
    model.set_inter_stream_ca(False)


def test_fgr_param_count():
    cfg = ModelConfig(n_factors=3, factor_sizes=(3, 6, 40))
    model = FGRDiT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 13_000_000 <= n_params <= 14_000_000  # ~13.24M
