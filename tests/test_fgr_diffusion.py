import torch
from fgr.diffusion import get_alpha_bars
from fgr.oracle import OracleClassifier


def test_alpha_bars_linear():
    ab = get_alpha_bars("linear")
    assert ab.shape == (1000,)
    assert 0.0 <= ab[-1] <= ab[0] <= 1.0  # monotonically decreasing from ~1 to ~0
    assert ab[0] > 0.999  # nearly clean at t=0
    assert ab[-1] < 0.001  # nearly pure noise at t=999


def test_alpha_bars_cosine():
    ab = get_alpha_bars("cosine")
    assert ab.shape == (1000,)
    assert 0.0 <= ab[-1] <= ab[0] <= 1.0


def test_alpha_bars_invalid():
    import pytest
    with pytest.raises(ValueError):
        get_alpha_bars("invalid")


def test_oracle_classifier_forward():
    oracle = OracleClassifier((3, 6, 40), in_channels=1)
    x = torch.randn(2, 1, 64, 64)
    preds = oracle(x)
    assert len(preds) == 3
    assert preds[0].shape == (2, 3)
    assert preds[1].shape == (2, 6)
    assert preds[2].shape == (2, 40)


def test_oracle_classifier_gradients():
    oracle = OracleClassifier((3, 6, 40), in_channels=1)
    oracle.train()
    x = torch.randn(2, 1, 64, 64)
    preds = oracle(x)
    loss = sum(p.sum() for p in preds)
    loss.backward()
    for name, param in oracle.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
