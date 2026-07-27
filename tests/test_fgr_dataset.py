import os
import pytest
from src.dataset import DSpritesDataset

DSPRITES = os.environ.get("DSPRITES_PATH", "/tmp/dsprites/dsprites.npz")
requires_dsprites = pytest.mark.skipif(not os.path.exists(DSPRITES), reason="dSprites dataset not found")


@requires_dsprites
def test_dsprites_factor_slice():
    ds = DSpritesDataset(DSPRITES, split="train", seed=42)
    img, factors = ds[0]["image"], ds[0]["factors"]
    assert factors.shape[0] == 3
    assert factors[0].item() in [0, 1, 2]
    assert factors[1].item() in range(6)
    assert factors[2].item() in range(40)
    assert -1.0 <= img.min() <= 1.0
    assert img.shape == (1, 64, 64)


@requires_dsprites
def test_dsprites_split():
    ds_train = DSpritesDataset(DSPRITES, split="train", seed=42)
    ds_test = DSpritesDataset(DSPRITES, split="test", seed=42)
    total = 737280
    assert 600_000 <= len(ds_train) <= 680_000
    assert len(ds_train) + len(ds_test) == total


@requires_dsprites
def test_dsprites_reproducibility():
    ds1 = DSpritesDataset(DSPRITES, seed=42)
    ds2 = DSpritesDataset(DSPRITES, seed=42)
    assert (ds1[0]["factors"] == ds2[0]["factors"]).all().item()


@requires_dsprites
def test_dsprites_image_range():
    ds = DSpritesDataset(DSPRITES, split="train", seed=42)
    for i in range(100):
        assert -1.0 <= ds[i]["image"].min() <= 1.0
        assert -1.0 <= ds[i]["image"].max() <= 1.0
