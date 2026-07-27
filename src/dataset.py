"""
Dataset loaders for dSprites and 3DShapes.

- dSprites: shape(3), scale(6), orientation(40). Position x/y sent to base stream.
- 3DShapes: 6 independent factors, lazy HDF5 access.
- Splits: IID random, held-out pair/triple combinations.
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    """dSprites dataset. Factors: shape(3), scale(6), orientation(40).

    Position x/y are NOT conditioned — handled by base stream as nuisance.
    """

    def __init__(self, data_path: str, split: str = "train",
                 train_frac: float = 0.9, seed: int = 42):
        data = np.load(data_path, allow_pickle=True)
        imgs = data["imgs"]
        latents_classes = data["latents_classes"]
        n = len(imgs)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        split_idx = int(n * train_frac)
        if split == "train":
            idx = perm[:split_idx]
        elif split == "val":
            val_idx = int(n * (train_frac + 0.05))
            idx = perm[split_idx:val_idx]
        else:
            idx = perm[split_idx:]

        self.imgs = torch.from_numpy(imgs[idx])
        self.latents = torch.from_numpy(latents_classes[idx][:, 1:4])  # shape, scale, orientation

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].float().unsqueeze(0) * 2 - 1
        latents = self.latents[idx].long()
        return {"image": img, "factors": latents}


class Shapes3DDataset(Dataset):
    """3DShapes dataset with lazy HDF5 access.

    480,000 images × 64×64×3. Factors are independent (all Cartesian combinations).
    Raw: ~5.49 GB uint8, ~22 GB float32.
    """

    n_factors = 6
    factor_names = [
        "floor_hue", "wall_hue", "object_hue",
        "scale", "shape", "orientation",
    ]
    factor_sizes = [10, 10, 10, 8, 4, 15]

    def __init__(self, data_path: str, split: str = "train",
                 train_frac: float = 0.9, seed: int = 42):
        import h5py
        self.h5_path = data_path

        with h5py.File(data_path, "r") as f:
            labels = f["labels"][:]

        latents_classes = np.zeros_like(labels, dtype=np.int64)
        for i in range(self.n_factors):
            _, inverse = np.unique(labels[:, i], return_inverse=True)
            latents_classes[:, i] = inverse

        n = len(labels)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        split_idx = int(n * train_frac)
        if split == "train":
            idx = perm[:split_idx]
        elif split == "val":
            val_idx = int(n * (train_frac + 0.05))
            idx = perm[split_idx:val_idx]
        else:
            idx = perm[split_idx:]

        self.indices = idx
        self.latents = torch.from_numpy(latents_classes[idx])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import h5py
        real_idx = self.indices[idx]
        with h5py.File(self.h5_path, "r") as f:
            img = f["images"][real_idx]
        img = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1)
        img = img.permute(2, 0, 1)  # HWC → CHW
        return {"image": img, "factors": self.latents[idx]}


class HeldOutPairSplit(Dataset):
    """Wrapper: held-out factor-value pair combinations.

    train on selected (f_i, f_j) pairs, test on excluded pairs.
    Every individual factor value appears in training.
    """

    def __init__(self, base_dataset: Dataset, factor_a: int, factor_b: int,
                 holdout_pairs: list[tuple[int, int]], split: str = "train"):
        self.base = base_dataset
        self.factor_a = factor_a
        self.factor_b = factor_b
        self.holdout_pairs = set(holdout_pairs)

        self.indices = []
        for i in range(len(base_dataset)):
            fa = base_dataset[i]["factors"][factor_a].item()
            fb = base_dataset[i]["factors"][factor_b].item()
            is_holdout = (fa, fb) in self.holdout_pairs
            if split == "train" and not is_holdout:
                self.indices.append(i)
            elif split == "test" and is_holdout:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def make_heldout_pairs(
    factor_sizes: list[int],
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Generate held-out pair combinations."""
    rng = np.random.RandomState(seed)
    all_pairs = [(i, j) for i in range(factor_sizes[0]) for j in range(factor_sizes[1])]
    n_holdout = max(1, int(len(all_pairs) * holdout_fraction))
    holdout_idx = rng.choice(len(all_pairs), n_holdout, replace=False)
    return [all_pairs[i] for i in holdout_idx]
