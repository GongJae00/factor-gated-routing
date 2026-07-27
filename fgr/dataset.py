import numpy as np
import torch
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train", train_frac: float = 0.9, seed: int = 42):
        data = np.load(data_path, allow_pickle=True)
        imgs = data["imgs"]
        latents_classes = data["latents_classes"]
        n = len(imgs)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        split_idx = int(n * train_frac)
        if split == "train":
            idx = perm[:split_idx]
        else:
            idx = perm[split_idx:]
        self.imgs = torch.from_numpy(imgs[idx])
        self.latents = torch.from_numpy(latents_classes[idx][:, 1:4])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].float().unsqueeze(0) * 2 - 1
        latents = self.latents[idx].long()
        return {"image": img, "factors": latents}


class Shapes3DDataset(Dataset):
    n_factors = 6
    factor_names = [
        "floor_hue", "wall_hue", "object_hue",
        "scale", "shape", "orientation",
    ]
    factor_sizes = [10, 10, 10, 8, 4, 15]

    def __init__(self, data_path: str, split: str = "train", train_frac: float = 0.9, seed: int = 42):
        import h5py
        with h5py.File(data_path, "r") as f:
            images = f["images"][:]
            labels = f["labels"][:]

        latents_classes = np.zeros_like(labels, dtype=np.int64)
        for i in range(self.n_factors):
            _, inverse = np.unique(labels[:, i], return_inverse=True)
            latents_classes[:, i] = inverse

        images = images.astype(np.float32) / 255.0 * 2 - 1
        images = images.transpose(0, 3, 1, 2)

        n = len(images)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        split_idx = int(n * train_frac)
        if split == "train":
            idx = perm[:split_idx]
        else:
            idx = perm[split_idx:]

        self.images = torch.from_numpy(images[idx])
        self.latents = torch.from_numpy(latents_classes[idx])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return {"image": self.images[idx], "factors": self.latents[idx]}
