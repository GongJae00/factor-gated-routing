"""
Oracle classifier for Factor-Path Diffusion evaluation.

Multi-head CNN classifier predicting per-factor labels.
Includes training pipeline, validation, confusion matrix, and calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
import os
import argparse
from typing import Sequence


class OracleClassifier(nn.Module):
    """Multi-head CNN classifier for factor prediction."""

    def __init__(self, factor_sizes: Sequence[int], in_channels: int = 1):
        super().__init__()
        C = in_channels
        self.factor_sizes = list(factor_sizes)
        self.shared = nn.Sequential(
            nn.Conv2d(C, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(256, s) for s in factor_sizes
        ])

    def forward(self, x):
        feat = self.shared(x)
        return [h(feat) for h in self.heads]


def train_oracle(
    train_dataset,
    factor_sizes: Sequence[int],
    output_dir: str,
    *,
    in_channels: int = 1,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 20,
    val_frac: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
):
    """Train and validate oracle classifier."""
    torch.manual_seed(seed)

    n_total = len(train_dataset)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(train_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = OracleClassifier(factor_sizes, in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_ckpt = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            factors = batch["factors"].to(device)
            preds = model(img)
            loss = sum(F.cross_entropy(p, factors[:, i]) for i, p in enumerate(preds))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = [0] * len(factor_sizes)
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                factors = batch["factors"].to(device)
                preds = model(img)
                val_total += img.shape[0]
                for i, p in enumerate(preds):
                    val_correct[i] += (p.argmax(1) == factors[:, i]).sum().item()

        val_acc = [c / val_total for c in val_correct]
        mean_acc = np.mean(val_acc)

        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            best_ckpt = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_loader):.4f}, "
              f"val_acc={mean_acc:.4f} ({[f'{a:.3f}' for a in val_acc]})", flush=True)

    # Save best checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "oracle.pt")
    torch.save(best_ckpt, ckpt_path)

    # Save metadata
    meta_path = os.path.join(output_dir, "oracle_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "factor_sizes": list(factor_sizes),
            "in_channels": in_channels,
            "best_val_acc": best_val_acc,
            "per_factor_acc": val_acc,
            "epochs": epochs,
            "seed": seed,
        }, f, indent=2)

    print(f"Oracle saved to {ckpt_path} (val acc: {best_val_acc:.4f})")

    # Load best model
    model.load_state_dict(best_ckpt)

    # Compute confusion matrices on validation set
    model.eval()
    K = len(factor_sizes)
    confusions = [torch.zeros(s, s, dtype=torch.long) for s in factor_sizes]
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            factors = batch["factors"].to(device)
            preds = model(img)
            for i in range(K):
                pred_class = preds[i].argmax(1)
                for b in range(img.shape[0]):
                    confusions[i][factors[b, i].long(), pred_class[b].long()] += 1

    conf_path = os.path.join(output_dir, "oracle_confusion.json")
    conf_data = {}
    for i in range(K):
        cm = confusions[i].numpy()
        acc = cm.diagonal().sum() / cm.sum()
        conf_data[f"factor_{i}"] = {
            "confusion_matrix": cm.tolist(),
            "accuracy": float(acc),
        }
    with open(conf_path, "w") as f:
        json.dump(conf_data, f, indent=2)

    return model, {"best_val_acc": best_val_acc, "per_factor_acc": val_acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dsprites")
    parser.add_argument("--output-dir", type=str, default="output/oracle")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.config import get_data_path
    from src.dataset import DSpritesDataset, Shapes3DDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "dsprites":
        factor_sizes = (3, 6, 40)
        in_channels = 1
        data_path = get_data_path("dsprites")
        dataset = DSpritesDataset(data_path, split="train", seed=args.seed)
    else:
        factor_sizes = (10, 10, 10, 8, 4, 15)
        in_channels = 3
        data_path = get_data_path("3dshapes")
        dataset = Shapes3DDataset(data_path, split="train", seed=args.seed)

    train_oracle(dataset, factor_sizes, args.output_dir,
                 in_channels=in_channels, batch_size=args.batch_size,
                 epochs=args.epochs, device=device, seed=args.seed)


if __name__ == "__main__":
    main()
