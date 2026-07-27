import torch
import torch.nn as nn


class OracleClassifier(nn.Module):
    def __init__(self, factor_sizes, in_channels=1):
        super().__init__()
        C = in_channels
        self.shared = nn.Sequential(
            nn.Conv2d(C, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, s) for s in factor_sizes
        ])

    def forward(self, x):
        feat = self.shared(x)
        return [h(feat) for h in self.heads]
