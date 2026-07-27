import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def safe_load_state_dict(model, state_dict, model_name="model"):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"[{model_name}] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[{model_name}] Unexpected keys: {unexpected}")
    if not missing and not unexpected:
        logger.info(f"[{model_name}] All keys loaded successfully")


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, patch_size: int, image_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FactorEmbed(nn.Module):
    def __init__(self, factor_sizes: tuple, dim: int):
        super().__init__()
        self.n_factors = len(factor_sizes)
        self.embeds = nn.ModuleList([
            nn.Embedding(s, dim) for s in factor_sizes
        ])

    def forward(self, factor_classes):
        out = 0
        for i, embed in enumerate(self.embeds):
            out = out + embed(factor_classes[:, i])
        return out


class AdaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        scale_shift = self.linear(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.norm1 = AdaLN(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = AdaLN(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, cond):
        skip = x
        x = self.norm1(x, cond)
        x = self.attn(x, x, x, need_weights=False)[0]
        x = skip + x
        skip = x
        x = self.norm2(x, cond)
        x = self.mlp(x)
        x = skip + x
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.sa_norm = AdaLN(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ca_norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff_norm = AdaLN(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, cond, parent_states):
        skip = x
        x = self.sa_norm(x, cond)
        x = self.self_attn(x, x, x, need_weights=False)[0]
        x = skip + x
        if parent_states is not None:
            for p_state in parent_states:
                skip2 = x
                x = self.ca_norm1(x)
                x = self.cross_attn(x, p_state, p_state, need_weights=False)[0]
                x = skip2 + x
        skip = x
        x = self.ff_norm(x, cond)
        x = self.mlp(x)
        x = skip + x
        return x
