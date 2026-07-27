"""
Baseline models for Factor-Path Diffusion.

All baselines use the same trunk dimension and adaLN-Zero blocks.
Models:
- CanonicalDiT: Single-stream DiT with adaLN-Zero, sum of factor embeddings
- IndependentStreamDiT: K independent DiT streams, concatenated output
- CrossAttnDiT: Single-stream DiT with cross-attention conditioning
- AllToAllFactorStreamDiT: K streams with all-to-all synchronous cross-attention
- CFDiT: CanonicalDiT with classifier-free guidance (per-factor null tokens)
"""

import torch
import torch.nn as nn

from src.utils import timestep_embedding, PatchEmbed


def unpatchify(x, patch_size):
    B, N, D = x.shape
    H = W = int(N ** 0.5)
    x = x.permute(0, 2, 1).reshape(B, D, H, W)
    return nn.functional.pixel_shuffle(x, patch_size)


class AdaLNZeroBaseline(nn.Module):
    """adaLN-Zero for baseline DiT blocks."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * 6)

    def forward(self, x, cond):
        params = self.linear(cond).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        return x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


class DiTBlockZero(nn.Module):
    """DiT block with adaLN-Zero initialization."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.adaln = AdaLNZeroBaseline(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # Zero-init the MLP output and attention output projection
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        x_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, cond)
        h = x_norm * (1 + scale_msa) + shift_msa
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_msa * h

        x_norm2 = x * (1 + scale_mlp) + shift_mlp
        h = self.mlp(x_norm2)
        x = x + gate_mlp * h
        return x


class FactorEmbedSum(nn.Module):
    """Sum factor embeddings for single-stream models. +1 for null token per factor."""
    def __init__(self, factor_sizes: tuple[int, ...], dim: int):
        super().__init__()
        self.factor_sizes = factor_sizes
        self.embeds = nn.ModuleList([nn.Embedding(s + 1, dim) for s in factor_sizes])  # +1 for null

    @property
    def null_indices(self) -> list[int]:
        return [s for s in self.factor_sizes]  # null = last index = factor_size

    def forward(self, factor_classes):
        out = 0
        for i, emb in enumerate(self.embeds):
            out = out + emb(factor_classes[:, i])
        return out


class CanonicalDiT(nn.Module):
    """Single-stream DiT with adaLN-Zero, factor embeddings summed."""
    def __init__(self, config):
        super().__init__()
        dim = config.trunk_dim
        in_c = config.in_channels
        n_blocks = max(config.n_trunk_blocks, config.n_factors * 4)
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embed = FactorEmbedSum(config.factor_sizes, dim)
        self.blocks = nn.ModuleList([DiTBlockZero(dim, config.n_heads) for _ in range(n_blocks)])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = timestep_embedding(t, self.pos_embed.shape[-1])
        t_emb = self.t_embed(t_emb)
        f_emb = self.factor_embed(factor_classes)
        cond = t_emb + f_emb
        x = self.patch_embed(x_t) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond)
        return unpatchify(self.output(x), self.patch_size)


class IndependentStreamDiT(nn.Module):
    """K independent DiT streams, concatenated output. Formerly called CoInDDiT."""
    def __init__(self, config):
        super().__init__()
        dim = config.branch_dim
        in_c = config.in_channels
        n_factors = config.n_factors
        self.n_factors = n_factors
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embeds = nn.ModuleList([nn.Embedding(s, dim) for s in config.factor_sizes])
        self.streams = nn.ModuleList([
            nn.ModuleList([DiTBlockZero(dim, config.n_heads) for _ in range(config.n_branch_layers)])
            for _ in range(n_factors)
        ])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(
            nn.LayerNorm(dim * n_factors),
            nn.Linear(dim * n_factors, out_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = timestep_embedding(t, self.pos_embed.shape[-1])
        t_emb = self.t_embed(t_emb)
        tokens = self.patch_embed(x_t) + self.pos_embed
        stream_outputs = []
        for i in range(self.n_factors):
            f_emb = self.factor_embeds[i](factor_classes[:, i])
            cond = t_emb + f_emb
            x = tokens + f_emb.unsqueeze(1)
            for block in self.streams[i]:
                x = block(x, cond)
            stream_outputs.append(x)
        out = torch.cat(stream_outputs, dim=-1)
        return unpatchify(self.output(out), self.patch_size)


class CrossAttnBlock(nn.Module):
    """DiT block with cross-attention conditioning. Formerly called EncDiffDiT block."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.adaln = AdaLNZeroBaseline(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond, cond_tokens):
        x_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, cond)
        h = x_norm * (1 + scale_msa) + shift_msa
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_msa * h

        if cond_tokens is not None:
            h = self.norm_ca(x)
            h = self.cross_attn(h, cond_tokens, cond_tokens, need_weights=False)[0]
            x = x + h

        h = x * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h
        return x


class CrossAttnDiT(nn.Module):
    """Single-stream DiT with per-factor concept-token cross-attention. Formerly EncDiffDiT."""
    def __init__(self, config):
        super().__init__()
        dim = config.trunk_dim
        in_c = config.in_channels
        n_blocks = max(config.n_trunk_blocks, config.n_factors * 4)
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embeds = nn.ModuleList([nn.Embedding(s, dim) for s in config.factor_sizes])
        self.blocks = nn.ModuleList([CrossAttnBlock(dim, config.n_heads) for _ in range(n_blocks)])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = timestep_embedding(t, self.pos_embed.shape[-1])
        t_emb = self.t_embed(t_emb)
        # Per-factor concept tokens
        cond_tokens = torch.stack([emb(factor_classes[:, i]) for i, emb in enumerate(self.factor_embeds)], dim=1)
        x = self.patch_embed(x_t) + self.pos_embed
        for block in self.blocks:
            x = block(x, t_emb, cond_tokens)
        return unpatchify(self.output(x), self.patch_size)


class AllToAllFactorStreamDiT(nn.Module):
    """K streams with all-to-all synchronous cross-attention. Formerly MMDiT-k."""
    def __init__(self, config):
        super().__init__()
        dim = config.branch_dim
        in_c = config.in_channels
        n_factors = config.n_factors
        self.n_factors = n_factors
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embeds = nn.ModuleList([nn.Embedding(s, dim) for s in config.factor_sizes])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, config.n_heads, batch_first=True)
            for _ in range(n_factors - 1)
        ])
        self.stream_blocks = nn.ModuleList([
            nn.ModuleList([DiTBlockZero(dim, config.n_heads) for _ in range(config.n_branch_layers)])
            for _ in range(n_factors)
        ])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(
            nn.LayerNorm(dim * n_factors),
            nn.Linear(dim * n_factors, out_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = timestep_embedding(t, self.pos_embed.shape[-1])
        t_emb = self.t_embed(t_emb)
        tokens = self.patch_embed(x_t) + self.pos_embed
        f_embs = [emb(factor_classes[:, i]) for i, emb in enumerate(self.factor_embeds)]
        stream_states = [tokens + f_embs[i].unsqueeze(1) for i in range(self.n_factors)]

        for layer_idx in range(len(self.stream_blocks[0])):
            new_states = []
            for i in range(self.n_factors):
                s = self.stream_blocks[i][layer_idx](stream_states[i], t_emb + f_embs[i])
                # Cross-attend to other streams
                others = [stream_states[j] for j in range(self.n_factors) if j != i]
                attn_idx = 0
                for oj, other in enumerate(range(self.n_factors)):
                    if other == i:
                        continue
                    if attn_idx < len(self.cross_attns):
                        s = s + self.cross_attns[attn_idx](s, others[oj], others[oj], need_weights=False)[0]
                    attn_idx += 1
                new_states.append(s)
            stream_states = new_states

        out = torch.cat(stream_states, dim=-1)
        return unpatchify(self.output(out), self.patch_size)


class CFDiT(nn.Module):
    """CanonicalDiT with classifier-free guidance. Per-factor dedicated null tokens."""
    def __init__(self, config, p_uncond: float = 0.1):
        super().__init__()
        self.backbone = CanonicalDiT(config)
        self.p_uncond = p_uncond
        self.n_factors = config.n_factors

    def forward(self, x_t, t, factor_classes, **kwargs):
        if self.training and self.p_uncond > 0:
            for i in range(self.n_factors):
                drop_i = torch.rand(factor_classes.shape[0], device=factor_classes.device) < self.p_uncond
                null_idx = self.backbone.factor_embed.null_indices[i]
                factor_classes[drop_i, i] = null_idx
        return self.backbone(x_t, t, factor_classes)


# Build functions for registry
def build_canonical_dit(config):
    return CanonicalDiT(config)


def build_independent_stream_dit(config):
    return IndependentStreamDiT(config)


def build_cross_attn_dit(config):
    return CrossAttnDiT(config)


def build_all_to_all_factor_stream_dit(config):
    return AllToAllFactorStreamDiT(config)


def build_cf_dit(config, p_uncond=0.1):
    return CFDiT(config, p_uncond=p_uncond)
