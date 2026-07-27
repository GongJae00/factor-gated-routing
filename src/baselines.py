import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import timestep_embedding, PatchEmbed, FactorEmbed, DiTBlock


def unpatchify(x, patch_size):
    B, N, D = x.shape
    H = W = int(N ** 0.5)
    x = x.permute(0, 2, 1).reshape(B, D, H, W)
    x = F.pixel_shuffle(x, patch_size)
    return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.stream_dim
        in_c = getattr(config, "in_channels", 1)
        n_blocks = config.n_stream_blocks * config.n_factors
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embed = FactorEmbed(config.factor_sizes, dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, config.n_heads) for _ in range(n_blocks)
        ])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = self.t_embed(timestep_embedding(t, self.patch_embed.proj.out_channels))
        f_emb = self.factor_embed(factor_classes)
        cond = t_emb + f_emb
        x = self.patch_embed(x_t) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond)
        x = self.output(x)
        return unpatchify(x, self.patch_size)


class EncDiffBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, cond):
        skip = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, need_weights=False)[0]
        x = skip + x
        skip = x
        x = self.norm2(x)
        cond_exp = cond.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = self.cross_attn(x, cond_exp, cond_exp, need_weights=False)[0]
        x = skip + x
        skip = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = skip + x
        return x


class EncDiffDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.stream_dim
        in_c = getattr(config, "in_channels", 1)
        n_blocks = config.n_stream_blocks * config.n_factors
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embed = FactorEmbed(config.factor_sizes, dim)
        self.blocks = nn.ModuleList([
            EncDiffBlock(dim, config.n_heads) for _ in range(n_blocks)
        ])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = self.t_embed(timestep_embedding(t, self.patch_embed.proj.out_channels))
        f_emb = self.factor_embed(factor_classes)
        cond = t_emb + f_emb
        x = self.patch_embed(x_t) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond)
        x = self.output(x)
        return unpatchify(x, self.patch_size)


class MMDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, n_factors):
        super().__init__()
        self.sa_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ca_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_factors - 1)])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(dim, n_heads, batch_first=True) for _ in range(n_factors - 1)
        ])
        self.ff_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ada_mlp = nn.Linear(dim, dim * 2)

    def forward(self, x, cond, others):
        scale_shift = self.ada_mlp(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        skip = x
        x = self.sa_norm(x)
        x = x * (1 + scale) + shift
        x = self.self_attn(x, x, x, need_weights=False)[0]
        x = skip + x
        for norm, attn, other in zip(self.ca_norms, self.cross_attns, others):
            skip = x
            x = norm(x)
            x = attn(x, other, other, need_weights=False)[0]
            x = skip + x
        skip = x
        x = self.ff_norm(x)
        x = x * (1 + scale) + shift
        x = self.mlp(x)
        x = skip + x
        return x


class MMDiTk(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.stream_dim
        in_c = getattr(config, "in_channels", 1)
        n_factors = config.n_factors
        self.n_factors = n_factors
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embeds = nn.ModuleList([
            nn.Embedding(s, dim) for s in config.factor_sizes
        ])
        self.stream_blocks = nn.ModuleList([
            nn.ModuleList([
                MMDiTBlock(dim, config.n_heads, n_factors)
                for _ in range(config.n_stream_blocks)
            ])
            for _ in range(n_factors)
        ])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(nn.LayerNorm(dim * n_factors), nn.Linear(dim * n_factors, out_dim))
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = self.t_embed(timestep_embedding(t, self.patch_embed.proj.out_channels))
        tokens = self.patch_embed(x_t) + self.pos_embed
        f_embs = [emb(factor_classes[:, i]) for i, emb in enumerate(self.factor_embeds)]
        stream_states = [tokens + f_embs[i].unsqueeze(1) for i in range(self.n_factors)]
        for layer_idx in range(len(self.stream_blocks[0])):
            new_states = []
            for i in range(self.n_factors):
                others = [stream_states[j] for j in range(self.n_factors) if j != i]
                s = self.stream_blocks[i][layer_idx](stream_states[i], t_emb + f_embs[i], others)
                new_states.append(s)
            stream_states = new_states
        out = torch.cat(stream_states, dim=-1)
        out = self.output(out)
        return unpatchify(out, self.patch_size)


class CoInDDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.stream_dim
        in_c = getattr(config, "in_channels", 1)
        n_factors = config.n_factors
        self.n_factors = n_factors
        self.patch_embed = PatchEmbed(in_c, dim, config.patch_size, config.image_size)
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )
        self.factor_embeds = nn.ModuleList([
            nn.Embedding(s, dim) for s in config.factor_sizes
        ])
        self.streams = nn.ModuleList([
            nn.ModuleList([
                DiTBlock(dim, config.n_heads) for _ in range(config.n_stream_blocks)
            ])
            for _ in range(n_factors)
        ])
        out_dim = config.patch_size * config.patch_size * in_c
        self.output = nn.Sequential(nn.LayerNorm(dim * n_factors), nn.Linear(dim * n_factors, out_dim))
        self.patch_size = config.patch_size

    def forward(self, x_t, t, factor_classes, **kwargs):
        t_emb = self.t_embed(timestep_embedding(t, self.patch_embed.proj.out_channels))
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
        out = self.output(out)
        return unpatchify(out, self.patch_size)


class CFDiT(nn.Module):
    def __init__(self, config, p_uncond: float = 0.1):
        super().__init__()
        self.backbone = SingleStreamDiT(config)
        self.p_uncond = p_uncond

    def forward(self, x_t, t, factor_classes, **kwargs):
        if self.training and self.p_uncond > 0:
            drop = torch.rand(factor_classes.shape[0], 1, device=factor_classes.device) < self.p_uncond
            factor_classes = torch.where(drop, torch.zeros_like(factor_classes), factor_classes)
        return self.backbone(x_t, t, factor_classes)


def build_baseline(name, config):
    if name == "SDiT":
        return SingleStreamDiT(config)
    elif name == "EncDiff":
        return EncDiffDiT(config)
    elif name == "MMDiT-k":
        return MMDiTk(config)
    elif name == "CoInD":
        return CoInDDiT(config)
    elif name == "CF-DiT":
        return CFDiT(config)
    else:
        raise ValueError(f"Unknown baseline: {name}")
