import torch
import torch.nn as nn

from src.utils import timestep_embedding, PatchEmbed, FactorEmbed, DiTBlock, CrossAttnBlock
from src.baselines import unpatchify


class FGRStream(nn.Module):
    def __init__(self, dim: int, n_blocks: int, n_heads: int,
                 factor_idx: int, factor_size: int,
                 use_cross_attn: bool = False):
        super().__init__()
        self.factor_idx = factor_idx
        self.factor_embed = nn.Embedding(factor_size, dim)
        self.use_cross_attn = use_cross_attn
        block_cls = CrossAttnBlock if use_cross_attn else DiTBlock
        self.blocks = nn.ModuleList([
            block_cls(dim, n_heads) for _ in range(n_blocks)
        ])

    def forward(self, tokens, t_emb, factor_class, parent_states=None, gate=1.0):
        f_emb = self.factor_embed(factor_class[:, self.factor_idx])
        cond = t_emb + f_emb
        x = tokens + f_emb.unsqueeze(1)
        for block in self.blocks:
            if isinstance(block, CrossAttnBlock):
                x = block(x, cond, parent_states)
            else:
                x = block(x, cond)
        x = x * gate
        return x

    def set_inter_stream_ca(self, enabled: bool):
        if self.use_cross_attn and not enabled:
            for i, block in enumerate(self.blocks):
                if isinstance(block, CrossAttnBlock):
                    new_block = DiTBlock(block.dim, block.n_heads).to(
                        next(block.parameters()).device)
                    self.blocks[i] = new_block
        elif not self.use_cross_attn and enabled:
            for i, block in enumerate(self.blocks):
                if not isinstance(block, CrossAttnBlock):
                    new_block = CrossAttnBlock(block.dim, block.n_heads).to(
                        next(block.parameters()).device)
                    self.blocks[i] = new_block


class FGRDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        dim = config.stream_dim
        in_c = getattr(config, "in_channels", 1)
        self.patch_embed = PatchEmbed(
            in_channels=in_c, out_dim=dim,
            patch_size=config.patch_size,
            image_size=config.image_size,
        )
        n_tokens = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        self.streams = nn.ModuleList([
            FGRStream(
                dim=dim,
                n_blocks=config.n_stream_blocks,
                n_heads=config.n_heads,
                factor_idx=i,
                factor_size=config.factor_sizes[i],
                use_cross_attn=config.use_cross_attn,
            ) for i in range(config.n_factors)
        ])

        self.n_factors = config.n_factors
        self.factor_names = config.factor_sizes
        self.use_gating = config.use_gating
        self._ca_mode = "dag"

        out_dim = config.patch_size * config.patch_size * in_c
        self.output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )

    def set_inter_stream_ca(self, enabled: bool):
        for stream in self.streams:
            stream.set_inter_stream_ca(enabled)

    def set_ca_mode(self, mode: str = "dag"):
        if mode == "full":
            self._ca_mode = "full"
        elif mode == "dag":
            self._ca_mode = "dag"
        else:
            raise ValueError(f"Unknown ca_mode: {mode}")

    def forward(self, x_t, t, factor_classes, gates=None):
        B = x_t.shape[0]
        t_emb = timestep_embedding(t, self.cfg.stream_dim)
        t_emb = self.t_embed(t_emb)
        tokens = self.patch_embed(x_t)
        tokens = tokens + self.pos_embed

        if gates is None:
            gates = [1.0] * self.n_factors

        dag = self.cfg.dag_edges
        ca_mode = getattr(self, "_ca_mode", "dag")
        stream_outputs = []
        for i, stream in enumerate(self.streams):
            parent_states = None
            if ca_mode == "full" and i > 0:
                parent_states = list(stream_outputs)
            elif dag:
                parent_indices = [e[0] for e in dag if e[1] == i]
                if parent_indices:
                    parent_states = [stream_outputs[pi] for pi in parent_indices if pi < len(stream_outputs)]
                    parent_states = parent_states or None
            gate_val = gates[i] if self.use_gating else 1.0
            so = stream(tokens, t_emb, factor_classes, parent_states, gate_val)
            stream_outputs.append(so)

        if self.cfg.stream_aggregation == "sum":
            out = sum(stream_outputs)
        elif self.cfg.stream_aggregation == "mean":
            out = sum(stream_outputs) / self.n_factors
        else:
            raise ValueError(f"Unknown aggregation: {self.cfg.stream_aggregation}")

        out = self.output_head(out)
        return unpatchify(out, self.cfg.patch_size)
