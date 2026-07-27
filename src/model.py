"""
ROST-FRG: Read-Only Shared Trunk + Factor Residual Graph.

Primary architecture for Factor-Path Diffusion (spec v3.0).

Shared factor-agnostic DiT trunk processes x_t. Factor-specific adapter
branches read trunk outputs (read-only, no write-back). Inter-branch
communication via explicit graph edge messages. Per-branch output heads
with additive aggregation in noise space.

All 8 canonical intervention modes supported via CompiledIntervention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union

from src.types import (
    InterventionMode,
    InterventionSpec,
    GraphSpec,
    GraphType,
    FactorSpec,
    CategoricalFactorSpec,
    CompiledIntervention,
)
from src.graph import build_graph_spec, validate_graph
from src.interventions import compile_intervention
from src.utils import timestep_embedding, PatchEmbed


class AdaLNZero(nn.Module):
    """adaLN-Zero modulation block. Cond should be [B, D] (flat, not expanded)."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * 3)

    def forward(self, x, cond):
        params = self.linear(cond).unsqueeze(1)
        shift, scale, gate = params.chunk(3, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x, gate


class AdaLN(nn.Module):
    """Standard adaLN modulation (for branches). Cond can be [B, D] or [B, 1, D]."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        params = self.linear(cond)
        if params.dim() == 2:
            params = params.unsqueeze(1)
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TrunkBlock(nn.Module):
    """DiT block for shared trunk with adaLN-Zero."""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.norm1 = AdaLNZero(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = AdaLNZero(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, t_emb):
        h, gate1 = self.norm1(x, t_emb)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate1 * h

        h, gate2 = self.norm2(x, t_emb)
        h = self.mlp(h)
        x = x + gate2 * h
        return x


class EdgeMessage(nn.Module):
    """Shared edge message module per layer. Conditioned on (parent,child) edge key."""
    def __init__(self, dim: int, num_factors: int):
        super().__init__()
        self.edge_key_embed = nn.Embedding(num_factors * num_factors, dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, parent_state: torch.Tensor, parent_idx: int, child_idx: int, num_factors: int) -> torch.Tensor:
        key = parent_idx * num_factors + child_idx
        key_emb = self.edge_key_embed(torch.tensor(key, device=parent_state.device))
        key_emb = key_emb.unsqueeze(0).unsqueeze(0).expand(parent_state.shape[0], parent_state.shape[1], -1)
        return self.proj(torch.cat([parent_state, key_emb], dim=-1))


class FactorInit(nn.Module):
    """Initialize branch state from trunk z^(0) + factor encoding."""
    def __init__(self, trunk_dim: int, branch_dim: int, factor_size: int):
        super().__init__()
        self.factor_embed = nn.Embedding(factor_size + 1, branch_dim)  # +1 for null
        self.proj = nn.Linear(trunk_dim, branch_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(branch_dim, branch_dim * 2),
            nn.SiLU(),
            nn.Linear(branch_dim * 2, branch_dim),
        )

    def forward(self, trunk_z0: torch.Tensor, factor_class: torch.Tensor, factor_idx: int, t_emb: torch.Tensor) -> torch.Tensor:
        f_emb = self.factor_embed(factor_class[:, factor_idx])
        trunk_info = self.proj(trunk_z0)
        t_info = self.t_proj(t_emb).unsqueeze(1)
        return trunk_info + f_emb.unsqueeze(1) + t_info


class FactorAdapter(nn.Module):
    """Per-factor branch adapter at one trunk layer."""
    def __init__(self, trunk_dim: int, branch_dim: int, n_heads: int, factor_size: int):
        super().__init__()
        self.factor_embed = nn.Embedding(factor_size + 1, branch_dim)
        self.trunk_read = nn.Sequential(
            nn.Linear(trunk_dim, branch_dim),
            nn.GELU(),
        )
        self.self_attn = nn.MultiheadAttention(branch_dim, n_heads, batch_first=True)
        self.norm_self = AdaLN(branch_dim)
        self.norm_parent = nn.LayerNorm(branch_dim)
        self.parent_proj = nn.Linear(branch_dim, branch_dim)
        self.norm_ff = AdaLN(branch_dim)
        self.mlp = nn.Sequential(
            nn.Linear(branch_dim, branch_dim * 4),
            nn.GELU(),
            nn.Linear(branch_dim * 4, branch_dim),
        )

    def forward(
        self,
        branch_state: torch.Tensor,
        trunk_read: torch.Tensor,
        factor_class: torch.Tensor,
        factor_idx: int,
        t_emb: torch.Tensor,
        parent_aggregate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f_emb = self.factor_embed(factor_class[:, factor_idx])
        cond = t_emb + f_emb

        trunk_info = self.trunk_read(trunk_read)
        h = branch_state + trunk_info

        h = self.norm_self(h, cond)
        h = self.self_attn(h, h, h, need_weights=False)[0]

        if parent_aggregate is not None:
            parent_info = self.parent_proj(self.norm_parent(parent_aggregate))
            h = h + parent_info

        h = self.norm_ff(h, cond)
        h = h + self.mlp(h)

        return h


class FactorHead(nn.Module):
    """Per-factor output head: Norm_i → P_i → noise contribution."""
    def __init__(self, branch_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(branch_dim)
        self.proj = nn.Linear(branch_dim, out_dim)

        # Zero initialization
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class ROSTFRG(nn.Module):
    """Read-Only Shared Trunk + Factor Residual Graph.

    Primary architecture for Factor-Path Diffusion.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 1,
        n_factors: int = 3,
        factor_sizes: tuple[int, ...] = (3, 6, 40),
        trunk_dim: int = 512,
        branch_dim: int = 256,
        n_trunk_blocks: int = 12,
        n_branch_layers: int = 4,
        n_heads: int = 8,
        graph_type: GraphType | str = GraphType.INDEPENDENT,
        dag_edges: Sequence[tuple[int, int]] | None = None,
        use_base: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_factors = n_factors
        self.factor_sizes = factor_sizes
        self.trunk_dim = trunk_dim
        self.branch_dim = branch_dim
        self.n_trunk_blocks = n_trunk_blocks
        self.n_branch_layers = n_branch_layers
        self.n_heads = n_heads
        self.use_base = use_base

        # Graph
        if isinstance(graph_type, str):
            graph_type = GraphType(graph_type)
        self.graph_spec = build_graph_spec(graph_type, n_factors, list(dag_edges) if dag_edges else None)

        out_dim = patch_size * patch_size * in_channels
        n_patches = (image_size // patch_size) ** 2

        # Shared trunk
        self.patch_embed = PatchEmbed(in_channels, trunk_dim, patch_size, image_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, trunk_dim) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim * 4),
            nn.SiLU(),
            nn.Linear(trunk_dim * 4, trunk_dim),
        )
        self.branch_t_embed = nn.Sequential(
            nn.Linear(trunk_dim, branch_dim * 2),
            nn.SiLU(),
            nn.Linear(branch_dim * 2, branch_dim),
        )
        self.trunk_blocks = nn.ModuleList([
            TrunkBlock(trunk_dim, n_heads) for _ in range(n_trunk_blocks)
        ])

        # Factor branches
        self.factor_inits = nn.ModuleList([
            FactorInit(trunk_dim, branch_dim, factor_sizes[i])
            for i in range(n_factors)
        ])

        branch_layers_per = max(1, n_branch_layers // max(1, n_trunk_blocks))
        self.branch_layers = nn.ModuleList([
            nn.ModuleList([
                FactorAdapter(trunk_dim, branch_dim, n_heads, factor_sizes[i])
                for i in range(n_factors)
            ])
            for _ in range(n_branch_layers)
        ])

        # Edge messages
        self.edge_messages = nn.ModuleList([
            EdgeMessage(branch_dim, n_factors)
            for _ in range(n_branch_layers)
        ])

        # Output heads
        if use_base:
            self.base_head = nn.Sequential(
                nn.LayerNorm(trunk_dim),
                nn.Linear(trunk_dim, out_dim),
            )
        else:
            self.base_head = None

        self.factor_heads = nn.ModuleList([
            FactorHead(branch_dim, out_dim) for _ in range(n_factors)
        ])

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        factor_classes: torch.Tensor,
        intervention: Optional[CompiledIntervention] = None,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        device = x_t.device

        # Timestep embedding
        t_emb = timestep_embedding(t, self.trunk_dim)
        t_emb_flat = self.t_embed(t_emb)  # [B, trunk_dim]
        t_branch = self.branch_t_embed(t_emb)  # [B, branch_dim]

        # Patch embed
        z = self.patch_embed(x_t) + self.pos_embed

        # Extract gates from intervention
        src_gate = intervention.source_gate.to(device) if intervention is not None else torch.ones(B, self.n_factors, device=device)
        node_gate = intervention.node_gate.to(device) if intervention is not None else torch.ones(B, self.n_factors, device=device)
        out_gate = intervention.output_gate.to(device) if intervention is not None else torch.ones(B, self.n_factors, device=device)
        edge_gate_all = intervention.edge_gate.to(device) if intervention is not None else torch.ones(B, self.n_branch_layers, self.n_factors, self.n_factors, device=device)

        # Trunk forward
        z0_for_branch = z
        trunk_outputs = [z0_for_branch]
        for l, block in enumerate(self.trunk_blocks):
            z = block(z, t_emb_flat)
            trunk_outputs.append(z)

        # Initialize branch states
        branch_states: list[torch.Tensor] = []
        for i in range(self.n_factors):
            a = self.factor_inits[i](z0_for_branch, factor_classes, i, t_branch)
            branch_states.append(a)

        # Branch layers (synchronous)
        trunk_stride = max(1, self.n_trunk_blocks // max(1, self.n_branch_layers))
        for l in range(self.n_branch_layers):
            trunk_layer_out = trunk_outputs[min((l + 1) * trunk_stride, self.n_trunk_blocks)]

            # Compute edge messages from snapshot
            edge_msgs: dict[tuple[int, int], torch.Tensor] = {}
            for u, v in self.graph_spec.edges:
                if edge_gate_all[0, l, u, v] > 0:
                    msg = self.edge_messages[l](branch_states[u], u, v, self.n_factors)
                    edge_msgs[(u, v)] = msg * edge_gate_all[:, l, u, v].view(B, 1, 1)

            # Aggregate parent messages per child
            parent_agg: dict[int, torch.Tensor] = {}
            for v in range(self.n_factors):
                parents = []
                for u, child in self.graph_spec.edges:
                    if child == v and (u, v) in edge_msgs:
                        parents.append(edge_msgs[(u, v)])
                if parents:
                    parent_agg[v] = torch.stack(parents).mean(0)

            # Update all branches (synchronous from snapshot)
            new_states = []
            for i in range(self.n_factors):
                if node_gate[0, i] < 0.5:
                    new_states.append(branch_states[i])
                    continue
                a = self.branch_layers[l][i](
                    branch_states[i],
                    trunk_layer_out,
                    factor_classes,
                    i,
                    t_branch,
                    parent_agg.get(i),
                )
                new_states.append(a)
            branch_states = new_states

        # Output heads
        epsilon: torch.Tensor | float = 0.0
        if self.base_head is not None:
            epsilon = self.base_head(z)  # type: ignore[assignment]

        for i in range(self.n_factors):
            if out_gate[0, i] > 0:
                eps_i = self.factor_heads[i](branch_states[i])
                gate_val = out_gate[:, i].view(B, 1, 1)
                if isinstance(epsilon, float):
                    epsilon = gate_val * eps_i
                else:
                    epsilon = epsilon + gate_val * eps_i  # type: ignore[operator]

        # Unpatchify
        return unpatchify(epsilon, self.patch_size)


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, N, D = x.shape
    H = W = int(N ** 0.5)
    x = x.permute(0, 2, 1).reshape(B, D, H, W)
    return F.pixel_shuffle(x, patch_size)
