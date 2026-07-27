"""
Model registry and baseline builders for Factor-Path Diffusion.

ROST-FRG is the primary architecture.
Baselines: CanonicalDiT, IndependentStreamDiT, CrossAttnDiT,
           AllToAllFactorStreamDiT, CFDiT.
"""

from src.model import ROSTFRG
from src.config import ModelConfig


def build_rostfrg(config: ModelConfig):
    return ROSTFRG(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        n_factors=config.n_factors,
        factor_sizes=config.factor_sizes,
        trunk_dim=config.trunk_dim,
        branch_dim=config.branch_dim,
        n_trunk_blocks=config.n_trunk_blocks,
        n_branch_layers=config.n_branch_layers,
        n_heads=config.n_heads,
        graph_type=config.graph_type,
        dag_edges=config.dag_edges,
        use_base=config.use_base,
    )


from src.baselines import (
    build_canonical_dit,
    build_independent_stream_dit,
    build_cross_attn_dit,
    build_all_to_all_factor_stream_dit,
    build_cf_dit,
)

MODEL_REGISTRY = {
    "ROSTFRG": build_rostfrg,
    "FGR": build_rostfrg,  # backward compat alias
    "CanonicalDiT": lambda cfg: build_canonical_dit(cfg),
    "SDiT": lambda cfg: build_canonical_dit(cfg),  # backward compat
    "IndependentStreamDiT": lambda cfg: build_independent_stream_dit(cfg),
    "CrossAttnDiT": lambda cfg: build_cross_attn_dit(cfg),
    "AllToAllFactorStreamDiT": lambda cfg: build_all_to_all_factor_stream_dit(cfg),
    "CFDiT": lambda cfg: build_cf_dit(cfg),
    "CF-DiT": lambda cfg: build_cf_dit(cfg),  # backward compat
}
