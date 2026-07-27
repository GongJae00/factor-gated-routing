"""
Configuration for Factor-Path Diffusion.

YAML-based config with environment variable resolution.
Lazy-loading of dataset paths (no import-time enforcement).
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence
import json


@dataclass
class ModelConfig:
    image_size: int = 64
    patch_size: int = 4
    in_channels: int = 1
    n_factors: int = 3
    factor_sizes: tuple[int, ...] = (3, 6, 40)
    factor_names: tuple[str, ...] = ("shape", "scale", "orientation")
    trunk_dim: int = 512
    branch_dim: int = 256
    n_trunk_blocks: int = 12
    n_branch_layers: int = 4
    n_heads: int = 8
    graph_type: str = "independent"
    dag_edges: list[tuple[int, int]] = field(default_factory=list)
    use_base: bool = True
    use_gating: bool = True
    schedule: str = "cosine"
    diffusion_steps: int = 1000


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-6
    n_steps: int = 400000
    warmup_steps: int = 5000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    log_every: int = 5000
    ckpt_every: int = 50000
    num_workers: int = 2
    prefetch_factor: int = 2
    seed: int = 42
    output_dir: str = "output"
    mixed_precision: str = "fp16"


def load_config(path: str) -> dict:
    """Load config from YAML or JSON file."""
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for .yaml configs: pip install pyyaml")
    elif path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown config format: {path}")


def build_model_config(cfg: dict) -> ModelConfig:
    """Build ModelConfig from dict (from YAML or inline)."""
    return ModelConfig(
        image_size=cfg.get("image_size", 64),
        patch_size=cfg.get("patch_size", 4),
        in_channels=cfg.get("in_channels", 1),
        n_factors=cfg.get("n_factors", 3),
        factor_sizes=tuple(cfg.get("factor_sizes", (3, 6, 40))),
        factor_names=tuple(cfg.get("factor_names", [])),
        trunk_dim=cfg.get("trunk_dim", 512),
        branch_dim=cfg.get("branch_dim", 256),
        n_trunk_blocks=cfg.get("n_trunk_blocks", 12),
        n_branch_layers=cfg.get("n_branch_layers", 4),
        n_heads=cfg.get("n_heads", 8),
        graph_type=cfg.get("graph_type", "independent"),
        dag_edges=cfg.get("dag_edges", []),
        use_base=cfg.get("use_base", True),
        use_gating=cfg.get("use_gating", True),
        schedule=cfg.get("schedule", "cosine"),
        diffusion_steps=cfg.get("diffusion_steps", 1000),
    )


def model_config_to_dict(cfg: ModelConfig) -> dict:
    """Serialize ModelConfig to dict for checkpoint storage."""
    return {
        "image_size": cfg.image_size,
        "patch_size": cfg.patch_size,
        "in_channels": cfg.in_channels,
        "n_factors": cfg.n_factors,
        "factor_sizes": list(cfg.factor_sizes),
        "factor_names": list(cfg.factor_names),
        "trunk_dim": cfg.trunk_dim,
        "branch_dim": cfg.branch_dim,
        "n_trunk_blocks": cfg.n_trunk_blocks,
        "n_branch_layers": cfg.n_branch_layers,
        "n_heads": cfg.n_heads,
        "graph_type": cfg.graph_type,
        "dag_edges": cfg.dag_edges,
        "use_base": cfg.use_base,
        "use_gating": cfg.use_gating,
        "schedule": cfg.schedule,
        "diffusion_steps": cfg.diffusion_steps,
    }


def get_data_path(dataset: str) -> str:
    """Lazy resolution of dataset path from env var."""
    env_map = {
        "dsprites": "DSPRITES_PATH",
        "3dshapes": "SHAPES3D_PATH",
    }
    path = os.environ.get(env_map.get(dataset, ""))
    if path:
        return path
    raise RuntimeError(
        f"Environment variable {env_map[dataset]} is not set. "
        f"Set it to the {dataset} dataset path."
    )


def get_output_dir(tag: str) -> str:
    base = os.environ.get("FGR_OUTPUT_DIR", "output")
    return os.path.join(base, tag)


def get_oracle_path() -> str:
    return os.environ.get("FGR_ORACLE_PATH", "output/oracle.pt")


DSPRITES_CFG = dict(
    image_size=64, patch_size=4, in_channels=1,
    n_factors=3, factor_sizes=(3, 6, 40),
    factor_names=("shape", "scale", "orientation"),
    trunk_dim=512, branch_dim=256,
    n_trunk_blocks=12, n_branch_layers=4, n_heads=8,
    graph_type="independent", dag_edges=[],
    use_base=True, schedule="cosine",
)

SHAPES3D_CFG = dict(
    image_size=64, patch_size=4, in_channels=3,
    n_factors=6, factor_sizes=(10, 10, 10, 8, 4, 15),
    factor_names=("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"),
    trunk_dim=512, branch_dim=256,
    n_trunk_blocks=24, n_branch_layers=6, n_heads=8,
    graph_type="independent", dag_edges=[],
    use_base=True, schedule="cosine",
)
