import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    image_size: int = 64
    patch_size: int = 4
    in_channels: int = 1
    n_factors: int = 3
    factor_sizes: tuple = (3, 6, 40)
    stream_dim: int = 256
    n_stream_blocks: int = 4
    n_heads: int = 8
    d_parent: Optional[int] = None
    n_time_embed: int = 256
    stream_aggregation: str = "sum"
    use_cross_attn: bool = False
    dag_edges: list = field(default_factory=list)
    use_gating: bool = True
    intervention_gate: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.0
    n_steps: int = 200000
    warmup_steps: int = 5000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    log_every: int = 1000
    save_every: int = 50000
    eval_every: int = 25000
    n_epochs: int = 0
    data_root: str = os.environ.get("DSPRITES_PATH", "")
    output_dir: str = os.environ.get("FGR_OUTPUT_DIR", "output")


@dataclass
class FGRConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_data_path(dataset: str) -> str:
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
