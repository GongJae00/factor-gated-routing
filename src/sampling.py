"""
DDPM/DDIM sampler with NoiseTrace for paired-noise evaluation.

Supports:
- deterministic DDIM (η=0) for path invariance verification
- stochastic DDPM with counter-seed NoiseTrace for paired coupling
- shared-noise reproducibility
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

from src.types import DeterministicNoiseTrace, CompiledIntervention

T_1000 = 1000


def get_alpha_bars(schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, T_1000)
    elif schedule == "cosine":
        t = torch.linspace(0, 1, T_1000)
        return (torch.cos(t * math.pi / 2) ** 2).clamp(min=1e-5)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def make_trace_ddim(
    batch_size: int,
    in_channels: int,
    image_size: int,
    n_steps: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> DeterministicNoiseTrace:
    """Generate DeterministicNoiseTrace with explicit x_T for DDIM η=0."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x_T = torch.randn(batch_size, in_channels, image_size, image_size, generator=g)

    dt = T_1000 // n_steps
    timesteps = tuple(range(T_1000 - 1, 0, -dt))
    hash_str = f"ddim_n{n_steps}_seed{seed}"

    return DeterministicNoiseTrace(
        x_T=x_T,
        timesteps=timesteps,
        sampler_config_hash=hash_str,
    )


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    factors: torch.Tensor,
    device: torch.device,
    image_size: int,
    in_channels: int,
    n_steps: int = 200,
    alpha_bars: Optional[torch.Tensor] = None,
    intervention: Optional[CompiledIntervention] = None,
    trace: Optional[DeterministicNoiseTrace] = None,
) -> torch.Tensor:
    """Deterministic DDIM (η=0) sampling. Accepts optional NoiseTrace for paired generation."""
    model.eval()
    n = factors.shape[0]

    if alpha_bars is None:
        alpha_bars = get_alpha_bars("linear").to(device)

    if trace is not None:
        x = trace.x_T.to(device)
        timesteps = trace.timesteps
    else:
        x = torch.randn(n, in_channels, image_size, image_size, device=device)
        dt = T_1000 // n_steps
        timesteps = tuple(range(T_1000 - 1, 0, -dt))

    for t_val in timesteps:
        t_batch = torch.full((n,), t_val, device=device, dtype=torch.long)

        if intervention is not None:
            pred = model(x, t_batch, intervention.effective_factor_values.to(device),
                        intervention=intervention)
        else:
            pred = model(x, t_batch, factors)

        ab_t = alpha_bars[t_val]
        t_next = max(t_val - (T_1000 // n_steps), 0)
        ab_next = alpha_bars[t_next]

        alpha_t = ab_t / ab_next.clamp(min=1e-8)
        beta_t = 1.0 - alpha_t

        x0_pred = (x - torch.sqrt(1.0 - ab_t) * pred) / torch.sqrt(ab_t).clamp(min=1e-8)
        x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * pred if t_next > 0 else x0_pred

    return x


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    factors: torch.Tensor,
    device: torch.device,
    image_size: int,
    in_channels: int,
    n_steps: int = 200,
    alpha_bars: Optional[torch.Tensor] = None,
    intervention: Optional[CompiledIntervention] = None,
    trace: Optional[DeterministicNoiseTrace] = None,
    shared_noise_trace: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Stochastic DDPM sampling. Accepts shared NoiseTrace for paired generation."""
    model.eval()
    n = factors.shape[0]

    if alpha_bars is None:
        alpha_bars = get_alpha_bars("linear").to(device)

    if trace is not None:
        x = trace.x_T.to(device)
    else:
        x = torch.randn(n, in_channels, image_size, image_size, device=device)

    dt = T_1000 // n_steps
    timesteps = list(range(T_1000 - 1, 0, -dt))
    noise_idx = 0

    for t_val in timesteps:
        t_batch = torch.full((n,), t_val, device=device, dtype=torch.long)

        if intervention is not None:
            pred = model(x, t_batch, intervention.effective_factor_values.to(device),
                        intervention=intervention)
        else:
            pred = model(x, t_batch, factors)

        ab_t = alpha_bars[t_val]
        t_next = max(t_val - dt, 0)
        ab_next = alpha_bars[t_next]
        alpha_t = ab_t / ab_next.clamp(min=1e-8)
        beta_t = 1.0 - alpha_t

        sigma_t = (torch.sqrt((1.0 - ab_next) / (1.0 - ab_t).clamp(min=1e-8) * beta_t)
                    if t_next > 0 else torch.zeros_like(x))

        x = (x - beta_t / torch.sqrt(1.0 - ab_t).clamp(min=1e-8) * pred) / torch.sqrt(alpha_t).clamp(min=1e-8)

        if t_next > 0:
            if shared_noise_trace is not None and noise_idx < len(shared_noise_trace):
                z = shared_noise_trace[noise_idx].to(device)
            else:
                z = torch.randn_like(x)
            x = x + sigma_t * z
            noise_idx += 1

    return x


def generate_noise_trace(
    batch_size: int,
    in_channels: int,
    image_size: int,
    n_steps: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Generate a shared noise trace for DDPM paired evaluation."""
    g = torch.Generator()
    g.manual_seed(seed)
    dt = T_1000 // n_steps
    timesteps = list(range(T_1000 - 1, 0, -dt))
    trace = []
    for _ in timesteps[:-1]:
        trace.append(torch.randn(batch_size, in_channels, image_size, image_size, generator=g))
    return trace


__all__ = [
    "get_alpha_bars",
    "make_trace_ddim",
    "sample_ddim",
    "sample_ddpm",
    "generate_noise_trace",
    "T_1000",
]
