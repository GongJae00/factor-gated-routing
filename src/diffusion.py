import math
import torch
from typing import Optional

T_1000 = 1000


def get_alpha_bars(schedule="linear"):
    if schedule == "linear":
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, T_1000)
    elif schedule == "cosine":
        t = torch.linspace(0, 1, T_1000)
        return (torch.cos(t * math.pi / 2) ** 2).clamp(min=1e-5)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    alphas = 1 - betas
    return torch.cumprod(alphas, dim=0)


@torch.no_grad()
def sample_images(model, factors, device, config, gates=None, n_steps=200,
                  cfg_scale=0.0, uncondition_factors=None, alpha_bars=None):
    model.eval()
    n = factors.shape[0]
    in_c = getattr(config, "in_channels", 1)
    x = torch.randn(n, in_c, config.image_size, config.image_size, device=device)
    if alpha_bars is None:
        alpha_bars = get_alpha_bars("linear").to(device)

    dt = T_1000 // n_steps
    for t_val in range(999, 0, -dt):
        t_batch = torch.full((n,), t_val, device=device, dtype=torch.long)

        if cfg_scale > 0 and uncondition_factors is not None:
            pred_cond = model(x, t_batch, factors, gates=gates)
            pred_uncond = model(x, t_batch, uncondition_factors, gates=gates)
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = model(x, t_batch, factors, gates=gates)

        ab_t = alpha_bars[t_val]
        t_next = max(t_val - dt, 0)
        ab_next = alpha_bars[t_next]
        alpha = ab_t / ab_next.clamp(min=1e-8)
        beta = 1 - alpha

        sigma = (torch.sqrt((1 - ab_next) / (1 - ab_t).clamp(min=1e-8) * beta)
                 if t_next > 0 else torch.zeros_like(x))
        x = (1 / torch.sqrt(alpha).clamp(min=1e-8)) * (
            x - beta / torch.sqrt(1 - ab_t).clamp(min=1e-8) * pred
        )
        if t_next > 0:
            x = x + sigma * torch.randn_like(x)

    return x
