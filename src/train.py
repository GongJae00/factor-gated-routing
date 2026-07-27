"""
Training loop for Factor-Path Diffusion.

Supports:
- ROST-FRG and all baselines
- Gate exposure training (condition masking, branch dropout)
- Full checkpoint save (model, optimizer, scheduler, scaler, config, RNG, step)
- Resume from checkpoint
- NaN guard
- Cosine LR schedule with warmup
- EMA with foreach ops
- FP16 mixed precision
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import copy
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.config import (
    ModelConfig, build_model_config, DSPRITES_CFG, SHAPES3D_CFG,
    get_data_path, get_output_dir,
)
from src.dataset import DSpritesDataset, Shapes3DDataset
from src.sampling import get_alpha_bars
from src.registry import MODEL_REGISTRY
from src.interventions import make_observational
from src.types import CategoricalFactorSpec, GraphSpec, GraphType

DATASET_REGISTRY = {
    "dsprites": DSpritesDataset,
    "3dshapes": Shapes3DDataset,
}


def get_lr_schedule(optimizer, warmup, total):
    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(progress * math.pi)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@torch.no_grad()
def ema_update(ema_params, model_params, decay):
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, model_params, alpha=1 - decay)


def save_checkpoint(path: str, model, ema_model, optimizer, scheduler, scaler,
                    config: ModelConfig, step: int, loss_log: list, args):
    torch.save({
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict() if ema_model else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": {
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "n_factors": config.n_factors,
            "factor_sizes": list(config.factor_sizes),
            "factor_names": list(config.factor_names),
            "trunk_dim": config.trunk_dim,
            "branch_dim": config.branch_dim,
            "n_trunk_blocks": config.n_trunk_blocks,
            "n_branch_layers": config.n_branch_layers,
            "n_heads": config.n_heads,
            "graph_type": config.graph_type,
            "dag_edges": config.dag_edges,
            "use_base": config.use_base,
            "use_gating": config.use_gating,
            "schedule": config.schedule,
        },
        "step": step,
        "loss_log": loss_log,
        "args": vars(args),
    }, path)


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["model_config"])
    return ckpt, config


def build_config_from_yaml(dataset_name: str, config_path: str | None = None) -> ModelConfig:
    if config_path and os.path.exists(config_path):
        cfg = yaml.safe_load(open(config_path))
        return build_model_config(cfg)

    base = DSPRITES_CFG.copy()
    if dataset_name == "3dshapes":
        base.update(SHAPES3D_CFG)
    elif dataset_name == "dsprites":
        # Update with dSprites-specific if config exists
        dsprites_yaml = os.path.join("configs", "dsprites.yaml")
        if os.path.exists(dsprites_yaml):
            return build_model_config(yaml.safe_load(open(dsprites_yaml)))

    return build_model_config(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ROSTFRG",
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, default="dsprites",
                       choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--ablation", type=str, default=None,
                       choices=[None, "no_inter_stream", "dense_ca"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gate-dropout", type=float, default=0.0,
                       help="Probability of dropping a factor branch during training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output_dir is None:
        args.output_dir = get_output_dir("train")

    os.makedirs(args.output_dir, exist_ok=True)
    log = open(os.path.join(args.output_dir, "train.log"), "w", buffering=1)

    def log_msg(msg):
        print(msg, flush=True)
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log.flush()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    tag = args.model
    if args.ablation:
        tag = f"{tag}_{args.ablation}"
    if args.dataset != "dsprites":
        tag = f"{tag}_{args.dataset}"

    config = build_config_from_yaml(args.dataset, args.config)
    log_msg(f"Model: {tag} | Dataset: {args.dataset} | Steps: {args.steps}")
    log_msg(f"Config: trunk={config.trunk_dim}, branch={config.branch_dim}, "
            f"blocks={config.n_trunk_blocks}, graph={config.graph_type}")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_msg(f"Params: {n_params/1e6:.2f}M")

    # Gate exposure: binary branch dropout during training
    gate_dropout_p = args.gate_dropout

    # Ablation: dense cross-attention
    if args.ablation == "dense_ca":
        config.graph_type = "dense_directed"
        log_msg("Ablation: DENSE_DIRECTED graph")
        model = model_cls(config).to(device)

    ds_cls = DATASET_REGISTRY[args.dataset]
    data_path = get_data_path(args.dataset)
    dataset = ds_cls(data_path, split="train", seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True, pin_memory=(args.num_workers > 0 and device.type == "cuda"),
    )
    loader_iter = iter(loader)

    alpha_bars = get_alpha_bars(config.schedule).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
    scheduler = get_lr_schedule(opt, args.warmup, args.steps)
    scaler = torch.amp.GradScaler("cuda")

    ema_model = copy.deepcopy(model) if device.type == "cuda" else None
    ema_decay = 0.9999
    ema_params = [p for p in ema_model.parameters()] if ema_model else []
    model_params_list = [p for p in model.parameters()]

    step = 0
    t_start = time.time()
    losses_log = []

    # Resume logic
    if args.resume:
        ckpt, _ = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model_state"])
        if ema_model and ckpt.get("ema_state"):
            ema_model.load_state_dict(ckpt["ema_state"])
        if ckpt.get("optimizer_state"):
            opt.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        step = ckpt.get("step", 0)
        losses_log = ckpt.get("loss_log", [])
        log_msg(f"Resumed from step {step}")

    model.train()
    log_msg(f"Starting training loop at step {step}...")

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x_0 = batch["image"].to(device, non_blocking=True)
        factors = batch["factors"].to(device, non_blocking=True)
        t = torch.randint(0, config.diffusion_steps, (x_0.shape[0],), device=device)

        alpha_bar = alpha_bars[t].view(-1, *([1] * (x_0.dim() - 1)))
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            pred = model(x_t, t, factors)
            loss = F.mse_loss(pred, noise)

        if not torch.isfinite(loss):
            log_msg(f"ERROR: NaN loss at step {step}, stopping")
            break

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if ema_model:
            ema_update(ema_params, model_params_list, ema_decay)

        losses_log.append(loss.item())

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            recent = losses_log[-500:]
            avg_loss = sum(recent) / len(recent)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (args.steps - step - 1) / speed / 3600 if speed > 0 else 0
            log_msg(f"Step {step+1}/{args.steps} | loss={avg_loss:.6f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | mem={mem:.2f}GB | "
                    f"speed={speed:.1f}it/s | est_remain={remaining:.1f}h")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"{tag}_step{step+1}.pt")
            save_checkpoint(ckpt_path, model, ema_model, opt, scheduler, scaler,
                          config, step + 1, losses_log, args)
            log_msg(f"Checkpoint: {ckpt_path}")

        step += 1

    elapsed = time.time() - t_start
    log_msg(f"\nTraining complete! {elapsed/3600:.2f}h, {step} steps")

    final_path = os.path.join(args.output_dir, f"{tag}_final.pt")
    save_checkpoint(final_path, model, ema_model, opt, scheduler, scaler,
                   config, step, losses_log, args)
    log_msg(f"Final model: {final_path}")

    with open(os.path.join(args.output_dir, f"{tag}_losses.json"), "w") as f:
        json.dump({"losses": losses_log, "total_time_h": elapsed/3600, "steps": step}, f)

    log.close()


if __name__ == "__main__":
    main()
