import os, sys, json, time, math, copy, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.config import ModelConfig, get_data_path, get_output_dir
from src.dataset import DSpritesDataset, Shapes3DDataset
from src.diffusion import get_alpha_bars
from src.registry import MODEL_REGISTRY


DATASET_REGISTRY = {
    "dsprites": DSpritesDataset,
    "3dshapes": Shapes3DDataset,
}

DSPRITES_CFG = dict(
    image_size=64, patch_size=4, n_factors=3,
    factor_sizes=(3, 6, 40), stream_dim=256,
    n_stream_blocks=4, n_heads=8, use_gating=True,
)

SHAPES3D_CFG = dict(
    image_size=64, patch_size=4, n_factors=6,
    factor_sizes=(10, 10, 10, 8, 4, 15), stream_dim=256,
    n_stream_blocks=4, n_heads=8, use_gating=True, in_channels=3,
)

DATASET_PATHS = {
    "dsprites": get_data_path("dsprites"),
    "3dshapes": get_data_path("3dshapes"),
}

T_1000 = 1000


def get_lr_schedule(optimizer, warmup, total):
    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1 + math.cos(progress * math.pi))
        return max(0.0, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@torch.no_grad()
def ema_update(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


def build_config(dataset_name: str, model_name: str):
    base = DSPRITES_CFG.copy()
    if dataset_name == "3dshapes":
        base.update(SHAPES3D_CFG)
    base["use_gating"] = (model_name == "FGR")
    return ModelConfig(**base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FGR", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, default="dsprites", choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--ablation", type=str, default=None, choices=[None, "no_inter_stream", "full_ca"])
    parser.add_argument("--output-dir", type=str, default=get_output_dir("train"),
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "train.log")
    log = open(log_file, "w", buffering=1)

    def p(msg):
        print(msg, flush=True)
        log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log.flush()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    tag = args.model
    if args.ablation:
        tag = f"{args.model}_{args.ablation}"
    if args.dataset != "dsprites":
        tag = f"{tag}_{args.dataset}"

    config = build_config(args.dataset, args.model)
    p(f"Config: {config}")
    p(f"Schedule: {args.schedule}")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    p(f"Model: {tag} | Params: {n_params/1e6:.2f}M")

    if args.ablation == "no_inter_stream" and hasattr(model, "set_inter_stream_ca"):
        if config.use_cross_attn:
            model.set_inter_stream_ca(False)
            p("Ablation: inter-stream CA disabled")
        else:
            p("WARNING: no_inter_stream ablation with use_cross_attn=False is a no-op. "
              "For independent factors (dSprites), use 'gate_artifact' ablation instead.")
    if args.ablation == "full_ca" and hasattr(model, "set_ca_mode"):
        model.set_inter_stream_ca(True)
        model.set_ca_mode("full")
        p("Ablation: using full bidirectional CA (not DAG)")

    ds_cls = DATASET_REGISTRY[args.dataset]
    data_path = DATASET_PATHS[args.dataset]
    dataset = ds_cls(data_path, split="train", seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True, pin_memory=(args.num_workers > 0),
    )
    loader_iter = iter(loader)

    alpha_bars = get_alpha_bars(args.schedule).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
    scheduler = get_lr_schedule(opt, args.warmup, args.steps)
    scaler = torch.amp.GradScaler("cuda")

    ema_model = copy.deepcopy(model)
    ema_decay = 0.9999
    for ema_p in ema_model.parameters():
        ema_p.requires_grad_(False)
    ema_params = [p for p in ema_model.parameters()]
    model_params = [p for p in model.parameters()]

    model.train()
    step = 0
    t_start = time.time()
    losses_log = []
    p("Starting training loop...")

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x_0 = batch["image"].to(device, non_blocking=True)
        factors = batch["factors"].to(device, non_blocking=True)
        t = torch.randint(0, T_1000, (x_0.shape[0],), device=device)

        alpha_bar = alpha_bars[t].view(-1, *([1] * (x_0.dim() - 1)))
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        with torch.amp.autocast("cuda"):
            if args.model == "FGR":
                gates = [1.0] * config.n_factors
                pred = model(x_t, t, factors, gates=gates)
            else:
                pred = model(x_t, t, factors)

            loss = F.mse_loss(pred, noise)

        if not torch.isfinite(loss):
            p(f"ERROR: NaN loss at step {step}, stopping training")
            break

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        torch._foreach_mul_(ema_params, ema_decay)
        torch._foreach_add_(ema_params, model_params, alpha=1 - ema_decay)
        losses_log.append(loss.item())

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            avg_loss = sum(losses_log[-500:]) / min(len(losses_log[-500:]), 500)
            mem = torch.cuda.max_memory_allocated() / 1e9
            speed = (step + 1) / elapsed
            remaining = (args.steps - step - 1) / speed / 3600 if speed > 0 else 0
            p(f"Step {step+1}/{args.steps} | loss={avg_loss:.6f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | mem={mem:.2f}GB | "
              f"speed={speed:.1f}it/s | est_remain={remaining:.1f}h")
            torch.cuda.reset_peak_memory_stats()

            if (step + 1) % args.ckpt_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"{tag}_step{step+1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                ema_path = os.path.join(args.output_dir, f"{tag}_ema_step{step+1}.pt")
                torch.save(ema_model.state_dict(), ema_path)
                p(f"Checkpoints: {ckpt_path}, {ema_path}")

        step += 1

    elapsed = time.time() - t_start
    p(f"\nTraining complete! Total time: {elapsed/3600:.2f}h")
    p(f"Final avg loss (last 500): {sum(losses_log[-500:])/500:.6f}")

    final_path = os.path.join(args.output_dir, f"{tag}_final.pt")
    ema_final_path = os.path.join(args.output_dir, f"{tag}_ema_final.pt")
    torch.save(model.state_dict(), final_path)
    torch.save(ema_model.state_dict(), ema_final_path)
    with open(os.path.join(args.output_dir, f"{tag}_losses.json"), "w") as f:
        json.dump({"losses": losses_log, "total_time_h": elapsed/3600}, f)
    p(f"Model saved to {final_path}")
    p(f"EMA model saved to {ema_final_path}")
    log.close()


if __name__ == "__main__":
    main()
