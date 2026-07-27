import os, json, math, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import ModelConfig, get_data_path, get_output_dir, get_oracle_path
from src.dataset import DSpritesDataset
from src.utils import safe_load_state_dict
from src.diffusion import get_alpha_bars, sample_images
from src.oracle import OracleClassifier
from src.registry import MODEL_REGISTRY


DSPRITES_FACTORS = dict(n_factors=3, factor_sizes=(3, 6, 40), in_channels=1)
SHAPES3D_FACTORS = dict(n_factors=6, factor_sizes=(10, 10, 10, 8, 4, 15), in_channels=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--oracle", type=str, default=get_oracle_path(),
                        help="Path to oracle classifier checkpoint")
    parser.add_argument("--output-dir", type=str, default=get_output_dir("eval"),
                        help="Output directory for evaluation results")
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--cfg-scale", type=float, default=0.0,
                        help="Classifier-free guidance scale (0=disabled)")
    parser.add_argument("--gate-sweep", action="store_true",
                        help="Test gate sensitivity: sweep gate in [0.0, 0.5, 1.0]")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--dataset", type=str, default="dsprites", choices=["dsprites", "3dshapes"],
                        help="Dataset to evaluate on")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    factor_cfg = SHAPES3D_FACTORS if args.dataset == "3dshapes" else DSPRITES_FACTORS
    config = ModelConfig(
        image_size=64, patch_size=4,
        stream_dim=256, n_stream_blocks=4, n_heads=8,
        use_gating=True,
        **factor_cfg,
    )

    alpha_bars = get_alpha_bars(args.schedule).to(device)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(config).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {args.model}: "
            f"missing keys={len(missing)}, unexpected keys={len(unexpected)}. "
            f"Check that --dataset matches the trained config."
        )
    print(f"Checkpoint loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    model.eval()

    oracle = OracleClassifier(config.factor_sizes, in_channels=getattr(config, "in_channels", 1)).to(device)
    oracle.load_state_dict(torch.load(args.oracle, map_location="cpu", weights_only=True))
    oracle.eval()

    if args.dataset == "dsprites":
        dataset = DSpritesDataset(get_data_path("dsprites"), split="test", seed=args.seed)
    else:
        from src.dataset import Shapes3DDataset
        dataset = Shapes3DDataset(get_data_path("3dshapes"), split="test", seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.n_samples, shuffle=True)
    ref_batch = next(iter(loader))
    ref_factors = ref_batch["factors"][:args.n_samples].to(device)
    null_factors = torch.zeros_like(ref_factors) if args.cfg_scale > 0 else None

    results = {}
    for intervene_idx in range(config.n_factors):
        normal_factors = ref_factors.clone()
        samples_normal = sample_images(model, normal_factors, device, config,
                                        n_steps=args.n_steps, cfg_scale=args.cfg_scale,
                                        uncondition_factors=null_factors, alpha_bars=alpha_bars)

        intervened_factors = normal_factors.clone()
        new_val = torch.randint(0, config.factor_sizes[intervene_idx],
                                (normal_factors.shape[0],), device=device)
        intervened_factors[:, intervene_idx] = new_val
        gates = [1.0] * config.n_factors
        if args.model == "FGR":
            gates[intervene_idx] = 0.0
        samples_intervened = sample_images(model, intervened_factors, device, config,
                                            gates=gates, n_steps=args.n_steps,
                                            cfg_scale=args.cfg_scale,
                                            uncondition_factors=null_factors,
                                            alpha_bars=alpha_bars)

        samples_vis = (samples_normal + 1) / 2
        samples_inter_vis = (samples_intervened + 1) / 2
        diff = np.abs(samples_inter_vis.cpu().numpy() - samples_vis.cpu().numpy()).reshape(args.n_samples, -1)
        change_mag = diff.mean(axis=1)
        unchanged = (change_mag < 0.05).mean()
        results[f"factor_{intervene_idx}_mean_change"] = float(change_mag.mean())
        results[f"factor_{intervene_idx}_nonintervention_stability"] = float(unchanged)

        with torch.no_grad():
            normal_preds = [p.argmax(1) for p in oracle(samples_normal)]
            cond_acc = [(normal_preds[fi] == normal_factors[:, fi]).float().mean().item()
                         for fi in range(config.n_factors)]
            for fi in range(config.n_factors):
                results[f"cond_accuracy_f{fi}"] = float(cond_acc[fi])
            results["mean_cond_accuracy"] = float(np.mean(cond_acc))
            inter_preds = [p.argmax(1) for p in oracle(samples_intervened)]
            for fi in range(config.n_factors):
                changed = (normal_preds[fi] != inter_preds[fi]).float().mean().item()
                results[f"factor_{intervene_idx}_oracle_change_f{fi}"] = float(changed)

            if args.gate_sweep and args.model == "FGR":
                for test_gate in [0.0, 0.5, 1.0]:
                    tg_gates = [1.0] * config.n_factors
                    tg_gates[intervene_idx] = test_gate
                    samples_tg = sample_images(model, intervened_factors, device, config,
                                                gates=tg_gates, n_steps=args.n_steps,
                                                cfg_scale=args.cfg_scale,
                                                uncondition_factors=null_factors,
                                                alpha_bars=alpha_bars)
                    tg_preds = [p.argmax(1) for p in oracle(samples_tg)]
                    for fi in range(config.n_factors):
                        changed = (normal_preds[fi] != tg_preds[fi]).float().mean().item()
                        results[f"factor_{intervene_idx}_gate{test_gate}_oracle_change_f{fi}"] = float(changed)

        print(f"[Factor {intervene_idx}] change={change_mag.mean():.4f} "
              f"stability={unchanged:.4f}", flush=True)

    mean_stability = np.mean([v for k, v in results.items() if "stability" in k])
    results["mean_nonintervention_stability"] = float(mean_stability)
    print(f"[{args.model}] mean_stability={mean_stability:.4f}", flush=True)

    out_path = os.path.join(args.output_dir, f"{args.model}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
