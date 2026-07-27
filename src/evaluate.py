"""
Paired-noise evaluation pipeline for Factor-Path Diffusion.

Three protocols:
1. factor_edit — change factor value, measure target success + off-target leakage
2. factor_source_cut — verify Path Non-Interference (output invariant to f_i)
3. neural_graph_surgery — cut incoming edges + inject v' + preserve outgoing

All comparisons use shared NoiseTrace for paired evaluation.
"""

from __future__ import annotations

import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import (
    ModelConfig, build_model_config, DSPRITES_CFG, SHAPES3D_CFG,
    get_data_path, get_output_dir, get_oracle_path,
)
from src.dataset import DSpritesDataset, Shapes3DDataset
from src.sampling import get_alpha_bars, sample_ddim, make_trace_ddim
from src.oracle import OracleClassifier
from src.registry import MODEL_REGISTRY
from src.types import (
    InterventionMode, InterventionSpec, GraphSpec, GraphType,
    CategoricalFactorSpec,
)
from src.interventions import compile_intervention, make_observational
from src.metrics import (
    compute_target_value_success,
    compute_target_change_rate,
    compute_off_target_change_matrix,
    compute_no_op_change,
    compute_source_invariance_error_trajectory,
    compute_oracle_conditional_accuracy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--oracle", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=250)
    parser.add_argument("--dataset", type=str, default="dsprites")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schedule", type=str, default="cosine")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output_dir is None:
        args.output_dir = get_output_dir("eval")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_config" in ckpt:
        config = ModelConfig(**ckpt["model_config"])
    else:
        cfg_dict = SHAPES3D_CFG if args.dataset == "3dshapes" else DSPRITES_CFG
        config = build_model_config(cfg_dict)

    print(f"Config: {config.n_factors} factors, trunk={config.trunk_dim}, "
          f"branch={config.branch_dim}, graph={config.graph_type}")

    # Build factor specs for intervention compiler
    factor_specs = [CategoricalFactorSpec(f"f{i}", config.factor_sizes[i])
                    for i in range(config.n_factors)]
    graph_spec = GraphSpec(
        graph_type=GraphType(config.graph_type),
        num_nodes=config.n_factors,
        edges=tuple(config.dag_edges),
    )

    # Load model
    model_fn = MODEL_REGISTRY[args.model]
    model = model_fn(config).to(device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "ema_state" in ckpt:
        model.load_state_dict(ckpt["ema_state"])
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params/1e6:.2f}M params)")

    # Load oracle
    oracle_path = args.oracle or get_oracle_path()
    oracle = OracleClassifier(config.factor_sizes, in_channels=config.in_channels).to(device)
    if os.path.exists(oracle_path):
        oracle.load_state_dict(torch.load(oracle_path, map_location="cpu", weights_only=True))
    oracle.eval()

    # Load dataset for reference
    ds_cls = DSpritesDataset if args.dataset == "dsprites" else Shapes3DDataset
    dataset = ds_cls(get_data_path(args.dataset), split="test", seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.n_samples, shuffle=True)
    ref_batch = next(iter(loader))
    ref_factors = ref_batch["factors"][:args.n_samples].to(device)

    alpha_bars = get_alpha_bars(args.schedule).to(device)
    n_branch_layers = config.n_branch_layers

    results = {}
    K = config.n_factors

    # Generate shared NoiseTrace
    trace = make_trace_ddim(args.n_samples, config.in_channels, config.image_size,
                           args.n_steps, device, seed=args.seed)

    # --- Protocol 0: Observational (normal generation) ---
    obs_intervention = make_observational(
        ref_factors[0].tolist(), graph_spec, factor_specs,
        args.n_samples, n_branch_layers, device=str(device),
    )
    samples_normal = sample_ddim(model, ref_factors, device, config.image_size,
                                 config.in_channels, args.n_steps, alpha_bars,
                                 intervention=obs_intervention, trace=trace)

    with torch.no_grad():
        normal_preds = oracle(samples_normal)
    per_factor_acc, overall_acc = compute_oracle_conditional_accuracy(normal_preds, ref_factors)
    results["observational_per_factor_accuracy"] = per_factor_acc.tolist()
    results["observational_overall_accuracy"] = overall_acc

    # --- Protocol 1: Factor Edit ---
    all_edit_preds = []  # all_edit_preds[i] = oracle preds for edit on factor i
    target_success = torch.zeros(K)
    target_change = torch.zeros(K)

    for intervene_idx in range(K):
        new_val = _offset_sample(ref_factors[:, intervene_idx], config.factor_sizes[intervene_idx])

        edit_intervention = compile_intervention(
            InterventionSpec(
                mode=InterventionMode.FACTOR_EDIT,
                factor_edits={intervene_idx: int(new_val[0].item())},
            ),
            graph_spec, factor_specs,
            factor_values=ref_factors[0].tolist(),
            batch_size=args.n_samples, num_layers=n_branch_layers, device=str(device),
        )

        samples_edit = sample_ddim(model, ref_factors, device, config.image_size,
                                    config.in_channels, args.n_steps, alpha_bars,
                                    intervention=edit_intervention, trace=trace)

        with torch.no_grad():
            edit_preds = oracle(samples_edit)

        all_edit_preds.append(edit_preds)

        # Target success: P[Oracle_i(x_edit) = v']
        ts = (edit_preds[intervene_idx].argmax(dim=1) == new_val).float().mean()
        target_success[intervene_idx] = ts
        results[f"factor_{intervene_idx}_target_success"] = float(ts)

        # Target change: P[Oracle_i(x_edit) != Oracle_i(x_orig)]
        tc = (edit_preds[intervene_idx].argmax(dim=1) != normal_preds[intervene_idx].argmax(dim=1)).float().mean()
        target_change[intervene_idx] = tc
        results[f"factor_{intervene_idx}_target_change"] = float(tc)

        print(f"[Edit {intervene_idx}] target_success={ts:.4f}, target_change={tc:.4f}", flush=True)

    # Off-target leakage matrix
    leakage = compute_off_target_change_matrix(all_edit_preds, normal_preds)
    results["off_target_change_matrix"] = leakage.tolist()
    results["mean_off_target_change"] = float(leakage.nanmean())

    # --- Protocol 2: Factor Source Cut (Path Non-Interference) ---
    if args.model in ("ROSTFRG", "FGR"):
        for intervene_idx in range(K):
            cut_intervention = compile_intervention(
                InterventionSpec(
                    mode=InterventionMode.FACTOR_SOURCE_CUT,
                    source_cut_factors=frozenset([intervene_idx]),
                ),
                graph_spec, factor_specs,
                factor_values=ref_factors[0].tolist(),
                batch_size=args.n_samples, num_layers=n_branch_layers, device=str(device),
            )
            samples_cut = sample_ddim(model, ref_factors, device, config.image_size,
                                      config.in_channels, args.n_steps, alpha_bars,
                                      intervention=cut_intervention, trace=trace)

            # Source invariance: change factor value with source cut → identical output
            new_val = _offset_sample(ref_factors[:, intervene_idx], config.factor_sizes[intervene_idx])
            cut_diff_intervention = compile_intervention(
                InterventionSpec(
                    mode=InterventionMode.FACTOR_SOURCE_CUT,
                    source_cut_factors=frozenset([intervene_idx]),
                    factor_edits={intervene_idx: int(new_val[0].item())},
                ),
                graph_spec, factor_specs,
                factor_values=ref_factors[0].tolist(),
                batch_size=args.n_samples, num_layers=n_branch_layers, device=str(device),
            )
            samples_cut_diff = sample_ddim(model, ref_factors, device, config.image_size,
                                            config.in_channels, args.n_steps, alpha_bars,
                                            intervention=cut_diff_intervention, trace=trace)

            inv_err = compute_source_invariance_error_trajectory(samples_cut, samples_cut_diff)
            results[f"factor_{intervene_idx}_source_invariance_rmse"] = inv_err["rmse"]
            results[f"factor_{intervene_idx}_source_invariance_max"] = inv_err["max_abs"]
            print(f"[Cut {intervene_idx}] invariance: rmse={inv_err['rmse']:.6f}, "
                  f"max={inv_err['max_abs']:.6f}", flush=True)

    # Save
    out_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _offset_sample(old_vals: torch.Tensor, factor_size: int) -> torch.Tensor:
    """Sample new value guaranteed ≠ old value."""
    offset = torch.randint(1, factor_size, old_vals.shape, device=old_vals.device)
    return (old_vals + offset) % factor_size


if __name__ == "__main__":
    main()
