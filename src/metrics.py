"""
Metric primitives for Factor-Path Diffusion evaluation.

All metrics operate on oracle predictions tensors.
Categorical and continuous factor support.
"""

from __future__ import annotations

import torch
from typing import Optional


def compute_target_value_success(
    oracle_preds: list[torch.Tensor],
    target_values: torch.Tensor,
) -> torch.Tensor:
    """TargetValueSuccess: P[Oracle_i(x_edit) = v_i'] — K-vector."""
    K = len(oracle_preds)
    result = torch.zeros(K, device=target_values.device)
    for i in range(K):
        pred_class = oracle_preds[i].argmax(dim=1)
        result[i] = (pred_class == target_values[:, i]).float().mean()
    return result


def compute_target_change_rate(
    oracle_preds_edit: list[torch.Tensor],
    oracle_preds_orig: list[torch.Tensor],
) -> torch.Tensor:
    """TargetChangeRate: P[O_i(x_edit) != O_i(x_orig)] — K-vector."""
    K = len(oracle_preds_edit)
    result = torch.zeros(K, device=oracle_preds_edit[0].device)
    for i in range(K):
        edit_class = oracle_preds_edit[i].argmax(dim=1)
        orig_class = oracle_preds_orig[i].argmax(dim=1)
        result[i] = (edit_class != orig_class).float().mean()
    return result


def compute_off_target_change(
    oracle_preds_edit: list[torch.Tensor],
    oracle_preds_orig: list[torch.Tensor],
) -> torch.Tensor:
    """OffTargetChange: L_ij = P[O_j(x_edit_i) != O_j(x_orig)] — K×K, diagonal NaN."""
    K = len(oracle_preds_edit)
    result = torch.full((K, K), float("nan"), device=oracle_preds_edit[0].device)
    for i in range(K):
        edit_preds_i = oracle_preds_edit[i]
        orig_preds_i = oracle_preds_orig[i]
        for j in range(K):
            if i == j:
                continue
            edit_class = edit_preds_i[j].argmax(dim=1) if isinstance(edit_preds_i, list) else None
            orig_class = orig_preds_i[j].argmax(dim=1) if isinstance(orig_preds_i, list) else None
    return result


def compute_off_target_change_matrix(
    all_edit_preds: list[list[torch.Tensor]],
    orig_preds: list[torch.Tensor],
) -> torch.Tensor:
    """OffTargetChange K×K matrix.

    all_edit_preds[i] = oracle predictions for edit on factor i
    L_ij = P[O_j(x_edit_i) != O_j(x_original)], i≠j, diagonal NaN.
    """
    K = len(orig_preds)
    device = orig_preds[0].device
    result = torch.full((K, K), float("nan"), device=device)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            edit_class = all_edit_preds[i][j].argmax(dim=1)
            orig_class = orig_preds[j].argmax(dim=1)
            result[i, j] = (edit_class != orig_class).float().mean()

    return result


def compute_no_op_change(
    oracle_preds_noop: list[torch.Tensor],
    oracle_preds_orig: list[torch.Tensor],
) -> torch.Tensor:
    """NoOpChange matrix. Should be all-zero with same NoiseTrace."""
    K = len(oracle_preds_noop)
    device = oracle_preds_noop[0].device
    result = torch.zeros(K, K, device=device)
    for i in range(K):
        for j in range(K):
            noop_class = oracle_preds_noop[j].argmax(dim=1)
            orig_class = oracle_preds_orig[j].argmax(dim=1)
            result[i, j] = (noop_class != orig_class).float().mean()
    return result


def compute_source_invariance_error_denoiser(
    epsilon_original: torch.Tensor,
    epsilon_cut: torch.Tensor,
) -> dict[str, float]:
    """SourceInvarianceError at denoiser level. Expects zero difference."""
    diff = (epsilon_original - epsilon_cut).pow(2)
    return {
        "mse": diff.mean().item(),
        "max_abs": diff.max().item(),
        "rmse": diff.mean().sqrt().item(),
    }


def compute_source_invariance_error_trajectory(
    x0_original: torch.Tensor,
    x0_cut: torch.Tensor,
) -> dict[str, float]:
    """SourceInvarianceError at trajectory level."""
    diff = (x0_original - x0_cut).pow(2)
    return {
        "mse": diff.mean().item(),
        "max_abs": diff.max().item(),
        "rmse": diff.mean().sqrt().item(),
    }


def compute_direct_contribution_effect(
    epsilon_with_output: torch.Tensor,
    epsilon_ablated: torch.Tensor,
) -> dict[str, float]:
    """Effect of DIRECT_OUTPUT_ABLATION on denoiser output."""
    diff = (epsilon_with_output - epsilon_ablated).pow(2)
    return {
        "mse": diff.mean().item(),
        "max_abs": diff.max().item(),
        "norm_ratio": (epsilon_ablated.norm() / epsilon_with_output.norm().clamp(min=1e-8)).item(),
    }


def compute_oracle_conditional_accuracy(
    oracle_preds: list[torch.Tensor],
    factor_values: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Per-factor and overall conditional accuracy."""
    K = len(oracle_preds)
    per_factor = torch.zeros(K)
    all_correct = torch.ones(factor_values.shape[0], dtype=torch.bool)

    for i in range(K):
        pred_class = oracle_preds[i].argmax(dim=1)
        correct = (pred_class == factor_values[:, i])
        per_factor[i] = correct.float().mean()
        all_correct = all_correct & correct

    overall = all_correct.float().mean().item()
    return per_factor, overall


__all__ = [
    "compute_target_value_success",
    "compute_target_change_rate",
    "compute_off_target_change_matrix",
    "compute_no_op_change",
    "compute_source_invariance_error_denoiser",
    "compute_source_invariance_error_trajectory",
    "compute_direct_contribution_effect",
    "compute_oracle_conditional_accuracy",
]
