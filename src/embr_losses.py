from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def per_sample_bce_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """BCE + soft Dice, returned per sample."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3))
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - (2.0 * inter + eps) / (den + eps)
    return 0.5 * bce + 0.5 * dice_loss


def _soft_morphological_edge(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    k = 2 * int(radius) + 1
    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    ero = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=radius)
    return (dil - ero).clamp(0.0, 1.0)


def per_sample_boundary_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable boundary Dice using a soft morphological contour."""
    p = torch.sigmoid(logits)
    pe = _soft_morphological_edge(p, radius=1)
    ge = _soft_morphological_edge(target.float(), radius=1)
    inter = (pe * ge).sum(dim=(1, 2, 3))
    den = pe.sum(dim=(1, 2, 3)) + ge.sum(dim=(1, 2, 3))
    score = (2.0 * inter + eps) / (den + eps)
    return 1.0 - score


def per_sample_multiscale_boundary_distance(
    logits: torch.Tensor,
    target: torch.Tensor,
    radii: tuple[int, ...] = (1, 2, 4, 8, 16),
    weights: tuple[float, ...] = (1.0, 1.0, 2.0, 4.0, 8.0),
    eps: float = 1e-6,
) -> torch.Tensor:
    """GPU-only surface-displacement surrogate.

    A boundary point contributes at every radius for which it remains unmatched.
    Large displacements therefore survive more scales and receive more weight. The
    loss is symmetric and does not require a learned metric or CPU distance map.
    """
    if len(radii) != len(weights):
        raise ValueError("radii and weights must have the same length")
    p = torch.sigmoid(logits)
    pe = _soft_morphological_edge(p, radius=1)
    ge = _soft_morphological_edge(target.float(), radius=1)
    # Numerical guard: when a prediction is nearly constant, soft edge mass can be
    # extremely small. Dividing by ~1e-6 keeps the forward value finite but can
    # create enormous backward derivatives. A one-pixel mass floor preserves the
    # interpretation as a normalized mismatch while keeping gradients bounded.
    pe_mass = pe.sum(dim=(1, 2, 3)).clamp_min(1.0)
    ge_mass = ge.sum(dim=(1, 2, 3)).clamp_min(1.0)
    total = torch.zeros_like(pe_mass)
    norm = float(sum(weights))
    for r, w in zip(radii, weights):
        k = 2 * int(r) + 1
        g_near = F.max_pool2d(ge, kernel_size=k, stride=1, padding=r).clamp(0.0, 1.0)
        p_near = F.max_pool2d(pe, kernel_size=k, stride=1, padding=r).clamp(0.0, 1.0)
        miss_p = (pe * (1.0 - g_near)).sum(dim=(1, 2, 3)) / pe_mass
        miss_g = (ge * (1.0 - p_near)).sum(dim=(1, 2, 3)) / ge_mass
        total = total + float(w) * 0.5 * (miss_p + miss_g)
    return total / max(norm, eps)


def per_sample_adaptive_tversky(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha_fp: torch.Tensor,
    beta_fn: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-case asymmetric Tversky loss from a precomputed training-only policy."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - p) * target).sum(dim=(1, 2, 3))
    alpha_fp = alpha_fp.reshape(-1).to(dtype=logits.dtype, device=logits.device)
    beta_fn = beta_fn.reshape(-1).to(dtype=logits.dtype, device=logits.device)
    score = (tp + eps) / (tp + alpha_fp * fp + beta_fn * fn + eps)
    return 1.0 - score


def _boundary_band(target: torch.Tensor, radius: int = 3) -> torch.Tensor:
    k = 2 * int(radius) + 1
    dil = F.max_pool2d(target, kernel_size=k, stride=1, padding=radius)
    ero = -F.max_pool2d(-target, kernel_size=k, stride=1, padding=radius)
    return (dil - ero).clamp(0.0, 1.0)


def per_sample_correctness_anchor(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    boundary_radius: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Preserve source behavior only where the source is already correct and away from GT boundaries.

    The boundary region is intentionally excluded so the refiner is free to move the
    contour. This is a safety anchor, not a general knowledge-distillation loss.
    """
    p = torch.sigmoid(logits)
    with torch.no_grad():
        t = torch.sigmoid(teacher_logits)
        teacher_hard = (t >= 0.5).to(target.dtype)
        correct = (teacher_hard == target).to(target.dtype)
        far_from_boundary = 1.0 - _boundary_band(target, radius=boundary_radius)
        mask = correct * far_from_boundary
    sq = (p - t).square() * mask
    num = sq.sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)) + eps
    return num / den


def per_sample_residual_penalty(residual_logits: torch.Tensor) -> torch.Tensor:
    return residual_logits.abs().mean(dim=(1, 2, 3))


@dataclass
class EMBRLossWeights:
    boundary: float = 0.25
    multiscale_distance: float = 0.35
    tversky: float = 0.20
    anchor: float = 0.08
    residual_l1: float = 0.002


def geometry_ramp(epoch: int, warmup_epochs: int = 1, ramp_epochs: int = 3) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - warmup_epochs) / float(ramp_epochs))))
