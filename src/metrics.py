from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return mask ^ binary_erosion(mask)


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return float("inf"), float("inf")

    ps = _surface(pred)
    gs = _surface(gt)
    dt_g = distance_transform_edt(~gs)
    dt_p = distance_transform_edt(~ps)
    d_pg = dt_g[ps]
    d_gp = dt_p[gs]
    both = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(both, 95)) if both.size else 0.0
    assd = float((d_pg.mean() + d_gp.mean()) / 2.0) if d_pg.size and d_gp.size else float("inf")
    return hd95, assd


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2, eps: float = 1e-7) -> float:
    """Boundary F1 using a symmetric pixel tolerance.

    A predicted boundary pixel is correct when it falls within `tolerance` pixels of
    a GT boundary pixel, and vice versa for recall. This is intentionally exact and
    non-differentiable; GMGD uses it as a teacher signal, not as a backprop loss.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    ps = _surface(pred)
    gs = _surface(gt)
    if not ps.any() and not gs.any():
        return 1.0
    if not ps.any() or not gs.any():
        return 0.0

    tol = max(0, int(tolerance))
    if tol > 0:
        structure = np.ones((3, 3), dtype=bool)
        gs_match = binary_dilation(gs, structure=structure, iterations=tol)
        ps_match = binary_dilation(ps, structure=structure, iterations=tol)
    else:
        gs_match, ps_match = gs, ps

    precision = float(np.logical_and(ps, gs_match).sum()) / (float(ps.sum()) + eps)
    recall = float(np.logical_and(gs, ps_match).sum()) / (float(gs.sum()) + eps)
    if precision + recall <= eps:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def case_metrics(
    prob: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
    bf1_tolerance: int = 2,
) -> dict:
    pred = prob >= threshold
    gt = gt >= 0.5
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    hd95, assd = _surface_distances(pred, gt)
    bf1 = boundary_f1(pred, gt, tolerance=bf1_tolerance, eps=eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "hd95": float(hd95),
        "assd": float(assd),
        "bf1": float(bf1),
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
