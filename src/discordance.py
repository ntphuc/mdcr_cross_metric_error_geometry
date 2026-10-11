from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_opening, label, shift as ndi_shift

from .metrics import case_metrics

ROUTES = ("no_correction", "recover_fn", "remove_fp", "boundary_align", "tail_cleanup")
ROUTE_TO_ID = {name: i for i, name in enumerate(ROUTES)}


def finite_surface_score(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.exp(-max(0.0, float(value)) / max(float(scale), 1e-6)))


def normalized_distance_error(value: float, scale: float) -> float:
    return 1.0 - finite_surface_score(value, scale)


def exact_utility(metrics: dict, hd95_scale: float = 16.0, assd_scale: float = 4.0,
                  weights: dict | None = None) -> float:
    w = weights or {"dice": 0.35, "bf1": 0.25, "hd95": 0.25, "assd": 0.15}
    return float(
        w.get("dice", 0.35) * float(metrics["dice"]) +
        w.get("bf1", 0.25) * float(metrics["bf1"]) +
        w.get("hd95", 0.25) * finite_surface_score(float(metrics["hd95"]), hd95_scale) +
        w.get("assd", 0.15) * finite_surface_score(float(metrics["assd"]), assd_scale)
    )


def binary_entropy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float32), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return m
    return np.logical_xor(m, binary_erosion(m))


def uncertainty_features(prob: np.ndarray, pred: np.ndarray | None = None) -> dict[str, float]:
    p = np.asarray(prob, dtype=np.float32)
    pred = (p >= 0.5) if pred is None else np.asarray(pred).astype(bool)
    ent = binary_entropy(p)
    fg_rel = np.logical_or(pred, binary_dilation(pred, iterations=3))
    if fg_rel.any():
        mean_ent = float(ent[fg_rel].mean())
        low_conf = float((np.abs(p[fg_rel] - 0.5) < 0.15).mean())
    else:
        mean_ent = float(ent.mean())
        low_conf = float((np.abs(p - 0.5) < 0.15).mean())
    b = boundary(pred)
    band = binary_dilation(b, iterations=3) if b.any() else b
    boundary_ent = float(ent[band].mean()) if band.any() else mean_ent
    margin = float(np.mean(np.abs(p[fg_rel] - 0.5))) if fg_rel.any() else float(np.mean(np.abs(p - 0.5)))
    return {
        "uncertainty_mean": mean_ent,
        "uncertainty_boundary": boundary_ent,
        "low_conf_fraction": low_conf,
        "confidence_margin": margin,
    }


def metric_profile_features(metrics: dict, hd95_scale: float = 16.0, assd_scale: float = 4.0) -> dict[str, float]:
    dice = float(metrics["dice"])
    bf1 = float(metrics["bf1"])
    hd = normalized_distance_error(float(metrics["hd95"]), hd95_scale)
    assd = normalized_distance_error(float(metrics["assd"]), assd_scale)
    gt = max(float(metrics.get("gt_pixels", 0.0)), 1.0)
    pred = max(float(metrics.get("pred_pixels", 0.0)), 1.0)
    fp = float(metrics.get("fp", 0.0))
    fn = float(metrics.get("fn", 0.0))
    fp_ratio_gt = fp / gt
    fn_ratio_gt = fn / gt
    fp_fraction_pred = fp / pred
    fn_fraction_gt = fn / gt
    precision_err = 1.0 - float(metrics.get("precision", 1.0))
    recall_err = 1.0 - float(metrics.get("recall", 1.0))
    # Discordance terms preserve the fact that metrics disagree rather than
    # collapsing them into a single scalar quality value.
    return {
        "dice": dice,
        "bf1": bf1,
        "hd95_err": hd,
        "assd_err": assd,
        "dice_err": 1.0 - dice,
        "bf1_err": 1.0 - bf1,
        "surface_tail_gap": hd - assd,
        "overlap_boundary_gap": dice - bf1,
        "overlap_surface_gap": (1.0 - dice) - assd,
        "boundary_surface_gap": (1.0 - bf1) - assd,
        "fp_ratio_gt": fp_ratio_gt,
        "fn_ratio_gt": fn_ratio_gt,
        "fp_fraction_pred": fp_fraction_pred,
        "fn_fraction_gt": fn_fraction_gt,
        "fp_fn_asymmetry": fn_ratio_gt - fp_ratio_gt,
        "precision_err": precision_err,
        "recall_err": recall_err,
    }


def scalar_quality_from_features(f: dict[str, float]) -> float:
    return float(0.35 * f["dice"] + 0.25 * f["bf1"] + 0.25 * (1.0 - f["hd95_err"]) + 0.15 * (1.0 - f["assd_err"]))


def feature_columns(feature_set: str) -> list[str]:
    fs = feature_set.lower()
    if fs == "dice_only":
        return ["dice"]
    if fs == "fpfn":
        return ["fp_ratio_gt", "fn_ratio_gt", "fp_fraction_pred", "fn_fraction_gt", "precision_err", "recall_err"]
    if fs == "uncertainty":
        return ["uncertainty_mean", "uncertainty_boundary", "low_conf_fraction", "confidence_margin"]
    if fs == "uncertainty_mc":
        return ["mc_predictive_entropy", "mc_variance", "mc_mutual_information", "mc_boundary_entropy"]
    if fs == "scalar_quality":
        return ["scalar_quality"]
    if fs == "metric_values":
        return ["dice", "bf1", "hd95_err", "assd_err"]
    if fs == "metric_discordance":
        return [
            "dice", "bf1", "hd95_err", "assd_err",
            "surface_tail_gap", "overlap_boundary_gap", "overlap_surface_gap", "boundary_surface_gap",
        ]
    if fs == "metric_discordance_plus_fpfn":
        return feature_columns("metric_discordance") + feature_columns("fpfn")
    if fs == "all_combined":
        return feature_columns("metric_discordance") + feature_columns("fpfn") + feature_columns("uncertainty")
    raise ValueError(f"Unknown feature_set={feature_set}")


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y <= r*r)


def remove_small_components(mask: np.ndarray, min_pixels: int = 64, keep_largest_if_empty: bool = True) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    lab, n = label(m)
    if n == 0:
        return m
    counts = np.bincount(lab.ravel())
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n + 1):
        if counts[i] >= int(min_pixels):
            out[lab == i] = True
    if keep_largest_if_empty and not out.any() and n > 0:
        i = int(np.argmax(counts[1:]) + 1)
        out[lab == i] = True
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    lab, n = label(m)
    if n <= 1:
        return m
    counts = np.bincount(lab.ravel())
    i = int(np.argmax(counts[1:]) + 1)
    return lab == i


def correction_candidates(pred: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """Post-processing candidates used only to define an oracle correction family.

    The oracle is an analysis/training-label device. The deployed neural refiner does
    not execute these operators and never sees ground truth at inference.
    """
    m = np.asarray(pred).astype(bool)
    c: list[tuple[str, str, np.ndarray]] = [("no_correction", "identity", m.copy())]
    # Missing-region / under-segmentation family.
    for r in (1, 2, 3):
        c.append(("recover_fn", f"dilate_r{r}", binary_dilation(m, structure=_disk(r))))
    c.append(("recover_fn", "fill_holes", binary_fill_holes(m)))
    c.append(("recover_fn", "close_r2_fill", binary_fill_holes(binary_closing(m, structure=_disk(2)))))
    # Excess-region / FP family.
    for r in (1, 2):
        c.append(("remove_fp", f"erode_r{r}", binary_erosion(m, structure=_disk(r))))
    for r in (1, 2):
        c.append(("remove_fp", f"open_r{r}", binary_opening(m, structure=_disk(r))))
    # Shallow but spatially widespread boundary irregularity.
    for r in (1, 2):
        opened = binary_opening(m, structure=_disk(r))
        closed = binary_closing(m, structure=_disk(r))
        c.append(("boundary_align", f"open_close_r{r}", binary_closing(opened, structure=_disk(r))))
        c.append(("boundary_align", f"close_open_r{r}", binary_opening(closed, structure=_disk(r))))
    if m.any():
        u8 = (m.astype(np.uint8) * 255)
        med = cv2.medianBlur(u8, 5) > 127
        c.append(("boundary_align", "median5", med))
    for dy, dx in ((-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1)):
        shifted = ndi_shift(m.astype(np.uint8), shift=(dy, dx), order=0, mode="constant", cval=0) > 0
        c.append(("boundary_align", f"shift_{dy}_{dx}", shifted))
    # Distant-component / high-tail surface-distance family.
    c.append(("tail_cleanup", "keep_largest", keep_largest_component(m)))
    for minp in (16, 32, 64, 128):
        c.append(("tail_cleanup", f"remove_components_lt{minp}", remove_small_components(m, min_pixels=minp)))
    return c


def choose_oracle_route(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5,
                        bf1_tolerance: int = 2, min_gain: float = 0.002,
                        hd95_scale: float = 16.0, assd_scale: float = 4.0) -> dict:
    """Choose a correction family using pixel-error reduction, not the diagnostic metrics.

    This avoids a circular experiment: Dice/BF1/HD95/ASSD are candidate *predictors* of
    correction type, while the oracle label is defined by reduction of FP+FN pixels among
    a predeclared correction-operator library. Exact geometric utility is recorded only as
    a secondary descriptive outcome.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    base_metrics = case_metrics(pred.astype(np.float32), gt.astype(np.float32), threshold=threshold, bf1_tolerance=bf1_tolerance)
    base_utility = exact_utility(base_metrics, hd95_scale, assd_scale)
    gt_scale = max(float(base_metrics.get("gt_pixels", 0)), 1.0)
    base_error = float(base_metrics.get("fp", 0) + base_metrics.get("fn", 0))
    best = {"route": "no_correction", "operator": "identity", "pixel_error": base_error, "metrics": base_metrics, "utility": base_utility}
    family_best: dict[str, tuple[float, str, dict, float]] = {}
    for route, op, candidate in correction_candidates(pred):
        met = case_metrics(candidate.astype(np.float32), gt.astype(np.float32), threshold=0.5, bf1_tolerance=bf1_tolerance)
        err = float(met.get("fp", 0) + met.get("fn", 0))
        util = exact_utility(met, hd95_scale, assd_scale)
        prev = family_best.get(route)
        if prev is None or err < prev[0]:
            family_best[route] = (err, op, met, util)
        if err < best["pixel_error"]:
            best = {"route": route, "operator": op, "pixel_error": err, "metrics": met, "utility": util}
    gain = float((base_error - best["pixel_error"]) / gt_scale)
    if gain < float(min_gain):
        best = {"route": "no_correction", "operator": "identity", "pixel_error": base_error, "metrics": base_metrics, "utility": base_utility}
        gain = 0.0
    return {
        "oracle_route": best["route"],
        "oracle_route_id": ROUTE_TO_ID[best["route"]],
        "oracle_operator": best["operator"],
        "oracle_gain": gain,
        "oracle_pixel_error_reduction": float(base_error - best["pixel_error"]),
        "base_pixel_error": base_error,
        "oracle_pixel_error": float(best["pixel_error"]),
        "base_utility": base_utility,
        "oracle_utility": float(best["utility"]),
        "family_best": {k: {"pixel_error": float(v[0]), "operator": v[1], "utility": float(v[3])} for k, v in family_best.items()},
    }


@dataclass(frozen=True)
class RouteWeights:
    boundary: float
    msbd: float
    tversky: float
    anchor: float
    residual: float
    alpha_fp: float
    beta_fn: float


DEFAULT_ROUTE_WEIGHTS = {
    "no_correction": RouteWeights(0.05, 0.05, 0.05, 0.30, 0.020, 0.50, 0.50),
    "recover_fn": RouteWeights(0.15, 0.15, 0.45, 0.06, 0.003, 0.30, 0.70),
    "remove_fp": RouteWeights(0.15, 0.20, 0.45, 0.08, 0.004, 0.70, 0.30),
    "boundary_align": RouteWeights(0.45, 0.35, 0.15, 0.05, 0.003, 0.50, 0.50),
    "tail_cleanup": RouteWeights(0.20, 0.60, 0.25, 0.08, 0.004, 0.65, 0.35),
}


def route_name(route_id: int) -> str:
    return ROUTES[int(route_id)]
