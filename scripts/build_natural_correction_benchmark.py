from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.discordance import (
    binary_entropy, boundary, choose_oracle_route, metric_profile_features,
    scalar_quality_from_features, uncertainty_features,
)
from src.metrics import case_metrics
from src.model_factory import architecture_from_config, build_segmentor
from src.utils import get_device


def first(v):
    if isinstance(v, (list, tuple)):
        return v[0]
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v[0].item()
    return v


def enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_features(model, image: torch.Tensor, samples: int, deterministic_prob: np.ndarray) -> dict[str, float]:
    if samples <= 1:
        ent = binary_entropy(deterministic_prob)
        pred = deterministic_prob >= 0.5
        b = boundary(pred)
        if b.any():
            from scipy.ndimage import binary_dilation
            band = binary_dilation(b, iterations=3)
            be = float(ent[band].mean()) if band.any() else float(ent.mean())
        else:
            be = float(ent.mean())
        return {
            "mc_predictive_entropy": float(ent.mean()),
            "mc_variance": 0.0,
            "mc_mutual_information": 0.0,
            "mc_boundary_entropy": be,
        }
    enable_dropout_only(model)
    probs = []
    with torch.no_grad():
        for _ in range(int(samples)):
            probs.append(torch.sigmoid(model(image)).detach().cpu().numpy()[0, 0])
    model.eval()
    arr = np.stack(probs, axis=0).astype(np.float32)
    mean_p = arr.mean(axis=0)
    pe = binary_entropy(mean_p)
    exp_ent = np.mean(np.stack([binary_entropy(x) for x in arr], axis=0), axis=0)
    mi = np.maximum(pe - exp_ent, 0.0)
    var = arr.var(axis=0)
    pred = mean_p >= 0.5
    rel = np.logical_or(pred, mean_p >= 0.2)
    if not rel.any(): rel = np.ones_like(pred, dtype=bool)
    b = boundary(pred)
    if b.any():
        from scipy.ndimage import binary_dilation
        band = binary_dilation(b, iterations=3)
    else:
        band = rel
    return {
        "mc_predictive_entropy": float(pe[rel].mean()),
        "mc_variance": float(var[rel].mean()),
        "mc_mutual_information": float(mi[rel].mean()),
        "mc_boundary_entropy": float(pe[band].mean()) if band.any() else float(pe[rel].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="Natural source-error benchmark and oracle correction-family labels.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mc-samples", type=int, default=1)
    ap.add_argument("--min-oracle-gain", type=float, default=0.002)
    ap.add_argument("--bf1-tolerance", type=int, default=2)
    ap.add_argument("--hd95-scale", type=float, default=16.0)
    ap.add_argument("--assd-scale", type=float, default=4.0)
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_segmentor(architecture_from_config(cfg), pretrained_encoder=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    ds = BRISC2025Dataset(args.manifest, args.split, args.data_root, int(cfg["image_size"]), False, cfg.get("imagenet_norm", True))
    if args.max_cases and args.max_cases < len(ds):
        # Deterministic prefix is enough because the manifest itself is randomized/fixed by split seed.
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(args.max_cases)))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    threshold = float(cfg.get("threshold", 0.5))
    rows = []
    for images, masks, meta in tqdm(loader, desc=f"natural-{args.split}", disable=os.environ.get("TQDM_DISABLE") == "1"):
        image = images.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(image)
            prob = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]
        gt = masks.numpy()[0, 0]
        pred = prob >= threshold
        met = case_metrics(prob, gt, threshold, bf1_tolerance=args.bf1_tolerance)
        f = metric_profile_features(met, args.hd95_scale, args.assd_scale)
        u = uncertainty_features(prob, pred)
        oracle = choose_oracle_route(pred, gt >= 0.5, threshold=0.5, bf1_tolerance=args.bf1_tolerance,
                                     min_gain=args.min_oracle_gain, hd95_scale=args.hd95_scale,
                                     assd_scale=args.assd_scale)
        row = {
            "case_id": str(first(meta["case_id"])), "split": args.split,
            "plane_code": str(first(meta["plane_code"])), "tumor_code": str(first(meta["tumor_code"])),
            **met, **f, **u,
            "scalar_quality": scalar_quality_from_features(f),
            "oracle_route": oracle["oracle_route"], "oracle_route_id": oracle["oracle_route_id"],
            "oracle_operator": oracle["oracle_operator"], "oracle_gain": oracle["oracle_gain"],
            "oracle_pixel_error_reduction": oracle.get("oracle_pixel_error_reduction", 0.0),
            "base_pixel_error": oracle.get("base_pixel_error", np.nan), "oracle_pixel_error": oracle.get("oracle_pixel_error", np.nan),
            "base_utility": oracle["base_utility"], "oracle_utility": oracle["oracle_utility"],
        }
        row.update(mc_features(model, image, args.mc_samples, prob))
        rows.append(row)

    final_rows = rows
    df = pd.DataFrame(final_rows)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = {
        "checkpoint": str(args.checkpoint), "split": args.split, "n_cases": int(len(df)),
        "architecture": architecture_from_config(cfg), "mc_samples": int(args.mc_samples),
        "min_oracle_gain": float(args.min_oracle_gain), "route_counts": df.oracle_route.value_counts().to_dict(),
        "mean_oracle_gain": float(df.oracle_gain.mean()),
        "role": "Ground truth is used only for diagnostic analysis and training-route labels, never at deployment.",
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.groupby("oracle_route")[["dice", "bf1", "hd95", "assd", "oracle_gain"]].mean(numeric_only=True).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
