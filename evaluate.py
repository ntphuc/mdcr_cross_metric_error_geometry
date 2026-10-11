from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.metrics import case_metrics
from src.model_factory import architecture_from_config, build_segmentor
from src.utils import get_device, save_json


def first(v):
    if isinstance(v, (list, tuple)):
        return v[0]
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v[0].item()
    return v


def finite_mean(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    gcfg = ckpt.get("gmgd_v2_1_config") or {}
    threshold = float(cfg.get("threshold", 0.5) if args.threshold is None else args.threshold)
    bf1_tolerance = int(gcfg.get("bf1_tolerance", cfg.get("bf1_tolerance", 2)))
    model = build_segmentor(architecture_from_config(cfg), pretrained_encoder=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds = BRISC2025Dataset(args.manifest, args.split, args.data_root, cfg["image_size"], False, cfg.get("imagenet_norm", True))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    with torch.no_grad():
        for images, masks, meta in tqdm(loader, disable=os.environ.get("TQDM_DISABLE") == "1"):
            logits = model(images.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            gt = masks.numpy()[0, 0]
            m = case_metrics(prob, gt, threshold, bf1_tolerance=bf1_tolerance)
            row = {
                "case_id": first(meta["case_id"]),
                "plane_code": first(meta["plane_code"]), "plane_label": first(meta["plane_label"]),
                "tumor_code": first(meta["tumor_code"]), "tumor_label": first(meta["tumor_label"]),
                "image_path": first(meta["image_path"]), "mask_path": first(meta["mask_path"]),
                **m,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_case_metrics.csv", index=False)

    summary = {
        "checkpoint": str(args.checkpoint), "split": args.split, "n_cases": len(df), "threshold": threshold, "bf1_tolerance_px": bf1_tolerance,
        "dice_mean": float(df.dice.mean()), "dice_std": float(df.dice.std(ddof=0)),
        "iou_mean": float(df.iou.mean()), "precision_mean": float(df.precision.mean()),
        "recall_mean": float(df.recall.mean()), "specificity_mean": float(df.specificity.mean()),
        "bf1_mean": float(df.bf1.mean()),
        "hd95_mean_finite": finite_mean(df.hd95), "assd_mean_finite": finite_mean(df.assd),
        "missed_lesion_cases": int(((df.gt_pixels > 0) & (df.pred_pixels == 0)).sum()),
    }
    save_json(summary, out / "summary.json")
    print(pd.Series(summary).to_string())
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
