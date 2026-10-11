from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, shift as ndi_shift
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.discordance import metric_profile_features, scalar_quality_from_features
from src.metrics import case_metrics


def disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y <= r*r)


def add_distant_island(mask: np.ndarray, rng: np.random.Generator, radius: int) -> np.ndarray:
    m = mask.astype(bool).copy()
    h, w = m.shape
    # Prefer corners/edges far from the lesion. Try deterministic random candidates.
    yy, xx = np.where(m)
    center = np.array([float(yy.mean()), float(xx.mean())]) if len(yy) else np.array([h / 2.0, w / 2.0])
    candidates = [
        (radius + 2, radius + 2), (radius + 2, w - radius - 3),
        (h - radius - 3, radius + 2), (h - radius - 3, w - radius - 3),
    ]
    for _ in range(12):
        candidates.append((int(rng.integers(radius+2, h-radius-2)), int(rng.integers(radius+2, w-radius-2))))
    cy, cx = max(candidates, key=lambda p: float(np.linalg.norm(np.array(p) - center)))
    y, x = np.ogrid[:h, :w]
    island = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    m[island] = True
    return m


def cut_hole(mask: np.ndarray, rng: np.random.Generator, radius: int) -> np.ndarray:
    m = mask.astype(bool).copy()
    yy, xx = np.where(m)
    if len(yy) == 0:
        return m
    i = int(rng.integers(0, len(yy)))
    cy, cx = int(yy[i]), int(xx[i])
    y, x = np.ogrid[:m.shape[0], :m.shape[1]]
    hole = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    m[hole] = False
    return m


def corruptions(gt: np.ndarray, rng: np.random.Generator) -> list[tuple[str, str, np.ndarray]]:
    m = gt.astype(bool)
    out: list[tuple[str, str, np.ndarray]] = [("no_correction", "identity", m.copy())]
    for r in (1, 3):
        out.append(("recover_fn", f"erosion_r{r}", binary_erosion(m, structure=disk(r))))
    out.append(("recover_fn", "interior_hole", cut_hole(m, rng, radius=max(3, int(np.sqrt(max(m.sum(), 1)) * 0.04)))))
    for r in (1, 3):
        out.append(("remove_fp", f"dilation_r{r}", binary_dilation(m, structure=disk(r))))
    # Shallow widespread boundary displacement: rigid shifts preserve area but move the contour.
    for s in (2, 4):
        sh = ndi_shift(m.astype(np.uint8), shift=(s, -s), order=0, mode="constant", cval=0) > 0
        out.append(("boundary_align", f"shift_{s}px", sh))
    # Tail geometry: components far from the lesion; use enough boundary mass to influence high-percentile distances.
    base_r = max(5, min(12, int(np.sqrt(max(m.sum(), 1)) * 0.06)))
    out.append(("tail_cleanup", f"distant_island_r{base_r}", add_distant_island(m, rng, base_r)))
    out.append(("tail_cleanup", f"distant_island_r{base_r+3}", add_distant_island(m, rng, base_r + 3)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Controlled corruption benchmark for metric-discordance hypotheses.")
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    ap.add_argument("--seed", type=int, default=4201)
    ap.add_argument("--bf1-tolerance", type=int, default=2)
    ap.add_argument("--hd95-scale", type=float, default=16.0)
    ap.add_argument("--assd-scale", type=float, default=4.0)
    args = ap.parse_args()

    ds = BRISC2025Dataset(args.manifest, args.split, args.data_root, args.image_size, False, False)
    indices = np.arange(len(ds))
    rng = np.random.default_rng(args.seed)
    if args.max_cases and args.max_cases < len(indices):
        indices = np.sort(rng.choice(indices, size=args.max_cases, replace=False))

    rows = []
    for idx in tqdm(indices, desc=f"synthetic-{args.split}", disable=os.environ.get("TQDM_DISABLE") == "1"):
        _, mask_t, meta = ds[int(idx)]
        gt = mask_t.numpy()[0] >= 0.5
        local_rng = np.random.default_rng(args.seed + int(idx) * 7919)
        for route, corruption, pred in corruptions(gt, local_rng):
            met = case_metrics(pred.astype(np.float32), gt.astype(np.float32), 0.5, bf1_tolerance=args.bf1_tolerance)
            f = metric_profile_features(met, args.hd95_scale, args.assd_scale)
            row = {
                "case_id": str(meta["case_id"]), "split": args.split,
                "corruption_type": corruption, "target_route": route,
                **met, **f,
            }
            row["scalar_quality"] = scalar_quality_from_features(f)
            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = {
        "split": args.split, "n_source_cases": int(len(indices)), "n_rows": int(len(df)),
        "classes": df.target_route.value_counts().to_dict(),
        "purpose": "controlled error-geometry benchmark; no model predictions are used",
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.groupby("target_route")[["dice", "bf1", "hd95", "assd"]].mean(numeric_only=True).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
