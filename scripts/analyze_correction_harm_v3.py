from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.discordance import ROUTES
from src.mdcr_v3_model import MDCRV3Model
from src.metrics import case_metrics
from src.model_factory import architecture_from_config, build_segmentor
from src.utils import get_device


def first(v):
    if isinstance(v, (list, tuple)): return v[0]
    if torch.is_tensor(v): return v.item() if v.numel() == 1 else v[0].item()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-checkpoint", required=True); ap.add_argument("--refined-checkpoint", required=True)
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv"); ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--split", choices=["val", "test"], default="val"); ap.add_argument("--device", default="auto"); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--relevant-dilation", type=int, default=5)
    args = ap.parse_args(); device = get_device(args.device)
    sck = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False); rc = torch.load(args.refined_checkpoint, map_location="cpu", weights_only=False)
    bcfg = sck["config"]; cfg = rc.get("mdcr_v3_config", {}) or {}; arch = architecture_from_config(bcfg)
    source = build_segmentor(arch, pretrained_encoder=False); source.load_state_dict(sck["model"]); source.to(device).eval()
    base = build_segmentor(arch, pretrained_encoder=False)
    model = MDCRV3Model(base, router_width=int(cfg.get("router_width", 32)), router_hidden=int(cfg.get("router_hidden", 96)),
                        router_dropout=float(cfg.get("router_dropout", 0.15)), expert_width=int(cfg.get("expert_width", 32)),
                        residual_scale=float(cfg.get("residual_scale", 2.0)), correction_margin=float(rc.get("correction_margin", 0.0)))
    model.load_state_dict(rc["model"]); model.to(device).eval(); threshold = float(cfg.get("threshold", bcfg.get("threshold", 0.5))); bf1 = int(cfg.get("bf1_tolerance", 2))
    ds = BRISC2025Dataset(args.manifest, args.split, args.data_root, int(bcfg["image_size"]), False, bcfg.get("imagenet_norm", True)); loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows = []; identity_max = 0.0
    with torch.no_grad():
        for images, masks, meta in tqdm(loader, desc=f"harm-v3-{args.split}", disable=os.environ.get("TQDM_DISABLE") == "1"):
            im = images.to(device); gt = masks.numpy()[0, 0] >= 0.5
            sp = torch.sigmoid(source(im)).cpu().numpy()[0, 0]
            out = model(im, return_parts=True); rp = torch.sigmoid(out[0]).cpu().numpy()[0, 0]; gp = out[4].cpu().numpy()[0]
            rid = int(np.argmax(gp))
            if rid == 0: identity_max = max(identity_max, float((out[0] - out[1]).abs().max().cpu()))
            sh, rh = sp >= threshold, rp >= threshold
            source_correct = sh == gt; source_wrong = ~source_correct; refined_correct = rh == gt
            relevant = binary_dilation(np.logical_or(np.logical_or(gt, sh), rh), iterations=max(0, args.relevant_dilation))
            corrected = np.logical_and(source_wrong, refined_correct); harmed = np.logical_and(source_correct, ~refined_correct)
            corrected_rel = np.logical_and(corrected, relevant); harmed_rel = np.logical_and(harmed, relevant)
            sw_rel = np.logical_and(source_wrong, relevant); sc_rel = np.logical_and(source_correct, relevant)
            sm = case_metrics(sp, gt.astype(np.float32), threshold, bf1_tolerance=bf1); rm = case_metrics(rp, gt.astype(np.float32), threshold, bf1_tolerance=bf1)
            rows.append({"case_id": str(first(meta["case_id"])), "predicted_route": ROUTES[rid],
                         "source_wrong_relevant": int(sw_rel.sum()), "source_correct_relevant": int(sc_rel.sum()), "corrected_pixels": int(corrected_rel.sum()), "harmed_pixels": int(harmed_rel.sum()),
                         "correction_rate": float(corrected_rel.sum() / max(sw_rel.sum(), 1)), "harm_rate": float(harmed_rel.sum() / max(sc_rel.sum(), 1)),
                         "net_corrected_pixels": int(corrected_rel.sum() - harmed_rel.sum()), "dice_delta": rm["dice"] - sm["dice"], "bf1_delta": rm["bf1"] - sm["bf1"],
                         "hd95_reduction": sm["hd95"] - rm["hd95"] if np.isfinite(sm["hd95"]) and np.isfinite(rm["hd95"]) else np.nan,
                         "assd_reduction": sm["assd"] - rm["assd"] if np.isfinite(sm["assd"]) and np.isfinite(rm["assd"]) else np.nan})
    df = pd.DataFrame(rows); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "correction_harm_per_case.csv", index=False)
    no = df[df.predicted_route == "no_correction"]
    summary = {"split": args.split, "n_cases": len(df), "mean_correction_rate": float(df.correction_rate.mean()), "mean_harm_rate": float(df.harm_rate.mean()),
               "total_corrected_pixels": int(df.corrected_pixels.sum()), "total_harmed_pixels": int(df.harmed_pixels.sum()), "net_corrected_pixels": int(df.net_corrected_pixels.sum()),
               "cases_dice_improved": int((df.dice_delta > 1e-9).sum()), "cases_dice_harmed": int((df.dice_delta < -1e-9).sum()),
               "no_correction_cases": int(len(no)), "no_correction_total_harmed_pixels": int(no.harmed_pixels.sum()) if len(no) else 0,
               "no_correction_identity_max_abs_logit_diff": float(identity_max),
               "preservation_contract_pass": bool(identity_max == 0.0 and (int(no.harmed_pixels.sum()) if len(no) else 0) == 0),
               "interpretation": "In v3, route no_correction is exact identity because the source segmentor is permanently frozen."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
