from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import BRISC2025Dataset
from src.debug_utils import setup_logger
from src.losses import BCEDiceLoss
from src.model_factory import architecture_from_config, build_segmentor
from src.utils import get_device, load_yaml, seed_everything


def epoch_pass(model, loader, criterion, device, threshold, amp, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    losses, dices, ious = [], [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= threshold).float()
                inter = (pred * masks).sum((1, 2, 3))
                ps, gs = pred.sum((1, 2, 3)), masks.sum((1, 2, 3))
                d = (2 * inter + 1e-7) / (ps + gs + 1e-7)
                u = ps + gs - inter
                j = (inter + 1e-7) / (u + 1e-7)
            losses.append(float(loss.detach().cpu()))
            dices += d.detach().cpu().tolist()
            ious += j.detach().cpu().tolist()
    return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser(description="Compute-matched BCE+Dice continuation control for GMGD.")
    ap.add_argument("--config", default="configs/gmgd.yaml")
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    gcfg = load_yaml(args.config)
    seed = int(gcfg.get("seed", 42) if args.seed is None else args.seed)
    seed_everything(seed)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint or gcfg["baseline_checkpoint"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bcfg = ckpt["config"]

    out = Path(args.output_dir or f"outputs/gmgd_control_seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out / "train.log", f"control_seed{seed}")

    image_size = int(bcfg["image_size"])
    batch_size = int(gcfg.get("batch_size", bcfg.get("batch_size", 8)))
    num_workers = int(gcfg.get("num_workers", bcfg.get("num_workers", 2)))
    train_ds = BRISC2025Dataset(args.manifest, "train", args.data_root, image_size, True, bcfg.get("imagenet_norm", True))
    val_ds = BRISC2025Dataset(args.manifest, "val", args.data_root, image_size, False, bcfg.get("imagenet_norm", True))
    kw = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        kw["persistent_workers"] = bool(gcfg.get("persistent_workers", True))
        kw["prefetch_factor"] = int(gcfg.get("prefetch_factor", 2))
    seg_gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=seg_gen, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)

    model = build_segmentor(architecture_from_config(bcfg), pretrained_encoder=False).to(device)
    model.load_state_dict(ckpt["model"])
    criterion = BCEDiceLoss(gcfg.get("bce_weight", 0.5), gcfg.get("dice_weight", 0.5))
    epochs = int(gcfg.get("segmentor_epochs", 12))
    opt = AdamW(model.parameters(), lr=float(gcfg.get("segmentor_lr", 8e-5)), weight_decay=float(gcfg.get("weight_decay", 1e-4)))
    sch = CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = GradScaler("cuda", enabled=bool(gcfg.get("amp", True) and device.type == "cuda"))
    threshold = float(gcfg.get("threshold", bcfg.get("threshold", 0.5)))

    initial_val = epoch_pass(model, val_loader, criterion, device, threshold, gcfg.get("amp", True))
    best_dice = float(initial_val[1])
    initial_payload = {"model": model.state_dict(), "config": bcfg, "gmgd_v2_1_config": gcfg, "seed": seed, "epoch": 0,
                       "source_checkpoint": str(ckpt_path), "branch": "GMGD_V2_1_COMPUTE_MATCHED_CONTROL_SOURCE"}
    torch.save(initial_payload, out / "best_dice.pt")
    log.info("control epoch0 source val_dice=%.6f included in validation checkpoint selection", best_dice)
    fields = ["epoch", "lr", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou", "seconds"]
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for epoch in range(1, epochs + 1):
            seg_gen.manual_seed(seed + 10000 + epoch)
            random.seed(seed + 30000 + epoch)
            t0 = time.time()
            tr = epoch_pass(model, train_loader, criterion, device, threshold, gcfg.get("amp", True), opt, scaler)
            va = epoch_pass(model, val_loader, criterion, device, threshold, gcfg.get("amp", True))
            sec = time.time() - t0
            row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=tr[0], train_dice=tr[1], train_iou=tr[2], val_loss=va[0], val_dice=va[1], val_iou=va[2], seconds=sec)
            w.writerow(row); f.flush()
            log.info("epoch=%d/%d train_loss=%.5f train_dice=%.5f val_loss=%.5f val_dice=%.5f val_iou=%.5f sec=%.1f", epoch, epochs, tr[0], tr[1], va[0], va[1], va[2], sec)
            payload = {"model": model.state_dict(), "config": bcfg, "gmgd_v2_1_config": gcfg, "seed": seed, "epoch": epoch, "source_checkpoint": str(ckpt_path), "branch": "GMGD_V2_1_COMPUTE_MATCHED_CONTROL"}
            torch.save(payload, out / "final.pt")
            if va[1] > best_dice:
                best_dice = va[1]
                torch.save(payload, out / "best_dice.pt")
            sch.step()
    log.info("CONTROL DONE final=%s best_val_dice=%.6f", out / "final.pt", best_dice)


if __name__ == "__main__":
    main()
