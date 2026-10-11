from __future__ import annotations

import argparse
import csv
import faulthandler
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BRISC2025Dataset
from src.debug_utils import gpu_memory_string, log_runtime_environment, setup_logger
from src.losses import BCEDiceLoss
from src.model_factory import architecture_from_config, build_segmentor
from src.utils import get_device, load_yaml, seed_everything


# Makes low-level Python crashes much easier to diagnose in Slurm logs.
faulthandler.enable(all_threads=True)


class RuntimeState:
    logger = None
    model = None
    optimizer = None
    run_dir = None
    cfg = None
    seed = None
    epoch = 0
    batch = 0
    phase = "startup"


STATE = RuntimeState()


def _save_interrupted_checkpoint(reason: str) -> None:
    if STATE.model is None or STATE.run_dir is None:
        return
    try:
        path = Path(STATE.run_dir) / "interrupted.pt"
        payload = {
            "model": STATE.model.state_dict(),
            "config": STATE.cfg,
            "seed": STATE.seed,
            "epoch": STATE.epoch,
            "batch": STATE.batch,
            "phase": STATE.phase,
            "reason": reason,
        }
        if STATE.optimizer is not None:
            payload["optimizer"] = STATE.optimizer.state_dict()
        torch.save(payload, path)
        if STATE.logger:
            STATE.logger.error("Emergency checkpoint saved: %s", path)
    except Exception:
        if STATE.logger:
            STATE.logger.exception("Could not save interrupted checkpoint")


def _signal_handler(signum, frame):
    name = signal.Signals(signum).name if signum in [s.value for s in signal.Signals] else str(signum)
    msg = f"Received signal {name} ({signum}) at phase={STATE.phase}, epoch={STATE.epoch}, batch={STATE.batch}."
    if STATE.logger:
        STATE.logger.error(msg)
        STATE.logger.error("This commonly means the scheduler/job manager stopped the process (for example TIME LIMIT or cancellation).")
    else:
        print(msg, file=sys.stderr, flush=True)
    _save_interrupted_checkpoint(msg)
    raise SystemExit(128 + signum)


for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, _signal_handler)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    use_amp=True,
    threshold=0.5,
    *,
    logger=None,
    epoch=0,
    phase="train",
    log_every=25,
    concise=False,
):
    train_mode = optimizer is not None
    model.train(train_mode)
    scaler = run_epoch.scaler if train_mode else None
    losses, dices, ious = [], [], []

    n_batches = len(loader)
    STATE.phase = phase
    STATE.epoch = int(epoch)
    last_end = time.perf_counter()
    epoch_start = last_end

    progress = loader if concise else tqdm(
        loader,
        total=n_batches,
        leave=False,
        file=sys.stdout,
        dynamic_ncols=False,
        desc=f"{phase} e{epoch:03d}",
    )

    for batch_idx, (images, masks, _) in enumerate(progress, start=1):
        STATE.batch = batch_idx
        now = time.perf_counter()
        data_wait_s = now - last_end
        step_start = now

        try:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            enabled = bool(use_amp and device.type == "cuda")
            with autocast(device_type=device.type, enabled=enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {phase} epoch={epoch} batch={batch_idx}: {loss.item()}")

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            losses.append(float(loss.detach().cpu()))
            pred = (torch.sigmoid(logits.detach()) >= threshold).float()
            dims = (1, 2, 3)
            inter = (pred * masks).sum(dim=dims)
            pred_sum = pred.sum(dim=dims)
            gt_sum = masks.sum(dim=dims)
            dice_batch = (2.0 * inter + 1e-7) / (pred_sum + gt_sum + 1e-7)
            union = pred_sum + gt_sum - inter
            iou_batch = (inter + 1e-7) / (union + 1e-7)
            dices.extend(dice_batch.cpu().tolist())
            ious.extend(iou_batch.cpu().tolist())

            step_s = time.perf_counter() - step_start
            if not concise:
                progress.set_postfix(loss=f"{losses[-1]:.4f}", dice=f"{float(dice_batch.mean()):.3f}")

            should_log = (not concise) and (batch_idx == 1 or batch_idx % max(1, int(log_every)) == 0 or batch_idx == n_batches)
            if should_log and logger is not None:
                elapsed = time.perf_counter() - epoch_start
                avg_batch_s = elapsed / batch_idx
                eta_s = avg_batch_s * (n_batches - batch_idx)
                logger.info(
                    "%s epoch=%d batch=%d/%d loss=%.5f batch_dice=%.5f "
                    "data_wait=%.3fs step=%.3fs elapsed=%.1fs eta=%.1fs %s",
                    phase,
                    epoch,
                    batch_idx,
                    n_batches,
                    losses[-1],
                    float(dice_batch.mean()),
                    data_wait_s,
                    step_s,
                    elapsed,
                    eta_s,
                    gpu_memory_string(device),
                )
        except Exception:
            if logger is not None:
                logger.exception(
                    "FAILED inside %s loop at epoch=%d batch=%d/%d | image_shape=%s mask_shape=%s | %s",
                    phase,
                    epoch,
                    batch_idx,
                    n_batches,
                    tuple(images.shape) if hasattr(images, "shape") else "unknown",
                    tuple(masks.shape) if hasattr(masks, "shape") else "unknown",
                    gpu_memory_string(device),
                )
            raise
        finally:
            last_end = time.perf_counter()

    if not losses:
        raise RuntimeError(f"No batches were processed for phase={phase}")
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices)), "iou": float(np.mean(ious))}


run_epoch.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--log-every", type=int, default=None, help="Write a detailed batch log every N batches.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    seed_everything(seed)
    device = get_device(args.device)

    run_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{cfg['experiment_name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "train.log", name=f"train_seed{seed}")
    STATE.logger = logger
    STATE.run_dir = run_dir
    STATE.cfg = cfg
    STATE.seed = seed

    try:
        logger.info("===== TRAIN START =====")
        logger.info("command=%s", " ".join(sys.argv))
        logger.info("cwd=%s", Path.cwd())
        logger.info("config=%s", cfg)
        log_runtime_environment(logger, device)

        STATE.phase = "dataset_init"
        train_ds = BRISC2025Dataset(args.manifest, "train", args.data_root, cfg["image_size"], True, cfg.get("imagenet_norm", True))
        val_ds = BRISC2025Dataset(args.manifest, "val", args.data_root, cfg["image_size"], False, cfg.get("imagenet_norm", True))
        num_workers = int(cfg.get("num_workers", 4))
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and num_workers > int(slurm_cpus):
            logger.warning(
                "num_workers=%d is greater than SLURM_CPUS_PER_TASK=%s. This can slow/stall loading. "
                "Consider num_workers <= allocated CPUs.",
                num_workers,
                slurm_cpus,
            )

        loader_kwargs = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=False, **loader_kwargs)

        logger.info(
            "dataset backend train=%s val=%s | train_samples=%d val_samples=%d | batch_size=%d | train_batches=%d val_batches=%d",
            getattr(train_ds, "backend", "unknown"),
            getattr(val_ds, "backend", "unknown"),
            len(train_ds),
            len(val_ds),
            int(cfg["batch_size"]),
            len(train_loader),
            len(val_loader),
        )
        logger.info(
            "loader num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
            num_workers,
            device.type == "cuda",
            loader_kwargs.get("persistent_workers", False),
            loader_kwargs.get("prefetch_factor", "n/a"),
        )

        # Read one item before constructing the model. Dataset/path errors then show immediately.
        sample_image, sample_mask, sample_meta = train_ds[0]
        logger.info(
            "dataset preflight OK | image_shape=%s mask_shape=%s image_dtype=%s mask_dtype=%s case_id=%s lesion_pixels=%s",
            tuple(sample_image.shape),
            tuple(sample_mask.shape),
            sample_image.dtype,
            sample_mask.dtype,
            sample_meta.get("case_id"),
            sample_meta.get("lesion_pixels"),
        )

        STATE.phase = "model_init"
        model = build_segmentor(architecture_from_config(cfg), pretrained_encoder=bool(cfg.get("pretrained_encoder", True))).to(device)
        STATE.model = model
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model initialized | params=%d trainable=%d | %s", n_params, n_trainable, gpu_memory_string(device))

        criterion = BCEDiceLoss(cfg.get("bce_weight", 0.5), cfg.get("dice_weight", 0.5))
        optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))
        STATE.optimizer = optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["epochs"])))
        log_every = int(cfg.get("log_every_batches", 25) if args.log_every is None else args.log_every)
        concise = bool(cfg.get("concise_epoch_logging", False))

        best = -1.0
        bad_epochs = 0
        history_path = run_dir / "history.csv"
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou", "seconds"])
            writer.writeheader()

            for epoch in range(1, int(cfg["epochs"]) + 1):
                STATE.epoch = epoch
                STATE.batch = 0
                t0 = time.time()
                if not concise:
                    logger.info("===== EPOCH %d/%d START | lr=%.8g =====", epoch, int(cfg["epochs"]), optimizer.param_groups[0]["lr"])

                train_m = run_epoch(
                    model, train_loader, criterion, device, optimizer,
                    cfg.get("amp", True), cfg.get("threshold", 0.5),
                    logger=logger, epoch=epoch, phase="train", log_every=log_every, concise=concise,
                )
                with torch.no_grad():
                    val_m = run_epoch(
                        model, val_loader, criterion, device, None,
                        cfg.get("amp", True), cfg.get("threshold", 0.5),
                        logger=logger, epoch=epoch, phase="val", log_every=log_every, concise=concise,
                    )
                scheduler.step()

                epoch_seconds = time.time() - t0
                row = {
                    "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_m["loss"], "train_dice": train_m["dice"], "train_iou": train_m["iou"],
                    "val_loss": val_m["loss"], "val_dice": val_m["dice"], "val_iou": val_m["iou"],
                    "seconds": epoch_seconds,
                }
                writer.writerow(row)
                f.flush()
                improved = val_m["dice"] > best + float(cfg.get("min_delta", 0.0005))
                if improved:
                    best = val_m["dice"]
                    bad_epochs = 0
                    ckpt_path = run_dir / "best.pt"
                    torch.save({
                        "model": model.state_dict(), "config": cfg, "seed": seed,
                        "epoch": epoch, "best_val_dice": best,
                    }, ckpt_path)
                else:
                    bad_epochs += 1

                stop_now = bad_epochs >= int(cfg.get("patience", 8))
                status = "best" if improved else ("early_stop" if stop_now else f"bad={bad_epochs}")
                logger.info(
                    "epoch=%03d/%03d train_loss=%.4f train_dice=%.4f train_iou=%.4f "
                    "val_loss=%.4f val_dice=%.4f val_iou=%.4f sec=%.1f status=%s",
                    epoch, int(cfg["epochs"]), train_m["loss"], train_m["dice"], train_m["iou"],
                    val_m["loss"], val_m["dice"], val_m["iou"], epoch_seconds, status,
                )
                if stop_now:
                    break

        STATE.phase = "finished"
        logger.info("===== TRAIN FINISHED =====")
        logger.info("Best val Dice: %.4f", best)
        logger.info("Checkpoint: %s", run_dir / "best.pt")
        logger.info("History: %s", history_path)
    except BaseException as exc:
        STATE.phase = f"crashed:{STATE.phase}"
        if isinstance(exc, KeyboardInterrupt):
            logger.error("Training interrupted by keyboard.")
        elif isinstance(exc, SystemExit):
            logger.error("Training exited: %s", exc)
        else:
            logger.exception(
                "FATAL TRAINING ERROR | phase=%s epoch=%d batch=%d | %s",
                STATE.phase,
                STATE.epoch,
                STATE.batch,
                gpu_memory_string(device),
            )
            _save_interrupted_checkpoint(repr(exc))
        raise


if __name__ == "__main__":
    main()
