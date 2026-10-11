from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import torch


def setup_logger(log_path: str | Path, name: str = "brisc") -> logging.Logger:
    """Create a timestamped logger that writes to both stdout and a persistent file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Re-create handlers so repeated invocations in notebooks/tests do not duplicate lines.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging to %s", log_path.resolve())
    return logger


def _safe_cuda_device_name(index: int = 0) -> str:
    try:
        return torch.cuda.get_device_name(index)
    except Exception as exc:
        return f"<unavailable: {exc}>"


def gpu_memory_string(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "gpu_mem=n/a"
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(idx) / 1024**2
        reserved = torch.cuda.memory_reserved(idx) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(idx) / 1024**2
        free, total = torch.cuda.mem_get_info(idx)
        return (
            f"gpu_mem_alloc={allocated:.0f}MB reserved={reserved:.0f}MB "
            f"peak={max_allocated:.0f}MB free={free / 1024**2:.0f}MB total={total / 1024**2:.0f}MB"
        )
    except Exception as exc:
        return f"gpu_mem=<unavailable: {exc}>"


def log_runtime_environment(logger: logging.Logger, device: torch.device) -> None:
    logger.info("===== RUNTIME ENVIRONMENT =====")
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    logger.info("torch=%s", torch.__version__)
    logger.info("requested/resolved device=%s", device)
    logger.info("torch.cuda.is_available=%s", torch.cuda.is_available())
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    logger.info("SLURM_JOB_ID=%s", os.environ.get("SLURM_JOB_ID", "<unset>"))
    logger.info("SLURM_JOB_NAME=%s", os.environ.get("SLURM_JOB_NAME", "<unset>"))
    logger.info("SLURM_CPUS_PER_TASK=%s", os.environ.get("SLURM_CPUS_PER_TASK", "<unset>"))
    logger.info("SLURM_MEM_PER_NODE=%s", os.environ.get("SLURM_MEM_PER_NODE", "<unset>"))
    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            logger.info("cuda_device_count=%d", count)
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    "cuda[%d]=%s capability=%d.%d total_memory=%.1fGB",
                    i,
                    _safe_cuda_device_name(i),
                    props.major,
                    props.minor,
                    props.total_memory / 1024**3,
                )
            logger.info("current_cuda_device=%d", torch.cuda.current_device())
            logger.info("%s", gpu_memory_string(device))
        except Exception:
            logger.exception("Failed while reading CUDA device information")
    logger.info("===============================")
