from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _to_2d(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D sample, got {arr.shape}")
    if arr.shape[0] <= 4 and arr.shape[1] > 4 and arr.shape[2] > 4:
        return arr.max(axis=0) if is_mask else arr.astype(np.float32).mean(axis=0)
    if arr.shape[-1] <= 4 and arr.shape[0] > 4 and arr.shape[1] > 4:
        return arr.max(axis=-1) if is_mask else arr.astype(np.float32).mean(axis=-1)
    raise ValueError(f"Cannot infer channel axis from shape {arr.shape}")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN/Inf")
    if float(image.max()) > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(mask).all():
        raise ValueError("Mask contains NaN/Inf")
    if float(mask.max()) > 1.5:
        mask = mask / 255.0
    return (mask > 0.5).astype(np.float32)


def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.25:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    if random.random() < 0.35:
        k = random.choice([1, 2, 3])
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
    if random.random() < 0.30:
        gain = random.uniform(0.9, 1.1)
        bias = random.uniform(-0.04, 0.04)
        image = np.clip(image * gain + bias, 0.0, 1.0)
    return image, mask


class BRISC2025Dataset(Dataset):
    """Manifest-driven loader compatible with raw BRISC2025 and the preprocessed NPY archive."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        data_root: str | Path = "datasets/brisc2025",
        image_size: int = 256,
        train: bool = False,
        imagenet_norm: bool = True,
    ):
        self.data_root = Path(data_root).resolve()
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples for split={split}")
        self.image_size = int(image_size)
        self.train = bool(train)
        self.imagenet_norm = bool(imagenet_norm)

        if "data_backend" in self.df.columns:
            vals = self.df["data_backend"].dropna().astype(str).str.lower().unique().tolist()
            self.backend = vals[0] if vals else "raw"
        elif {"x_path", "y_path", "sample_index"}.issubset(self.df.columns):
            self.backend = "npy"
        else:
            self.backend = "raw"

        self.X = self.Y = None
        if self.backend == "npy":
            x_paths = self.df["x_path"].astype(str).unique().tolist()
            y_paths = self.df["y_path"].astype(str).unique().tolist()
            if len(x_paths) != 1 or len(y_paths) != 1:
                raise ValueError("Each split must point to one X array and one Y array")
            self.x_path = self._resolve(x_paths[0])
            self.y_path = self._resolve(y_paths[0])
            try:
                self.X = np.load(self.x_path, mmap_mode="r", allow_pickle=False)
                self.Y = np.load(self.y_path, mmap_mode="r", allow_pickle=False)
            except (EOFError, ValueError, OSError) as exc:
                def _desc(path: Path) -> str:
                    try:
                        return f"{path} ({path.stat().st_size} bytes)"
                    except OSError:
                        return str(path)
                raise RuntimeError(
                    "BRISC2025 manifest selects the preprocessed NPY backend, but an array "
                    "is unreadable. This is commonly caused by a zero-byte/partial NPY file. "
                    f"X={_desc(self.x_path)}; Y={_desc(self.y_path)}. "
                    "Regenerate the manifest with scripts/prepare_brisc2025.py; when the raw "
                    "segmentation_task tree exists, the pipeline will now prefer it automatically."
                ) from exc
            if len(self.X) != len(self.Y):
                raise RuntimeError(
                    f"BRISC2025 NPY X/Y length mismatch: X={len(self.X)} Y={len(self.Y)} "
                    f"for split={split}"
                )

    def _resolve(self, p: str) -> Path:
        p = Path(str(p))
        return p if p.is_absolute() else self.data_root / p

    def __len__(self) -> int:
        return len(self.df)

    def _read(self, row):
        if self.backend == "npy":
            i = int(row.sample_index)
            image = _normalize_image(_to_2d(self.X[i], is_mask=False))
            mask = _normalize_mask(_to_2d(self.Y[i], is_mask=True))
            image_locator = f"{self.x_path}#{i}"
            mask_locator = f"{self.y_path}#{i}"
        else:
            ip = self._resolve(row.image_path)
            mp = self._resolve(row.mask_path)
            image = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(f"Could not read image/mask: {ip} / {mp}")
            image = _normalize_image(image)
            mask = _normalize_mask(mask)
            image_locator, mask_locator = str(ip), str(mp)
        return image, mask, image_locator, mask_locator

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image, mask, image_locator, mask_locator = self._read(row)
        if self.train:
            image, mask = _augment(image, mask)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

        image3 = np.repeat(image[..., None], 3, axis=-1).astype(np.float32)
        if self.imagenet_norm:
            image3 = (image3 - IMAGENET_MEAN) / IMAGENET_STD

        image_t = torch.from_numpy(np.ascontiguousarray(image3.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        meta = {
            "case_id": str(row.get("case_id", f"{self.backend}_{idx:05d}")),
            "plane_code": str(row.get("plane_code", "unknown")),
            "plane_label": str(row.get("plane_label", "unknown")),
            "tumor_code": str(row.get("tumor_code", "unknown")),
            "tumor_label": str(row.get("tumor_label", "unknown")),
            "image_path": image_locator,
            "mask_path": mask_locator,
            "sample_index": int(row.get("sample_index", -1)) if not pd.isna(row.get("sample_index", -1)) else -1,
            "lesion_pixels": float(mask.sum()),
        }
        return image_t, mask_t, meta
