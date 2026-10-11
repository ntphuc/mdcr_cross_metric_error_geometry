from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_brisc2025 import _npy_ready  # noqa: E402
from src.dataset import BRISC2025Dataset  # noqa: E402


def _write_raw_case(root: Path, split: str, name: str) -> None:
    images = root / "segmentation_task" / split / "images"
    masks = root / "segmentation_task" / split / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32), 100, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    assert cv2.imwrite(str(images / f"{name}.jpg"), img)
    assert cv2.imwrite(str(masks / f"{name}.png"), mask)


def test_zero_byte_npy_is_not_ready(tmp_path: Path):
    p = tmp_path / "preprocessed"
    p.mkdir()
    for name in ("X_train.npy", "Y_train.npy", "X_test.npy", "Y_test.npy"):
        (p / name).touch()
    assert not _npy_ready(tmp_path)


def test_dataset_reports_unreadable_npy_cleanly(tmp_path: Path):
    p = tmp_path / "preprocessed"
    p.mkdir()
    for name in ("X_train.npy", "Y_train.npy"):
        (p / name).touch()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {
            "case_id": "bad", "split": "train", "data_backend": "npy", "sample_index": 0,
            "x_path": "preprocessed/X_train.npy", "y_path": "preprocessed/Y_train.npy",
        }
    ]).to_csv(manifest, index=False)
    with pytest.raises(RuntimeError, match="zero-byte/partial NPY"):
        BRISC2025Dataset(manifest, "train", tmp_path)


def test_prepare_recovers_from_stale_npy_by_using_raw(tmp_path: Path):
    project = tmp_path / "project"
    dataset = tmp_path / "dataset"
    (project / "datasets").mkdir(parents=True)
    (project / "datasets" / "source.json").write_text(json.dumps({
        "url": "https://example.invalid/not-used.zip",
        "archive_name": "unused.zip",
        "expected_train_samples": 2,
        "expected_test_samples": 1,
    }))

    # Broken NPY cache that used to pass the existence-only check.
    p = dataset / "preprocessed"
    p.mkdir(parents=True)
    for name in ("X_train.npy", "Y_train.npy", "X_test.npy", "Y_test.npy"):
        (p / name).touch()

    _write_raw_case(dataset, "train", "brisc2025_train_0001_gl_ax_a")
    _write_raw_case(dataset, "train", "brisc2025_train_0002_me_co_b")
    _write_raw_case(dataset, "test", "brisc2025_test_0001_pi_sa_c")

    manifest = project / "data" / "manifest.csv"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "prepare_brisc2025.py"),
        "--project-root", str(project),
        "--data-root", str(dataset),
        "--manifest", str(manifest),
        "--prefer-backend", "raw",
        "--val-fraction", "0.5",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    df = pd.read_csv(manifest)
    assert set(df["data_backend"]) == {"raw"}
    assert set(df["split"]) == {"train", "val", "test"}
