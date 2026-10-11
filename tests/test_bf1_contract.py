from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.metrics import boundary_f1, case_metrics


def square(h=32, w=32, y0=8, y1=16, x0=8, x1=16):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def main():
    empty = np.zeros((32, 32), dtype=np.uint8)
    a = square()
    same = a.copy()
    near = square(y0=9, y1=17, x0=9, x1=17)
    far = square(y0=22, y1=30, x0=22, x1=30)

    assert boundary_f1(empty, empty, tolerance=2) == 1.0
    assert boundary_f1(a, empty, tolerance=2) == 0.0
    assert boundary_f1(empty, a, tolerance=2) == 0.0
    assert boundary_f1(a, same, tolerance=2) > 0.999
    assert boundary_f1(a, near, tolerance=2) > 0.95
    assert boundary_f1(a, far, tolerance=2) == 0.0

    m = case_metrics(far.astype(np.float32), a.astype(np.float32), threshold=0.5, bf1_tolerance=2)
    assert m["tp"] == 0
    assert m["bf1"] == 0.0, m
    assert m["dice"] < 1e-6, m
    print("BF1_CONTRACT_PASS")


if __name__ == "__main__":
    main()
