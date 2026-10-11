from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

NPY_REQUIRED = ("X_train.npy", "Y_train.npy", "X_test.npy", "Y_test.npy")
NPY_OPTIONAL = ("planes_train.npy", "planes_test.npy", "tumors_train.npy", "tumors_test.npy")
PLANE = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary"}
NAME_RE = re.compile(r"brisc2025_(train|test)_(\d+)_([a-z]+)_([a-z]+)_([a-z0-9]+)$")


def extract_drive_id(url: str) -> str | None:
    m = re.search(r"/file/d/([^/?#]+)", url)
    if m:
        return m.group(1)
    try:
        q = parse_qs(urlparse(url).query).get("id")
        return q[0] if q else None
    except Exception:
        return None


def _supported_kwargs(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def download_file(url: str, output: Path, force: bool = False) -> Path:
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"[prepare] reuse archive: {output}")
        return output
    if force and output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Missing gdown. Run: pip install -r requirements.txt") from exc

    fn = gdown.download
    file_id = extract_drive_id(url)
    base = {"output": str(output), "quiet": False, "use_cookies": False}
    try:
        supports_id = "id" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        supports_id = False
    if file_id and supports_id:
        kwargs = {**base, "id": file_id}
    else:
        direct = f"https://drive.google.com/uc?id={file_id}" if file_id else url
        kwargs = {**base, "url": direct}
    result = fn(**_supported_kwargs(fn, kwargs))
    if result is None or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("gdown did not produce the BRISC2025 archive. Check Drive sharing/network.")
    return output


def _safe_extract(zip_path: Path, target: Path):
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if target != dest and target not in dest.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        zf.extractall(target)


def _raw_ready(data_root: Path) -> bool:
    seg = data_root / "segmentation_task"
    return all((seg / s / k).is_dir() for s in ("train", "test") for k in ("images", "masks"))


def _validate_required_npy_file(path: Path) -> tuple[bool, str]:
    """Validate an NPY file without loading the full array into RAM.

    A stale/partially-created zero-byte NPY previously passed the old
    existence-only check and later crashed with ``EOFError: No data left in file``.
    This helper verifies that NumPy can read the header and that the first
    dimension is non-empty.
    """
    if not path.is_file():
        return False, "missing"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat failed: {exc}"
    if size <= 0:
        return False, "empty file (0 bytes)"
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim < 1 or len(arr) == 0:
            return False, f"invalid/empty array shape={getattr(arr, 'shape', None)}"
        # Touch only metadata; mmap_mode keeps the payload on disk.
        _ = arr.shape, arr.dtype
        del arr
    except (EOFError, ValueError, OSError) as exc:
        return False, f"cannot read NPY header: {type(exc).__name__}: {exc}"
    return True, "ok"


def _npy_ready(data_root: Path, verbose: bool = False) -> bool:
    p = data_root / "preprocessed"
    ok = True
    problems = []
    for name in NPY_REQUIRED:
        valid, reason = _validate_required_npy_file(p / name)
        if not valid:
            ok = False
            problems.append(f"{p / name}: {reason}")
    if verbose and problems:
        print("[prepare] Ignoring unusable preprocessed NPY backend:")
        for msg in problems:
            print(f"  - {msg}")
    return ok


def _find_raw(root: Path) -> Path | None:
    for p in [root, *root.rglob("segmentation_task")]:
        seg = p if p.name == "segmentation_task" else p / "segmentation_task"
        if all((seg / s / k).is_dir() for s in ("train", "test") for k in ("images", "masks")):
            return seg
    return None


def _find_npy(root: Path) -> dict[str, Path] | None:
    dirs = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for d in dirs:
        files = {p.name.lower(): p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".npy"}
        if all(n.lower() in files for n in NPY_REQUIRED):
            out = {n: files[n.lower()] for n in NPY_REQUIRED}
            out.update({n: files[n.lower()] for n in NPY_OPTIONAL if n.lower() in files})
            return out
    all_files = {p.name.lower(): p for p in root.rglob("*.npy")}
    if all(n.lower() in all_files for n in NPY_REQUIRED):
        out = {n: all_files[n.lower()] for n in NPY_REQUIRED}
        out.update({n: all_files[n.lower()] for n in NPY_OPTIONAL if n.lower() in all_files})
        return out
    return None


def normalize_archive(archive: Path, data_root: Path, force_data: bool = False) -> str:
    if not force_data:
        if _npy_ready(data_root):
            return "npy"
        if _raw_ready(data_root):
            return "raw"
    if force_data and data_root.exists():
        shutil.rmtree(data_root)

    staging = data_root.parent / ".brisc_extract"
    if staging.exists():
        shutil.rmtree(staging)
    _safe_extract(archive, staging)
    try:
        raw = _find_raw(staging)
        if raw is not None:
            target = data_root / "segmentation_task"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(raw, target, dirs_exist_ok=True)
            return "raw"
        npy = _find_npy(staging)
        if npy is not None:
            target = data_root / "preprocessed"
            target.mkdir(parents=True, exist_ok=True)
            for name, src in npy.items():
                shutil.copy2(src, target / name)
            return "npy"
        raise RuntimeError("Archive contains neither raw segmentation_task layout nor required NPY arrays")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _plane_values(path: Path, n: int) -> list[str]:
    if not path.exists():
        return ["unknown"] * n
    arr = np.load(path, allow_pickle=True)
    if len(arr) != n:
        raise RuntimeError(f"Plane array length mismatch: {path}")
    return [str(x).lower() for x in arr.tolist()]


def _tumor_values(path: Path, n: int) -> list[str]:
    """Read optional tumor-code arrays when supplied by the preprocessed release."""
    if not path.exists():
        return ["unknown"] * n
    arr = np.load(path, allow_pickle=True)
    if len(arr) != n:
        raise RuntimeError(f"Tumor array length mismatch: {path}")
    vals=[]
    aliases={"glioma":"gl","meningioma":"me","pituitary":"pi","gl":"gl","me":"me","pi":"pi"}
    for x in arr.tolist():
        vals.append(aliases.get(str(x).lower(), str(x).lower()))
    return vals


def build_npy_manifest(data_root: Path, out_csv: Path, val_fraction: float, seed: int) -> pd.DataFrame:
    p = data_root / "preprocessed"
    xtr, ytr, xte, yte = [p / n for n in NPY_REQUIRED]
    ntr = len(np.load(xtr, mmap_mode="r"))
    nte = len(np.load(xte, mmap_mode="r"))
    tr_planes = _plane_values(p / "planes_train.npy", ntr)
    te_planes = _plane_values(p / "planes_test.npy", nte)
    tr_tumors = _tumor_values(p / "tumors_train.npy", ntr)
    te_tumors = _tumor_values(p / "tumors_test.npy", nte)
    rel = lambda q: q.resolve().relative_to(data_root.resolve()).as_posix()

    tr = pd.DataFrame({
        "case_id": [f"brisc_npy_train_{i:05d}" for i in range(ntr)],
        "data_backend": "npy", "source_split": "train", "sample_index": np.arange(ntr),
        "x_path": rel(xtr), "y_path": rel(ytr),
        "plane_code": tr_planes, "plane_label": [PLANE.get(x, x) for x in tr_planes],
        "tumor_code": tr_tumors, "tumor_label": [TUMOR.get(x, x) for x in tr_tumors],
    })
    te = pd.DataFrame({
        "case_id": [f"brisc_npy_test_{i:05d}" for i in range(nte)],
        "data_backend": "npy", "source_split": "test", "sample_index": np.arange(nte),
        "x_path": rel(xte), "y_path": rel(yte),
        "plane_code": te_planes, "plane_label": [PLANE.get(x, x) for x in te_planes],
        "tumor_code": te_tumors, "tumor_label": [TUMOR.get(x, x) for x in te_tumors],
    })
    strat = tr["plane_code"] if tr["plane_code"].nunique() > 1 and (tr["plane_code"].value_counts() >= 2).all() else None
    try:
        _, idx_val = train_test_split(tr.index, test_size=val_fraction, random_state=seed, shuffle=True, stratify=strat)
    except ValueError as exc:
        print(f"[prepare] Stratified NPY split unavailable ({exc}); using seeded random split.")
        _, idx_val = train_test_split(tr.index, test_size=val_fraction, random_state=seed, shuffle=True, stratify=None)
    tr["split"] = "train"
    tr.loc[idx_val, "split"] = "val"
    te["split"] = "test"
    out = pd.concat([tr, te], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _parse_name(stem: str) -> dict:
    m = NAME_RE.match(stem)
    if not m:
        return {"tumor_code": "unknown", "tumor_label": "unknown", "plane_code": "unknown", "plane_label": "unknown"}
    _, _, tumor, plane, _ = m.groups()
    return {"tumor_code": tumor, "tumor_label": TUMOR.get(tumor, tumor), "plane_code": plane, "plane_label": PLANE.get(plane, plane)}


def build_raw_manifest(data_root: Path, out_csv: Path, val_fraction: float, seed: int) -> pd.DataFrame:
    seg = data_root / "segmentation_task"
    rows = []
    for source_split in ("train", "test"):
        images = seg / source_split / "images"
        masks = seg / source_split / "masks"
        for ip in sorted(images.glob("*.jpg")):
            mp = masks / f"{ip.stem}.png"
            if not mp.exists():
                raise FileNotFoundError(mp)
            rows.append({
                "case_id": ip.stem, "data_backend": "raw", "source_split": source_split,
                "image_path": ip.resolve().relative_to(data_root.resolve()).as_posix(),
                "mask_path": mp.resolve().relative_to(data_root.resolve()).as_posix(),
                **_parse_name(ip.stem),
            })
    df = pd.DataFrame(rows)
    tr = df[df.source_split == "train"].copy()
    te = df[df.source_split == "test"].copy()
    strat = tr["tumor_code"].astype(str) + "_" + tr["plane_code"].astype(str)
    try:
        _, idx_val = train_test_split(tr.index, test_size=val_fraction, random_state=seed, shuffle=True, stratify=strat)
    except ValueError:
        _, idx_val = train_test_split(tr.index, test_size=val_fraction, random_state=seed, shuffle=True)
    tr["split"] = "train"
    tr.loc[idx_val, "split"] = "val"
    te["split"] = "test"
    out = pd.concat([tr, te], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _check_counts(train_n: int, test_n: int, expected_train: int | None, expected_test: int | None, allow_count_drift: bool):
    mismatches = []
    if expected_train is not None and train_n != expected_train:
        mismatches.append(f"train={train_n} expected={expected_train}")
    if expected_test is not None and test_n != expected_test:
        mismatches.append(f"test={test_n} expected={expected_test}")
    if mismatches and not allow_count_drift:
        raise RuntimeError("BRISC2025 sample-count mismatch: " + "; ".join(mismatches) + ". Use --allow-count-drift only if this is intentional.")
    if mismatches:
        print("[prepare] WARNING count drift: " + "; ".join(mismatches))


def verify_npy(data_root: Path, expected_train: int | None = None, expected_test: int | None = None, allow_count_drift: bool = False) -> dict:
    p = data_root / "preprocessed"
    xtr, ytr, xte, yte = [np.load(p / n, mmap_mode="r") for n in NPY_REQUIRED]
    if len(xtr) != len(ytr) or len(xte) != len(yte):
        raise RuntimeError("X/Y sample count mismatch")
    _check_counts(len(xtr), len(xte), expected_train, expected_test, allow_count_drift)
    # Lightweight finite/range sanity check on a few samples without loading all arrays.
    for name, arr in (("X_train", xtr), ("Y_train", ytr), ("X_test", xte), ("Y_test", yte)):
        for i in sorted(set([0, max(0, len(arr)//2), max(0, len(arr)-1)])):
            sample = np.asarray(arr[i])
            if not np.isfinite(sample).all():
                raise RuntimeError(f"{name}[{i}] contains NaN/Inf")
    return {
        "backend": "npy", "train_samples": len(xtr), "test_samples": len(xte),
        "x_train_shape": list(xtr.shape), "y_train_shape": list(ytr.shape),
        "x_test_shape": list(xte.shape), "y_test_shape": list(yte.shape),
    }


def verify_raw(data_root: Path, expected_train: int | None = None, expected_test: int | None = None, allow_count_drift: bool = False) -> dict:
    seg = data_root / "segmentation_task"
    result = {"backend": "raw"}
    for split in ("train", "test"):
        ni = len(list((seg / split / "images").glob("*.jpg")))
        nm = len(list((seg / split / "masks").glob("*.png")))
        if ni != nm or ni == 0:
            raise RuntimeError(f"Raw {split} pair count mismatch: images={ni}, masks={nm}")
        result[f"{split}_samples"] = ni
    _check_counts(result["train_samples"], result["test_samples"], expected_train, expected_test, allow_count_drift)
    return result


def main():
    ap = argparse.ArgumentParser(description="Prepare BRISC2025 (raw image/mask layout or preprocessed NPY arrays) and write the split manifest.")
    ap.add_argument("--project-root", default=str(ROOT))
    ap.add_argument("--data-root", default=None, help="Existing BRISC2025 dataset root; defaults to <project>/datasets/brisc2025")
    ap.add_argument("--manifest", default=None, help="Manifest output path; defaults to <project>/data/brisc2025_manifest.csv")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--allow-count-drift", action="store_true")
    ap.add_argument(
        "--prefer-backend", choices=("auto", "raw", "npy"), default="auto",
        help=("Backend preference when both layouts exist. 'auto' prefers raw images "
              "because they are easier to validate and cannot be broken by a partial NPY cache."),
    )
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    source = json.loads((root / "datasets/source.json").read_text(encoding="utf-8"))
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else (root / "datasets/brisc2025")
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else (root / "data/brisc2025_manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    archive = root / "downloads" / source.get("archive_name", "data_preprocessed.zip")

    raw_ok = _raw_ready(data_root)
    npy_ok = _npy_ready(data_root, verbose=True)

    if args.force_data or not (raw_ok or npy_ok):
        if not source.get("url"):
            raise SystemExit(
                f"[prepare] No BRISC2025 data found under {data_root} and no download URL is set in "
                "datasets/source.json. Download BRISC2025 from its official source and place it under "
                "--data-root (see README, section 'Data')."
            )
        download_file(source["url"], archive, force=args.force_download)
        backend = normalize_archive(archive, data_root, force_data=args.force_data)
    else:
        if args.prefer_backend == "raw":
            backend = "raw" if raw_ok else "npy"
        elif args.prefer_backend == "npy":
            backend = "npy" if npy_ok else "raw"
        else:
            # Prefer the canonical image/mask tree when available. This also
            # recovers automatically from an old manifest pointing at a broken
            # or zero-byte preprocessed NPY cache.
            backend = "raw" if raw_ok else "npy"
        print(f"[prepare] Selected backend={backend} (raw_ok={raw_ok}, npy_ok={npy_ok})")

    expected_train = source.get("expected_train_samples")
    expected_test = source.get("expected_test_samples")
    if backend == "npy":
        report = verify_npy(data_root, expected_train, expected_test, args.allow_count_drift)
        df = build_npy_manifest(data_root, manifest, args.val_fraction, args.seed)
    else:
        report = verify_raw(data_root, expected_train, expected_test, args.allow_count_drift)
        df = build_raw_manifest(data_root, manifest, args.val_fraction, args.seed)

    report["manifest"] = str(manifest)
    report["split_counts"] = {k: int(v) for k, v in df["split"].value_counts().to_dict().items()}
    (root / "data/brisc2025_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("[prepare] Official test is preserved as split=test and is not used for training/validation.")


if __name__ == "__main__":
    main()
