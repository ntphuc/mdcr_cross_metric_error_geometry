#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Override with environment variables, e.g. DATA_ROOT=/path/to/brisc2025 VENV_ROOT=/path/to/venv bash RUN_MDCR_V3_EXPERIMENTS.sh
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/brisc2025}"
VENV_ROOT="${VENV_ROOT:-}"
MANIFEST="data/brisc2025_manifest.csv"
SUITE_CONFIG="configs/mdcr_v3_suite.yaml"
DEVICE="${DEVICE:-cuda}"
RESUME=1
EVAL_TEST=1
mkdir -p logs data outputs downloads
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ -d "$DATA_ROOT" ]] || { echo "[FATAL] Dataset missing: $DATA_ROOT" >&2; exit 2; }
if [[ -n "$VENV_ROOT" ]]; then
  [[ -f "$VENV_ROOT/bin/activate" ]] || { echo "[FATAL] venv missing: $VENV_ROOT/bin/activate" >&2; exit 2; }
  source "$VENV_ROOT/bin/activate"
  PYTHON_BIN="$VENV_ROOT/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] Python missing: $PYTHON_BIN" >&2; exit 2; }
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
printf '%s\n' "[$(date '+%F %T')] Starting MDCR-v3 FINAL ROUND" "Project: $ROOT" "Python: $PYTHON_BIN" "Dataset: $DATA_ROOT" "Manifest: $MANIFEST" "Suite: $SUITE_CONFIG" "Device: $DEVICE"
"$PYTHON_BIN" - <<'PY'
import importlib
mods=["torch","torchvision","numpy","pandas","sklearn","cv2","yaml","scipy"]
missing=[]
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append(f"{m}: {type(e).__name__}: {e}")
if missing: raise SystemExit("[FATAL] dependency check failed:\n  - "+"\n  - ".join(missing))
print("[preflight] dependencies OK")
PY
# Refresh the manifest. The raw image/mask layout is preferred when present;
# the paper's runs used the preprocessed NPY arrays (no raw tree present).
"$PYTHON_BIN" -u scripts/prepare_brisc2025.py --data-root "$DATA_ROOT" --manifest "$MANIFEST" --prefer-backend raw
args=(--suite-config "$SUITE_CONFIG" --device "$DEVICE" --data-root "$DATA_ROOT" --manifest "$MANIFEST")
[[ "$RESUME" == "1" ]] || args+=(--no-resume)
[[ "$EVAL_TEST" == "1" ]] || args+=(--skip-test)
"$PYTHON_BIN" -u RUN_MDCR_V3_SUITE.py "${args[@]}"
echo "[$(date '+%F %T')] MDCR-v3 FINAL ROUND completed."
echo "Main analysis: outputs/mdcr_publication_suite_v3_final/analysis/"
