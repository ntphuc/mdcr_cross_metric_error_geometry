from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.discordance import ROUTES, feature_columns


def make_model(kind: str, seed: int):
    if kind == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, multi_class="auto")),
        ])
    if kind == "forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=350, max_depth=10, min_samples_leaf=4,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
            )),
        ])
    raise ValueError(kind)


def available_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def metrics(y_true, y_pred) -> dict:
    labels = [r for r in ROUTES if r in set(y_true) or r in set(y_pred)]
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=["no_correction"], average=None, zero_division=0)
    return {
        "accuracy": float(acc), "balanced_accuracy": float(bal), "macro_f1": float(macro),
        "no_correction_precision": float(p[0]) if len(p) else 0.0,
        "no_correction_recall": float(r[0]) if len(r) else 0.0,
        "no_correction_f1": float(f[0]) if len(f) else 0.0,
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare error-diagnosis feature sets against oracle correction families.")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--label-column", default="oracle_route")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--feature-sets", nargs="+", default=["dice_only", "fpfn", "uncertainty", "scalar_quality", "metric_values", "metric_discordance", "metric_discordance_plus_fpfn", "all_combined"])
    ap.add_argument("--model", choices=["forest", "logistic"], default="forest")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--augment-train-output", default=None, help="Write train CSV with OOF route_<feature_set> columns.")
    ap.add_argument("--augment-test-output", default=None, help="Write test CSV with predicted route_<feature_set> columns.")
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv).copy(); test = pd.read_csv(args.test_csv).copy()
    if args.label_column not in train or args.label_column not in test:
        raise ValueError(f"Missing label column {args.label_column}")
    y_train = train[args.label_column].astype(str).to_numpy()
    y_test = test[args.label_column].astype(str).to_numpy()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for fs in args.feature_sets:
        requested = feature_columns(fs)
        cols = [c for c in requested if c in train.columns and c in test.columns]
        if not cols:
            print(f"SKIP {fs}: no requested columns present")
            continue
        # Keep the comparison explicit if an uncertainty-MC set is requested on deterministic data.
        Xtr = train[cols].replace([np.inf, -np.inf], np.nan)
        Xte = test[cols].replace([np.inf, -np.inf], np.nan)
        model = make_model(args.model, args.seed)
        counts = pd.Series(y_train).value_counts()
        min_count = int(counts.min()) if len(counts) else 0
        n_splits = max(2, min(5, min_count)) if min_count >= 2 else 0
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            oof = cross_val_predict(make_model(args.model, args.seed), Xtr, y_train, cv=cv, method="predict", n_jobs=1)
        else:
            # Extremely rare route classes: leave-one-model prediction is an explicitly marked fallback.
            model.fit(Xtr, y_train); oof = model.predict(Xtr)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        train[f"route_{fs}_oof"] = oof
        test[f"route_{fs}"] = pred
        m = metrics(y_test, pred)
        m.update({"feature_set": fs, "model": args.model, "n_features": len(cols), "features": ";".join(cols)})
        summary_rows.append(m)
        labels = list(ROUTES)
        cm = confusion_matrix(y_test, pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f"confusion_{fs}.csv")
        joblib.dump(model, out / f"router_{fs}.joblib")
        pd.DataFrame({"case_id": test.get("case_id", pd.Series(np.arange(len(test)))), "y_true": y_test, "y_pred": pred}).to_csv(out / f"predictions_{fs}.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(out / "route_diagnostic_summary.csv", index=False)
    if args.augment_train_output:
        p = Path(args.augment_train_output); p.parent.mkdir(parents=True, exist_ok=True); train.to_csv(p, index=False)
    if args.augment_test_output:
        p = Path(args.augment_test_output); p.parent.mkdir(parents=True, exist_ok=True); test.to_csv(p, index=False)
    meta = {
        "train_csv": args.train_csv, "test_csv": args.test_csv, "label": args.label_column,
        "model": args.model, "seed": args.seed, "feature_sets": args.feature_sets,
        "interpretation": "Higher macro-F1 means the signal better predicts which correction family the oracle says is useful, including no correction.",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
