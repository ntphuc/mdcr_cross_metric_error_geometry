"""Recompute the numbers reported in the ICEIR 2026 paper from the released results.

No GPU, dataset, or checkpoint is needed. Usage:

    python scripts/reproduce_paper_tables.py \
        --results results/mdcr_publication_suite_v3_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUTES = ["no_correction", "recover_fn", "remove_fp", "boundary_align", "tail_cleanup"]
FEATURE_ORDER = ["dice_only", "scalar_quality", "uncertainty", "metric_values", "metric_discordance",
                 "fpfn", "metric_discordance_plus_fpfn", "all_combined"]


def runs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "architectures").glob("*/seed*") if p.is_dir())


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table1(root: Path) -> None:
    header("Table 1 - controlled correction-type diagnosis (synthetic benchmarks)")
    broad = pd.read_csv(root / "synthetic/diagnostics/route_diagnostic_summary.csv").set_index("feature_set")
    matched = pd.read_csv(root / "synthetic/matched_diagnostics/route_diagnostic_summary.csv").set_index("feature_set")
    out = broad[["accuracy", "balanced_accuracy", "macro_f1"]].copy()
    out["matched_fp_accuracy"] = matched["accuracy"].reindex(out.index)
    print(out.round(3).to_string())


def table2(root: Path) -> None:
    header("Table 2 - natural correction-type diagnosis (validation, 8 runs)")
    rows = []
    for r in runs(root):
        d = pd.read_csv(r / "diagnostics/route_prediction/route_diagnostic_summary.csv")
        d["run"] = f"{r.parent.name}/{r.name}"
        rows.append(d)
    d = pd.concat(rows)
    agg = d.groupby("feature_set")[["accuracy", "balanced_accuracy", "macro_f1"]].mean()
    print(agg.reindex([f for f in FEATURE_ORDER if f in agg.index]).round(3).to_string())
    p = d.pivot(index="run", columns="feature_set", values="macro_f1")
    gain = p["metric_discordance_plus_fpfn"] - p["fpfn"]
    print(f"\nMacro-F1 gain of discordance+FP/FN over FP/FN: mean {gain.mean():.4f}, "
          f"min {gain.min():.4f}, max {gain.max():.4f}, positive in {(gain > 0).sum()}/{len(gain)} runs")


def prevalence(root: Path) -> None:
    header("Oracle route prevalence (mean +/- SD over 8 runs)")
    for split in ("train", "val"):
        counts = [json.loads((r / f"diagnostics/natural_{split}_raw.json").read_text())["route_counts"] for r in runs(root)]
        n = json.loads((runs(root)[0] / f"diagnostics/natural_{split}_raw.json").read_text())["n_cases"]
        print(f"{split} (n={n}):")
        for route in ROUTES:
            v = np.array([c.get(route, 0) for c in counts], dtype=float)
            print(f"  {route:15s} {v.mean():7.1f} +/- {v.std(ddof=1):5.1f} ({100 * v.mean() / n:4.1f}%)")


def table3(root: Path) -> None:
    header("Table 3 - raw true-oracle improvement over the compute-matched control (validation)")
    rows = []
    for r in runs(root):
        ctrl = json.loads((r / "eval_val/control/summary.json").read_text())
        orc = json.loads((r / "upper_bounds/true_oracle_raw/summary.json").read_text())
        rows.append({"architecture": r.parent.name, "seed": r.name,
                     "d_dice": orc["dice_mean"] - ctrl["dice_mean"], "d_bf1": orc["bf1_mean"] - ctrl["bf1_mean"],
                     "hd95_red": ctrl["hd95_mean_finite"] - orc["hd95_mean_finite"],
                     "assd_red": ctrl["assd_mean_finite"] - orc["assd_mean_finite"],
                     "decision": json.loads((r / "FINAL_ROUND_DECISION.json").read_text())["decision"]})
    d = pd.DataFrame(rows)
    cols = ["d_dice", "d_bf1", "hd95_red", "assd_red"]
    out = d.groupby("architecture")[cols].mean()
    out.loc["all_runs"] = d[cols].mean()
    print(out.round(5).to_string())
    print("\nBottleneck decisions:", d["decision"].value_counts().to_dict())


def router(root: Path) -> None:
    header("Router quality against oracle routes (validation, 8 runs)")
    sel, dep = [], []
    for r in runs(root):
        sel.append(json.loads((r / "router_v3/router_summary.json").read_text())["best_validation"])
        dep.append(json.loads((r / "router_v3_eval/metrics_oracle_route.json").read_text()))
    for name, items in (("as deployed in MDCR (argmax of composed probabilities)", dep),
                        ("at selection, with validation-calibrated margin", sel)):
        print(f"{name}:")
        for k in ("accuracy", "balanced_accuracy", "macro_f1"):
            print(f"  {k:18s} {np.mean([m[k] for m in items]):.3f}")
        print("  recall: " + ", ".join(f"{rt}={np.mean([m['per_route_recall'][rt] for m in items]):.4f}" for rt in ROUTES))


def segmentation(root: Path) -> None:
    for title, fname in (("Table 4 - validation segmentation (mean +/- SD over seeds)", "performance_multiseed_summary.csv"),
                         ("Table 5 - locked official test (mean +/- SD over seeds)", "locked_test_multiseed_summary.csv")):
        header(title)
        d = pd.read_csv(root / "analysis" / fname, header=[0, 1], index_col=[0, 1])
        d.index.names = ["architecture", "method"]
        out = pd.DataFrame(index=d.index)
        for m in ("dice", "bf1", "hd95", "assd", "misses"):
            nd = 5 if m in ("dice", "bf1") else 3
            out[m] = [f"{a:.{nd}f} +/- {b:.{nd}f}" for a, b in zip(d[(m, "mean")], d[(m, "std")])]
        print(out.to_string())
    print("\nNote: 'mdcr_v3' rows are MDCR (fallback); the safety gate selected the frozen source in every run.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/mdcr_publication_suite_v3_final")
    root = Path(ap.parse_args().results)
    table1(root); table2(root); prevalence(root); table3(root); router(root); segmentation(root)


if __name__ == "__main__":
    main()
