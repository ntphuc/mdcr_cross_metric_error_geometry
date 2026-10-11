# Beyond FP/FN Counts: Cross-Metric Error Geometry for Diagnosing Correction Needs in Brain Tumor MRI Segmentation

Code and released results for the ICEIR 2026 paper by **Nguyen Thien Phuc** and **Nguyen Quang Tuyen** (Department of Software Engineering, FPT University, Vietnam).

The paper asks whether disagreement among overlap, boundary, and surface-distance metrics ("cross-metric error geometry") tells us *which kind of correction* a segmentation needs, beyond how many pixels are wrong. It then uses MDCR (Metric-Discordance Correction Routing) as a testbed to check whether that diagnostic knowledge can be turned into deployable correction.

Main findings, all reproducible from this repository:

- On natural validation errors, adding metric geometry to FP/FN features raises correction-type macro-F1 from 0.476 to 0.551, positive in all 8 architecture-seed runs.
- Route-specialized experts have correction headroom under true-oracle routing in 7 of 8 runs.
- The ground-truth-free router is the bottleneck (0.216 macro-F1 as deployed). The safety gate therefore falls back to the source model in every run, so MDCR does not change the locked-test segmentation.

## Repository layout

```text
.
├── RUN_MDCR_V3_EXPERIMENTS.sh     # one-command entry point (full suite)
├── RUN_MDCR_V3_SUITE.py           # resumable suite runner (all stages below)
├── run_slurm.example.sh           # example Slurm submission
├── train.py                       # source segmentor (EfficientNet-B0 / ResNet18 U-Net)
├── train_control.py               # compute-matched continuation control
├── train_router_v3.py             # hierarchical ground-truth-free router
├── train_mdcr_v3.py               # route-specialized residual experts + safety gate
├── evaluate.py, evaluate_mdcr_v3.py
├── configs/                       # baseline, MDCR-v3 and suite configs used in the paper
├── src/                           # models, losses, metrics, oracle route definition
├── scripts/                       # data preparation, benchmarks, diagnostics, analysis
│   └── reproduce_paper_tables.py  # recompute the paper's numbers from results/
├── data/brisc2025_split_case_ids.csv  # exact train/val/test case IDs used
├── results/mdcr_publication_suite_v3_final/  # released outputs of the paper's runs
├── patches/                       # optional post-publication fix (see "Known behavior")
└── tests/
```

## Reproduce the paper's numbers without a GPU

All tables and the router/prevalence statistics can be recomputed from the released results:

```bash
pip install numpy pandas
python scripts/reproduce_paper_tables.py --results results/mdcr_publication_suite_v3_final
```

| Paper item | Released file(s) |
|---|---|
| Table 1 (synthetic, matched-FP) | `results/.../synthetic/{diagnostics,matched_diagnostics}/route_diagnostic_summary.csv` |
| Table 2, Fig. 3 (natural diagnosis) | `results/.../architectures/*/seed*/diagnostics/route_prediction/route_diagnostic_summary.csv` |
| Route prevalence | `results/.../architectures/*/seed*/diagnostics/natural_{train,val}_raw.json` |
| Table 3, Fig. 4 (true-oracle ceiling) | `results/.../analysis/final_round_decisions.csv`, `upper_bounds/true_oracle_raw/summary.json` |
| Router quality (Sec. 5.4) | `router_v3_eval/metrics_oracle_route.json` (deployed), `router_v3/router_summary.json` (selection) |
| Tables 4 and 5 | `results/.../analysis/{performance,locked_test}_multiseed_summary.csv` |
| Per-case metrics | `eval_val/*/per_case_metrics.csv`, `locked_test/*/*/per_case_metrics.csv` |

Absolute paths in the released files were replaced by `<DATA_ROOT>`; no images, masks, or checkpoints are included.

## Installation

The paper's runs used Python 3.12, PyTorch 2.7.1 (CUDA 12.8), and one NVIDIA H200 MIG 2g.35gb slice.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # install the PyTorch build matching your CUDA first if needed
```

## Data

The experiments use the segmentation subset of **BRISC2025** (Fateh et al., *Scientific Data* 13, 2026, [doi:10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y)). Obtain the dataset from its official source and follow its license; this repository does not redistribute it.

`scripts/prepare_brisc2025.py` accepts either layout under `--data-root` (default `datasets/brisc2025`):

- the raw tree `segmentation_task/{train,test}/{images,masks}`, or
- preprocessed arrays `X_train.npy`, `Y_train.npy`, `X_test.npy`, `Y_test.npy` (optionally `planes_*.npy`, `tumors_*.npy`).

It writes `data/brisc2025_manifest.csv`, keeping the official 860-image test partition locked and holding out 15% of the 3,933 development images for validation (seed 42).

**Exact split.** The paper's runs used the preprocessed NPY arrays (case IDs `brisc_npy_<split>_<index>`), with validation stratified by imaging plane. `data/brisc2025_split_case_ids.csv` lists every case ID with its split, so you can check that your manifest matches. The raw-image backend stratifies by tumor type and plane and therefore produces a different, though comparable, split.

## Run the full experiment suite

```bash
DATA_ROOT=/path/to/brisc2025 bash RUN_MDCR_V3_EXPERIMENTS.sh
# optional: VENV_ROOT=/path/to/venv  DEVICE=cuda
# on Slurm: edit and submit run_slurm.example.sh
```

The suite is resumable: re-running the same command skips completed stages. It runs, in order:

1. Synthetic structured and matched-FP benchmarks (Table 1).
2. For each architecture and seed (EfficientNet-B0 U-Net: 42, 123, 2024, 2026, 3407; ResNet18 U-Net: 42, 123, 2024):
   - source segmentor and compute-matched continuation control;
   - natural correction benchmark with oracle route labels defined by FP+FN pixel-error reduction (`src/discordance.py`), and diagnostic probes (Table 2);
   - hierarchical router (`train_router_v3.py`) and route-specialized experts (`train_mdcr_v3.py`);
   - learned-routing evaluation, validation-only true-oracle and metric-teacher forced routing, and the bottleneck decision (`scripts/final_round_decision.py`).
3. Locked official-test evaluation of baseline, control, and the safety-selected MDCR checkpoint (oracle routing is never used on test).
4. Aggregation into `outputs/mdcr_publication_suite_v3_final/analysis/`.

## Known behavior: router margin at deployment

`train_router_v3.py` selects the router checkpoint on validation using a calibrated correction margin. All selected margins were negative (−0.08 to −0.18), so the calibrated rule favors correction. In the code used for the paper, `MDCRV3Model._hard_route_ids` ignores margins ≤ 0. As a result, MDCR deployed plain argmax routing over the composed route probabilities.

The paper reports both figures: macro-F1 is 0.216 as deployed and 0.289 with the calibrated margin. The experts (trained with forced oracle routes) and the locked test (fallback to the source in every run) are unaffected.

`patches/0001-apply-router-margin-of-any-sign.patch` applies the margin for any sign. It is **not** applied in this release so that the code matches the reported results:

```bash
git apply patches/0001-apply-router-margin-of-any-sign.patch
```

With the patch, re-run the expert stage (`train_mdcr_v3.py`) and the evaluations; routers and baselines can be reused. Results may differ from the paper.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

## Provenance

This is the MDCR-v3.1 code that produced the paper's results. Earlier exploratory variants (GMGD, EMBR, MDCR-v1/v2) are not included. Changes relative to the code as run are limited to:

- the default dataset path and the run script, which now use configurable paths instead of cluster paths;
- the data-download URL in `datasets/source.json`, which is left empty, with a clear error message in `prepare_brisc2025.py`;
- added files: `scripts/reproduce_paper_tables.py`, `data/brisc2025_split_case_ids.csv`, `run_slurm.example.sh`, `patches/`, and documentation.

No training, evaluation, or metric code was changed.

## Citation

```bibtex
@inproceedings{phuc2026crossmetric,
  title     = {Beyond {FP/FN} Counts: Cross-Metric Error Geometry for Diagnosing
               Correction Needs in Brain Tumor {MRI} Segmentation},
  author    = {Nguyen Thien Phuc and Nguyen Quang Tuyen},
  booktitle = {Proceedings of ICEIR 2026},
  year      = {2026},
  note      = {To appear; update with the final proceedings reference}
}
```

## License and contact

Code is released under the MIT License (see `LICENSE`). The BRISC2025 data remain subject to their own license.

Corresponding author: Nguyen Thien Phuc, phucnt77@fe.edu.vn.
