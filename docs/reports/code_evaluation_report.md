# QMLKit Code Evaluation Report

**Date:** 2026-08-23 · **Repo state evaluated:** `efb9075` (main) · **Method:** static review + test execution

---

## 1. Executive summary

The codebase is well-layered (data → quantum → classical → evaluation → explainability → API) with honest
leak-free splitting where implemented and correct fidelity-kernel math. However, at evaluation time it
*simulated its own success*: its only data source was a generator engineered to be easy, the headline
BioZZ feature map was effectively disabled as wired, and the explainability layer would have misled on
real data. All integration blockers and high-priority defects identified below have been fixed in this
pass; remaining items are documented for follow-up.

## 2. Findings and disposition

| # | Severity | Location | Finding | Status |
|---|---|---|---|---|
| 1 | High | `src/qmlkit/evaluation/benchmark_suite.py` | BioZZ covariance computed as `corrcoef(PCA scores)` — orthogonal by construction ≈ identity → entanglement weights ≈ 0; headline map silently inert | **Fixed**: `compute_qubit_covariance()` derives qubit weights from raw-feature correlations (top-loading aggregation for PCA; direct correlation for raw-subset selectors) |
| 2 | High | `src/qmlkit/explainability/biomarker_mapper.py` | Hardcoded synthetic geometry (24 compounds / 64 sensors); undefined `Any` name; unused `pandas` import; hardcoded pathway slices | **Fixed**: schema-generalized `_pool_to_compounds()` (direct-name match → legacy schema → contiguous pooling), adaptive pathway blocks, imports repaired |
| 3 | High | `src/qmlkit/api/server.py` | QSVM trained on full cohort without split (leakage); same PCA-covariance defect; deprecated `on_event`; pseudo-SHAP delta while real explainer imported unused | **Fixed**: stratified leak-free split; covariance helper reused; FastAPI lifespan handler; Kernel-SHAP wired behind `deep_explain=true` with graceful fallback |
| 4 | High | repo-wide | No external data ingestion existed (`read_csv`/parsers absent); only synthetic generator fed the pipeline | **Fixed**: new `qmlkit.data.dataset_loader` (markdown-table + CSV loaders) powering `run_benchmark.py --source markdown/csv` |
| 5 | Medium | `src/qmlkit/quantum/vqc.py`, `qcnn.py` | Floor-division batching dropped up to batch−1 samples per epoch; unseeded shuffles | **Fixed**: ceil-batching keeps every sample; seeded `default_rng` |
| 6 | Medium | `src/qmlkit/quantum/qcnn.py` | Pool block crashed on odd `n_qubits` (`i+1` unbounded); no min-qubit validation | **Fixed**: modular wire index + validation |
| 7 | Medium | `src/qmlkit/explainability/quantum_shap.py` | Top-level `import shap` broke package import on minimal installs | **Fixed**: lazy import with actionable error |
| 8 | Medium | `pyproject.toml` | `httpx` missing from dev deps (API e2e fails in clean envs) | **Fixed** |
| 9 | Low | multiple | Unused imports (`StratifiedKFold`, `BaseFeatureMap`, …); unsorted imports | **Fixed** via ruff config + autofix (ruff now clean: E4/E7/E9/F/I/B selected) |
| 10 | Low | `scripts/run_benchmark.py` | Plot `ylim(0.5, 1.05)` clipped sub-chance bars (visual honesty issue) | **Fixed**: ylim `(0, 1.05)` |
| 11 | Info | `src/qmlkit/classical/baselines.py` | sklearn FutureWarning: `SVC(probability=True)` deprecated since 1.9 → migrate to `CalibratedClassifierCV` | Open (tracked) |
| 12 | Info | `README.md` | Advertises Qiskit badge; no Qiskit dependency or code exists | Open (badge should be removed or Qiskit path built) |
| 13 | Info | `src/qmlkit/evaluation/hardware_profiler.py`, `quantum/qcnn.py`, `qrc.py` | Dead/unwired modules (profiler arithmetic-only with `parameter_count=0`; QCNN/QRC not in benchmark suite) | Open (wire into suite or remove) |

## 3. Synthetic-vs-real sanity check (context)

Committed historical artifact (`outputs/benchmark/benchmark_metrics.csv`) showed classical baselines at
exactly 1.0 ROC-AUC against the synthetic cohort — i.e., the generator produced trivially separable data.
On the real 427-patient dataset classical models remain strong (0.97–1.0 AUC) but quantum models land far
lower (see `real_data_benchmark_report.md`) — evidence that prior synthetic numbers said little about
real-world quantum-model behavior.

## 4. Verification after fixes

| Gate | Result |
|---|---|
| `python -m pytest -q` | **40 passed** (13 pre-existing + 13 loader + 14 hardware-ingest) |
| `python -m ruff check src scripts tests packaging qmlkit_desktop.py` | **All checks passed** |
| `pnpm --filter frontend lint && build` | clean; static export of `/`, `/diagnostics`, `/collect` |
