# Workstream A — QMLKit Code Fixes & Dataset Integration

## A1. Bug fixes (blockers + high-priority)

| # | File | Defect | Fix |
|---|---|---|---|
| 1 | `src/qmlkit/evaluation/benchmark_suite.py` | BioZZ covariance computed from orthogonal PCA scores ≈ identity → entanglement silently disabled | New `compute_qubit_covariance()`: raw-subset selectors → Pearson corr of selected columns; PCA → max abs raw-feature correlation among top-k loadings per component pair |
| 2 | `src/qmlkit/explainability/biomarker_mapper.py` | Undefined `Any`; unused `pd`; hardcoded 24-compound / 64-sensor geometry; hardcoded pathway slices | Import `Any`; generalize `_pool_to_compounds()` (direct-name match → legacy synthetic schema → generic contiguous pooling); pathway blocks derived from compound count |
| 3 | `src/qmlkit/explainability/quantum_shap.py` | Hard top-level `import shap` breaks package import on minimal installs | Lazy import inside `__init__` with helpful error message |
| 4 | `src/qmlkit/quantum/qcnn.py` | Pool block crashes for odd `n_qubits` (`source_wire=i+1` unbounded); floor-division drops remainder samples per epoch; unseeded permutation; no min-qubit validation | `(i+1) % n_qubits`; ceil-division batching; seeded `default_rng`; raise if `< 2` qubits; drop unused imports |
| 5 | `src/qmlkit/quantum/vqc.py` | Same remainder-drop + unseeded shuffle | Ceil-division batching + seeded rng |
| 6 | `src/qmlkit/data/preprocessor.py`, `benchmark_suite.py` | Unused `StratifiedKFold` imports | Removed |
| 7 | `src/qmlkit/api/server.py` | Trains QSVM on full cohort (no split); covariance from PCA scores; deprecated `on_event`; pseudo-SHAP delta while importing real explainer unused; unused imports | Leak-free stratified split at startup; `compute_qubit_covariance` reuse; FastAPI lifespan handler; real `QuantumKernelSHAP` with graceful delta fallback |
| 8 | `pyproject.toml` | `httpx` missing from dev deps (API e2e test fails in clean envs) | Added to `[project.optional-dependencies].dev` |

Already fixed by prior commits (verified, no action): `feature_selector.py`
fit-once angle scaling; VQC gradient/scaling leakage.

## A2. Dataset integration

### Lung VOC (primary) — `docs/Lung_Cancer_VOC_Dataset_427.md`

- New module **`src/qmlkit/data/dataset_loader.py`**:
  - `load_markdown_table(path)` — generic pipe-delimited parser (skips alignment row).
  - `load_lung_voc_dataset(path)` → `(df_features[27 compounds], y∈{0,1}, patient_ids)`;
    labels: Control→0, Cancer→1.
  - `load_csv_dataset(path, label_column)` — future external CSVs.
- Extend `scripts/run_benchmark.py`: `--source {synthetic,markdown,csv}`,
  `--data-path`, `--max-samples` (balanced subsample guard for O(N²) kernels).
- Artifacts → `outputs/benchmark_real/`.
- Tests: parser fixture table, separator-row skip, numeric coercion, unknown-label
  errors, balanced subsampling.

### Dog ECG + DogInfo (secondary) — `docs/dataset_1.md`, `docs/DogInfo.md`

Constraints (documented): WAV audio files are absent from the repo and there are
no diagnostic labels → annotation-based integration only.

- Loader parses the 1,123-row markdown table incl. list-valued cells
  (`segments_br`, `segments_hr`, `ecg_pulses`, `bad_ecg`) via safe literal eval.
- Per-recording features from R-peak timestamps: HR mean/min/max, SDNN, RMSSD;
  bradycardia episode count/duration; bad-ECG quality fraction.
- Join `DogInfo.md` on `pet_id ↔ DogID` (breed/weight/age-months/gender/neutering).
- Curated tables → `data/ecg/ecg_recordings_features.csv`, `data/ecg/dog_info.csv`.

## A3. Reports → `docs/reports/`

1. `code_evaluation_report.md` — findings table (severity, file:line, status).
2. `dataset_integration_report.md` — schemas, mappings, assumptions, constraints.
3. `real_data_benchmark_report.md` — quantum vs classical metrics on the real
   427-patient dataset + comparison against the synthetic-run baseline.

Verification gates: full pytest green; ruff clean.
