# 15 — Results and Findings

Every number the project has committed, in one place, with the caveats attached.

---

## 1. Where the artifacts live

```
outputs/
├─ benchmark/benchmark_metrics.csv            historical SYNTHETIC run
├─ benchmark_real/
│  ├─ benchmark_metrics.csv                   real Lung VOC, single split
│  ├─ benchmark_comparison.png                ROC-AUC bar chart
│  └─ run_metadata.txt                        source, N, splits, seed
├─ hybrid_search_log.txt
├─ lab_kennel_search_log.txt
├─ lab_kennel_robust_log.txt
├─ lab_map_ablation_log.txt
└─ lab/
   ├─ 20260823_082122/  full 12-config search + ensembles, voc_real
   │  ├─ run.json · leaderboard.csv · _cli_meta.json
   │  └─ configs/*.json   (per-config spec + per-fold metrics + NISQ profile)
   ├─ 20260823_084010/  map ablation (4 configs)
   ├─ 20260823_084047/  kennel_synth search
   ├─ 20260823_085411/  kennel robustness — CRASHED (KeyError 'leaderboard'), artifacts present, no _cli_meta.json
   └─ 20260823_085700/  kennel robustness — rerun via CLI, has _cli_meta.json
```

`outputs/lab/<run_id>/configs/*.json` is the **complete evidence trail**: every
leaderboard cell is backed by a per-fold metric dict and a circuit profile in the
corresponding config file. This is unusually good practice and should be
highlighted in the presentation.

---

## 2. Experiment A — real-data single-split benchmark

**Command**

```bash
python scripts/run_benchmark.py --source markdown --voc-task cancer_vs_control \
    --max-samples 120 --vqc-epochs 8 --n-qubits 6 --output-dir outputs/benchmark_real
```

**Configuration**

| Parameter | Value | Rationale |
|---|---|---|
| Dataset | real `Lung_Cancer_VOC_Dataset_427` (cancer_vs_control) | first non-synthetic run |
| N after cap | 120 = 60 Control + 60 Cancer | bounds the $O(N^2)$ kernel cost |
| Split | stratified 80/20 ⇒ **96 train / 24 test** | leak-free; scaler + PCA fit on train only |
| Qubits | 6 (PCA from 27 compounds) | NISQ-realistic |
| VQC epochs | 8 (default 20) | runtime bound |
| Seed | 42 | reproducibility |

**Results (ranked by ROC-AUC)**

| Model | Paradigm | Accuracy | Bal. Acc | Sens. | Spec. | **ROC-AUC** | Brier | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | Classical | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.010 | 0.13 |
| Random Forest | Classical | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.041 | 0.13 |
| MLP NeuralNet | Classical | 0.958 | 0.958 | 0.917 | 1.000 | **1.000** | 0.020 | 0.16 |
| SVM Linear | Classical | 0.958 | 0.958 | 0.917 | 1.000 | 0.979 | 0.061 | 0.00 |
| SVM RBF | Classical | 0.917 | 0.917 | 0.917 | 0.917 | 0.972 | 0.074 | 0.00 |
| VQC StronglyEntangled | Quantum | 0.667 | 0.667 | 0.500 | 0.833 | **0.625** | 0.245 | 343.5 |
| QSVM BioZZ | Quantum | 0.542 | 0.542 | 0.417 | 0.667 | **0.507** | 0.250 | 0.74 |

### 2.1 Findings

1. **The real dataset is highly separable for strong classical learners.** Even a
   *linear* SVM reaches 0.979 AUC. A linear model on 27 features separating cancer
   from control means the signal is close to a hyperplane in concentration space —
   precisely the regime where a fidelity kernel cannot help, because the kernel's
   advantage is in modelling nonlinear structure.
2. **Quantum models are at or near chance.** QSVM 0.507 is a coin flip; VQC 0.625
   is weak. Measured **with** the covariance fix active.
3. **The synthetic history understated the gap.** The committed synthetic artifact
   (`outputs/benchmark/`) had QSVM at 0.64 AUC against trivially separable
   baselines. On real data quantum falls to 0.51–0.63 while baselines stay
   ≥0.97.
4. **Cost asymmetry is 2,600×.** VQC 343.5 s vs XGBoost 0.13 s.

### 2.2 Statistical caveats the report states

- Single split, $n=24$ test ⇒ wide confidence intervals. A perfect 24/24 is
  1.000 with a Wilson 95% CI of roughly **[0.86, 1.00]** — the 1.000s should be
  read as "at ceiling", not exact.
- VQC under-trained relative to defaults (8 vs 20 epochs).
- "Numbers are evidence, not tuned results."

---

## 3. Experiment B — hybrid search, 5-fold CV

**Command**

```bash
python scripts/run_hybrid_search.py --dataset voc_real --experiment search \
    --max-samples 120 --vqc-epochs 6 --n-splits 5
```

**Protocol:** stratified 5-fold CV; **identical partitions across all
configurations**; a fresh pipeline, scaler, selector and correlation matrix per
fold; $n_\text{components}=6$; seed 42 throughout. Dataset: real Lung VOC
(cancer_vs_control), 120 balanced.

**Leaderboard (mean ± population sd over folds)**

| Rank | Config | ROC-AUC | Acc | Sens | Spec | Train s |
|---|---|---|---|---|---|---|
| 1 | **Stacked-LR ensemble {RF, XGB, QSVM, VQC}** | **0.986 ± 0.018** | 0.925 | 0.917 | 0.933 | 135.7 |
| 2 | **Quantum-Augmented-XGB** | **0.974 ± 0.023** | 0.900 | 0.883 | 0.917 | 15.4 |
| 3 | SoftVote {SVM-RBF, RF, XGB, QSVM} | 0.913 ± 0.050 | 0.808 | 0.733 | 0.883 | 1.1 |
| 4 | PCA-XGBoost (classical control) | 0.910 ± 0.053 | 0.867 | 0.833 | 0.900 | 0.16 |
| 5 | Raw-XGBoost (classical control) | 0.887 ± 0.076 | 0.783 | 0.750 | 0.817 | 0.12 |
| 6 | MI-CWZZ-QSVM | 0.757 ± 0.083 | 0.692 | 0.733 | 0.650 | 0.84 |
| 7 | CWZZ-permuted control | 0.689 ± 0.095 | 0.642 | 0.600 | 0.683 | 1.99 |
| 8 | CWZZ-QSVM (flagship) | 0.658 ± 0.156 | 0.617 | 0.633 | 0.600 | 2.55 |
| 9 | AE-VQC | 0.576 ± 0.151 | 0.508 | 0.533 | 0.483 | 20.96 |
| 10 | QCNN | 0.558 ± 0.109 | 0.592 | 0.633 | 0.550 | 13.39 |
| 11 | ZZ-QSVM | 0.557 ± 0.115 | 0.517 | 0.533 | 0.500 | 2.48 |
| 12 | Angle-QSVM | 0.557 ± 0.115 | 0.517 | 0.533 | 0.500 | 2.30 |

**Kennel-synth run** (80 synthetic three-phase trials, 55 features): 
`Quantum-Augmented-XGB` dominates at **0.906**; stacking underperformed there
(small trial count).

### 3.1 Findings

1. **Hybrid wins.** The stacked quantum+classical ensemble tops the board
   (0.986), and quantum-augmented XGBoost is the best single *pipeline* (0.974)
   at 9× less training cost than full stacking. This is the "best of both worlds"
   thesis: **quantum representations add value through strong classical
   learners.**
2. **Cost reality check:** backprop-diff VQC is ~5× faster than parameter-shift
   after that fix, but still costs ~100× an XGBoost fit; ensembles multiply it.
3. **Kennel synthetic:** again hybrid > pure quantum or pure classical.

### 3.2 Statistical caveat on the headline

`Quantum-Augmented-XGB` 0.974 ± 0.023 vs `PCA-XGBoost` 0.910 ± 0.053. The
difference is +0.064 AUC. The fold-to-fold sds are 0.023 and 0.053, so on a
5-fold comparison with $n=120$ samples **the gap is of the same order as the
fold-to-fold spread**. A paired test across the *same* partitions (which the
protocol makes possible, since partitions are identical) is required before
claiming significance. The committed artifacts contain per-fold metrics in
`configs/*.json`, so a Wilcoxon signed-rank or paired bootstrap is a few lines of
analysis away — **and has not been done.** Any comparative claim should be
phrased as "consistent improvement across folds under an identical protocol"
rather than "statistically significant improvement".

---

## 4. Experiment C — feature-map ablation (RQ3)

**Command**

```bash
python scripts/run_hybrid_search.py --dataset voc_real --experiment map_ablation
```

Four variants, all `reduction="pca"`, `head="qsvm"`, differing only in
`embedding` — identical gate budget, identical head, identical partitions, **one
variable**.

| Variant | `embedding` | Circuit | ROC-AUC |
|---|---|---|---|
| `Angle-QSVM` | `angle` | $K\cdot r$ RY, **no entanglement** | 0.557 ± 0.115 |
| `ZZ-QSVM` | `zz` | conventional 2nd-order Pauli-ZZ | 0.557 ± 0.115 |
| `CWZZ-QSVM` | `cwzz` | covariance-weighted ZZ (flagship) | 0.658 ± 0.156 |
| `CWZZ-permuted-control` | `cwzz_permuted` | **negative control** | 0.689 ± 0.095 |

Artifacts: `outputs/lab/20260823_084010/`.

### 4.1 The negative result — stated by the project

> Angle ≈ ZZ ≈ 0.557 < CW-ZZ 0.658 — correlation-aware encoding helps over
> conventional maps. **However the permuted-correlation control scored 0.689 ≥
> flagship 0.658: at n=96 train samples the advantage is within noise, so we do
> not yet claim correlation-weighting benefit on this dataset (honest negative
> control; rerun at larger N).**

This is the single most important methodological moment in the project, and it
should be presented as a *strength*, not a weakness:

- The permutation preserves the correlation **distribution** exactly and only
  breaks the **assignment** of correlations to qubit pairs.
- If BioZZ's benefit came from chemically-meaningful correlation weighting, the
  permuted arm should be worse. It is not.
- Therefore the +0.101 over plain ZZ appears to come from **any** non-trivial
  phase structure, not from correlation semantics.
- The flagship's own fold-to-fold sd is 0.156 — the largest of any configuration.
  That is the signature of a configuration that is not robustly better.

**What would settle it:** $n \geq 1000$, or a $K$ sweep, or repeated CV with
different seeds so the permuted control's position can be assessed
distributionally rather than from one draw.

---

## 5. Experiment D — modality ablation (RQ2)

`run_modality_ablation` builds $1 + 2G$ configurations from $G$ feature groups:
`full_multimodal`, `only_<group>`, `remove_<group>`. For `voc_real` the 27 VOC
columns are split into 3 contiguous thirds (9/9/9); for `kennel_synth` the 4
physical groups (pressure / proximity / motion / physiology) are used.

All configurations share the same `y` and the same `n_splits`/`seed`
stratification, so per-group deltas are attributable to the removed modality.

Test coverage: 2 groups (5 configurations). **No committed run artifact** for
this experiment exists in `outputs/`. If modality contribution is to be claimed,
this experiment must be run and its `leaderboard.csv` committed.

---

## 6. Experiment E — robustness

```bash
python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness
```

Noise levels $(0.0, 0.05, 0.1, 0.2, 0.3)\,\sigma$ where $\sigma$ is the
per-column standard deviation, dropout 0.0. **Train on clean, evaluate under
degradation** — the right protocol.

**Result:** kennel `CWZZ-QSVM` degrades **0.60 → 0.51** AUC as feature noise rises
0 → 0.3σ. Graceful, monotonic decay with no cliff — the expected behaviour for a
smooth degradation.

**Caveats:** train/infer times are hardcoded to 0.0 in this experiment
(`FoldResult(..., 0.0, 0.0)`), so the leaderboard's `train_time_s_mean` column is
always 0.0. The first API attempt at this experiment crashed
(`outputs/lab/20260823_085411/` has artifacts but no `_cli_meta.json`); the CLI
was fixed and the rerun at `20260823_085700` succeeded. The **API route was not
fixed** — `kennel_server.py:449` still reads a `leaderboard` key that
`run_robustness` does not return.

---

## 7. Dataset integration results

From `docs/reports/dataset_integration_report.md`.

### 7.1 Lung VOC

| Property | Value |
|---|---|
| Rows | 427 (PatientID 1–427) |
| Features | 27 VOC compound concentration columns |
| Classes | Control 193 · Cancer 157 · Benign 77 |
| `cancer_vs_control` | (350, 27) — 193 / 157 |
| `disease_vs_control` | (427, 27) — 193 / 234 |

**Data quality:** all 427 rows parse cleanly. Duplicate-looking rows exist
(patients 21/45/46 share the first 14 values) — preserved as-is. **Precision
changes mid-file**: patients ≈1–88 carry 2–3 decimals, later rows full float
precision — consistent with a simulated augmentation appended to a real cohort.
Treat absolute performance numbers with that caveat.

### 7.2 Canine ECG (secondary, annotation-only)

| Metric | Value |
|---|---|
| Recordings parsed | 1,123 |
| Distinct dogs | 40 (DogInfo covers 45, but only 444 rows match) |
| Median HR | 62.6 bpm (plausible resting canine) |
| Recordings with ≥1 bradycardia episode | 1,063 |
| Recordings >50% bad signal | 149 |

**Quality flags:** `hr_min_bpm_min = 0.62` bpm ⇒ some rows contain corrupted or
placeholder beat lists; filter on quality metrics. The exported
`has_bradycardia` target is **circular by construction** (derived from the same
annotation) and is for pipeline demos, not clinical claims.

The `ecg_data/*.wav` audio is **not distributed**, so no waveform processing is
possible and there are no diagnostic labels.

---

## 8. Code-evaluation results

From `docs/reports/code_evaluation_report.md` (repo state `efb9075`).

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | High | BioZZ covariance computed as `corrcoef(PCA scores)` ⇒ ≈identity ⇒ headline map silently inert | **Fixed** (`compute_qubit_covariance`) |
| 2 | High | `biomarker_mapper.py` hardcoded the synthetic 24/64 geometry; would mislead on real data | **Fixed** (three-tier pooling) |
| 3 | High | `server.py` trained QSVM on the full cohort without a split; same PCA-covariance defect; pseudo-SHAP | **Fixed** |
| 4 | High | No external data ingestion existed | **Fixed** (`dataset_loader`) |
| 5 | Medium | Floor-division batching dropped up to `batch−1` samples/epoch; unseeded shuffles | **Fixed** (ceil + `default_rng`) |
| 6 | Medium | QCNN pool block crashed on odd `n_qubits` | **Fixed** (modular index + validation) |
| 7 | Medium | Top-level `import shap` broke package import on minimal installs | **Fixed** (lazy import) |
| 8 | Medium | `httpx` missing from dev deps | **Fixed** (in `pyproject`; still `httpx2` in `requirements.txt`) |
| 9 | Low | Unused imports, unsorted imports | **Fixed** (ruff E4/E7/E9/F/I/B clean) |
| 10 | Low | Benchmark plot `ylim(0.5, 1.05)` clipped sub-chance bars | **Fixed** (`(0, 1.05)`) |
| 11 | Info | `SVC(probability=True)` deprecated in sklearn 1.9 | **Open in `lab/pipeline.py`** (fixed in `baselines.py` and `qsvm.py`) |
| 12 | Info | README advertises Qiskit; no Qiskit dependency or code | **Open** — and the frontend footer still says "Qiskit" |
| 13 | Info | `hardware_profiler.py`, `qcnn.py`, `qrc.py` dead/unwired | **Partially open** (profiler and QCNN now wired; QRC still dead) |

**Verification at the time of the report:** `pytest -q` → 40 passed;
`ruff check` → all checks passed; frontend lint + build → clean static export of
`/`, `/diagnostics`, `/collect`. The current tree has **59** tests (the
19-test `test_lab_pipeline.py` postdates the report), so "40 passed" is stale.

---

## 9. Kennel subsystem status

| Component | Status |
|---|---|
| ESP32 firmware (13 sensors, 6-state machine, NDJSON telemetry) | implemented, **not compiled** (`secrets.h` missing) |
| TCP ingestion + reconnect | implemented, untested against a real board |
| Simulation fallback (100 Hz, seed 42) | implemented, 1 test |
| 45-feature window extractor | implemented, 3 tests |
| 55-feature trial extractor | implemented |
| Session recorder + CSV format | implemented, 3 tests |
| RandomForest trainer + LOSO | implemented, **0 tests**, no artifact |
| Live server + WebSockets | implemented, 2 tests |
| Next.js console (5 pages, 8 components) | implemented, static export clean |
| PyInstaller + Inno Setup + macOS DMG + Linux deb | scripted, CI-configured, no PR gate |
| **Labelled multi-dog kennel data** | **absent** — 1 dog, 2 labels, no `control/` |
| **`models/kennel_model.joblib`** | **absent** ⇒ diagnostics report `"untrained"` |

---

## 10. Summary scorecard

| Dimension | Assessment | Evidence |
|---|---|---|
| **Engineering completeness** | High | 3 subsystems, 59 tests, static console, desktop packaging for 3 OSes, CI release pipeline |
| **Leak-freeness discipline** | High | train-only scaler/selector/bounds/covariance; identical CV partitions; OOF augmentation; LOSO for the kennel model |
| **Quantum circuit realism** | Medium | genuine multi-qubit circuits, correct fidelity kernel math, but simulated and small; no hardware execution |
| **Hybrid thesis** | Supported | 0.974 hybrid vs 0.910 classical control vs 0.658 pure quantum, identical protocol |
| **BioZZ correlation-weighting claim** | **Not supported** | permuted control 0.689 ≥ flagship 0.658 |
| **Pure-quantum advantage** | **Not observed** | QSVM 0.507 / VQC 0.625 on real data vs 0.972–1.000 classical |
| **Explainability** | Partial | kernel SHAP exists; compound-level naming on real data is a positional heuristic; non-PCA selector returns a uniform constant |
| **Kennel hardware** | Prototype | firmware written but not compiled; a real 100 Hz↔10 Hz sampling mismatch breaks two frequency features |
| **Statistical rigour** | Medium | 5-fold CV with identical partitions and per-fold artifacts, but no paired significance test and `ddof=0` fold sd |
| **Documentation honesty** | High | negative controls shipped and reported; known defects recorded rather than hidden |
