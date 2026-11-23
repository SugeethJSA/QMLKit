# 07 — Classical Baselines and Evaluation Metrics

A quantum comparison is only meaningful if the classical arm is strong, fitted
under an identical protocol, and scored with the same function. This document
covers both sides of that requirement.

---

## 1. `classical/baselines.py` — the benchmark-suite baselines

`get_all_classical_baselines(random_state=42)` → `ClassicalBaselineSuite.get_baselines()`.

### 1.1 Exact hyperparameters

```python
calibrated_svm = lambda **kw: CalibratedClassifierCV(SVC(**kw), ensemble=False)
```

| Key | Estimator | Hyperparameters |
|---|---|---|
| `SVM_RBF` | `CalibratedClassifierCV(SVC(...), ensemble=False)` | `kernel="rbf", C=1.5, gamma="scale", random_state=seed` |
| `SVM_Linear` | same wrapper | `kernel="linear", C=1.0, random_state=seed` |
| `Logistic_Regression` | `LogisticRegression` | `C=1.0, max_iter=1000, random_state=seed` |
| `Random_Forest` | `RandomForestClassifier` | `n_estimators=150, max_depth=8, min_samples_split=4, random_state=seed` |
| `MLP_NeuralNet` | `MLPClassifier` | `hidden_layer_sizes=(64,32), activation="relu", alpha=0.001, max_iter=300, random_state=seed` |
| `XGBoost` *(if `HAS_XGBOOST`)* | `XGBClassifier` | `n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, eval_metric="logloss", random_state=seed` |
| `Gradient_Boosting` *(fallback)* | `GradientBoostingClassifier` | `n_estimators=100, max_depth=4, learning_rate=0.08, random_state=seed` |

`HAS_XGBOOST` is a guarded import (`:18-22`); `xgboost` is in the **dev extra**,
not core dependencies, so a minimal install silently gets the sklearn gradient
booster instead. That changes the leaderboard row label — worth stating in any
report.

**Why `CalibratedClassifierCV` rather than `SVC(probability=True)`:** the source
comment (`:103-104`) records two reasons — the sklearn 1.9 deprecation, and
"better-behaved screening-risk scores". Internal 5-fold Platt scaling /
isotonic-free sigmoid calibration gives probabilities usable for the Brier score
and the ROC-AUC, which `SVC(probability=True)`'s internal
`libsvm` cross-validation also provides but with a different variance profile.

### 1.2 `Temporal1DCNN` — the deep baseline

`baselines.py:25-94`. A proper `BaseEstimator, ClassifierMixin`, so it can go
into sklearn utilities. Consumes a **3-D tensor** `(N, n_sensors, timesteps)`.

```
Conv1d(16 → 32, k=5, pad=2) → BatchNorm1d(32) → ReLU → MaxPool1d(2)
Conv1d(32 → 64, k=3, pad=1) → BatchNorm1d(64) → ReLU
AdaptiveAvgPool1d(1) → Flatten → Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 2)
```

Training: `Adam(lr=1e-3, weight_decay=1e-4)`, `CrossEntropyLoss`,
`DataLoader(batch_size=32, shuffle=True)`, `epochs=25`. No validation split, no
early stopping, no per-epoch shuffling seed control.

**Status: defined and tested, but never in a leaderboard.** It is not returned by
`get_baselines()` and not referenced by `BenchmarkSuite`. It is the natural
consumer of `BiomimeticVOCGenerator.raw_time_series` — the same unused tensor
`QuantumReservoirClassifier` would use. `tests/test_classical_baselines.py` does
fit it on a `(12, 16, 50)` tensor, so the code path is exercised.

### 1.3 Baseline quality gate

The `Random_Forest`, `MLP_NeuralNet`, `XGBoost` and SVM configurations were
chosen to be *strong* — `max_depth=8` with 150 trees, XGBoost with `depth=4` and
100 rounds. On the real VOC dataset they reach 0.97–1.000 ROC-AUC
([`15_results_and_findings.md`](15_results_and_findings.md) §2). This matters:
a weak baseline would make quantum models look better than they are.

---

## 2. `compute_clinical_metrics` — the 10 metrics

`evaluation/benchmark_suite.py:47-97`.

```python
@dataclass
class ModelEvaluationMetrics:
    model_name: str
    paradigm: str            # "Quantum" | "Classical" | "Hybrid"
    accuracy: float
    balanced_accuracy: float
    sensitivity_recall: float
    specificity: float
    precision_ppv: float
    negative_predictive_val: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float
    brier_score: float
    train_time_sec: float = 0.0
    inference_time_ms: float = 0.0
```

All values `round(x, 4)`.

### 2.1 Formulas

| Metric | Source | Notes |
|---|---|---|
| `accuracy` | `accuracy_score` | |
| `balanced_accuracy` | `balanced_accuracy_score` | mean of per-class recall |
| `sensitivity_recall` | `recall_score(y, pred, zero_division=0)` | TP / (TP+FN) |
| `precision_ppv` | `precision_score(y, pred, zero_division=0)` | TP / (TP+FP) |
| `f1_macro` / `f1_weighted` | `f1_score(average="macro"/"weighted")` | |
| `specificity` | from confusion matrix | `tn / (tn + fp)`, else `0.0` |
| `negative_predictive_val` | from confusion matrix | `tn / (tn + fn)`, else `0.0` |
| `roc_auc` | `roc_auc_score(y, p_pos)` | **fallback**: `= balanced_accuracy` |
| `brier_score` | `brier_score_loss(y, p_pos)` | **fallback**: `mean((pred - y)²)` |

Confusion matrix (`:65-66`):

```python
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
```

Using `labels=[0, 1]` pins the matrix shape to 2×2 even if a fold contains only
one class — without it, `ravel()` would produce 4 elements for a 1×1 matrix or
raise on a 2×1.

### 2.2 Why clinical metrics rather than accuracy

The screening framing makes **sensitivity** and **specificity** the operationally
meaningful quantities, and they trade off. A model that achieves 1.000 accuracy
by always predicting the majority class would have balanced accuracy ≈ 0.5 and
sensitivity 0. The suite reports all ten so a reviewer can see the whole
operating point, and the `paradigm` field lets the leaderboard be grouped by
quantum/classical/hybrid (which is exactly what the seaborn bar plot does, coloured
`Quantum: #7c3aed`, `Classical: #2563eb`).

### 2.3 Metric fallbacks — a real caveat

```python
if y_prob is not None:
    try:    auc_val = float(roc_auc_score(y_true, pos_prob))
    except Exception: auc_val = float(bal_acc)          # ← silent substitution
    brier = float(brier_score_loss(y_true, pos_prob))
else:
    auc_val = float(bal_acc)
    brier   = float(np.mean((y_pred - y_true) ** 2))
```

**A reported `roc_auc` may not be a ROC-AUC.** If `roc_auc_score` raises (single
class in the test fold, or all-identical probabilities), the value is replaced by
the balanced accuracy *with no marker in the output*. In the lab's
`summarise_folds`, NaN values are filtered out of the mean/std — so a fallback
would silently enter the average as a balanced accuracy. A reviewer should check
`leaderboard.csv` for suspiciously-equal `roc_auc_mean` and
`balanced_accuracy_mean` columns.

`f1_weighted` is computed and stored in the dataclass but **omitted from the
`training_service.py` API response dict** (`:332-361`).

---

## 3. `BenchmarkSuite.run_benchmark`

`benchmark_suite.py:156-278`.

```python
def __init__(self, n_qubits=6, random_state=42, vqc_epochs=20)
```

### 3.1 Order of operations

```python
pipeline = BiomedicalDataPipeline(scaler_type="standard").fit(X_train_raw)  # train only
X_tr_scaled = pipeline.transform(X_train_raw)
X_te_scaled = pipeline.transform(X_test_raw)

selector = QuantumFeatureSelector(n_qubits=self.n_qubits, method="pca").fit(X_tr_scaled)
X_tr_q = selector.transform(X_tr_scaled)
X_te_q = selector.transform(X_te_scaled)

cov_matrix = compute_qubit_covariance(X_tr_scaled, selector)
```

Note the classical baselines are fitted on **`X_tr_scaled`** (the 27-dim scaled
matrix) while the quantum models see **`X_tr_q`** (the 6-dim reduced matrix). The
comparison is therefore *not* purely "algorithm A vs algorithm B" — it is also
"full feature set vs 6-component projection". This is a defensible choice (the
quantum models *cannot* use 27 qubits at NISQ scale) but it must be stated.

### 3.2 Models evaluated

| Order | Model | paradigm | head input |
|---|---|---|---|
| 1 | `QSVM_BioZZ` | Quantum | `X_tr_q` (6-dim) |
| 2 | `VQC_StronglyEntangled` | Quantum | `X_tr_q` (6-dim) |
| 3 | `Quantum_Kernel_XGB` | Hybrid | **raw `X_train_raw`** (whole `HybridPipeline`) |
| 4 | `Quantum_Augmented_XGB` | Hybrid | raw `X_train_raw` |
| 5–10 | all classical baselines | Classical | `X_tr_scaled` (27-dim) |

The two hybrid arms are wrapped in a `try/except Exception: pass` (`:258-259`)
with the comment *"hybrid is best-effort; keep quantum/classical on failure"*.
Consequence: **if a hybrid arm fails, the leaderboard silently has 2 fewer rows
and nothing in the output records that.** A reviewer must check row counts.

The hybrid arms receive the **raw** `X_train_raw` (unscaled DataFrame) because
`HybridPipeline` does its own imputation + scaling internally. The classical arms
receive the pre-scaled matrix from the outer `BiomedicalDataPipeline`. So
`Quantum_Augmented_XGB`'s `X_scaled` is its *own* StandardScaler output, not the
suite's — two scalers in the same run, both train-fitted, so no leak, but a
detail to note.

### 3.3 Timing methodology

```python
t0 = time.time(); qsvm.fit(X_tr_q, y_train); t_train_qsvm = time.time() - t0
t0 = time.time(); preds = qsvm.predict(X_te_q); probs = qsvm.predict_proba(X_te_q)
t_infer_qsvm = (time.time() - t0) * 1000.0 / len(y_test)
```

Inference time is **wall time for the whole prediction call divided by test-set
size**, in ms/sample. For QSVM this includes **two** independent
`compute_kernel_matrix` calls (`predict` and `predict_proba` each recompute the
kernel), so it is roughly 2× the cost of a single prediction — a conservative
over-estimate, not an under-estimate. For VQC, `predict` + `predict_proba` each
run the full QNode sweep, again 2×.

### 3.4 Output

```python
df_res = pd.DataFrame([asdict(r) for r in results])
return df_res.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
```

Sorted by ROC-AUC descending. `scripts/run_benchmark.py` then writes
`benchmark_metrics.csv`, renders a seaborn ROC-AUC bar plot
(`ylim=(0, 1.05)` — deliberately including 0 so sub-chance bars are not clipped;
that was finding #10), and dumps `run_metadata.txt` with source, task, N,
train/test sizes, class counts, qubits, and seed.

---

## 4. Real-data results — the honest picture

`python scripts/run_benchmark.py --source markdown --voc-task cancer_vs_control \
    --max-samples 120 --vqc-epochs 8 --n-qubits 6 --output-dir outputs/benchmark_real`

Split: stratified 80/20 → **96 train / 24 test**.

| Model | Paradigm | Accuracy | Bal. Acc | Sensitivity | Specificity | **ROC-AUC** | Brier | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | Classical | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.010 | 0.13 |
| Random Forest | Classical | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.041 | 0.13 |
| MLP NeuralNet | Classical | 0.958 | 0.958 | 0.917 | 1.000 | **1.000** | 0.020 | 0.16 |
| SVM Linear | Classical | 0.958 | 0.958 | 0.917 | 1.000 | 0.979 | 0.061 | 0.00 |
| SVM RBF | Classical | 0.917 | 0.917 | 0.917 | 0.917 | 0.972 | 0.074 | 0.00 |
| VQC StronglyEntangled | Quantum | 0.667 | 0.667 | 0.500 | 0.833 | **0.625** | 0.245 | 343.5 |
| QSVM BioZZ | Quantum | 0.542 | 0.542 | 0.417 | 0.667 | **0.507** | 0.250 | 0.74 |

Artifacts: `outputs/benchmark_real/benchmark_metrics.csv`,
`benchmark_comparison.png`, `run_metadata.txt`.

### 4.1 What this table says

1. **The real VOC dataset is highly separable for strong classical learners.**
   Even a *linear* SVM reaches 0.979 AUC. A linear model on 27 features
   separating cancer from control means the signal is close to a hyperplane in
   concentration space — which is exactly the regime where a fidelity kernel
   cannot help, because the kernel's advantage is in modelling *nonlinear*
   structure.
2. **Quantum models are at or near chance.** QSVM 0.507 ≈ coin flip; VQC 0.625 is
   weak. This is measured **with** the covariance fix active.
3. **Cost asymmetry is 2,600×.** VQC 343.5 s vs XGBoost 0.13 s.
4. **$n=24$ on the test set.** A perfect 24/24 is 1.000 with a Wilson 95% CI of
   roughly [0.86, 1.00]. The 1.000s should be read as "at ceiling", not exact.

### 4.2 Why the quantum arms are weak — root causes

| Cause | Mechanism | Evidence |
|---|---|---|
| Information loss in reduction | 27 → 6 PCA components discards most of the variance that carries the signal | 6-qubit depth 94 vs 27-qubit depth 706; a linear SVM already solves the full space |
| Tiny training set for a kernel method | Fidelity kernels have no generalisation advantage at $n=96$; SVC's margin has 96 points to work with | 96 train / 24 test |
| Covariance is a proxy, not a measurement | With PCA reduction, $\rho_{ij}$ is a max-correlation between top-3 loading features, not a compound-pair correlation | [`03_preprocessing_and_reduction.md`](03_preprocessing_and_reduction.md) §5.2b |
| Degenerate separability target | The task is near-linearly separable, so the kernel has no nonlinear structure to exploit | SVM_Linear 0.979 |
| VQC under-trained | 8 epochs instead of 20; `sigmoid(2·raw)` cannot produce confident probabilities | Brier 0.245 |

### 4.3 What the project recommends next

From `docs/reports/real_data_benchmark_report.md:55-62`:

- k-fold CV instead of one split (the `n_splits_cv` config field exists but is
  unused in this path)
- sweep `n_qubits ∈ {4,6,8}` × selector `{pca, mutual_info}`; `mutual_info`
  selects *raw compounds*, which gives BioZZ genuinely interpretable
  correlations
- re-scale for angle encoding (per-feature min–max on raw concentrations spans
  decades between compounds)
- full-N runs on faster hardware

The lab has since implemented the k-fold CV and the `mutual_info` selector — see
[`08_experiment_lab.md`](08_experiment_lab.md) and the `MI-CWZZ-QSVM` result
(0.757 ± 0.083, the best *pure-quantum* arm in the 5-fold leaderboard). The
recommendations are therefore partly actioned, which is a good story for the
presentation.

---

## 5. Test coverage

`tests/test_classical_baselines.py` (3 tests):

1. All baselines `fit`/`predict` with probability shape `(6, 2)` — validates the
   `CalibratedClassifierCV` wrapper and the `HAS_XGBOOST` branch.
2. `Temporal1DCNN` on a `(12, 16, 50)` tensor.
3. `compute_clinical_metrics` against a hand-built confusion matrix: asserts
   `accuracy == sensitivity == specificity == 0.75` and `roc_auc ∈ [0,1]`.

`tests/test_e2e_workflow.py` (2 tests):

1. `BenchmarkSuite.run_benchmark` on a synthetic cohort → ≥5 rows, and both
   `QSVM_BioZZ` and `SVM_RBF` present in `model_name`.
2. FastAPI `TestClient`: `/api/v1/health` returns `"HEALTHY"`, `/` contains
   `"QMLKit"`, and `POST /api/v1/predict` with a 64-element `sensor_readings`
   payload returns `cancer_probability`, `top_biomarkers`, `biochemical_pathways`.

**Not covered:** the `roc_auc`/`brier` fallback branches; the `roc_auc_score`
`except` path; the `HybridPipeline` `except Exception: pass` in the benchmark
suite.
