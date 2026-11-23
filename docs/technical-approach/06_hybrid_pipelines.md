# 06 — Hybrid Pipelines

`src/qmlkit/lab/pipeline.py` (423 lines) — the composable pipeline that the
experiment lab, the console's `/lab` page, and the Training Studio's hybrid
options all instantiate.

---

## 1. Design: one declarative spec, many configurations

The module docstring states the design intent (`pipeline.py:1-16`):

```
reduction  : none | pca | mutual_info | autoencoder
embedding  : none | angle | zz | cwzz | cwzz_permuted
head       : qsvm | vqc | qcnn | logistic_regression | svm_rbf | svm_linear |
             random_forest | xgboost | mlp | quantum_augmented_xgb
```

> **Doc drift:** `quantum_kernel_xgb` is fully implemented (`:125`, `:259`,
> `:381`) and shipped as preset #8, but is missing from this docstring tuple.

```python
REDUCTION_METHODS = ("none", "pca", "mutual_info", "autoencoder")     # :43
EMBEDDINGS        = ("none", "angle", "zz", "cwzz", "cwzz_permuted")  # :44
CLASSICAL_HEADS   = ("logistic_regression", "svm_rbf", "svm_linear",
                     "random_forest", "xgboost", "mlp")                # :45-52
```

### 1.1 `PipelineSpec`

```python
@dataclass
class PipelineSpec:
    name: str
    reduction: str = "pca"
    n_components: int = 6
    embedding: str = "cwzz"
    head: str = "qsvm"
    covariance_mode: str = "train"     # "train" | "identity" | permuted via suffix
    vqc_epochs: int = 8
    n_landmarks: int = 12
    seed: int = 42
    meta: Dict[str, Any] = field(default_factory=dict)
```

`to_dict()` (`:159-171`) flattens `**self.meta` into the top level, so meta keys
**override** the fixed keys — useful for annotating run artifacts.

### 1.2 Two orthogonal concepts

| Axis | Question it answers |
|---|---|
| `embedding` | *How does the quantum map weight pairwise interactions?* — `cwzz` uses train correlations, `cwzz_permuted` is the correlation-control experiment |
| `head` | *Who makes the final decision?* — a quantum model, a classical model, or a fusion of both |

`embedding="none"` + `head="xgboost"` is a pure-classical control.
`embedding="cwzz"` + `head="qsvm"` is pure quantum.
`embedding="cwzz"` + `head="quantum_augmented_xgb"` is the hybrid thesis.

---

## 2. `_make_head` — the head factory

`pipeline.py:69-141`. Embedding → feature-map type mapping (repeated in three
places: `:75-80`, `:91-96`, `:260-265`):

| `spec.embedding` | `feature_map_type` |
|---|---|
| `angle` | `Angle` |
| `zz` | `ZZ` |
| `cwzz` | `BioZZ` |
| `cwzz_permuted` | `BioZZ` |
| `none` / unknown | `ZZ` |

| `head` | Construction | Hyperparameters |
|---|---|---|
| `qsvm` | `QSVMClassifier` | `n_qubits=n_components`, `c_param` = default 1.0 |
| `vqc` | `VariationalQuantumClassifier` | `n_qubits=n_components`, **`n_layers=2` hardcoded**, `epochs=vqc_epochs`, lr default 0.03 |
| `qcnn` | `QuantumConvolutionalClassifier` | **`n_qubits=max(2, n_components)`**, **`feature_map_type="BioZZ"` hardcoded (ignores `embedding`)**, `epochs=vqc_epochs` |
| `logistic_regression` | `LogisticRegression` | `max_iter=1000, random_state=seed` |
| `svm_rbf` | `SVC` | `kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=seed` |
| `svm_linear` | `SVC` | `kernel="linear", C=1.0, probability=True, random_state=seed` |
| `random_forest` | `RandomForestClassifier` | `n_estimators=150, max_depth=8, random_state=seed` |
| `xgboost` / `quantum_*_xgb` | `XGBClassifier` if `HAS_XGB` else `GradientBoostingClassifier` | XGB: `n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, eval_metric="logloss", random_state=seed`. GB fallback drops `subsample`/`eval_metric` |
| `mlp` | `MLPClassifier` | `hidden_layer_sizes=(64,32), max_iter=300, random_state=seed` |
| anything else | `raise ValueError(f"Unknown head: {spec.head}")` | |

**Divergences from `classical/baselines.py`** (worth flagging in review — the
"same" model is not identical between the benchmark suite and the lab):

| Model | `baselines.py` | `lab/pipeline.py` |
|---|---|---|
| SVMs | `CalibratedClassifierCV(SVC(...), ensemble=False)` | `SVC(probability=True)` — the deprecated path |
| RandomForest | `+ min_samples_split=4` | no `min_samples_split` |
| MLP | `+ activation="relu", alpha=0.001` | omitted |

So a `Random_Forest` head in the lab and a `Random_Forest` baseline in the
benchmark suite are **not the same estimator**. Comparisons across the two code
paths should be treated as indicative, not exact.

---

## 3. `HybridPipeline.fit` — the leak-free sequence

`pipeline.py:231-366`.

```
1. X_arr = X.values if DataFrame else np.asarray(X)
2. X_arr = self._impute(X_arr, fitted=False)              # train medians stored
3. self.scaler = StandardScaler().fit(X_arr)               # TRAIN ONLY
4. X_scaled = self.scaler.transform(X_arr)
5. if reduction != "none":
       self.selector = QuantumFeatureSelector(n_qubits, method=reduction).fit(X_scaled, y)
6. X_reduced = self._reduce(self.selector, X_scaled)      # + angle re-scaling
7. if embedding in ("cwzz", "cwzz_permuted"):
       base_cov = compute_qubit_covariance(X_scaled, self.selector) or eye
       shape guard → eye(n) on mismatch
       self.cov_matrix = _permute_correlation(base_cov, seed) if permuted else base_cov
   else:
       self.cov_matrix = None
8. head dispatch (three branches, §4)
9. self.head.fit(X_head, y)
```

The critical property: **the same fitted objects (`scaler`, `selector`,
`cov_matrix`, landmark states, OOF scaler state) are reused unchanged at
transform time** in `transform_input` (`:368-376`):

```python
def transform_input(self, X):
    if not self.is_fitted: raise RuntimeError("Pipeline must be fitted first.")
    X_arr = self._values(X); X_arr = self._impute(X_arr, fitted=True)
    return self.scaler.transform(X_arr), self._reduce(self.selector, X_arr_scaled)
```

### 3.1 `_reduce` when `reduction == "none"`

```python
n = min(self.spec.n_components, X_scaled.shape[1]) if X_scaled.shape[1] > self.spec.n_components \
    else X_scaled.shape[1]
return X_scaled[:, :n]
```

Effectively a column truncation to the first `n_components` columns. The only
presets that use it are the classical controls (`Raw-XGBoost`, `Raw-XGBoost` with
`n_components=32`), and with no selector there is no angle re-scaling — raw
standardised features go straight to a tree ensemble, where bounded angles are
irrelevant.

---

## 4. The three head branches

### 4.1 `quantum_kernel_xgb` — quantum-kernel feature fusion

`pipeline.py:259-293`.

```
fit:
  transformer = QuantumKernelFeatureTransformer(
      n_qubits = X_reduced.shape[1],
      feature_map_type = <from embedding>,
      covariance_matrix = self.cov_matrix,
      n_landmarks = spec.n_landmarks,        # 12
      seed = spec.seed)
  quantum_features = transformer.fit_transform(X_reduced, y)      # (N, L)
  X_head = np.hstack([X_reduced, quantum_features])               # (N, K + L)
  head    = _make_head(replace(spec, head="xgboost"), cov_matrix=None)

predict_proba:
  qf = transformer.transform(X_reduced)          # against STORED landmark states
  proba = head.predict_proba(np.hstack([X_reduced, qf]))
```

**What the quantum part contributes:** $L=12$ fidelity similarities to
class-balanced landmark samples, in addition to the 6 reduced angles. The
XGBoost head can then split on "is this sample quantum-similar to a known
positive?" — a genuinely different feature family from a linear angle.

**Leak-freeness:** landmarks are selected in `fit` from training data only and
their statevectors are cached; `transform` reuses them unchanged.

**Cost:** $O((N + L))$ circuit simulations at fit, $O(N_{\text{batch}} + L)$ per
inference batch. Cheap compared to the VQC.

### 4.2 `quantum_augmented_xgb` — out-of-fold VQC feature augmentation

`pipeline.py:295-354`. **This branch contains the project's most important
methodological fix.**

```python
# Leak-free: OOF VQC predictions for XGB training
# (prevents in-sample optimism that inflated 100% scores)
n_splits = 3
_, counts = np.unique(y, return_counts=True)
if min(counts) >= n_splits and len(y) >= n_splits * 2:
    oof = np.zeros((len(y), 1), dtype=float)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=spec.seed)
    for tr_idx, va_idx in skf.split(X_reduced, y):
        fold_aug = VariationalQuantumClassifier(
            n_qubits=X_reduced.shape[1], n_layers=1,
            feature_map_type="BioZZ" if spec.embedding.startswith("cwzz") else "ZZ",
            epochs=max(1, spec.vqc_epochs // 2),       # cheaper per fold
            covariance_matrix=self.cov_matrix)
        fold_aug.fit(X_reduced[tr_idx], y[tr_idx])
        oof[va_idx, 0] = fold_aug.predict_proba(X_reduced[va_idx])[:, 1]
    aug_proba = oof
else:
    raise ValueError("insufficient samples for OOF")
```

**The defect this fixes:** if the VQC's probability on the *same* rows it was
trained on is used as an XGBoost feature, the feature is a near-perfect training
label in-sample. XGBoost will lean on it, the training score saturates, and the
held-out score collapses. The comment records that the un-OOF version produced
**100% scores**.

**Three-fold, not 5:** $3 \times \lceil \text{epochs}/2 \rceil$ circuit
evaluations must fit in a reasonable wall time. `vqc_epochs // 2` halves the
per-fold cost so total cost stays near one full VQC fit.

**Then:**

```python
X_head = np.hstack([X_scaled, aug_proba])   # NOTE: full-dimension X_scaled, not X_reduced
self.head = _make_head(replace(spec, head="xgboost"), cov_matrix=None)
# Separate inference-time augmenter, refit on ALL training data
self.augmenter = VariationalQuantumClassifier(..., epochs=spec.vqc_epochs)
self.augmenter.fit(X_reduced, y)
```

Two consequences worth naming in review:

1. **Asymmetric feature spaces between fit and predict.** At fit, the head sees
   `X_scaled` (all $M$ columns) + OOF VQC probability. At predict, it sees
   `X_scaled` + the probability of an augmenter refit on all training data. The
   *column* layout is consistent, but the *distribution* of the quantum feature
   differs between the two (OOF probabilities are systematically less confident
   than in-sample ones). This is a known limitation of the OOF-augmentation
   pattern; the code comment at `:296` is candid about the trade.
2. **The VQC sees the reduced features, the head sees the full features.** The
   quantum branch operates on 6 angles; the classical branch on all $M$ scaled
   columns plus 1 quantum scalar. So the "quantum contribution" to this head is
   exactly **one scalar column**.

**Fallback:** a bare `except Exception` (`:320`) falls back to a single in-sample
VQC on the full training set — i.e. it silently reverts to the leaky behaviour
when the OOF guard fails. The intent is documented ("original behavior if OOF
fails") but the bare `except` will also swallow genuine bugs (e.g. a
`QuantumFeatureSelector` failure) with no signal in the artifacts.

### 4.3 Everything else — `X_head = X_reduced`

`pipeline.py:356-362`. Pure quantum heads and pure classical heads consume the
reduced (and angle-bounded) representation.

---

## 5. `predict_proba` and `predict`

`pipeline.py:378-420`.

```python
def predict_proba(self, X):
    X_scaled, X_reduced = self.transform_input(X)
    if head == "quantum_kernel_xgb":       ... hstack([X_reduced, qf])  → head
    elif head == "quantum_augmented_xgb":  ... hstack([X_scaled, aug])  → head
    else:                                                  X_reduced   → head
    return proba if proba.ndim == 2 else np.vstack([1 - proba, proba]).T

def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1).astype(int)
```

The `ndim == 1` → 2-column `vstack` normalisation handles estimators that return
a flat positive-class score (`RidgeClassifier`-style). All the configured heads
already return `(N, 2)`, so this is defensive.

---

## 6. `_describe()` — NISQ profile emission

`pipeline.py:214-228`.

```python
record = QuantumHardwareProfiler.describe_model(
    feature_map_type={"cwzz": "BioZZ", "cwzz_permuted": "BioZZ"}
                      .get(spec.embedding, spec.embedding.title()),
    n_qubits=spec.n_components, reps=2,
    variational={"ansatz_type": "StronglyEntangling",
                 "n_qubits": spec.n_components, "n_layers": 2}
                 if spec.head in ("vqc", "quantum_augmented_xgb") else None)
record["spec"] = spec.to_dict()
return record
```

Emitted per configuration into `outputs/lab/<run_id>/configs/*.json`. Verdict
logic and full accounting: [`14_nisq_feasibility.md`](14_nisq_feasibility.md).

Note `embedding="none"` → `"None".title()` = `"None"` → the profiler's else
branch (the non-ZZ, non-Angle path). `QuantumHardwareProfiler` treats anything
that is not ZZ-family and not `"angle"` as Angle-like, so classical controls get
`ry_rotations = n·reps` gate counts — a nominal profile, not a real circuit.

---

## 7. Measured hybrid results

Real Lung VOC (350 → 120 balanced cap), stratified 5-fold CV, identical
partitions across all configurations, seed 42
(source: `outputs/lab/20260823_082122/leaderboard.csv`):

| Rank | Config | reduction | embedding | head | ROC-AUC | Train s |
|---|---|---|---|---|---|---|
| 1 | **Stacked-LR ensemble {RF, XGB, QSVM, VQC}** | — | — | stacking | **0.986 ± 0.018** | 135.7 |
| 2 | **Quantum-Augmented-XGB** | pca | cwzz | `quantum_augmented_xgb` | **0.974 ± 0.023** | 15.4 |
| 3 | SoftVote {SVM-RBF, RF, XGB, QSVM} | — | — | soft voting | 0.913 ± 0.050 | 1.1 |
| 4 | PCA-XGBoost (classical control) | pca | none | xgboost | 0.910 ± 0.053 | 0.16 |
| 5 | Raw-XGBoost (classical control) | none (32) | none | xgboost | 0.887 ± 0.076 | 0.12 |
| 6 | MI-CWZZ-QSVM | mutual_info | cwzz | qsvm | 0.757 ± 0.083 | 0.84 |
| 7 | CWZZ-permuted control | pca | cwzz_permuted | qsvm | 0.689 ± 0.095 | 1.99 |
| 8 | CWZZ-QSVM (flagship) | pca | cwzz | qsvm | 0.658 ± 0.156 | 2.55 |
| 9 | AE-VQC | autoencoder | cwzz | vqc | 0.576 ± 0.151 | 20.96 |
| 10 | QCNN | pca | cwzz | qcnn | 0.558 ± 0.109 | 13.39 |
| 11 | ZZ-QSVM | pca | zz | qsvm | 0.557 ± 0.115 | 2.48 |
| 12 | Angle-QSVM | pca | angle | qsvm | 0.557 ± 0.115 | 2.30 |

On synthetic kennel trials (80 trials, 55 features) `Quantum-Augmented-XGB`
dominates at **0.906**.

### 7.1 The result that matters

> Hybrid > pure quantum, and hybrid > pure classical. Quantum-Augmented-XGB
> (0.974) beats PCA-XGBoost (0.910) by +0.064 AUC at 96× the training cost
> (15.4 s vs 0.16 s). The Stacked-LR ensemble reaches 0.986 but costs 135.7 s.

The project's thesis — *"quantum representations add value **through** strong
classical learners"* — is supported by these numbers, with the caveat that both
hybrid arms are also 1–2 orders of magnitude more expensive and that the CV
standard deviations (0.023–0.053) are comparable to the hybrid-vs-classical gap
in the single-split benchmark. See
[`15_results_and_findings.md`](15_results_and_findings.md) for the full
statistical discussion.

### 7.2 The result that does not support the headline

`CWZZ-QSVM` (0.658) vs `CWZZ-permuted control` (0.689) vs `ZZ-QSVM` (0.557).
The correlation-weighted map beats plain ZZ by +0.101, but the **permuted
control beats the flagship by +0.031**. Since the permutation preserves the
correlation distribution and only breaks the assignment to qubit pairs, the
+0.101 over ZZ appears to come from *any* non-trivial phase structure rather than
from chemically-meaningful correlation weighting. The project reports this
explicitly as a negative result.
