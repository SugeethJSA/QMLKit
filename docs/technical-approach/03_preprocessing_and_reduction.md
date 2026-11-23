# 03 — Preprocessing and Dimensionality Reduction

Everything in this document is concerned with one constraint: **a quantum
register has a fixed number of qubits, and the input does not.** The reduction
stage is where information is lost, so its leak-freeness and its behaviour are
both load-bearing.

---

## 1. Stage order

```
raw features  →  [1] NaN imputation  →  [2] StandardScaler
              →  [3] QuantumFeatureSelector (fit on train only)
              →  [4] min–max re-scaling to [angle_low, angle_high]
              →  [5] compute_qubit_covariance (train-only correlation matrix)
              →  circuit input angles x ∈ [-π, π]^n_qubits
```

Stages 1–2 live in `HybridPipeline`; stage 3–4 in
`data/feature_selector.py`; stage 5 in `evaluation/benchmark_suite.py`.
`BiomedicalDataPipeline` provides stage 2 (and the split) for the benchmark path.

---

## 2. Stage 1 — NaN imputation (median, train-only)

`HybridPipeline._impute()` (`lab/pipeline.py:192-207`):

```python
X = np.array(X_arr, dtype=float, copy=True)   # DataFrame views are read-only
if not np.isnan(X).any():
    return X
if fitted:                                    # transform path
    medians = self.impute_medians             # from training
    if medians is None:
        raise RuntimeError("Imputer state missing.")
else:                                         # fit path
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    self.impute_medians = medians
idx = np.where(np.isnan(X))
X[idx] = medians[idx[1]]
```

Why it matters: the kennel telemetry protocol uses `-1.0` as the
"physiology probe not attached" sentinel, and `kennel_synth.py:156` converts
those to NaN. Without imputation, a window without the MAX30102 probe would
produce NaN angles and a corrupt Hilbert state. Columns that are *entirely* NaN
in train get median 0.0 rather than propagating NaN.

`scripts/train_kennel_model.py:103-104` performs the equivalent
`np.nanmedian` imputation for the RandomForest artifact path.

Verified by `tests/test_lab_pipeline.py::TestHybridPipeline` — NaN train-median
imputation yields finite probabilities.

---

## 3. Stage 2 — StandardScaler (fitted on train only)

`HybridPipeline.fit` (`lab/pipeline.py:236-237`):

```python
self.scaler = StandardScaler().fit(X_arr)      # train only
X_scaled   = self.scaler.transform(X_arr)
```

`BiomedicalDataPipeline` (`data/preprocessor.py:27-57`) is the benchmark-path
equivalent and adds `feature_names` capture, `RobustScaler` as an alternative,
and `save`/`load` via pickle:

```python
self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
```

`create_leak_free_split` is the only place `fit_transform` on train is used, and
test/val are always `transform`ed. `BenchmarkSuite.run_benchmark` does
`pipeline.fit(X_train_raw)` then `pipeline.transform(...)` on both partitions
(`benchmark_suite.py:175-177`).

**Note on `Transform` after `fit`:** the pipeline raises
`RuntimeError("Pipeline must be fitted on training data before transforming.")`
if `transform` is called before `fit` — a defensive guard against the most likely
leakage bug.

---

## 4. Stage 3 — QuantumFeatureSelector (register-size reduction)

`QuantumFeatureSelector` (`data/feature_selector.py:40-203`) maps an
$M$-dimensional feature vector to $K = \min(n_\text{qubits}, M)$ dimensions.

### 4.1 Methods

| `method` | Model | Notes |
|---|---|---|
| `pca` | `sklearn.decomposition.PCA(n_components=K, random_state=42)` | default; linear, deterministic |
| `kpca` | `sklearn.decomposition.KernelPCA(n_components=K, kernel="rbf", random_state=42)` | nonlinear |
| `mutual_info` | `sklearn.feature_selection.mutual_info_classif` | **supervised**: requires `y_train`, else `ValueError`; keeps `selected_indices` for covariance derivation |
| `autoencoder` | `LatentAutoencoder` (PyTorch) | see §4.3 |

An unrecognised method falls through both `if` chains and then hits
`z_train = self.autoencoder.encoder(...)` — effectively an error path. The lab
restricts this via `REDUCTION_METHODS = ("none", "pca", "mutual_info", "autoencoder")`.

### 4.2 Why reduce at all

| Without reduction | With reduction |
|---|---|
| 27 VOC columns → 27 qubits | 27 columns → 6 qubits |
| $2^{27}$ ≈ 134 M amplitudes | $2^6$ = 64 amplitudes |
| 27-qubit BioZZ depth: $2(2+351)=706$ | 6-qubit depth: $2(2+15)=94$ |
| 351 CNOT pairs × 2 reps = 702 CNOTs | 30 CNOTs |

The reduction is therefore not a nicety; it is what makes the circuit NISQ-shaped
at all. See [`14_nisq_feasibility.md`](14_nisq_feasibility.md).

### 4.3 Latent autoencoder

`LatentAutoencoder` (`feature_selector.py:15-37`):

```
Encoder:  Linear(M→32) → BatchNorm1d(32) → LeakyReLU(0.2) → Linear(32→K) → Tanh
Decoder:  Linear(K→32) → BatchNorm1d(32) → LeakyReLU(0.2) → Linear(32→M)
```

`Tanh` on the latent bounds the code to $[-1,1]$, which is then re-scaled again
in `transform`. Trained by `_train_autoencoder` (`:109-122`): Adam `lr=0.01`,
`MSELoss`, `epochs=40` **full-batch** (no `DataLoader`, no minibatching).

Caveats: `BatchNorm1d` in eval mode uses running statistics, so `transform` is
consistent with `fit`; but the full-batch, fixed-epoch, unseeded training means
the AE is the least reproducible component in the pipeline (its result depends on
the ambient torch seed, which `set_seed()` does set).

### 4.4 Fit / transform split

`fit(X_train, y_train=None)` (`:62-107`) fits the model and **captures the
latent bounds**:

```python
self.latent_min = np.min(z_train, axis=0)
self.latent_max = np.max(z_train, axis=0)
```

`transform(X)` (`:125-159`) never re-derives bounds:

```python
span     = self.latent_max - self.latent_min
safe_span = np.where(span > 0, span, 1.0)     # avoid /0 when min == max
z_scaled = (z - self.latent_min) / safe_span
z_scaled = np.clip(z_scaled, 0.0, 1.0)        # train-range clamping
low, high = self.angle_range
return low + z_scaled * (high - low)          # default angle_range = (-π, π)
```

Three deliberate design choices:

1. **Bounds from train only** — a leak-free operation by construction.
2. **Clip to [0,1] before de-scaling** — a test sample outside the training
   latent range is clamped, not extrapolated. Extrapolation into $[-\pi,\pi]$
   would wrap angles and silently create a *different* sample.
3. **`safe_span = 1.0` for constant columns** — prevents NaN when a component is
   identically zero (can happen with `mutual_info` on a low-variance feature).

`inverse_transform(z)` (`:165-192`) undoes the affine map back to latent space
and then inverts PCA or the AE decoder. Supported only for `pca` and
`autoencoder`; other methods raise `NotImplementedError`. This is what enables
biomarker attribution back into raw compound space
([`09_explainability.md`](09_explainability.md)).

Tested: `tests/test_data_pipeline.py` asserts the selector is bounded in
$[-\pi,\pi]$ and that `inverse_transform` returns the original dimensionality.

---

## 5. Stage 4 — The covariance matrix (the most instructive function in the repo)

`compute_qubit_covariance(X_scaled, selector, top_k=3)`
(`evaluation/benchmark_suite.py:100-153`) produces the $K \times K$ matrix that
weights BioZZ's entangling phases.

### 5.1 The bug this function exists to fix

The original implementation computed `np.corrcoef(X_pca_scores.T)`. PCA scores
are **orthogonal by construction**, so that matrix is the identity up to float
noise. Identity covariance ⇒ every off-diagonal weight is 1 ⇒ BioZZ collapses to
plain Pauli-ZZ, and the project's headline feature map was **silently inert** on
the exact data it was designed for. This is recorded as finding #1 (High) in
`docs/reports/code_evaluation_report.md:20`.

### 5.2 The three branches

```python
n_q = min(selector.n_qubits, X_scaled.shape[1])
```

**(a) Raw-subset selectors (`mutual_info`)** — direct Pearson correlation of the
selected raw columns:

```python
sel  = np.asarray(selector.selected_indices)[:n_q]
corr = np.corrcoef(X_scaled[:, sel].T)
return np.nan_to_num(corr[:len(sel), :len(sel)], nan=0.0)
```

**(b) PCA** — loadings are used to proxy which raw compounds drive each
component, and the component-pair weight is the **maximum absolute raw-feature
correlation between the two components' top-|loading| features**:

```python
loadings = np.abs(selector.pca_model.components_)          # (K, M)
raw_corr  = np.nan_to_num(np.corrcoef(X_scaled.T), nan=0.0) # (M, M)
cov = np.eye(n_q)
top_feats = [np.argsort(loadings[i])[::-1][:top_k] for i in range(n_q)]
for i, j in pairs:
    set_i, set_j = set(top_feats[i]), set(top_feats[j])
    a = sorted(set_i - set_j); b = sorted(set_j - set_i)    # DISJOINT ONLY
    if a.size and b.size:
        cov[i, j] = cov[j, i] = np.abs(raw_corr[np.ix_(a, b)]).max()
    else:                                                   # fully overlapping
        union = sorted(set_i | set_j)
        block = np.abs(raw_corr[np.ix_(union, union)])
        iu = np.triu_indices(len(union), k=1)
        cov[i, j] = cov[j, i] = block[iu].mean() if iu[0].size else 0.5
```

The **disjointness** constraint is load-bearing: if the top-loading sets share a
feature, the cross-block would include that feature against *itself*, whose
correlation is 1, and the weight would be pinned to 1 — reintroducing exactly the
degeneracy the function is meant to remove. Falls back to the mean |corr| over
the union, then to 0.5.

**(c) Fallback** — `np.eye(n_q)` when the selector is neither a raw-subset method
nor has a fitted PCA (i.e. `none` reduction, or an AE).

**Important:** when reduction is `"none"`, `HybridPipeline` calls
`compute_qubit_covariance(X_scaled, self.selector)` **only if a selector exists**
(`lab/pipeline.py:247`); otherwise it uses `np.eye(X_reduced.shape[1])`. This
means the classical control arms (`Raw-XGBoost`, `PCA-XGBoost` with
`embedding="none"`) never get a covariance matrix at all — they have no feature
map, so this is correct.

### 5.3 Properties of the output

| Property | Guaranteed | Why |
|---|---|---|
| Symmetric | yes | written as `cov[i,j] = cov[j,i]` |
| Unit diagonal | yes | initialised `np.eye`, only off-diagonals overwritten |
| Values in $[-1, 1]$ | yes | derived from \|Pearson r\| ≤ 1 |
| NaN-free | yes | `np.nan_to_num(..., nan=0.0)` |
| Derived from train only | yes | callers pass `X_tr_scaled` |

`tests/test_lab_pipeline.py` asserts permuted-covariance differs from the train
covariance, has unit diagonal, and is symmetric.

### 5.4 The negative control — permuted correlations

`cwzz_permuted` is a first-class embedding. `_permute_correlation(cov, seed)`
(`lab/pipeline.py:55-66`):

```python
rng = np.random.default_rng(seed)
iu  = np.triu_indices(n, k=1)
upper = np.asarray(cov)[iu].copy()
rng.shuffle(upper)                          # destroy the *assignment* of
mat = np.zeros((n, n)); mat[iu] = upper     # correlations to qubit pairs
mat[(iu[1], iu[0])] = upper                 # while keeping the multiset
np.fill_diagonal(mat, 1.0)
```

This preserves the correlation *distribution* exactly while breaking the
*structure*. If BioZZ's advantage comes from correlation-aware weighting, the
permuted arm should be worse. If the permuted arm matches or beats the
flagship, the advantage is noise. **On the current dataset it scores 0.689 vs
the flagship's 0.658** — i.e. the negative control passed the negative test, and
the project reports that
([`15_results_and_findings.md`](15_results_and_findings.md) §4.2).

---

## 6. DataFrames vs arrays

Two entry points coexist, and the distinction matters for code review:

- `BenchmarkSuite.run_benchmark` accepts `pd.DataFrame | np.ndarray` and
  immediately converts; `X_test_raw` is a DataFrame so feature names survive.
- `HybridPipeline.fit` does `X.values if isinstance(X, pd.DataFrame) else
  np.asarray(X)` and then makes a **writable copy** — `DataFrame.values` can
  return a read-only view, and in-place NaN imputation would then raise.
- `run_modality_ablation` slices feature groups with `X.iloc[:, idx]`, which
  requires a DataFrame; `lab/cv.py::_slice` handles both
  (`X.iloc[idx]` vs `np.asarray(X)[idx]`).

---

## 7. Reproducibility

`qmlkit.config.set_seed(seed)` (`config.py:14-21`):

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
```

Called by `scripts/run_benchmark.py:80` and `scripts/run_hybrid_search.py:44`.
In addition, most components carry their own fixed internal seed that does *not*
depend on the global one:

| Component | Internal seed | Location |
|---|---|---|
| `QuantumFeatureSelector` PCA/KPCA/MI | `random_state=42` | `feature_selector.py:68,71,76` |
| `QSVMClassifier` SVC | `random_state=42` | `qsvm.py:108` |
| VQC weight init | `default_rng(42)` | `vqc.py:86-89` |
| VQC batch shuffle | `default_rng(42)` | `vqc.py:103` |
| QCNN weight init + shuffle | `default_rng(42)` | `qcnn.py:108` |
| QRC Hamiltonian | `default_rng(random_state)` | `qrc.py:27` |
| `SimulatedKennelSource` | `default_rng(42)` | `kennel_streaming.py:150` |
| `QuantumKernelSHAP` background | `default_rng(42)` | `quantum_shap.py` |
| `LabRun` job ids | `uuid4().hex[:12]` | `kennel_server.py:458` |

Consequence: **changing `--seed` does not change the quantum circuits' internal
randomness.** It changes the split, the classical models, and the CV
partitioning. Worth stating explicitly in any comparative claim.
