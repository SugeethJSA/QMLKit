# 05 — Quantum Models

`src/qmlkit/quantum/` — five models over one shared encoding. All are
PennyLane `QNode`s on `default.qubit` (exact statevector, no shot noise,
differentiable by backprop).

| Class | File | Type | Trained params | In benchmark? |
|---|---|---|---|---|
| `QuantumKernel` | `qsvm.py` | not a classifier | 0 | via QSVM |
| `QSVMClassifier` | `qsvm.py` | kernel method | SVC support vectors | ✅ |
| `VariationalQuantumClassifier` | `vqc.py` | PQC | $3 n_L K$ + 1 bias | ✅ |
| `QuantumConvolutionalClassifier` | `qcnn.py` | PQC (hierarchical) | 14 fixed | ✅ (lab) |
| `QuantumReservoirClassifier` | `qrc.py` | reservoir + ridge readout | 0 (Hamiltonian random) | ❌ unwired |
| `QuantumKernelFeatureTransformer` | `kernel_features.py` | feature transform | 0 | ✅ (hybrid) |

---

## 1. `QuantumKernel` — exact fidelity kernel

`qsvm.py:16-72`.

```python
@qml.qnode(self.device)                       # device = qml.device("default.qubit", wires=K)
def state_circuit(x):
    self.feature_map.apply(x, wires=range(self.n_qubits))
    return qml.state()                         # |Phi(x)> ∈ C^{2^K}
```

### 1.1 Kernel

$$K(x,x')=\left|\langle \Phi(x)\mid\Phi(x')\rangle\right|^{2}$$

Implemented as a vectorised Hilbert projection rather than a statevector
outer-product loop (`compute_kernel_matrix`, `:50-72`):

```python
states1 = self.get_statevectors(X1)                       # (N1, 2^K) complex
inner   = np.dot(states1, states2.conj().T)               # (N1, N2)
K       = np.abs(inner) ** 2
```

Symmetric path (`X2 is None or X2 is X1`) adds numerical hygiene (`:64-67`):

```python
np.fill_diagonal(K, 1.0)        # exact unit diagonal
K = np.clip(K, 0.0, 1.0)        # fidelity is in [0,1]
return (K + K.T) / 2.0          # enforce exact symmetry
```

The fidelity kernel (rather than the "fidelity" $\langle\Phi(x)|\Phi(x')\rangle$
linear overlap) is required for the Gram matrix to be a **positive semi-definite
kernel** — i.e. to admit a RKHS and therefore a valid SVM. Using the bare overlap
would produce a Gram matrix that is generally indefinite and SVC's convexity
guarantee would be void.

### 1.2 Cost

$N$ samples ⇒ $N$ circuit evaluations per kernel-matrix column ⇒ $O(N^2)$
simulations. At $K=6$ each `qml.state()` is cheap but not free; the practical
limit on a laptop is $N \approx 200$–400 in reasonable time. This is the reason
for `balanced_subsample` and for the `--max-samples 120` cap in the reported
runs.

### 1.3 Properties asserted in tests

`tests/test_quantum_circuits.py`:
- symmetry within `atol=1e-6`
- unit diagonal
- all entries in $[0,1]$

---

## 2. `QSVMClassifier`

`qsvm.py:75-159`. A faithful scikit-learn-shaped wrapper: `fit`, `predict`,
`predict_proba`, `decision_function`, `score`, `save`, `load`.

```python
def __init__(self, n_qubits=6, feature_map_type="BioZZ", c_param=1.0,
             covariance_matrix=None):
    self.feature_map  = get_feature_map(feature_map_type, n_qubits, covariance_matrix)
    self.kernel_engine = QuantumKernel(feature_map=self.feature_map, n_qubits=n_qubits)
    self.X_train_cached = None; self.train_kernel_matrix = None; self.is_fitted = False
```

`c_param` is **not** passed to the `QuantumKernel`; it is used only by the SVC.

### 2.1 Adaptive calibration

`_make_calibrated_svc` (`qsvm.py:103-113`):

```python
counts   = np.bincount(np.asarray(y_train).astype(int))
min_class = int(counts[counts > 0].min()) if counts.size else 0
base = SVC(C=c_param, kernel="precomputed", random_state=42)
if len(counts) == 2 and min_class >= 2:
    cv = int(max(2, min(5, min_class)))
    return CalibratedClassifierCV(base, ensemble=False, cv=cv)
return base        # degenerate class balance → plain SVC
```

Two things to note for review:

1. **`CalibratedClassifierCV` instead of `SVC(probability=True)`.** The latter is
   deprecated from scikit-learn 1.9; this is the migration. `ensemble=False` keeps
   it a single held-out calibrated model rather than an ensemble of $cv$ models.
2. **Fold count derived from the minority class size.** With 4 positives, `cv=4`
   — a 5-fold split would produce a fold with zero positives and raise. The
   `ensemble=False` calibration therefore also *shrinks* the effective training
   set: the SVC is fit on $(cv-1)/cv$ of the data and calibrated on the rest,
   then (with `ensemble=False`) it is the model that is returned. This is
   correct but slightly under-trains; it is a deliberate robustness-over-fidelity
   trade.

### 2.2 Fit / predict

```python
def fit(self, X_train, y_train):
    self.X_train_cached = np.copy(X_train)                              # needed for predict
    self.train_kernel_matrix = self.kernel_engine.compute_kernel_matrix(X_train)
    self.svc_model = self._make_calibrated_svc(self.c_param, y_train)
    self.svc_model.fit(self.train_kernel_matrix, y_train)
    self.is_fitted = True
    return self

def predict(self, X_test):
    if not self.is_fitted or self.X_train_cached is None:
        raise RuntimeError("Model must be fitted before predicting.")
    K_test = self.kernel_engine.compute_kernel_matrix(X_test, self.X_train_cached)
    return self.svc_model.predict(K_test)
```

The kernel is recomputed against the cached training states for every prediction
call — **no train-state caching in `QuantumKernel`**, so inference is $O(N_{test}
\cdot N_{train})$ circuit simulations per call. This dominates the reported
"inference time per sample" figures. `predict_proba` and `decision_function` each
recompute the same kernel matrix independently (three separate `compute_kernel_matrix`
calls in `benchmark_suite.py:198-199` for preds + probs).

### 2.3 Measured performance

Real Lung VOC (n=120, 6 qubits, 8 VQC epochs, seed 42 — full table in
[`15_results_and_findings.md`](15_results_and_findings.md)):

| Model | ROC-AUC | Accuracy | Sensitivity | Specificity | Train s |
|---|---|---|---|---|---|
| QSVM BioZZ | **0.507** | 0.542 | 0.417 | 0.667 | 0.74 |
| VQC StronglyEntangled | **0.625** | 0.667 | 0.500 | 0.833 | 343.5 |
| XGBoost (classical) | 1.000 | 1.000 | 1.000 | 1.000 | 0.13 |

QSVM at 0.507 is chance. Root-cause analysis in
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §2:
6-qubit PCA compression of 27 compounds discards most of the variance that
carries the signal, and the fidelity kernel over 96 training points has almost
no room to separate.

---

## 3. `VariationalQuantumClassifier` (VQC)

`vqc.py:15-172`.

### 3.1 Circuit

```python
@qml.qnode(self.device, interface="autograd", diff_method="backprop")
def vqc_qnode(inputs, weights):
    encoded = inputs if inputs.ndim == 1 else pnp.transpose(inputs)   # batch → feature-major
    self.feature_map.apply(encoded, wires=range(self.n_qubits))       # 1. encoding
    if self.ansatz_type == "StronglyEntangling":
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
    else:                                                             # RealAmplitudes
        for layer in range(self.n_layers):
            for i in range(self.n_qubits): qml.RY(weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1):  qml.CNOT(wires=[i, i+1])
            if self.n_qubits > 2:               qml.CNOT(wires=[self.n_qubits-1, 0])
    return qml.expval(qml.PauliZ(0))                                   # 2. readout
```

Weight shapes (`_build_circuit`, `:51-84`):

- `StronglyEntangling`: `qml.StronglyEntanglingLayers.shape(n_layers, n_wires)`
  ⇒ $(n_L, K, 3)$ ⇒ $3 n_L K$ parameters.
- `RealAmplitudes`: $(n_L, K)$ ⇒ $n_L K$ parameters.

Plus one trainable scalar bias added to the expectation value.

Default readout is $\langle Z_0\rangle$ only — a single-qubit measurement, which
is the cheapest possible expectation and avoids a multi-qubit observable
requiring basis rotations.

### 3.2 Optimisation

```python
opt        = qml.AdamOptimizer(stepsize=self.learning_rate)   # 0.03
weights_var = pnp.array(self.weights, requires_grad=True)
bias_var    = pnp.array(self.bias, requires_grad=True)
batch_rng   = np.random.default_rng(42)

def cost_fn(w, b, x_batch, y_batch):
    preds = pnp.stack([self._qnode(x, w) + b for x in x_batch])
    return pnp.mean((preds - y_batch) ** 2)                    # margin MSE
```

- **Loss:** mean squared error against **signed labels**
  `y_signed = np.where(y_train == 0, -1.0, 1.0)` (`:94`) — matches the range of
  $\langle Z_0\rangle \in [-1,1]$.
- **`diff_method="backprop"`** — gradients flow analytically through the
  statevector simulator. The alternative, parameter-shift, needs $2P$ circuit
  evaluations per parameter per sample. For $3\cdot2\cdot6 = 36$ parameters that
  is a 72× multiplier; the source comment (`:107-108`) records the switch and
  the measured ~5× end-to-end speedup.
- **Mini-batching:** `batch_size = min(32, len(X_train))`,
  `n_batches = ceil(len(X_train)/batch_size)` (`:112-113`). `ceil` is deliberate —
  the original `//` silently dropped up to `batch-1` samples per epoch
  (`docs/reports/code_evaluation_report.md:24`).
- **Seeded shuffle** per epoch: `indices = batch_rng.permutation(len(X_train))`.

### 3.3 Inference and probability

```python
def predict_raw(self, X):   return np.array([self._qnode(x, self.weights) + self.bias for x in X])
def predict_proba(self, X):
    raw = self.predict_raw(X)
    prob_pos = 1.0 / (1.0 + np.exp(-2.0 * raw))      # sigmoid(2·raw) → [0.1, 0.9] for raw ∈ [-1,1]
    return np.vstack([1.0 - prob_pos, prob_pos]).T
def predict(self, X):        return np.where(self.predict_raw(X) >= 0.0, 1, 0)
```

Note the logistic map is `sigmoid(2·raw)`, so with the bias unconstrained the
model **cannot reach a calibrated 0.0 or 1.0** — reported probabilities are
bounded in roughly $[0.1, 0.9]$ when $|{\rm bias}|$ is small. This is visible in
the reported VQC Brier score of 0.245, which is high partly because probabilities
are never confident.

### 3.4 Cost

At 6 qubits / 2 layers / 8 epochs on 96 training samples, the reported wall time
is **343.5 s** — roughly 2,600× the XGBoost fit (0.13 s) and 2,000× the QSVM
kernel fit (0.74 s). The minibatch loop evaluates each sample's QNode
individually inside `cost_fn`; a batched `@qml.qnode` would be substantially
faster but is not used.

---

## 4. `QuantumConvolutionalClassifier` (QCNN)

`qcnn.py:14-165`. A 2-layer quantum convolution + pooling network with a
**fixed parameter count of 14**.

### 4.1 Blocks

```python
def _conv_block(self, params, wires):                       # 5 params
    qml.RY(params[0], wires[0]);  qml.RY(params[1], wires[1])
    qml.CNOT([wires[0], wires[1]]);  qml.RZ(params[2], wires[1])
    qml.RY(params[3], wires[0]);  qml.RY(params[4], wires[1])
    qml.CNOT([wires[1], wires[0]])

def _pool_block(self, params, source_wire, sink_wire):      # 2 params
    qml.CRZ(params[0], wires=[source_wire, sink_wire])
    qml.PauliX(wires=source_wire)
    qml.CRX(params[1], wires=[source_wire, sink_wire])
```

Parameter budget (`_build_circuit`, `:62-65`): `5 (Conv1) + 2 (Pool1) + 5 (Conv2)
+ 2 (Pool2) = 14`. The same `weights[0:5]` slice is reused across *all* adjacent
pairs in layer 1 — it is a **convolution in the strict sense** (shared weights),
not a per-pair parameter set. That is what keeps the count at 14 regardless of
qubit number.

### 4.2 Circuit topology

```
1. encoding (BioZZ)
2. Conv layer 1: for i in range(0, K, 2):  conv([i, (i+1) % K])     # even starts
                 for i in range(1, K-1, 2): conv([i, (i+1) % K])   # odd starts
3. Pool layer 1: for i in range(0, K, 2):  pool(source=(i+1)%K, sink=i)
                 active_qubits_layer1 = [0, 2, 4, ...]             # K → ceil(K/2)
4. Conv layer 2: for each adjacent pair in active_qubits_layer1
5. Pool layer 2: pool(source=active[1], sink=active[0])
6. readout: qml.expval(qml.PauliZ(final_sink))
```

The original pool loop indexed `i+1` without a modulus and crashed on odd
`n_qubits`; the fix is the modular wire index and the explicit
`if n_qubits < 2: raise ValueError(...)` guard (`qcnn.py:26-27`,
`docs/reports/code_evaluation_report.md:25`).

### 4.3 Training

Same structure as the VQC: `y_signed ∈ {-1,1}`, MSE loss, `AdamOptimizer(0.03)`,
`batch_size = min(32, N)`, `ceil` batching, `default_rng(42)` for init and
shuffling, `diff_method="backprop"`, sigmoid(2·raw) probabilities.

### 4.4 Constraint and measured result

`training_service.py:181-188` **rebases the register** for QCNN:

```python
actual_qubits = 8 if req.n_qubits >= 8 else 4
# a separate selector + covariance matrix are fitted for actual_qubits
```

because the two-conv/two-pool topology needs ≥4 qubits to be meaningful. The lab
uses `n_qubits=max(2, spec.n_components)` (`lab/pipeline.py:101`).

Measured: **ROC-AUC 0.558 ± 0.109** on real VOC 5-fold CV — the weakest arm,
and 13.4 s mean training time.

---

## 5. `QuantumReservoirClassifier` (QRC)

`qrc.py:12-117`. The only model in the package that consumes **raw time
series** rather than a feature vector.

### 5.1 Fixed Hamiltonian

```python
self.rng = np.random.default_rng(random_state)            # 42
self.J = self.rng.uniform(-1, 1, size=(K, K))
self.J = (self.J + self.J.T) * coupling_strength          # 0.5
self.h = self.rng.uniform(-1, 1, size=K)
```

A random transverse-field Ising model. **Never trained** — the reservoir's
parameters are fixed at construction.

### 5.2 Reservoir step

```python
@qml.qnode(self.device)
def reservoir_step(inputs, prev_angles):
    for i in range(K): qml.RZ(prev_angles[i], wires=i)         # 1. state recurrence
    for i in range(K): qml.RX(inputs[i % len(inputs)], wires=i)  # 2. input injection
    for i in range(K):                                          # 3. Ising evolution
        qml.RZ(self.h[i], wires=i)
        for j in range(i+1, K): qml.IsingXX(self.J[i, j], wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(K)]       # 4. multi-qubit readout
```

`prev_angles = expvals * np.pi` closes the recurrence — the previous step's
$Z$-expectations are fed back as $R_Z$ rotations. This is the reservoir's
memory mechanism.

### 5.3 Feature extraction over a series

`transform_series(time_series_tensor)` — input shape
`(N_samples, N_sensors, N_timesteps)`:

```python
sample_step = max(1, timesteps // 15)      # sub-sample to ~15 steps
prev_angles = np.zeros(K)
trajectory = []
for t in range(0, timesteps, sample_step):
    expvals = np.array(self._reservoir_step(series[idx, :, t], prev_angles))
    prev_angles = expvals * np.pi
    trajectory.append(expvals)
traj_mat = np.array(trajectory)             # (T_steps, K)
reservoir_features[idx] = concat([mean(traj, 0), max(traj, 0), traj[-1]])   # 3K features
```

Output `(N, 3K)` = mean-state, max-state, and final-state summaries. Readout is
`RidgeClassifier(alpha=1.0)`.

### 5.4 Status

**Unwired.** It is exported from `qmlkit.quantum.__init__` but is not in
`BenchmarkSuite`, not in `default_presets`, not in `training_service`, and has
no test. Recorded as finding #13 (Info) in
`docs/reports/code_evaluation_report.md:32`. It is also the natural consumer of
`BiomimeticVOCGenerator.raw_time_series` — the only component that uses the
`(N, 16, 100)` raw tensor that the generator produces and that the rest of the
pipeline discards after feature extraction.

---

## 6. `QuantumKernelFeatureTransformer`

`kernel_features.py:13-207`. The bridge into the hybrid arms: turns samples into
a **fixed-length quantum-kernel similarity vector** suitable for any
scikit-learn estimator.

### 6.1 Algorithm

```python
def fit(self, X, y=None):
    indices = self._select_landmarks(X, y)                   # class-balanced sample
    self.landmarks_        = np.copy(X[indices])
    self.landmark_states_  = self.kernel_engine.get_statevectors(self.landmarks_)
    return self

def transform(self, X):
    states         = self.kernel_engine.get_statevectors(X)          # (N, 2^K)
    inner_products = np.dot(states, self.landmark_states_.conj().T)   # (N, L)
    return np.clip(np.abs(inner_products) ** 2, 0.0, 1.0)            # (N, L)
```

So each sample is represented by its fidelity similarity to $L$ landmark
samples: a "quantum landmark kernel" representation in $\mathbb{R}^{L}$.

### 6.2 Landmark selection

`_select_landmarks` (`:82-137`) is **stratified and seeded**:

- `count = min(n_landmarks, N)`; `rng = np.random.default_rng(self.seed)`.
- With ≥2 classes, indices are split into per-class pools, each shuffled, then
  drained round-robin until `count` is reached — guarantees both classes
  contribute landmarks.
- With 1 class or `y is None`, uniform sampling without replacement.

### 6.3 Validation

`_check_X` (`:63-80`) raises on: non-2D input, zero rows, `X.shape[1] !=
n_qubits`, and any non-finite value. The constructor additionally validates
`covariance_matrix.shape == (n_qubits, n_qubits)` and `n_qubits >= 1`,
`n_landmarks >= 1`.

### 6.4 Where it is used

`HybridPipeline` head `quantum_kernel_xgb` (`lab/pipeline.py:259-293`):

```python
quantum_features = self.quantum_transformer.fit_transform(X_reduced, y)   # (N, L)
X_head = np.hstack([X_reduced, quantum_features])                          # (N, K + L)
self.head = _make_head(dataclasses.replace(spec, head="xgboost"), cov_matrix=None)
```

With `n_components=6, n_landmarks=12` the head sees **18** features: the 6
reduced angles plus 12 quantum similarities. Tested in
`tests/test_lab_pipeline.py` for `landmarks_.shape == (6,3)` and the fused
dimension `(5,8)` on a 3-qubit / 5-landmark toy case.

At inference, `transform` is called against the **stored** landmark states — no
landmark re-selection, no test-set information. Leak-free.

---

## 7. Comparison table

| | QSVM | VQC | QCNN | QRC | QKFT |
|---|---|---|---|---|---|
| Input | feature vector | feature vector | feature vector | **time series** | feature vector |
| Trained quantum params | 0 | $3 n_L K$+1 | 14 | 0 (random fixed) | 0 |
| Readout | calibrated SVC | $\langle Z_0\rangle$ + bias | $\langle Z_{\rm sink}\rangle$ + bias | `RidgeClassifier` | downstream estimator |
| Differentiation | n/a | autograd backprop | autograd backprop | n/a | n/a |
| Probabilities | `CalibratedClassifierCV` | `sigmoid(2·raw)` | `sigmoid(2·raw)` | none | downstream |
| Fidelity kernel | ✅ | ❌ | ❌ | ❌ | ✅ |
| Complexity driver | $O(N^2)$ circuit sims | epochs × N sims | epochs × N sims | T × N sims | $O((N+L))$ sims |
| Real-VOC ROC-AUC | 0.507 | 0.625 | 0.558 | not measured | via `Quantum_Kernel_XGB` |
| Lab 5-fold AUC (mean±sd) | 0.658 ± 0.156 | 0.576 ± 0.151 (AE-VQC) | 0.558 ± 0.109 | — | 0.910 (kennel synth) |
| Train time (real VOC) | 0.74 s | 343.5 s | 13.4 s | — | 15.4 s (augmented) |
