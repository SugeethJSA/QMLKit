# 04 — Quantum Feature Maps

`src/qmlkit/quantum/feature_maps.py` (134 lines) — the entire encoding
vocabulary. This is the file to read first for any technical review.

---

## 1. Contract

Every feature map is a **stateless PennyLane operation** that mutates the active
quantum state. It is not a QNode and not a device-bound object; the QNode that
wraps it owns the device.

```python
def <map>_feature_map(x: np.ndarray, wires: list[int] | range, ...) -> None
```

`x` is **feature-major**: `x[i]` drives wire `wires[i]`. All three maps assume
`len(x) >= len(wires)`; extra features are ignored. The VQC and QCNN QNodes
transpose batched `(batch, features)` inputs before calling `apply`
(`vqc.py:64`, `qcnn.py:71`).

---

## 2. `angle_feature_map` — first-order encoding

```python
for i, wire in enumerate(wires):
    qml.RY(x[i], wires=wire)
```

Product state, zero entanglement:

$$|\Phi(x)\rangle=\bigotimes_{i} \mathrm{RY}(x_i)|0\rangle$$

- **Depth:** 1 layer of single-qubit rotations.
- **Expressivity:** linear in $x$ only. No interactions. For a QSVM this yields a
  kernel that is a function of pairwise *distances* only.
- **Cost:** $K$ RY gates, 0 CNOTs.

Included as the *lower bound* of the ablation, not as a candidate.

---

## 3. `pauli_zz_feature_map` — conventional second-order ZZ

```python
for _ in range(reps):                       # default reps = 2
    for wire in wires:  qml.Hadamard(wires=wire)          # layer 1: superposition
    for i, wire in enumerate(wires): qml.RZ(2.0*x[i], wire)   # layer 2: linear phase
    for i in range(n):
        for j in range(i+1, n):
            phi = 2.0*(np.pi - x[i])*(np.pi - x[j])
            qml.CNOT([wires[i], wires[j]]); qml.RZ(phi, wires[j]); qml.CNOT([wires[i], wires[j]])
```

Per rep: $K$ H + $K$ RZ + $\binom{K}{2}$×(2 CNOT + 1 RZ).

- **Depth:** $\text{reps}\cdot(2 + 3\binom{K}{2})$
- **Gates:** $\text{reps}\cdot(2K + 3\binom{K}{2})$
- **Expressivity:** full second-order interactions, but *every* pair coupled
  identically — i.e. the interaction structure assumes all VOC compounds are
  equally co-varied, which is chemically false.

For $K=6$, reps=2: depth 94, 126 gates, 60 CNOTs.

---

## 4. `bio_zz_feature_map` — covariance-weighted ZZ (the headline)

```python
def bio_zz_feature_map(x, wires, covariance_matrix=None, reps=2):
    n = len(wires)
    cov = covariance_matrix if covariance_matrix is not None else np.eye(n)
    for _ in range(reps):
        for wire in wires: qml.Hadamard(wires=wire)
        for i, wire in enumerate(wires): qml.RZ(2.0*x[i], wires=wire)
        for i in range(n):
            for j in range(i+1, n):
                weight = float(cov[i, j]) if i < cov.shape[0] and j < cov.shape[1] else 0.5
                phi_ij = 2.0*weight*(np.pi - x[i])*(np.pi - x[j])
                qml.CNOT([wires[i], wires[j]]); qml.RZ(phi_ij, wires[j]); qml.CNOT([wires[i], wires[j]])
```

**The only difference from `pauli_zz_feature_map` is the `weight` factor.** That
is the entire contribution:

$$\phi_{ij} \;=\; 2\,\underbrace{\rho_{ij}}_{\text{empirical correlation}}\,
(\pi - x_i)(\pi - x_j)$$

where $\rho_{ij} = \texttt{cov}[i,j] \in [-1,1]$ is the absolute Pearson
correlation between the raw features that drive qubit $i$ and qubit $j$
(derived per [`03_preprocessing_and_reduction.md`](03_preprocessing_and_reduction.md) §5).

### 4.1 What the weighting buys, mechanically

| $\rho_{ij}$ | Effect |
|---|---|
| ≈ 1 | strong ZZ interaction — near-complete control exchange; the pair behaves almost classically |
| ≈ 0 | the CNOT–RZ–CNOT block is (locally) a product of rotations — no effective interaction |
| < 0 | sign-flipped interaction |

Consequences:

1. **Sparse correlation structure ⇒ sparse effective entanglers.** If VOC
   compounds cluster by chemical family, the covariance matrix is block-diagonal
   and BioZZ builds a circuit with strong within-family entanglers and weak
   cross-family ones — a chemically-motivated connectivity graph, obtained for
   free from the data.
2. **Identical gate sequence, different angles.** BioZZ is a *specialisation* of
   Pauli-ZZ, not a different circuit family. It is therefore directly
   comparable to the ZZ arm under an identical gate budget, which is exactly what
   the ablation in [`08_experiment_lab.md`](08_experiment_lab.md) does.
3. **No extra cost.** Same CNOT count, same depth, same parameter count. The
   correlation information is folded into the *values* of the RZ angles.

### 4.2 Honest limitations of the mechanism

- $\rho_{ij}$ is an **absolute** correlation; the sign is discarded. Two
  anti-correlated compounds are treated as two strongly co-varying compounds.
- $\rho$ is a *global, static* statistic. It does not adapt per sample, so a
  sample whose internal structure is atypical is still encoded with the
  population-average coupling.
- With PCA reduction, $\rho_{ij}$ is a *proxy*: the max |raw correlation| between
  the two components' top-3 loading features, not a direct measurement
  ([`03_preprocessing_and_reduction.md`](03_preprocessing_and_reduction.md) §5.2b).
- The map is deterministic given $(\rho, x)$; there is no learned parameter, so
  there is nothing to train in the encoding itself. All learning happens in the
  head (SVC weights, VQC angles, XGBoost trees).
- **The advantage is not yet demonstrated** on the bundled dataset: the permuted
  control scored 0.689 vs the flagship's 0.658 ([`15_results_and_findings.md`](15_results_and_findings.md) §4.2).

---

## 5. Class wrappers and the factory

```python
class BaseFeatureMap:                 # abstract: n_qubits, name, apply()
class AngleFeatureMap(BaseFeatureMap)     name = "Angle"
class PauliZZFeatureMap(BaseFeatureMap)   name = "ZZ"    (+ reps)
class BioZZFeatureMap(BaseFeatureMap)     name = "BioZZ"  (+ covariance_matrix, reps)
class CovarianceFeatureMap(BioZZFeatureMap)   # alias, no behaviour change
```

`BioZZFeatureMap.set_covariance_matrix(cov)` allows late injection of the
correlation matrix after construction.

### 5.1 `get_feature_map` dispatch

```python
def get_feature_map(map_type, n_qubits, covariance_matrix=None):
    t = map_type.lower()
    if "bio" in t or "cov" in t:  return BioZZFeatureMap(n_qubits, covariance_matrix)
    elif "zz" in t or "pauli" in t: return PauliZZFeatureMap(n_qubits)
    elif "angle" in t:            return AngleFeatureMap(n_qubits)
    else:                        return BioZZFeatureMap(n_qubits, covariance_matrix)   # default
```

Substring matching, so `"BioZZ"`, `"bio"`, `"Covariance"`, `"cov"` all resolve to
BioZZ. **The default branch is BioZZ**, so an unrecognised or empty name silently
gets the weighted map (with `covariance_matrix=None` ⇒ identity ⇒ plain ZZ).
`QuantumModelConfig.feature_map_type` in `config.py:57-59` constrains the
string to `{"BioZZ","ZZ","Angle","Covariance"}`, but the lab's `embedding` field
maps through a separate dict with a `"ZZ"` default
(`lab/pipeline.py:75-80, 91-96, 260-265`):

| `embedding` | `feature_map_type` |
|---|---|
| `angle` | `Angle` |
| `zz` | `ZZ` |
| `cwzz` | `BioZZ` |
| `cwzz_permuted` | `BioZZ` (with a permuted covariance) |
| `none` or anything else | `ZZ` |

---

## 6. Circuit gate accounting (verified against artifacts)

Let $K$ = qubits, $r$ = reps, $P = \binom{K}{2}$ = pairs.

| Quantity | ZZ family (BioZZ / PauliZZ) | Angle |
|---|---|---|
| Hadamards | $K r$ | 0 |
| Single-qubit RZ (linear) | $K r$ | 0 |
| RY | 0 | $K r$ |
| Pair RZ | $P r$ | 0 |
| CNOTs | $2 P r$ | 0 |
| **Depth** | $r(2 + 3P)$ | $K r$ |
| **1-qubit gates** | $2Kr + Pr$ | $K r$ |
| **Total gates** | $2Kr + 3Pr$ | $K r$ |

Worked example — $K=6$, $r=2$, $P=15$:

```
depth   = 2 * (2 + 45)          = 94
1q      = 12 + 12 + 30          = 66
CNOT    = 2 * 15 * 2            = 60
total   = 66 + 60               = 126
```

This matches the committed artifact
`outputs/lab/20260823_082122/configs/CWZZ-QSVM__flagship_.json` exactly, and is
what `QuantumHardwareProfiler.profile_feature_map` computes analytically
([`14_nisq_feasibility.md`](14_nisq_feasibility.md)). The `hardware_profiler`
detects the ZZ family with `any(t in map_lower for t in ("zz","cov","bio"))`
(`hardware_profiler.py:49-50`) — a substring test on the same naming convention
used by the factory.

### 6.1 Scaling in qubit count

| $K$ | $P$ | depth ($r{=}2$) | CNOTs | total gates |
|---|---|---|---|---|
| 4 | 6 | 40 | 24 | 60 |
| 6 | 15 | 94 | 60 | 126 |
| 8 | 28 | 176 | 112 | 238 |
| 12 | 66 | 404 | 264 | 564 |
| 24 | 276 | 1664 | 1104 | 2352 |

Quadratic growth in pairs ⇒ the reduction stage, not the model, is what sets the
qubit budget. This is the single most important scalability fact in the design.

---

## 7. Where each map is consumed

| Consumer | `feature_map_type` / `embedding` | Notes |
|---|---|---|
| `QSVMClassifier` (benchmark) | `BioZZ` + train covariance | `benchmark_suite.py:193` |
| `VariationalQuantumClassifier` (benchmark) | `BioZZ`, `n_layers=2` | `benchmark_suite.py:208-214` |
| `HybridPipeline` presets 1–9 | `cwzz` / `zz` / `angle` / `cwzz_permuted` | `lab/experiments.py:33-55` |
| `QuantumKernelFeatureTransformer` | `BioZZ` under `cwzz` | `lab/pipeline.py:267-273` |
| `QuantumConvolutionalClassifier` | `BioZZ` **hardcoded**, ignores `spec.embedding` | `lab/pipeline.py:100-105` |
| OOF fold augmenters in `quantum_augmented_xgb` | `BioZZ` if `embedding.startswith("cwzz")` else `ZZ` | `lab/pipeline.py:311` |
| `training_service.py` dispatch | `BioZZ` (QSVM/VQC), `ZZ` (QSVM_PauliZZ), `Angle` (VQC_RealAmplitudes), `BioZZ` (QCNN) | `training_service.py:83-208` |
| `QuantumReservoirClassifier` | **none** — uses a fixed random Ising Hamiltonian, no feature map | `qrc.py:38-60` |
| `server.py` default screening model | `BioZZ` | `server.py:112-115` |

The QCNN's hardcoded `BioZZ` means the `qcnn` preset cannot actually be run with
an alternative embedding; the ablation covers QSVM only.
