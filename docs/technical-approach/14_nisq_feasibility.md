# 14 — NISQ Feasibility

`src/qmlkit/evaluation/hardware_profiler.py` (151 lines). **This document exists
because "is this real quantum computing?" is the first question a technical
reviewer will ask.** The answer must be precise and honest.

---

## 1. The honest answer in one paragraph

The circuits in this project are **genuine quantum circuits** — real
`CNOT`/`RZ`/`Hadamard`/`IsingXX` gates composed by PennyLane, with a real
statevector of dimension $2^K$. They are **executed on an exact classical
statevector simulator** (`qml.device("default.qubit")`), not on a QPU. The
profiler's job is to show that the *circuits themselves* are small enough to be
executed on near-term hardware: at 6 qubits and 2 reps, the flagship feature map
is **94 gates deep with 60 two-qubit CNOTs and 0 trainable parameters** — which is
small for a 6-qubit device but not small enough to run without error mitigation.
The project's quantum contribution is therefore a **faithful simulation-based
NISQ feasibility study**, and its scientific claim is about the *modelling
approach* (correlation-weighted encoding + hybrid fusion), not about quantum
hardware advantage.

---

## 2. `CircuitProfile`

```python
@dataclass
class CircuitProfile:
    n_qubits: int
    circuit_depth: int
    total_gates: int
    single_qubit_gates: int
    two_qubit_cnot_gates: int
    parameter_count: int
    nisq_compatibility_verdict: str
```

---

## 3. `profile_feature_map` — analytic gate accounting

`hardware_profiler.py:35-85`. The docstring states the methodology:

> Gate counts are derived analytically from the exact circuit structures used by
> `qmlkit.quantum.feature_maps` and the VQC ansätze, so profiles match what is
> actually executed.

**No circuit is built or simulated** — it is arithmetic from the known gate
structure. That makes it O(1) and exact-for-this-architecture, but it is
architecture-level accounting, not a transpiler output: it does not model
hardware connectivity constraints, gate synthesis (e.g. a native `RZZ` instead of
`CNOT-RZ-CNOT`), or routing overhead.

```python
is_zz_family = any(t in map_lower for t in ("zz", "cov", "bio"))

if is_zz_family:
    pairs      = n * (n - 1) // 2
    hadamards  = n * reps
    rz_single  = n * reps
    cnots      = 2 * pairs * reps
    rz_pair    = pairs * reps
    depth      = reps * (2 + 3 * pairs)      # H layer + RZ layer + serialised CNOT-RZ-CNOT
else:
    hadamards = rz_single = cnots = rz_pair = 0
    depth     = n * reps                     # RY only

single_qubit = hadamards + rz_single + rz_pair + ry_rotations
total_gates  = single_qubit + cnots
verdict = "EXCELLENT - Fully Executable on Near-Term NISQ QPUs" if depth <= 60 \
          else "MODERATE - Requires Error Mitigation"
```

The depth model `reps × (2 + 3·pairs)` treats each `CNOT–RZ–CNOT` block as 3
serialised gates and assumes **no parallelism between pairs** — a conservative
(upper-bound) model, since disjoint qubit pairs could in principle execute
concurrently on hardware with all-to-all connectivity.

### 3.1 ZZ-family scaling table ($r = 2$)

| $K$ | $P = \binom{K}{2}$ | **depth** | CNOTs | 1-qubit gates | total gates | verdict |
|---|---|---|---|---|---|---|
| 4 | 6 | 40 | 24 | 36 | 60 | EXCELLENT |
| 5 | 10 | 64 | 40 | 50 | 90 | MODERATE |
| **6** | **15** | **94** | **60** | **66** | **126** | **MODERATE** |
| 8 | 28 | 176 | 112 | 124 | 236 | MODERATE |
| 12 | 66 | 404 | 264 | 348 | 668 | MODERATE |
| 24 | 276 | 1664 | 1104 | 1152 | 2256 | MODERATE |

**Verified against the committed artifact**
`outputs/lab/20260823_082122/configs/CWZZ-QSVM__flagship_.json`:
`n_qubits: 6, circuit_depth: 94, total_gates: 126, single_qubit_gates: 66,
two_qubit_cnot_gates: 60, parameter_count: 0,
nisq_compatibility_verdict: "MODERATE - Requires Error Mitigation"`.

### 3.2 The `depth ≤ 60` threshold

An internally-chosen cutoff, not derived from a specific device. It is calibrated
roughly to IBM's 5-qubit `ibm_brisbane`-class devices (typical 2-qubit gate
depth budget before error correction becomes necessary for meaningful results is
100–300, so 94 is genuinely borderline). Stating the device the threshold targets
would be an improvement.

### 3.3 `parameter_count` is always 0 for feature maps

`profile_feature_map` accepts `trainable_params=0` but **`describe_model` never
passes it** (`hardware_profiler.py:122-124`), so the feature-map block always
reports `parameter_count: 0`. This is *correct* — a data-encoding circuit has no
trainable parameters — but it is worth stating explicitly, because "0 trainable
parameters" invites the question "so what is being learned?" The answer is: the
*head* (SVC support vectors / VQC angles / XGBoost trees), never the encoding.

---

## 4. `profile_variational_ansatz` — the trainable block

`hardware_profiler.py:87-112`.

| Branch | parameters | rotations | CNOTs | depth |
|---|---|---|---|---|
| `StronglyEntangling` | $n_L \cdot K \cdot 3$ | $n_L \cdot K$ | $n_L \cdot K$ (ring ladder) | $n_L \cdot 3$ |
| RealAmplitudes-style | $n_L \cdot K$ | $n_L \cdot K$ | $n_L \cdot (K-1)$ | $n_L \cdot 2$ |

For the lab's VQC default ($K{=}6$, $n_L{=}2$): **36 parameters, 12 rotations,
12 CNOTs, ansatz depth 6.**

### 4.1 A discrepancy in the CNOT count

PennyLane's `StronglyEntanglingLayers` with default `ranges='Circular'` emits
`n_wires` CNOTs per layer ⇒ $n_L \cdot K = 12$ for the defaults. The profiler
matches. But note that with a full 6-qubit data-encoding layer of 30 CNOTs in
front, the ansatz contributes only 12 more — **the encoding dominates the two-qubit
budget by 2.5×**. Optimising `reps` (1 instead of 2) halves the encoding CNOTs
and is the single highest-leverage hardware knob, at the cost of expressivity.

### 4.2 Hardcoded profiles in `training_service.py`

For the VQC/QCNN branches the API does **not** call the profiler — it hardcodes
the profile numbers (`training_service.py:123-208`):

| `model_type` | depth | total_gates | cnots | verdict |
|---|---|---|---|---|
| `VQC_StronglyEntangled` | **24** | `n_qubits*6*2` | `n_qubits*2` | "OPTIMIZED - Native Parameterized Quantum Ansatz" |
| `VQC_RealAmplitudes` | **14** | `n_qubits*4` | `(n_qubits-1)*2` | "LOW LATENCY - Real Amplitudes NISQ Circuit" |
| `QCNN` | **18** | 36 | 12 | "HIERARCHICAL - Barren Plateau Immune QCNN" |

These are **ansatz-only** numbers that ignore the feature map entirely, and they
contradict the profiler's own arithmetic for the same circuits. So the Training
Studio's "NISQ Quantum Hardware Compilation Profile" panel shows depth 24 for a
VQC whose actual total depth is $94 + 6 = 100$. Only the QSVM branches
(`QSVM_BioZZ`, `QSVM_PauliZZ`) and the two hybrid branches use the live profiler.
This is a presentation-critical inconsistency — see
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §1.

---

## 5. `evaluate_shot_noise_robustness`

`hardware_profiler.py:130-151`.

```python
shots_list = shot_levels or [500, 1000, 4000, 8192]
k_exact = kernel_engine.compute_kernel_matrix(X_sample[:5], X_sample[:5])
for shots in shots_list:
    noise_sigma = np.sqrt(0.25 / shots)
    noise_matrix = np.random.default_rng(42).normal(0, noise_sigma, size=k_exact.shape)
    noise_matrix = (noise_matrix + noise_matrix.T) / 2.0      # symmetric
    np.fill_diagonal(noise_matrix, 0.0)                       # diagonal unaffected
    k_noisy = np.clip(k_exact + noise_matrix, 0.0, 1.0)
    robustness_scores[f"shots_{shots}_mae"] = round(mean(|k_exact - k_noisy|), 4)
```

**The $\sigma$ model.** $\sigma = \sqrt{0.25/\text{shots}}$ comes from the
binomial standard deviation of a measured frequency, $\sigma_p = \sqrt{p(1-p)/N}$,
maximised at $p = 0.5$ (hence $0.25$). For $N=500$, $\sigma \approx 0.0224$;
for $N=8192$, $\sigma \approx 0.0055$.

**This is an analytic surrogate, not a shot simulation.** It adds Gaussian noise
to the *already-computed exact* kernel matrix. It does **not** sample
computational basis states from the statevector, does not apply measurement
bit-flip channels, and does not model readout error. The three symmetrisation
steps (symmetric, zeroed diagonal, clipped to $[0,1]$) keep the perturbed matrix
a plausible kernel, but the absolute MAE numbers should be described as
"kernel-entry perturbation sensitivity", not "hardware fidelity".

Also note: the same seed 42 creates a **new** `default_rng(42)` per shot level,
so the noise matrices are not nested — increasing shots does not monotonically
reduce the perturbation of a fixed realisation. For a monotonic sweep the RNG
should be created once outside the loop.

**No committed results.** This function is not called from any runner, CLI, or
test. To make the NISQ claim concrete, a real shot-sampling experiment
(`qml.sample_statevector` / `qml.qnode` with a shot-based device) at
$N \in \{500, 1000, 4000, 8192\}$ should be run and its output committed.

---

## 6. Realistic hardware requirements

What would be needed to actually run the flagship configuration.

### 6.1 6-qubit CW-ZZ QSVM

| Resource | Requirement | Notes |
|---|---|---|
| Qubits | 6 | any 6-qubit device; needs all-to-all or a routing-aware transpiler |
| Circuit depth | 94 (upper bound) | actual hardware depth would be lower with pair parallelism |
| 2-qubit gates | 60 CNOTs | IBM Eagle/Heron: 2Q error ~0.5–1% ⇒ ~30–60% chance of a clean run without mitigation |
| Measurements | 1 expectation per kernel entry | 36 Gram entries × 96 train = 3,456 kernel evaluations for a full fit |
| Practical cost | $10^5$–$10^6$ shots for a usable Gram matrix | 96² = 9,216 kernel entries; at 4,096 shots each ≈ 38 M shots |
| Error mitigation needed | ZNE / PEC / measurement error mitigation | the profiler's own verdict string |
| Estimated queue time | hours to days | dominates the experiment wall time |

**The dominant cost is the fidelity kernel.** $O(N^2)$ circuit executions with
$O(10^3)$–$O(10^4)$ shots each is the reason quantum kernel methods have not
displaced classical kernels in practice, and it is worth stating this plainly.

### 6.2 6-qubit VQC

| Resource | Requirement |
|---|---|
| Qubits | 6 |
| Total depth | 94 (encoding) + 6 (ansatz) ≈ 100 |
| Trainable parameters | 36 + 1 bias |
| Iterations | epochs × n_batches × batch_size QNode evaluations; 8 epochs × 3 batches × 32 = 768 forward + 768 backward |
| Parameter-shift cost | 2 × 37 = 74 circuit evals per parameter per sample (hence `backprop` on the simulator) |
| Hardware gradient | parameter-shift or adjoint differentiation; the current implementation optimises the *simulator*, so running on hardware requires re-plumbing the optimizer |

That last point is important and often glossed over: the VQC is trained with
`qml.AdamOptimizer` on **differentiable simulator statevectors**. To train on a
QPU you need a shot-based device and either parameter-shift gradients (2P circuit
evals per parameter — a 74× cost multiplier) or the adjoint method. The
architecture supports it, but the training loop would need to change.

---

## 7. Claims that are supported

| Claim | Evidence | Status |
|---|---|---|
| The feature map is a genuine multi-qubit circuit with real two-qubit gates | 60 CNOTs analytically; statevector dimension 64 | ✅ supported |
| The circuit is small enough to be NISQ-relevant at 6 qubits | depth 94, 126 gates, 0 trainable params in the encoder | ✅ supported, with the "MODERATE" caveat |
| Quantum representation + classical head beats both pure arms | 0.974 (hybrid) vs 0.910 (classical control) vs 0.658 (pure quantum), 5-fold CV | ✅ supported on this dataset |
| Cost of quantum is 2–3 orders of magnitude higher | 15.4 s vs 0.16 s (hybrid vs control); 343.5 s vs 0.13 s (VQC vs XGBoost) | ✅ supported |
| Quantum models underperform strong classical baselines on real VOC data | 0.507 / 0.625 vs 0.972–1.000 | ✅ supported and honestly reported |
| Correlation-weighting of the feature map improves on plain ZZ | 0.658 vs 0.557 | ⚠️ **not** supported — the permuted control scored 0.689 |
| The circuits run on real NISQ hardware | — | ❌ **not claimed, not tested** |
| Shot-noise robustness | analytic surrogate only | ⚠️ surrogate, no committed results |

---

## 8. Recommended next steps for a hardware claim

1. **Run on a real device** at $K=4$ (depth 40, verdict "EXCELLENT") to get one
   end-to-end datapoint. 4 qubits is within reach of current hardware and the
   `mutual_info` selector gives genuinely interpretable compound-to-qubit
   assignments.
2. **Replace the analytic shot-noise surrogate** with real shot sampling
   (`qml.device("default.qubit", shots=N)`), and commit the output.
3. **Sweep `reps ∈ {1, 2}`** — `reps=1` halves CNOTs to 30 and depth to 47, which
   crosses the profiler's own "EXCELLENT" threshold at 6 qubits. This is a
   one-line change with a large hardware-feasibility payoff.
4. **Cache train statevectors** in `QuantumKernel` so inference is $O(N_{test})$
   statevector evaluations rather than $O(N_{test} \cdot N_{train})$ — the
   single largest practical optimisation available and a prerequisite for any
   hardware run.
5. **Use `n_qubits=4` for the flagship** and report the $K$ sweep as a scaling
   study rather than claiming parity at $K=6$.

---

## 9. Where the profiler output appears

| Consumer | Line | Behaviour |
|---|---|---|
| `HybridPipeline._describe()` | `lab/pipeline.py:214-228` | live profiler; attaches `record["spec"]` |
| `run_hybrid_search` | `lab/experiments.py:81-84` | best-effort `circuit_profile` per config, `try/except` |
| `training_service.py` | `:83-299` | live for `QSVM_BioZZ` / `QSVM_PauliZZ` / hybrids; **hardcoded for the VQC/QCNN branches** |
| `TrainingStudio.tsx` | `components/TrainingStudio.tsx` | renders the "NISQ Quantum Hardware Compilation Profile" panel (qubits, depth, CNOTs, total gates, verdict chip) |
| Committed artifacts | `outputs/lab/*/configs/*.json` | one profile per configuration, with per-fold metrics |

Because `_evaluate_config` wraps `describe()` in `try/except`, a profile can be
**absent** from a committed config JSON with no marker. Reviewers should check
that every `configs/*.json` contains a `circuit_profile` key.
