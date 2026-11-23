# 17 — Presentation Guide

Two artefacts: a **slide deck** (18 slides, 30–45 min) and a **live demo script**
(8 min). Speaker notes are written as talk-track, not prose.

---

## 0. The one-sentence thesis

> Canine-biomimetic VOC sensing plus hybrid quantum machine learning: we built the
> full pipeline, we measured it honestly, and **the honest result is that hybrid
> quantum+classical beats both pure arms, while pure quantum does not beat
> classical on this data** — and the correlation-weighted encoding, our own
> headline idea, did not survive its negative control.

Lead with that. It buys credibility for everything after it.

---

## 1. Deck outline (18 slides)

### Slide 1 — Title

**QMLKit: Hybrid Quantum Machine Learning for Canine-Olfactory Early Disease Screening**
Problem Statement 26139 · Hybrid Quantum ML Platform for Early Disease Detection

*Subtitle line:* ESP32 kennel telemetry · PennyLane quantum encoding · FastAPI +
Next.js console · MIT licensed · v0.1.4

> **Note:** do **not** put a Qiskit logo on this slide. There is no Qiskit in the
> codebase. If the deck template has one, remove it (risk register D2/X11).

---

### Slide 2 — The problem (2 min)

Three constraints that make this hard:

1. **Biological signal, not an image.** Cancer VOC signatures are
   cross-reactive: one sensor responds to many compounds, one compound triggers
   many sensors. There is no clean feature.
2. **Few samples, high dimension.** 427 patients, 27 compounds — and the target
   is imbalanced (193 control / 157 cancer / 77 benign).
3. **Screening, not diagnosis.** Sensitivity and specificity are the operating
   points that matter; a 100%-accuracy always-positive model is worthless.

**Body:** dogs detect cancer by smell at sensitivities above most instruments.
The gap is instrumentation and quantification. We built the instrument; the
question is what to do with its output.

---

### Slide 3 — System architecture (2 min)

```
ESP32 kennel node ──TCP JSON :3333 (10 Hz)──► FastAPI ingest ──► 45 window features
   4 FSR · 6 IR · 2 US                        (400-frame window)  │
   · collar MPU6050 · MAX30102                                    ▼
        ▲                                            RandomForest (LOSO) + EMA
        │                                                    │
        │                                            Next.js Console ◄── diagnostics WS
        │
   ── parallel offline path ──
VOC dataset (427 × 27) ──► scaler ──► PCA(6) ──► BioZZ feature map ──► QSVM/VQC/Hybrid ──► 10 metrics
```

Three subsystems, one feature contract. Emphasise: **the same 45-feature
extractor runs on a recorded CSV and on a live TCP stream** — that is what
prevents train/serve skew.

---

### Slide 4 — Data (2 min)

| Dataset | Size | Classes |
|---|---|---|
| **Lung VOC (real, bundled)** | 427 × 27 | Control 193 · Cancer 157 · Benign 77 |
| → `cancer_vs_control` | 350 × 27 | 193 / 157 |
| Kennel recordings | 3 files, 1 dog | cancer 1 · calm 2 |
| Synthetic VOC cohort | on demand | 6 cancer types |
| Canine ECG (secondary) | 1,123 recordings, 40 dogs | annotation only — no audio, no labels |

**Say the caveat out loud:** "The VOC file's precision changes mid-table —
patients 1–88 at 2–3 decimals, later rows full float. That pattern is consistent
with a simulated augmentation appended to a real cohort. We preserve the data and
flag it; absolute numbers carry that caveat."

That sentence is worth 30 seconds of credibility.

---

### Slide 5 — The preprocessing chain (2 min)

```
raw (27)  →  NaN impute (train medians)
          →  StandardScaler        (fit on train only)
          →  PCA(6)                 (fit on train only)
          →  min–max → [-π, π]     (bounds from train only)
          →  correlation matrix ρ  (train only, from RAW features)
```

**The point of this slide:** every stage is fitted on the training fold and
reused unchanged at transform time. No test-fold statistic touches the encoding.

**The reduction is not optional:** 27 qubits ⇒ depth 706, 702 CNOTs. 6 qubits ⇒
depth 94, 60 CNOTs. The reduction is what makes the circuit NISQ-shaped at all.

---

### Slide 6 — The BioZZ feature map (3 min) — *the core idea*

Per repetition:

```
H  on all qubits            → superposition
RZ(2·x_i)  on all qubits    → linear encoding
for each pair (i,j):
    CNOT(i→j)
    RZ( 2 · ρ_ij · (π−x_i)(π−x_j) )   ← THE CONTRIBUTION
    CNOT(i→j)
```

$$\phi_{ij} = 2\,\rho_{ij}\,(\pi - x_i)(\pi - x_j)$$

**One factor of difference from a standard Pauli-ZZ map**, and it is a *data*
factor. $\rho_{ij}$ is the empirical correlation between the compounds driving
qubits $i$ and $j$.

**What that buys, mechanically:** $\rho\approx1$ ⇒ strong ZZ interaction;
$\rho\approx0$ ⇒ the CNOT–RZ–CNOT block is locally a product of rotations, i.e.
no effective entangler. If VOC compounds cluster by chemical family, the
covariance matrix is block-diagonal and BioZZ builds a *chemically-motivated
connectivity graph* for free.

**What it costs:** nothing. Same gate count, same depth, same parameter count.
The correlation information is folded into the *values* of the RZ angles.

**The three bug stories that make this credible** (pick one, ~40 s):

> "The first version computed the correlation matrix as `corrcoef` of the *PCA
> scores*. PCA scores are orthogonal by construction, so that matrix was the
> identity — which silently reduced our headline feature map to a plain ZZ map,
> on exactly the data it was designed for. The fix derives correlations from the
> *raw* features; for PCA it proxies component-pair correlation via the maximum
> raw correlation between their top-3 loading features, using disjoint feature
> sets so a feature is never correlated with itself."

That is a real, high-severity, found-and-fixed defect. Showing it is stronger
than hiding it.

---

### Slide 7 — Quantum models (2 min)

| Model | Circuit | Trained params | Notes |
|---|---|---|---|
| **QSVM** | BioZZ → fidelity kernel $K=\|\langle\Phi(x)\|\Phi(x')\rangle\|^2$ → precomputed-kernel SVC | 0 in the encoder | exact statevector; $O(N^2)$ sims |
| **VQC** | BioZZ + `StronglyEntanglingLayers(2)` | 36 + 1 bias | Adam 0.03, backprop diff, margin-MSE on $\langle Z_0\rangle$ |
| **QCNN** | 2 conv + 2 pool blocks | **14** (shared weights) | weakest arm |
| **QRC** | fixed random Ising Hamiltonian + ridge readout | 0 | implemented, not wired in |

Two implementation choices worth naming:

- **Fidelity kernel, not bare overlap.** $|\langle\Phi(x)|\Phi(x')\rangle|^2$ is
  a positive semi-definite kernel, so it admits an RKHS and SVC's convexity
  guarantee holds. The bare overlap generally does not.
- **Backprop, not parameter-shift.** On a differentiable simulator, backprop
  avoids the $2P$ circuit-evaluation multiplier — 36 parameters would otherwise
  mean 72 circuit evaluations per sample per step. Measured ~5× end-to-end.

---

### Slide 8 — The hybrid thesis (2 min)

Two ways to fuse quantum and classical:

| Head | Mechanism | Head sees |
|---|---|---|
| `quantum_kernel_xgb` | 12 landmark fidelity similarities **concatenated** with the 6 reduced angles | 18 features |
| `quantum_augmented_xgb` | **out-of-fold** VQC probability as one extra column beside the full 27 scaled features | 28 features |

**The OOF point is the most defensible thing in the project:**

> "If you use a VQC's probability on the rows it was trained on as an XGBoost
> feature, that feature is a near-perfect training label. XGBoost leans on it, the
> training score saturates at 100%, and the held-out score collapses. We caught
> that, and now every augmented probability comes from a 3-fold
> out-of-fold VQC — the feature a row sees was predicted by a model that never
> saw that row."

---

### Slide 9 — Baselines and metrics (1 min)

Six classical baselines, fitted under the **identical** protocol:
SVM-RBF (C=1.5, calibrated), SVM-Linear, LogReg, RandomForest (150 × depth 8),
XGBoost (100 × depth 4, lr 0.08), MLP (64-32, α=0.001).

Ten clinical metrics — the point of which is that **accuracy alone is
insufficient for screening**: sensitivity, specificity, PPV, NPV, macro/weighted
F1, balanced accuracy, ROC-AUC, Brier.

Note the **deprecation fix**: `SVC(probability=True)` is deprecated from
scikit-learn 1.9; we use `CalibratedClassifierCV`, with the internal fold count
derived from the minority class size so small folds cannot produce a
zero-positive split.

---

### Slide 10 — Experiment protocol (2 min)

Four invariants that make the comparison fair:

1. **Identical stratified partitions across every configuration** — the splitter
   runs on `np.zeros(len(y))`, so partitions are a pure function of $y$ and seed.
   A score difference is attributable to configuration, not fold luck.
2. **A fresh pipeline per fold** — scaler, selector, angle bounds, correlation
   matrix, and landmark states all rebuilt inside the fold.
3. **Correlation matrix from the training fold only.**
4. **Seed fixed at 42 throughout**, with per-fold metrics committed to JSON.

**Shipped ablations:** feature-map ablation (RQ3, with a negative control),
modality ablation (RQ2, 1 + 2G configurations), noise robustness (train clean,
evaluate degraded). Every run writes `run.json` + `leaderboard.csv` +
per-config JSON with **per-fold metrics and a circuit profile**.

---

### Slide 11 — Result 1: real-data benchmark (2 min)

Real Lung VOC, 120 balanced, stratified 80/20 ⇒ **96 train / 24 test**, 6 qubits.

| Model | Paradigm | ROC-AUC | Brier | Train s |
|---|---|---:|---:|---:|
| XGBoost | Classical | **1.000** | 0.010 | 0.13 |
| Random Forest | Classical | **1.000** | 0.041 | 0.13 |
| SVM Linear | Classical | 0.979 | 0.061 | 0.00 |
| SVM RBF | Classical | 0.972 | 0.074 | 0.00 |
| VQC StronglyEntangled | Quantum | **0.625** | 0.245 | 343.5 |
| QSVM BioZZ | Quantum | **0.507** | 0.250 | 0.74 |

**Read it honestly, out loud:**

> "The real dataset is highly separable — a *linear* SVM reaches 0.979, which
> means the signal is close to a hyperplane in concentration space. That is
> precisely the regime where a fidelity kernel has nothing to add. The quantum
> arms are at or near chance, and the VQC costs 2,600× the XGBoost fit. This is
> the result, and it is the result we report."

Also state the caveat: 24 test samples ⇒ a perfect 24/24 has a Wilson 95% CI of
about [0.86, 1.00]. The 1.000s are "at ceiling".

---

### Slide 12 — Result 2: the hybrid leaderboard (3 min)

5-fold CV, identical partitions, 11 curated configurations + 2 ensembles.

| Rank | Config | ROC-AUC | Train s |
|---|---|---|---|
| 1 | **Stacked-LR {RF, XGB, QSVM, VQC}** | **0.986 ± 0.018** | 135.7 |
| 2 | **Quantum-Augmented-XGB** | **0.974 ± 0.023** | 15.4 |
| 3 | SoftVote {SVM-RBF, RF, XGB, QSVM} | 0.913 ± 0.050 | 1.1 |
| 4 | PCA-XGBoost (classical control) | 0.910 ± 0.053 | 0.16 |
| 5 | Raw-XGBoost (classical control) | 0.887 ± 0.076 | 0.12 |
| 8 | CWZZ-QSVM (pure quantum, flagship) | 0.658 ± 0.156 | 2.55 |

**The finding, stated carefully:**

> "Hybrid beats both pure arms. Quantum-Augmented-XGB reaches 0.974 against a
> classical control's 0.910 and a pure-quantum 0.658. The quantum
> representation earns its keep — but **through** a strong classical learner, at
> roughly 100× the training cost. And note the fold-to-fold standard deviations:
> 0.023 and 0.053. The gap and the spread are the same order of magnitude, so I
> am claiming a consistent improvement across folds under an identical protocol,
> not statistical significance. The per-fold numbers are committed, so a paired
> test is a short piece of analysis away and we have not yet run it."

That last sentence is the mark of a careful researcher. Use it.

---

### Slide 13 — Result 3: the negative control (3 min) — *the honesty slide*

Feature-map ablation. Identical gate budget, identical head, identical partitions,
**one variable changed**.

| Variant | Circuit | ROC-AUC |
|---|---|---|
| Angle | $K\cdot r$ RY, no entanglement | 0.557 |
| Pauli-ZZ | conventional 2nd-order | 0.557 |
| **CW-ZZ (ours)** | covariance-weighted | **0.658** |
| **CW-ZZ permuted (control)** | **same correlations, shuffled assignment** | **0.689** |

**The presentation of a negative result:**

> "We built a feature map whose entire contribution is weighting the entangling
> phases by inter-compound correlation. To test it, we built a negative control:
> take the same correlation matrix, shuffle which correlation goes to which qubit
> pair, and re-run. If the weighting matters chemically, the shuffled version must
> be worse.
>
> **It is not worse. It scored 0.689 against our flagship's 0.658.** And the
> flagship has the largest fold-to-fold variance of any configuration we ran,
> ±0.156.
>
> So the +0.101 over plain ZZ appears to come from *any* non-trivial phase
> structure, not from correlation semantics. We are not claiming correlation
> weighting helps on this dataset. We built the control specifically so that this
> was a falsifiable claim, and it did not survive. Re-running at larger $N$ is the
> next step."

**This is the strongest slide in the deck.** A project that ships and reports a
negative control against its own headline idea is doing science, not marketing.

---

### Slide 14 — NISQ feasibility (2 min)

| Resource | 6 qubits, 2 reps |
|---|---|
| Circuit depth | **94** (upper bound) |
| Total gates | 126 |
| Two-qubit CNOTs | **60** |
| Trainable params in the encoder | 0 |
| Profiler verdict | "MODERATE — Requires Error Mitigation" |

Scaling: $K{=}4$ ⇒ depth 40 (**"EXCELLENT"** by our own threshold); $K{=}8$ ⇒ 176;
$K{=}12$ ⇒ 404. Depth grows **quadratically** in pairs, so the reduction stage —
not the model — sets the qubit budget.

**Be precise about what this is:**

> "These are genuine multi-qubit circuits with real CNOTs. They are executed on
> an exact statevector simulator, not on a QPU. The profiler's job is to show the
> circuits are *small enough to be relevant*: 6 qubits and 60 CNOTs is a
> near-term device scale. The claim is about the modelling approach, not about
> hardware advantage."

**And name the real cost driver honestly:**

> "The dominant cost is the fidelity kernel: $O(N^2)$ circuit executions, each
> needing $O(10^3)$–$O(10^4)$ shots on hardware. At $N=96$ that is roughly 9,200
> kernel entries. This is exactly why quantum kernel methods have not displaced
> classical kernels in practice, and we would rather say so than not."

Optionally mention the two highest-leverage hardware changes: **`reps=1` halves
CNOTs 60→30 and depth 94→47, crossing our own "EXCELLENT" threshold**, and
caching train statevectors turns $O(N_{test}\cdot N_{train})$ inference into
$O(N_{test})$.

---

### Slide 15 — Explainability (1 min)

Two layers:

1. **Quantum-kernel SHAP** — `shap.KernelExplainer` over the reduced features.
   Weighted-linear-regression Shapley values; attribution on the positive-class
   probability so values sum to $f(x) - \mathbb{E}[f]$. Gated behind
   `?deep_explain=true` because it costs ~3,000 QNode sweeps; the default path is
   a mean-delta approximation over angle space.
2. **Latent → chemical back-projection** — the PCA back-projection is *exact*;
   per-compound attribution then pools by a three-tier strategy (direct name
   match → the synthetic 64-sensor schema → contiguous positional pooling) and
   aggregates into four biochemical pathways.

**State the current limitation plainly:** on the real dataset the compound
*names* attached to the attributions come from a positional mapping, because the
columns are molecular formulas rather than compound names. The *magnitudes* are
computed from the real model; the *labels* need a curated formula→compound
dictionary. That is the honest current state, and it is on the risk register.

---

### Slide 16 — Hardware and realtime (2 min)

**ESP32-WROOM-32, 13 sensors:**

| Group | Count | Pins | Rate |
|---|---|---|---|
| FSR load cells | 4 | 34–39 (ADC1, input-only, Wi-Fi-safe) | 20 Hz |
| IR proximity | 6 | 19,18,17,16,5,23 (active LOW) | 20 Hz |
| HC-SR04 | 2 | 26/27, 14/13 | 20 Hz |
| MPU6050 IMU | 1 | I²C 21/22 @ 400 kHz | 100 Hz |
| MAX30102 | 1 | same I²C | 2 Hz |
| **Telemetry frame** | — | TCP :3333, NDJSON, mDNS `kennel.local` | **10 Hz** |

**6-state machine:** BOOT → CALIBRATE → IDLE → OCCUPIED → SNIFF (8 s) →
COOLDOWN (10 s) → IDLE. Occupancy is a **fusion** of load and distance
($\Delta$ADC > 400 *and* < 60 cm), so neither a passing tail nor an object on the
mat can trigger a session.

**45 features** in four groups: pressure 13 · proximity 11 · motion 18 ·
physiology 3. Plus 9 baseline deltas and a recovery ratio ⇒ **55 trial features**.

**Realtime path:** producer thread → 4,000-frame ring buffer (oldest-drop) →
single consumer → every 10th frame to the UI over WebSocket, every 400th frame
to the RandomForest → EMA(α=0.3) → two-sided confidence gate at 0.6.

**And the honest hardware status:**

> "The kennel model is a RandomForest trained with leave-one-dog-out validation —
> LOSO rather than a random split, because a dog's posture and physiology are
> individual traits and a random split would leak subject identity. We currently
> have recordings for **one** dog, so LOSO has no usable fold and the live
> diagnostics correctly report `untrained`. The console renders a 'Model Awaiting
> Calibration' state pointing at the Data Lab, which is the designed first-run
> experience, not a failure."

---

### Slide 17 — Engineering and delivery (1.5 min)

- **Backend:** FastAPI + uvicorn, **two disjoint apps** (live kennel service with
  thread/queue lifecycle; screening service with in-process model fit and a
  Bootstrap portal).
- **Console:** Next.js 16 static export + React 19 + Tailwind v4, five pages
  (Dashboard, Diagnostics, Training Studio, Data Lab, Hybrid Lab), **zero
  state-management library** — the WebSocket and the polled REST state endpoint
  are the only data paths.
- **Packaging:** pnpm workspace; PyInstaller **onedir** bundle; Inno Setup `.exe`,
  macOS `.dmg`, Linux `.deb` + self-installing tarball; CI publishes a GitHub
  Release when `version.properties` is bumped.
- **Tests:** **59** across 8 files, including a FastAPI `TestClient` end-to-end
  path and the real bundled dataset load.
- **Reproduction:** every published number has a committed artifact directory with
  per-fold metrics.

**Two engineering items to raise proactively** (a reviewer will find them):

> "Two things you should know. First, there is currently no CI job that runs tests
> or lint on a pull request — CI triggers only on a version bump. That is a real
> gap we have on the fix list. Second, the release build installs the core
> dependency set only, so the packaged app falls back from XGBoost to
> scikit-learn's gradient booster; the released app's leaderboard is therefore not
> identical to source mode. We are fixing both."

Naming your own gaps is the fastest way to a clean review.

---

### Slide 18 — Conclusions and next steps (1.5 min)

**What we established**

1. A complete, leak-free, reproducible hybrid QML screening pipeline — three
   subsystems, one feature contract, 59 tests, packaged for three platforms.
2. **Hybrid wins.** Quantum representation + strong classical learner
   (0.974) beats both a classical control (0.910) and pure quantum (0.658) under
   an identical 5-fold protocol.
3. **Pure quantum does not beat classical here** (0.507 / 0.625 vs 0.972–1.000)
   — and the classical baselines are strong enough that the comparison is
   meaningful.
4. **Our own headline idea did not survive its negative control** (permuted
   correlations 0.689 ≥ flagship 0.658). Reported, not buried.
5. The circuits are genuinely NISQ-shaped (6 qubits, depth 94, 60 CNOTs) but
   executed on a simulator, and quantum inference is 100–2,600× more expensive.

**Next steps, in priority order**

| Priority | Action | Why |
|---|---|---|
| 1 | Derive the sample interval from frame timestamps | Fixes the 100 Hz ↔ 10 Hz mismatch; two frequency features are currently unreachable on live data |
| 2 | Cache train statevectors in `QuantumKernel` | $O(N_{test}\cdot N_{train}) \to O(N_{test})$; prerequisite for any hardware run |
| 3 | Paired significance test over the committed per-fold metrics | Converts the headline from "suggestive" to "demonstrated" — the data is already committed |
| 4 | Add CI test + lint jobs on pull requests | Nothing validates a PR today |
| 5 | `reps=1` sweep | Halves CNOTs and depth, crossing our own "EXCELLENT" threshold |
| 6 | Re-run the ablation at $N \geq 1000$ | The only way to settle the correlation-weighting question |
| 7 | Collect ≥2 dogs per label; train the kennel model | Unblocks LOSO and the live diagnostic path |
| 8 | One real hardware execution at $K=4$ | Turns the feasibility study into a feasibility *result* |

**Closing line**

> "The platform is real, the measurements are real, and the two findings that
> matter are both uncomfortable: hybrid is the way quantum adds value here, and
> our correlation-weighted encoding has not earned its name. Both are things we
> found by building the controls that could disprove us — which is the part of the
> work we would most like to be judged on."

---

## 2. Live demo script (8 min)

### Pre-flight (5 min before)

```bash
# 1. hardware: confirm the board answers BEFORE starting the backend
#    (the auto probe is one-shot — see risk register M13)
Test-NetConnection -ComputerName <board-ip> -Port 3333

# 2. backend + console
pnpm dev

# 3. confirm simulation fallback works
curl -s http://localhost:8000/api/v1/kennel/state
#    expect: {"stream_source":"simulation", "connected":true, "model_status":"untrained", ...}
```

If the board is up, set `QMLKIT_KENNEL_STREAM_SOURCE=hardware` so the probe is
skipped entirely.

### Beat sheet

| Time | Action | Say |
|---|---|---|
| 0:00–0:45 | `/` Dashboard | "This is live telemetry — 10 Hz over TCP. The kennel diagram shows the four load cells; the corner intensity *is* the dog's weight distribution. The IMU waveforms scroll the collar accelerometer. Right now this is the simulator, because we have one dog recorded and no trained artifact." |
| 0:45–1:30 | Point at the state machine | "BOOT, CALIBRATE, IDLE, OCCUPIED, SNIFF, COOLDOWN. Occupancy needs load *and* distance to agree — a passing tail or a bag on the mat won't trigger a session." |
| 1:30–2:30 | `/diagnostics` | "The VOC screening view. The confidence ring is an EMA-smoothed probability with a two-sided confidence gate at 0.6." **If the chart shows the seeded placeholder series, say so** — "the first 60 points are placeholder; the server replays the last real diagnostic on connect." |
| 2:30–4:00 | `/train` → run **QSVM BioZZ**, 30/class, 6 qubits | "This is the full pipeline in the browser: synthesise a cohort, leak-free split, scale, PCA to 6, build the correlation matrix from raw features, encode with BioZZ, compute the fidelity kernel, fit a calibrated SVC on the precomputed Gram matrix. Ten metrics, a confusion matrix, and the NISQ gate profile." |
| 4:00–5:00 | **Run VQC StronglyEntangled** with 3 epochs | "Same data, same split, same encoding — now a trainable ansatz. Watch the loss trajectory drop. And watch the training time: seconds for the kernel, minutes for the VQC. That cost asymmetry is one of our actual results." |
| 5:00–5:45 | Switch to **Quantum_Augmented_XGB** | "The hybrid arm. The VQC is run out-of-fold, so the probability XGBoost sees for a row was predicted by a model that never saw that row. Then XGBoost gets the quantum scalar *plus* all 27 raw features." |
| 5:45–6:30 | NISQ profile panel | "Six qubits, 60 two-qubit gates, and the parameter count is zero — because the *encoder* has nothing to train. The learning all happens in the head." |
| 6:30–7:15 | `/collect` | "This is the loop that closes the hardware story. Pick a dog, pick a label, record. That CSV is exactly what `train_kennel_model.py` consumes — same 45-feature extractor, so there is no train/serve skew." |
| 7:15–8:00 | `/lab` (if pre-warmed) | "The experiment lab: eleven curated configurations plus two ensembles, five-fold cross-validation with identical partitions across all of them, and per-fold metrics committed to JSON. Note that the leaderboard includes a classical control and a permuted-correlation negative control." |

### Demo failure playbook

| Failure | Recovery |
|---|---|
| Backend not started / `auto` probe already fired | Restart the backend, or set `QMLKIT_KENNEL_STREAM_SOURCE=hardware` |
| `/lab` search takes >3 min | Pre-warm a run 10 min before; or present the committed `outputs/lab/20260823_082122/leaderboard.csv` |
| `/lab` robustness shows an error | Expected — known defect D1. Say so: "the robustness arm through the API reads a result key the runner doesn't return; the CLI path works and the artifacts are on disk" |
| VQC training exceeds the demo budget | Use 3 epochs and 20 samples/class; quote the full-budget number from Slide 12 |
| `xgboost` missing in the local env | The row relabels to `Gradient_Boosting`; say so rather than hiding it |

---

## 3. Anticipated questions

### On the quantum claim

**"Isn't this just a classical SVM with extra steps?"**
Partly, and we say so. On this dataset the classical baselines reach 0.972–1.000
and the quantum arms 0.507–0.625. The quantum kernel's advantage is in modelling
nonlinear structure, and this task is close to linearly separable — a linear SVM
gets 0.979. What the quantum arm buys is a *representation* (fidelity
similarities to landmark samples), and that representation helps a strong
classical learner (0.974 vs 0.910) rather than replacing it.

**"Did you run this on a real quantum computer?"**
No. Exact statevector simulation. What we show is that the circuits are small
enough to matter: 6 qubits, depth 94, 60 CNOTs, and a `reps=1` variant that would
cross a conservative NISQ threshold. The scientific claim is about the modelling
approach; the hardware execution is future work, and slide 18 lists it.

**"Your feature map's contribution is one multiplicative factor. Isn't that thin?"**
It is thin, and the ablation says so. The permuted-correlation control scored
0.689 against our flagship's 0.658 — we do not claim correlation weighting helps
on this dataset. The interesting part is what the factor *buys* structurally: a
block-diagonal covariance gives a block-diagonal effective entangler, i.e. a
chemically-motivated connectivity graph, at zero gate cost. Whether that
converts to accuracy is an empirical question and the answer here is "not yet".

**"Isn't 94 circuit depth fine? Why 'MODERATE'?"**
Our threshold is `depth ≤ 60`, chosen conservatively. At 4 qubits the same map is
depth 40 and crosses it. The threshold is internally-chosen rather than tied to a
specific device — we should state which device it targets, and we have that on the
fix list.

### On the data

**"How do you know the VOC file is real data?"**
We don't — and we flag it. The precision changes mid-table: patients ≈1–88 at
2–3 decimals, later rows at full float precision. That pattern is consistent with
a simulated augmentation appended to a real cohort. There are also
duplicate-looking rows. We preserve the data, we report the pattern, and we
qualify the absolute numbers.

**"350 samples and 27 features — isn't that overfitting territory?"**
Yes, which is exactly why the leak-free protocol and the capped sample count
matter more than usual. It is also why we do not claim statistical significance on
the hybrid gap: the fold-to-fold standard deviations are the same order as the
gap.

**"Why cap at 120 samples?"**
The fidelity kernel is $O(N^2)$ circuit simulations. 96 training points means
~9,200 kernel entries. The cap is a runtime bound, and we state it as such rather
than presenting the capped cohort as the full dataset.

### On the kennel subsystem

**"Has the firmware ever been flashed?"**
No. The sketch requires a `secrets.h` with WiFi credentials, which is gitignored
and absent, so it has not been compiled in this repository's history. The
simulation source, the protocol parser, the feature extractor, and the server
are implemented and tested against each other; the hardware path is
implemented but unexercised.

**"Why a RandomForest for the kennel and quantum models for the VOC data?"**
Different problems with different sample regimes. The VOC path has 427 labelled
patients and is the quantum contribution. The kennel path currently has
recordings for one dog, and a quantum model on $n=1$ dog would be indefensible. The
kennel design target is a RandomForest with LOSO validation — cheap, robust at
small $n$, and subject-leakage-aware — with the quantum path reserved for when a
multi-dog cohort exists.

**"There's a sampling-rate mismatch in your feature extractor."**
Yes, and it is the top item on our fix list. The features assume 100 Hz; the wire
rate is 10 Hz. `acc_band_tremor_4_8hz` is therefore unreachable on live data
because the FFT axis tops out at 5 Hz, and the jerk scale is 10× optimistic.
Training and serving both use 400 frames so they are self-consistent in frame
count, but the committed CSVs are 100 Hz, so there is a real distribution shift.
The fix is to derive the sample interval from the frame timestamps — about five
lines.

### On the engineering

**"How do you know there's no leakage?"**
Four mechanisms, and we can point at the code. (1) Every fitted object — scaler,
selector, angle bounds, correlation matrix, landmark states — is constructed
inside the training branch and reused unchanged at transform time.
(2) The CV splitter runs on `np.zeros(len(y))` so partitions are identical across
all configurations and a fresh pipeline is built per fold. (3) The VQC
augmentation is out-of-fold. (4) The kennel model uses leave-one-dog-out.
We also found and fixed the highest-severity leak in the project — training the
screening service's QSVM on the full cohort without a split.

**"What's your test coverage like?"**
59 tests across 8 files. The ones that matter most: the fidelity kernel's Gram
properties (symmetry, unit diagonal, values in $[0,1]$); the leak-free split;
PCA bounds in $[-\pi,\pi]$ with a working `inverse_transform`; the protocol
parser's rejection paths; feature-extraction determinism; a FastAPI `TestClient`
end-to-end path; and loading the real bundled dataset. Gaps we know about: the
quantum reservoir, the hardware profiler, the kernel-SHAP path, and the lab API
routes are untested.

**"Does it run on hardware?"**
The firmware is written; the desktop app is packaged for Windows, macOS and
Linux. Both have caveats we've documented: the firmware has not been compiled,
and the release build installs the core dependency set only, so the packaged app
falls back from XGBoost to scikit-learn's gradient booster.

### Questions to redirect

| Question | Redirect |
|---|---|
| "Can this diagnose cancer?" | "It is a screening research platform. Every UI surface carries a research-aid disclaimer, the dataset has a documented augmentation artifact, and nothing here is a clinical claim." |
| "How much would this cost to deploy?" | "Not a question we've scoped. The binding constraint is quantum inference cost, which is 100–2,600× the classical arms in our measurements." |
| "Can you add [model]?" | "The lab is declarative — a new head is one function in `_make_head` and one `PipelineSpec` entry. The CV harness, ablations, and registry pick it up automatically." |
| "What about [disease] X?" | "The generator and the screening service already parameterise 5 cancer types. The real dataset currently only has lung." |

---

## 4. Timing guide

| Duration | Slides | Cut |
|---|---|---|
| **15 min** | 1, 3, 6, 11, 12, 13, 18 | Drop 2, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17. Slides 12+13 must both survive — they are the finding |
| **30 min** | 1–13, 18 | Compress 7 and 9 to bullets; keep 6, 12, 13 intact |
| **45 min** | All 18 + demo | Full |
| **Workshop** | All 18 + demo + open `lab/pipeline.py` and `benchmark_suite.py:100-153` live | The covariance function and the OOF augmentation are the two best code-walkthrough targets |
