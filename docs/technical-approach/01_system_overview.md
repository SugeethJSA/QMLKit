# 01 — System Overview

## 1. What QMLKit is

QMLKit is an integrated, installable **screening research platform** for early
disease detection that combines canine-biomimetic olfactory sensing with hybrid
quantum machine learning. It is not a diagnostic device; every UI surface
carries a research-aid disclaimer.

It has three subsystems that share one feature contract:

| Subsystem | Location | Responsibility |
|---|---|---|
| **Kennel node firmware** | `firmware/kennel_node` | ESP32-WROOM-32 telemetry: 4× FSR load cells, 6× IR proximity, 2× HC-SR04 ultrasonic, collar MPU6050 IMU, MAX30102 pulse oximeter, over TCP JSON on port 3333 |
| **QML backend** | `src/qmlkit` | Quantum + classical + hybrid modelling (`quantum/`, `classical/`, `lab/`), leak-free benchmarking (`evaluation/`), VOC and kennel data ingestion (`data/`, `hardware/`), biomarker attribution (`explainability/`), FastAPI + WebSocket services (`api/`) |
| **Next.js console** | `frontend` | Live kennel dashboard, VOC diagnostics view with explainability, Training Studio, Data Lab recording controller, Hybrid Lab experiment launcher |

Packaging follows a pnpm-workspace + PyInstaller-onedir methodology
(`package.json`, `packaging/qmlkit_desktop.spec`, `scripts/build_desktop.ps1`).

---

## 2. The two problem domains

The platform addresses two related screening problems, which is why two model
families exist.

### 2.1 VOC screening (breath biomarkers)

**Input:** concentrations of volatile organic compounds in exhaled breath.
**Question:** can a *chemical* signature separate cancer from control?
**Modelling path:** concentration vector → scaling → register reduction →
BioZZ feature map → QSVM / VQC / hybrid head.
**Data:** the bundled real dataset `docs/Lung_Cancer_VOC_Dataset_427.md`
(427 patients × 27 compounds), plus a physics-based synthetic generator.
**See:** [`02_data_layer.md`](02_data_layer.md), [`05_quantum_models.md`](05_quantum_models.md).

### 2.2 Kennel micro-movement screening

**Input:** an instrumented examination kennel in which a dog stands/sniffs,
producing load, proximity, distance, and 6-axis inertial signatures.
**Question:** can canine *behavioural* micro-movements be screened
non-invasively?
**Modelling path:** streamed telemetry → 400-frame window → 45 window features
(+10 baseline deltas +1 recovery ratio = **55 trial features**) → median
imputation → RandomForest (joblib artifact) → EMA-smoothed probability.
**Data:** recorded sessions under `data/kennel/<label>/<dog>_<label><trial>.csv`.
**See:** [`10_kennel_hardware.md`](10_kennel_hardware.md), [`11_realtime_pipeline.md`](11_realtime_pipeline.md).

The VOC path is the quantum contribution. The kennel path is the hardware /
systems contribution. They share the leakage discipline, the feature-extraction
contract, and the console.

---

## 3. End-to-end data flow

### 3.1 Offline (batch) VOC path

```
docs/Lung_Cancer_VOC_Dataset_427.md  (or synthetic generator / user CSV)
  │
  ├─ qmlkit.data.dataset_loader            markdown pipe-table parser, label mapping,
  │                                       pd.to_numeric coercion, balanced_subsample
  ▼
DataFrame X (N × 27) , y ∈ {0,1}
  │
  ├─ stratified 80/20 split (train_test_split, stratify=y, random_state=seed)
  ▼
X_train_raw / X_test_raw
  │
  ├─ qmlkit.data.preprocessor.BiomedicalDataPipeline   StandardScaler fit on TRAIN only
  ▼
X_tr_scaled / X_te_scaled
  │
  ├─ qmlkit.data.feature_selector.QuantumFeatureSelector (PCA/KPCA/mutual_info/autoencoder)
  │                                               → n_qubits dims, min-max → [-π, π]
  ▼
X_tr_q / X_te_q
  │
  ├─ qmlkit.evaluation.benchmark_suite.compute_qubit_covariance
  │                                     raw-feature correlation matrix (n_qubits × n_qubits)
  ▼
cov_matrix
  │
  ├──────────────────────────┬───────────────────────────┬──────────────────────┐
  ▼                          ▼                           ▼                      ▼
QSVMClassifier            VariationalQuantum-      HybridPipeline          Classical
(fidelity kernel →        Classifier (BioZZ +      (kernel-fused or       baselines
 precomputed SVC)          StronglyEntangling)      VQC-augmented XGB)     (sklearn)
  │                          │                           │                      │
  └──────────────┬───────────┴───────────────┬───────────┴──────────────────────┘
                 ▼                           ▼
        compute_clinical_metrics()  →  10 metrics + train/infer timing
                 │
                 ▼
   BenchmarkSuite.run_benchmark → pd.DataFrame sorted by ROC-AUC
                 │
                 ├─► outputs/benchmark*/benchmark_metrics.csv
                 ├─► outputs/benchmark*/benchmark_comparison.png
                 └─► outputs/benchmark*/run_metadata.txt
```

**Leak-freeness is structural, not incidental.** The scaler, the selector, the
min–max bounds, and the covariance matrix are each fitted on the training
partition only and reused unchanged at transform time. `HybridPipeline.fit`
(`src/qmlkit/lab/pipeline.py:231`) and `BenchmarkSuite.run_benchmark`
(`src/qmlkit/evaluation/benchmark_suite.py:164`) both do the transform of the
test partition with the *already-fitted* train objects. See
[`03_preprocessing_and_reduction.md`](03_preprocessing_and_reduction.md) §5.

### 3.2 Online (realtime) kennel path

```
ESP32 kennel_node.ino
  • IMU 100 Hz • FSR/IR/US 20 Hz • MAX30102 2 Hz
  • JSON frame every 100 ms (10 Hz) over TCP :3333, newline-delimited
  ▼
qmlkit.hardware.kennel_streaming.create_stream_source()
  • source=hardware → TcpKennelSource (newline split, parse_frame validation)
  • source=simulation → SimulatedKennelSource (100 Hz synthetic, seed 42)
  • source=auto (default) → one-shot 2.0 s TCP probe, OSError → simulation
  ▼
queue.Queue(maxsize=4000), oldest-drop on overflow
  ▼
predictor_worker thread  (qmlkit/api/kennel_server.py:163)
  1. recorder.add_frame()            → Data Lab CSV if a session is active
  2. every 10th frame → asyncio.Queue(maxsize=200) via call_soon_threadsafe
  3. append to predictor_buffer; at 400 frames:
        extract 45 features → impute → RF.predict_proba
        EMA(α=0.3) → status ok|uncertain (max(p,1−p) ≥ 0.6)
  ▼
  ├─► raw_stream_broadcaster  → WebSocket /ws/stream   (10 Hz raw frames)
  └─► broadcast_prediction   → WebSocket /ws/diagnostic (~0.25 Hz results)
                                   ▲
                                   └── Next.js console (Diagnostics page)
  ▼
GET /api/v1/kennel/state (2 s poll) → stream source, frames seen, model status,
                                     recorder progress, buffer depth
```

**No labelled kennel data → no artifact → `status: "untrained"`.** This is the
designed first-run state; the console renders a "Model Awaiting Calibration"
panel pointing at the Training Studio. See
[`11_realtime_pipeline.md`](11_realtime_pipeline.md) §4.

---

## 4. Key architectural decisions

| # | Decision | Rationale | Where |
|---|---|---|---|
| D1 | PennyLane `default.qubit` statevector, **not** a QPU backend | The project is a NISQ-*feasibility* study, not a hardware deployment; exact kernels are needed for correct fidelity math and reproducibility | `quantum/qsvm.py:27`, `quantum/vqc.py:42`, `QuantumModelConfig.device_name` |
| D2 | One feature contract (`src/qmlkit/hardware/kennel_features.py`) shared by training, serving, and the synthetic generator | Prevents train/serve skew; the same 45 features are produced from a recorded CSV and from a live TCP stream | `kennel_features.extract_window_features` |
| D3 | `StandardScaler` fitted on train only; min–max bounds captured inside the selector | Quantum rotations need bounded angles; the bounds must not see test data | `feature_selector.py:103-105, 151-159` |
| D4 | Covariance matrix derived from **raw scaled features**, not PCA scores | PCA scores are orthogonal by construction → `corrcoef` ≈ identity → BioZZ entanglement would be silently inert. This was a real defect, fixed | `benchmark_suite.py:100-153` |
| D5 | Classical baselines live in the *same* CV loop as quantum models | Any quantum/classical comparison is only meaningful under identical partitions | `lab/cv.py:29-77` |
| D6 | Identical stratified partitions across all configurations (split on `np.zeros`) | Makes per-config score differences attributable to configuration, not fold luck | `lab/cv.py:38` |
| D7 | Permuted-correlation control (`cwzz_permuted`) ships as a first-class preset | An ablation that cannot fail is not an ablation; the project needed a falsifiable test of its own headline claim | `lab/pipeline.py:55-66`, preset #4 |
| D8 | Out-of-fold VQC augmentation for the hybrid head | In-sample VQC probabilities produced 100% scores — a classic self-confirmation leak. OOF fixed it | `lab/pipeline.py:295-354` |
| D9 | Two disjoint FastAPI apps (`server.py`, `kennel_server.py`) | Different domains and lifecycle; keeps the screening service free of thread/queue machinery | `api/__init__.py` |
| D10 | Next.js `output: "export"` (static) + `app.mount("/")` from FastAPI | One process ships the console; no Node runtime required at deploy | `frontend/next.config.ts:3-7`, `kennel_server.py:373-379` |
| D11 | Simulation fallback in `auto` mode | The platform must boot and demo without hardware | `kennel_streaming.py:226-245` |
| D12 | Reported results include the negative control | Reviewers trust a project that reports its own disconfirming evidence | `docs/reports/hybrid_search_report.md:39-43` |

---

## 5. Request/response surface (both services)

### 5.1 Live kennel service — `qmlkit.api.kennel_server:app` (port 8000)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/kennel/state` | stream source, connected, frames seen, model status, recorder progress, buffer/batch sizes |
| POST | `/api/v1/kennel/start` | start labelled recording session (409 if already recording) |
| POST | `/api/v1/kennel/stop` | flush + write CSV, returns path (409 if not recording) |
| POST | `/api/v1/train` | synchronous VOC-domain training job |
| POST | `/api/v1/benchmark/run` | multi-model leaderboard |
| GET | `/api/v1/dataset/stats` | VOC dataset metadata |
| POST | `/api/v1/kennel/frame` | inject a pre-parsed frame (test/utility hook) |
| POST/GET | `/api/v1/lab/runs`, `/api/v1/lab/runs/{job_id}` | background hybrid-lab experiment jobs |
| WS | `/ws/stream` | raw frames at 10 Hz |
| WS | `/ws/diagnostic` | windowed inference results, last result replayed on connect |
| GET | `/` | static console when `QMLKIT_FRONTEND_DIST` is set |

### 5.2 Screening service — `qmlkit.api.server:create_app()`

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | liveness + backend string |
| GET | `/api/v1/demo-sample/{preset}` | a held-out test row as 64 floats (`healthy_control`, `lung_positive`) |
| POST | `/api/v1/predict` | single-sample screening + biomarker attribution (`?deep_explain=true` enables kernel SHAP) |
| POST | `/api/v1/train` | same trainer as above |
| POST | `/api/v1/benchmark/run` | same suite, typed request model |
| GET | `/api/v1/dataset/stats` | dataset metadata |
| GET | `/` | self-contained Bootstrap 5.3 dark SPA inlined in Python |

Full route tables with request/response shapes:
[`11_realtime_pipeline.md`](11_realtime_pipeline.md).

---

## 6. Repository module map

| Module | Files | Purpose |
|---|---|---|
| `qmlkit.config` | `config.py` | Pydantic config models (`SensorArrayConfig`, `VOCBiomarkerConfig`, `QuantumModelConfig`, `BenchmarkConfig`, `QMLKitAppConfig`) + `set_seed()` seeding `random`/`numpy`/`torch`/CUDA/`PYTHONHASHSEED` |
| `qmlkit.data` | `dataset_loader.py`, `preprocessor.py`, `feature_selector.py`, `biomimetic_voc_generator.py` | ingestion, scaling, register reduction, synthetic cohort physics |
| `qmlkit.quantum` | `feature_maps.py`, `qsvm.py`, `vqc.py`, `qcnn.py`, `qrc.py`, `kernel_features.py` | circuits and quantum models |
| `qmlkit.classical` | `baselines.py` | 6 sklearn baselines + `Temporal1DCNN` (PyTorch) |
| `qmlkit.evaluation` | `benchmark_suite.py`, `hardware_profiler.py` | 10-metric comparison, NISQ gate/depth accounting |
| `qmlkit.explainability` | `biomarker_mapper.py`, `quantum_shap.py` | latent→chemical attribution, kernel SHAP |
| `qmlkit.lab` | `pipeline.py`, `experiments.py`, `cv.py`, `registry.py`, `stacking.py`, `data_sources.py`, `kennel_synth.py` | composable hybrid pipelines and the experiment protocol |
| `qmlkit.hardware` | `protocol.py`, `kennel_streaming.py`, `kennel_features.py`, `session_recording.py` | telemetry contract, ingestion, windowed features, recording |
| `qmlkit.api` | `kennel_server.py`, `server.py`, `training_service.py` | HTTP/WS services and the blocking trainer |

---

## 7. Where to start reading the code

Recommended review order for a new reviewer:

1. `src/qmlkit/quantum/feature_maps.py` — the headline idea, 134 lines
2. `src/qmlkit/quantum/qsvm.py` — how the feature map becomes a classifier
3. `src/qmlkit/evaluation/benchmark_suite.py:100-153` — the covariance fix (the most instructive function in the repo)
4. `src/qmlkit/lab/pipeline.py:231-366` — `HybridPipeline.fit`, the leak-free composition
5. `src/qmlkit/lab/cv.py` — the CV contract everything else obeys
6. `src/qmlkit/hardware/kennel_features.py` — the 45-feature contract
7. `src/qmlkit/api/kennel_server.py:163-224` — the realtime threading model
8. `docs/reports/real_data_benchmark_report.md`, `docs/reports/hybrid_search_report.md` — what the system actually achieves
