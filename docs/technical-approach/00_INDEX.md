# QMLKit — Technical Approach Documentation

**Scope:** end-to-end technical description of QMLKit for (a) a full technical
review and (b) a conference/assessment presentation.

**Repository state described:** `main` @ `e306dbb`, `version.properties` = `0.1.4`.
**Stack:** Python 3.11+ / PennyLane ≥ 0.40 / FastAPI / Next.js 16 / React 19 /
ESP32-WROOM-32 (PlatformIO).

Every statement in these documents is traceable to a file and line in the
repository. Where a documented value is stale relative to the code, the code is
treated as ground truth and the discrepancy is called out explicitly in
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md).

---

## Document map

| # | Document | What it answers |
|---|---|---|
| 00 | This index | Where to look for what |
| 01 | [`01_system_overview.md`](01_system_overview.md) | What the system is, how the three subsystems connect, end-to-end data flow, key design decisions |
| 02 | [`02_data_layer.md`](02_data_layer.md) | All datasets, schemas, loaders, label mappings, subsampling, split protocol |
| 03 | [`03_preprocessing_and_reduction.md`](03_preprocessing_and_reduction.md) | Leak-free scaling, quantum feature selection, angle re-scaling, the covariance matrix |
| 04 | [`04_quantum_feature_maps.md`](04_quantum_feature_maps.md) | Angle / Pauli-ZZ / BioZZ circuit math, gate-level structure, the correlation-weighting mechanism |
| 05 | [`05_quantum_models.md`](05_quantum_models.md) | QuantumKernel, QSVM, VQC, QCNN, QRC, QuantumKernelFeatureTransformer |
| 06 | [`06_hybrid_pipelines.md`](06_hybrid_pipelines.md) | `HybridPipeline` / `PipelineSpec`, the two hybrid fusion heads, ensembles |
| 07 | [`07_baselines_and_metrics.md`](07_baselines_and_metrics.md) | Classical baselines + exact hyperparameters, the 10 clinical metrics, benchmark suite |
| 08 | [`08_experiment_lab.md`](08_experiment_lab.md) | Cross-validation protocol, presets, ablations, robustness, run registry |
| 09 | [`09_explainability.md`](09_explainability.md) | Latent→chemical attribution, pathway aggregation, quantum-kernel SHAP |
| 10 | [`10_kennel_hardware.md`](10_kennel_hardware.md) | ESP32 firmware, sensors, state machine, telemetry protocol, 45 window features |
| 11 | [`11_realtime_pipeline.md`](11_realtime_pipeline.md) | TCP ingestion, simulation fallback, server threads/queues, WebSocket protocols, REST API |
| 12 | [`12_frontend_console.md`](12_frontend_console.md) | Next.js console: pages, components, state, Tailwind v4, API auto-detect |
| 13 | [`13_packaging_and_ops.md`](13_packaging_and_ops.md) | pnpm workspace, PyInstaller bundle, Inno Setup installer, config, run commands |
| 14 | [`14_nisq_feasibility.md`](14_nisq_feasibility.md) | Circuit depth / gate counts / parameter counts, shot-noise analysis, honest NISQ verdict |
| 15 | [`15_results_and_findings.md`](15_results_and_findings.md) | All measured numbers, artifacts, interpretation, reproduce commands |
| 16 | [`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) | Known defects, doc/code drift, statistical caveats, mitigation status |
| 17 | [`17_presentation_guide.md`](17_presentation_guide.md) | Slide-by-slide deck, talk track, demo script, likely Q&A |
| 18 | [`18_review_checklist.md`](18_review_checklist.md) | Reviewer checklist and claims-to-evidence map |

---

## How to use this set

**For a presentation (30–45 min):** read `01`, `04`, `05`, `06`, `15`, then
`17` for the deck and talk track. `14` is the "is this real quantum computing?"
answer slide.

**For a technical review (half to full day):** read in numerical order. `16` is
mandatory reading for anyone signing off — it is the honest assessment of what
the system does and does not yet prove.

---

## One-paragraph summary of the technical approach

QMLKit ingests canine-exhalation VOC biomarker concentrations (real 427-patient
Lung Cancer VOC dataset, or a physics-based synthetic generator) or live kennel
micro-movement telemetry from an ESP32 node. Sensor data is normalised with a
strictly train-fitted `StandardScaler`, then compressed to a register-sized
vector (6 qubits) by PCA, KernelPCA, mutual-information selection, or a PyTorch
autoencoder, and min–max re-scaled into bounded rotation angles $[-\pi,\pi]$. A
**BioZZ feature map** — a covariance-weighted second-order Pauli-Z circuit —
encodes each sample into a Hilbert-space statevector, where the entangling phase
$\phi_{ij}=2\,\mathrm{cov}[i,j]\,(\pi-x_i)(\pi-x_j)$ is *weighted by the empirical
inter-compound correlation*, so VOC cross-reactivity is mapped directly onto ZZ
coupling strength. This is consumed three ways: (i) an exact fidelity quantum
kernel $K(x,x')=|\langle\Phi(x)|\Phi(x')\rangle|^2$ feeding a precomputed-kernel
SVC (QSVM), (ii) a parameterised VQC with a `StronglyEntanglingLayers` ansatz
trained by PennyLane autograd with backprop differentiation, and (iii) a landmark
quantum-kernel feature transformer whose similarity features are fused with the
classical features into XGBoost. A parallel classical path (LogReg, linear/RBF
SVM, RandomForest, XGBoost, MLP, 1-D CNN) is benchmarked under an identical
leak-free protocol using 10 clinical metrics. A composable experiment lab
(`PipelineSpec` × CV × ablations × robustness) establishes that **hybrid**
configurations — quantum representation feeding a strong classical learner —
outperform both pure-quantum and pure-classical arms, and that at current sample
sizes the correlation-weighting advantage of BioZZ is **within noise**, which the
project reports rather than hides. All of this is wrapped in a live FastAPI +
WebSocket backend and a static Next.js console, packaged as a PyInstaller
desktop app.

---

## Repository layout

```
qmlkit/
├─ src/qmlkit/
│  ├─ config.py                 Pydantic config models + global seeding
│  ├─ data/                     dataset_loader, preprocessor, feature_selector, synthetic generator
│  ├─ quantum/                  feature_maps, qsvm, vqc, qcnn, qrc, kernel_features
│  ├─ classical/                baselines.py (sklearn + Temporal1DCNN)
│  ├─ evaluation/               benchmark_suite, hardware_profiler
│  ├─ explainability/           biomarker_mapper, quantum_shap
│  ├─ lab/                      pipeline, experiments, cv, registry, stacking, data_sources, kennel_synth
│  ├─ hardware/                 protocol, kennel_streaming, kennel_features, session_recording
│  └─ api/                      kennel_server (live), server (screening), training_service
├─ firmware/kennel_node/        ESP32 PlatformIO sketch + README
├─ frontend/                    Next.js 16 console (app/, components/, lib/)
├─ scripts/                     run_benchmark, run_hybrid_search, train_kennel_model, build_ecg_tables,
│                               explain_patient, generate_voc_data, launch_portal, build/start_*.ps1|sh
├─ packaging/                   PyInstaller spec + Inno Setup iss + linux/
├─ tests/                       8 files, 59 tests
├─ docs/                        init/, workings/, upgrade/, reports/, technical-approach/ (this set)
├─ outputs/                     committed benchmark + lab artifacts
└─ data/                        kennel recordings, ecg tables, sample cohort
```
