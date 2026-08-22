# 07. Implementation Roadmap, Milestones, and Execution Plan

## 1. Project Phase Breakdown

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   QMLKIT DEVELOPMENT ROADMAP                                     │
├─────────┬─────────────────────────────────────────────────┬──────────────────────┬───────────────┤
│ Phase   │ Objective & Key Deliverables                    │ Core Tech / Modules  │ Milestone Ref │
├─────────┼─────────────────────────────────────────────────┼──────────────────────┼───────────────┤
│ **PHASE 1**│ Synthetic Biomimetic VOC Data Engine &          │ `numpy`, `scipy`,    │ **M1: Data &**│
│         │ Preprocessing Pipeline (Savitzky-Golay, drift)  │ `pandas`, `sklearn`  │ **Pipeline**  │
├─────────┼─────────────────────────────────────────────────┼──────────────────────┼───────────────┤
│ **PHASE 2**│ Core Quantum Circuits & Models                  │ `PennyLane`, `Qiskit`│ **M2: Quantum**│
│         │ (BioZZFeatureMap, QSVM, VQC, QCNN, QRC)         │ `PyTorch`            │ **Core**      │
├─────────┼─────────────────────────────────────────────────┼──────────────────────┼───────────────┤
│ **PHASE 3**│ Classical Baselines & Leak-Free Benchmarking   │ `scikit-learn`,      │ **M3: Baseline**│
│         │ (SVM-RBF, RF, XGBoost, 1D-CNN, Metrics Suite)   │ `xgboost`, `pytest`  │ **Benchmark** │
├─────────┼─────────────────────────────────────────────────┼──────────────────────┼───────────────┤
│ **PHASE 4**│ Quantum Explainability & Reverse Bio-Mapping    │ `shap`, Custom XQAI, │ **M4: Clinical**│
│         │ (Quantum Kernel SHAP, VOC Biomarker Ranking)    │ `matplotlib`         │ **Explain**   │
├─────────┼─────────────────────────────────────────────────┼──────────────────────┼───────────────┤
│ **PHASE 5**│ Diagnostic REST API & Web Screening Portal      │ `FastAPI`, `uvicorn`,│ **M5: Portal**│
│         │ (Interactive dashboard, report generator)       │ `pydantic`, HTML/JS  │ **& Release** │
└─────────┴─────────────────────────────────────────────────┴──────────────────────┴───────────────┘
```

---

## 2. Directory Layout & Module Blueprint

```
QMLKit/
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   └── init/
│       ├── 01_problem_statement_and_vision.md
│       ├── 02_system_architecture.md
│       ├── 03_canine_olfactory_voc_dataset_spec.md
│       ├── 04_quantum_algorithms_and_circuits.md
│       ├── 05_classical_baselines_and_benchmarking.md
│       ├── 06_explainability_and_biomarker_attribution.md
│       └── 07_roadmap_and_implementation_plan.md
├── src/
│   └── qmlkit/
│       ├── __init__.py
│       ├── config.py                 # Pydantic schemas, experiment configs, seed controls
│       ├── data/                     # Data synthesis, ingestion, and preprocessing
│       │   ├── __init__.py
│       │   ├── biomimetic_voc_generator.py # Realistic canine olfactory sensor synthesizer
│       │   ├── preprocessor.py       # Baseline drift correction, filtering, normalizers
│       │   └── feature_selector.py   # PCA, Kernel PCA, Autoencoder, Mutual Information
│       ├── quantum/                  # Quantum Circuits & Hybrid Classifiers
│       │   ├── __init__.py
│       │   ├── feature_maps.py       # BioZZFeatureMap, AngleEmbedding, CovarianceMap
│       │   ├── qsvm.py               # QSVM with exact Quantum Kernel Matrix calculation
│       │   ├── vqc.py                # Variational Quantum Classifier (PennyLane/PyTorch)
│       │   ├── qrc.py                # Quantum Reservoir Computing for kinetic signals
│       │   └── qcnn.py               # Quantum Convolutional Neural Network
│       ├── classical/                # Classical Machine Learning Baselines
│       │   ├── __init__.py
│       │   └── baselines.py          # SVM-RBF, Linear SVM, RF, XGBoost, MLP, 1D-CNN
│       ├── explainability/           # Explainable Quantum AI (XQAI)
│       │   ├── __init__.py
│       │   ├── quantum_shap.py       # Quantum Kernel SHAP & State Saliency
│       │   └── biomarker_mapper.py   # Projection of latent quantum states to VOCs
│       ├── evaluation/               # Metrics & NISQ Hardware Profiler
│       │   ├── __init__.py
│       │   ├── benchmark_suite.py    # Stratified K-Fold CV, ROC-AUC, Sensitivity, Specificity
│       │   └── hardware_profiler.py  # Circuit depth, CNOT count, shot noise analysis
│       └── api/                      # REST API & Web Delivery
│           ├── __init__.py
│           └── server.py             # FastAPI screening endpoints & telemetry
├── scripts/                          # Executable CLI Workflows
│   ├── generate_voc_data.py          # CLI to synthesize or ingest datasets
│   ├── run_benchmark.py              # CLI to run full quantum vs classical benchmarks
│   ├── explain_patient.py            # CLI to generate clinical biomarker report
│   └── launch_portal.py              # CLI to start the FastAPI diagnostic dashboard
└── tests/                            # Automated Pytest Suite
    ├── test_data_pipeline.py         # Leak-free splitters, drift corrections
    ├── test_quantum_circuits.py      # Kernel symmetry, unit diagonal, VQC forward pass
    ├── test_classical_baselines.py   # Baseline training, convergence, metric sanity
    ├── test_explainability.py        # Biomarker attribution ranking validity
    └── test_e2e_workflow.py          # Full synthetic-to-prediction-to-report pipeline
```

---

## 3. Detailed Phase Deliverables

### Phase 1: Data Ingestion & Biomimetic Synthesis
- Implement `biomimetic_voc_generator.py`: Generates multi-channel sensor arrays and underlying VOC biomarker concentrations across healthy controls and target cancer types (Lung, Breast, Colorectal, Prostate, Ovarian).
- Implement `preprocessor.py`: Baseline subtraction $\frac{\Delta R}{R_0}$, Savitzky-Golay smoothing, and extraction of dynamic transient descriptors.
- Implement leak-free data splitters guaranteeing no test set statistics contaminate training scalers.

### Phase 2: Quantum Algorithms Implementation
- Implement `feature_maps.py`: `BioZZFeatureMap`, `AngleEmbedding`, and `CovarianceFeatureMap` in PennyLane and Qiskit.
- Implement `qsvm.py`: Parallelized quantum kernel matrix generator and dual SVM optimizer.
- Implement `vqc.py`: Multi-qubit variational circuit with parameter-shift rule differentiation and PyTorch hybrid layer bindings.
- Implement `qcnn.py` & `qrc.py`: Quantum Convolutional Classifier and Quantum Reservoir Computer.

### Phase 3: Classical Baselines & Comparative Benchmarking
- Implement `baselines.py`: Standardized scikit-learn and PyTorch wrappers for SVM-RBF, Random Forest, XGBoost, MLP, and 1D-CNN.
- Implement `benchmark_suite.py`: Automated multi-metric evaluator computing Sensitivity, Specificity, ROC-AUC, PR-AUC, F1, and Brier score under identical 5-fold cross-validation.
- Implement `hardware_profiler.py`: Circuit depth, 2-qubit gate counts, and shot noise robustness curves ($shots \in [500, 8192]$ vs. analytic statevectors).

### Phase 4: Quantum Explainability & Reverse Bio-Mapping
- Implement `quantum_shap.py`: Kernel SHAP implementation operating on quantum state embeddings.
- Implement `biomarker_mapper.py`: Projects latent qubit attributions back to original VOC chemical analytes (Hexanal, 2-Butanone, Benzaldehyde, etc.), generating clinical waterfall charts and radar profiles.

### Phase 5: REST API & Interactive Clinical Portal
- Implement `server.py`: FastAPI server with endpoints `/api/v1/predict`, `/api/v1/benchmark`, and `/api/v1/explain`.
- Web diagnostic interface allowing clinicians to upload sensor profiles, view quantum vs classical diagnostic confidence, inspect biomarker attributions, and export PDF/HTML diagnostic summaries.

---

## 4. Quality Assurance & Testing Standards

1. **Deterministic Seeds:** Global seeding across `numpy`, `torch`, and quantum simulator devices to ensure complete test reproducibility.
2. **Defect-Free Math:**
   - Assert $K(\mathbf{x}_i, \mathbf{x}_i) = 1.0 \pm 10^{-6}$ for all diagonal kernel entries.
   - Assert $K(\mathbf{x}_i, \mathbf{x}_j) = K(\mathbf{x}_j, \mathbf{x}_i)$ within numerical tolerance.
   - Assert all probabilities $p \in [0, 1]$ and loss functions monotonically decrease during training.
3. **Continuous Verification:** Pytest test suite targeting $>90\%$ code coverage across data, quantum, classical, and explainability modules.
