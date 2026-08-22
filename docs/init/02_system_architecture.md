# 02. System Architecture: End-to-End Hybrid Quantum-Classical Platform

## 1. Architectural Overview & Design Philosophy

The QMLKit platform is engineered with a strict **layered, modular, and leak-free** hybrid computing paradigm. It bridges high-throughput biomimetic chemical sensing instruments (e-Nose / GC-MS / Bio-electronic arrays) with near-term Noisy Intermediate-Scale Quantum (NISQ) devices and simulators.

```
+===================================================================================================+
|                                    QMLKIT SYSTEM ARCHITECTURE                                     |
+===================================================================================================+

 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   1. SENSING & INGESTION LAYER                                  │
 │  - Biomimetic Sensor Matrix (MOS / Bio-FET / FAIMS / GC-MS)                                     │
 │  - Real-time Stream & Batch Ingestion (CSV / JSON / HDF5 / Feather)                            │
 │  - Canine Olfactory Biomarker Spectrum (Aldehydes, Ketones, Alkanes, Aromatics)                 │
 └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                 2. BIOMEDICAL PREPROCESSING LAYER                               │
 │  - Baseline Drift Correction (Polynomial & Asymmetric Least Squares Baseline Removal)          │
 │  - Dynamic Transient Denoising (Savitzky-Golay Filtering, Wavelet Denoising)                    │
 │  - Kinetic Response Extraction: Max Response ($\Delta R/R_0$), Integral Area, Rise/Decay Rates  │
 │  - Leak-free Chronological & Stratified Data Splitting (Scalers fit strictly on train slice)   │
 └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                              3. QUANTUM-AWARE REDUCTION & EMBEDDING                             │
 │  - High-Dim -> Qubit Dimension Reduction: Supervised PCA, Kernel PCA, Autoencoder, Mutual Info  │
 │  - Scaling to $[0, \pi]$ / $[-\pi, \pi]$ bounded angles for unitary rotation quantum gates       │
 │  - Quantum Feature Mappings: Angle Embedding, ZZFeatureMap, Covariance-Weighted BioFeatureMap    │
 └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                 4. HYBRID QUANTUM COMPUTING CORE                                │
 │  ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────────┐  │
 │  │        QSVM Engine        │ │        VQC / QNN          │ │      Quantum Reservoir (QRC)  │  │
 │  │ - Exact Kernel Matrix     │ │ - StronglyEntanglingLayers│ │ - Dynamic Unitary Reservoir   │  │
 │  │ - Quantum Kernel Ridge    │ │ - Parameterized Ansätze   │ │ - Temporal Kinetic Readout    │  │
 │  │ - C-SVC Dual Optimization │ │ - Adam / COBYLA / SPSA    │ │ - Recurrent State Memory      │  │
 │  └───────────────────────────┘ └───────────────────────────┘ └───────────────────────────────┘  │
 │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
 │  │ Hardware / Simulator Backends: PennyLane (`default.qubit`), Qiskit Aer, IBM Quantum Cloud │  │
 │  └───────────────────────────────────────────────────────────────────────────────────────────┘  │
 └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                            5. EXPLAINABILITY & BIOMARKER ATTRIBUTION                            │
 │  - Quantum Kernel SHAP (Shapley Additive Explanations in Quantum Feature Space)                 │
 │  - Quantum Fisher Information Matrix (QFIM) & Parameter Saliency Gradients                      │
 │  - Reverse Bio-Mapping: Attribution from Hilbert space states to specific chemical VOC molecules│
 └────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                             6. BENCHMARKING & CLINICAL DELIVERY LAYER                           │
 │  - Classical Baseline Comparison: SVM-RBF, Random Forest, XGBoost, LightGBM, MLP, 1D-CNN        │
 │  - Diagnostic Metrics: Sensitivity, Specificity, ROC-AUC, PR-AUC, F1-Score, Brier Loss         │
 │  - NISQ Resource Profiler: 2-Qubit Gate Count, Circuit Depth, Shot Noise Tolerance Curve        │
 │  - Clinical Delivery: FastAPI REST Service + Interactive Web Dashboard                         │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Layer 1: Ingestion & Data Formats
- **Input Modalities:** Multi-sensor transient responses (time-series voltage/conductance per sensor channel) and static chemical concentration matrices (ppm/ppb VOC concentrations).
- **Data Validation:** Pydantic schema validation for sensor count, sampling frequency, baseline duration, exposure duration, and clinical metadata (Patient ID, Age, Smoking Status, Histological Ground Truth).

### 2.2 Layer 2: Biomedical Preprocessing
1. **Drift Removal:**
   $$R_{\text{norm}}(t) = \frac{R(t) - R_0}{R_0}$$
   where $R_0$ is the baseline resistance in purified zero-air.
2. **Kinetic Feature Extraction:**
   - Peak Amplitude: $\Delta R_{\max} = \max_t |R_{\text{norm}}(t)|$
   - Response Area: $A = \int_{t_{\text{on}}}^{t_{\text{off}}} R_{\text{norm}}(t) \, dt$
   - Slope of Adsorption: $\left.\frac{dR}{dt}\right|_{t_{\text{rise}}}$
   - Desorption Time Constant: $\tau_{\text{decay}}$
3. **Leakage Prevention Standard:**
   Scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) and Dimensionality Reducers (`PCA`, `KernelPCA`) MUST be instantiated and fit **exclusively** on the training partition:
   ```python
   # GUARANTEED LEAK-FREE SPLIT
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   scaler = StandardScaler().fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

### 2.3 Layer 3: Dimensionality Reduction & Quantum Feature Encoding
Given a target NISQ register of $n$ qubits (typically $n \in [4, 16]$):
- **Reduction Strategy:** Supervised PCA or Autoencoder mapping $D$-dimensional sensor features to $n$-dimensional latent space $\mathbf{z} \in [-\pi, \pi]^n$.
- **Quantum Feature Map Operators:**
  - **Angle Embedding:** $U(\mathbf{z}) = \bigotimes_{i=1}^n R_y(z_i)$
  - **Second-Order Pauli-Z (ZZFeatureMap):**
    $$U_{\Phi}(\mathbf{z}) = \exp\left(i \sum_{j=1}^n z_j Z_j + i \sum_{j < k} (\pi - z_j)(\pi - z_k) Z_j Z_k \right) H^{\otimes n}$$
  - **Canine-Biomimetic Covariance Feature Map:** Weights the 2-qubit entangling angles by the empirical cross-covariance of VOC functional groups.

### 2.4 Layer 4: Quantum Machine Learning Engines
1. **QSVM (Quantum Support Vector Machine):**
   Computes the $N \times N$ Quantum Kernel Gram matrix $K_{ij} = |\langle 0^{\otimes n} | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0^{\otimes n} \rangle|^2$.
   Solves the classical dual quadratic optimization problem:
   $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K_{ij} \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \sum_{i=1}^N \alpha_i y_i = 0$$
2. **Variational Quantum Classifier (VQC):**
   Parameterized ansatz circuit $W(\boldsymbol{\theta})$ with $L$ entangling layers:
   $$\hat{y}(\mathbf{x}) = \langle 0^{\otimes n} | U^\dagger(\mathbf{x}) W^\dagger(\boldsymbol{\theta}) M W(\boldsymbol{\theta}) U(\mathbf{x}) | 0^{\otimes n} \rangle$$
   Trained via gradient descent (Adam / Parameter-Shift Rule / SPSA) minimizing binary cross-entropy or margin loss.
3. **Quantum Reservoir Computing (QRC):**
   Processes non-linear time-series sensor kinetics using a fixed, complex entangling Hamiltonian reservoir, reading out expectation values $\langle Z_i \rangle, \langle Z_i Z_j \rangle$ to train a classical ridge regressor.

### 2.5 Layer 5: Explainability & Biomarker Attribution
- **Quantum SHAP:** Approximates Shapley values in the quantum feature space by sampling feature subsets and computing the marginal contribution of each VOC dimension to the quantum decision boundary.
- **Reverse Spectral Mapping:** Projects quantum feature importance back through the PCA / Autoencoder decoder weights to individual VOC compounds (e.g. Hexanal, 2-Butanone).

### 2.6 Layer 6: Classical Baselines & Verification
- Side-by-side benchmarking against **Support Vector Machine (RBF/Linear)**, **Random Forest**, **XGBoost**, **LightGBM**, **Multi-Layer Perceptron (MLP)**, and **1D-CNN**.
- Outputs comprehensive metrics: Sensitivity, Specificity, Balanced Accuracy, ROC-AUC, PR-AUC, F1-Score, Confusion Matrices, and NISQ gate execution stats.

---

## 3. Technology Stack & Dependencies

| Category | Primary Technology | Purpose |
|---|---|---|
| **Quantum Framework** | `PennyLane` (v0.40+) / `Qiskit` (v1.3+) | Quantum circuit compilation, QNodes, variational optimizers, simulators |
| **Classical Machine Learning** | `scikit-learn` (v1.6+), `torch` (v2.6+), `xgboost` | Preprocessing, classical baselines, PyTorch hybrid QNode layers |
| **Numerical & Data Handling** | `numpy`, `pandas`, `scipy` | Signal processing, matrix operations, data manipulation |
| **Explainable AI (XAI)** | `shap`, custom Quantum-SHAP engine | Feature attribution and biomarker ranking |
| **Backend & Web API** | `FastAPI`, `uvicorn`, `pydantic` | Microservice REST API for automated screening |
| **Visualization & Reporting** | `matplotlib`, `seaborn` | ROC curves, confusion matrices, quantum kernel heatmaps, radar plots |
| **Testing & CI/CD** | `pytest`, `ruff` | Automated unit testing, linting, regression prevention |
