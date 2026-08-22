# 05. Classical Baselines, Leak-Free Validation, and Benchmarking Protocol

## 1. Benchmarking Philosophy & Anti-Defect Standards

A core requirement of Problem Statement 26139 is **rigorous, scientifically defensible benchmarking against classical baselines**. 

Drawing directly from the audit principles outlined in our codebase remediation standards (e.g. `ibm-qff25-hackathon/REMEDIATION_PLAN.md`), the QMLKit platform enforces strict anti-leakage and anti-circularity guidelines:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             CRITICAL ANTI-DEFECT VALIDATION RULES                                │
├────┬─────────────────────────────┬───────────────────────────────────────────────────────────────┤
│ ID │ Defect Risk                 │ QMLKit Enforcement Rule                                       │
├────┼─────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ **L1** │ Feature Space Mismatch      │ Unified transformation pipeline; training scalers and PCA     │
│    │                             │ objects are serialized and reused identically at inference.   │
├────┼─────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ **L2** │ Pre-split Leakage           │ Scalers, normalizers, and dimensionality reducers MUST be fit │
│    │                             │ EXCLUSIVELY on training folds (never on full dataset).        │
├────┼─────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ **L3** │ Circular Metric Inflation   │ Ground truth is derived from biological/histological labels,   │
│    │                             │ never from internal heuristic rules matching predicted states.│
├────┼─────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ **L4** │ Validation Contamination    │ Independent 3-way split (Train / Validation / Test) or Nested │
│    │                             │ Stratified K-Fold. Test partition is strictly held out.       │
└────┴─────────────────────────────┴───────────────────────────────────────────────────────────────┘
```

---

## 2. Classical Machine Learning Baseline Suite

To evaluate quantum advantage objectively, the platform benchmarks all quantum algorithms against 6 industry-standard classical machine learning models trained on identical, leak-free feature partitions:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CLASSICAL BASELINE ALGORITHMS                                    │
├─────────────────────┬──────────────────────────┬─────────────────────────────────────────────────┤
│ Model               │ Implementation           │ Primary Hyperparameter Search Space             │
├─────────────────────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ **Support Vector    │ `sklearn.svm.SVC`        │ $C \in [10^{-2}, 10^3]$, $\gamma \in [\text{scale}, 10^{-3}, 10^1]$ │
│ Classifier (RBF)**  │                          │ Kernel: Radial Basis Function                   │
├─────────────────────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ **Linear SVM**      │ `sklearn.svm.SVC`        │ $C \in [10^{-3}, 10^2]$, Linear Kernel          │
├─────────────────────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ **Random Forest**   │ `sklearn.ensemble.RF`    │ `n_estimators` $\in [50, 500]$, `max_depth` $\in [3, 15]$, │
│                     │                          │ `min_samples_split` $\in [2, 10]$               │
├─────────────────────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ **XGBoost / GBDT**  │ `xgboost.XGBClassifier`  │ `learning_rate` $\in [0.01, 0.2]$, `max_depth` $\in [3, 8]$,│
│                     │                          │ `subsample` $\in [0.6, 1.0]$, `n_estimators` $\in [100, 300]$│
├─────────────────────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ **Multi-Layer       │ `torch.nn` / `sklearn`   │ Hidden layers: $[128, 64, 32]$, Dropout: $0.3$, │
│ Perceptron (MLP)**  │                          │ Optimizer: Adam ($lr = 10^{-3}$), ReLU / GELU   │
├─────────────────────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ **1D-CNN (Temporal│ `torch.nn`                 │ Conv1D filters: $[32, 64]$, Kernel size: $3, 5$,│
│ Sensor Kinetics)**  │                          │ BatchNorm, MaxPool1d, Dense Readout             │
└─────────────────────┴──────────────────────────┴─────────────────────────────────────────────────┘
```

---

## 3. Clinical & Statistical Evaluation Metrics

In oncology screening, general accuracy is often misleading due to class imbalances and differing clinical penalties for False Negatives vs. False Positives. QMLKit evaluates models across 8 rigorous clinical metrics:

```
                    PREDICTED POSITIVE (Cancer)       PREDICTED NEGATIVE (Healthy)
 ACTUAL POSITIVE            True Positive (TP)               False Negative (FN)  <-- CATASTROPHIC
 ACTUAL NEGATIVE           False Positive (FP)                True Negative (TN)
```

### 3.1 Metric Formulations

1. **Sensitivity (Recall / True Positive Rate):**
   $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *Clinical Significance:* Critical for early detection. High sensitivity ensures malignant tumors are not missed.
2. **Specificity (True Negative Rate):**
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
   *Clinical Significance:* Prevents unnecessary panic, biopsies, and healthcare burdens from false alarms.
3. **Positive Predictive Value (Precision):**
   $$\text{PPV} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
4. **Negative Predictive Value (NPV):**
   $$\text{NPV} = \frac{\text{TN}}{\text{TN} + \text{FN}}$$
5. **Balanced Accuracy:**
   $$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$
6. **Macro & Weighted F1-Score:**
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
7. **Area Under the Receiver Operating Characteristic (ROC-AUC):**
   Measures discrimination across all diagnostic probability thresholds.
8. **Brier Score (Probabilistic Calibration Loss):**
   $$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$
   Measures the accuracy and reliability of probabilistic risk scores.

---

## 4. Cross-Validation and Data Splitting Protocol

```
Full Cohort (N = 1000)
  │
  ├── 80% Train-Val Set (N = 800) ─────────────────────────────┐
  │     │                                                      │
  │     ├── 5-Fold Stratified Cross-Validation                 │
  │     │     ├── Fold 1: Train (640) / Val (160)              │
  │     │     ├── Fold 2: Train (640) / Val (160)              │
  │     │     ├── Fold 3: Train (640) / Val (160)              │
  │     │     ├── Fold 4: Train (640) / Val (160)              │
  │     │     └── Fold 5: Train (640) / Val (160)              │
  │     │           │                                          │
  │     │           ▼ (Fit scalers, tune hyperparams)          │
  │     │                                                      │
  │     └── Final Fit on All 800 Samples (Fitted Scaler saved) │
  │                                                            │
  └── 20% Held-Out Test Set (N = 200) ─────────────────────────┘
        │
        ▼ (Strictly evaluated ONCE with frozen pipeline)
      Reporting Metrics (ROC-AUC, Sensitivity, Specificity)
```

---

## 5. Quantum Hardware & Computational Profiling

In addition to classification metrics, the platform profiles quantum execution metrics to verify NISQ readiness:

| Metric | Definition | Practical NISQ Target |
|---|---|---|
| **Qubit Register Size ($n$)** | Number of active qubits utilized | $n \in [4, 16]$ |
| **Circuit Depth** | Longest path of dependent quantum gates | Depth $\le 50$ gates |
| **2-Qubit Gate Count (CNOT/CZ)** | Number of multi-qubit entangling gates | CNOT count $\le 40$ |
| **Shot Noise Sensitivity** | Performance drop when transitioning from statevector to finite shots ($shots \in [500, 8192]$) | $\Delta \text{AUC} < 0.03$ |
| **Execution Latency** | Time per sample inference | $< 100\text{ ms}$ (Simulator) / $< 5\text{ s}$ (QPU Queue) |
