# 06. Explainability and Biomarker Attribution: From Quantum States to Chemical Diagnosis

## 1. The Clinical Need for Explainable Quantum AI (XQAI)

In healthcare and oncology, "black-box" predictions are unacceptable for clinical adoption. An oncologist requires actionable biochemical explanations:
- **Which specific chemical compounds (VOCs)** triggered the high-risk cancer classification?
- **Are the attributed compounds biologically consistent** with known oncological dysregulations (e.g. lipid peroxidation aldehydes, altered ketone metabolism)?
- **What is the confidence and robustness** of the quantum decision boundary?

This document outlines QMLKit's **Explainable Quantum AI (XQAI)** engine, which bridges abstract quantum Hilbert space states back to interpretable physical volatile biomarkers.

```
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  QMLKIT EXPLAINABILITY PIPELINE                                  |
|                                                                                                  |
|   1. Quantum Feature Space Decision   ──> [Quantum Kernel SHAP / Parameter Saliency Gradients]   |
|                                                               │                                  |
|                                                               ▼                                  |
|   2. Latent Component Attribution    ──> [Shapley Values on Quantum Feature Map Latent Axes]     |
|                                                               │                                  |
|                                                               ▼                                  |
|   3. Reverse Biochemical Projection  ──> [Transpose Mapping via PCA / Autoencoder Decoders]      |
|                                                               │                                  |
|                                                               ▼                                  |
|   4. Chemical Compound Ranking       ──> [Hexanal: +38%, Benzaldehyde: +24%, 2-Butanone: +19%]   |
|                                                               │                                  |
|                                                               ▼                                  |
|   5. Clinical Diagnostic Report      ──> [Radar Fingerprints + Biomarker Pathway Visualizations] |
+──────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Quantum Kernel SHAP (Shapley Additive Explanations)

Shapley values allocate the payout of a cooperative game among players based on their marginal contributions. In QMLKit, the "players" are input sensor/chemical features $i \in \{1, \dots, D\}$, and the "game" is the quantum model prediction $f_{\text{quantum}}(\mathbf{x})$.

### 2.1 Formulation for Quantum Classifiers
For a quantum prediction function $f(\mathbf{x}) = \sum_{j=1}^N \alpha_j y_j K(\mathbf{x}_j, \mathbf{x}) + b$, the Shapley attribution $\phi_i$ for feature $i$ is:
$$\phi_i(f, \mathbf{x}) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
where:
- $F$ is the total set of input features.
- $S$ is a subset of active features.
- $f(S)$ evaluates the quantum circuit expectation value when features outside $S$ are masked with baseline reference values (e.g., zero-air ambient background).

### 2.2 Quantum Kernel Matrix Perturbation
To compute marginal contributions efficiently without re-simulating the entire quantum circuit for all $2^{|F|}$ subsets, QMLKit implements **Quantum Kernel Perturbation Approximation**:
$$\Delta K_{i}(\mathbf{x}, \mathbf{x}_j) = \left| \langle \Phi(\mathbf{x} + \epsilon \mathbf{e}_i) | \Phi(\mathbf{x}_j) \rangle \right|^2 - \left| \langle \Phi(\mathbf{x}) | \Phi(\mathbf{x}_j) \rangle \right|^2$$

---

## 3. Quantum Fisher Information Matrix (QFIM) & Parameter Saliency

For Variational Quantum Classifiers (VQC), model sensitivity is evaluated using the Quantum Fisher Information Matrix (QFIM), which quantifies the geometry of the parameterized quantum state space:

$$\mathcal{F}_{k, l}(\boldsymbol{\theta}) = 4 \, \text{Re} \left[ \langle \partial_k \psi(\boldsymbol{\theta}) | \partial_l \psi(\boldsymbol{\theta}) \rangle - \langle \partial_k \psi(\boldsymbol{\theta}) | \psi(\boldsymbol{\theta}) \rangle \langle \psi(\boldsymbol{\theta}) | \partial_l \psi(\boldsymbol{\theta}) \rangle \right]$$

### 3.1 QFIM Diagnostics
- **Expressibility vs. Overfitting:** Monitors the spectrum of eigenvalues of $\mathcal{F}$. A well-conditioned QFIM ensures that the quantum circuit learns rich chemical representations without entering barren plateaus.
- **Quantum Parameter Saliency:** Identifies which variational ansatz parameters and quantum entangling gates contribute most directly to separating cancer subtypes.

---

## 4. Reverse Biomarker Mapping: From Latent Quantum Space to Chemical Compounds

Because quantum circuits operate on reduced latent registers ($n \in [4, 16]$ qubits) derived from classical dimensionality reduction (PCA or Autoencoders), QMLKit maps latent feature attributions $\boldsymbol{\phi}_{\text{latent}} \in \mathbb{R}^n$ back to the original physical $D$-dimensional sensor / chemical space:

### 4.1 Linear Transformation Inversion (PCA / Kernel PCA)
Given the orthogonal projection matrix $\mathbf{V} \in \mathbb{R}^{D \times n}$ where $\mathbf{z} = \mathbf{V}^T (\mathbf{x} - \boldsymbol{\mu})$:
$$\boldsymbol{\phi}_{\text{chemical}} = \mathbf{V} \boldsymbol{\phi}_{\text{latent}}$$
where $\boldsymbol{\phi}_{\text{chemical}} \in \mathbb{R}^D$ yields the individual contribution score of each physical VOC analyte (e.g. Hexanal, Heptanal, Acetone).

### 4.2 Non-Linear Autoencoder Decoding
When using a deep autoencoder encoder $E(\mathbf{x}) = \mathbf{z}$ and decoder $D(\mathbf{z}) = \hat{\mathbf{x}}$:
$$\boldsymbol{\phi}_{\text{chemical}} = \mathbf{J}_{D}(\mathbf{z}) \boldsymbol{\phi}_{\text{latent}}$$
where $\mathbf{J}_D(\mathbf{z}) = \frac{\partial D(\mathbf{z})}{\partial \mathbf{z}} \in \mathbb{R}^{D \times n}$ is the Jacobian matrix of the decoder evaluated at the patient's latent state.

---

## 5. Clinical Explainability Visualizations

The platform generates three clinical explainability artifacts for each diagnostic screening:

```
1. VOC Biomarker Waterfall Plot
   --------------------------------------------------------------
   Hexanal (Lipid Peroxidation)       |=====================>  +0.38
   Benzaldehyde (Aromatic Aldehyde)   |===============>        +0.24
   2-Butanone (Ketone Metabolism)     |==========>             +0.19
   DMDS (Sulfur Cleavage)             |=====>                  +0.08
   Isoprene (Depleted Control)        |<=====                  -0.11
   --------------------------------------------------------------
   Baseline Risk: 0.12  ───>  Final Diagnostic Probability: 0.88 (Cancer Positive)

2. Multi-Analyte Radar Fingerprint
   A circular radar chart overlaying the patient's chemical fingerprint against the reference "Healthy Cohort Range" and "Cancer Type Signature".

3. Quantum Kernel Similarity Graph
   Displays the patient's topological nearest neighbors in the quantum Hilbert space, showing which verified historical cancer/healthy patients share the closest quantum entanglement state.
```
