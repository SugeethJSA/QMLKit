# 04. Quantum Algorithms and Circuits: Mathematical Formulations and Quantum Architectures

## 1. Overview of Quantum Models in QMLKit

QMLKit implements four complementary Quantum Machine Learning paradigms optimized for noisy intermediate-scale quantum (NISQ) devices:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 QMLKIT QUANTUM ALGORITHM PORTFOLIO                               │
├─────────────────────┬──────────────────────────┬─────────────────────────┬───────────────────────┤
│ Model Family        │ Quantum Mechanism        │ Mathematical Objective  │ Primary Clinical Role │
├─────────────────────┼──────────────────────────┼─────────────────────────┼───────────────────────┤
│ **QSVM**            │ Quantum Kernel Feature   │ Quadratic programming   │ Trace-level early     │
│ (Quantum Kernel)    │ Space Embedding (ZZ/Bio) │ on Hilbert inner product│ cancer classification │
├─────────────────────┼──────────────────────────┼─────────────────────────┼───────────────────────┤
│ **VQC / QNN**       │ Parameterized Unitaries  │ Gradient descent on     │ Multi-class cancer    │
│ (Variational QNN)   │ + Entangling Ansätze     │ variational parameters  │ subtyping & staging   │
├─────────────────────┼──────────────────────────┼─────────────────────────┼───────────────────────┤
│ **QCNN / CQSV-Net** │ Quantum Convolution +    │ Hierarchical spatial /  │ Multi-channel sensor  │
│ (Quantum ConvNet)   │ Quantum Pooling          │ spectral feature filter │ array pattern mining  │
├─────────────────────┼──────────────────────────┼─────────────────────────┼───────────────────────┤
│ **QRC**             │ High-dimensional fixed   │ Classical linear readout│ Dynamic sensor        │
│ (Quantum Reservoir) │ entangled Hamiltonian    │ on quantum expectation  │ adsorption kinetics   │
└─────────────────────┴──────────────────────────┴─────────────────────────┴───────────────────────┘
```

---

## 2. Quantum Feature Maps & State Embeddings

Before quantum operations can occur, classical sensor feature vectors $\mathbf{x} \in \mathbb{R}^n$ (after dimensionality reduction to $n$ qubits, typically $n \in [4, 12]$) must be encoded into quantum states $|\Phi(\mathbf{x})\rangle = U_{\Phi}(\mathbf{x})|0\rangle^{\otimes n}$.

### 2.1 First-Order Angle Embedding
Applies independent single-qubit rotations along the Y-axis:
$$U_{\text{Angle}}(\mathbf{x}) = \bigotimes_{i=1}^n R_y(x_i), \quad R_y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$
*Property:* Fast, $O(1)$ circuit depth, but lacks multi-qubit entanglement to capture cross-analyte interactions.

### 2.2 Second-Order Pauli-Z Feature Map (ZZFeatureMap)
Encodes non-linear pairwise sensor interactions via controlled phase evolution:
$$U_{\Phi}(\mathbf{x}) = \mathcal{U}_{\Phi}(\mathbf{x}) H^{\otimes n} \mathcal{U}_{\Phi}(\mathbf{x}) H^{\otimes n}$$
where:
$$\mathcal{U}_{\Phi}(\mathbf{x}) = \exp\left( i \sum_{i=1}^n \phi_{\{i\}}(\mathbf{x}) Z_i + i \sum_{i < j}^n \phi_{\{i, j\}}(\mathbf{x}) Z_i Z_j \right)$$
$$\phi_{\{i\}}(\mathbf{x}) = x_i, \quad \phi_{\{i, j\}}(\mathbf{x}) = (\pi - x_i)(\pi - x_j)$$

```
  |0> ── H ── Rz(2*x_1) ──●──────────────●── H ── Rz(2*x_1) ──●──────────────●──
                          │              │                    │              │
  |0> ── H ── Rz(2*x_2) ──X── Rz(2*φ_12) ─X── H ── Rz(2*x_2) ──X── Rz(2*φ_12) ─X──
```

### 2.3 Biomimetic Covariance-Weighted Feature Map (`BioZZFeatureMap`)
To directly encode biochemical functional group correlations (e.g. aldehydes co-varying during lipid peroxidation), we introduce the domain-adapted feature map:
$$\phi_{\{i, j\}}^{\text{bio}}(\mathbf{x}) = C_{i, j} \cdot (\pi - x_i)(\pi - x_j)$$
where $C_{i, j} \in [-1, 1]$ is the empirical cross-correlation coefficient between latent chemical factors $i$ and $j$ computed from training cohorts. This heavily entangles biologically correlated channels while keeping unrelated channels separable.

---

## 3. Quantum Support Vector Machine (QSVM)

The QSVM evaluates similarities between patients in the $2^n$-dimensional quantum Hilbert space using the **Quantum Kernel Matrix**.

### 3.1 Quantum Kernel Computation
For any pair of patient samples $(\mathbf{x}_i, \mathbf{x}_j)$, the quantum kernel element is:
$$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \Phi(\mathbf{x}_i) | \Phi(\mathbf{x}_j) \rangle|^2 = \left| \langle 0^{\otimes n} | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0^{\otimes n} \rangle \right|^2$$

### 3.2 Transition Amplitude Measurement Circuit
The kernel value is measured directly on quantum hardware or simulators via fidelity testing:

```
  |0> ───[ U_Φ(x_i) ]───[ U_Φ(x_j)† ]─── Measure (Pauli-Z) ──> Probability of |00...0>
```
The probability of observing the all-zero state $|0\dots 0\rangle$ equals $|\langle \Phi(\mathbf{x}_j) | \Phi(\mathbf{x}_i) \rangle|^2 = K(\mathbf{x}_i, \mathbf{x}_j)$.

### 3.3 Mathematical Properties of the Quantum Kernel
1. **Symmetry:** $K(\mathbf{x}_i, \mathbf{x}_j) = K(\mathbf{x}_j, \mathbf{x}_i)$
2. **Unit Diagonal:** $K(\mathbf{x}_i, \mathbf{x}_i) = 1.0$
3. **Positive Semi-Definiteness (Mercer's Condition):** $\forall \mathbf{c} \in \mathbb{R}^N, \mathbf{c}^T \mathbf{K} \mathbf{c} \ge 0$.
4. **Dual Formulation:**
   $$\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) - \sum_{i=1}^N \alpha_i \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \sum_{i=1}^N \alpha_i y_i = 0$$

---

## 4. Variational Quantum Classifier (VQC) / Quantum Neural Network (QNN)

The VQC couples a quantum feature map $U_{\Phi}(\mathbf{x})$ with a parameterized variational ansatz $W(\boldsymbol{\theta})$, followed by expectation value measurements.

```
  |0> ──[ U_Φ(x) ]──[ W_1(θ_1) ]──●──[ W_2(θ_2) ]──●── Measure <Z_0>
  |0> ──[ U_Φ(x) ]──[ W_1(θ_1) ]──X──[ W_2(θ_2) ]──│── Measure <Z_1>
  |0> ──[ U_Φ(x) ]──[ W_1(θ_1) ]─────[ W_2(θ_2) ]──X── Measure <Z_2>
```

### 4.1 Variational Ansätze
1. **Strongly Entangling Layers (`StronglyEntanglingLayers`):**
   Composed of single-qubit rotations $R(\alpha, \beta, \gamma) = R_z(\gamma) R_y(\beta) R_z(\alpha)$ on every qubit, followed by cyclic CNOT entangling gates with configurable entangling range $r$.
2. **RealAmplitudes Ansatz:**
   Consists of alternating layers of $R_y(\theta)$ rotations and nearest-neighbor CNOT gates. It maintains real-valued state vectors, reducing computational complexity and barren plateau vulnerabilities.

### 4.2 Measurement & Loss Function
Expectation value of the Pauli-Z operator on qubit 0 (or a linear combination of all qubits):
$$\hat{y}(\mathbf{x}; \boldsymbol{\theta}) = \langle \Phi(\mathbf{x}) | W^\dagger(\boldsymbol{\theta}) Z_0 W(\boldsymbol{\theta}) | \Phi(\mathbf{x}) \rangle \in [-1, 1]$$
Probability of cancer:
$$p(\text{Cancer} \mid \mathbf{x}) = \sigma(w \cdot \hat{y}(\mathbf{x}; \boldsymbol{\theta}) + b) = \frac{1}{1 + e^{-(w \cdot \hat{y} + b)}}$$
Optimized via Binary Cross-Entropy (BCE) Loss using the **Parameter-Shift Rule** for exact quantum analytic gradients:
$$\frac{\partial \langle Z \rangle}{\partial \theta_k} = \frac{\langle Z(\theta_k + \frac{\pi}{2}) \rangle - \langle Z(\theta_k - \frac{\pi}{2}) \rangle}{2}$$

---

## 5. Quantum Convolutional Neural Networks (QCNN / CQSV-Net)

Inspired by convolutional architectures and CQSV-Net principles, QCNN applies translationally invariant multi-qubit unitary operations to extract hierarchical chemical patterns without exponential parameter scaling:

1. **Quantum Convolution Layer ($QConv$):** Applies 2-qubit parameterized unitary operators $U_{conv}(\boldsymbol{\theta})$ to neighboring pairs of qubits.
2. **Quantum Pooling Layer ($QPool$):** Reduces dimensionality by measuring a subset of qubits and conditionally rotating remaining qubits based on measurement outcomes, halving the register size.
3. **Barren Plateau Resilience:** Because QCNN uses $O(\log n)$ parameters and local cost functions, it is mathematically immune to the barren plateau phenomenon that impairs deep unstructured QNNs.

---

## 6. Quantum Reservoir Computing (QRC) for Dynamic Kinetics

For time-series sensor curves where temporal kinetics matter (adsorption slopes, desorption decay):

1. **Fixed Entangled Quantum Reservoir:** A multi-qubit system evolving under a fixed, randomly coupled transverse-field Ising Hamiltonian:
   $$\hat{H}_{\text{res}} = \sum_{i=1}^n h_i Z_i + \sum_{i < j} J_{ij} X_i X_j$$
2. **Sequential Input Injection:** At time step $t$, sensor readings are encoded via angle rotations $R_x(s(t))$ into the reservoir state.
3. **Quantum Dynamics & Readout:** The reservoir state evolves under $U_{\text{res}} = \exp(-i \hat{H}_{\text{res}} \Delta t)$. Expectation values $\{\langle Z_i(t) \rangle, \langle Z_i(t) Z_j(t) \rangle\}$ form a high-dimensional non-linear feature space fed into a fast classical Ridge Classifier.
