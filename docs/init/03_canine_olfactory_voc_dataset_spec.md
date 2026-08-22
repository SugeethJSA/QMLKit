# 03. Canine Olfactory VOC Dataset Specification

## 1. Biomimetic Chemical Profiling Overview

Canines discriminate cancer by sensing patterns of **Volatile Organic Compounds (VOCs)** in the headspace of exhaled breath, urine, blood, or tissue samples. These VOCs are byproducts of altered cellular metabolism (e.g. lipid peroxidation, aerobic glycolysis, mutated enzymatic pathways).

This document specifies:
1. The **Chemical Taxonomy of VOC Biomarkers** across major target cancers.
2. The **Biomimetic Sensor Array Architecture** (16 to 32 sensor channels).
3. The **Dynamic Sensor Response Simulation Model** (adsorption/desorption physics).
4. The **Dataset Schema & Data Synthesis Specification** for benchmarking.

---

## 2. Chemical Taxonomy of Cancer VOC Biomarkers

The platform models 24 key volatile organic compounds across four primary biochemical functional classes:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CANCER VOC BIOMARKER MATRIX                                      │
├───────────────────┬────────────────────────────┬────────────────────────────┬────────────────────┤
│ Chemical Class    │ Key Compounds              │ Biological Origin / Pathway│ Target Cancers     │
├───────────────────┼────────────────────────────┼────────────────────────────┼────────────────────┤
│ **Aldehydes**     │ Hexanal, Heptanal,         │ Lipid peroxidation of cell │ Lung Cancer,       │
│                   │ Octanal, Nonanal,          │ membrane polyunsaturated   │ Breast Cancer,     │
│                   │ Benzaldehyde, Decanal      │ fatty acids (PUFAs)        │ Colorectal Cancer  │
├───────────────────┼────────────────────────────┼────────────────────────────┼────────────────────┤
│ **Ketones**       │ Acetone, 2-Butanone,       │ Altered mitochondrial      │ Lung Cancer,       │
│                   │ 2-Pentanone, 3-Octanone,   │ $\beta$-oxidation & ketone │ Ovarian Cancer,    │
│                   │ Acetophenone, Cyclohexanone│ body metabolism            │ Prostate Cancer    │
├───────────────────┼────────────────────────────┼────────────────────────────┼────────────────────┤
│ **Aromatic        │ Ethylbenzene, Styrene,     │ Altered cytochrome P450    │ Lung Cancer,       │
│ Hydrocarbons**    │ Toluene, o-Xylene,         │ enzyme clearance and       │ Colorectal,        │
│                   │ 1,2,4-Trimethylbenzene     │ environmental sequestration│ Breast Cancer      │
├───────────────────┼────────────────────────────┼────────────────────────────┼────────────────────┤
│ **Alkanes /       │ Isoprene, Octane,          │ Cholesterol biosynthesis   │ Lung Cancer,       │
│ Terpenes / Other**│ Decane, D-Limonene,        │ intermediate & oxidative   │ Colorectal Cancer, │
│                   │ Dimethyl disulfide (DMDS)  │ stress cleavage products   │ Melanoma           │
└───────────────────┴────────────────────────────┴────────────────────────────┴────────────────────┤
```

### 2.1 Cancer-Specific VOC Signatures

| Cancer Indication | Signature Pattern | Typical Concentrations in Headspace |
|---|---|---|
| **Non-Small Cell Lung Cancer (NSCLC)** | Markedly elevated Hexanal, Heptanal, Benzaldehyde; depleted Isoprene; elevated Ethylbenzene | $0.5 - 25.0\text{ ppb}$ |
| **Breast Cancer** | Elevated Heptanal, Octanal, 2-Pentanone, 3-Octanone, 1,2,4-Trimethylbenzene | $0.2 - 15.0\text{ ppb}$ |
| **Colorectal Cancer (CRC)** | High Dimethyl disulfide (DMDS), Benzaldehyde, Ethylbenzene, Cyclohexanone | $1.0 - 50.0\text{ ppb}$ |
| **Prostate Cancer (Urine VOCs)** | Elevated 2-Butanone, 2-Pentanone, Dimethyl sulfide, Toluene | $0.8 - 35.0\text{ ppb}$ |
| **Ovarian Cancer** | Specific ratios of Nonanal, Decanal, Acetophenone, 2-Hexanone | $0.1 - 10.0\text{ ppb}$ |
| **Healthy Control** | Baseline physiological homeostasis: low aldehydes, stable acetone & isoprene | Variable baseline |

---

## 3. Biomimetic Sensor Array Model

To mimic the canine olfactory epithelium, the simulated and physical platform utilizes an array of **$S = 16$ cross-reactive sensor elements** (e.g. Metal-Oxide Semiconductor, Quartz Crystal Microbalance, and Functionalized Carbon Nanotube receptors).

### 3.1 Sensor Affinity Matrix ($A \in \mathbb{R}^{S \times C}$)
Each sensor channel $s \in \{1, \dots, 16\}$ possesses a distinct affinity vector across all chemical compounds $c \in \{1, \dots, 24\}$:
$$A_{s, c} = \kappa_s \cdot \exp\left(-\frac{\|\mathbf{p}_c - \mathbf{q}_s\|^2}{2 \sigma_s^2}\right)$$
where $\mathbf{p}_c$ represents the physicochemical descriptor vector of VOC $c$ (molecular weight, dipole moment, polarizability, boiling point), and $\mathbf{q}_s$ represents the sensor's functional surface coating properties.

### 3.2 Sensor Transient Dynamic Kinetics
When exposed to sample gas at time $t_0$, the fractional conductance response $G_s(t) = \frac{\Delta R_s(t)}{R_{0, s}}$ follows a Langmuir-adsorption dynamic differential equation:
$$\frac{d G_s(t)}{dt} = k_{\text{ads}, s} \left( \sum_{c=1}^C A_{s, c} C_c \right) (G_{\max, s} - G_s(t)) - k_{\text{des}, s} G_s(t) + \xi_s(t)$$
where:
- $C_c$ is the concentration of VOC compound $c$ in the sample.
- $k_{\text{ads}, s}$ and $k_{\text{des}, s}$ are adsorption and desorption rate constants.
- $\xi_s(t) \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$ is physical sensor noise (1/f flicker noise + Johnson thermal noise).

```
  Conductance
  Response G(t)
       ▲
       │                  ┌───────────────┐ Peak Steady-State: G_max
       │                .'                 `.
       │              .'                     `.  Desorption Decay
       │            .'                         `.
       │  Adsorption                             `.
       │  Rise Phase                               `..
       │                                              `─────── Baseline
       └──────────────────┬───────────────────┬────────────────────► Time (s)
                         t_on                t_off
```

---

## 4. Feature Extraction & Dataset Schema

For each sample, the sensor array produces a multi-channel time-series tensor $\mathbf{T} \in \mathbb{R}^{S \times T_{\text{steps}}}$.

### 4.1 Extracted Feature Vector $\mathbf{x} \in \mathbb{R}^{D}$
From each sensor $s \in [1, 16]$, 4 kinetic features are extracted ($16 \times 4 = 64$ features total):
1. **Steady-State Maximum Amplitude:** $\Delta G_{s, \max} = \max_t G_s(t)$
2. **Integral Response Area:** $\text{AUC}_s = \int_{t_{\text{on}}}^{t_{\text{off}}} G_s(t) \, dt$
3. **Maximum Adsorption Rate:** $\left.\frac{dG_s}{dt}\right|_{\max}$
4. **Desorption Decay Half-Life:** $t_{1/2, s}$

### 4.2 Tabular Data Schema

| Field Name | Type | Unit / Format | Description |
|---|---|---|---|
| `sample_id` | String | `SMPL_XXXXXX` | Unique anonymized identifier |
| `patient_age` | Integer | Years | Patient demographic |
| `patient_sex` | Categorical | `M` / `F` | Biological sex |
| `smoking_status` | Categorical | `Never` / `Former` / `Current` | Clinical confounder |
| `sample_matrix` | Categorical | `Exhaled_Breath` / `Urine_Headspace` | Biofluid specimen type |
| `label_cancer_type` | Categorical | `Healthy`, `Lung_Cancer`, `Breast_Cancer`, `Colorectal_Cancer`, `Prostate_Cancer`, `Ovarian_Cancer` | Ground truth diagnosis |
| `label_stage` | Categorical | `Stage_I`, `Stage_II`, `Stage_III`, `Stage_IV`, `Control` | Tumor staging |
| `sensor_01_max` ... `sensor_16_max` | Float | a.u. (normalized) | Peak response amplitude |
| `sensor_01_auc` ... `sensor_16_auc` | Float | a.u. $\cdot$ s | Integrated area under curve |
| `sensor_01_rise` ... `sensor_16_rise` | Float | $\text{s}^{-1}$ | Maximum adsorption slope |
| `sensor_01_decay` ... `sensor_16_decay` | Float | s | Exponential desorption time constant |
| `true_voc_hexanal_ppb` ... (24 compounds) | Float | ppb | Ground truth simulated VOC concentrations |

---

## 5. Benchmark Synthesis Engine Specifications

The synthetic benchmark generator (`src/qmlkit/data/biomimetic_voc_generator.py`) generates chemically grounded, physiologically calibrated benchmark cohorts:

1. **Configurable Sample Sizes:** Default $N = 500 - 2000$ subjects with configurable class balance (e.g. 50% Healthy, 50% Cancer, or multi-class).
2. **Biologically Realistic Covariance:** Compound concentrations are drawn from multivariate log-normal distributions $\ln(\mathbf{C}) \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ where $\boldsymbol{\Sigma}_k$ encodes known biochemical pathway correlations.
3. **Controlled Noise Injection:**
   - Additive Gaussian White Noise ($SNR \in [5\text{ dB}, 30\text{ dB}]$).
   - Baseline Drift (low-frequency sinusoidal + linear drift).
   - Cross-sensor calibration offsets ($\pm 5\%$).
4. **Clinical Confounders:** Includes realistic smoking status and age-correlated background volatiles to test model robustness against confounding artifacts.
