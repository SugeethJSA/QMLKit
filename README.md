# QMLKit: Hybrid Quantum Machine Learning Platform for Early Disease Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PennyLane: v0.40+](https://img.shields.io/badge/PennyLane-v0.40%2B-teal.svg)](https://pennylane.ai/)
[![Status: Integrated Platform](https://img.shields.io/badge/Platform-Firmware%20%2B%20API%20%2B%20Console-brightgreen.svg)]()

> **Problem Statement ID:** 26139
> **Problem Statement Title:** Hybrid Quantum Machine Learning Platform for Early Disease Detection
> **Core Innovation:** Canine-biomimetic olfactory VOC sensing with NISQ-compatible quantum feature maps, QSVM / VQC classifiers, and an instrumented examination kennel capturing canine micro-movements during sniffing.

---

## 🌟 What QMLKit is

An integrated, packagable screening platform combining **three subsystems**:

1. **Kennel Node firmware** (`firmware/kennel_node`) — ESP32-WROOM telemetry for a canine examination kennel: 4× FSR corner load cells, 6× IR proximity (4 lower corners + 2 top-front), 2× ultrasonic, and a collar-mounted MPU6050 streaming ~100 Hz micro-movement data over TCP JSON.
2. **QML backend** (`src/qmlkit`) — leak-free benchmarking of quantum vs classical models on real VOC biomarker data (`docs/Lung_Cancer_VOC_Dataset_427.md`), plus live kennel ingestion (`src/qmlkit/hardware`) with windowed micro-movement features and Data-Lab session recording.
3. **Next.js Console** (`frontend`) — live dashboard (kennel diagram, IMU waveforms, sensor bars), diagnostics view with confidence handling, and a guided recording lab.

Packaging follows the **repomono/GAIT methodology**: pnpm workspace orchestration, PyInstaller onedir desktop bundle with embedded dashboard snapshot, PowerShell build scripts.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+, Node 18+ & pnpm 9+
- Optional hardware: ESP32 kennel node streaming on port 3333 (auto-fallback to simulation)

```bash
# 1. Backend deps
python -m pip install -e .[dev]

# 2. Everything at once (backend :8000 + frontend :3000)
pnpm install && pnpm dev

# 3. Or separately
python -m uvicorn --app-dir src qmlkit.api.kennel_server:app --port 8000   # kennel API
pnpm --filter frontend dev                                                  # console UI
```

The dashboard auto-detects the API host; override with `?api=http://host:port`.

### Benchmarks (real VOC data)

```bash
python scripts/run_benchmark.py --source markdown --voc-task cancer_vs_control \
    --max-samples 120 --vqc-epochs 8 --output-dir outputs/benchmark_real
```

Latest results and interpretation: [`docs/reports/real_data_benchmark_report.md`](docs/reports/real_data_benchmark_report.md).

### Kennel ML loop

```bash
# collect labelled sessions via the Data Lab (/collect in the console)
python scripts/train_kennel_model.py     # leave-one-dog-out validation -> models/kennel_model.joblib
```

Diagnostics stream reports `"untrained"` until an artifact exists.

---

## 📚 Documentation map

| Path | Content |
|---|---|
| `docs/init/01…07_*.md` | Original design corpus (problem statement → roadmap) |
| `docs/upgrade/*.md` | Integration plan for this release (firmware/server/frontend/packaging decisions) |
| `docs/reports/code_evaluation_report.md` | Code evaluation findings & fix disposition |
| `docs/reports/dataset_integration_report.md` | Real dataset schemas, mappings, caveats |
| `docs/reports/real_data_benchmark_report.md` | Quantum vs classical results on real VOC data |
| `firmware/kennel_node/README.md` | Wiring tables, state machine, flashing, telemetry protocol |

---

## 🏗️ Architectural flow

```
ESP32 Kennel Node ──TCP JSON :3333──► FastAPI Ingest ──► Windowed Features (single source of truth)
        ▲                                       │                        │
  4×FSR · 6×IR · 2×US                    simulation fallback          RF baseline (LOSO)
  · collar MPU6050                                     │                        ▼
                                              Next.js Console ◄──── diagnostics WS
```

## 🔬 Core technologies

- **Quantum:** PennyLane (BioZZ covariance-weighted ZZ feature map, QSVM fidelity kernels, VQC)
- **Classical:** scikit-learn, XGBoost, PyTorch (baselines + kennel RandomForest artifact via joblib)
- **Backend:** FastAPI, WebSockets, uvicorn
- **Frontend:** Next.js 16, React 19, TailwindCSS v4, TypeScript
- **Firmware:** Arduino framework / PlatformIO, ESP32-WROOM-32
- **QA:** pytest (40 tests), ruff, eslint
