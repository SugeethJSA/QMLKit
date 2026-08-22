# QMLKit Integrated System Upgrade — Overview

**Date:** 2026-08-23
**Status:** Approved — implementation in progress
**Reference methodology:** `C:\Users\sugee\Github\repomono` (GAIT Mood Detection monorepo)

---

## 1. Mission

Extend QMLKit from a VOC-benchmark-only codebase into a complete, packagable
screening platform:

```
[Dog + Collar MPU6050]──┐
                        │  ~100 Hz telemetry (TCP JSON @ :3333)
[Kennel: FSR×4, IR×6,   ├──────────────► FastAPI Ingest Server ──► Feature Extractor ──► ML Models
 Ultrasonic×2]──────────┘                (src/qmlkit)            (single source     (RF baseline →
                                                                  of truth)          QSVM/VQC path)
                                                                                          │
                                                              Next.js Live Dashboard ◄────┘
                                                              (dashboard/diagnostics/collect)
                                                                                          │
                                                              PyInstaller desktop bundle ◄┘
```

## 2. Confirmed hardware decisions

| Decision | Choice |
|---|---|
| IR placement | 4 near bottom corners (with FSRs) + 2 top front left/right |
| Transport | ESP32 runs TCP JSON server on port **3333**; Python connects (repomono GAIT contract) |
| Temperature | **No dedicated sensor** — MPU6050 die-temp logged as bonus field only |
| MPU6050 mount | **Dog collar/harness** (body-worn micro-movement capture) |

### Final sensor manifest

| Sensor | Count | Placement / role | Interface |
|---|---|---|---|
| FSR | 4 | Bottom kennel corners — weight distribution, posture, presence | ADC1 (GPIO 34/35/36/39) |
| IR | 6 | 4 lower corners (approach detection), 2 top-front L/R (head position) | Digital out (GPIO 19/18/17/16, 5/23) |
| Ultrasonic HC-SR04 | 2 | Bottom (floor/body distance), Top (head height) | TRIG/ECHO (26/27 bottom, 14/13 top) |
| MPU6050 | 1 | Dog collar — micro-movements while sniffing center sample | I²C (SDA 21, SCL 22) |

## 3. Workstreams

| # | Document | Scope |
|---|---|---|
| A | [01_qmlkit_code_fixes.md](01_qmlkit_code_fixes.md) | Bug fixes from code evaluation + Lung VOC & dog-ECG dataset integration + reports |
| B | [02_esp32_firmware.md](02_esp32_firmware.md) | `firmware/kennel_node` — Arduino/PlatformIO firmware |
| C | [03_server_ingestion.md](03_server_ingestion.md) | `src/qmlkit/hardware/` ingest modules + FastAPI endpoints |
| D | [04_nextjs_frontend.md](04_nextjs_frontend.md) | `frontend/` Next.js 16 live dashboard |
| E | [05_packaging.md](05_packaging.md) | pnpm workspace scripts, PyInstaller bundle, PS1 build scripts |
| F | [06_execution_checklist.md](06_execution_checklist.md) | Execution order + verification gates |

## 4. Environment facts (verified)

- Node v26.7.0 + pnpm 11.22.0 available.
- No PlatformIO / arduino-cli installed → firmware ships flash-ready; validated by static review only.
- Python 3.14 environment with full QMLKit deps; baseline pytest = **13 passed**.
- Repo state at `efb9075`, clean tree, up to date with origin/main.

## 5. Honesty constraints

- No labelled kennel data exists yet → server ships with simulation source +
  Data-Lab collection tooling; models report `"uncertain"` until trained.
- Kennel modality (movement biomarkers) is separate from the VOC chemical
  modality; both share the same benchmark/metrics layer.
