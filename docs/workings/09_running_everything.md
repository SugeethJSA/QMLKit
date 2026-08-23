# Running Everything — Cheat Sheet

All commands assume you are in the repo root. Python 3.11+, Node 18+, pnpm 9+.

## One-time setup

```bash
python -m pip install -e .[dev]      # backend + dev tools (pytest, ruff, httpx, shap)
pnpm install                         # frontend + task runner
```

## Daily driving

| I want to… | Command |
|---|---|
| Run backend + frontend together | `pnpm dev` |
| Backend only (kennel API :8000) | `pnpm backend:dev` |
| Frontend only (:3000) | `pnpm --filter frontend dev` |

## Benchmarks & experiments

```bash
# Quantum vs classical on the REAL VOC chemistry (single split, quick)
python scripts/run_benchmark.py --source markdown --voc-task cancer_vs_control \
    --max-samples 120 --vqc-epochs 6 --output-dir outputs/benchmark_real

# Hybrid Training Lab: the full curated recipe search (5-fold CV)
python scripts/run_hybrid_search.py --dataset voc_real --experiment search \
    --max-samples 120 --vqc-epochs 6

# Paper ablations / robustness
python scripts/run_hybrid_search.py --dataset voc_real --experiment map_ablation
python scripts/run_hybrid_search.py --dataset voc_real --experiment modality_ablation
python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness

# Synthetic-only legacy generator benchmark
python scripts/generate_voc_data.py && python scripts/run_benchmark.py   # defaults
```

Results land in `outputs/lab/<timestamp>/` and `outputs/benchmark*/`.

## Kennel ML loop (with real dogs)

1. Open the console → **Data Lab** → fill dog/sample/label → Start recording.
2. Repeat per sample; CSVs accumulate under `data/kennel/<label>/`.
3. Train: `python scripts/train_kennel_model.py`
4. Live diagnostics stream on the **Diagnostics** page; until a model exists they
   honestly report `untrained`.

## Firmware (ESP32)

```bash
cd firmware/kennel_node
pio run -e kennel -t upload        # PlatformIO path
```
(Arduino IDE: open `kennel_node.ino`, add Adafruit MPU6050 + SparkFun MAX3010x
libraries, board = ESP32 Dev Module.) Wiring tables:
[`firmware/kennel_node/README.md`](../../firmware/kennel_node/README.md).

## Desktop app packaging

```bash
pnpm desktop:build     # builds frontend export + PyInstaller bundle -> dist/
pnpm desktop:start     # launches packaged exe (falls back to python)
```

## Quality gates

```bash
python -m pytest -q                          # all tests
python -m ruff check src scripts tests       # lint
pnpm --filter frontend lint                  # UI lint
pnpm --filter frontend build                 # UI static export
```

## Where results live

| Path | Content |
|---|---|
| `outputs/lab/*/leaderboard.csv` | hybrid-search rankings |
| `outputs/benchmark_real/` | single-split quantum-vs-classical metrics |
| `data/ecg/` | curated dog ECG feature tables |
| `data/kennel/<label>/*.csv` | your recorded Data-Lab sessions |
| `models/kennel_model.joblib` | trained kennel model artifact |
| `docs/reports/*.md` | written-up analyses of each run family |
