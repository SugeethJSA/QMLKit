# Workstream F — Execution Order & Verification Checklist

## Order

1. **A-fixes:** finish `qcnn.py` (pool modulo, fit batching/seed), `vqc.py`,
   `preprocessor.py` import, `server.py` (split, lifespan, SHAP), `pyproject.toml`.
2. **B-firmware:** `firmware/kennel_node/` (.ino, platformio.ini, config.h, README).
3. **C-ingest:** `src/qmlkit/hardware/*`, kennel API routes, train CLI + tests.
4. **D-frontend:** scaffold + pages/components + lint/build.
5. **E-packaging:** root package.json/workspace/spec/PS1 scripts.
6. **A-datasets:** `dataset_loader.py` + tests; extend `run_benchmark.py`;
   dog-ECG ingestion script.
7. **Run** real-data Lung VOC benchmark → `outputs/benchmark_real/`.
8. **Reports** → `docs/reports/{code_evaluation,dataset_integration,
   real_data_benchmark}_report.md`.
9. **Root README** rewrite (three subsystems + quickstart).

## Gates

| Gate | Command | Expectation |
|---|---|---|
| Backend tests | `python -m pytest -q` | all green incl. new loader/hardware tests |
| Lint | `python -m ruff check src scripts tests packaging qmlkit_desktop.py` | clean |
| Frontend | `pnpm --filter frontend lint && pnpm --filter frontend build` | clean |
| Benchmark artifacts | `outputs/benchmark_real/benchmark_metrics.csv` | produced from 427 real patients |
| Firmware | static review vs docs/upgrade/02 pin map | consistent |

## Risks / notes

- QSVM kernel O(N²) on 427 samples (~182k circuits) — mitigated by
  `--max-samples` balanced subsampling; documented in benchmark report.
- No hardware in the loop during this pass: firmware validated statically;
  ingest verified against simulation source.
- Kennel model intentionally ships untrained (`uncertain` state) until Data-Lab
  sessions are collected on real dogs.
