# 16 — Limitations and Risk Register

**Mandatory reading for anyone signing off on QMLKit.** Every item is
traceable to a file and line. Severity reflects impact on the project's
*claims*, not on code aesthetics.

Legend: **S1** = blocks a claim · **S2** = materially weakens a claim ·
**S3** = should fix before release · **S4** = cosmetic / hygiene.

---

## 1. Defects that must be fixed before a presentation

| # | Sev | Location | Defect | Impact | Fix |
|---|---|---|---|---|---|
| **D1** | **S1** | `kennel_server.py:449` | `result["leaderboard"]` on the robustness branch, but `run_robustness` returns `{"run_dir", "levels"}` ⇒ `KeyError` caught by the broad `except` ⇒ the job **always** ends `status: "error"` even though the run succeeded on disk | The `/lab` page can never show a robustness result. Confirmed by `outputs/lab/20260823_085411/` having artifacts but no `_cli_meta.json`, while the CLI-fixed rerun `20260823_085700` has one | `result.get("leaderboard") or result.get("levels", [])`, mirroring the CLI fix at `run_hybrid_search.py:90` |
| **D2** | **S1** | `frontend/app/layout.tsx:82` | Footer renders "PennyLane · **Qiskit** · PyTorch · Scikit-Learn" | Qiskit is **not a dependency and has no code in the repository**. A reviewer who checks will find the claim false. Same issue as report finding #12 (README badge), still open | Remove "Qiskit" from the footer, or add a genuine Qiskit backend path |
| **D3** | **S1** | `firmware/kennel_node/secrets.h` | Absent and gitignored | **The sketch does not compile.** No hardware demo is possible until the user creates it | Ship `secrets.h.example`; document the two `#define`s in the firmware README |
| **D4** | **S1** | `kennel_features.py:79, 85`; `train_kennel_model.py:38` vs `kennel_node.ino` `TX_INTERVAL_MS = 100` | Features assume a **100 Hz** stream; the wire rate is **10 Hz** | `acc_dom_freq_hz` axis tops out at 5 Hz ⇒ **`acc_band_tremor_4_8hz` is always 0**; `acc_jerk_rms` is ~10× off and aliased; a 400-frame window is 40 s, not 4 s. Models trained on the committed 100 Hz CSVs will distribution-shift on live 10 Hz data | Derive `dt` from `median(diff(ts_ms))` inside `extract_window_features` — ~5 lines, removes the assumption entirely. See [`10_kennel_hardware.md`](10_kennel_hardware.md) §7.1 |
| **D5** | **S1** | `frontend/app/diagnostics/page.tsx:44` | `history` seeded with a **fabricated** rising series `[0.15 … 0.88]` | The "Risk History" chart shows a plausible positive trend **before any data arrives**. A reviewer glancing at it will see a result that does not exist | Remove the seed, or render it as visually-marked placeholder data |
| **D6** | **S1** | `hybrid_search_report.md:39-43` | The BioZZ correlation-weighting claim is **not supported** by the committed ablation | The project's headline scientific contribution is not demonstrated. It must not be presented as a positive result | Already reported honestly in the source report. **The presentation must repeat this framing**, not soften it |

---

## 2. Methodological weaknesses

| # | Sev | Location | Issue | Impact | Fix |
|---|---|---|---|---|---|
| M1 | **S2** | `evaluation/hardware_profiler.py` vs `training_service.py:123-208` | The Training Studio's "NISQ Compilation Profile" panel shows **hardcoded ansatz-only** depths (VQC 24, QCNN 18) that ignore the 94-deep feature map; the lab's own profiler computes 94 + 6 = 100 for the same circuit | The console **understates** the circuit depth by ~4×, on the panel specifically meant to convey NISQ feasibility | Route all branches through `QuantumHardwareProfiler.describe_model`. Detail in [`14_nisq_feasibility.md`](14_nisq_feasibility.md) §4.2 |
| M2 | **S2** | `benchmark_suite.py:71-80` | `roc_auc` silently falls back to `balanced_accuracy` when `roc_auc_score` raises; `brier` falls back to MSE when probabilities are absent | A reported `roc_auc` **may not be a ROC-AUC**, with no marker in the output | Add an `"auc_is_fallback": true` field, or `raise` instead |
| M3 | **S2** | `benchmark_suite.py:258-259` | The two hybrid arms are wrapped in `except Exception: pass` | If a hybrid arm fails, the leaderboard silently loses 2 rows with no record | Log the exception; emit a placeholder row with `status: "failed"` |
| M4 | **S2** | `server.py:87-91` | `QuantumKernelSHAP(background_data=X_te_q)` — the **held-out test** matrix is the SHAP background distribution | Test-set information leaks into the explanation's reference distribution; also a train/serve inconsistency | Pass the **training** reduced matrix |
| M5 | **S2** | `biomarker_mapper.py:85-88` | Non-PCA selectors return a **uniform constant** `1/n_comp` instead of raising | `mutual_info` or `autoencoder` produce a waterfall with 24 identical bars and no error | Scatter into a full-width zero vector, or raise |
| M6 | **S2** | `biomarker_mapper.py:137-141` | The summary text hardcodes lipid peroxidation as "the primary biochemical driver" regardless of the computed percentages | The prose contradicts the rendered percentages when a different pathway dominates | `max(pathways, key=pathways.get)` — one line |
| M7 | **S2** | `lab/cv.py:88` | `np.std(vals)` uses `ddof=0` (population sd) | For 5 folds this is ≈0.949× the sample sd; the ± values in every leaderboard understate the true spread | `ddof=1` |
| M8 | **S2** | `hybrid_search_report.md` §3 | No paired significance test on the hybrid-vs-classical gap (+0.064 AUC) | Fold sds are 0.023–0.053, the same order as the gap. The claim is suggestive, not demonstrated | The per-fold metrics are already in `configs/*.json`; a Wilcoxon signed-rank or paired bootstrap is a few lines away |
| M9 | **S3** | `benchmark_suite.py:262-275` | Classical baselines are fitted on the **27-dim** scaled matrix while quantum models see the **6-dim** projection | The comparison conflates algorithm choice with feature dimensionality | State it explicitly in any comparison, or add a PCA-6 classical control (the lab already has one) |
| M10 | **S3** | `lab/pipeline.py:320-330` | `except Exception` silently falls back from **OOF to in-sample** VQC augmentation | Reverts to exactly the leakage the OOF logic exists to prevent, with no marker in the run artifacts | Narrow the `except` to the guard's own `ValueError`, and record the fallback in `spec.meta` |
| M11 | **S3** | `lab/pipeline.py:332-335` vs `:396-406` | `quantum_augmented_xgb` trains XGBoost on `X_scaled` + OOF VQC probability, but at inference supplies the probability of an augmenter **refit on all training data** | The quantum feature's distribution differs between fit and predict (OOF is systematically less confident than in-sample) | Document the limitation in the class docstring; or ship the fold models and average at inference |
| M12 | **S3** | `server.py:174` vs `:168-174` | The default attribution path is `q_feat[0] - np.mean(q_feat)`, a **descriptive statistic, not a model attribution** | The API response does not say which method produced the attributions | Add `"attribution_method": "shap" \| "mean_delta"` to the response — one line |
| M13 | **S3** | `kennel_streaming.py:241` | The `auto` probe is **one-shot with no retry** | If the ESP32 boots slowly, the process stays in simulation for its lifetime | Retry loop, or surface a "board not reachable — retry?" affordance |
| M14 | **S3** | `kennel_streaming.py:100` | Fixed 2.0 s reconnect, **no exponential backoff** | A downed board produces a 0.5 Hz reconnect storm | `min(2 ** attempt, 30)` |
| M15 | **S3** | `kennel_server.py:185` | The inference window is **non-overlapping** (`del buffer[:400]`), while training uses 50% overlap (`STEP = 200`) | Train/serve mismatch in window placement; a hard discontinuity at each boundary | Slide by `STEP` instead of consuming the whole buffer |

---

## 3. Data and sampling issues

| # | Sev | Location | Issue | Impact |
|---|---|---|---|---|
| **D4** | **S1** | see §1 | 100 Hz ↔ 10 Hz sampling mismatch | see §1 |
| D7 | **S2** | `docs/Lung_Cancer_VOC_Dataset_427.md` | Precision changes mid-file (patients ≈1–88 at 2–3 dp, later rows full fp) — **a simulated augmentation appended to a real cohort** | Absolute performance numbers carry that caveat; the 1.000 AUCs may reflect the augmentation's separability |
| D8 | **S2** | same | Duplicate-looking rows (patients 21/45/46 share the first 14 compound values) | Potential optimistic bias; preserved as-is, deduplication left to the analyst |
| D9 | **S2** | `data/kennel/` | **1 dog, 2 labels, no `control/`.** `models/` absent | LOSO has no usable fold; `loso_accuracy_mean` would be `null`; diagnostics permanently `"untrained"` |
| D10 | **S3** | `kennel_node.ino:338-346` | SpO₂ uses an uncalibrated linear ratio model `constrain(110 − 25R, 70, 100)` | Not a clinical SpO₂ reading; should be labelled a relative oximetry index |
| D11 | **S3** | `kennel_node.ino:1058` | Gyro range ±500 °/s, DLPF 44 Hz | ~10× coarser than needed, and the 44 Hz filter attenuates the 2–8 Hz sniff/tremor band the features target |
| D12 | **S3** | `kennel_synth.py:53-57` vs `kennel_streaming.py:190-194` | The synthetic lab data uses noise σ 0.05/0.02 and `ir_top` p=0.8; the live simulator uses 0.02/0.01 and p=0.5 | Lab numbers and live-demo numbers are not from the same distribution — do not compare them directly |
| D13 | **S3** | `session_recording.py:51-55` vs `:99-119` | `sample_id` is captured at session start and **never written to disk** | The trial identifier is lost; only the directory-derived label survives |
| D14 | **S3** | `session_recording.py:77-120` | `stop_and_save` clears meta/rows **before** the empty-file check | A failed save also loses the session metadata |
| D15 | **S3** | `hardware/protocol.py:61-101` | Every parse failure returns `None` **silently** — no exception, no log, no counter | A systematically malformed stream is indistinguishable from an idle one | Add a `malformed_frames` counter to `/api/v1/kennel/state` — cheap, high-value observability |
| D16 | **S4** | `data/ecg/summary_stats.csv` | `hr_min_bpm_min = 0.62` bpm ⇒ some rows have corrupted/placeholder beat lists | Any ECG modelling must filter on quality metrics first |

---

## 4. Code correctness and maintainability

| # | Sev | Location | Issue | Impact | Fix |
|---|---|---|---|---|---|
| C1 | **S2** | `requirements.txt:47` | `httpx2` — a **typo** for `httpx>=0.27.0`, a different unrelated PyPI package | `pip install -r requirements.txt` installs the wrong package; the API e2e test would then fail | `httpx>=0.27.0`, or regenerate the file from `pyproject` |
| C2 | **S2** | `kennel_server.py:485` | `app = create_kennel_app()` at **module import**, and the factory performs a blocking 2.0 s TCP probe | `import qmlkit.api.kennel_server` performs network I/O | Construct the source in `lifespan`, not the factory |
| C3 | **S2** | `api/kennel_server.py` vs `api/server.py` vs `lab/pipeline.py` vs `classical/baselines.py` | **"The same model" is not the same estimator across code paths**: SVMs use `CalibratedClassifierCV` in `baselines.py` but the deprecated `SVC(probability=True)` in `lab/pipeline.py`; `Random_Forest` differs by `min_samples_split=4`; `MLP` differs by `activation`/`alpha` | Cross-path comparisons are indicative, not exact | Unify the estimator factory |
| C4 | **S2** | `.github/workflows/ci.yml:9-14` | CI triggers only on `version.properties` changes to `main` or `workflow_dispatch`. **No tests or lint run on any pull request** | Regressions can land through any gate | Add a `test` + `lint` job on `pull_request` |
| C5 | **S2** | `ci.yml:410` vs `pyproject.toml:35-42` | The build matrix runs `pip install -e .` (**core only**) ⇒ the released binaries have **no `xgboost` and no `shap`** | The packaged app's leaderboard uses `GradientBoosting` in place of XGBoost — **different numbers than source mode** | `pip install -e ".[dev]"` in the matrix, or a `desktop` extra with `xgboost` + `shap` + `pywebview` |
| C6 | **S2** | `packaging/qmlkit_desktop.spec:25-40` | `hiddenimports` omits `xgboost`, `shap`, `pywebview`; `pywebview` is **not a dependency at all** | The "native window" always degrades to `webbrowser.open` | Add a `desktop` extra and the three hidden imports |
| C7 | **S3** | `qmlkit_desktop.py:21, 58` | The spec comment claims logs go to `<app dir>/logs/qmlkit.log`, but `logging.basicConfig` writes to **stderr**, which is `None` in a windowed app | No logs from the packaged app | Add a `FileHandler` to `<app dir>/logs/qmlkit.log` |
| C8 | **S3** | `lab/pipeline.py:262` | The `qcnn` head hardcodes `feature_map_type="BioZZ"`, ignoring `spec.embedding` | The QCNN preset cannot be run with an alternative embedding; the ablation covers QSVM only | Use the same embedding→map dict as the other heads |
| C9 | **S3** | `lab/pipeline.py:1-16` | The docstring's `head` tuple omits `quantum_kernel_xgb` | Documentation drift on the primary config surface | Add it |
| C10 | **S3** | `run_hybrid_search.py:65-71` | `--n-components` overwrites **every** preset, including `Raw-XGBoost` which ships at 32 | Silently changes the classical control's dimensionality | Skip presets whose `n_components` was deliberately set |
| C11 | **S3** | `stacking.py:103` | `m.predict_proba(X)` is called **twice** per member per inference | 2× waste of kernel/simulation time on the most expensive arm (135.7 s) | Hoist into a local |
| C12 | **S3** | `feature_selector.py:109-122` | The autoencoder is trained full-batch for a fixed 40 epochs with no validation split and no early stopping | Least reproducible component; no signal if 40 epochs is wrong | Add a validation split and early stopping |
| C13 | **S3** | `qmlkit.config` | `QMLKitAppConfig` and friends are **never instantiated** | Config models are documentation, not configuration — a reader may assume they drive the system | Either wire them or label them explicitly as the design-space definition |
| C14 | **S4** | `kennel_server.py:9` | Docstring says `--port 8001`; every launcher uses 8000 | Stale |
| C15 | **S4** | `kennel_node.ino:75` | `CALIBRATION_MS = 2000` declared, never referenced; actual zeroing is ~160 ms | Dead constant implying a calibration duration that does not happen |
| C16 | **S4** | `TrainingStudio.tsx:434-435, 576` | `Â·` mojibake (double-encoded `·`) | Visible rendering defect |
| C17 | **S4** | `globals.css:3-14, 64-67, 126-133` | 6 dead CSS custom properties, plus `.panel-glow` and `.animate-pulse-glow` | Misleads contributors |
| C18 | **S4** | `layout.tsx:27` | `className="dark"` with no `dark:` variants anywhere | Vestigial |
| C19 | **S4** | `page.tsx:49-56` | The FPS `setInterval` is re-created on **every frame** (10×/s) | Works, but wasteful; interval identity churns |
| C20 | **S4** | `SensorBars.tsx`, `KennelDiagram.tsx`, `ImuWaveforms.tsx` | No placeholder states — zero bars / empty canvas / absent diagram before the first frame | Reads as broken rather than loading |
| C21 | **S4** | `PredictionCard.tsx:90` | The ring colour threshold is one-sided (`p ≥ 0.5`) while the server `status` is two-sided (`max(p,1−p) ≥ 0.6`) | A confident-healthy frame (`p=0.1`, `status:"ok"`) renders with the malignancy gradient |
| C22 | **S4** | Multiple pages | ~15 unused `lucide-react` imports | Lint noise |
| C23 | **S4** | `api/kennel_server.py.bak` | A 486-line **indented duplicate** of `kennel_server.py` is committed | Dead artifact; pure leading-whitespace diff |
| C24 | **S4** | `protocol.py` / `KennelFrame` | `us_bottom`/`us_top` are flat internally and re-nested in `to_dict()` | Minor impedance mismatch |

---

## 5. Documentation drift

Every entry is a place where a document and the code disagree. **The code is
ground truth.**

| # | Document | Says | Code does |
|---|---|---|---|
| X1 | `docs/reports/code_evaluation_report.md:46` | "40 passed" | **59** tests in the current tree |
| X2 | `docs/reports/real_data_benchmark_report.md:66` | feature extractor "(42 features)" | **45** window / **55** trial |
| X3 | `docs/upgrade/03_server_ingestion.md:46` | `<dog_id>_<label>_<trial>.csv` | `<dog_id>_<label><trial>.csv` (no underscore before the trial) |
| X4 | `docs/upgrade/03_server_ingestion.md:26` | "~36 features" | 45 |
| X5 | `docs/upgrade/04_nextjs_frontend.md:23-24` | `PathwayAttribution.tsx`, `ConnectionBadge.tsx` | Neither exists; actual components are the 8 listed in [`12_frontend_console.md`](12_frontend_console.md) §4 |
| X6 | `firmware/kennel_node/README.md:58-66` | NVS/`Preferences` credential storage; SoftAP `QMLKit-Kennel` / `sniff1234`; loop watchdog | **None of these exist.** `Preferences.h` is included but never used; no `WiFi.softAP` call; no watchdog |
| X7 | `firmware/kennel_node/README.md:48-51` | example JSON | omits `hr_bpm` / `spo2_pct`, which the firmware emits |
| X8 | `kennel_node.ino:6-11` | header comment lists FSR/IR/US/MPU6050 | the **MAX30102** is also fitted (declared `:174`, init `:1083-1099`) |
| X9 | `kennel_server.py:9` | `--port 8001` | 8000 everywhere |
| X10 | `README.md:22` | "40 tests" | 59 |
| X11 | `README.md` | Qiskit in the tech framing | no Qiskit dependency or code (report finding #12, still open) |
| X12 | `lab/pipeline.py:1-16` | `head` tuple | omits `quantum_kernel_xgb` |
| X13 | `backend/benchmark` config field `n_splits_cv` | exists in `BenchmarkConfig` | **never used** in the benchmark path |
| X14 | `frontend/package.json` | — | no `typecheck` script; types verified only by `next build` |
| X15 | `kennel_server.py` (module docstring) | — | the `robustness` lab branch is broken (D1) |
| X16 | `hardware_profiler.py` docstring | "profiles match what is actually executed" | true for `lab/`, **false** for the hardcoded `training_service.py` profiles (M1) |
| X17 | `qmlkit_desktop.spec:58` | "logs go to `<app dir>/logs/qmlkit.log`" | no file handler exists (C7) |

---

## 6. Statistical caveats

| # | Caveat | Consequence |
|---|---|---|
| S1 | **Single split, $n=24$ test set** in the real-data benchmark | The 1.000 AUCs have a Wilson 95% CI of ≈[0.86, 1.00]. Read as "at ceiling" |
| S2 | **No paired significance test** on the hybrid-vs-classical gap | The +0.064 AUC advantage has the same order of magnitude as the fold sd (0.023–0.053). Phrase as "consistent across folds under an identical protocol", not "significant" |
| S3 | **`ddof=0` fold sd** (`lab/cv.py:88`) | Every ± in the leaderboard is ≈0.949× the sample sd |
| S4 | **`roc_auc` can silently be `balanced_accuracy`** | Check for suspiciously equal `roc_auc_mean` and `balanced_accuracy_mean` columns in `leaderboard.csv` |
| S5 | **`balanced_subsample` fixes the class ratio at 1:1** | The real cohort is 193/157 (55:45); the cap rebalances it, so the reported metrics are not the cohort's natural operating point |
| S6 | **Real VOC precision drift** (D7) | Absolute numbers may reflect the appended augmentation's separability |
| S7 | **`--seed` does not change quantum internal randomness** | Most quantum components carry a fixed internal seed 42 (`vqc.py:86,103`; `qcnn.py:108`; `qsvm.py:108`; `feature_selector.py:68,71,76`). Repeating a run with different `--seed` re-rolls the split and the classical models, **not** the circuits |
| S8 | **No committed `modality_ablation` artifact** | RQ2 has code and tests but no published result |
| S9 | **No real shot-sampling experiment** | `evaluate_shot_noise_robustness` is an **analytic surrogate** (adds Gaussian noise to the exact kernel), and it is not called from any runner. The fresh `default_rng(42)` per shot level also makes the sweep non-monotonic |
| S10 | **Sample-ID leakage of fold assignment** | `Quantum_Kernel_XGB` selects landmarks stratified by `y`; the head then also sees `y`-derived probabilities. Under identical CV partitions this is valid, but it means the quantum feature is partly a smoothed label — worth stating in any interpretation |

---

## 7. Security and operational notes

| # | Sev | Item | Assessment |
|---|---|---|---|
| SEC1 | S3 | CORS `allow_origins=["*"]` on `kennel_server.py:245-250` | Acceptable for a localhost research tool (and the desktop app binds `127.0.0.1`), but must not be deployed. No `allow_credentials`, so no cookie exposure |
| SEC2 | S3 | `POST /api/v1/kennel/frame` reaches `source._enqueue()` via a private method | An unauthenticated frame-injection hook. Fine locally; gate it in any shared deployment |
| SEC3 | S3 | No authentication on **any** endpoint | Every model-training, recording, and lab-launch route is open. Acceptable for a single-user research tool, not for a network service |
| SEC4 | S3 | Lab jobs unbounded and in-process (`kennel_server.py:390-391`) | No concurrency cap, no eviction, no persistence ⇒ a repeated-launch loop grows memory monotonically and loses everything on restart |
| SEC5 | S4 | `firmware/kennel_node/secrets.h` holds WiFi credentials | Correctly gitignored. Ship a `.example` so the build is not blocked (D3) |
| SEC6 | S4 | `POST /api/v1/train` and `/api/v1/benchmark/run` are **synchronous** inside `async def` | Blocks the whole event loop for the duration of a full quantum fit (VQC 343.5 s ⇒ the console freezes). Should be background jobs like the lab routes |
| SEC7 | S4 | No dependency scanning, no SBOM, no signed artifacts | `ci.yml` uses pinned major versions of first-party actions; the Python/JS dependency trees are unconstrained above the minimum |

---

## 8. Prioritised fix list

**Before any presentation or review (S1):** D1, D2, D3, D4, D5, D6.

**Before any comparative claim (S2):** M1, M2, M3, M4, M7, M8, C1, C2, C5.

**Before a release (S3):** M9–M15, D7–D15, C3, C4, C6–C13, SEC1–SEC4.

**Hygiene (S4):** C14–C24, S9, S10, X1–X17.

### The five highest-value engineering changes

| Rank | Change | Effort | Payoff |
|---|---|---|---|
| 1 | **Derive `dt` from frame timestamps** in `extract_window_features` (D4) | ~5 lines | Makes every rate-dependent feature correct and removes the 100 Hz assumption entirely |
| 2 | **Cache train statevectors** in `QuantumKernel` | ~15 lines | Turns $O(N_{test}\cdot N_{train})$ inference into $O(N_{test})$; prerequisite for any hardware run and for the 2× double-computation waste |
| 3 | **Add a `test` + `lint` CI job on `pull_request`** (C4) | ~30 lines | Prevents all regressions; currently nothing validates a PR |
| 4 | **Paired significance test** over the committed per-fold metrics (M8) | ~20 lines | Converts the headline claim from "suggestive" to "demonstrated" — and the data is already committed |
| 5 | **`reps=1` sweep** (halves CNOTs 60→30, depth 94→47, crossing the profiler's own "EXCELLENT" threshold) | ~1 line | The single highest-leverage NISQ-feasibility change available |

---

## 9. What the project does *not* claim

Stated explicitly so nothing in this documentation set can be read as overclaiming:

- It does **not** run on quantum hardware. All circuits are executed on an exact
  classical statevector simulator.
- It does **not** demonstrate a quantum advantage over classical ML. On the real
  dataset classical baselines reach 0.972–1.000 ROC-AUC while the quantum arms
  reach 0.507–0.625.
- It does **not** demonstrate that correlation-weighting the feature map helps.
  The permuted negative control outperformed the flagship.
- It does **not** produce clinically valid biomarkers. The compound-level
  labelling on the real dataset is a positional heuristic; the SpO₂ channel is
  uncalibrated; the kennel model has never been trained on real data.
- It does **not** have hardware-in-the-loop validation. The ESP32 firmware has
  never been compiled or flashed in this repository's history.
- It does **not** have statistical significance testing on its comparative
  claims.
