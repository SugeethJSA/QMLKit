# 18 — Review Checklist

A reviewer's working checklist, with the specific evidence to look for and the
questions that separate "read the code" from "verified the claims".

---

## 1. Reproduce first (30 min) — before reading any claim

Do these in order. If any fails, note it and continue; the failure itself is a
finding.

```bash
# 1. install
python -m pip install -e ".[dev]"
pnpm install

# 2. tests
python -m pytest -q
#    EXPECT: 59 passed
#    NOTE: docs/reports/code_evaluation_report.md:46 says "40 passed" — stale

# 3. lint
python -m ruff check src scripts tests packaging qmlkit_desktop.py
#    EXPECT: All checks passed

# 4. frontend
pnpm --filter frontend lint
pnpm --filter frontend build
#    EXPECT: clean; static export to frontend/out/

# 5. boot
pnpm dev
curl -s http://localhost:8000/api/v1/kennel/state
#    EXPECT: {"stream_source":"simulation","connected":true,
#             "model_status":"untrained", "batch_size":400, ...}
```

**Then confirm the negative claim:** the frontend footer claims "Qiskit"
(`frontend/app/layout.tsx:82`) and the README carries a Qiskit badge. There is no
Qiskit dependency and no Qiskit code. `grep -ri qiskit src/ frontend/package.json`
returns nothing.

---

## 2. Claim-to-evidence map

Every substantive claim the project makes, with the file that supports or
contradicts it. Review the right-hand column before accepting the claim.

| # | Claim | Supporting evidence | Check for |
|---|---|---|---|
| C1 | "Leak-free preprocessing" | `preprocessor.py:59-113`; `feature_selector.py:62-107`; `benchmark_suite.py:175-186`; `lab/pipeline.py:231-257`; `lab/cv.py:38,47` | Confirm **no** transform re-fits on test. Grep for `fit(` outside the train branch |
| C2 | "Correct fidelity-kernel math" | `qsvm.py:50-72` — $\|⟨\Phi(x)\|\Phi(x')\rangle\|^2$, `fill_diagonal(1.0)`, `clip(0,1)`, symmetrised | Confirm the *fidelity* (squared modulus), not the bare overlap. Bare overlap is generally not PSD |
| C3 | "Correlation-weighted feature map" | `feature_maps.py:41-71`; `benchmark_suite.py:100-153` | Confirm the covariance is from **raw** features, not PCA scores. `corrcoef(PCA)` ≈ identity ⇒ inert map |
| C4 | "Identical CV partitions across configurations" | `lab/cv.py:38` — `skf.split(np.zeros(len(y)), y_arr)` | Confirm the split input is independent of `X` |
| C5 | "Out-of-fold VQC augmentation" | `lab/pipeline.py:295-354` | Confirm `StratifiedKFold` and that `oof[va_idx]` is written, not `oof[tr_idx]` |
| C6 | "Leave-one-dog-out validation" | `train_kennel_model.py:109-121` | Confirm the test index is **all windows of one dog** and the train index is all others |
| C7 | "Real-data benchmark: classical 0.972–1.000, quantum 0.507–0.625" | `outputs/benchmark_real/benchmark_metrics.csv` | Recompute from the CSV. Note $n=24$ test |
| C8 | "Hybrid beats classical control" | `outputs/lab/20260823_082122/leaderboard.csv` — 0.974 vs 0.910 | **No paired test exists.** Fold sds 0.023 / 0.053 are the same order as the gap |
| C9 | "BioZZ outperforms ZZ" | same leaderboard — 0.658 vs 0.557 | ✅ holds |
| C10 | "BioZZ outperforms plain ZZ **because of correlation weighting**" | same leaderboard — permuted control **0.689** > 0.658 | ❌ **NOT supported.** The project reports this honestly |
| C11 | "NISQ-relevant circuit sizes" | `outputs/lab/20260823_082122/configs/CWZZ-QSVM__flagship_.json`; `hardware_profiler.py:35-85` | Verify depth 94 / gates 126 / CNOTs 60 by hand: $r(2+3P)=2(2+45)=94$ ✔. Check the Training Studio's *hardcoded* profiles (M1) — they say 24 |
| C12 | "45 kennel features, 55 trial features" | `kennel_features.py` (`N_KENNEL_FEATURES`, `N_TRIAL_FEATURES`) | Docs say 42 and ~36 (X2, X4). Verify at runtime |
| C13 | "13 sensors, 100 Hz IMU, 10 Hz telemetry" | `kennel_node.ino:41-62, 69-88` | Confirm `TX_INTERVAL_MS = 100`. Then check the feature extractor's `d=0.01` (D4) |
| C14 | "13 sensors" | README, firmware header comment | The firmware header comment **omits the MAX30102** (X8) |
| C15 | "59 tests" | `tests/` (8 files) | `docs/reports/code_evaluation_report.md:46` says 40 (X1) |
| C16 | "Ablation harness with a negative control" | `lab/experiments.py:174-200`; `lab/pipeline.py:55-66` | Confirm `cwzz_permuted` is a shipped preset and that it shuffles only the *assignment* |
| C17 | "Reproducible" | `config.py:14-21`; committed `_cli_meta.json` | **Caveat:** most quantum components carry a fixed internal seed 42, so `--seed` does not change their randomness (S7) |
| C18 | "Explainable" | `biomarker_mapper.py`; `quantum_shap.py` | On real data, compound *names* come from positional pooling (M-tier 3); non-PCA selectors return a uniform constant (M5) |
| C19 | "Desktop app for 3 platforms" | `packaging/`; `.github/workflows/ci.yml:365-487` | CI installs **core only** ⇒ no `xgboost`/`shap` in the released binary (C5) |
| C20 | "CI-validated" | `.github/workflows/ci.yml:9-14` | ❌ **No test/lint job on PRs.** Triggers only on a version bump (C4) |

---

## 3. Code review — high-value targets

Ordered by how much a careful read will teach you.

### 3.1 `evaluation/benchmark_suite.py:100-153` — `compute_qubit_covariance`

The most instructive function in the repository. Read it *before* the feature map
and you will understand the project's scientific integrity.

Questions to answer:
- Why is `np.corrcoef` on PCA scores useless? (orthogonal by construction)
- Why must the top-loading feature sets be made **disjoint**? (a shared feature
  correlates 1.0 with itself ⇒ the weight pins to 1 ⇒ the original degeneracy)
- What happens if the reduction is `"none"`? (no selector ⇒ `np.eye`)
- Is the output guaranteed symmetric, unit-diagonal, NaN-free? (yes, all three)

### 3.2 `lab/pipeline.py:295-354` — OOF VQC augmentation

- Confirm the source comment at `:296` names the defect it fixes: *"prevents
  in-sample optimism that inflated 100% scores"*.
- Note that the bare `except Exception` at `:320` **silently reverts to in-sample**
  when the OOF guard fails (M10).
- Note the fit/predict asymmetry: XGBoost trains on `X_scaled` + OOF probability
  but predicts with `X_scaled` + a full-data augmenter's probability (M11).

### 3.3 `data/feature_selector.py:125-159` — angle re-scaling

- Confirm the bounds come from `fit` and are **not** re-derived in `transform`.
- Confirm `np.clip(z_scaled, 0.0, 1.0)` before de-scaling — a test sample outside
  the training latent range is clamped, not extrapolated. Extrapolating into
  $[-\pi,\pi]$ would wrap angles and create a *different* sample.
- Confirm `safe_span = 1.0` for constant columns.
- Confirm `inverse_transform` raises `NotImplementedError` for unsupported methods
  (M-tier for explainability).

### 3.4 `quantum/qsvm.py:103-113` — adaptive calibration

- Confirm the fold count is derived from `min_class`, so a 4-positive fold cannot
  produce a zero-positive split.
- Note the trade-off: with `ensemble=False`, the SVC is fit on $(cv-1)/cv$ of the
  data and calibrated on the rest, and *that* is the model returned — correct, but
  slightly under-trained.

### 3.5 `quantum/vqc.py:99-136` — the training loop

- Confirm `y_signed ∈ {-1,+1}` matches the range of $\langle Z_0\rangle$.
- Confirm `n_batches = ceil(...)`, not `//` (finding #5's fix).
- Confirm `default_rng(42)` for the shuffle.
- Note the per-sample QNode loop inside `cost_fn` (`:108`) — a batched QNode would
  be substantially faster and is the obvious optimisation.
- Note `predict_proba` uses `sigmoid(2·raw)`, so probabilities are bounded in
  roughly $[0.1, 0.9]$ — visible in the reported Brier of 0.245.

### 3.6 `hardware/kennel_features.py` — the shared contract

- Confirm the same function serves training, serving, and the synthetic generator.
- Count to 45 and verify the `FEATURE_GROUPS` partition (13 + 11 + 18 + 3).
- **Find the `d=0.01` at `:85` and the `*100.0` at `:79`.** Then find
  `TX_INTERVAL_MS = 100` in the firmware. This is D4, the top fix item.

### 3.7 `api/kennel_server.py:163-224` — the threading model

- Trace the four steps of `predictor_worker` and the queue sizes
  (4,000 sync / 200 async).
- Confirm the non-overlapping window consumption at `:181-184` versus training's
  50% overlap (M15).
- Confirm the two-sided confidence gate at `:204` and the EMA at `:197-203`.
- Find `:449` and compare against `run_robustness`'s return dict. This is D1.

---

## 4. Statistical review

| Check | Question | Current state |
|---|---|---|
| Split discipline | Any test-fold statistic in the encoding? | No — verify §3.1, §3.3 |
| Fold identity | Are partitions identical across configurations? | Yes — verify `lab/cv.py:38` |
| Per-fold reuse | Is any fitted object shared across folds? | No — a fresh pipeline per fold |
| Metric honesty | Can `roc_auc` be something other than a ROC-AUC? | **Yes** — silent `balanced_accuracy` fallback (M2). Check `leaderboard.csv` for suspiciously equal `roc_auc_mean` / `balanced_accuracy_mean` |
| Dispersion statistic | Is the reported ± a sample or population sd? | Population (`ddof=0`, `lab/cv.py:88`) ⇒ ≈0.949× the sample sd (M7) |
| Significance | Is there a paired test on the headline gap? | **No** (M8). The per-fold data is committed, so this is recoverable |
| Class balance | Does the cap distort the operating point? | Yes — `balanced_subsample` forces 1:1 from a 55:45 cohort (S5) |
| Test-set size | Is $n=24$ sufficient for the reported AUCs? | Wilson 95% CI ≈ [0.86, 1.00] for 24/24. Read as "at ceiling" |
| Feature-set asymmetry | Do quantum and classical arms see the same features? | **No** — quantum sees 6 PCA dims, classical sees 27 scaled (M9) |
| Data provenance | Is the VOC file homogeneous? | **No** — precision changes mid-file; duplicate-looking rows (D7, D8) |
| Robustness | Does the noise sweep train clean and evaluate degraded? | Yes — the correct protocol |
| Seed sensitivity | Does a different seed change the result? | Partially — the split and classical models re-roll; **the quantum circuits do not** (S7) |

---

## 5. Reproducibility review

| Item | Status |
|---|---|
| Committed benchmark artifacts | ✅ `outputs/benchmark_real/` (CSV + PNG + metadata) |
| Committed lab artifacts | ✅ `outputs/lab/*/` with `run.json`, `leaderboard.csv`, **per-config per-fold metrics**, circuit profiles |
| CLI argument capture | ✅ `_cli_meta.json` per run |
| Environment pinning | ⚠️ minimum-version constraints only, no lock file for Python |
| JS lock file | ✅ `pnpm-lock.yaml` committed |
| Python lock file | ❌ none — `pyproject.toml` uses `>=` throughout |
| Seed recorded | ✅ in `run.json` metadata and `run_metadata.txt` |
| `requirements.txt` | ❌ contains the `httpx2` typo (C1) |
| `model` artifact for the kennel | ❌ `models/` does not exist |

---

## 6. Security review

| # | Item | Verdict |
|---|---|---|
| 1 | CORS `allow_origins=["*"]` | Acceptable for a localhost research tool; **must not** be deployed. No `allow_credentials`, so no cookie exposure |
| 2 | No authentication on any endpoint | Acceptable for a single-user local tool; a gap for a network service |
| 3 | `POST /api/v1/kennel/frame` reaches `source._enqueue()` | Unauthenticated frame injection; gate it in a shared deployment |
| 4 | Lab jobs unbounded and in-process | No concurrency cap, no eviction, no persistence ⇒ monotonic memory growth |
| 5 | `POST /train` and `/benchmark/run` are synchronous inside `async def` | Blocks the event loop for the full quantum fit (VQC 343.5 s ⇒ the console freezes) |
| 6 | `secrets.h` gitignored | ✅ correct; ship a `.example` (D3) |
| 7 | `allow_origins=["*"]` + `0.0.0.0` bind in `backend:dev` | `package.json:14` binds `--host 0.0.0.0`, so the wide-open CORS is reachable from the LAN during dev. The desktop app binds `127.0.0.1` — the difference is deliberate but worth noting |
| 8 | No dependency scanning / SBOM / signed artifacts | Minimal supply-chain posture |
| 9 | `firmware` credentials | Compile-time `#define`s in a gitignored header — standard practice |

---

## 7. Frontend review

| # | Item | Verdict |
|---|---|---|
| 1 | Footer claims "Qiskit" | ❌ **False claim** (D2). No Qiskit dependency or code |
| 2 | `/diagnostics` risk history seeded with a fabricated rising series | ❌ Could mislead (D5) |
| 3 | `/lab` not in the navigation | ⚠️ The lab produces the headline numbers; it should be reachable |
| 4 | `PredictionCard` colour threshold (one-sided 0.5) vs server `status` (two-sided 0.6) | ⚠️ A confident-healthy frame renders with the malignancy gradient (C21) |
| 5 | `/lab` robustness always errors | ⚠️ D1 |
| 6 | No `typecheck` script | ⚠️ Types verified only at `next build` (X14) |
| 7 | `?api=` override not re-read on change | ⚠️ Minor (C12-family) |
| 8 | No placeholder states in `SensorBars` / `KennelDiagram` / `ImuWaveforms` | ⚠️ Reads as broken, not loading (C20) |
| 9 | `Â·` mojibake in `TrainingStudio` | ⚠️ Visible (C16) |
| 10 | 6 dead CSS custom properties, `.panel-glow`, `.animate-pulse-glow` | ⚠️ Hygiene (C17) |
| 11 | `className="dark"` with no `dark:` variants | ⚠️ Vestigial (C18) |
| 12 | FPS interval re-created every frame | ⚠️ Wasteful (C19) |
| 13 | Two independent WebSocket connections (Dashboard + Diagnostics) | ⚠️ Inherent to the no-store design; each page re-derives stream state |
| 14 | `useTelemetrySocket` holds the callback in a `useRef` | ✅ **Correct** — the connection survives parent re-renders |
| 15 | Exponential reconnect backoff, capped at 8 s | ✅ **Correct** |
| 16 | `strict: true` in `tsconfig.json` | ✅ |
| 17 | Tailwind v4 CSS-first config (no `tailwind.config.js`) | ✅ Idiomatic v4 |

---

## 8. Documentation review

Full inventory in
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §5.
The 17 known drifts are:

| Doc | Says | Code |
|---|---|---|
| `code_evaluation_report.md:46` | 40 tests | 59 |
| `real_data_benchmark_report.md:66` | 42 features | 45 / 55 |
| `upgrade/03_server_ingestion.md:26` | ~36 features | 45 |
| `upgrade/03_server_ingestion.md:46` | `<dog>_<label>_<trial>.csv` | `<dog>_<label><trial>.csv` |
| `upgrade/04_nextjs_frontend.md:23-24` | `PathwayAttribution.tsx`, `ConnectionBadge.tsx` | Neither exists |
| `firmware/README.md:58-66` | NVS credentials, SoftAP, watchdog | None exist |
| `firmware/README.md:48-51` | example JSON | Omits `hr_bpm`, `spo2_pct` |
| `kennel_node.ino:6-11` | header sensor list | Omits the MAX30102 |
| `kennel_server.py:9` | `--port 8001` | 8000 |
| `README.md:22` | 40 tests | 59 |
| `README.md` | Qiskit framing | No Qiskit |
| `lab/pipeline.py:1-16` | `head` tuple | Omits `quantum_kernel_xgb` |
| `BenchmarkConfig.n_splits_cv` | exists | Never used in the benchmark path |
| `frontend/package.json` | — | No `typecheck` script |
| `qmlkit_desktop.spec:58` | logs to a file | No file handler (C7) |
| `hardware_profiler.py` docstring | "profiles match what is executed" | False for `training_service.py`'s hardcoded profiles (M1) |
| `firmware/kennel_node/.gitignore` | excludes `secrets.h` | ✅ correct, but no `.example` (D3) |

**Assessment: the project's own *reports* are honest** — the negative control and
the real-data result are reported plainly. The drift is concentrated in the
older `docs/init/`, `docs/upgrade/`, and `docs/workings/` design documents and
in the firmware README, which predate the code they describe.

---

## 9. Consolidated findings

Ordered by severity. Full detail in
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md).

### Must fix before presentation (S1)

| # | Finding | Location |
|---|---|---|
| D1 | Robustness lab arm always errors through the API (`KeyError: 'leaderboard'`) | `kennel_server.py:449` |
| D2 | Frontend footer claims Qiskit — no Qiskit exists | `layout.tsx:82` |
| D3 | `secrets.h` absent ⇒ **firmware does not compile** | `firmware/kennel_node/` |
| D4 | 100 Hz feature assumption vs 10 Hz wire rate ⇒ `acc_band_tremor_4_8hz` always 0 on live data; 10× jerk error; train/serve shift | `kennel_features.py:79,85` |
| D5 | `/diagnostics` risk chart pre-seeded with a fabricated positive trend | `page.tsx:44` |
| D6 | The correlation-weighting claim is not supported by its own negative control | `hybrid_search_report.md:39-43` |

### Materially weakens a claim (S2)

M1 hardcoded NISQ profiles understate depth ~4× · M2 silent `roc_auc` fallback ·
M3 silent hybrid-arm failure · M4 SHAP background is the test split · M5 uniform
constant for non-PCA selectors · M6 prose hardcodes the wrong pathway · M7
`ddof=0` · M8 no paired significance test · C1 `httpx2` typo · C2 import-time
network probe · C5 release build lacks xgboost/shap.

### Should fix before release (S3)

M9–M15, D7–D15, C3, C4, C6–C13, SEC1–SEC4.

### Hygiene (S4)

C14–C24, S9, S10, X1–X17.

---

## 10. Reviewer's summary template

```
Overall:            <strong / adequate / weak>
Methodology:        <leak-freeness, controls, statistics>
Engineering:        <tests, packaging, CI>
Documentation:      <accuracy, honesty>
Scientific claims:  <supported / partially / unsupported>

Strongest elements:
  1. Hybrid-leads-both-arms result under an identical 5-fold protocol (C8)
  2. Shipped negative control that disconfirmed the project's own headline (C10)
  3. Four independent leak-prevention mechanisms (C1, C4, C5, C6)
  4. Per-fold metrics + circuit profiles committed for every published number
  5. The covariance fix — a real high-severity defect, found and fixed

Weakest elements:
  1. Firmware never compiled; hardware path unexercised (D3)
  2. 100 Hz ↔ 10 Hz sampling mismatch breaks two frequency features (D4)
  3. No paired significance test on the headline gap (M8)
  4. No CI gate on pull requests (C4)
  5. Released binary runs different models than source mode (C5)
  6. Compound-level explainability is a positional heuristic on real data (M5, §9 of 09)

Blockers for any comparative claim:
  - D6 must be stated as a negative result, not a positive one
  - M8: either run the paired test or phrase the claim as non-significant
  - S5: disclose that the sample cap rebalances 55:45 to 1:1
```
