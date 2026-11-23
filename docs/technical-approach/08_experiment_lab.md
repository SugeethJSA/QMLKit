# 08 — Experiment Lab

`src/qmlkit/lab/` + `scripts/run_hybrid_search.py` + `POST /api/v1/lab/runs`.
This is the subsystem that produces the project's comparative claims.

---

## 1. The protocol contract

`lab/experiments.py:1-9` states it:

> identical stratified partitions shared across configurations (§VII-B/F),
> fresh pipeline + correlation matrix per fold (§VI-C/VII-F), feature-map
> ablation incl. permuted control (§VII-D), sensor-modality ablation (§VII-D),
> noise + dropout robustness (§VII-E).

Four invariants make the numbers comparable:

| Invariant | Implementation | Why it matters |
|---|---|---|
| **Identical partitions across configs** | `cross_validate_config` splits on `np.zeros(len(y))`, not on `X` (`cv.py:38`) | a score difference is attributable to configuration, not fold luck |
| **Fresh pipeline per fold** | `pipeline = make_pipeline_fn(spec)` inside the fold loop (`cv.py:47`) | no scaler/selector/covariance/landmark state crosses a fold boundary |
| **Correlation matrix from training fold only** | `HybridPipeline.fit` calls `compute_qubit_covariance(X_scaled, self.selector)` on the fold's training data | no test-fold statistics leak into the encoding |
| **Seed fixed everywhere** | `StratifiedKFold(shuffle=True, random_state=seed)` + per-model `random_state` | reproducibility |

---

## 2. `cross_validate_config`

`lab/cv.py:29-77`.

```python
@dataclass
class FoldResult:
    fold: int
    metrics: Dict[str, float]
    train_time_s: float
    infer_time_ms_per_sample: float
```

```python
def cross_validate_config(spec, X, y, make_pipeline_fn, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr)):
        pipeline = make_pipeline_fn(spec)                       # FRESH
        t0 = time.time(); pipeline.fit(X_tr, y_tr)
        train_time = time.time() - t0
        proba = pipeline.predict_proba(X_val)
        infer_ms = (time.time() - t0) * 1000.0 / max(1, len(va_idx))
        preds = np.argmax(proba if proba.ndim == 2 else np.vstack([1-proba, proba]), axis=1)
        m = compute_clinical_metrics(spec.name, "Hybrid", y_val, preds, proba,
                                     train_time=train_time, infer_time=infer_ms/1000.0)
        folds.append(FoldResult(fold, {k: v for k, v in vars(m).items()
                                       if isinstance(v, (int, float))}, train_time, infer_ms))
    return folds
```

Notes:

- **Hardcoded `paradigm="Hybrid"`.** Every lab configuration is labelled Hybrid
  in its metrics record, including pure-quantum and pure-classical arms. The
  paradigm label in the lab leaderboard is therefore not informative.
- `infer_ms` is measured from the *start of fit*, so it is
  `train + inference` divided by validation size. The field is named
  `infer_time_ms_per_sample` but the value is a **combined** cost. When passed
  to `compute_clinical_metrics` it is divided by 1000 and re-labelled
  `inference_time_ms`, i.e. **seconds per sample** under that name.
- Non-numeric dataclass fields (`model_name`, `paradigm`) are filtered out before
  storage; the metrics dict holds the 10 numbers + 2 timings.

## 3. `summarise_folds`

`lab/cv.py:80-94`.

```python
keys = ["accuracy", "balanced_accuracy", "sensitivity_recall", "specificity",
        "precision_ppv", "negative_predictive_val", "f1_macro",
        "roc_auc", "brier_score"]
out = {"config": name, "n_folds": len(folds), "train_time_s_mean": mean(...)}
for k in keys:
    vals = [f.metrics[k] for f in folds if not np.isnan(f.metrics[k])]
    out[f"{k}_mean"] = np.mean(vals)
    out[f"{k}_std"]  = np.std(vals)          # POPULATION std, ddof=0
```

**`np.std` with default `ddof=0`** reports the standard deviation *of the observed
folds*, which is smaller than the sample standard deviation by a factor
$\sqrt{(n-1)/n}$ — for $n=5$ folds, $\approx 0.949$. Reporting fold-to-fold spread
as a confidence interval would need `ddof=1`; the current value is best read as
"consistency across folds", not a CI. Stated in
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §4.

---

## 4. `default_presets` — the 11 configurations

`lab/experiments.py:33-55`. `base = dict(n_components=6, vqc_epochs=vqc_epochs, seed=seed)`.

| # | Name | reduction | embedding | head | n_components |
|---|---|---|---|---|---|
| 1 | `CWZZ-QSVM (flagship)` | pca | cwzz | qsvm | 6 |
| 2 | `ZZ-QSVM` | pca | zz | qsvm | 6 |
| 3 | `Angle-QSVM` | pca | angle | qsvm | 6 |
| 4 | `CWZZ-permuted-QSVM (control)` | pca | cwzz_permuted | qsvm | 6 |
| 5 | `MI-CWZZ-QSVM` | mutual_info | cwzz | qsvm | 6 |
| 6 | `AE-VQC` | autoencoder | cwzz | vqc | 6 |
| 7 | `Quantum-Augmented-XGB` | pca | cwzz | `quantum_augmented_xgb` | 6 |
| 8 | `Quantum-Kernel-XGB (CG-ZZ+XGB)` | pca | cwzz | `quantum_kernel_xgb` | 6 |
| 9 | `QCNN` | pca | cwzz | qcnn | 6 |
| 10 | `PCA-XGBoost (classical control)` | pca | none | xgboost | 6 |
| 11 | `Raw-XGBoost (classical control)` | none | none | xgboost | **32** |

The set is deliberately constructed to contain a **classical control** (10, 11), a
**negative control** (4), and a **lower-bound feature map** (3) — so the headline
comparison is not "quantum vs nothing".

---

## 5. Ensembles

`ensemble_base_specs` (`:58-67`): `stack-rf` (none/none/random_forest, 32),
`stack-xgb` (none/none/xgboost, 32), `stack-qsvm` (pca/cwzz/qsvm, 6),
`stack-vqc` (pca/cwzz/vqc, 6).

### 5.1 `SoftVotingEnsemble` (`lab/stacking.py:18-43`)

Fits every member on the same `(X, y)`; `predict_proba` stacks `p[:, 1]` columns
and averages (or applies normalised weights via `np.tensordot`).

Soft-voting members in `run_hybrid_search` (`:115-157`):
`vote-svm-rbf` (none/none/svm_rbf, 8), `vote-rf` (…, random_forest, 8),
`vote-xgb` (…, xgboost, 8), `vote-qsvm` (pca/cwzz/qsvm, 6).

The `soft_vote_factory` **ignores the passed spec** and builds
`SoftVotingEnsemble([make_pipeline(s) for s in specs_list])` — the per-fold
evaluation therefore instantiates the full 4-member ensemble on each fold. The
wrapper is registered with `{"kind": "soft_voting"}`.

### 5.2 `StackingEnsemble` (`lab/stacking.py:46-110`)

```
1. _oof_matrix: StratifiedKFold(n_splits, shuffle=True, random_state=seed)
     for each fold × each base spec:
         fresh estimator via make_pipeline_fn, fit on X_tr
         oof[val_idx, j] = proba[:, 1]
2. refit every base member on the FULL training set (for inference)
3. LogisticRegression(max_iter=1000, random_state=seed) fit on the OOF matrix
```

This is textbook stacking and it is the **most leak-safe** configuration in the
project — the meta-learner never sees an in-sample base prediction.

**Known inefficiency** (`:103`): `m.predict_proba(X)` is called twice per member
inside the inference comprehension, a 2× waste of kernel/simulation time. This is
why the Stacked-LR arm costs 135.7 s.

---

## 6. Experiment types

### 6.1 `search` (default) — `run_hybrid_search` (`:89-171`)

```
specs = presets or default_presets(seed=seed)
for spec in specs:  registry.add_result(_evaluate_config(spec, ...))
if include_ensembles:  register SoftVote and Stacked-LR
registry.finalise(extra_meta={dataset_shape, class_counts, n_splits, seed, presets})
return {"run_dir", "leaderboard"}   # sorted by roc_auc_mean DESC
```

Each `_evaluate_config` (`:74-86`) also emits a best-effort NISQ profile:

```python
details["circuit_profile"] = HybridPipeline(spec).describe()   # inside try/except
```

### 6.2 `map_ablation` (RQ3) — `run_feature_map_ablation` (`:174-200`)

Four variants, all `reduction="pca"`, `head="qsvm"`, differing **only** in
`embedding`:

| Name | `embedding` | feature map |
|---|---|---|
| `Angle-QSVM` | `angle` | first-order RY, no entanglement |
| `ZZ-QSVM` | `zz` | conventional second-order Pauli-ZZ |
| `CWZZ-QSVM` | `cwzz` | covariance-weighted ZZ (flagship) |
| `CWZZ-permuted-control` | `cwzz_permuted` | **negative control** |

This is the cleanest experiment in the project: identical gate budget, identical
head, identical partitions, one variable.

Result: `Angle 0.557 = ZZ 0.557 < CWZZ 0.658`, but `permuted 0.689 > CWZZ 0.658`.
**Conclusion: the ordering is Angle ≈ ZZ < weighted, but the weighting's
*correlation structure* is not the source of the gain at $n=96$.**

### 6.3 `modality_ablation` (RQ2) — `run_modality_ablation` (`:203-238`)

Builds $1 + 2G$ configurations from $G$ feature groups:

| Config | Columns |
|---|---|
| `full_multimodal` | all |
| `only_<group>` | just that group |
| `remove_<group>` | union of all other groups |

Default `base_spec = PipelineSpec(name="CWZZ-QSVM-modality", reduction="pca",
embedding="cwzz", head="qsvm")`. All configurations share the same `y` and the
same `n_splits`/`seed` stratification. For the VOC dataset, the 3 groups are
contiguous column thirds (9/9/9); for kennel synth, the 4 physical groups
(pressure / proximity / motion / physiology).

`extra_meta` records `{"experiment": "modality_ablation", "groups": {g: len(c)}}`.

### 6.4 `robustness` — `run_robustness` (`:241-289`)

```python
noise_levels: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3)
dropout_fraction: float = 0.0
```

**Design: train on clean data, evaluate under degradation** (source comment at
`:261`). Rolls its own `StratifiedKFold`, fits a fresh pipeline per fold on
`X_tr`, then perturbs **`X_val` only**.

`_add_noise_and_dropout` (`:292-304`):

```python
stds  = values.std(axis=0); stds[stds == 0] = 1.0     # scale-relative noise
noisy = values + rng.normal(0.0, noise_level, size) * stds
if dropout_fraction > 0:
    mask = rng.random(shape) < dropout_fraction
    noisy[mask] = np.nanmedian(values, axis=0)[mask_column]
```

Noise is therefore **scale-relative** ($\sigma \times$ per-column std), which is
the right choice given the VOC columns span orders of magnitude.

**Two implementation notes:**

1. `FoldResult(..., 0.0, 0.0)` — train and infer times are **hardcoded to 0.0**,
   so `train_time_s_mean` in a robustness leaderboard is always 0.0 (visible in
   `outputs/lab/20260823_085700/leaderboard.csv`).
2. `run_robustness` returns **`{"run_dir", "levels"}` — not `{"leaderboard"}`**.
   `kennel_server.py:449` reads `result["leaderboard"]` → `KeyError` → caught by
   the broad `except Exception` → the job always ends as `status: "error"` even
   though the run succeeded on disk. This is confirmed by
   `outputs/lab/20260823_085411/` having artifacts but no `_cli_meta.json`, while
   the re-run `20260823_085700` has one. The CLI was fixed with
   `if "leaderboard" in result` / `result.get("levels", [])`; **the API route was
   not**. See [`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §1.

**Result:** kennel `CWZZ-QSVM` degrades 0.60 → 0.51 AUC as feature noise rises
0 → 0.3σ — graceful decay, no cliff.

---

## 7. `RunRegistry` — run artifacts

`lab/registry.py:28-66`.

```
outputs/lab/<YYYYmmdd_HHMMSS>/
├─ run.json              full run metadata
├─ leaderboard.csv       12 columns, sorted by roc_auc_mean DESC
└─ configs/
   ├─ CWZZ-QSVM__flagship_.json      (name sanitised: non [A-Za-z0-9-_] → _)
   ├─ Stacked-LR-ensemble.json
   └─ ... one per record
```

`LEADERBOARD_FIELDS` (`:11-24`) = `config, n_folds, roc_auc_mean, roc_auc_std,
accuracy_mean, accuracy_std, balanced_accuracy_mean, sensitivity_recall_mean,
specificity_mean, f1_macro_mean, brier_score_mean, train_time_s_mean`.

Each `configs/*.json` contains the full `PipelineSpec` dict, **per-fold metric
dicts**, and the NISQ `circuit_profile` — i.e. the complete evidence trail for
every number in the leaderboard.

---

## 8. CLI: `scripts/run_hybrid_search.py`

| Arg | Choices / default |
|---|---|
| `--dataset` | `voc_real` \| `kennel_synth`; default `voc_real` |
| `--experiment` | `search` \| `map_ablation` \| `modality_ablation` \| `robustness`; default `search` |
| `--max-samples` | int, `0` = all |
| `--vqc-epochs` | int, default 8 |
| `--n-splits` | int, default 5 |
| `--n-components` | int, default 6 |
| `--seed` | int, default 42 |
| `--output-root` | str, default `outputs/lab` |

`set_seed(args.seed)` is called first. Output is a markdown table filtered to
`config, roc_auc_mean, roc_auc_std, accuracy_mean, sensitivity_recall_mean,
specificity_mean, train_time_s_mean` (robustness uses
`config, roc_auc_mean, roc_auc_std, sensitivity_recall_mean, specificity_mean`),
plus `_cli_meta.json` recording `{"argv": sys.argv[1:], "dataset": args.dataset}`.

**Mutation caveat** (`:65-71`): the `search` branch overwrites
`p.n_components = args.n_components` on **every** preset — including
`Raw-XGBoost`, which ships at 32. So `--n-components 6` silently changes the
classical control from 32 columns to 6.

```bash
# reproduce the headline leaderboard
python scripts/run_hybrid_search.py --dataset voc_real --experiment search \
    --max-samples 120 --vqc-epochs 6 --n-splits 5

# reproduce the feature-map ablation
python scripts/run_hybrid_search.py --dataset voc_real --experiment map_ablation

# reproduce the robustness sweep
python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness
```

---

## 9. API surface: `POST /api/v1/lab/runs`

`kennel_server.py:384-482` + `register_lab_routes`.

```python
class LabRunRequest(BaseModel):
    dataset:      str = "voc_real"    # voc_real | kennel_synth
    experiment:   str = "search"      # search | map_ablation | modality_ablation | robustness
    max_samples:  int = 120   (ge=0)
    vqc_epochs:   int = 8     (ge=1, le=40)
    n_splits:     int = 5     (ge=2, le=10)
    n_components: int = 6     (ge=2, le=12)
    seed:         int = 42
```

| Method | Path | Behaviour |
|---|---|---|
| POST | `/api/v1/lab/runs` | spawns a daemon thread; returns `{"job_id": <12 hex>, "status": "running"}` |
| GET | `/api/v1/lab/runs` | list newest-first with `progress[-5:]` |
| GET | `/api/v1/lab/runs/{job_id}` | full job dict; `404 "Unknown lab run."` |

Job record (`:458-464`):

```python
{"job_id": uuid4().hex[:12], "status": "running", "request": req.model_dump(),
 "started_at": time.time(), "progress": []}
```

Terminal transitions (`:445-455`): success → `status="done"`,
`finished_at`, `result={"run_dir", "leaderboard"}`; exception →
`logger.exception(...)`, `status="error"`, `error=str(exc)`.

**Dispatch and progress callbacks** (`:419-443`):

| `experiment` | Runner | `progress_cb`? |
|---|---|---|
| `search` | `default_presets(...)`, then `p.n_components = req.n_components` per preset, then `run_hybrid_search(..., progress_cb=progress)` | ✅ |
| `map_ablation` | `run_feature_map_ablation(...)` | ❌ |
| `modality_ablation` | base spec `CWZZ-QSVM-modality` + `run_modality_ablation(...)` | ❌ |
| anything else (incl. `robustness`) | base spec `CWZZ-QSVM-robust` + `run_robustness(...)` | ❌ |

Operational notes:

- **No validation of `req.experiment`.** An unrecognised string silently becomes
  a robustness run.
- **Jobs are in-process, unbounded, and lost on restart.** No persistence, no
  eviction, no concurrency cap. `lab_jobs` is a plain dict guarded by a
  `threading.Lock` (`:390-391`).
- **Only `search` reports progress**, so the `/lab` page shows a running pill
  with no progress lines for the other three experiment types.
- **The `robustness` branch always errors** (KeyError on `"leaderboard"`, §6.4).
- The runners execute synchronously inside the thread and can run for minutes
  (Stacked-LR 135.7 s; a full search including VQC arms took ~10 min per the
  committed logs).

---

## 10. Test coverage

`tests/test_lab_pipeline.py` — **19 tests**, the largest test file.

| Group | Tests | What it pins down |
|---|---|---|
| `TestHybridPipeline` | 6 | QSVM fit/predict proba shape `(30,2)`; permuted covariance differs / unit diagonal / symmetric; `quantum_augmented_xgb` head; NaN train-median imputation → finite probs; classical head + `describe()` has `feature_map` and `spec`; `quantum_kernel_xgb` (`landmarks_.shape == (6,3)`, fused dim `(5,8)`) |
| `TestCVHarness` | 1 | fold metric dict shape |
| `TestExperiments` | 4 | map ablation produces 4 configs incl. `CWZZ-permuted-control`; modality ablation produces 5 (= full + 2 only + 2 remove-one); robustness over 2 levels; `run_hybrid_search` writes registry files |
| `TestRegistry` | 1 | leaderboard sorted descending |
| `TestKennelSynth` | 2 | shapes/labels; CV on synthetic trials (parametrised `head=["qsvm"]`) |
| module-level | 4 | `SoftVotingEnsemble` shape; `StackingEnsemble` OOF shape; `default_presets` completeness; Angle vs ZZ feature-map distinction |

Module fixture `toy_data`: `rng(7)`, $n=90$, 10 features,
$y = (f_0 + 0.6 f_1 - 0.4 f_2 + N(0,0.4)) > 0.2$. `_fast_spec` defaults:
`n_components=3, vqc_epochs=1, seed=42` — the tests are fast because the quantum
budget is minimised, not because the code paths are mocked.

**Not covered:** `StackingEnsemble.predict_proba` (only the OOF matrix shape is
checked), the `modality_ablation` 3+ group case (only 2 groups), the `search`
ensemble branches, and `run_robustness` through the API route.
