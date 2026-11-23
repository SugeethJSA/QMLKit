# 02 — Data Layer

All data enters through `src/qmlkit/data/` (VOC domain) or
`src/qmlkit/hardware/` + `src/qmlkit/lab/kennel_synth.py` (kennel domain).

---

## 1. Bundled datasets

| Dataset | Path | Loader | Size |
|---|---|---|---|
| Lung Cancer VOC (real, primary) | `docs/Lung_Cancer_VOC_Dataset_427.md` | `load_lung_voc_dataset()` | 427 patients × 27 compounds, 3 classes |
| Canine ECG metadata (secondary) | `docs/dataset_1.md` (17.8 MB) | `load_dog_ecg_metadata()` + `extract_ecg_features()` | 1,123 recordings, 40 dogs |
| Dog registry (secondary) | `docs/DogInfo.md` | `load_dog_info()` | 45 dogs |
| Kennel recordings (live/hardware) | `data/kennel/<label>/*.csv` | `load_csv_rows()` → window features | 3 files, 1 dog (`Rex`) |
| Synthetic VOC cohort | generated | `BiomimeticVOCGenerator` | on demand |
| Synthetic kennel trials | generated | `generate_synthetic_trials()` | on demand |
| User CSV | `--data-path` | `load_csv_dataset()` | arbitrary |

Path constants: `dataset_loader.py:19-23`
```python
REPO_ROOT        = Path(__file__).resolve().parents[3]
LUNG_VOC_MARKDOWN = REPO_ROOT / "docs" / "Lung_Cancer_VOC_Dataset_427.md"
DOG_ECG_MARKDOWN  = REPO_ROOT / "docs" / "dataset_1.md"
DOG_INFO_MARKDOWN = REPO_ROOT / "docs" / "DogInfo.md"
```

---

## 2. Lung Cancer VOC dataset (real)

### 2.1 Schema as shipped

The markdown is a pandas `to_markdown()` dump, `Sheet1`, pipe-delimited.

| Property | Value |
|---|---|
| Rows | 427 (`PatientID` 1–427) |
| Columns | `PatientID` + **27 VOC columns** + `Class` = 29 |
| Classes | Control 193 · Cancer 157 · Benign 77 |

**The 27 feature columns, in file order:**

```
CH2O, C2H4O, C3H6O, C4H8O, C5H10O, C6H12O, C7H14O, C8H16O, C9H18O,
C10H20O, C11H22O, C12H24O, C13H26O, C4HO8O2, C2H4O2, C3H4O, C6H10O2,
C9H16O2, C3H4O2, C4H6O2, C4H6O, C4H4O2, C5H8O, C7H6O, C7H11O, C13H22O, C15H10O
```

These are molecular-formula identifiers for breath VOC families (aldehydes,
ketones, aromatics, terpenes/sulphur compounds) on a ppb-like scale.

### 2.2 Markdown table parser

`load_markdown_table(path)` (`dataset_loader.py:44-69`):

1. Read UTF-8.
2. Keep only lines where `line.strip()` starts **and** ends with `|`.
3. Split `line[1:-1].split("|")`, strip each cell.
4. Drop rows where **every** cell satisfies `set(cell) <= set(":- ")` — i.e. markdown alignment separators such as `|---:|---:|`.
5. Require ≥ 2 surviving rows, else `ValueError("No markdown table found in {path}")`.
6. Header = row 0; keep data rows only where `len(row) == len(header)`.

This is a generic pipe-table parser, not a hard-coded format assumption — it is
why the same function parses the ECG tables.

### 2.3 Label mapping and binarisation

```python
LUNG_VOC_LABEL_NAMES = {0: "Control", 1: "Cancer"}                    # :25
LUNG_VOC_TASKS = {                                                      # :26-31
    "cancer_vs_control": {"control": 0, "cancer": 1},
    "disease_vs_control": {"control": 0, "benign": 1, "cancer": 1},
}
```

| Task | Mapping | Resulting shape | Class counts |
|---|---|---|---|
| `cancer_vs_control` *(default)* | Control→0, Cancer→1; **Benign dropped** | (350, 27) | 193 / 157 |
| `disease_vs_control` | Control→0; Benign+Cancer pooled→1 | (427, 27) | 193 / 234 |

`load_lung_voc_dataset()` (`:72-118`) requires `PatientID` and `Class`
(`ValueError` otherwise), coerces all other columns with
`pd.to_numeric(errors="coerce")`, lowercases/strips labels, and — when
`drop_unmapped` is set — removes rows whose label is not in the task mapping.
An unknown label that survives masking raises
`ValueError("Unknown class label(s) …; expected one of …")` rather than being
silently coerced.

### 2.4 Data-quality caveats (documented in the integration report)

- All 427 rows parse cleanly; no NaN drop occurs on the bundled file.
- Duplicate-looking rows exist (patients 21/45/46 share the first 14 compound
  values). Preserved as-is; deduplication left to the analyst.
- **Precision changes mid-file**: patients ≈1–88 carry 2–3 decimals, later rows
  full float precision — consistent with a simulated augmentation appended to a
  real cohort. Absolute performance numbers should be read with that caveat.
- Source: `docs/reports/dataset_integration_report.md:10-39`.

---

## 3. Balanced subsampling

`balanced_subsample(X, y, max_samples, seed=42)` (`dataset_loader.py:121-140`):

```python
per_class = max(1, max_samples // len(classes))
for cls in classes:
    idx = np.flatnonzero(y == cls)
    take = min(per_class, len(idx))
    chosen.append(rng.choice(idx, size=take, replace=False))
sel = np.sort(np.concatenate(chosen))      # order-preserving
return X.iloc[sel].reset_index(drop=True), y[sel]
```

- No-op (with `reset_index`) when `max_samples >= len(y)`.
- Guarantees **both classes survive** — important because dropping a class would
  silently break stratification downstream.
- Rationale (docstring): bounds the $O(N^2)$ quantum-kernel cost. A 6-qubit
  statevector kernel over 96 samples means 96 circuit evaluations per Gram
  column; at 4,096 samples it is 16,384×.

---

## 4. Split protocol

`BiomedicalDataPipeline.create_leak_free_split()` (`preprocessor.py:59-113`):

1. `train_test_split(..., test_size=0.20, stratify=y, random_state=seed)`
2. Optional second stratified split for validation, with the relative size
   recomputed as `val_size / (1 − test_size)` (so 10% val of the *whole* set).
3. Scaler fitted on `X_train` **only**; then `transform` applied to test and val.

`scripts/run_benchmark.py:96-106` uses the simpler two-way split directly:

```python
idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.20,
                                       stratify=y, random_state=args.seed)
```

Both are leak-free because the scaler, selector, angle bounds, and covariance
matrix are all fitted inside the training branch.

---

## 5. Synthetic VOC cohort (physics-based generator)

`BiomimeticVOCGenerator` (`biomimetic_voc_generator.py`) is the historical data
source. It is **not** used for the headline real-data numbers, but it powers the
`Training Studio` in the console and the `--source synthetic` benchmark path.

### 5.1 Sensor array model — `config.SensorArrayConfig`

| Field | Default | Meaning |
|---|---|---|
| `n_sensors` | 16 | cross-reactive sensor channels (4–64) |
| `sampling_freq_hz` | 10.0 | nominal |
| `transient_steps` | 100 | timesteps per exposure |
| `baseline_noise_sigma` | 0.08 | i.i.d. Gaussian noise σ |
| `drift_amplitude` | 0.04 | baseline ramp amplitude |
| `adsorption_rate` (`k_ads`) | 0.15 | Langmuir adsorption rate |
| `desorption_rate` (`k_des`) | 0.08 | first-order desorption rate |

### 5.2 Chemical classes — `config.VOCBiomarkerConfig`

24 compounds in **four blocks of 6** (the block structure is what the pathway
attribution in [`09_explainability.md`](09_explainability.md) relies on):

| Block | Pathway | Compounds |
|---|---|---|
| 0 | Lipid peroxidation (aldehydes) | Hexanal, Heptanal, Octanal, Nonanal, Benzaldehyde, Decanal |
| 1 | Mitochondrial / fatty-acid (ketones) | Acetone, 2-Butanone, 2-Pentanone, 3-Octanone, Acetophenone, Cyclohexanone |
| 2 | Cytochrome P450 (aromatics) | Ethylbenzene, Styrene, Toluene, o-Xylene, 1,2,4-Trimethylbenzene, Benzene |
| 3 | Alkanes / terpenes / sulphur | Isoprene, Octane, Decane, D-Limonene, Dimethyl_disulfide, Dimethyl_sulfide |

`cancer_types` = Healthy, Lung_Cancer, Breast_Cancer, Colorectal_Cancer,
Prostate_Cancer, Ovarian_Cancer.

### 5.3 Sensor affinity matrix

`_build_sensor_affinity_matrix()` (`:49-66`) draws a fixed 16×24 matrix from
`np.random.default_rng(random_state + 100)`:

- **Preferred** class (sensor's own class block): `uniform(0.45, 0.85)`
- **Cross-reactive**: `uniform(0.15, 0.50)`

The source comment (`:54`) is explicit that these bounds were **retuned downward**
because the original separation (preferred 0.6–1.0 vs cross 0.05–0.35) made the
data trivially separable and every classical model hit 1.000 ROC-AUC.

### 5.4 VOC concentration model

`_generate_voc_profiles(n, cancer_type)` (`:68-111`):

- `mean_log`: 24 baseline log-concentrations, e.g. aldehydes
  `0.5,0.4,0.3,0.3,0.6,0.2`; ketones `2.5,1.0,0.5,0.3,0.4,0.3`; aromatics
  `0.8,0.4,1.2,0.5,0.3,0.3`; alkanes `2.2,0.6,0.4,0.5,0.3,0.3`.
- `std_log = 0.42` everywhere — also retuned (was 0.25 → "perfect separation for
  XGB").
- Per-cancer multipliers on `exp(log_conc)`, floor at 0.01:

| cancer | elevated (×) | depleted (×) |
|---|---|---|
| Lung | `[0,1,4]×2.2`, `[12]×1.8` | `[18]×0.65` |
| Breast | `[1,2,8,9,16]×2.0` | — |
| Colorectal | `[22,23,4,11]×2.3` | — |
| Prostate | `[7,8,14,23]×2.1` | — |
| Ovarian | `[3,5,10,11]×2.2` | — |
| Healthy | all 1.0 | — |

### 5.5 Langmuir transient kinetics

`_simulate_transient_signals()` (`:113-153`) — the physics core:

```
stim[s]  = affinity_matrix[s, :] @ voc_matrix[i]        (16,)
g_max[s] = log1p(stim[s]) * 1.5                        # log saturation, not linear
drift[t] = drift_amplitude * linspace(-1, 1, T) + N(0, 0.005)

t <  t_on (=20):  g = 0
t_on <= t < t_off (=70):  g += k_ads * (g_max[s] - g)   # Langmuir adsorption
t >= t_off:              g += -k_des * g                # first-order desorption

signals[i, s, t] = max(0, g + drift[t] + N(0, 0.08))
```

This produces the characteristic **fast rise / slow decay** transients that
olfactory e-nose sensors exhibit, and the four extracted kinetic features
(§5.6) are exactly the shape descriptors of that transient.

### 5.6 Feature extraction — 4 per sensor, 64 total

`_extract_kinetic_features()` (`:155-197`) smooths with
`savgol_filter(window_length=min(11, T), polyorder=2)` then computes per sensor
`sensor_{SS}_{max|auc|rise|decay}`:

| Feature | Definition |
|---|---|
| `max` | `np.max(smooth)` |
| `auc` | `np.sum(smooth[t_on:t_off])` — exposure window only |
| `rise` | `(smooth[peak] − smooth[t_on]) / (peak − t_on)`, else 0.01; stored as `max(0, ·)` |
| `decay` | `(smooth[t_off] − smooth[-1]) / (T − t_off)`, else 0.01; stored as `max(0, ·)` |

### 5.7 `generate_cohort()` metadata

Per sample (`generate_cohort`, `:199-246`):

- `sample_id = f"SMPL_{counter:06d}"`
- `patient_age`: Healthy `N(52, 14)`, cancer `N(60, 12)`, clipped `[21, 88]`
- `patient_sex` ∈ {M, F}; `smoking_status` ∈ {Never, Former, Current} with
  `p = [0.5, 0.3, 0.2]`
- `sample_matrix = "Exhaled_Breath"`
- `label_cancer_type`, `label_binary` (0 Healthy / 1 else), `label_stage`
  (Stage_I..IV with `p = [0.35, 0.35, 0.2, 0.1]`)

Returns `SyntheticCohort(df_features (N,64), raw_time_series (N,16,100),
voc_ground_truth (N,24), metadata)`. Verified in `tests/test_data_pipeline.py:24-32`.

> Note: column names are `f"voc_{compound.lower()}_ppb"`, which retains spaces
> and parentheses (e.g. `voc_2-butanone_ppb`).

---

## 6. Canine ECG metadata (secondary, annotation-only)

`extract_ecg_features()` (`dataset_loader.py:223-270`) — **annotation-only**; the
`ecg_data/*.wav` files referenced by `docs/dataset_1.md` are not distributed, so
no waveform processing is possible, and there are no diagnostic labels, so no
supervised disease classification.

Derived per recording: `n_beats`, HR mean/min/max from RR intervals
(`60000/rr_ms`), SDNN `std(rr_ms, ddof=1)`, RMSSD `sqrt(mean(diff(rr_ms)²))`,
bradycardia episode count / total duration / mean value (from `segments_br`
`deb`/`fin` pairs), and `bad_ecg_fraction`.

Derived CSV outputs (`scripts/build_ecg_tables.py` → `data/ecg/`):

| File | Content |
|---|---|
| `ecg_recordings_features.csv` | 1,123 recordings × derived features |
| `dog_info.csv` | 45 dogs (breed/weight/age-months/gender/neutering) |
| `ecg_features_with_dog_info.csv` | joined on `pet_id ↔ DogID` (444 matched) |
| `summary_stats.csv` | headline statistics |

**Headline statistics:** 1,123 recordings · 40 distinct dogs · median HR 62.6 bpm
· 1,063 recordings with ≥1 bradycardia episode · 149 recordings >50% bad signal.

**Quality flags:** `hr_min_bpm_min = 0.62` bpm ⇒ some rows contain corrupted or
placeholder beat lists — filter on quality metrics before modelling. The exported
`has_bradycardia` target is circular by construction (derived from the same
annotation) and is exported for pipeline demos, not clinical claims.

---

## 7. Kennel recordings

### 7.1 On-disk format

```
data/kennel/<label>/<dog_id>_<label><trial>.csv
```

`session_recording.py:92` — note there is **no underscore before the trial
number** (an older doc shows `<dog>_<label>_<trial>.csv`; see
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md)).

**25-column header, exact order** (`:99-109`):

```
ts_ms, seq, state,
fsr_0, fsr_1, fsr_2, fsr_3,
ir_0, ir_1, ir_2, ir_3, ir_4, ir_5,
us_bottom, us_top,
acc_x, acc_y, acc_z,
gyr_x, gyr_y, gyr_z,
imu_temp_c, hr_bpm, spo2_pct, wall_ts
```

The label is recoverable from the **parent directory name**
(`train_kennel_model.py:74`); the dog id from the filename prefix. The
`sample_id` captured at session start is held in memory but **never written to
disk** (`session_recording.py:51-55` vs `:99-119`).

Example header row from the committed `data/kennel/calm/Rex_calm1.csv`:

```csv
ts_ms,seq,state,fsr_0,...,spo2_pct,wall_ts
62799,6280,IDLE,22.0,36.0,6.0,54.0,0,0,0,0,0,0,38.67,99.77,-0.048,9.724,-0.103,
0.0094,-0.029,0.0017,37.69,-1.0,-1.0,1787467669.058
```

Consecutive `ts_ms` differ by 10 ms → ~100 Hz source data in these files, not the
10 Hz wire rate (see [`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §3).

### 7.2 Current corpus

| File | Rows |
|---|---|
| `data/kennel/calm/Rex_calm1.csv` | 391 |
| `data/kennel/calm/Rex_calm2.csv` | 387 |
| `data/kennel/cancer/Rex_cancer1.csv` | 719 |

One dog, two labels, **no `control/` directory**. Consequence: LOSO validation has
no usable fold and `models/kennel_model.joblib` does not exist, so live
diagnostics report `"untrained"`.

### 7.3 Synthetic kennel trials

`generate_synthetic_trials()` (`lab/kennel_synth.py:76-83`):

```python
trials_per_class=40, baseline_s=4.0, exposure_s=8.0, post_s=4.0,
rate_hz=100, seed=42
```

**Class signatures** (`:93-103`):

| Quantity | Control (label 0) | Cancer (label 1) |
|---|---|---|
| `response_gain` | 1.0 | **1.6** |
| `dog_hr` (per-dog baseline) | `U(55, 95)` bpm | `U(55, 95)` bpm |
| `dog_spo2` (per-dog baseline) | `U(94, 99)` % | `U(94, 99)` % |
| `tremor_amp` | `0.05 × gain` = 0.05 | 0.08 |
| `hr_rise` | `N(8.0, 2.5)` bpm | `N(12.8, 2.5)` bpm |
| `spo2_drop` | `\|N(0.9, 0.3)\|` % | `\|N(1.44, 0.3)\|` % |

Dog ids: `f"{'AB'[label]}{trial % n_dogs_per_class:02d}"` with
`n_dogs_per_class = max(4, trials_per_class // 10)` → `A00..A03` / `B00..B03`.

**Three-phase trial structure** (`:105-146`), each phase sub-sampled to
`window = int(rate_hz)` steps so every phase yields 100 frames at defaults:

| Phase | `start_ms` | `sniff` | HR | SpO2 | `total_load` |
|---|---|---|---|---|---|
| baseline | 0 | False | `dog_hr` | `dog_spo2` | 120.0 |
| exposure | 4000 | True | `dog_hr + hr_rise` | `dog_spo2 − spo2_drop` | **900.0** |
| post | 12000 | False | `dog_hr + 0.4·hr_rise` | `dog_spo2 − 0.4·spo2_drop` | 120.0 |

Per-frame motion model:

| Signal | Formula |
|---|---|
| `sniff_wave` | `sin(2π·3.0·t)·0.35 + N(0, 0.03)` if sniff else 0 |
| `tremor` | `sin(2π·6.0·t + 0.7)·tremor_amp` |
| `acc_x` | `N(0, 0.05) + tremor + sniff_wave` |
| `acc_y` | `9.78 + N(0, 0.05) + 0.5·sniff_wave` |
| `acc_z` | `N(0, 0.05) + (0.15·sniff_wave if sniff else 0)` |
| `gyr[k]` | `N(0, 0.02) + (0.06·sniff_wave if sniff else 0)` |
| `us_bottom` | `clip(38.0 − 12.0·sniff + N(0, 3), 5, 200)` cm |
| `us_top` | `clip(95.0 − 45.0·sniff + N(0, 5), 5, 200)` cm |
| `imu_temp_c` | `37.2 + N(0, 0.2)` °C |

Posture shares come from `rng.dirichlet(ones(4) × 20.0)` per phase, with
`clip(share + N(0, 0.01, 4), 0.05, 0.85)` renormalisation while sniffing.

**Output:** `X` = `(80, 55)` DataFrame, then `X = X.replace(-1.0, np.nan)`
(`:156`) — the "physiology unavailable" sentinels become NaN and are handled by
`HybridPipeline`'s train-median imputation. Feature groups are prefix-matched
(`:161-166`): pressure = `("fsr_","total_load","lr_imbalance","fb_imbalance","cop_drift")`,
proximity = `("ir_","us_")`, motion = `("acc_","gyr_")`, physiology = `("imu_temp","hr_","spo2_")`.

> Note: these prefix sets also capture the 10 `_delta`/recovery columns, so they do
> not partition the 55 features the way `FEATURE_GROUPS` partitions the 45 window
> features. See [`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md).

---

## 8. Lab dataset loading

`load_lab_dataset(name, max_samples=0, seed=42)` (`lab/data_sources.py`) returns
`(X, y, feature_groups)`:

| `name` | Source | `feature_groups` |
|---|---|---|
| `voc_real` | `load_lung_voc_dataset(task="cancer_vs_control")` | 3 contiguous column thirds via `third = max(1, len(cols)//3)` → 9/9/9 for the 27 VOC columns |
| `kennel_synth` | `generate_synthetic_trials(seed=seed)` | 4 physical groups |

`max_samples` is applied via `balanced_subsample`; index is reset; unknown names
raise `ValueError(f"Unknown lab dataset: {name}")`.

The VOC thirds give the modality ablation (RQ2) a physically-meaningful grouping
(early columns = aldehydes/ketones, late = aromatics/sulphur) even though the
bundled dataset's column names are molecular formulas rather than compound names.

---

## 9. Test coverage of the data layer

`tests/test_dataset_loader.py` (13 tests):

- `TestMarkdownTableParser` (2) — alignment-row skipping, ragged-row rejection
- `TestLungVocLoader` (5) — label mapping + numeric coercion, unknown label
  raises, unmapped dropped, **the real bundled file loads with 27 features and
  350 = 193 + 157 rows**, `disease_vs_control` pools benign (427; 193/234)
- `TestBalancedSubsample` (2)
- `TestCsvLoader` (2) — including a deliberately malformed row (`"bad"`)
- `TestDogEcg` (2)

`tests/test_data_pipeline.py` (4): selector single-vs-batch consistency; VOC
cohort shapes `(20,64) / (20,16,100) / (20,24)`; leak-free split (35/5/10 with
`X_train.mean() ≈ 0`); PCA selector bounded in $[-\pi,\pi]$ with a working
`inverse_transform → (30,64)`.
