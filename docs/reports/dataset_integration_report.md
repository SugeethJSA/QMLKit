# Dataset Integration Report

**Date:** 2026-08-23 · **New ingestion code:** `src/qmlkit/data/dataset_loader.py`

This pass added the first external-data ingestion paths to QMLKit and integrated
the two datasets found in `/docs`.

---

## 1. Lung Cancer VOC Dataset (primary) — `docs/Lung_Cancer_VOC_Dataset_427.md`

### Schema as shipped

| Property | Value |
|---|---|
| Rows | 427 patients (PatientID 1–427) |
| Features | 27 VOC compound concentration columns (CH2O … C15H10O), numeric ppb-scale |
| Labels | **3-class**: Control 193 · Cancer 157 · Benign 77 |

The markdown file is a pandas `to_markdown()` dump; the loader parses any
pipe-delimited table, skips alignment rows and coerces numerics.

### Binarisation tasks implemented

| Task | Mapping | N used |
|---|---|---|
| `cancer_vs_control` *(default)* | Control→0, Cancer→1; Benign dropped | 350 |
| `disease_vs_control` | Control→0; Benign+Cancer pooled→1 | 427 |

### Notes / caveats

- Rows with unparseable feature values are dropped (none observed in the
  bundled file — all 427 parse cleanly).
- Duplicate-looking rows exist in the source (e.g., patients 21/45/46 share the
  first 14 compound values). We preserve them as-is; deduplication is left to
  the analyst. Flagged for future review.
- Precision changes mid-file: patients ≈1–88 carry 2–3 decimals, later rows are
  full float precision — consistent with a simulated augmentation appended to a
  real cohort. Treat absolute performance numbers with that caveat.

## 2. Canine ECG metadata + DogInfo (secondary)

### Constraints (documented honestly)

- The `ecg_data/*.wav` audio files referenced by `docs/dataset_1.md` are **not**
  distributed with this repo → no waveform processing is possible.
- No diagnostic labels exist → supervised disease classification is not
  possible from these tables alone.

Integration is therefore annotation-based:

| Output (`data/ecg/`) | Content |
|---|---|
| `ecg_recordings_features.csv` | 1,123 recordings × derived features |
| `dog_info.csv` | 45 dogs (breed/weight/age-months/gender/neutering) |
| `ecg_features_with_dog_info.csv` | features joined on `pet_id ↔ DogID` (444 matched — DogInfo covers only 45 of the dogs appearing in the ECG dump) |
| `summary_stats.csv` | headline statistics |

### Derived per-recording features (from R-peak timestamps & annotations only)

`n_beats`, HR mean/min/max (from RR intervals), SDNN + RMSSD (HRV),
bradycardia episode count / total duration / mean value (`segments_br`),
bad-signal fraction (`bad_ecg` spans ÷ duration).

### Headline statistics

| Metric | Value |
|---|---|
| Recordings parsed | 1,123 |
| Distinct dogs | 40 |
| Median HR | 62.6 bpm (plausible resting canine) |
| Recordings with ≥1 bradycardia episode | 1,063 |
| Recordings >50 % bad signal | 149 |

Data-quality flags for downstream users:
- `hr_min_bpm_min = 0.62` bpm ⇒ some rows contain corrupted/placeholder beat
  lists; filter on quality metrics before modeling.
- A transparent weak target (`has_bradycardia`) exists but is circular by
  construction (derived from the same annotation); it is exported for pipeline
  demos, not clinical claims.

## 3. Where things live now

```
src/qmlkit/data/dataset_loader.py   # loaders + balanced_subsample + ECG feature extraction
scripts/run_benchmark.py            # --source synthetic|markdown|csv --data-path --voc-task --max-samples
scripts/build_ecg_tables.py         # docs/*.md -> data/ecg/*.csv
tests/test_dataset_loader.py        # 13 tests covering parsing/mapping/subsampling/ECG
outputs/benchmark_real/             # real-data benchmark artifacts
```
