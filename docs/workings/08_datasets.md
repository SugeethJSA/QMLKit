# The Datasets

QMLKit currently touches three datasets. They serve very different purposes and
are kept clearly separated.

## 1. Real Lung-Cancer VOC chemistry — the modelling dataset

`docs/Lung_Cancer_VOC_Dataset_427.md` · loaded by `qmlkit.data.dataset_loader`

- **427 human breath samples**, each described by the measured concentrations of
  **27 volatile organic compounds** (formaldehyde, acetone-family, aromatic
  hydrocarbons, …).
- Labels: **Control (193)**, **Cancer (157)**, **Benign (77)**. Our default
  binary task is Cancer-vs-Control; a `disease_vs_control` variant pools Benign
  with Cancer.
- This is the dataset that replaced our early synthetic generator for benchmarking
  — it is real chemistry with real messiness (a few duplicate-ish rows, mixed
  decimal precision, which we document rather than silently "fix").

Caveat: this is *human* breath chemistry. In the full system vision, the dog
sniffs these samples and the kennel records its reaction; until those paired
sessions are collected, VOC features stand in as the classification target
domain.

## 2. Dog ECG metadata + dog registry — auxiliary canine physiology

`docs/dataset_1.md` (1,123 ECG recordings from 40 dogs) and `docs/DogInfo.md`
(45 dogs: breed, weight, age, sex, neuter status).

- The raw audio files are **not** in the repo, so we work from the annotations:
  beat timestamps let us compute heart-rate statistics and HRV (SDNN/RMSSD);
  bradycardia episodes give episode counts and durations.
- Purpose today: reference physiology ranges and a template for how we'll ingest
  collar-based physiological signals. Curated tables are written to `data/ecg/`.
- Honest limitation: no disease labels → no supervised claims from this data.

## 3. Synthetic kennel trials — the pipeline test-bed

Generated on demand by `src/qmlkit/lab/kennel_synth.py`.

- Simulates three-phase screening visits (baseline → exposure → post) for two
  classes, embedding moderate class-dependent differences in tremor amplitude,
  sniffing-band power, heart-rate rise and SpO₂ dip.
- Deliberately **not** trivially separable — good enough to exercise every gear
  of the lab (features, deltas, ablations, robustness) before real Data-Lab
  sessions exist.
- Any result on this dataset is a plumbing check, not biology.

## The one to watch

The dataset that matters most is the one that doesn't exist yet: **paired
trials** where a labelled sample sits in the kennel and a dog's recorded
reaction is joined to that label. The Data Lab page exists precisely to collect
it — every recording session produces one training row under
`data/kennel/<label>/`.
