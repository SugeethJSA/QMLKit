# Real-Data Benchmark Report — Lung Cancer VOC

**Date:** 2026-08-23
**Command:**
```
python scripts/run_benchmark.py --source markdown --voc-task cancer_vs_control \
    --max-samples 120 --vqc-epochs 8 --n-qubits 6 --output-dir outputs/benchmark_real
```

## Run configuration & honesty notes

| Parameter | Value | Rationale |
|---|---|---|
| Dataset | Real `Lung_Cancer_VOC_Dataset_427` (cancer_vs_control) | First non-synthetic run of the pipeline |
| N (after balanced cap) | 120 = 60 Control + 60 Cancer | Bounds O(N²) QSVM kernel cost on this machine |
| Split | Stratified 80/20 → 96 train / 24 test | Leak-free; scalers + PCA fit on train only |
| Quantum register | 6 qubits (PCA reduction from 27 compounds) | NISQ-realistic |
| VQC epochs | 8 (default 20) | Runtime bound on this environment |
| Seed | 42 throughout | Reproducibility |

**Caveats:** single split, small test set (n=24) → wide confidence intervals;
VQC under-trained relative to default settings. Numbers are evidence, not
tuned results.

## Results (ranked by ROC-AUC)

| Model | Paradigm | Accuracy | Bal. Acc | Sensitivity | Specificity | ROC-AUC | Brier | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | Classical | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.010 | 0.13 |
| Random Forest | Classical | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.041 | 0.13 |
| MLP NeuralNet | Classical | 0.958 | 0.958 | 0.917 | 1.000 | **1.000** | 0.020 | 0.16 |
| SVM Linear | Classical | 0.958 | 0.958 | 0.917 | 1.000 | 0.979 | 0.061 | 0.00 |
| SVM RBF | Classical | 0.917 | 0.917 | 0.917 | 0.917 | 0.972 | 0.074 | 0.00 |
| VQC StronglyEntangled | Quantum | 0.667 | 0.667 | 0.500 | 0.833 | 0.625 | 0.245 | 343.5 |
| QSVM BioZZ | Quantum | 0.542 | 0.542 | 0.417 | 0.667 | 0.507 | 0.250 | 0.74 |

Artifacts: `outputs/benchmark_real/benchmark_metrics.csv`, `benchmark_comparison.png`, `run_metadata.txt`.

## Interpretation

1. **The real dataset is highly separable for strong classical learners** — even
   a linear SVM reaches ~0.98 AUC. With only 24 test samples the perfect 1.0s
   should be read as "≥ ceiling", not exact.
2. **Quantum models lag far behind** (QSVM ≈ chance at this scale/epochs).
   Notably, this is *with* the BioZZ covariance fix active — real entanglement
   weights did not rescue kernel performance at n=96 training points and 6-qubit
   PCA compression, which discards most variance signal.
3. **Contrast with synthetic history:** the committed synthetic-run artifact had
   QSVM at 0.64 AUC against trivially separable baselines; on real data quantum
   models fall to 0.51–0.63 while baselines stay ≥0.97. The prior synthetic gap
   understated reality.
4. **Cost asymmetry is stark:** VQC training consumed ~344 s vs ≤0.16 s for any
   baseline — an important practical result for the platform's value story.

### Recommended next steps for quantum credibility

- k-fold CV (config field `n_splits_cv` exists but unused) instead of one split.
- Sweep `n_qubits ∈ {4,6,8}` × selector method `{pca, mutual_info}`; raw-compound
  selection feeds BioZZ genuinely interpretable correlations.
- Data re-scaling for angle encoding (current min-max per-feature on raw
  concentrations spans decades between compounds).
- Full-N runs on faster hardware once firmware/hardware bring-up completes.

## Kennel subsystem status (related)

- Ingestion server, simulation source, feature extractor (42 features),
  recording manager and trainer CLI are implemented and tested
  (`tests/test_hardware_ingest.py`, 14 passing).
- No labelled kennel sessions exist yet → diagnostics report `"untrained"`
  until Data-Lab recordings are collected and
  `python scripts/train_kennel_model.py` is run (leave-one-dog-out validation).
