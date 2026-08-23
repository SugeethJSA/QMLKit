# Workstream G — Hybrid Training Lab & Paper-Parity Implementation

**Date:** 2026-08-23
**Trigger:** Root `*.docx` manuscript (Correlation-Aware Quantum Kernel Learning for Canine-Assisted
Breath-Based Multi-Cancer Screening) + user request for a "best of both worlds" hybrid model search.

## G1. Manuscript audit → gap list

Implemented already: CW-ZZ/BioZZ map (φᵢⱼ = 2Cᵢⱼ(π−xᵢ)(π−xⱼ)), train-only correlation estimation,
fidelity-kernel QSVM, stratified leak-free splits.

Gaps being closed in this pass:

| # | Gap | Fix |
|---|---|---|
| 1 | HR + SpO₂ sensing missing from firmware | MAX30102 in `kennel_node.ino` v2; frame fields `hr_bpm`, `spo2_pct`; optional in parser |
| 2 | Three-phase trials (baseline/exposure/post) with Δx = xₑ−xᵦ (§V-B) | Phase segmentation + delta features in kennel feature layer |
| 3 | Table I behavioural features (approach latency, withdrawal time, dwell, contact duration) | Extended extractor |
| 4 | Feature-map ablation Angle vs ZZ vs CW-ZZ (§VII-D) | `run_feature_map_ablation` |
| 5 | Correlation-control: permuted-C CW-ZZ (§VII-D) | `covariance_mode="permuted"` |
| 6 | Modality ablation incl. remove-one (RQ2) | Feature-group registry + `run_modality_ablation` |
| 7 | Noise-injection + dropout robustness (§VII-E) | `run_robustness` |
| 8 | Logistic Regression baseline (§VII-B) | Added to suite |
| 9 | Identical partitions across models; stratified k-fold | `lab/cv.py` honoring `n_splits_cv` |
| 10 | Calibrated screening-risk score (§VI-F) | CalibratedClassifierCV migration (resolves SVC deprecation too) |
| 11 | Reproducibility stats: qubits/reps/gates/params (§VII-F) | Real profiler implementation wired into run records |

Explicitly out of scope per user: dual IMU and dedicated activity monitor.

## G2. Dead-code resolutions

- `TorchVQC` (vqc.py) — **deleted** (never referenced).
- `hardware_profiler.parameter_count=0` — **implemented** analytic gate/param counts.
- `SVC(probability=True)` — migrated to `CalibratedClassifierCV(SVC(...), ensemble=False)`.

## G3. Hybrid lab architecture

```
src/qmlkit/lab/
├── pipeline.py     PipelineSpec(reduction × embedding × head × covariance_mode)
│                   HybridPipeline.fit/predict_proba — every stage train-only
├── stacking.py     OOF meta-features, StackingEnsemble(LR meta), SoftVotingEnsemble
├── cv.py           stratified k-fold harness; fresh pipeline+correlation per fold
├── experiments.py  presets + run_hybrid_search / run_feature_map_ablation /
│                   run_modality_ablation / run_robustness
├── registry.py     outputs/lab/<run_id>/run.json + leaderboard.csv (+ circuit stats)
└── kennel_synth.py two-class synthetic trial generator (baseline/exposure windows,
                    class-dependent signatures) feeding the same lab on tabular features
```

### Curated presets (5-fold CV)

1. pca→CWZZ-QSVM (flagship) 2. pca→ZZ-QSVM 3. pca→Angle-QSVM 4. pca→permuted-CWZZ-QSVM (control)
5. mutual-info→CWZZ-QSVM 6. autoencoder→VQC 7. quantum-augmented XGB (VQC proba ⊕ raw features)
8. QCNN arm 9. soft-voting {SVM-RBF, RF, XGB, QSVM} 10. stacked {RF, XGB, QSVM, VQC}→LogReg
11. plain XGBoost control. Runtime bounded by `--max-samples`, `--vqc-epochs`.

## G4. Interfaces

- `scripts/run_hybrid_search.py` — dataset (voc_real | kennel_synth), preset selection, budget, seed;
  writes `outputs/lab/<run_id>/`.
- API `/api/v1/lab/runs` (POST start background search, GET list/status/result).
- Frontend `/lab` page — launch runs, live status, leaderboard table.
- `docs/workings/*.md` — layman-language explainer series covering the whole system.

## G5. Verification gates

pytest green (new `tests/test_lab_pipeline.py` etc.) · ruff clean · frontend lint/build · one real
VOC hybrid-search run producing leaderboard + report (`docs/reports/hybrid_search_report.md`).
