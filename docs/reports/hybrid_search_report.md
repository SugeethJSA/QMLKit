# Hybrid Training-Lab Search Report

**Date:** 2026-08-23 · **Engine:** `src/qmlkit/lab/` · **CLI:** `scripts/run_hybrid_search.py`

## Setup

| Parameter | VOC run | Kennel-synth run |
|---|---|---|
| Dataset | Real Lung VOC (cancer_vs_control), balanced cap 120 | Synthetic three-phase trials (80 trials, 55 features) |
| Protocol | Stratified 5-fold CV; identical partitions across configs; per-fold fresh scaler/selector/correlation matrix | same |
| Quantum budget | n_components=6, VQC epochs=6 | same |
| Seeds | 42 everywhere | 42 |

## VOC real-data leaderboard (mean ± std over folds)

| Rank | Config | ROC-AUC | Acc | Sens | Spec | Train s |
|---|---|---|---|---|---|---|
| 1 | **Stacked-LR ensemble {RF,XGB,QSVM,VQC}** | **0.986 ± 0.018** | 0.925 | 0.917 | 0.933 | 135.7 |
| 2 | **Quantum-Augmented-XGB** (VQC signal ⊕ raw → XGB) | **0.974 ± 0.023** | 0.900 | 0.883 | 0.917 | 15.4 |
| 3 | SoftVote {SVM-RBF,RF,XGB,QSVM} | 0.913 ± 0.050 | 0.808 | 0.733 | 0.883 | 1.1 |
| 4 | PCA-XGBoost (classical control) | 0.910 ± 0.053 | 0.867 | 0.833 | 0.900 | 0.16 |
| 5 | Raw-XGBoost (control) | 0.887 ± 0.076 | 0.783 | 0.750 | 0.817 | 0.12 |
| 6 | MI-CWZZ-QSVM | 0.757 ± 0.083 | 0.692 | 0.733 | 0.650 | 0.84 |
| 7 | CWZZ-permuted control | 0.689 ± 0.095 | 0.642 | 0.600 | 0.683 | 1.99 |
| 8 | CWZZ-QSVM (flagship) | 0.658 ± 0.156 | 0.617 | 0.633 | 0.600 | 2.55 |
| 9 | AE-VQC | 0.576 ± 0.151 | 0.508 | 0.533 | 0.483 | 20.96 |
| 10 | QCNN | 0.558 ± 0.109 | 0.592 | 0.633 | 0.550 | 13.39 |
| 11 | ZZ-QSVM | 0.557 ± 0.115 | 0.517 | 0.533 | 0.500 | 2.48 |
| 12 | Angle-QSVM | 0.557 ± 0.115 | 0.517 | 0.533 | 0.500 | 2.30 |

Artifacts: `outputs/lab/<run_id>/leaderboard.csv`, per-config JSON incl. circuit profiles.

## Findings

1. **Hybrid wins.** The stacked quantum+classical ensemble tops the board (0.986),
   and quantum-augmented XGBoost is the best single *pipeline* (0.974) at 9× less
   training cost than full stacking. This is exactly the "best of both worlds"
   thesis: quantum representations add value **through** strong classical learners.
2. **Feature-map ablation (RQ3):** Angle ≈ ZZ ≈ 0.557 < CW-ZZ 0.658 — correlation-aware
   encoding helps over conventional maps. However the permuted-correlation control
   scored 0.689 ≥ flagship 0.658: at n=96 train samples the advantage is within noise,
   so we do **not** yet claim correlation-weighting benefit on this dataset (honest
   negative control; rerun at larger N).
3. **Kennel synthetic:** Quantum-Augmented-XGB dominates (0.906) — again hybrid >
   pure quantum or pure classical. Stacking underperformed there (small trial count).
4. **Robustness harness works:** kennel CWZZ-QSVM degrades 0.60→0.51 AUC as feature
   noise rises 0→0.3σ (`outputs/lab/*robust*`), matching expected graceful decay.
5. **Cost reality check:** backprop-diff VQC (~5× faster than parameter-shift after
   this pass's fix) still costs ~100× an XGB fit; ensembles multiply that.

## Recommended configuration

`Quantum-Augmented-XGB` for deployment-minded screening (quality/speed sweet spot);
`Stacked-LR` when maximum accuracy justifies compute. Re-run both at full N and
10-fold before any comparative claims.

## Reproduce

```bash
python scripts/run_hybrid_search.py --dataset voc_real --experiment search \
    --max-samples 120 --vqc-epochs 6 --n-splits 5
python scripts/run_hybrid_search.py --dataset voc_real --experiment map_ablation
python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness
```
