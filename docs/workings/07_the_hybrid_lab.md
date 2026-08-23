# The Hybrid Training Lab

The Training Lab is a systematic "mix-and-match kitchen" for models. Instead of
hand-picking one pipeline and hoping, it builds many recipes from the same
three slots and ranks them under identical test conditions.

## The three slots

```
   SLOT 1              SLOT 2                 SLOT 3
 REDUCTION    →     QUANTUM EMBEDDING   →      HEAD
 (squeeze the       (how numbers enter        (who makes the final call)
  feature count      quantum space)
  to qubit budget)
```

| Slot | Options |
|---|---|
| Reduction | none · PCA · mutual-information top-k · autoencoder |
| Embedding | none (skip quantum) · Angle · ZZ · **BioZZ/CW-ZZ** · BioZZ-with-shuffled-correlations *(the control)* |
| Head | QSVM · VQC · QCNN · logistic regression · SVM-RBF/linear · Random Forest · XGBoost · MLP · **quantum-augmented XGB** |

## The curated menu (~12 recipes)

1. PCA → BioZZ-QSVM — the paper's flagship pure-quantum path
2. PCA → ZZ-QSVM and 3. Angle-QSVM — conventional-map baselines
4. PCA → shuffled-BioZZ-QSVM — correlation control
5. Mutual-info → BioZZ-QSVM — chemically-selected inputs
6. Autoencoder → VQC — learned compression into the quantum register
7. **Quantum-Augmented-XGB** — VQC opinion as an extra feature for XGBoost
8. QCNN — hierarchical quantum pattern reader
9–10. XGBoost controls (PCA / raw) — what classical alone achieves
11. **Soft voting** of four mixed models
12. **Stacking**: RF + XGB + QSVM + VQC opinions → small meta-learner

Every recipe runs through the same stratified 5-fold cross-validation with
identical fold boundaries, so the leaderboard is apples-to-apples.

## How to read the leaderboard

- `roc_auc_mean ± std` is the headline: how well the model separates classes,
  averaged over folds; the spread tells you how stable that estimate is.
- `train_time_s_mean` is the reality check — a 0.98 AUC that takes 2 minutes per
  fold may not be worth it over a 0.97 at 15 seconds.
- Compare any hybrid against rows 9–10: beating *Raw-XGBoost* is the bar a
  quantum component must clear to justify itself.

## Current champion

On the real VOC dataset: **Stacked-LR ensemble 0.986** and
**Quantum-Augmented-XGB 0.974** — hybrids beat every pure approach. Full tables:
[`docs/reports/hybrid_search_report.md`](../reports/hybrid_search_report.md).

## Running it

From the console's **Training Lab** page (pick dataset/experiment/budget and
press Launch), or:

```bash
python scripts/run_hybrid_search.py --dataset voc_real --experiment search --max-samples 120
python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness
```

Results land in `outputs/lab/<timestamp>/` as `leaderboard.csv` plus a JSON
dossier per recipe (including its circuit-gate profile for the manuscript).
