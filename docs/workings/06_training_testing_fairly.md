# Training and Testing Fairly

Machine-learning results are only as trustworthy as the plumbing around them.
These are the rules QMLKit enforces, why each exists, and where the code lives.

## Rule 1: Never let test data peek

If you normalise numbers using averages computed from *all* data — including
the part you'll test on — your score is quietly inflated. Every scaler,
feature-selector, PCA, and correlation matrix in QMLKit is fitted **only on the
training portion** and then applied unchanged to the test portion.
(`src/qmlkit/data/preprocessor.py`, `lab/pipeline.py`.)

## Rule 2: One split is an anecdote; k splits are evidence

With small datasets, one lucky/unlucky train-test split can swing results
wildly. We use **stratified k-fold cross-validation**: slice the data into k=5
parts keeping class balance identical in each, train five models each leaving a
different part out, and report mean ± spread. All compared configurations share
the *exact same* fold boundaries so differences come from the models, not luck.
(`lab/cv.py`.)

## Rule 3: Rebuild the pipeline inside every fold

The BioZZ correlation matrix, PCA rotation, and scalers are all re-derived from
each fold's training slice. This matches how the model would be deployed (train
once, freeze, then judge new samples) and is stricter than most papers bother
with.

## Rule 4: Ablate to attribute

"Is the whole system good?" is less useful than "**which part** earns its keep?"
The lab can:

- drop one sensor family at a time (**modality ablation**) → shows what each
  modality contributes;
- swap the quantum encoding Angle ↔ plain-ZZ ↔ BioZZ ↔ shuffled-BioZZ
  (**feature-map ablation**) → isolates what correlation-awareness adds.

## Rule 5: Stress-test before believing

Real kennels have noisy cables and flaky sensors. The robustness runner adds
fake noise of increasing size to features, or blanks out random ones, and
reports how gently (or badly) accuracy falls. A model whose score collapses with
a 5 % perturbation isn't ready for a kennel.

## Rule 6: Report the honest metrics clinicians care about

Accuracy alone lies on imbalanced data. We always report **sensitivity** (of the
sick samples, how many did we catch?), **specificity** (of healthy samples, how
many did we correctly clear?), **ROC-AUC**, F1, and Brier score (are the
confidence levels calibrated?). `evaluation/benchmark_suite.py` computes all of
them from a confusion matrix + probability outputs.

## The honesty ledger

Things our own runs currently show that we will not sugar-coat: quantum-only
models still trail classical baselines on the real VOC dataset; the permuted-
correlation control matched flagship BioZZ within noise at n≈96 training
samples; synthetic kennel results are pipeline demos, not biology. See
[`docs/reports/hybrid_search_report.md`](../reports/hybrid_search_report.md).
