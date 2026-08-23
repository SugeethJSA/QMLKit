"""Stratified k-fold evaluation harness for hybrid pipelines.

Every fold builds a *fresh* pipeline: scaler, feature reduction, and the
CW-ZZ correlation matrix are all recomputed from that fold's training split
only — identical partitions are shared across all compared configurations
(manuscript §VII-B/F).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from qmlkit.evaluation.benchmark_suite import compute_clinical_metrics


@dataclass
class FoldResult:
    fold: int
    metrics: Dict[str, float]
    train_time_s: float
    infer_time_ms_per_sample: float


def cross_validate_config(
    spec,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    make_pipeline_fn: Callable,
    n_splits: int = 5,
    seed: int = 42,
) -> List[FoldResult]:
    """Evaluate one configuration over stratified folds; returns per-fold metrics."""
    y_arr = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: List[FoldResult] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr)):
        import time

        X_tr, y_tr = _slice(X, train_idx), y_arr[train_idx]
        X_val, y_val = _slice(X, val_idx), y_arr[val_idx]

        pipeline = make_pipeline_fn(spec)
        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        train_time = time.time() - t0

        t0 = time.time()
        proba = pipeline.predict_proba(X_val)
        infer_ms = (time.time() - t0) * 1000.0 / max(1, len(val_idx))
        preds = np.argmax(proba if proba.ndim == 2 else np.vstack([1 - proba, proba]).T, axis=1)

        metrics = compute_clinical_metrics(
            model_name=spec.name,
            paradigm="Hybrid",
            y_true=y_val,
            y_pred=preds,
            y_prob=proba if proba.ndim == 2 else np.vstack([1 - proba, proba]).T,
            train_time=train_time,
            infer_time=infer_ms / 1000.0,
        )
        from dataclasses import asdict

        results.append(
            FoldResult(
                fold=fold_idx,
                metrics={k: v for k, v in asdict(metrics).items() if isinstance(v, (int, float))},
                train_time_s=train_time,
                infer_time_ms_per_sample=infer_ms,
            )
        )
    return results


def summarise_folds(name: str, folds: List[FoldResult]) -> Dict[str, float]:
    """mean +/- std across folds for the headline clinical metrics."""
    keys = [
        "accuracy", "balanced_accuracy", "sensitivity_recall", "specificity",
        "precision_ppv", "negative_predictive_val", "f1_macro", "roc_auc",
        "brier_score",
    ]
    summary: Dict[str, float] = {"config": name, "n_folds": len(folds)}
    for key in keys:
        values = [f.metrics.get(key, np.nan) for f in folds]
        values = [v for v in values if not np.isnan(v)]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
        summary[f"{key}_std"] = float(np.std(values)) if values else float("nan")
    summary["train_time_s_mean"] = float(np.mean([f.train_time_s for f in folds]))
    return summary


def _slice(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return np.asarray(X)[idx]
