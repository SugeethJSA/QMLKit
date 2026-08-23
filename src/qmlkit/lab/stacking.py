"""Ensemble strategies combining hybrid pipelines.

- ``SoftVotingEnsemble``  : average predicted probabilities of fitted members.
- ``StackingEnsemble``    : out-of-fold (OOF) base predictions -> logistic-regression meta learner,
  avoiding meta-learner leakage (predictions for training are produced on held-out folds).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class SoftVotingEnsemble:
    """Averages ``predict_proba`` outputs across member estimators (or pipelines)."""

    def __init__(self, members: Sequence[Any], weights: Optional[Sequence[float]] = None):
        self.members = list(members)
        self.weights = np.asarray(weights) if weights is not None else None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "SoftVotingEnsemble":
        for m in self.members:
            m.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.members]
        stacked = np.stack([p[:, 1] if p.ndim == 2 else p for p in probas], axis=0)
        if self.weights is not None:
            w = self.weights / self.weights.sum()
            avg = np.tensordot(w, stacked, axes=1)
        else:
            avg = stacked.mean(axis=0)
        return np.vstack([1 - avg, avg]).T

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(int)


class StackingEnsemble:
    """Trains a logistic-regression meta learner on out-of-fold base predictions."""

    def __init__(
        self,
        base_specs: Sequence[Any],
        make_pipeline_fn,
        n_splits: int = 5,
        seed: int = 42,
    ):
        """
        ``base_specs``  : sequence of PipelineSpec-like objects.
        ``make_pipeline_fn(spec) -> estimator`` : factory producing a fresh,
        unfitted pipeline/estimator for each fold (keeps every fold leak-free).
        """
        self.base_specs = list(base_specs)
        self.make_pipeline_fn = make_pipeline_fn
        self.n_splits = n_splits
        self.seed = seed
        self.fitted_members: List[Any] = []
        self.meta: Optional[LogisticRegression] = None
        self.is_fitted = False

    def _oof_matrix(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> np.ndarray:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(y), len(self.base_specs)))
        y_arr = np.asarray(y)
        for train_idx, val_idx in skf.split(np.zeros(len(y_arr)), y_arr):
            X_tr, X_val = _slice(X, train_idx), _slice(X, val_idx)
            for j, spec in enumerate(self.base_specs):
                est = self.make_pipeline_fn(spec)
                est.fit(X_tr, y_arr[train_idx])
                proba = est.predict_proba(X_val)
                oof[val_idx, j] = proba[:, 1] if proba.ndim == 2 else proba
        return oof

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        y_arr = np.asarray(y)
        oof = self._oof_matrix(X, y_arr)

        # Refit each base member on the full training set for inference.
        self.fitted_members = []
        for spec in self.base_specs:
            est = self.make_pipeline_fn(spec)
            est.fit(X, y_arr)
            self.fitted_members.append(est)

        self.meta = LogisticRegression(max_iter=1000, random_state=self.seed)
        self.meta.fit(oof, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted first.")
        base = np.column_stack(
            [
                (m.predict_proba(X)[:, 1] if m.predict_proba(X).ndim == 2 else m.predict_proba(X))
                for m in self.fitted_members
            ]
        )
        return self.meta.predict_proba(base)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(int)


def _slice(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return np.asarray(X)[idx]
