"""Quantum Kernel SHAP for Explainable Quantum AI."""

from __future__ import annotations

from typing import Any, Callable, List, Optional
import numpy as np


class QuantumKernelSHAP:
    """Computes Shapley values for Quantum Classifiers (QSVM, VQC, QCNN)."""

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background_data: np.ndarray,
        n_samples_background: int = 25
    ):
        """Initialize Quantum Kernel SHAP explainer with a reference background subset."""
        try:
            import shap
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "QuantumKernelSHAP requires the optional 'shap' package. "
                "Install it via `pip install shap` or `pip install qmlkit[dev]`."
            ) from exc

        if len(background_data) > n_samples_background:
            indices = np.random.default_rng(42).choice(
                len(background_data), size=n_samples_background, replace=False
            )
            bg = background_data[indices]
        else:
            bg = background_data

        self.predict_fn = predict_fn
        self.background_data = bg
        self.explainer = shap.KernelExplainer(self._predict_wrapper, self.background_data)

    def _predict_wrapper(self, x: np.ndarray) -> np.ndarray:
        """Wrapper ensuring probability outputs are 2D for SHAP."""
        probs = self.predict_fn(x)
        if probs.ndim == 1:
            return np.vstack([1.0 - probs, probs]).T
        elif probs.shape[1] == 2:
            return probs[:, 1]  # Return positive class probability
        return probs

    def explain(self, x_instance: np.ndarray, n_evals: int = 150) -> np.ndarray:
        """Compute Shapley attribution vector for single sample or batch.

        Returns array of shape (N, n_features).
        """
        x_mat = np.atleast_2d(x_instance)
        shap_vals = self.explainer.shap_values(x_mat, nsamples=n_evals, silent=True)
        if isinstance(shap_vals, list):
            # Binary classification returned list of [class0, class1]
            return np.asarray(shap_vals[1])
        return np.asarray(shap_vals)
