"""Quantum-kernel feature transformer for hybrid quantum-classical models."""

from __future__ import annotations

from typing import Optional

import numpy as np

from qmlkit.quantum.feature_maps import get_feature_map
from qmlkit.quantum.qsvm import QuantumKernel


class QuantumKernelFeatureTransformer:
    """Transform reduced classical features into quantum-kernel similarities."""

    def __init__(
        self,
        n_qubits: int,
        feature_map_type: str = "BioZZ",
        covariance_matrix: Optional[np.ndarray] = None,
        n_landmarks: int = 12,
        seed: int = 42,
    ):
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1.")

        if n_landmarks < 1:
            raise ValueError("n_landmarks must be at least 1.")

        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.covariance_matrix = covariance_matrix
        self.n_landmarks = n_landmarks
        self.seed = seed

        if covariance_matrix is not None:
            cov = np.asarray(covariance_matrix, dtype=float)
            expected = (n_qubits, n_qubits)

            if cov.shape != expected:
                raise ValueError(
                    f"covariance_matrix must have shape {expected}, got {cov.shape}."
                )

            self.covariance_matrix = cov

        self.feature_map = get_feature_map(
            feature_map_type,
            n_qubits=n_qubits,
            covariance_matrix=self.covariance_matrix,
        )

        self.kernel_engine = QuantumKernel(
            feature_map=self.feature_map,
            n_qubits=n_qubits,
        )

        self.landmarks_: Optional[np.ndarray] = None
        self.landmark_indices_: Optional[np.ndarray] = None
        self.landmark_states_: Optional[np.ndarray] = None
        self.is_fitted = False

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features, got {X.shape[1]}."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        return X

    def _select_landmarks(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
    ) -> np.ndarray:
        count = min(self.n_landmarks, X.shape[0])
        rng = np.random.default_rng(self.seed)

        if y is None:
            return rng.choice(
                X.shape[0],
                size=count,
                replace=False,
            )

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if len(y) != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        classes = np.unique(y)

        if len(classes) < 2:
            return rng.choice(
                X.shape[0],
                size=count,
                replace=False,
            )

        pools = []

        for cls in classes:
            indices = np.flatnonzero(y == cls)
            indices = rng.permutation(indices)
            pools.append(list(indices))

        selected = []

        while len(selected) < count:
            added = False

            for pool in pools:
                if len(selected) >= count:
                    break

                if pool:
                    selected.append(pool.pop())
                    added = True

            if not added:
                break

        return np.asarray(selected, dtype=int)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> QuantumKernelFeatureTransformer:
        """Choose training-only landmarks and cache their quantum states."""

        X = self._check_X(X)

        indices = self._select_landmarks(X, y)

        self.landmark_indices_ = indices
        self.landmarks_ = np.copy(X[indices])

        self.landmark_states_ = self.kernel_engine.get_statevectors(
            self.landmarks_
        )

        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return quantum-kernel similarity features against stored landmarks."""

        if (
            not self.is_fitted
            or self.landmarks_ is None
            or self.landmark_states_ is None
        ):
            raise RuntimeError(
                "QuantumKernelFeatureTransformer must be fitted before transform()."
            )

        X = self._check_X(X)

        states = self.kernel_engine.get_statevectors(X)

        inner_products = np.dot(
            states,
            self.landmark_states_.conj().T,
        )

        kernel_features = np.abs(inner_products) ** 2

        return np.clip(kernel_features, 0.0, 1.0)

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit the transformer and return quantum features for X."""

        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Return names for generated quantum similarity features."""

        if not self.is_fitted or self.landmarks_ is None:
            raise RuntimeError(
                "QuantumKernelFeatureTransformer must be fitted first."
            )

        return [
            f"qkernel_{i}"
            for i in range(self.landmarks_.shape[0])
        ]