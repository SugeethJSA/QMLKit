"""Quantum Support Vector Machine (QSVM) and Quantum Kernel Methods."""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import pennylane as qml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from qmlkit.quantum.feature_maps import BaseFeatureMap, BioZZFeatureMap, get_feature_map


class QuantumKernel:
    """Computes exact quantum kernel Gram matrices in Hilbert space."""

    def __init__(
        self,
        feature_map: Optional[BaseFeatureMap] = None,
        n_qubits: int = 6,
        device_name: str = "default.qubit"
    ):
        self.n_qubits = n_qubits
        self.feature_map = feature_map or BioZZFeatureMap(n_qubits=n_qubits)
        self.device = qml.device(device_name, wires=self.n_qubits)

        # QNode to extract statevector |Phi(x)>
        @qml.qnode(self.device)
        def state_circuit(x: np.ndarray):
            self.feature_map.apply(x, wires=range(self.n_qubits))
            return qml.state()

        self._state_circuit = state_circuit

    def get_statevector(self, x: np.ndarray) -> np.ndarray:
        """Compute quantum statevector |Phi(x)> for single sample."""
        return np.asarray(self._state_circuit(x), dtype=complex)

    def get_statevectors(self, X: np.ndarray) -> np.ndarray:
        """Batch compute statevectors for all samples in X (Shape: N x 2^n)."""
        n_samples = X.shape[0]
        dim = 2 ** self.n_qubits
        states = np.zeros((n_samples, dim), dtype=complex)
        for i in range(n_samples):
            states[i] = self._state_circuit(X[i])
        return states

    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Quantum Kernel Matrix K(X1, X2) = |<Phi(X1) | Phi(X2)>|^2.

        Uses vectorized Hilbert inner product projection for maximum efficiency and exact mathematical fidelity.
        """
        states1 = self.get_statevectors(X1)  # (N1, 2^n)
        if X2 is None or X2 is X1:
            # Symmetric Gram Matrix: K_ij = |states1[i] . states1[j]*|^2
            inner_prod = np.dot(states1, states1.conj().T)  # (N1, N1)
            K = np.abs(inner_prod) ** 2
            # Numerical cleanup: enforce exact unit diagonal and symmetry
            np.fill_diagonal(K, 1.0)
            K = np.clip(K, 0.0, 1.0)
            return (K + K.T) / 2.0
        else:
            states2 = self.get_statevectors(X2)  # (N2, 2^n)
            inner_prod = np.dot(states1, states2.conj().T)  # (N1, N2)
            K = np.abs(inner_prod) ** 2
            return np.clip(K, 0.0, 1.0)


class QSVMClassifier:
    """Scikit-Learn compliant Quantum Support Vector Classifier."""

    def __init__(
        self,
        n_qubits: int = 6,
        feature_map_type: str = "BioZZ",
        c_param: float = 1.0,
        covariance_matrix: Optional[np.ndarray] = None
    ):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.c_param = c_param
        self.covariance_matrix = covariance_matrix

        self.feature_map = get_feature_map(
            feature_map_type,
            n_qubits=n_qubits,
            covariance_matrix=covariance_matrix
        )
        self.kernel_engine = QuantumKernel(feature_map=self.feature_map, n_qubits=n_qubits)
        # Calibrated SVC is constructed in fit() with fold-count adaptive to the
        # training class balance (SVC(probability=True) deprecated in sklearn 1.9).

        self.X_train_cached: Optional[np.ndarray] = None
        self.train_kernel_matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    @staticmethod
    def _make_calibrated_svc(c_param: float, y_train: np.ndarray, max_cv: int = 5):
        """Calibrated SVC whose internal CV adapts to the smallest class count."""
        counts = np.bincount(np.asarray(y_train).astype(int))
        min_class = int(counts[counts > 0].min()) if counts.size else 0
        base = SVC(C=c_param, kernel="precomputed", random_state=42)
        if len(counts) == 2 and min_class >= 2:
            cv = int(max(2, min(max_cv, min_class)))
            return CalibratedClassifierCV(base, ensemble=False, cv=cv)
        # Degenerate class balance in this training fold - plain SVC fallback.
        return base

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> QSVMClassifier:
        """Fit QSVM on training data by computing quantum kernel Gram matrix."""
        self.X_train_cached = np.copy(X_train)
        self.train_kernel_matrix = self.kernel_engine.compute_kernel_matrix(X_train)
        self.svc_model = self._make_calibrated_svc(self.c_param, y_train)
        self.svc_model.fit(self.train_kernel_matrix, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels for test samples."""
        if not self.is_fitted or self.X_train_cached is None:
            raise RuntimeError("Model must be fitted before predicting.")
        K_test = self.kernel_engine.compute_kernel_matrix(X_test, self.X_train_cached)
        return self.svc_model.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Compute calibrated probability estimates."""
        if not self.is_fitted or self.X_train_cached is None:
            raise RuntimeError("Model must be fitted before predicting probabilities.")
        K_test = self.kernel_engine.compute_kernel_matrix(X_test, self.X_train_cached)
        return self.svc_model.predict_proba(K_test)

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        """Compute raw distance to quantum separating hyperplane."""
        if not self.is_fitted or self.X_train_cached is None:
            raise RuntimeError("Model must be fitted before computing decision function.")
        K_test = self.kernel_engine.compute_kernel_matrix(X_test, self.X_train_cached)
        return self.svc_model.decision_function(K_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute mean accuracy on test set."""
        preds = self.predict(X_test)
        return float(np.mean(preds == y_test))

    def save(self, filepath: str) -> None:
        """Persist trained model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> QSVMClassifier:
        """Load trained model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
