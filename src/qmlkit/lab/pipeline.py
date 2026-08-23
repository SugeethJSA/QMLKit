"""Composable hybrid (quantum + classical) pipelines for the training lab.

A ``PipelineSpec`` describes one point in the combination space:

    reduction  : none | pca | mutual_info | autoencoder
    embedding  : none | angle | zz | cwzz | cwzz_permuted
                 (how quantum maps weight pairwise interactions;
                  cwzz uses train-derived correlations, cwzz_permuted is the
                  manuscript's correlation-control experiment)
    head       : qsvm | vqc | qcnn | logistic_regression | svm_rbf | svm_linear |
                 random_forest | xgboost | mlp | quantum_augmented_xgb

Every stage is fitted strictly on the training fold (leak-free by construction),
mirroring the manuscript's protocol (normalisation, correlation matrix, and
reduction all derive from training data only).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:  # optional dependency
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover - env dependent
    HAS_XGB = False

from qmlkit.data.feature_selector import QuantumFeatureSelector
from qmlkit.evaluation.benchmark_suite import compute_qubit_covariance
from qmlkit.evaluation.hardware_profiler import QuantumHardwareProfiler
from qmlkit.quantum.qcnn import QuantumConvolutionalClassifier
from qmlkit.quantum.qsvm import QSVMClassifier
from qmlkit.quantum.vqc import VariationalQuantumClassifier

from qmlkit.quantum.kernel_features import QuantumKernelFeatureTransformer

REDUCTION_METHODS = ("none", "pca", "mutual_info", "autoencoder")
EMBEDDINGS = ("none", "angle", "zz", "cwzz", "cwzz_permuted")
CLASSICAL_HEADS = (
    "logistic_regression",
    "svm_rbf",
    "svm_linear",
    "random_forest",
    "xgboost",
    "mlp",
)


def _permute_correlation(cov: np.ndarray, seed: int) -> np.ndarray:
    """Correlation-control experiment: shuffle off-diagonal entries (§VII-D)."""
    rng = np.random.default_rng(seed)
    n = cov.shape[0]
    iu = np.triu_indices(n, k=1)
    upper = np.asarray(cov)[iu].copy()
    rng.shuffle(upper)
    mat = np.zeros((n, n))
    mat[iu] = upper
    mat[(iu[1], iu[0])] = upper
    np.fill_diagonal(mat, 1.0)
    return mat


def _make_head(spec: "PipelineSpec", cov_matrix: Optional[np.ndarray]) -> Any:
    """Instantiate the head classifier described by the spec."""
    seed = spec.seed
    embedding_map = None if spec.embedding == "none" else spec.embedding

    if spec.head == "qsvm":
        map_type = {
            "angle": "Angle",
            "zz": "ZZ",
            "cwzz": "BioZZ",
            "cwzz_permuted": "BioZZ",
        }.get(embedding_map or "", "ZZ")

        return QSVMClassifier(
            n_qubits=spec.n_components,
            feature_map_type=map_type,
            covariance_matrix=cov_matrix,
        )
    if spec.head == "vqc":
        return VariationalQuantumClassifier(
            n_qubits=spec.n_components,
            n_layers=2,
            feature_map_type={
                "angle": "Angle",
                "zz": "ZZ",
                "cwzz": "BioZZ",
                "cwzz_permuted": "BioZZ",
            }.get(embedding_map or "", "ZZ"),            epochs=spec.vqc_epochs,
            covariance_matrix=cov_matrix,
        )
    if spec.head == "qcnn":
        return QuantumConvolutionalClassifier(
            n_qubits=max(2, spec.n_components),
            feature_map_type="BioZZ",
            covariance_matrix=cov_matrix,
            epochs=spec.vqc_epochs,
        )
    if spec.head == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    if spec.head == "svm_rbf":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=seed)
    if spec.head == "svm_linear":
        from sklearn.svm import SVC

        return SVC(kernel="linear", C=1.0, probability=True, random_state=seed)
    if spec.head == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=150, max_depth=8, random_state=seed)
    if spec.head in (
        "xgboost",
        "quantum_augmented_xgb",
        "quantum_kernel_xgb",
    ):
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.8, eval_metric="logloss", random_state=seed,
            )
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=seed
        )
    if spec.head == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
    raise ValueError(f"Unknown head: {spec.head}")


@dataclass
class PipelineSpec:
    """Declarative description of one hybrid pipeline configuration."""

    name: str
    reduction: str = "pca"
    n_components: int = 6
    embedding: str = "cwzz"
    head: str = "qsvm"
    covariance_mode: str = "train"  # train | identity | permuted handled via embedding suffix
    vqc_epochs: int = 8
    n_landmarks: int = 12
    seed: int = 42
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "reduction": self.reduction,
            "n_components": self.n_components,
            "embedding": self.embedding,
            "head": self.head,
            "covariance_mode": self.covariance_mode,
            "vqc_epochs": self.vqc_epochs,
            "n_landmarks": self.n_landmarks,
            "seed": self.seed,
            **self.meta,
        }

class HybridPipeline:
    """Leak-free composition: scaler -> reduction -> [quantum head | classical head].

    ``fit`` computes the correlation matrix from training data only; the same
    matrix is reused unchanged at transform time (manuscript §VI-C).
    """

    def __init__(self, spec: PipelineSpec):
        self.spec = spec
        self.scaler: Optional[StandardScaler] = None
        self.selector: Optional[QuantumFeatureSelector] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.head: Optional[Any] = None
        self.augmenter: Optional[VariationalQuantumClassifier] = None
        self.quantum_transformer: Optional[QuantumKernelFeatureTransformer] = None
        self.impute_medians: Optional[np.ndarray] = None  # train-set medians (NaN handling)
        self.is_fitted = False

    # -- internals ---------------------------------------------------------
    def _impute(self, X_arr: np.ndarray, fitted: bool) -> np.ndarray:
        X = np.array(X_arr, dtype=float, copy=True)  # writable copy (DataFrame views are read-only)
        if not np.isnan(X).any():
            return X
        if fitted:
            medians = self.impute_medians
            if medians is None:
                raise RuntimeError("Imputer state missing.")
        else:
            with np.errstate(all="ignore"):
                medians = np.nanmedian(X, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            self.impute_medians = medians
        idx = np.where(np.isnan(X))
        X[idx] = medians[idx[1]]
        return X
    def _reduce(self, selector: QuantumFeatureSelector, X_scaled: np.ndarray) -> np.ndarray:
        if self.spec.reduction == "none":
            n = min(self.spec.n_components, X_scaled.shape[1]) if X_scaled.shape[1] > self.spec.n_components else X_scaled.shape[1]
            return X_scaled[:, :n]
        return selector.transform(X_scaled)

    def _describe(self) -> Dict[str, Any]:
        record = QuantumHardwareProfiler.describe_model(
            feature_map_type={"cwzz": "BioZZ", "cwzz_permuted": "BioZZ"}.get(
                self.spec.embedding, self.spec.embedding.title()
            ),
            n_qubits=self.spec.n_components,
            reps=2,
            variational=(
                {"ansatz_type": "StronglyEntangling", "n_qubits": self.spec.n_components, "n_layers": 2}
                if self.spec.head in ("vqc", "quantum_augmented_xgb")
                else None
            ),
        )
        record["spec"] = self.spec.to_dict()
        return record

    # -- public API --------------------------------------------------------
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> "HybridPipeline":
        spec = self.spec
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_arr = self._impute(X_arr, fitted=False)

        self.scaler = StandardScaler().fit(X_arr)          # train-only normalisation
        X_scaled = self.scaler.transform(X_arr)

        if spec.reduction != "none":
            self.selector = QuantumFeatureSelector(
                n_qubits=spec.n_components, method=spec.reduction
            ).fit(X_scaled, y)
        X_reduced = self._reduce(self.selector, X_scaled)

        # Correlation matrix for CW-ZZ embeddings (training data only).
        if spec.embedding in ("cwzz", "cwzz_permuted"):
            base_cov = compute_qubit_covariance(X_scaled, self.selector) if self.selector else np.eye(X_reduced.shape[1])
            n = X_reduced.shape[1]
            base_cov = np.atleast_2d(base_cov)
            if base_cov.shape != (n, n):
                base_cov = np.eye(n)
            if spec.embedding == "cwzz_permuted":
                self.cov_matrix = _permute_correlation(base_cov, seed=spec.seed)
            else:
                self.cov_matrix = base_cov
        else:
            self.cov_matrix = None

        if spec.head == "quantum_kernel_xgb":
            map_type = {
                "angle": "Angle",
                "zz": "ZZ",
                "cwzz": "BioZZ",
                "cwzz_permuted": "BioZZ",
            }.get(spec.embedding, "ZZ")

            self.quantum_transformer = QuantumKernelFeatureTransformer(
                n_qubits=X_reduced.shape[1],
                feature_map_type=map_type,
                covariance_matrix=self.cov_matrix,
                n_landmarks=spec.n_landmarks,
                seed=spec.seed,
            )

            quantum_features = self.quantum_transformer.fit_transform(
                X_reduced,
                y,
            )

            X_head = np.hstack([
                X_reduced,
                quantum_features,
            ])

            plain = dataclasses.replace(
                spec,
                head="xgboost",
            )

            self.head = _make_head(
                plain,
                cov_matrix=None,
            )

        elif spec.head == "quantum_augmented_xgb":
            self.augmenter = VariationalQuantumClassifier(
                n_qubits=X_reduced.shape[1],
                n_layers=1,
                feature_map_type="BioZZ" if spec.embedding.startswith("cwzz") else "ZZ",
                epochs=spec.vqc_epochs,
                covariance_matrix=self.cov_matrix,
            )

            self.augmenter.fit(X_reduced, y)

            aug_proba = self.augmenter.predict_proba(
                X_reduced
            )[:, 1].reshape(-1, 1)

            X_head = np.hstack([
                X_scaled,
                aug_proba,
            ])

            plain = dataclasses.replace(
                spec,
                head="xgboost",
            )

            self.head = _make_head(
                plain,
                cov_matrix=None,
            )

        else:
            self.head = _make_head(
                spec,
                cov_matrix=self.cov_matrix,
            )

            X_head = X_reduced

        self.head.fit(X_head, y)
        self.is_fitted = True
        return self

        self.head.fit(X_head, y)
        self.is_fitted = True
        return self

    def transform_input(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply stored transforms; returns (scaled_full, reduced)."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_arr = self._impute(X_arr, fitted=True)
        X_scaled = self.scaler.transform(X_arr)
        X_reduced = self._reduce(self.selector, X_scaled)
        return X_scaled, X_reduced

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_scaled, X_reduced = self.transform_input(X)

        if self.spec.head == "quantum_kernel_xgb":
            if self.quantum_transformer is None:
                raise RuntimeError("Quantum kernel transformer is missing.")

            quantum_features = self.quantum_transformer.transform(
                X_reduced
            )

            X_head = np.hstack([
                X_reduced,
                quantum_features,
            ])

            proba = self.head.predict_proba(X_head)

        elif self.spec.head == "quantum_augmented_xgb":
            aug = self.augmenter.predict_proba(
                X_reduced
            )[:, 1].reshape(-1, 1)

            proba = self.head.predict_proba(
                np.hstack([
                    X_scaled,
                    aug,
                ])
            )

        else:
            proba = self.head.predict_proba(
                X_reduced
            )

        return (
            proba
            if proba.ndim == 2
            else np.vstack([1 - proba, proba]).T
        )

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (np.argmax(self.predict_proba(X), axis=1)).astype(int)

    def describe(self) -> Dict[str, Any]:
        return self._describe()
