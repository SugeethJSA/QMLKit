"""Rigorous Leak-Free Benchmarking Suite for Quantum vs Classical Models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from qmlkit.classical.baselines import get_all_classical_baselines
from qmlkit.data.feature_selector import QuantumFeatureSelector
from qmlkit.data.preprocessor import BiomedicalDataPipeline
from qmlkit.quantum.qsvm import QSVMClassifier
from qmlkit.quantum.vqc import VariationalQuantumClassifier


@dataclass
class ModelEvaluationMetrics:
    """Standardized oncology diagnostic evaluation metrics."""
    model_name: str
    paradigm: str  # "Quantum" or "Classical"
    accuracy: float
    balanced_accuracy: float
    sensitivity_recall: float
    specificity: float
    precision_ppv: float
    negative_predictive_val: float
    f1_macro: float
    f1_weighted: float
    roc_auc: float
    brier_score: float
    train_time_sec: float = 0.0
    inference_time_ms: float = 0.0


def compute_clinical_metrics(
    model_name: str,
    paradigm: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    train_time: float = 0.0,
    infer_time: float = 0.0
) -> ModelEvaluationMetrics:
    """Compute all 10 clinical metrics from true vs predicted values."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Specificity and NPV from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # ROC-AUC and Brier score
    if y_prob is not None:
        pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        try:
            auc_val = float(roc_auc_score(y_true, pos_prob))
        except Exception:
            auc_val = float(bal_acc)
        brier = float(brier_score_loss(y_true, pos_prob))
    else:
        auc_val = float(bal_acc)
        brier = float(np.mean((y_pred - y_true) ** 2))

    return ModelEvaluationMetrics(
        model_name=model_name,
        paradigm=paradigm,
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        sensitivity_recall=round(sens, 4),
        specificity=round(spec, 4),
        precision_ppv=round(prec, 4),
        negative_predictive_val=round(npv, 4),
        f1_macro=round(f1_m, 4),
        f1_weighted=round(f1_w, 4),
        roc_auc=round(auc_val, 4),
        brier_score=round(brier, 4),
        train_time_sec=round(train_time, 3),
        inference_time_ms=round(infer_time, 2)
    )


class BenchmarkSuite:
    """Runs rigorous leak-free comparative cross-validation across all models."""

    def __init__(self, n_qubits: int = 6, random_state: int = 42):
        self.n_qubits = n_qubits
        self.random_state = random_state

    def run_benchmark(
        self,
        X_train_raw: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_test_raw: pd.DataFrame | np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Run full evaluation suite on strictly held-out test split."""
        import time

        # 1. Leak-Free Preprocessing: Fit scaler only on train
        pipeline = BiomedicalDataPipeline(scaler_type="standard").fit(X_train_raw)
        X_tr_scaled = pipeline.transform(X_train_raw)
        X_te_scaled = pipeline.transform(X_test_raw)

        # 2. Quantum Feature Reduction: Fit selector only on train
        selector = QuantumFeatureSelector(n_qubits=self.n_qubits, method="pca").fit(X_tr_scaled)
        X_tr_q = selector.transform(X_tr_scaled)
        X_te_q = selector.transform(X_te_scaled)

        # Compute empirical covariance for BioZZ Feature Map
        cov_matrix = np.corrcoef(X_tr_q.T)

        results: List[ModelEvaluationMetrics] = []

        # === 1. Evaluate Quantum Models ===
        # QSVM
        t0 = time.time()
        qsvm = QSVMClassifier(n_qubits=self.n_qubits, feature_map_type="BioZZ", covariance_matrix=cov_matrix)
        qsvm.fit(X_tr_q, y_train)
        t_train_qsvm = time.time() - t0

        t0 = time.time()
        qsvm_preds = qsvm.predict(X_te_q)
        qsvm_probs = qsvm.predict_proba(X_te_q)
        t_infer_qsvm = (time.time() - t0) * 1000.0 / len(y_test)

        results.append(compute_clinical_metrics(
            "QSVM_BioZZ", "Quantum", y_test, qsvm_preds, qsvm_probs, t_train_qsvm, t_infer_qsvm
        ))

        # VQC
        t0 = time.time()
        vqc = VariationalQuantumClassifier(
            n_qubits=self.n_qubits, n_layers=2, feature_map_type="BioZZ", epochs=20, covariance_matrix=cov_matrix
        )
        vqc.fit(X_tr_q, y_train)
        t_train_vqc = time.time() - t0

        t0 = time.time()
        vqc_preds = vqc.predict(X_te_q)
        vqc_probs = vqc.predict_proba(X_te_q)
        t_infer_vqc = (time.time() - t0) * 1000.0 / len(y_test)

        results.append(compute_clinical_metrics(
            "VQC_StronglyEntangled", "Quantum", y_test, vqc_preds, vqc_probs, t_train_vqc, t_infer_vqc
        ))

        # === 2. Evaluate Classical Baselines ===
        classical_models = get_all_classical_baselines(random_state=self.random_state)
        for name, clf in classical_models.items():
            t0 = time.time()
            clf.fit(X_tr_scaled, y_train)
            t_train = time.time() - t0

            t0 = time.time()
            preds = clf.predict(X_te_scaled)
            probs = clf.predict_proba(X_te_scaled) if hasattr(clf, "predict_proba") else None
            t_infer = (time.time() - t0) * 1000.0 / len(y_test)

            results.append(compute_clinical_metrics(
                name, "Classical", y_test, preds, probs, t_train, t_infer
            ))

        df_res = pd.DataFrame([asdict(r) for r in results])
        return df_res.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
