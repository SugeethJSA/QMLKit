"""Unit tests for classical baseline models and metric evaluation functions."""

import numpy as np

from qmlkit.classical.baselines import Temporal1DCNN, get_all_classical_baselines
from qmlkit.evaluation.benchmark_suite import compute_clinical_metrics


def test_classical_baselines_execution():
    X_train = np.random.randn(20, 10)
    y_train = np.array([0, 1] * 10)
    X_test = np.random.randn(6, 10)

    baselines = get_all_classical_baselines(random_state=42)
    for _name, model in baselines.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == 6
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            assert probs.shape == (6, 2)


def test_temporal_1d_cnn():
    X_tensor = np.random.rand(12, 16, 50)
    y = np.array([0, 1] * 6)

    cnn = Temporal1DCNN(n_sensors=16, timesteps=50, epochs=3, batch_size=4)
    cnn.fit(X_tensor, y)

    preds = cnn.predict(X_tensor[:4])
    probs = cnn.predict_proba(X_tensor[:4])

    assert len(preds) == 4
    assert probs.shape == (4, 2)


def test_clinical_metrics_computation():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])  # 1 FN, 1 FP, 3 TP, 3 TN
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.4, 0.3, 0.85, 0.7])

    metrics = compute_clinical_metrics(
        model_name="TestModel",
        paradigm="Quantum",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )

    assert metrics.model_name == "TestModel"
    assert metrics.accuracy == 0.75
    assert metrics.sensitivity_recall == 0.75
    assert metrics.specificity == 0.75
    assert 0.0 <= metrics.roc_auc <= 1.0
