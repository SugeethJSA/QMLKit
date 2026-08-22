"""Unit tests for Quantum Circuits, Feature Maps, QSVM, and VQC."""

import numpy as np
import pytest

from qmlkit.quantum.feature_maps import BioZZFeatureMap, PauliZZFeatureMap, AngleFeatureMap
from qmlkit.quantum.qsvm import QSVMClassifier, QuantumKernel
from qmlkit.quantum.vqc import VariationalQuantumClassifier


def test_quantum_kernel_properties():
    n_qubits = 4
    X = np.random.uniform(-np.pi, np.pi, size=(5, n_qubits))

    feat_map = BioZZFeatureMap(n_qubits=n_qubits)
    kernel = QuantumKernel(feature_map=feat_map, n_qubits=n_qubits)
    K = kernel.compute_kernel_matrix(X)

    # 1. Symmetry: K_ij == K_ji
    assert np.allclose(K, K.T, atol=1e-6)
    # 2. Unit Diagonal: K_ii == 1.0
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    # 3. Bounds: 0 <= K_ij <= 1
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0 + 1e-6)


def test_qsvm_classifier_fit_predict():
    n_qubits = 4
    X_train = np.random.uniform(-np.pi, np.pi, size=(12, n_qubits))
    y_train = np.array([0, 1] * 6)
    X_test = np.random.uniform(-np.pi, np.pi, size=(4, n_qubits))

    qsvm = QSVMClassifier(n_qubits=n_qubits, feature_map_type="BioZZ", c_param=1.0)
    qsvm.fit(X_train, y_train)

    preds = qsvm.predict(X_test)
    probs = qsvm.predict_proba(X_test)

    assert len(preds) == 4
    assert probs.shape == (4, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_vqc_training():
    vqc = VariationalQuantumClassifier(
    n_qubits=n_qubits,
    n_layers=1,
    epochs=3,
    learning_rate=0.05
    )

    initial_weights = vqc._init_weights().copy()

    vqc.fit(X_train, y_train)

    assert vqc.is_fitted
    assert len(vqc.loss_history) == 3
    assert not np.allclose(initial_weights, vqc.weights)

    preds = vqc.predict(X_test)
    assert len(preds) == 8
