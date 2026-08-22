"""Unit tests for biomimetic VOC generator and leak-free preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.data.feature_selector import QuantumFeatureSelector
from qmlkit.data.preprocessor import BiomedicalDataPipeline


def test_voc_generator_cohort_shapes():
    gen = BiomimeticVOCGenerator(random_state=42)
    cohort = gen.generate_cohort(samples_per_class=10, cancer_types=["Healthy", "Lung_Cancer"])

    assert len(cohort.metadata) == 20
    assert cohort.df_features.shape == (20, 64)  # 16 sensors * 4 features
    assert cohort.raw_time_series.shape == (20, 16, 100)
    assert cohort.voc_ground_truth.shape == (20, 24)
    assert set(cohort.metadata["label_cancer_type"].unique()) == {"Healthy", "Lung_Cancer"}


def test_leak_free_split_integrity():
    gen = BiomimeticVOCGenerator(random_state=42)
    cohort = gen.generate_cohort(samples_per_class=25, cancer_types=["Healthy", "Breast_Cancer"])
    y = cohort.metadata["label_binary"].values

    splits, pipeline = BiomedicalDataPipeline.create_leak_free_split(
        df_features=cohort.df_features,
        y=y,
        test_size=0.20,
        val_size=0.10,
        random_state=42
    )

    assert len(splits.X_train) == 35
    assert len(splits.X_val) == 5
    assert len(splits.X_test) == 10
    assert pipeline.is_fitted

    # Scaler mean must match X_train mean
    assert np.allclose(np.mean(splits.X_train, axis=0), 0.0, atol=1e-5)


def test_quantum_feature_selector_pca():
    gen = BiomimeticVOCGenerator(random_state=42)
    cohort = gen.generate_cohort(samples_per_class=15, cancer_types=["Healthy", "Colorectal_Cancer"])

    selector = QuantumFeatureSelector(n_qubits=6, method="pca")
    X_q = selector.fit_transform(cohort.df_features.values)

    assert X_q.shape == (30, 6)
    # Check bounded within [-pi, pi]
    assert np.all(X_q >= -np.pi - 1e-5)
    assert np.all(X_q <= np.pi + 1e-5)

    # Test inverse transformation
    recon = selector.inverse_transform(X_q)
    assert recon.shape == (30, 64)
