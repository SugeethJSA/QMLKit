"""Unit tests for Quantum Explainability and VOC Biomarker Attribution."""

import numpy as np

from qmlkit.data.feature_selector import QuantumFeatureSelector
from qmlkit.explainability.biomarker_mapper import BiomarkerAttributionEngine


def test_biomarker_attribution_engine():
    X_train = np.random.randn(30, 64)
    selector = QuantumFeatureSelector(n_qubits=6, method="pca").fit(X_train)

    engine = BiomarkerAttributionEngine(feature_selector=selector)
    latent_shap = np.array([0.2, -0.1, 0.4, 0.05, -0.3, 0.15])

    chem_shap = engine.map_latent_to_chemical(latent_shap)
    assert chem_shap.shape == (1, 24)

    explanation = engine.generate_explanation(
        sample_id="SMPL_TEST_001",
        cancer_probability=0.88,
        latent_shap=latent_shap
    )

    assert explanation.sample_id == "SMPL_TEST_001"
    assert "Cancer Positive" in explanation.predicted_class
    assert len(explanation.top_biomarkers) == 5
    assert "Lipid_Peroxidation_Aldehydes" in explanation.pathway_contributions
    assert len(explanation.summary_text) > 20
