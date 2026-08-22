"""End-to-end integration and API tests for QMLKit."""

import numpy as np
from fastapi.testclient import TestClient

from qmlkit.api.server import create_app
from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.evaluation.benchmark_suite import BenchmarkSuite


def test_e2e_synthetic_to_benchmark():
    gen = BiomimeticVOCGenerator(random_state=42)
    cohort = gen.generate_cohort(samples_per_class=12, cancer_types=["Healthy", "Lung_Cancer"])
    y = cohort.metadata["label_binary"].values

    suite = BenchmarkSuite(n_qubits=4, random_state=42)
    df_metrics = suite.run_benchmark(
        X_train_raw=cohort.df_features.iloc[:16],
        y_train=y[:16],
        X_test_raw=cohort.df_features.iloc[16:],
        y_test=y[16:]
    )

    assert len(df_metrics) >= 5
    assert "QSVM_BioZZ" in df_metrics["model_name"].values
    assert "SVM_RBF" in df_metrics["model_name"].values


def test_fastapi_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

    # 2. Index portal HTML
    res = client.get("/")
    assert res.status_code == 200
    assert "QMLKit" in res.text

    # 3. Prediction endpoint
    sample_payload = {
        "sample_id": "TEST_PATIENT_99",
        "patient_age": 60,
        "patient_sex": "M",
        "smoking_status": "Former",
        "sensor_readings": [float(np.sin(i * 0.1) * 0.5 + 0.8) for i in range(64)]
    }
    res = client.post("/api/v1/predict", json=sample_payload)
    assert res.status_code == 200
    data = res.json()
    assert "cancer_probability" in data
    assert "top_biomarkers" in data
    assert "biochemical_pathways" in data
