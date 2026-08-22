"""FastAPI Microservice and Interactive Screening Portal for QMLKit."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from qmlkit.api.training_service import TrainRequest, execute_training_job
from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.data.feature_selector import QuantumFeatureSelector
from qmlkit.data.preprocessor import BiomedicalDataPipeline
from qmlkit.evaluation.benchmark_suite import BenchmarkSuite, compute_qubit_covariance
from qmlkit.explainability.biomarker_mapper import BiomarkerAttributionEngine
from qmlkit.quantum.qsvm import QSVMClassifier


# Pydantic Request Schemas
class PredictionRequest(BaseModel):
    sample_id: str = "SMPL_TEST_001"
    patient_age: int = 58
    patient_sex: str = "M"
    smoking_status: str = "Former"
    sensor_readings: List[float] = Field(
        description="64 extracted kinetic features (4 per sensor across 16 sensors)",
        default_factory=lambda: [float(np.sin(i) * 0.5 + 0.8) for i in range(64)]
    )


class BenchmarkRequest(BaseModel):
    n_samples_per_class: int = Field(default=80, ge=20, le=300)
    target_cancer: str = Field(default="Lung_Cancer")
    n_qubits: int = Field(default=6, ge=4, le=12)


def create_app() -> FastAPI:
    """Factory creating configured FastAPI instance."""
    # In-memory cached model state for rapid inference
    app_state: Dict[str, Any] = {
        "pipeline": None,
        "selector": None,
        "qsvm_model": None,
        "explainer_engine": None,
        "shap_explainer": None,
        "demo_samples": None,
        "is_ready": False
    }

    def _ensure_default_models():
        if app_state["is_ready"]:
            return
        generator = BiomimeticVOCGenerator(random_state=42)
        cohort = generator.generate_cohort(samples_per_class=60, cancer_types=["Healthy", "Lung_Cancer"])

        y = cohort.metadata["label_binary"].values

        # Strict leak-free partition: scalers/selector/kernel see TRAIN only.
        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
        )
        df_train = cohort.df_features.iloc[idx_train]
        df_test = cohort.df_features.iloc[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        pipeline = BiomedicalDataPipeline(scaler_type="standard").fit(df_train)
        X_tr_scaled = pipeline.transform(df_train)
        X_te_scaled = pipeline.transform(df_test)

        selector = QuantumFeatureSelector(n_qubits=6, method="pca").fit(X_tr_scaled)
        X_tr_q = selector.transform(X_tr_scaled)
        X_te_q = selector.transform(X_te_scaled)

        cov = compute_qubit_covariance(X_tr_scaled, selector)
        qsvm = QSVMClassifier(n_qubits=6, feature_map_type="BioZZ", covariance_matrix=cov).fit(X_tr_q, y_train)
        attr_engine = BiomarkerAttributionEngine(feature_selector=selector)

        # Optional Kernel-SHAP explainer (requires `pip install shap`); the
        # predict endpoint falls back to a fast delta approximation otherwise.
        shap_explainer = None
        try:
            from qmlkit.explainability.quantum_shap import QuantumKernelSHAP
            shap_explainer = QuantumKernelSHAP(
                lambda x: QSVMClassifier.predict_proba(qsvm, np.atleast_2d(x)),
                background_data=X_te_q,
                n_samples_background=16,
            )
        except ImportError:
            pass

        # Demo presets drawn strictly from held-out test rows.
        test_labels = list(y_test)
        healthy_idx = test_labels.index(0) if 0 in test_labels else 0
        lung_idx = next((i for i, v in enumerate(test_labels) if v == 1), len(test_labels) - 1)
        demo_samples = {
            "healthy_control": df_test.iloc[healthy_idx].to_numpy(dtype=float).tolist(),
            "lung_positive": df_test.iloc[lung_idx].to_numpy(dtype=float).tolist(),
        }

        app_state["pipeline"] = pipeline
        app_state["selector"] = selector
        app_state["qsvm_model"] = qsvm
        app_state["explainer_engine"] = attr_engine
        app_state["shap_explainer"] = shap_explainer
        app_state["is_ready"] = True
        app_state["demo_samples"] = demo_samples

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _ensure_default_models()
        yield

    app = FastAPI(
        title="QMLKit Clinical Screening API",
        description="Hybrid Quantum Machine Learning Platform for Early Cancer Detection via Canine Olfactory VOC Sensing",
        version="0.1.0",
        lifespan=lifespan
    )

    @app.get("/api/v1/health")
    async def health_check():
        return {
            "status": "HEALTHY",
            "quantum_backend": "PennyLane (default.qubit)",
            "models_loaded": app_state["is_ready"],
            "platform": "QMLKit PS-26139"
        }

    @app.get("/api/v1/demo-sample/{preset}")
    async def get_demo_sample(preset: str):
        _ensure_default_models()

        demo_samples = app_state["demo_samples"]

        if preset not in demo_samples:
            raise HTTPException(
                status_code=404,
                detail="Unknown demo sample preset."
            )

        return {
            "preset": preset,
            "sensor_readings": demo_samples[preset]
        }

    

    @app.post("/api/v1/predict")
    async def predict_sample(req: PredictionRequest, deep_explain: bool = False):
        _ensure_default_models()
        if len(req.sensor_readings) != 64:
            raise HTTPException(status_code=400, detail="Must provide exactly 64 sensor features (16 sensors x 4 features).")

        raw_vec = np.array(req.sensor_readings).reshape(1, -1)
        scaled = app_state["pipeline"].transform(raw_vec)
        q_feat = app_state["selector"].transform(scaled)

        prob_pos = float(app_state["qsvm_model"].predict_proba(q_feat)[0, 1])
        pred_label = "Malignant / High Risk" if prob_pos >= 0.5 else "Healthy / Low Risk"

        # Attribution: exact Kernel-SHAP on demand (`deep_explain=true`, slower),
        # otherwise a fast mean-delta approximation for low-latency screening.
        latent_shap = None
        if deep_explain and app_state["shap_explainer"] is not None:
            try:
                latent_shap = app_state["shap_explainer"].explain(q_feat, n_evals=48)[0]
            except Exception:
                latent_shap = None
        if latent_shap is None:
            latent_shap = q_feat[0] - np.mean(q_feat)

        explanation = app_state["explainer_engine"].generate_explanation(
            sample_id=req.sample_id,
            cancer_probability=prob_pos,
            latent_shap=latent_shap,
            patient_metadata={"age": req.patient_age, "smoking": req.smoking_status}
        )

        return {
            "sample_id": req.sample_id,
            "prediction": pred_label,
            "cancer_probability": round(prob_pos, 4),
            "quantum_confidence_pct": round(prob_pos * 100 if prob_pos >= 0.5 else (1 - prob_pos) * 100, 1),
            "top_biomarkers": explanation.top_biomarkers,
            "biochemical_pathways": explanation.pathway_contributions,
            "clinical_summary": explanation.summary_text
        }

    @app.post("/api/v1/train")
    async def train_model_endpoint(req: TrainRequest):
        """Train any quantum or classical model on demand and return metrics & loss curve."""
        try:
            return execute_training_job(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    @app.post("/api/v1/benchmark/run")
    async def run_benchmark_endpoint(req: BenchmarkRequest):
        """Run multi-model comparative leaderboard across Quantum and Classical models."""
        try:
            generator = BiomimeticVOCGenerator(random_state=42)
            cohort = generator.generate_cohort(
                samples_per_class=req.n_samples_per_class,
                cancer_types=["Healthy", req.target_cancer]
            )
            y = cohort.metadata["label_binary"].values
            df_X = cohort.df_features

            idx_train, idx_test = train_test_split(
                np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
            )

            suite = BenchmarkSuite(n_qubits=req.n_qubits, random_state=42)
            df_res = suite.run_benchmark(
                X_train_raw=df_X.iloc[idx_train],
                y_train=y[idx_train],
                X_test_raw=df_X.iloc[idx_test],
                y_test=y[idx_test]
            )
            return {
                "target_cancer": req.target_cancer,
                "n_qubits": req.n_qubits,
                "leaderboard": df_res.to_dict(orient="records")
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}") from exc

    @app.get("/api/v1/dataset/stats")
    async def dataset_stats():
        """Provide dataset stats and biomarker definitions."""
        generator = BiomimeticVOCGenerator(random_state=42)
        return {
            "n_sensors": generator.n_sensors,
            "compounds": generator.compounds,
            "cancer_types": generator.voc_cfg.cancer_types,
            "features_per_sensor": ["max_amplitude", "auc_integral", "adsorption_rise", "desorption_decay"]
        }

    @app.get("/", response_class=HTMLResponse)
    async def index_portal():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>QMLKit - Hybrid Quantum Early Disease Detection Portal</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { background-color: #0b0f19; color: #f3f4f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
                .card { background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; }
                .quantum-badge { background: linear-gradient(135deg, #6366f1, #a855f7); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; }
                .btn-quantum { background: linear-gradient(135deg, #4f46e5, #9333ea); color: white; border: none; font-weight: 600; }
                .btn-quantum:hover { background: linear-gradient(135deg, #4338ca, #7e22ce); color: white; }
                .progress-bar-quantum { background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); }
                .text-quantum { color: #818cf8; }
            </style>
        </head>
        <body class="py-4">
            <div class="container">
                <header class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
                    <div>
                        <h2 class="fw-bold mb-0">QMLKit <span class="quantum-badge">Quantum-Enhanced</span></h2>
                        <p class="text-secondary mb-0">Hybrid Quantum Machine Learning Platform for Early Disease Detection (PS ID: 26139)</p>
                    </div>
                    <span class="badge bg-success py-2 px-3">Simulator Backend Active</span>
                </header>

                <div class="row g-4">
                    <!-- Left: Patient Sample Ingestion -->
                    <div class="col-md-5">
                        <div class="card p-4 h-100">
                            <h4 class="text-quantum mb-3">🐶 Canine Olfactory Ingestion</h4>
                            <div class="mb-3">
                                <label class="form-label text-secondary">Patient Specimen ID</label>
                                <input type="text" id="sampleId" class="form-control bg-dark text-white border-secondary" value="SMPL_ONC_8892">
                            </div>
                            <div class="row g-2 mb-3">
                                <div class="col-6">
                                    <label class="form-label text-secondary">Patient Age</label>
                                    <input type="number" id="patientAge" class="form-control bg-dark text-white border-secondary" value="62">
                                </div>
                                <div class="col-6">
                                    <label class="form-label text-secondary">Smoking Status</label>
                                    <select id="smokingStatus" class="form-select bg-dark text-white border-secondary">
                                        <option value="Former" selected>Former Smoker</option>
                                        <option value="Current">Current Smoker</option>
                                        <option value="Never">Never Smoker</option>
                                    </select>
                                </div>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-secondary">Canine Biomimetic Profile Preset</label>
                                <select id="profilePreset" class="form-select bg-dark text-white border-secondary" onchange="applyPreset()">
                                    <option value="lung_positive">Synthetic Lung Cancer Sample</option>
                                    <option value="healthy_control">Synthetic Healthy Control</option>
                                </select>
                            </div>
                            <button class="btn btn-quantum w-100 py-2 mt-auto" onclick="runDiagnostic()">⚛️ Run Quantum Diagnostic</button>
                        </div>
                    </div>

                    <!-- Right: Diagnostic Results & Explainability -->
                    <div class="col-md-7">
                        <div class="card p-4 h-100">
                            <h4 class="text-quantum mb-3">🔬 Diagnostic Result & Biomarker Attribution</h4>
                            
                            <div id="resultsBlock" style="display: none;">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <span class="fs-5 fw-bold" id="predClass">Malignant / High Risk</span>
                                    <span class="badge fs-6" id="riskBadge">88.4% Malignancy Risk</span>
                                </div>
                                
                                <div class="progress mb-4" style="height: 12px;">
                                    <div id="riskBar" class="progress-bar progress-bar-quantum" style="width: 88%;"></div>
                                </div>

                                <h6 class="text-secondary mb-2">Top Attributed VOC Biomarkers (Reverse Quantum Mapping):</h6>
                                <ul id="biomarkerList" class="list-group list-group-flush mb-3">
                                </ul>

                                <h6 class="text-secondary mb-2">Biochemical Pathway Contributions:</h6>
                                <div class="row g-2 text-center" id="pathwayBoxes">
                                </div>

                                <div class="alert alert-dark mt-3 border-secondary" id="clinicalSummary">
                                </div>
                            </div>

                            <div id="placeholderBlock" class="text-center py-5 text-secondary">
                                <p class="fs-5">Click "Run Quantum Diagnostic" to evaluate sample through BioZZ Quantum Kernel.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                function getSensorValues(preset) {
                    let vals = [];
                    for(let i=0; i<64; i++) {
                        if (preset === 'lung_positive') {
                            vals.push(Math.sin(i * 0.3) * 0.8 + 1.8 + Math.random() * 0.2);
                        } else if (preset === 'healthy_control') {
                            vals.push(Math.sin(i * 0.3) * 0.2 + 0.4 + Math.random() * 0.1);
                        } else {
                            vals.push(Math.cos(i * 0.25) * 0.7 + 1.4 + Math.random() * 0.2);
                        }
                    }
                    return vals;
                }

                async function runDiagnostic() {
                    const preset = document.getElementById('profilePreset').value;
                    const sampleId = document.getElementById('sampleId').value;
                    const age = parseInt(document.getElementById('patientAge').value);
                    const smoking = document.getElementById('smokingStatus').value;
                    const sensorVals = getSensorValues(preset);

                    try {
                        const res = await fetch('/api/v1/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                sample_id: sampleId,
                                patient_age: age,
                                patient_sex: 'M',
                                smoking_status: smoking,
                                sensor_readings: sensorVals
                            })
                        });
                        const data = await res.json();

                        document.getElementById('placeholderBlock').style.display = 'none';
                        document.getElementById('resultsBlock').style.display = 'block';

                        document.getElementById('predClass').innerText = data.prediction;
                        document.getElementById('riskBadge').innerText = (data.cancer_probability * 100).toFixed(1) + '% Malignancy Risk';
                        document.getElementById('riskBadge').className = data.cancer_probability >= 0.5 ? 'badge bg-danger fs-6' : 'badge bg-success fs-6';
                        document.getElementById('riskBar').style.width = (data.cancer_probability * 100) + '%';

                        const bList = document.getElementById('biomarkerList');
                        bList.innerHTML = '';
                        data.top_biomarkers.forEach(b => {
                            const li = document.createElement('li');
                            li.className = 'list-group-item bg-dark text-white border-secondary d-flex justify-content-between align-items-center';
                            li.innerHTML = `<span><strong>${b.compound}</strong> (${b.clinical_impact})</span> <span class="badge bg-primary">${b.importance_score}</span>`;
                            bList.appendChild(li);
                        });

                        const pBoxes = document.getElementById('pathwayBoxes');
                        pBoxes.innerHTML = '';
                        for (const [k, v] of Object.entries(data.biochemical_pathways)) {
                            const col = document.createElement('div');
                            col.className = 'col-6 col-md-3';
                            col.innerHTML = `<div class="p-2 border border-secondary rounded bg-black"><small class="text-secondary">${k.replace(/_/g, ' ')}</small><div class="fs-6 fw-bold text-quantum">${v}%</div></div>`;
                            pBoxes.appendChild(col);
                        }

                        document.getElementById('clinicalSummary').innerText = data.clinical_summary;
                    } catch (err) {
                        alert('Diagnostic failed: ' + err);
                    }
                }
            </script>
        </body>
        </html>
        """

    return app
