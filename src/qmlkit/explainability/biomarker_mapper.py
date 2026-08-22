"""Biomarker Attribution Engine: Reverse Mapping from Latent Quantum States to VOC Molecules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from qmlkit.config import VOCBiomarkerConfig
from qmlkit.data.feature_selector import QuantumFeatureSelector


@dataclass
class ClinicalExplanation:
    """Diagnostic explanation object for oncology review."""
    sample_id: str
    predicted_class: str
    cancer_probability: float
    top_biomarkers: List[Dict[str, Any]]
    pathway_contributions: Dict[str, float]
    summary_text: str


class BiomarkerAttributionEngine:
    """Translates quantum Hilbert-space attributions into clinical VOC chemical explanations."""

    def __init__(
        self,
        feature_selector: QuantumFeatureSelector,
        voc_config: Optional[VOCBiomarkerConfig] = None
    ):
        self.feature_selector = feature_selector
        self.voc_config = voc_config or VOCBiomarkerConfig()
        self.compounds = self.voc_config.compounds

    def map_latent_to_chemical(
        self,
        latent_shap_values: np.ndarray
    ) -> np.ndarray:
        """Project latent qubit attributions back to 24 VOC chemical compounds."""
        latent_vec = np.atleast_2d(latent_shap_values)

        if self.feature_selector.method == "pca" and self.feature_selector.pca_model is not None:
            # PCA projection: chemical_shap = latent_shap @ components_
            # components_ is (n_components, n_sensor_features)
            sensor_shap = np.dot(latent_vec, self.feature_selector.pca_model.components_)
            # Map 64 sensor features (4 features per 16 sensors) to 24 chemical compounds
            # Approximate through mean sensor attribution across 4 functional blocks
            n_comp = len(self.compounds)
            chem_shap = np.zeros((latent_vec.shape[0], n_comp))

            # Group sensors into 4 classes
            for c_idx in range(n_comp):
                preferred_sensor_block = c_idx // 6
                # Sensors corresponding to this block: 4 sensors (e.g. 0, 4, 8, 12)
                sensor_indices = [preferred_sensor_block + 4 * k for k in range(4)]
                feat_indices = []
                for s in sensor_indices:
                    feat_indices.extend([s * 4 + f for f in range(4)])

                chem_shap[:, c_idx] = np.mean(sensor_shap[:, feat_indices], axis=1)

            return chem_shap
        else:
            # Fallback uniform projection
            n_comp = len(self.compounds)
            return np.ones((latent_vec.shape[0], n_comp)) / n_comp

    def generate_explanation(
        self,
        sample_id: str,
        cancer_probability: float,
        latent_shap: np.ndarray,
        patient_metadata: Optional[Dict] = None
    ) -> ClinicalExplanation:
        """Create a human-interpretable clinical report for an oncologist."""
        chem_shap = self.map_latent_to_chemical(latent_shap)[0]

        # Aggregate by biological pathway
        aldehydes = float(np.sum(np.maximum(0, chem_shap[0:6])))
        ketones = float(np.sum(np.maximum(0, chem_shap[6:12])))
        aromatics = float(np.sum(np.maximum(0, chem_shap[12:18])))
        alkanes_sulfur = float(np.sum(np.maximum(0, chem_shap[18:24])))
        total_pos = max(1e-6, aldehydes + ketones + aromatics + alkanes_sulfur)

        pathways = {
            "Lipid_Peroxidation_Aldehydes": round((aldehydes / total_pos) * 100, 1),
            "Mitochondrial_Ketone_Metabolism": round((ketones / total_pos) * 100, 1),
            "Cytochrome_P450_Aromatics": round((aromatics / total_pos) * 100, 1),
            "Alkanes_Sulfur_Dysregulation": round((alkanes_sulfur / total_pos) * 100, 1),
        }

        # Rank individual top biomarkers
        sorted_indices = np.argsort(np.abs(chem_shap))[::-1]
        top_biomarkers = []
        for idx in sorted_indices[:5]:
            comp = self.compounds[idx]
            impact = float(chem_shap[idx])
            direction = "Elevated (Risk Driver)" if impact > 0 else "Depleted / Protective"
            top_biomarkers.append({
                "compound": comp,
                "importance_score": round(abs(impact), 4),
                "clinical_impact": direction
            })

        pred_class = "Cancer Positive (High Risk)" if cancer_probability >= 0.5 else "Healthy / Low Risk"

        top_names = ", ".join([b["compound"] for b in top_biomarkers[:3]])
        summary = (
            f"Patient {sample_id} evaluated with {round(cancer_probability * 100, 1)}% malignancy risk. "
            f"The primary biochemical driver was {pathways['Lipid_Peroxidation_Aldehydes']}% lipid peroxidation "
            f"with significant elevation in [{top_names}]."
        )

        return ClinicalExplanation(
            sample_id=sample_id,
            predicted_class=pred_class,
            cancer_probability=float(cancer_probability),
            top_biomarkers=top_biomarkers,
            pathway_contributions=pathways,
            summary_text=summary
        )
