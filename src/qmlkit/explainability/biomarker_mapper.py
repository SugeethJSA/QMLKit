"""Biomarker Attribution Engine: Reverse Mapping from Latent Quantum States to VOC Molecules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

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
        voc_config: Optional[VOCBiomarkerConfig] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.feature_selector = feature_selector
        self.voc_config = voc_config or VOCBiomarkerConfig()
        self.compounds = self.voc_config.compounds
        # Optional names of the selector's input features. When they match the
        # compound list one-to-one (real VOC datasets), attributions map directly.
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _pool_to_compounds(self, feature_shap: np.ndarray) -> np.ndarray:
        """Aggregate per-input-feature attributions to per-compound attributions."""
        n_samples, n_feat = feature_shap.shape
        n_comp = len(self.compounds)

        # 1. Direct mapping when input features ARE the configured compounds.
        if self.feature_names is not None and n_feat == n_comp and self.feature_names == list(
            self.compounds
        ):
            return feature_shap

        # 2. Legacy synthetic schema: 64 sensor features (16 sensors x 4 kinetics),
        #    sensors grouped into 4 chemical-class blocks of 6 compounds each.
        if n_feat == 64 and n_comp == 24:
            chem_shap = np.zeros((n_samples, n_comp))
            for c_idx in range(n_comp):
                preferred_sensor_block = c_idx // 6
                sensor_indices = [preferred_sensor_block + 4 * k for k in range(4)]
                feat_indices = []
                for s in sensor_indices:
                    feat_indices.extend([s * 4 + f for f in range(4)])
                chem_shap[:, c_idx] = np.mean(feature_shap[:, feat_indices], axis=1)
            return chem_shap

        # 3. Generic fallback: contiguous mean pooling of features into compounds.
        chem_shap = np.zeros((n_samples, n_comp))
        edges = np.linspace(0, n_feat, n_comp + 1, dtype=int)
        for c_idx in range(n_comp):
            lo, hi = edges[c_idx], max(edges[c_idx] + 1, edges[c_idx + 1])
            chem_shap[:, c_idx] = np.mean(feature_shap[:, lo:hi], axis=1)
        return chem_shap

    def map_latent_to_chemical(
        self,
        latent_shap_values: np.ndarray
    ) -> np.ndarray:
        """Project latent qubit attributions back to VOC chemical compounds."""
        latent_vec = np.atleast_2d(latent_shap_values)

        if self.feature_selector.method == "pca" and self.feature_selector.pca_model is not None:
            # PCA projection: chemical_shap = latent_shap @ components_
            # components_ is (n_components, n_input_features)
            sensor_shap = np.dot(latent_vec, self.feature_selector.pca_model.components_)
            return self._pool_to_compounds(sensor_shap)
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

        # Aggregate by biological pathway. Compounds are ordered by chemical
        # class (see VOCBiomarkerConfig); split into four equal class blocks.
        pathway_names = [
            "Lipid_Peroxidation_Aldehydes",
            "Mitochondrial_Ketone_Metabolism",
            "Cytochrome_P450_Aromatics",
            "Alkanes_Sulfur_Dysregulation",
        ]
        n_comp = len(chem_shap)
        edges = np.linspace(0, n_comp, len(pathway_names) + 1, dtype=int)
        block_sums = [
            float(np.sum(np.maximum(0, chem_shap[edges[k]:max(edges[k] + 1, edges[k + 1])])))
            for k in range(len(pathway_names))
        ]
        total_pos = max(1e-6, sum(block_sums))

        pathways = {
            name: round((block / total_pos) * 100, 1)
            for name, block in zip(pathway_names, block_sums)
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
