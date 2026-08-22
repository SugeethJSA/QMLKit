"""CLI script to explain patient prediction and extract chemical VOC biomarkers."""

import argparse
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.data.feature_selector import QuantumFeatureSelector
from qmlkit.data.preprocessor import BiomedicalDataPipeline
from qmlkit.explainability.biomarker_mapper import BiomarkerAttributionEngine
from qmlkit.quantum.qsvm import QSVMClassifier


def main():
    parser = argparse.ArgumentParser(description="Explain patient prediction with Quantum SHAP and VOC attributions")
    parser.add_argument("--sample-id", type=str, default="SMPL_000105", help="Sample ID to inspect")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    print(f"Generating cohort and training QSVM for explainability...")
    gen = BiomimeticVOCGenerator(random_state=args.seed)
    cohort = gen.generate_cohort(samples_per_class=60, cancer_types=["Healthy", "Lung_Cancer"])

    y = cohort.metadata["label_binary"].values
    splits, pipeline = BiomedicalDataPipeline.create_leak_free_split(
        df_features=cohort.df_features, y=y, test_size=0.2, val_size=0.0, random_state=args.seed
    )

    selector = QuantumFeatureSelector(n_qubits=6, method="pca").fit(splits.X_train)
    X_tr_q = selector.transform(splits.X_train)
    X_te_q = selector.transform(splits.X_test)

    cov = np.corrcoef(X_tr_q.T)
    qsvm = QSVMClassifier(n_qubits=6, feature_map_type="BioZZ", covariance_matrix=cov).fit(X_tr_q, splits.y_train)

    engine = BiomarkerAttributionEngine(feature_selector=selector)

    # Pick first test sample
    test_sample_q = X_te_q[0]
    true_label = "Malignant" if splits.y_test[0] == 1 else "Healthy"
    prob = qsvm.predict_proba(test_sample_q.reshape(1, -1))[0, 1]

    latent_delta = test_sample_q - np.mean(X_tr_q, axis=0)
    explanation = engine.generate_explanation(
        sample_id=args.sample_id,
        cancer_probability=float(prob),
        latent_shap=latent_delta
    )

    print(f"\n================ CLINICAL EXPLANATION REPORT ================")
    print(f"Sample ID:            {explanation.sample_id}")
    print(f"Ground Truth:         {true_label}")
    print(f"Quantum Prediction:   {explanation.predicted_class}")
    print(f"Cancer Probability:   {round(explanation.cancer_probability * 100, 2)}%")
    print(f"\n--- Biochemical Pathway Breakdown ---")
    for pathway, score in explanation.pathway_contributions.items():
        print(f"  * {pathway.replace('_', ' '):<35}: {score:>5.1f}%")

    print(f"\n--- Top Attributed VOC Biomarkers ---")
    for b in explanation.top_biomarkers:
        print(f"  * {b['compound']:<24}: Score {b['importance_score']:<7} ({b['clinical_impact']})")

    print(f"\n--- Oncologist Summary ---")
    print(explanation.summary_text)
    print(f"=============================================================\n")


if __name__ == "__main__":
    main()
