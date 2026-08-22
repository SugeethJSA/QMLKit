"""CLI script to run full leak-free benchmark comparing Quantum vs Classical models."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from qmlkit.config import set_seed
from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.data.preprocessor import BiomedicalDataPipeline
from qmlkit.evaluation.benchmark_suite import BenchmarkSuite
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Run Quantum vs Classical benchmark on Canine VOC Cancer Screening")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Samples per class")
    parser.add_argument("--n-qubits", type=int, default=6, help="Quantum register size")
    parser.add_argument("--target-cancer", type=str, default="Lung_Cancer", help="Target cancer type")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== QMLKit Benchmarking Suite (PS ID: 26139) ===")
    print(f"Target: {args.target_cancer} vs Healthy | Qubits: {args.n_qubits} | Samples/class: {args.samples_per_class}")

    # 1. Generate cohort
    generator = BiomimeticVOCGenerator(random_state=args.seed)
    cohort = generator.generate_cohort(
        samples_per_class=args.samples_per_class,
        cancer_types=["Healthy", args.target_cancer]
    )

    y = cohort.metadata["label_binary"].values
    df_X = cohort.df_features

    # 2. Strict leak-free train/test partition indices
    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.20,
        stratify=y,
        random_state=args.seed
    )

    X_train_raw = df_X.iloc[idx_train]
    y_train = y[idx_train]
    X_test_raw = df_X.iloc[idx_test]
    y_test = y[idx_test]

    print(f"Data Split: Train={len(y_train)} samples, Held-Out Test={len(y_test)} samples")
    print("Running quantum and classical model evaluations...\n")

    suite = BenchmarkSuite(n_qubits=args.n_qubits, random_state=args.seed)
    df_results = suite.run_benchmark(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_test_raw=X_test_raw,
        y_test=y_test
    )

    # Output table
    print("=== FINAL BENCHMARK RESULTS (Ranked by ROC-AUC) ===")
    print(df_results.to_markdown(index=False))

    csv_path = os.path.join(args.output_dir, "benchmark_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[OK] Metrics saved to {csv_path}")

    # Generate comparative visualization plot
    plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    bar = sns.barplot(
        data=df_results,
        x="model_name",
        y="roc_auc",
        hue="paradigm",
        palette={"Quantum": "#7c3aed", "Classical": "#2563eb"}
    )
    plt.title(f"ROC-AUC Performance: Quantum vs Classical Baselines ({args.target_cancer})", fontsize=12, fontweight="bold")
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("ROC-AUC Score", fontsize=11)
    plt.ylim(0.5, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "benchmark_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
