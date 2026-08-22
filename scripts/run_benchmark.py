"""CLI script to run full leak-free benchmark comparing Quantum vs Classical models.

Data sources:
  synthetic  - biomimetic cohort generator (default, backward compatible)
  markdown   - pipe-delimited markdown table (bundled Lung Cancer VOC dataset)
  csv        - generic CSV with a label column
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from sklearn.model_selection import train_test_split

from qmlkit.config import set_seed
from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.data.dataset_loader import LUNG_VOC_MARKDOWN, load_csv_dataset, load_lung_voc_dataset
from qmlkit.evaluation.benchmark_suite import BenchmarkSuite


def _load_dataset(args):
    """Return (df_features, y, label_names) for the requested source."""
    if args.source == "synthetic":
        generator = BiomimeticVOCGenerator(random_state=args.seed)
        cohort = generator.generate_cohort(
            samples_per_class=args.samples_per_class,
            cancer_types=["Healthy", args.target_cancer]
        )
        return (
            cohort.df_features,
            cohort.metadata["label_binary"].values,
            {0: "Healthy", 1: args.target_cancer},
        )

    if args.source == "markdown":
        loaded = load_lung_voc_dataset(
            path=args.data_path or LUNG_VOC_MARKDOWN, task=args.voc_task
        )
        return loaded.df_features, loaded.y, loaded.label_names

    if args.source == "csv":
        if not args.data_path:
            raise SystemExit("--data-path is required for --source csv")
        loaded = load_csv_dataset(args.data_path, label_column=args.label_column)
        return loaded.df_features, loaded.y, loaded.label_names

    raise SystemExit(f"Unknown source: {args.source}")


def main():
    parser = argparse.ArgumentParser(description="Run Quantum vs Classical benchmark on Canine VOC Cancer Screening")
    parser.add_argument("--source", choices=["synthetic", "markdown", "csv"], default="synthetic",
                        help="Data source: bundled real VOC markdown, generic CSV, or synthetic generator")
    parser.add_argument("--data-path", type=str, default=None, help="Path for markdown/csv sources")
    parser.add_argument("--voc-task", type=str, default="cancer_vs_control",
                        choices=["cancer_vs_control", "disease_vs_control"],
                        help="Binarisation of the 3-class Lung VOC dataset")
    parser.add_argument("--label-column", type=str, default="label", help="Label column for csv source")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Class-balanced subsample cap (0 = all). Bounds O(N^2) kernel cost")
    parser.add_argument("--samples-per-class", type=int, default=50, help="Samples per class (synthetic only)")
    parser.add_argument("--n-qubits", type=int, default=6, help="Quantum register size")
    parser.add_argument("--vqc-epochs", type=int, default=20,
                        help="VQC training epochs (lower to bound runtime on real data)")
    parser.add_argument("--target-cancer", type=str, default="Lung_Cancer", help="Target cancer type (synthetic)")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df_X, y, label_names = _load_dataset(args)
    if args.max_samples and args.max_samples < len(y):
        from qmlkit.data.dataset_loader import balanced_subsample
        df_X, y = balanced_subsample(df_X, y, max_samples=args.max_samples, seed=args.seed)

    print("=== QMLKit Benchmarking Suite (PS ID: 26139) ===")
    print(
        f"Source: {args.source} | Classes: {label_names} | N={len(y)} | "
        f"Qubits: {args.n_qubits}"
        + (f" | Samples/class: {args.samples_per_class}" if args.source == "synthetic" else "")
    )

    # Strict leak-free stratified train/test partition
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

    suite = BenchmarkSuite(n_qubits=args.n_qubits, random_state=args.seed, vqc_epochs=args.vqc_epochs)
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

    meta = {
        "source": args.source,
        "voc_task": args.voc_task if args.source == "markdown" else "",
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=True)},
        "n_qubits": args.n_qubits,
        "seed": args.seed,
    }
    meta_path = Path(args.output_dir) / "run_metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as fh:
        for key, value in meta.items():
            fh.write(f"{key}: {value}\n")

    # Generate comparative visualization plot
    plt.figure(figsize=(10, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=df_results,
        x="model_name",
        y="roc_auc",
        hue="paradigm",
        palette={"Quantum": "#7c3aed", "Classical": "#2563eb"}
    )
    plt.title("ROC-AUC Performance: Quantum vs Classical Baselines", fontsize=12, fontweight="bold")
    plt.xlabel("Model", fontsize=11)
    plt.ylabel("ROC-AUC Score", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "benchmark_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[OK] Comparison plot saved to {plot_path}")
    print(f"[OK] Run metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
