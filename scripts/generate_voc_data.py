"""CLI script to generate and save synthetic canine biomimetic VOC dataset."""

import argparse
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic canine biomimetic cancer screening dataset")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Number of samples per clinical cohort")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory to save CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating synthetic cohort with {args.samples_per_class} samples per class (Seed: {args.seed})...")

    generator = BiomimeticVOCGenerator(random_state=args.seed)
    cohort = generator.generate_cohort(samples_per_class=args.samples_per_class)

    features_path = os.path.join(args.output_dir, "sensor_features.csv")
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    voc_path = os.path.join(args.output_dir, "ground_truth_voc.csv")

    cohort.df_features.to_csv(features_path, index=False)
    cohort.metadata.to_csv(metadata_path, index=False)
    cohort.voc_ground_truth.to_csv(voc_path, index=False)

    print(f"[OK] Successfully generated {len(cohort.metadata)} total samples across 6 cohorts.")
    print(f"     Sensor Features: {features_path} ({cohort.df_features.shape})")
    print(f"     Metadata:        {metadata_path}")
    print(f"     VOC Ground Truth:{voc_path}")


if __name__ == "__main__":
    main()
