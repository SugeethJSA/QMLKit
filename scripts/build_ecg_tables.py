"""Build curated canine ECG + dog-info tables from the bundled markdown dumps.

Reads docs/dataset_1.md (1,123 recordings with beat timestamps, bradycardia
episodes, quality marks) and docs/DogInfo.md (dog metadata), derives
heart-rate/HRV features per recording, joins them on pet_id <-> DogID and
writes tidy CSVs to data/ecg/.

Constraint: the WAV audio files are NOT distributed with this repo, so all
features derive from annotations only (no waveform processing).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd

from qmlkit.data.dataset_loader import (
    DOG_ECG_MARKDOWN,
    DOG_INFO_MARKDOWN,
    REPO_ROOT,
    extract_ecg_features,
    load_dog_ecg_metadata,
    load_dog_info,
)


def main() -> None:
    out_dir = REPO_ROOT / "data" / "ecg"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing docs/dataset_1.md ...")
    ecg_meta = load_dog_ecg_metadata(DOG_ECG_MARKDOWN)
    print(f"  recordings parsed: {len(ecg_meta)}")
    print(f"  distinct dogs: {ecg_meta['pet_id'].nunique()}")

    print("Deriving HR/HRV features from beat timestamps ...")
    features = extract_ecg_features(ecg_meta)
    features_path = out_dir / "ecg_recordings_features.csv"
    features.to_csv(features_path, index=False)
    print(f"[OK] {features_path} ({len(features)} rows)")

    print("Parsing docs/DogInfo.md ...")
    dogs = load_dog_info(DOG_INFO_MARKDOWN)
    merged = features.merge(
        dogs,
        left_on="pet_id",
        right_on="DogID",
        how="left",
    )
    joined_path = out_dir / "ecg_features_with_dog_info.csv"
    # Drop duplicated breed/weight columns coming from both tables.
    merged = merged.drop(columns=["breeds"], errors="ignore").rename(
        columns={"Weight": "weight_kg", "Age months": "age_months", "Breed": "breed"}
    )
    merged.to_csv(joined_path, index=False)
    matched = int(merged["breed"].notna().sum())
    print(f"[OK] {joined_path} ({len(merged)} rows; dog-info matched on {matched})")

    dogs_path = out_dir / "dog_info.csv"
    dogs.to_csv(dogs_path, index=False)
    print(f"[OK] {dogs_path} ({len(dogs)} rows)")

    hr = features["hr_mean_bpm"].dropna()
    summary = pd.DataFrame(
        {
            "metric": [
                "recordings",
                "dogs",
                "hr_mean_bpm_median",
                "hr_min_bpm_min",
                "bradycardia_recordings",
                "bad_ecg_fraction_gt_50pct",
            ],
            "value": [
                len(features),
                int(ecg_meta["pet_id"].nunique()),
                round(float(hr.median()), 2) if len(hr) else None,
                round(float(features["hr_min_bpm"].min()), 2) if hr.any() else None,
                int(features["has_bradycardia"].sum()),
                int((features["bad_ecg_fraction"] > 0.5).sum()),
            ],
        }
    )
    summary_path = out_dir / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] {summary_path}")
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
