"""Tests for external dataset ingestion (markdown tables, CSV, dog ECG)."""

import textwrap

import numpy as np
import pandas as pd
import pytest

from qmlkit.data.dataset_loader import (
    balanced_subsample,
    extract_ecg_features,
    load_csv_dataset,
    load_dog_ecg_metadata,
    load_lung_voc_dataset,
    load_markdown_table,
)

MINI_VOC_TABLE = textwrap.dedent(
    """
    # Mini Lung VOC

    ## Sheet1

    | PatientID | CH2O | C3H6O | C7H6O | Class   |
    |----------:|-----:|------:|------:|:--------|
    | 1         | 8.29 | 4.38  | 1.27  | Control |
    | 2         | 0.81 | 4.77  | 0.72  | Control |
    | 3         | 2.18 | 10.55 | 0.19  | Cancer  |
    | 4         | bad  | 3.00  | 0.44  | Cancer  |
    """
)


class TestMarkdownTableParser:
    def test_parses_and_skips_separator_row(self, tmp_path):
        path = tmp_path / "mini.md"
        path.write_text(MINI_VOC_TABLE, encoding="utf-8")
        df = load_markdown_table(path)
        assert list(df.columns) == ["PatientID", "CH2O", "C3H6O", "C7H6O", "Class"]
        assert len(df) == 4

    def test_no_table_raises(self, tmp_path):
        path = tmp_path / "empty.md"
        path.write_text("just some prose\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_markdown_table(path)


class TestLungVocLoader:
    def test_label_mapping_and_numeric_coercion(self, tmp_path):
        path = tmp_path / "voc.md"
        path.write_text(MINI_VOC_TABLE, encoding="utf-8")
        loaded = load_lung_voc_dataset(path)
        assert loaded.df_features.shape == (3, 3)  # 'bad' row dropped
        assert loaded.y.tolist() == [0, 0, 1]
        assert loaded.label_names[1] == "Cancer"
        assert loaded.ids.tolist() == ["1", "2", "3"]

    def test_unknown_label_raises(self, tmp_path):
        table = MINI_VOC_TABLE.replace("| Cancer  |", "| Malignant |")
        path = tmp_path / "voc.md"
        path.write_text(table, encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown class label"):
            load_lung_voc_dataset(path, drop_unmapped=False)

    def test_unmapped_labels_dropped(self, tmp_path):
        table = MINI_VOC_TABLE.replace("Cancer", "Benign")
        path = tmp_path / "voc.md"
        path.write_text(table, encoding="utf-8")
        loaded = load_lung_voc_dataset(path)  # cancer_vs_control drops Benign
        assert loaded.y.tolist() == [0, 0]

    def test_real_bundled_dataset_loads(self):
        loaded = load_lung_voc_dataset()
        counts = dict(zip(*np.unique(loaded.y, return_counts=True)))
        assert loaded.df_features.shape[1] == 27
        assert set(counts) == {0, 1}
        assert sum(counts.values()) == 350  # 193 Control + 157 Cancer (Benign dropped)

    def test_disease_vs_control_pools_benign(self):
        loaded = load_lung_voc_dataset(task="disease_vs_control")
        counts = dict(zip(*np.unique(loaded.y, return_counts=True)))
        assert len(loaded.y) == 427
        assert counts[0] == 193 and counts[1] == 234


class TestBalancedSubsample:
    def test_caps_per_class_and_keeps_both(self):
        X = pd.DataFrame({"a": range(100)})
        y = np.array([0] * 50 + [1] * 50)
        Xs, ys = balanced_subsample(X, y, max_samples=20)
        assert len(Xs) <= 20
        assert set(np.unique(ys)) == {0, 1}
        assert len(set(ys)) == 2

    def test_noop_when_under_limit(self):
        y = np.array([0, 1, 1])
        _, ys = balanced_subsample(pd.DataFrame({"a": [1, 2, 3]}), y, max_samples=10)
        assert ys.tolist() == y.tolist()


class TestCsvLoader:
    def test_auto_binary_encoding(self, tmp_path):
        path = tmp_path / "tab.csv"
        pd.DataFrame(
            {"f1": [1.0, 2.0, 3.0, 4.0], "label": ["no", "no", "yes", "yes"]}
        ).to_csv(path, index=False)
        loaded = load_csv_dataset(path, label_column="label")
        assert loaded.y.tolist() == [0, 0, 1, 1]

    def test_explicit_mapping_and_unknown(self, tmp_path):
        path = tmp_path / "tab.csv"
        pd.DataFrame({"f1": [1.0], "dx": ["weird"]}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_csv_dataset(path, label_column="dx", label_mapping={"cancer": 1, "control": 0})


class TestDogEcg:
    def test_extract_features_from_synthetic_rows(self):
        ecg_df = pd.DataFrame(
            [
                {
                    "_id": "a",
                    "pet_id": 14,
                    "breeds": "boxer",
                    "duration": 300.0,
                    "ecg_pulses": [0.0, 0.8, 1.6, 2.4, 3.2],
                    "segments_br": [{"deb": 200.0, "fin": 240.0, "value": 13.24}],
                    "bad_ecg": None,
                }
            ]
        )
        feats = extract_ecg_features(ecg_df)
        row = feats.iloc[0]
        assert row["n_beats"] == 5
        assert row["hr_mean_bpm"] == pytest.approx(75.0, abs=0.1)  # RR = 800 ms
        assert row["br_episode_count"] == 1
        assert row["br_total_s"] == pytest.approx(40.0)
        assert row["has_bradycardia"] == 1

    def test_literal_list_parsing(self, tmp_path):
        md = textwrap.dedent(
            """
            | _id | pet_id | segments_br |
            |-----|--------|-------------|
            | x1  | 7      | [{'deb': 10.0, 'fin': 20.0, 'value': 9.5}] |
            """
        )
        f = tmp_path / "ecg.md"
        f.write_text(md, encoding="utf-8")
        df = load_dog_ecg_metadata(f)
        segs = df.loc[0, "segments_br"]
        assert isinstance(segs, list) and segs[0]["value"] == 9.5
