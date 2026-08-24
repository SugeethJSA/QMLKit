"""Biomimetic Canine Olfactory VOC & Sensor Array Generator.

Simulates biological volatilome emissions (lipid peroxidation aldehydes, ketones,
aromatics, alkanes) across cancer types and maps them through a 16-channel cross-reactive
sensor array simulating dynamic Langmuir adsorption kinetics with noise and drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from qmlkit.config import SensorArrayConfig, VOCBiomarkerConfig


@dataclass
class SyntheticCohort:
    """Container for simulated canine-biomimetic dataset."""
    df_features: pd.DataFrame
    raw_time_series: np.ndarray  # Shape: (N_samples, N_sensors, N_timesteps)
    voc_ground_truth: pd.DataFrame
    metadata: pd.DataFrame


class BiomimeticVOCGenerator:
    """Generates synthetic biomimetic canine olfactory cancer screening data."""

    def __init__(
        self,
        sensor_cfg: Optional[SensorArrayConfig] = None,
        voc_cfg: Optional[VOCBiomarkerConfig] = None,
        random_state: int = 42
    ):
        self.sensor_cfg = sensor_cfg or SensorArrayConfig()
        self.voc_cfg = voc_cfg or VOCBiomarkerConfig()
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

        # Precompute sensor affinity matrix A (n_sensors x n_compounds)
        self.n_sensors = self.sensor_cfg.n_sensors
        self.compounds = self.voc_cfg.compounds
        self.n_compounds = len(self.compounds)
        self.affinity_matrix = self._build_sensor_affinity_matrix()

    def _build_sensor_affinity_matrix(self) -> np.ndarray:
        """Construct biologically authentic cross-reactivity affinity matrix A."""
        # Fix seed for reproducible sensor characteristics
        gen = np.random.default_rng(self.random_state + 100)

        # Base affinity with chemical class preferences - tuned for realistic overlap (was too separable: preferred 0.6-1.0 vs cross 0.05-0.35 gave 100% classical)
        A = np.zeros((self.n_sensors, self.n_compounds))
        for s in range(self.n_sensors):
            preferred_class = s % 4  # 0: Aldehydes, 1: Ketones, 2: Aromatics, 3: Alkanes/Sulfur
            for c_idx in range(self.n_compounds):
                comp_class = c_idx // 6  # 6 compounds per class
                if comp_class == preferred_class:
                    base_aff = gen.uniform(0.45, 0.85)
                else:
                    # Higher cross-reactivity background -> more sensor confusion, reduces perfect separation
                    base_aff = gen.uniform(0.15, 0.50)
                A[s, c_idx] = base_aff
        return A

    def _generate_voc_profiles(
        self,
        n_samples: int,
        cancer_type: str
    ) -> np.ndarray:
        """Generate log-normal VOC concentrations (in ppb) for a specific clinical cohort."""
        # Baseline healthy log-mean and variance
        mean_log = np.array([
            # Aldehydes (Hexanal, Heptanal, Octanal, Nonanal, Benzaldehyde, Decanal)
            0.5, 0.4, 0.3, 0.3, 0.6, 0.2,
            # Ketones (Acetone, 2-Butanone, 2-Pentanone, 3-Octanone, Acetophenone, Cyclohexanone)
            2.5, 1.0, 0.5, 0.3, 0.4, 0.3,
            # Aromatics (Ethylbenzene, Styrene, Toluene, o-Xylene, 1,2,4-Trimethylbenzene, Benzene)
            0.8, 0.4, 1.2, 0.5, 0.3, 0.3,
            # Alkanes / Terpenes / Sulfur (Isoprene, Octane, Decane, D-Limonene, DMDS, DMS)
            2.2, 0.6, 0.4, 0.5, 0.3, 0.3
        ])
        # Increased variance for realistic overlap (was 0.25 -> perfect separation for XGB); multipliers reduced from 3-4x to ~2x to prevent 100% classical
        std_log = np.full(self.n_compounds, 0.42)

        # Alterations per cancer indication - reduced effect sizes for calibrated difficulty
        multipliers = np.ones(self.n_compounds)
        if cancer_type == "Lung_Cancer":
            # Aldehydes elevated (Hexanal, Heptanal, Benzaldehyde), Aromatics (Ethylbenzene), Isoprene depleted
            multipliers[[0, 1, 4]] *= 2.2
            multipliers[12] *= 1.8
            multipliers[18] *= 0.65
        elif cancer_type == "Breast_Cancer":
            # Heptanal, Octanal, 2-Pentanone, 3-Octanone, Trimethylbenzene elevated
            multipliers[[1, 2, 8, 9, 16]] *= 2.0
        elif cancer_type == "Colorectal_Cancer":
            # DMDS, DMS, Benzaldehyde, Cyclohexanone elevated
            multipliers[[22, 23, 4, 11]] *= 2.3
        elif cancer_type == "Prostate_Cancer":
            # 2-Butanone, 2-Pentanone, Toluene, DMS elevated
            multipliers[[7, 8, 14, 23]] *= 2.1
        elif cancer_type == "Ovarian_Cancer":
            # Nonanal, Decanal, Acetophenone, Cyclohexanone elevated
            multipliers[[3, 5, 10, 11]] *= 2.2

        # Generate correlated multivariate samples
        log_conc = self.rng.normal(loc=mean_log, scale=std_log, size=(n_samples, self.n_compounds))
        concentrations = np.exp(log_conc) * multipliers
        return np.maximum(concentrations, 0.01)

    def _simulate_transient_signals(
        self,
        voc_matrix: np.ndarray
    ) -> np.ndarray:
        """Simulate dynamic Langmuir adsorption/desorption differential response curves."""
        n_samples = voc_matrix.shape[0]
        timesteps = self.sensor_cfg.transient_steps
        t_on = int(timesteps * 0.2)
        t_off = int(timesteps * 0.7)

        signals = np.zeros((n_samples, self.n_sensors, timesteps))
        k_ads = self.sensor_cfg.adsorption_rate
        k_des = self.sensor_cfg.desorption_rate
        noise_std = self.sensor_cfg.baseline_noise_sigma
        drift_amp = self.sensor_cfg.drift_amplitude

        for i in range(n_samples):
            # Effective sensor stimulation: S_s = sum_c (A_{s,c} * C_c)
            stim = np.dot(self.affinity_matrix, voc_matrix[i])  # shape (n_sensors,)
            g_max = np.log1p(stim) * 1.5

            for s in range(self.n_sensors):
                g = 0.0
                # Baseline drift component
                drift = drift_amp * np.linspace(-1, 1, timesteps) + self.rng.normal(0, 0.005)

                for t in range(timesteps):
                    if t < t_on:
                        g = 0.0
                    elif t_on <= t < t_off:
                        # Adsorption phase: dg/dt = k_ads * (g_max - g)
                        g += k_ads * (g_max[s] - g)
                    else:
                        # Desorption phase: dg/dt = -k_des * g
                        g += -k_des * g

                    # Add physical sensor noise + drift
                    noise = self.rng.normal(0, noise_std)
                    signals[i, s, t] = max(0.0, g + drift[t] + noise)

        return signals

    def _extract_kinetic_features(
        self,
        raw_signals: np.ndarray
    ) -> pd.DataFrame:
        """Extract 4 key kinetic features per sensor: Max Amplitude, AUC, Rise Rate, Decay Rate."""
        n_samples, n_sensors, timesteps = raw_signals.shape
        t_on = int(timesteps * 0.2)
        t_off = int(timesteps * 0.7)

        features_dict: Dict[str, List[float]] = {}
        for s in range(n_sensors):
            features_dict[f"sensor_{s+1:02d}_max"] = []
            features_dict[f"sensor_{s+1:02d}_auc"] = []
            features_dict[f"sensor_{s+1:02d}_rise"] = []
            features_dict[f"sensor_{s+1:02d}_decay"] = []

        for i in range(n_samples):
            for s in range(n_sensors):
                sig = raw_signals[i, s]
                # Filter smoothing
                smooth_sig = savgol_filter(sig, window_length=min(11, timesteps), polyorder=2)

                s_max = float(np.max(smooth_sig))
                s_auc = float(np.sum(smooth_sig[t_on:t_off]))
                # Rise rate (slope between t_on and peak)
                peak_idx = int(np.argmax(smooth_sig))
                if peak_idx > t_on:
                    s_rise = float((smooth_sig[peak_idx] - smooth_sig[t_on]) / (peak_idx - t_on))
                else:
                    s_rise = 0.01

                # Decay rate (decay after t_off)
                if timesteps > t_off + 5:
                    s_decay = float((smooth_sig[t_off] - smooth_sig[-1]) / (timesteps - t_off))
                else:
                    s_decay = 0.01

                features_dict[f"sensor_{s+1:02d}_max"].append(s_max)
                features_dict[f"sensor_{s+1:02d}_auc"].append(s_auc)
                features_dict[f"sensor_{s+1:02d}_rise"].append(max(0.0, s_rise))
                features_dict[f"sensor_{s+1:02d}_decay"].append(max(0.0, s_decay))

        return pd.DataFrame(features_dict)

    def generate_cohort(
        self,
        samples_per_class: int = 150,
        cancer_types: Optional[List[str]] = None
    ) -> SyntheticCohort:
        """Generate a complete balanced or specified multi-cancer cohort."""
        c_types = cancer_types or self.voc_cfg.cancer_types
        all_voc_list: List[np.ndarray] = []
        metadata_rows: List[Dict] = []

        sample_counter = 1
        for c_type in c_types:
            voc_block = self._generate_voc_profiles(samples_per_class, c_type)
            all_voc_list.append(voc_block)

            for _ in range(samples_per_class):
                age = int(self.rng.normal(60, 12)) if c_type != "Healthy" else int(self.rng.normal(52, 14))
                age = max(21, min(88, age))
                sex = self.rng.choice(["M", "F"])
                smoking = self.rng.choice(["Never", "Former", "Current"], p=[0.5, 0.3, 0.2])
                stage = "Control" if c_type == "Healthy" else self.rng.choice(["Stage_I", "Stage_II", "Stage_III", "Stage_IV"], p=[0.35, 0.35, 0.2, 0.1])

                metadata_rows.append({
                    "sample_id": f"SMPL_{sample_counter:06d}",
                    "patient_age": age,
                    "patient_sex": sex,
                    "smoking_status": smoking,
                    "sample_matrix": "Exhaled_Breath",
                    "label_cancer_type": c_type,
                    "label_binary": 0 if c_type == "Healthy" else 1,
                    "label_stage": stage
                })
                sample_counter += 1

        all_voc = np.vstack(all_voc_list)
        df_metadata = pd.DataFrame(metadata_rows)
        df_voc = pd.DataFrame(all_voc, columns=[f"voc_{c.lower()}_ppb" for c in self.compounds])

        # Simulate physics sensors
        raw_signals = self._simulate_transient_signals(all_voc)
        df_features = self._extract_kinetic_features(raw_signals)

        return SyntheticCohort(
            df_features=df_features,
            raw_time_series=raw_signals,
            voc_ground_truth=df_voc,
            metadata=df_metadata
        )
