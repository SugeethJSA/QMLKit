"""Two-class synthetic kennel trial generator (cancer-associated vs control).

Simulates the manuscript's three-phase screening trials (baseline / exposure /
post-exposure, §V-B) with class-dependent canine-response signatures so the
hybrid lab, modality ablations and robustness harnesses can be exercised
end-to-end before real Data-Lab sessions exist. Signatures are intentionally
moderate (no trivially separable classes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from qmlkit.hardware.kennel_features import (
    TRIAL_FEATURE_NAMES,
    extract_trial_features,
)
from qmlkit.hardware.protocol import KennelFrame


@dataclass
class KennelTrialDataset:
    X: pd.DataFrame
    y: np.ndarray
    dog_ids: np.ndarray
    feature_groups: Dict[str, List[str]]
    feature_names: List[str]


def _make_frame(
    t_ms: int,
    seq: int,
    state: str,
    rng: np.random.Generator,
    *,
    sniff: bool,
    total_load: float,
    load_share: np.ndarray,
    tremor_amp: float,
    hr_bpm: float,
    spo2_pct: float,
) -> KennelFrame:
    sniff_wave = (
        np.sin(2 * np.pi * 3.0 * t_ms / 1000.0) * 0.35 + rng.normal(0, 0.03)
        if sniff else 0.0
    )
    tremor = np.sin(2 * np.pi * 6.0 * t_ms / 1000.0 + 0.7) * tremor_amp
    acc = [
        float(rng.normal(0, 0.05) + tremor + sniff_wave),
        float(9.78 + rng.normal(0, 0.05) + 0.5 * sniff_wave),
        float(rng.normal(0, 0.05) + (0.15 * sniff_wave if sniff else 0.0)),
    ]
    gyr = [float(rng.normal(0, 0.02) + (0.06 * sniff_wave if sniff else 0.0)) for _ in range(3)]
    fsr = [float(total_load * s) for s in load_share]
    return KennelFrame(
        ts_ms=t_ms,
        seq=seq,
        state="SNIFF" if sniff else "IDLE",
        fsr=fsr,
        ir=[int(v) for v in (rng.random(4) < (0.7 if sniff else 0.15))]
        + [int(v) for v in (rng.random(2) < (0.8 if sniff else 0.05))],
        us_bottom=float(np.clip(38.0 - (12.0 if sniff else 0.0) + rng.normal(0, 3), 5, 200)),
        us_top=float(np.clip(95.0 - (45.0 if sniff else 0.0) + rng.normal(0, 5), 5, 200)),
        acc=acc,
        gyr=gyr,
        imu_temp_c=float(37.2 + rng.normal(0, 0.2)),
        hr_bpm=float(hr_bpm),
        spo2_pct=float(spo2_pct),
    )


def generate_synthetic_trials(
    trials_per_class: int = 40,
    baseline_s: float = 4.0,
    exposure_s: float = 8.0,
    post_s: float = 4.0,
    rate_hz: int = 100,
    seed: int = 42,
) -> KennelTrialDataset:
    """Build a trial-level tabular dataset from simulated three-phase sessions."""
    rng = np.random.default_rng(seed)
    window = int(rate_hz)

    rows: List[np.ndarray] = []
    labels: List[int] = []
    dogs: List[str] = []
    n_dogs_per_class = max(4, trials_per_class // 10)

    for label in (0, 1):
        for trial in range(trials_per_class):
            # Per-dog physiological baseline (paper §V-B rationale: compare to own baseline).
            dog_hr = float(rng.uniform(55, 95))
            dog_spo2 = float(rng.uniform(94, 99))

            # Cancer-associated signature: stronger exposure response.
            response_gain = 1.6 if label == 1 else 1.0
            tremor_amp = 0.05 * response_gain
            hr_rise = float(rng.normal(8.0 * response_gain, 2.5))
            spo2_drop = float(abs(rng.normal(0.9 * response_gain, 0.3)))

            def make_phase(
                start_ms: int,
                duration_s: float,
                *,
                sniff: bool,
                hr: float,
                spo2: float,
                _tremor: float = tremor_amp,
                _spo2_drop: float = spo2_drop,
            ) -> List[KennelFrame]:
                share = rng.dirichlet(np.ones(4) * 20.0)
                frames = []
                n = int(duration_s * rate_hz)
                step = max(1, n // window)
                for i, k in enumerate(range(0, n, step)):
                    drift = share.copy()
                    if sniff:
                        shift = rng.normal(0, 0.01, size=4)
                        drift = np.clip(drift + shift, 0.05, 0.85)
                        drift /= drift.sum()
                    frames.append(
                        _make_frame(
                            start_ms + int(k / rate_hz * 1000), i, "SNIFF" if sniff else "IDLE",
                            rng, sniff=sniff,
                            total_load=900.0 if sniff else 120.0,
                            load_share=drift,
                            tremor_amp=_tremor if sniff else 0.04,
                            hr_bpm=hr + (rng.normal(0, 1.5) if sniff else rng.normal(0, 1.0)),
                            spo2_pct=max(70.0, spo2 - (_spo2_drop if sniff else 0.0) + rng.normal(0, 0.15)),
                        )
                    )
                return frames

            baseline = make_phase(0, baseline_s, sniff=False, hr=dog_hr, spo2=dog_spo2)
            exposure = make_phase(
                int(baseline_s * 1000), exposure_s, sniff=True,
                hr=dog_hr + hr_rise, spo2=dog_spo2 - spo2_drop,
            )
            post = make_phase(
                int((baseline_s + exposure_s) * 1000), post_s, sniff=False,
                hr=dog_hr + hr_rise * 0.4, spo2=dog_spo2 - spo2_drop * 0.4,
            )

            feats = extract_trial_features(baseline, exposure, post)
            rows.append(feats)
            labels.append(label)
            dogs.append(f"{'AB'[label]}{trial % n_dogs_per_class:02d}")

    X = pd.DataFrame(np.vstack(rows), columns=list(TRIAL_FEATURE_NAMES))
    # Unavailable physiology (-1 sentinels from the extractor) become NaN; the
    # lab's HybridPipeline imputes with train-set medians.
    X = X.replace(-1.0, np.nan)
    return KennelTrialDataset(
        X=X,
        y=np.asarray(labels, dtype=int),
        dog_ids=np.asarray(dogs),
        feature_groups={
            "pressure": [c for c in TRIAL_FEATURE_NAMES if c.startswith(("fsr_", "total_load", "lr_imbalance", "fb_imbalance", "cop_drift"))],
            "proximity": [c for c in TRIAL_FEATURE_NAMES if c.startswith(("ir_", "us_"))],
            "motion": [c for c in TRIAL_FEATURE_NAMES if c.startswith(("acc_", "gyr_"))],
            "physiology": [c for c in TRIAL_FEATURE_NAMES if c.startswith(("imu_temp", "hr_", "spo2_"))],
        },
        feature_names=list(TRIAL_FEATURE_NAMES),
    )
