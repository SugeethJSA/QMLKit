"""Windowed feature extraction from kennel telemetry.

Single source of truth used by BOTH dataset building and live serving
(no train/serve skew - repomono rule). Input is an ordered list of parsed
``KennelFrame`` dicts/values sampled at ~100 Hz.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from qmlkit.hardware.protocol import KennelFrame

FSR_NAMES = ["fsr_fl", "fsr_fr", "fsr_rl", "fsr_rr"]
IR_NAMES = ["ir_bottom_fl", "ir_bottom_fr", "ir_bottom_rl", "ir_bottom_rr", "ir_top_left", "ir_top_right"]

KENNEL_FEATURE_NAMES: List[str] = (
    [f"{n}_mean" for n in FSR_NAMES]
    + [f"{n}_std" for n in FSR_NAMES]
    + [
        "total_load_mean",
        "total_load_std",
        "lr_imbalance_mean",
        "fb_imbalance_mean",
        "cop_drift_std",
    ]
    + [f"{n}_active_frac" for n in IR_NAMES]
    + ["ir_transitions_total"]
    + [
        "us_bottom_mean",
        "us_top_mean",
        "us_bottom_range_std",
        "us_top_slope",
    ]
    + [f"acc_{ax}_{s}" for ax in ("x", "y", "z") for s in ("mean", "std", "rms")]
    + ["acc_jerk_rms", "acc_dom_freq_hz", "acc_band_tremor_4_8hz", "acc_band_sniff_2_5hz", "acc_xy_corr"]
    + [f"gyr_{ax}_std" for ax in ("x", "y", "z")]
    + ["gyr_speed_rms"]
    + ["imu_temp_mean", "hr_bpm_mean", "spo2_pct_mean"]
)

N_KENNEL_FEATURES = len(KENNEL_FEATURE_NAMES)


def _safe_float(v: float, default: float = 0.0) -> float:
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def extract_window_features(frames: Sequence[KennelFrame]) -> np.ndarray:
    """Compute the fixed-order feature vector for one window of frames."""
    if len(frames) < 4:
        return np.full(N_KENNEL_FEATURES, np.nan, dtype=float)

    fsr = np.array([f.fsr for f in frames], dtype=float)          # (T, 4)
    ir = np.array([f.ir for f in frames], dtype=float)            # (T, 6)
    us_b = np.array([_safe_float(f.us_bottom, np.nan) for f in frames], dtype=float)
    us_t = np.array([_safe_float(f.us_top, np.nan) for f in frames], dtype=float)
    acc = np.array([f.acc for f in frames], dtype=float)          # (T, 3)
    gyr = np.array([f.gyr for f in frames], dtype=float)
    temp = np.array([_safe_float(f.imu_temp_c, 0.0) for f in frames])

    total = fsr.sum(axis=1)
    left_right = fsr[:, :2].sum(axis=1) - fsr[:, 2:].sum(axis=1)   # front vs rear naming per corner map
    front_back = fsr[:, 0] + fsr[:, 2] - (fsr[:, 1] + fsr[:, 3])
    # Center-of-pressure proxy: normalized diagonal drift magnitude.
    norm = np.where(total > 0, total, 1.0)
    cop_x = (fsr[:, 1] + fsr[:, 3]) / norm - 0.5
    cop_y = (fsr[:, 2] + fsr[:, 3]) / norm - 0.5
    cop_drift = float(np.std(np.hypot(cop_x, cop_y)))

    ir_active = ir.mean(axis=0)
    transitions = int(np.abs(np.diff(ir > 0.5, axis=0)).sum())

    us_b_clean = us_b[us_b > 0]
    us_t_clean = us_t[us_t > 0]

    jerk = np.diff(acc, axis=0) * 100.0                            # per-second scale @100 Hz
    mag = np.linalg.norm(acc, axis=1)

    # Dominant tremor/sniff frequency via zero-padded FFT on accel magnitude.
    sig = mag - mag.mean()
    spectrum = np.abs(np.fft.rfft(sig, 512))
    freqs = np.fft.rfftfreq(512, d=0.01)                           # 100 Hz sampling
    dom_freq = float(freqs[int(np.argmax(spectrum[1:]) + 1)])
    band_tremor = float(spectrum[(freqs >= 4) & (freqs <= 8)].sum())
    band_sniff = float(spectrum[(freqs >= 2) & (freqs <= 5)].sum())

    if len(acc[:, 0]) > 2 and np.std(acc[:, 0]) > 1e-9 and np.std(acc[:, 1]) > 1e-9:
        xy_corr = float(np.corrcoef(acc[:, 0], acc[:, 1])[0, 1])
    else:
        xy_corr = 0.0

    def slope(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return 0.0
        t_idx = np.arange(len(x), dtype=float)
        return float(np.polyfit(t_idx, x, 1)[0])

    values: List[float] = []
    values += list(fsr.mean(axis=0))
    values += list(fsr.std(axis=0))
    values += [
        float(total.mean()),
        float(total.std()),
        float(left_right.mean()),
        float(front_back.mean()),
        cop_drift,
    ]
    values += list(ir_active)
    values += [transitions]
    values += [
        float(us_b_clean.mean()) if us_b_clean.size else -1.0,
        float(us_t_clean.mean()) if us_t_clean.size else -1.0,
        float(us_b_clean.std()) if us_b_clean.size else 0.0,
        slope(us_t),
    ]
    for axis in range(3):
        a = acc[:, axis]
        values += [float(a.mean()), float(a.std()), float(np.sqrt((a**2).mean()))]
    values += [
        float(np.sqrt((jerk**2).mean())),
        dom_freq,
        band_tremor,
        band_sniff,
        xy_corr,
    ]
    for axis in range(3):
        values.append(float(gyr[:, axis].std()))
    values += [float(np.sqrt((gyr**2).mean()))]
    values += [float(temp.mean())]

    hr = np.array([_safe_float(f.hr_bpm, np.nan) for f in frames], dtype=float)
    spo2 = np.array([_safe_float(f.spo2_pct, np.nan) for f in frames], dtype=float)
    hr_valid = hr[hr > 0]
    spo2_valid = spo2[spo2 > 0]
    values += [
        float(hr_valid.mean()) if hr_valid.size else -1.0,
        float(spo2_valid.mean()) if spo2_valid.size else -1.0,
    ]

    return np.asarray(values, dtype=float)


def frames_from_dicts(rows: Sequence[dict]) -> List[KennelFrame]:
    """Rehydrate parsed frame dicts (e.g., CSV rows) into KennelFrame objects."""
    frames = []
    for row in rows:
        us = row.get("us") or {}
        frames.append(
            KennelFrame(
                ts_ms=int(row.get("ts_ms", 0)),
                seq=int(row.get("seq", 0)),
                state=str(row.get("state", "IDLE")),
                fsr=[float(v) for v in row.get("fsr", [])],
                ir=[int(v) for v in row.get("ir", [])],
                us_bottom=float(us.get("bottom", -1.0)),
                us_top=float(us.get("top", -1.0)),
                acc=[float(v) for v in row.get("acc", [])],
                gyr=[float(v) for v in row.get("gyr", [])],
                imu_temp_c=float(row.get("imu_temp_c", -1.0)),
                hr_bpm=float(row.get("hr_bpm", -1.0)),
                spo2_pct=float(row.get("spo2_pct", -1.0)),
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Feature-group registry (sensor-modality ablation, manuscript §VII-D)
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Dict[str, List[str]] = {
    "pressure": (
        [f"{n}_{s}" for n in FSR_NAMES for s in ("mean", "std")]
        + ["total_load_mean", "total_load_std", "lr_imbalance_mean",
           "fb_imbalance_mean", "cop_drift_std"]
    ),
    "proximity": (
        [f"{n}_active_frac" for n in IR_NAMES]
        + ["ir_transitions_total", "us_bottom_mean", "us_top_mean",
           "us_bottom_range_std", "us_top_slope"]
    ),
    "motion": (
        [f"acc_{ax}_{s}" for ax in ("x", "y", "z") for s in ("mean", "std", "rms")]
        + ["acc_jerk_rms", "acc_dom_freq_hz", "acc_band_tremor_4_8hz",
           "acc_band_sniff_2_5hz", "acc_xy_corr"]
        + [f"gyr_{ax}_std" for ax in ("x", "y", "z")]
        + ["gyr_speed_rms"]
    ),
    "physiology": ["imu_temp_mean", "hr_bpm_mean", "spo2_pct_mean"],
}


# ---------------------------------------------------------------------------
# Trial-level extraction with baseline/exposure/post segmentation (§V-B)
# ---------------------------------------------------------------------------

DELTA_SOURCE_KEYS = [
    # (window-feature index lookup by name) baseline-relative channels
    "total_load_mean",
    "us_bottom_mean",
    "us_top_mean",
    "acc_x_mean",
    "acc_y_mean",
    "acc_z_mean",
    "acc_jerk_rms",
    "gyr_speed_rms",
    "acc_band_sniff_2_5hz",
]

TRIAL_FEATURE_NAMES = (
    KENNEL_FEATURE_NAMES                      # exposure-window features
    + [f"{k}_delta" for k in DELTA_SOURCE_KEYS]   # Δx = x_exposure - x_baseline
    + ["post_recovery_delta"]                 # return toward baseline after exposure
)

N_TRIAL_FEATURES = len(TRIAL_FEATURE_NAMES)


def _feature_index(name: str) -> int:
    return KENNEL_FEATURE_NAMES.index(name)


def extract_trial_features(
    baseline_frames: Sequence[KennelFrame],
    exposure_frames: Sequence[KennelFrame],
    post_frames: Optional[Sequence[KennelFrame]] = None,
) -> np.ndarray:
    """Trial-level vector with baseline-relative changes (manuscript §V-B/D).

    Δx = x_exposure − x_baseline for key physiological/motion/pressure channels,
    plus a recovery indicator from the post-exposure window when available.
    """
    base_feats = extract_window_features(baseline_frames)
    expo_feats = extract_window_features(exposure_frames)

    deltas = []
    for key in DELTA_SOURCE_KEYS:
        idx = _feature_index(key)
        deltas.append(expo_feats[idx] - base_feats[idx])

    if post_frames:
        post_feats = extract_window_features(post_frames)
        # Recovery: how far exposure moved from baseline, relative to post movement.
        exposure_shift = abs(expo_feats[_feature_index("total_load_mean")]
                             - base_feats[_feature_index("total_load_mean")])
        post_shift = abs(post_feats[_feature_index("total_load_mean")]
                         - base_feats[_feature_index("total_load_mean")])
        recovery = (post_shift / exposure_shift) if exposure_shift > 1e-6 else 0.0
    else:
        recovery = 0.0

    return np.concatenate([expo_feats, np.asarray(deltas, dtype=float), [recovery]])

