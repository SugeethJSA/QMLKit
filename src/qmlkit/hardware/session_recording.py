"""Labelled recording sessions for the kennel Data Lab (repomono pattern)."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from qmlkit.hardware.protocol import KennelFrame


class RecordingError(RuntimeError):
    pass


class RecordingManager:
    """Collects frames for one labelled session and flushes to CSV.

    Files land at ``<data_dir>/<label>/<dog_id>_<label><trial>.csv`` with
    auto-incrementing trial numbers per (dog_id, label) pair.
    """

    def __init__(self, data_dir: str | Path = "data/kennel"):
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._meta: Dict[str, str] = {}
        self._started_at: float = 0.0
        self._duration_s: float = 0.0

    @property
    def is_recording(self) -> bool:
        return bool(self._meta)

    def status(self) -> dict:
        with self._lock:
            elapsed = time.monotonic() - self._started_at if self.is_recording else 0.0
            return {
                "recording": self.is_recording,
                "samples": len(self._rows),
                "elapsed_s": round(elapsed, 1),
                **({"target_s": self._duration_s, **self._meta} if self.is_recording else {}),
            }

    def start(self, dog_id: str, sample_id: str, label: str, duration_s: float = 20.0) -> None:
        with self._lock:
            if self.is_recording:
                raise RecordingError("A recording session is already running.")
            self._meta = {
                "dog_id": dog_id,
                "sample_id": sample_id,
                "label": label,
            }
            self._duration_s = float(duration_s)
            self._started_at = time.monotonic()
            self._rows = []

    def add_frame(self, frame: KennelFrame) -> None:
        if not self.is_recording:
            return
        row = frame.to_dict()
        row["wall_ts"] = time.time()
        with self._lock:
            if self.is_recording:
                self._rows.append(row)

    def progress(self) -> float:
        if not self.is_recording:
            return 0.0
        return min(1.0, (time.monotonic() - self._started_at) / max(1e-6, self._duration_s))

    def should_finish(self) -> bool:
        return self.is_recording and self.progress() >= 1.0

    def stop_and_save(self) -> Optional[Path]:
        with self._lock:
            if not self.is_recording:
                raise RecordingError("No recording session is running.")
            meta, rows = self._meta, self._rows
            self._meta = {}
            self._rows = []

        if not rows:
            raise RecordingError("No frames captured; nothing to save.")

        out_dir = self.data_dir / meta["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        trial = 1
        while True:
            path = out_dir / f"{meta['dog_id']}_{meta['label']}{trial}.csv"
            if not path.exists():
                break
            trial += 1

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "ts_ms", "seq", "state",
                    "fsr_0", "fsr_1", "fsr_2", "fsr_3",
                    "ir_0", "ir_1", "ir_2", "ir_3", "ir_4", "ir_5",
                    "us_bottom", "us_top",
                    "acc_x", "acc_y", "acc_z",
                    "gyr_x", "gyr_y", "gyr_z",
                    "imu_temp_c", "hr_bpm", "spo2_pct", "wall_ts",
                ]
            )
            for r in rows:
                writer.writerow(
                    [r["ts_ms"], r["seq"], r["state"]]
                    + [float(v) for v in r["fsr"]]
                    + [int(v) for v in r["ir"]]
                    + [r["us"]["bottom"], r["us"]["top"]]
                    + [float(v) for v in r["acc"]]
                    + [float(v) for v in r["gyr"]]
                    +[r["imu_temp_c"], r["hr_bpm"], r["spo2_pct"], r["wall_ts"]]
                )
        return path

    def load_csv_rows(self, path: str | Path) -> list[dict]:
        """Read back a saved session CSV into frame-like dicts."""
        rows: list[dict] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for rec in reader:
                rows.append(
                    {
                        "ts_ms": int(float(rec["ts_ms"])),
                        "seq": int(float(rec["seq"])),
                        "state": rec["state"],
                        "fsr": [float(rec[f"fsr_{i}"]) for i in range(4)],
                        "ir": [int(float(rec[f"ir_{i}"])) for i in range(6)],
                        "acc": [float(rec[k]) for k in ("acc_x", "acc_y", "acc_z")],
                        "gyr": [float(rec[k]) for k in ("gyr_x", "gyr_y", "gyr_z")],
                        "imu_temp_c": float(rec["imu_temp_c"]),
                        "hr_bpm": float(rec.get("hr_bpm") or -1.0),
                        "spo2_pct": float(rec.get("spo2_pct") or -1.0),
                        "us": {"bottom": float(rec["us_bottom"]), "top": float(rec["us_top"])},
                    }
                )
        return rows
