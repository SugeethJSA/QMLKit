"""Kennel telemetry frame schema and validation.

Mirrors the firmware contract in ``firmware/kennel_node``: newline-delimited
JSON frames over TCP port 3333.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

FRAME_KEYS = {
    "ts_ms",
    "seq",
    "state",
    "fsr",
    "ir",
    "us",
    "acc",
    "gyr",
    "imu_temp_c",
}
VALID_STATES = {"BOOT", "CALIBRATE", "IDLE", "OCCUPIED", "SNIFF", "COOLDOWN"}


@dataclass
class KennelFrame:
    """Typed view of one telemetry frame."""

    ts_ms: int
    seq: int
    state: str
    fsr: List[float] = field(default_factory=lambda: [0.0] * 4)
    ir: List[int] = field(default_factory=lambda: [0] * 6)
    us_bottom: float = -1.0
    us_top: float = -1.0
    acc: List[float] = field(default_factory=lambda: [0.0] * 3)
    gyr: List[float] = field(default_factory=lambda: [0.0] * 3)
    imu_temp_c: float = -1.0
    # Physiology (firmware v2, MAX30102); -1 = unavailable.
    hr_bpm: float = -1.0
    spo2_pct: float = -1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "seq": self.seq,
            "state": self.state,
            "fsr": list(self.fsr),
            "ir": list(self.ir),
            "us": {"bottom": self.us_bottom, "top": self.us_top},
            "acc": list(self.acc),
            "gyr": list(self.gyr),
            "imu_temp_c": self.imu_temp_c,
            "hr_bpm": self.hr_bpm,
            "spo2_pct": self.spo2_pct,
        }


def parse_frame(payload: Union[str, bytes, Dict[str, Any]]) -> Optional[KennelFrame]:
    """Parse and validate one JSON frame; returns None when malformed."""
    try:
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    missing = FRAME_KEYS - set(data)
    if missing:
        return None

    try:
        fsr = [float(v) for v in data["fsr"]]
        ir = [int(bool(v)) for v in data["ir"]]
        acc = [float(v) for v in data["acc"]]
        gyr = [float(v) for v in data["gyr"]]
        if len(fsr) != 4 or len(ir) != 6 or len(acc) != 3 or len(gyr) != 3:
            return None
        state = str(data["state"])
        if state not in VALID_STATES:
            return None
        us_bottom = float(data["us"].get("bottom", -1.0))
        us_top = float(data["us"].get("top", -1.0))
        return KennelFrame(
            ts_ms=int(data["ts_ms"]),
            seq=int(data["seq"]),
            state=state,
            fsr=fsr,
            ir=ir,
            us_bottom=us_bottom,
            us_top=us_top,
            acc=acc,
            gyr=gyr,
            imu_temp_c=float(data.get("imu_temp_c", -1.0)),
            # Firmware v2 physiological channels (optional).
            hr_bpm=float(data.get("hr_bpm", -1.0)),
            spo2_pct=float(data.get("spo2_pct", -1.0)),
        )
    except (TypeError, ValueError, AttributeError):
        return None
