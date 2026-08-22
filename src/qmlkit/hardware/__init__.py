"""Hardware ingestion layer for the kennel telemetry node (ESP32)."""

from qmlkit.hardware.protocol import FRAME_KEYS, KennelFrame, parse_frame
from qmlkit.hardware.kennel_streaming import (
    SimulatedKennelSource,
    TcpKennelSource,
    create_stream_source,
)
from qmlkit.hardware.session_recording import RecordingManager

__all__ = [
    "FRAME_KEYS",
    "KennelFrame",
    "parse_frame",
    "SimulatedKennelSource",
    "TcpKennelSource",
    "create_stream_source",
    "RecordingManager",
]
