"""Telemetry stream sources: hardware TCP client and simulation fallback.

Mirrors the GAIT monorepo streaming pattern: a background thread feeds a
thread-safe queue; the server drains it for feature extraction/broadcast.
"""

from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from qmlkit.hardware.protocol import KennelFrame, parse_frame


@dataclass
class KennelSettings:
    """Environment-driven configuration (QMLKIT_* variables)."""

    esp32_host: str = "192.168.4.1"
    esp32_port: int = 3333
    source: str = "auto"  # auto | hardware | simulation
    batch_size: int = 400
    confidence_threshold: float = 0.6
    smoothing_alpha: float = 0.3
    data_dir: str = "data/kennel"
    sim_rate_hz: float = 100.0

    @classmethod
    def from_env(cls) -> "KennelSettings":
        import os

        get = os.environ.get
        return cls(
            esp32_host=get("QMLKIT_KENNEL_ESP32_IP", cls.esp32_host),
            esp32_port=int(get("QMLKIT_KENNEL_ESP32_PORT", cls.esp32_port)),
            source=get("QMLKIT_KENNEL_STREAM_SOURCE", cls.source),
            batch_size=int(get("QMLKIT_KENNEL_BATCH_SIZE", cls.batch_size)),
            confidence_threshold=float(get("QMLKIT_KENNEL_CONFIDENCE_THRESHOLD", cls.confidence_threshold)),
            smoothing_alpha=float(get("QMLKIT_KENNEL_SMOOTHING_ALPHA", cls.smoothing_alpha)),
            data_dir=get("QMLKIT_KENNEL_DATA_DIR", cls.data_dir),
        )


class StreamSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self, timeout: Optional[float] = None) -> Optional[KennelFrame]: ...


class _BaseSource:
    """Queue plumbing shared by both sources."""

    def __init__(self, queue_maxsize: int = 4000):
        self._queue: "queue.Queue[KennelFrame]" = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_frame(self, timeout: Optional[float] = None) -> Optional[KennelFrame]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # Queue with oldest-drop on overflow keeps the live dashboard responsive.
    def _enqueue(self, frame: KennelFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(frame)

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class TcpKennelSource(_BaseSource):
    """Connects to the ESP32 TCP JSON server and parses frames."""

    def __init__(self, host: str, port: int, reconnect_delay_s: float = 2.0):
        super().__init__()
        self.host = host
        self.port = port
        self.reconnect_delay_s = reconnect_delay_s
        self.connected = False

    def _run(self) -> None:
        buffer = b""
        while not self._stop_event.is_set():
            sock: Optional[socket.socket] = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.settimeout(1.0)
                self.connected = True
                while not self._stop_event.is_set():
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        frame = parse_frame(line.strip())
                        if frame is not None:
                            self._enqueue(frame)
            except OSError:
                pass
            finally:
                self.connected = False
                if sock:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay_s)


class SimulatedKennelSource(_BaseSource):
    """Generates realistic synthetic micro-movement telemetry.

    Produces rest tremor (4-8 Hz), sniffing oscillation bursts during simulated
    SNIFF windows, posture shifts via random-walk load distribution, and
    ultrasonic distances correlated with the state machine. Enables the whole
    pipeline/GUI to run without hardware.
    """

    def __init__(self, rate_hz: float = 100.0, seed: int = 42):
        super().__init__()
        self.rate_hz = rate_hz
        self.rng = np.random.default_rng(seed)
        self._seq = 0
        self._t = 0.0
        self._load_share = np.array([0.25, 0.25, 0.25, 0.25])
        self._state_until = 0.0
        self._state = "IDLE"

    def _advance_sim_state(self) -> None:
        if self._t >= self._state_until:
            cycle = {
                "IDLE": ("SNIFF", 8.0),
                "SNIFF": ("COOLDOWN", 4.0),
                "COOLDOWN": ("IDLE", 12.0),
            }
            self._state, dur = cycle[self._state]
            self._state_until = self._t + dur

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_ts = time.perf_counter()
        while not self._stop_event.is_set():
            self._advance_sim_state()
            t = self._t

            # Posture: slow random walk in load distribution.
            self._load_share += self.rng.normal(0, 0.002, size=4)
            self._load_share = np.clip(self._load_share, 0.05, 0.9)
            self._load_share /= self._load_share.sum()

            sniff_active = self._state == "SNIFF"
            # Micro-movements: resting tremor + sniffing bursts (~3 Hz sniffs).
            tremor = np.sin(2 * math.pi * 6.0 * t + 0.7) * 0.05
            sniff_wave = (
                math.sin(2 * math.pi * 3.0 * t) * 0.35 + self.rng.normal(0, 0.03)
                if sniff_active else 0.0
            )
            acc = [
                0.02 * self.rng.normal() + tremor + sniff_wave,
                9.78 + 0.04 * self.rng.normal() + 0.5 * sniff_wave,
                0.31 * self.rng.normal() + 0.15 * math.sin(2 * math.pi * 2.2 * t) * sniff_active,
            ]
            gyr = [0.01 * self.rng.normal() + 0.06 * sniff_wave * sniff_active for _ in range(3)]

            total_load = 900 if sniff_active or self._state == "OCCUPIED" else 120
            fsr = [int(total_load * share) for share in self._load_share]
            us_bottom = 38.0 + 4.0 * self.rng.normal() + (-12.0 if sniff_active else 0.0)
            us_top = 95.0 + 6.0 * self.rng.normal() + (-45.0 if sniff_active else 0.0)
            ir_top = [1 if sniff_active and v > 0.5 else 0 for v in self.rng.random(2)]
            ir_bottom = [1 if s else 0 for s in self.rng.random(4) < (0.7 if sniff_active else 0.15)]

            frame = KennelFrame(
                ts_ms=int(t * 1000),
                seq=self._seq,
                state=self._state,
                fsr=[float(v) for v in fsr],
                ir=[*ir_bottom, *ir_top],
                us_bottom=float(np.clip(us_bottom, 5, 200)),
                us_top=float(np.clip(us_top, 5, 200)),
                acc=[float(a) for a in acc],
                gyr=[float(g) for g in gyr],
                imu_temp_c=37.2 + 0.2 * self.rng.normal(),
            )
            self._enqueue(frame)
            self._seq += 1
            self._t += period
            next_ts += period
            sleep_for = next_ts - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_ts = time.perf_counter()


def create_stream_source(settings: KennelSettings, seed: int = 42) -> tuple[StreamSource, str]:
    """Factory honouring ``source`` = auto | hardware | simulation.

    ``auto`` prefers hardware but transparently falls back to simulation when
    the ESP32 is unreachable (checked with a quick probe).
    """
    if settings.source == "simulation":
        return SimulatedKennelSource(rate_hz=settings.sim_rate_hz, seed=seed), "simulation"

    tcp = TcpKennelSource(settings.esp32_host, settings.esp32_port)
    if settings.source == "hardware":
        return tcp, "hardware"

    # Auto: probe once; fall back to simulation when unreachable.
    try:
        probe = socket.create_connection((settings.esp32_host, settings.esp32_port), timeout=2.0)
        probe.close()
        return tcp, "hardware"
    except OSError:
        return SimulatedKennelSource(rate_hz=settings.sim_rate_hz, seed=seed), "simulation"


def frame_to_json(frame: KennelFrame) -> str:
    return json.dumps(frame.to_dict())
