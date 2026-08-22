"""Tests for kennel hardware ingestion: protocol, streaming, features, recording."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qmlkit.api.kennel_server import create_kennel_app
from qmlkit.hardware.kennel_features import (
    KENNEL_FEATURE_NAMES,
    N_KENNEL_FEATURES,
    extract_window_features,
    frames_from_dicts,
)
from qmlkit.hardware.kennel_streaming import (
    KennelSettings,
    SimulatedKennelSource,
    frame_to_json,
)
from qmlkit.hardware.protocol import KennelFrame, parse_frame
from qmlkit.hardware.session_recording import RecordingError, RecordingManager

VALID_FRAME = {
    "ts_ms": 1000,
    "seq": 7,
    "state": "SNIFF",
    "fsr": [512.0, 498.0, 602.0, 590.0],
    "ir": [1, 1, 0, 1, 0, 0],
    "us": {"bottom": 41.2, "top": 88.7},
    "acc": [-0.12, 9.78, 0.31],
    "gyr": [0.01, -0.02, 0.0],
    "imu_temp_c": 36.4,
}


class TestProtocol:
    def test_valid_frame_roundtrip(self):
        line = json.dumps(VALID_FRAME)
        frame = parse_frame(line)
        assert frame is not None
        assert frame.state == "SNIFF" and frame.seq == 7
        assert len(frame.fsr) == 4 and len(frame.ir) == 6
        assert frame.us_bottom == pytest.approx(41.2)

    def test_missing_key_returns_none(self):
        bad = dict(VALID_FRAME)
        del bad["gyr"]
        assert parse_frame(json.dumps(bad)) is None

    def test_bad_state_returns_none(self):
        bad = dict(VALID_FRAME, state="WARP")
        assert parse_frame(json.dumps(bad)) is None

    def test_malformed_json_returns_none(self):
        assert parse_frame("{not json") is None

    def test_frame_to_json_matches_schema(self):
        frame = parse_frame(json.dumps(VALID_FRAME))
        payload = json.loads(frame_to_json(frame))
        assert set(payload) == set(VALID_FRAME)


class TestSimulationSource:
    def test_produces_valid_frames(self):
        source = SimulatedKennelSource(rate_hz=500.0, seed=1)
        source.start()
        frames = []
        deadline = time.monotonic() + 3.0
        while len(frames) < 20 and time.monotonic() < deadline:
            f = source.get_frame(timeout=0.5)
            if f is not None:
                frames.append(f)
        source.stop()
        assert len(frames) >= 20
        seqs = [f.seq for f in frames]
        assert seqs == sorted(seqs)
        assert all(f.state in {"IDLE", "SNIFF", "COOLDOWN"} for f in frames)
        assert all(len(f.acc) == 3 for f in frames)


class TestFeatureExtraction:
    def _synthetic_window(self, n=400):
        rng = np.random.default_rng(7)
        t = np.arange(n) * 10  # ms
        sniff = 0.35 * np.sin(2 * np.pi * 3.0 * t / 1000.0)
        rows = []
        for i in range(n):
            rows.append(
                {
                    "ts_ms": int(t[i]),
                    "seq": i,
                    "state": "SNIFF",
                    "fsr": [220.0 + rng.normal(0, 5)] * 4,
                    "ir": [1, 0, 1, 0, 1, 0],
                    "us": {"bottom": 38.0 + rng.normal(), "top": 55.0},
                    "acc": [
                        float(sniff[i] + rng.normal(0, 0.05)),
                        9.78 + rng.normal(0, 0.05),
                        float(rng.normal(0, 0.05)),
                    ],
                    "gyr": [float(rng.normal(0, 0.02)) for _ in range(3)],
                    "imu_temp_c": 37.0,
                }
            )
        return frames_from_dicts(rows)

    def test_feature_vector_shape_and_order(self):
        feats = extract_window_features(self._synthetic_window())
        assert feats.shape == (N_KENNEL_FEATURES,)
        assert len(KENNEL_FEATURE_NAMES) == N_KENNEL_FEATURES
        assert np.isfinite(feats).all()

    def test_deterministic_for_same_window(self):
        window = self._synthetic_window()
        a = extract_window_features(window)
        b = extract_window_features(window)
        assert np.allclose(a, b)

    def test_short_window_returns_nan_vector(self):
        tiny = frames_from_dicts([{"ts_ms": 0, "seq": 0, "state": "IDLE", "fsr": [0] * 4, "ir": [0] * 6}])
        assert np.isnan(extract_window_features(tiny)).all()


class TestRecordingManager:
    def test_session_roundtrip(self, tmp_path):
        manager = RecordingManager(tmp_path)
        manager.start("Rex", "S1", "cancer", duration_s=60)
        for i in range(30):
            manager.add_frame(parse_frame(json.dumps(dict(VALID_FRAME, seq=i))))
        path = manager.stop_and_save()
        assert path is not None and path.exists()
        rows = manager.load_csv_rows(path)
        assert len(rows) == 30
        assert rows[0]["fsr"] == [512.0, 498.0, 602.0, 590.0]
        # Trial auto-increment on the next session.
        manager.start("Rex", "S1", "cancer", duration_s=60)
        manager.add_frame(parse_frame(json.dumps(VALID_FRAME)))
        path2 = manager.stop_and_save()
        assert path2.name != path.name

    def test_double_start_raises(self, tmp_path):
        manager = RecordingManager(tmp_path)
        manager.start("A", "B", "c")
        with pytest.raises(RecordingError):
            manager.start("D", "E", "f")

    def test_stop_without_start_raises(self, tmp_path):
        manager = RecordingManager(tmp_path)
        with pytest.raises(RecordingError):
            manager.stop_and_save()


class TestKennelServer:
    def test_state_and_stream_endpoints(self):
        settings = KennelSettings(source="simulation", batch_size=50, sim_rate_hz=400.0)
        app = create_kennel_app(settings=settings)
        # Context manager runs lifespan -> simulation source + worker threads.
        with TestClient(app) as client:
            res = client.get("/api/v1/kennel/state")
            assert res.status_code == 200
            body = res.json()
            assert body["stream_source"] == "simulation"
            assert body["model_status"] == "untrained"

            res = client.post(
                "/api/v1/kennel/start",
                json={"dog_id": "Rex", "sample_id": "S0", "label": "calm", "duration_s": 5},
            )
            assert res.status_code == 200
            time.sleep(1.0)  # let the simulation feed some frames
            state_mid = client.get("/api/v1/kennel/state").json()
            assert state_mid["recorder"]["samples"] > 0
            res = client.post("/api/v1/kennel/stop")
            assert res.status_code == 200
            assert res.json()["path"] is not None

            with client.websocket_connect("/ws/stream") as ws:
                payload = json.loads(ws.receive_text())
                assert "fsr" in payload and "acc" in payload

    def test_frame_injection_hook(self):
        settings = KennelSettings(source="simulation", batch_size=20, sim_rate_hz=200.0)
        app = create_kennel_app(settings=settings)
        client = TestClient(app)
        res = client.post("/api/v1/kennel/frame", json=VALID_FRAME)
        assert res.status_code == 200 and res.json()["queued"]
