"""Kennel live server: telemetry streaming, recording sessions, diagnostics.

FastAPI app mirroring the GAIT monorepo reference architecture:
  - background stream source thread (ESP32 TCP or simulation fallback)
  - thread-safe frame queue drained by a predictor worker
  - WebSocket broadcasts: /ws/stream (raw @10 Hz), /ws/diagnostic (windows)
  - Data-Lab recording endpoints

Run: python -m uvicorn --app-dir src qmlkit.api.kennel_server:app --port 8001
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qmlkit.api.training_service import TrainRequest, execute_training_job
from qmlkit.data.biomimetic_voc_generator import BiomimeticVOCGenerator
from qmlkit.evaluation.benchmark_suite import BenchmarkSuite
from qmlkit.hardware.kennel_features import KENNEL_FEATURE_NAMES, extract_window_features
from qmlkit.hardware.kennel_streaming import KennelSettings, create_stream_source
from qmlkit.hardware.protocol import KennelFrame
from qmlkit.hardware.session_recording import RecordingError, RecordingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kennel_server")


class ConnectionManager:
    def __init__(self) -> None:
        self.stream: List[WebSocket] = []
        self.diagnostic: List[WebSocket] = []

    async def connect(self, socket: WebSocket, bucket: List[WebSocket]) -> None:
        await socket.accept()
        bucket.append(socket)

    def disconnect(self, socket: WebSocket, bucket: List[WebSocket]) -> None:
        if socket in bucket:
            bucket.remove(socket)

    async def broadcast(self, payload: Dict[str, Any], bucket: List[WebSocket]) -> None:
        message = json.dumps(payload)
        for connection in list(bucket):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection, bucket)


class StartRecordingRequest(BaseModel):
    dog_id: str = Field(min_length=1, examples=["Rex"])
    sample_id: str = Field(default="S0", min_length=1)
    label: str = Field(min_length=1, description="Class label, e.g. cancer/control/any custom")
    duration_s: float = Field(default=20.0, gt=1.0, le=300.0)


class LabRunRequest(BaseModel):
    dataset: str = Field(default="voc_real", description="voc_real | kennel_synth")
    experiment: str = Field(default="search", description="search | map_ablation | modality_ablation | robustness")
    max_samples: int = Field(default=120, ge=0)
    vqc_epochs: int = Field(default=8, ge=1, le=40)
    n_splits: int = Field(default=5, ge=2, le=10)
    n_components: int = Field(default=6, ge=2, le=12)
    seed: int = Field(default=42)


class KennelPredictor:
    """Model-artifact holder with feature-order validation."""

    def __init__(self, settings: KennelSettings) -> None:
        self.settings = settings
        self.model: Optional[Dict[str, Any]] = None
        self.model_status = "untrained"
        self._load_model()

    def _load_model(self) -> None:
        path = Path("models/kennel_model.joblib")
        if not path.exists():
            logger.warning("No kennel model artifact at %s - diagnostics will be 'untrained'.", path)
            return
        try:
            import joblib

            bundle = joblib.load(path)
            names = list(bundle.get("feature_names", []))
            if names != list(KENNEL_FEATURE_NAMES):
                logger.warning("Model feature order mismatch - ignoring artifact.")
                return
            self.model = bundle
            self.model_status = "ready"
            logger.info("Loaded kennel model artifact (%s).", path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load model artifact: %s", exc)


def _predict_probability(model_bundle: Dict[str, Any], features: np.ndarray) -> Optional[float]:
    try:
        model = model_bundle["model"]
        proba = model.predict_proba(features.reshape(1, -1))[0]
        classes = list(model.classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(max(proba))
    except Exception:
        return None


def create_kennel_app(settings: Optional[KennelSettings] = None) -> FastAPI:
    settings = settings or KennelSettings.from_env()
    manager = ConnectionManager()
    recorder = RecordingManager(settings.data_dir)

    main_loop: Optional[asyncio.AbstractEventLoop] = None
    source, resolved_kind = create_stream_source(settings, seed=42)
    app_state: Dict[str, Any] = {
        "frames_seen": 0,
        "diagnostics": 0,
        "last_prediction": None,
        "ema": None,
        "ui_queue": None,
    }

    def bridge_emit(payload: Dict[str, Any]) -> None:
        if main_loop is not None:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(payload, manager.diagnostic), main_loop
            )

    predictor = KennelPredictor(settings)
    predictor_buffer: List[KennelFrame] = []
    predictor_thread_stop = threading.Event()

    def predictor_worker() -> None:
        """Single queue consumer: recording + diagnostics + UI feed."""
        ui_counter = 0
        while not predictor_thread_stop.is_set():
            frame = source.get_frame(timeout=0.25)
            if frame is None:
                continue
            recorder.add_frame(frame)
            app_state["frames_seen"] += 1

            # ~10 Hz UI feed into the async bridge.
            ui_counter += 1
            if main_loop is not None and ui_counter % 10 == 0:
                loop_queue = app_state["ui_queue"]
                if loop_queue is not None:
                    main_loop.call_soon_threadsafe(loop_queue.put_nowait, frame.to_dict())

            predictor_buffer.append(frame)
            if len(predictor_buffer) < settings.batch_size:
                continue
            window = predictor_buffer[: settings.batch_size]
            del predictor_buffer[: settings.batch_size]

            features = extract_window_features(window)
            if predictor.model is None:
                payload = {
                    "type": "diagnostic",
                    "status": "untrained",
                    "detail": "Collect labelled sessions and run scripts/train_kennel_model.py.",
                }
            else:
                proba_pos = _predict_probability(predictor.model, features)
                if proba_pos is None:
                    continue
                alpha = settings.smoothing_alpha
                ema = (
                    proba_pos
                    if app_state["ema"] is None
                    else alpha * proba_pos + (1 - alpha) * app_state["ema"]
                )
                app_state["ema"] = ema
                confident = max(proba_pos, 1 - proba_pos) >= settings.confidence_threshold
                payload = {
                    "type": "diagnostic",
                    "status": "ok" if confident else "uncertain",
                    "probability_cancer": round(proba_pos, 4),
                    "smoothed_probability": round(float(ema), 4),
                    "confidence_threshold": settings.confidence_threshold,
                }
            app_state["last_prediction"] = payload
            app_state["diagnostics"] += 1
            bridge_emit(payload)

    async def raw_stream_broadcaster() -> None:
        """Forward bridged frames to /ws/stream clients."""
        while True:
            queue_ref = app_state["ui_queue"]
            if queue_ref is None:
                await asyncio.sleep(0.05)
                continue
            payload = await queue_ref.get()
            await manager.broadcast(payload, manager.stream)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal main_loop
        main_loop = asyncio.get_running_loop()
        app_state["ui_queue"] = asyncio.Queue(maxsize=200)
        source.start()
        threading.Thread(target=predictor_worker, daemon=True).start()
        task = asyncio.create_task(raw_stream_broadcaster())
        yield
        task.cancel()
        predictor_thread_stop.set()
        source.stop()

    app = FastAPI(
        title="QMLKit Kennel Live API",
        description="Real-time kennel telemetry ingestion and micro-movement diagnostics",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/kennel/state")
    async def state() -> dict:
        return {
            "stream_source": resolved_kind,
            "connected": getattr(source, "connected", True),
            "frames_seen": app_state["frames_seen"],
            "model_status": predictor.model_status,
            "recorder": recorder.status(),
            "buffer_frames": len(predictor_buffer),
            "batch_size": settings.batch_size,
        }

    @app.post("/api/v1/kennel/start")
    async def start_recording(req: StartRecordingRequest) -> dict:
        try:
            recorder.start(req.dog_id, req.sample_id, req.label, req.duration_s)
        except RecordingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "started", **recorder.status()}

    @app.post("/api/v1/kennel/stop")
    async def stop_recording() -> dict:
        try:
            path = recorder.stop_and_save()
        except RecordingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "saved", "path": str(path) if path else None}

    @app.post("/api/v1/train")
    async def train_model_endpoint(req: TrainRequest):
        """Train any quantum or classical model on demand and return metrics & loss curve."""
        try:
            return execute_training_job(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    @app.post("/api/v1/benchmark/run")
    async def run_benchmark_endpoint(req: Dict[str, Any] = Body(...)):  # noqa: B008 - FastAPI DI
        """Run multi-model comparative leaderboard across Quantum and Classical models."""
        try:
            from sklearn.model_selection import train_test_split
            target_cancer = req.get("target_cancer", "Lung_Cancer")
            n_samples = int(req.get("n_samples_per_class", 60))
            n_qubits = int(req.get("n_qubits", 6))

            generator = BiomimeticVOCGenerator(random_state=42)
            cohort = generator.generate_cohort(
                samples_per_class=n_samples,
                cancer_types=["Healthy", target_cancer]
            )
            y = cohort.metadata["label_binary"].values
            df_X = cohort.df_features

            idx_train, idx_test = train_test_split(
                np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
            )

            suite = BenchmarkSuite(n_qubits=n_qubits, random_state=42)
            df_res = suite.run_benchmark(
                X_train_raw=df_X.iloc[idx_train],
                y_train=y[idx_train],
                X_test_raw=df_X.iloc[idx_test],
                y_test=y[idx_test]
            )
            return {
                "target_cancer": target_cancer,
                "n_qubits": n_qubits,
                "leaderboard": df_res.to_dict(orient="records")
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}") from exc

    @app.get("/api/v1/dataset/stats")
    async def dataset_stats():
        """Provide dataset stats and biomarker definitions."""
        generator = BiomimeticVOCGenerator(random_state=42)
        return {
            "n_sensors": generator.n_sensors,
            "compounds": generator.compounds,
            "cancer_types": generator.voc_cfg.cancer_types,
            "features_per_sensor": ["max_amplitude", "auc_integral", "adsorption_rise", "desorption_decay"]
        }

    @app.websocket("/ws/stream")
    async def ws_stream(socket: WebSocket) -> None:
        await manager.connect(socket, manager.stream)
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(socket, manager.stream)

    @app.websocket("/ws/diagnostic")
    async def ws_diagnostic(socket: WebSocket) -> None:
        await manager.connect(socket, manager.diagnostic)
        if app_state["last_prediction"]:
            await socket.send_text(json.dumps(app_state["last_prediction"]))
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(socket, manager.diagnostic)

    @app.post("/api/v1/kennel/frame")
    async def inject_frame(frame_data: Dict[str, Any] = Body(...)) -> dict:  # noqa: B008 - FastAPI DI pattern
        """Testing/utility hook: push one pre-parsed frame through the queue."""
        parsed = KennelFrame(
            ts_ms=int(frame_data.get("ts_ms", 0)),
            seq=int(frame_data.get("seq", 0)),
            state=str(frame_data.get("state", "IDLE")),
            fsr=[float(v) for v in frame_data.get("fsr", [0, 0, 0, 0])],
            ir=[int(v) for v in frame_data.get("ir", [0] * 6)],
            us_bottom=float(frame_data.get("us", {}).get("bottom", -1)),
            us_top=float(frame_data.get("us", {}).get("top", -1)),
            acc=[float(v) for v in frame_data.get("acc", [0, 0, 0])],
            gyr=[float(v) for v in frame_data.get("gyr", [0, 0, 0])],
            imu_temp_c=float(frame_data.get("imu_temp_c", 0)),
        )
        source._enqueue(parsed)
        return {"queued": True}

    # Serve the static dashboard snapshot when bundled (desktop mode).
    dist_dir = os.environ.get("QMLKIT_FRONTEND_DIST", "")
    if dist_dir and Path(dist_dir).is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="dashboard")
        logger.info("Serving dashboard snapshot from %s", dist_dir)

    return app


def register_lab_routes(app: FastAPI) -> FastAPI:
    """Attach training-lab endpoints (background-thread experiment execution)."""
    import threading
    import time
    import uuid

    lab_jobs: Dict[str, Dict[str, Any]] = {}
    lab_lock = threading.Lock()

    @app.post("/api/v1/lab/runs")
    async def start_lab_run(req: LabRunRequest) -> dict:
        job_id = uuid.uuid4().hex[:12]

        def worker() -> None:
            from qmlkit.config import set_seed
            from qmlkit.lab.data_sources import load_lab_dataset
            from qmlkit.lab.experiments import (
                run_feature_map_ablation,
                run_hybrid_search,
                run_modality_ablation,
                run_robustness,
            )
            from qmlkit.lab.pipeline import PipelineSpec

            try:
                set_seed(req.seed)
                X, y, groups = load_lab_dataset(req.dataset, req.max_samples, req.seed)

                common = dict(n_splits=req.n_splits, seed=req.seed, output_root="outputs/lab")

                def progress(msg: str) -> None:
                    with lab_lock:
                        lab_jobs[job_id]["progress"].append(msg)

                kwargs: Dict[str, Any] = dict(progress_cb=progress)
                if req.experiment == "search":
                    from qmlkit.lab.experiments import default_presets

                    presets = default_presets(vqc_epochs=req.vqc_epochs, seed=req.seed)
                    for p in presets:
                        p.n_components = req.n_components
                    result = run_hybrid_search(X, y, presets=presets, **common, **kwargs)
                elif req.experiment == "map_ablation":
                    result = run_feature_map_ablation(
                        X, y, n_components=req.n_components, vqc_epochs=req.vqc_epochs, **common
                    )
                elif req.experiment == "modality_ablation":
                    base = PipelineSpec(
                        name="CWZZ-QSVM-modality", reduction="pca", embedding="cwzz",
                        head="qsvm", n_components=req.n_components,
                        vqc_epochs=req.vqc_epochs, seed=req.seed,
                    )
                    result = run_modality_ablation(X, y, groups, base_spec=base, **common)
                else:
                    spec = PipelineSpec(
                        name="CWZZ-QSVM-robust", reduction="pca", embedding="cwzz",
                        head="qsvm", n_components=req.n_components,
                        vqc_epochs=req.vqc_epochs, seed=req.seed,
                    )
                    result = run_robustness(X, y, spec=spec, **common)

                with lab_lock:
                    lab_jobs[job_id].update({
                        "status": "done",
                        "finished_at": time.time(),
                        "result": {"run_dir": result["run_dir"],
                                   "leaderboard": result["leaderboard"]},
                    })
            except Exception as exc:
                logger.exception("Lab run %s failed", job_id)
                with lab_lock:
                    lab_jobs[job_id].update({"status": "error", "error": str(exc)})

        with lab_lock:
            lab_jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "request": req.model_dump(),
                "started_at": time.time(),
                "progress": [],
            }
        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job_id, "status": "running"}

    @app.get("/api/v1/lab/runs")
    async def list_lab_runs() -> dict:
        with lab_lock:
            jobs = sorted(lab_jobs.values(), key=lambda j: j["started_at"], reverse=True)
        return {"jobs": [dict(j, progress=j["progress"][-5:]) for j in jobs]}

    @app.get("/api/v1/lab/runs/{job_id}")
    async def get_lab_run(job_id: str) -> dict:
        with lab_lock:
            job = lab_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown lab run.")
        return job

    return app


app = create_kennel_app()
app = register_lab_routes(app)
