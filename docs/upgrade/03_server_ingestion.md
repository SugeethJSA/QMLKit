# Workstream C — Server-Side Ingestion & Kennel ML

**Location:** `src/qmlkit/hardware/` (new package) + FastAPI endpoints

```
src/qmlkit/hardware/
├── __init__.py
├── protocol.py        # Frame schema, parsing, validation (shared with firmware docs)
├── kennel_streaming.py  # TCP client to ESP32:3333 + simulation fallback source
├── kennel_features.py   # Single-source-of-truth windowed feature extractor
└── session_recording.py # Labelled Data-Lab recording sessions → CSV
```

## C1. Streaming (`kennel_streaming.py`)

Mirrors `gait/streaming.py` from repomono:

- Background thread connects to `KENNEL_ESP32_IP:3333`, parses newline-delimited
  JSON frames into a typed dict, pushes to a thread-safe queue.
- **Simulation fallback** (`KENNEL_STREAM_SOURCE=auto|hardware|simulation`):
  synthetic micro-movement generator produces realistic frames (resting tremor,
  sniffing oscillation bursts, posture shifts) so the GUI and pipeline work
  without hardware.
- Gap/reconnect handling; monotonic-sequence continuity checks.

## C2. Feature extraction (`kennel_features.py`) — ~36 features

Single code path used at BOTH training and serving time (no train/serve skew —
repomono rule). Window = configurable samples (default 400 @100 Hz):

| Group | Features |
|---|---|
| FSR (4ch) | per-channel mean/std, total load mean/std, corner imbalance L/R + F/B, center-of-pressure drift magnitude |
| IR (6ch) | per-channel active fraction, total transitions, bottom-vs-top activity ratio |
| Ultrasonic (2ch) | mean/min distance, range variance, approach slope |
| Collar accel | per-axis mean/std/RMS, jerk RMS (Δaccel), dominant spectral frequency + band energies (tremor 4–8 Hz, sniff 2–5 Hz), pairwise axis correlation |
| Collar gyro | per-axis std (stability), angular-speed RMS |
| Meta | imu_temp mean |

Output: fixed-ordered float vector + names list exported for dataset building.

## C3. Session recording (`session_recording.py`)

Repomono Data-Lab pattern: POST `/api/start {dog_id, sample_id, label,
duration_s}` → buffered rows auto-flush to
`data/kennel/<label>/<dog_id>_<label>_<trial>.csv`; trial numbers auto-increment;
live progress via `/api/state`.

## C4. Model layer (honest cold start)

- **Now:** RandomForest baseline trained via new CLI
  `scripts/train_kennel_model.py` (builds dataset from `data/kennel/**.csv`,
  leave-one-dog-out validation like GAIT's LOSO, artifact →
  `models/kennel_model.joblib` with feature order/classes/metrics manifest).
- **Serving:** background thread buffers queue rows → window every
  `KENNEL_BATCH_SIZE` → features (same extractor) → probability averaging +
  EMA smoothing → broadcast `/ws/diagnostic`. Confidence below threshold →
  `"uncertain"`.
- **Untrained state:** endpoints return explicit `{status:"untrained"}`
  diagnostics instead of pretending.
- Later: QSVM/VQC path reusing existing `BenchmarkSuite` machinery once labelled
  sessions exist.

## C5. API surface (new `api/kennel_routes.py`, mounted in server app)

| Endpoint | Type | Purpose |
|---|---|---|
| `/api/v1/kennel/state` | GET | connection, stream source, buffer depth, session status |
| `/api/v1/kennel/start` / `stop` | POST | begin/end labelled recording session |
| `/ws/stream` | WS | raw telemetry broadcast (~10 Hz downsampled for UI) |
| `/ws/diagnostic` | WS | prediction events (window-level) |

Env config (QMLKIT_ prefixed): `KENNEL_ESP32_IP`, `KENNEL_ESP32_PORT`,
`KENNEL_STREAM_SOURCE`, `KENNEL_BATCH_SIZE`, `KENNEL_CONFIDENCE_THRESHOLD`,
`KENNEL_SMOOTHING_ALPHA`, `KENNEL_DATA_DIR`.

Tests: frame parsing/validation, feature extractor shape+order stability on
synthetic windows, simulation source smoke, recording manager round-trip.
