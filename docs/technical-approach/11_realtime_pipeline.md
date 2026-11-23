# 11 — Realtime Pipeline

`src/qmlkit/hardware/kennel_streaming.py`, `kennel_features.py`,
`session_recording.py`, and `src/qmlkit/api/kennel_server.py`.

---

## 1. Ingestion sources

`create_stream_source()` (`kennel_streaming.py:226-245`) is the single entry
point. It returns `(source, label)`.

| `QMLKIT_KENNEL_STREAM_SOURCE` | Source | Label |
|---|---|---|
| `simulation` | `SimulatedKennelSource(rate_hz, seed=42)` | `"simulation"` |
| `hardware` | `TcpKennelSource(host, port)` | `"hardware"` |
| `auto` *(default)* | one-shot probe, then hardware or simulation | whichever applied |

```python
# auto: one 2.0 s probe, no retry
try:
    socket.create_connection((host, port), timeout=2.0).close()
    return TcpKennelSource(host, port), "hardware"
except OSError:
    return SimulatedKennelSource(rate_hz, seed=42), "simulation"
```

**Behavioural consequence:** the probe is **one-shot and never retried**. If the
ESP32 is merely slow to boot when the server starts, the process stays in
simulation for its entire lifetime. A retry loop (or a manual restart) is
required. Worth flagging in the demo script so the presenter restarts the backend
after the board is up.

### 1.1 `KennelSettings`

| Field | Default | Env var |
|---|---|---|
| `esp32_host` | `192.168.4.1` | `QMLKIT_KENNEL_ESP32_IP` |
| `esp32_port` | `3333` | `QMLKIT_KENNEL_ESP32_PORT` |
| `source` | `auto` | `QMLKIT_KENNEL_STREAM_SOURCE` |
| `batch_size` | `400` | `QMLKIT_KENNEL_BATCH_SIZE` |
| `confidence_threshold` | `0.6` | `QMLKIT_KENNEL_CONFIDENCE_THRESHOLD` |
| `smoothing_alpha` | `0.3` | `QMLKIT_KENNEL_SMOOTHING_ALPHA` |
| `data_dir` | `data/kennel` | `QMLKIT_KENNEL_DATA_DIR` |
| `sim_rate_hz` | `100.0` | *(constructor only)* |

`192.168.4.1` is the ESP32 SoftAP default — but the firmware does **not** start a
SoftAP (see [`10_kennel_hardware.md`](10_kennel_hardware.md) F2). On a shared
LAN the correct value is the board's DHCP address or `kennel.local`.

### 1.2 `TcpKennelSource`

| Aspect | Value | Line |
|---|---|---|
| connect | `socket.create_connection((host, port), timeout=5.0)` | `:112` |
| recv timeout | `settimeout(1.0)` (keep-alive on idle) | `:113` |
| recv size | `4096` bytes | |
| framing | accumulate → split on `b"\n"` → `parse_frame(line.strip())` | `:122-127` |
| reconnect delay | `2.0 s` **fixed — no exponential backoff** | `:100, :137-138` |
| error handling | `OSError` swallowed; `finally` closes and sets `connected = False` | `:128-136` |

`self.connected` is a plain `bool` written by the producer thread and read by the
HTTP handler — a benign race in CPython, but not formally synchronised.

### 1.3 `SimulatedKennelSource`

Constructor `(rate_hz=100.0, seed=42)`; `rng = np.random.default_rng(seed)`.

**State cycle** (`_advance_sim_state`, `:160-168`) — a hardcoded 3-state rotation
with a 24 s period:

| From | To | Duration |
|---|---|---|
| `IDLE` | `SNIFF` | 8.0 s |
| `SNIFF` | `COOLDOWN` | 4.0 s |
| `COOLDOWN` | `IDLE` | 12.0 s |

The simulator **never emits** `OCCUPIED`, `BOOT`, or `CALIBRATE`
(`tests/test_hardware_ingest.py:81` asserts the state is a subset of these three).

**Per-sample signal model** (`:170-223`):

| Quantity | Expression |
|---|---|
| load-share random walk | `+= N(0, 0.002, size=4)`, then `clip(0.05, 0.9)` and renormalise |
| `tremor` | `sin(2π·6.0·t + 0.7) · 0.05` — 6 Hz |
| `sniff_wave` (SNIFF only) | `sin(2π·3.0·t)·0.35 + N(0, 0.03)` — 3 Hz |
| `acc[0]` | `0.02·N(0,1) + tremor + sniff_wave` |
| `acc[1]` | `9.78 + 0.04·N(0,1) + 0.5·sniff_wave` |
| `acc[2]` | `0.31·N(0,1) + 0.15·sin(2π·2.2·t)·sniff_active` — 2.2 Hz |
| `gyr[k]` | `0.01·N(0,1) + 0.06·sniff_wave·sniff_active` |
| `total_load` | `900` if sniffing else `120` |
| `fsr[i]` | `int(total_load · load_share[i])` |
| `us_bottom` | `38.0 + 4.0·N(0,1) − 12.0` if sniffing, clipped `[5, 200]` |
| `us_top` | `95.0 + 6.0·N(0,1) − 45.0` if sniffing, clipped `[5, 200]` |
| `ir_top` | `1` if sniffing and `rand > 0.5` else `0` (per sensor) |
| `ir_bottom` | `rand < 0.7` if sniffing else `rand < 0.15` |
| `imu_temp_c` | `37.2 + 0.2·N(0,1)` |
| `hr_bpm`, `spo2_pct` | **not set** ⇒ default `-1.0` |

**Timing** uses absolute-deadline pacing (`next_ts += period`; if behind schedule,
`next_ts = now`, so no catch-up burst). `self._t` is *virtual* sim time, so
`ts_ms` is deterministic regardless of wall-clock jitter — which is what makes
the simulator reproducible.

The simulator's noise σ (0.02 / 0.04 / 0.31) is **~2.5× smaller** than
`kennel_synth.py`'s (0.05), and its `ir_top` probability is 0.5 vs 0.8. So the
live demo and the lab's synthetic dataset are not drawn from the same
distribution — worth knowing when comparing a live demo against lab numbers.

---

## 2. Queue plumbing (`_BaseSource`)

```python
self.queue = queue.Queue(maxsize=4000)              # :61-62
threading.Thread(target=self._run, daemon=True)      # producer
def stop(self): self._stop_event.set(); self._thread.join(timeout=3.0)   # :73-76
def get_frame(self, timeout): 
    try: return self.queue.get(timeout=timeout)
    except queue.Empty: return None                   # :78-82
def _enqueue(self, frame):
    if self.queue.full():
        self.queue.get_nowait()        # DROP THE OLDEST
    self.queue.put_nowait(frame)       # :85-91
```

**Oldest-drop on overflow** is the right policy for a live dashboard: fresh data
matters more than complete data. 4,000 frames at 10 Hz = 400 s of buffer, so
overflow only occurs if the consumer stalls for >6.5 minutes.

---

## 3. Windowed feature extraction — the 45-feature contract

`kennel_features.py`. **This is the single most important shared contract in the
platform**: the same function produces features for training (from a recorded
CSV), for serving (from a live TCP stream), and for the synthetic lab dataset.

```python
FSR_NAMES = ["fsr_fl", "fsr_fr", "fsr_rl", "fsr_rr"]
IR_NAMES  = ["ir_bottom_fl", "ir_bottom_fr", "ir_bottom_rl", "ir_bottom_rr",
             "ir_top_left",  "ir_top_right"]
N_KENNEL_FEATURES = 45
```

Minimum viable window: `len(frames) < 4` ⇒ returns an all-NaN vector of length 45.

### 3.1 The 45 features

| Idx | Name | Formula | Units |
|---|---|---|---|
| 0–3 | `fsr_{fl,fr,rl,rr}_mean` | `fsr.mean(axis=0)` | ADC counts |
| 4–7 | `fsr_{fl,fr,rl,rr}_std` | `fsr.std(axis=0)` | ADC counts |
| 8 | `total_load_mean` | `mean(Σ_i fsr_i)` | ADC counts |
| 9 | `total_load_std` | `std(Σ_i fsr_i)` | ADC counts |
| 10 | `lr_imbalance_mean` | `mean((fsr_FL + fsr_FR) − (fsr_RL + fsr_RR))` | ADC counts |
| 11 | `fb_imbalance_mean` | `mean((fsr_FL + fsr_RL) − (fsr_FR + fsr_RR))` | ADC counts |
| 12 | `cop_drift_std` | `std(hypot(cop_x, cop_y))` | normalised |
| 13–18 | `ir_{6 sensors}_active_frac` | `ir.mean(axis=0)` | fraction 0–1 |
| 19 | `ir_transitions_total` | `int(Σ_axis ‖diff(ir > 0.5, axis=0)‖)` | count |
| 20 | `us_bottom_mean` | `mean(us_bottom where > 0)` else **−1.0** | cm |
| 21 | `us_top_mean` | `mean(us_top where > 0)` else **−1.0** | cm |
| 22 | `us_bottom_range_std` | `std(us_bottom where > 0)` else **0.0** | cm |
| 23 | `us_top_slope` | `np.polyfit(arange(T), us_top[~isnan], 1)[0]` | cm/sample |
| 24–26 | `acc_{x,y,z}_mean` | `mean` | m/s² |
| 27–29 | `acc_{x,y,z}_std` | `std` | m/s² |
| 30–32 | `acc_{x,y,z}_rms` | `sqrt(mean(a²))` | m/s² |
| 33 | `acc_jerk_rms` | `sqrt(mean((diff(acc)·100.0)²))` | m/s³ |
| 34 | `acc_dom_freq_hz` | `argmax(\|FFT(mag − mean(mag), 512)\|[1:]) + 1` → `rfftfreq(512, d=0.01)` | Hz |
| 35 | `acc_band_tremor_4_8hz` | `Σ\|FFT\|` where 4 ≤ f ≤ 8 | magnitude |
| 36 | `acc_band_sniff_2_5hz` | `Σ\|FFT\|` where 2 ≤ f ≤ 5 | magnitude |
| 37 | `acc_xy_corr` | `np.corrcoef(acc_x, acc_y)[0,1]`, 0.0 if either std ≤ 1e-9 | Pearson r |
| 38–40 | `gyr_{x,y,z}_std` | `std` | rad/s |
| 41 | `gyr_speed_rms` | `sqrt(mean(gyr²))` over all 3 axes | rad/s |
| 42 | `imu_temp_mean` | `mean(imu_temp_c)`, NaN → 0.0 | °C |
| 43 | `hr_bpm_mean` | `mean(hr_bpm where > 0)` else **−1.0** | bpm |
| 44 | `spo2_pct_mean` | `mean(spo2_pct where > 0)` else **−1.0** | % |

### 3.2 Centre of pressure (`:68-71`)

```python
norm  = np.where(total > 0, total, 1.0)
cop_x = (fsr[:,1] + fsr[:,3]) / norm - 0.5      # (FR + RR)/total − 0.5
cop_y = (fsr[:,2] + fsr[:,3]) / norm - 0.5      # (RL + RR)/total − 0.5
cop_drift = float(np.std(np.hypot(cop_x, cop_y)))
```

The dog shifting weight between paws is a behavioural signal, so the *instability*
of the normalised CoP is a feature rather than the CoP position itself.

### 3.3 Filtering rules

- `us_b_clean = us_b[us_b > 0]`, `us_t_clean = us_t[us_t > 0]` — drops the `-1.0`
  no-echo sentinel and 0. `us_top_slope` uses the *unfiltered* series but drops
  NaN inside `slope()`.
- `_safe_float(v, default)` converts `None`/NaN to a default.
- `acc_xy_corr` guards on `std > 1e-9` to avoid a division warning on constant
  axes.
- `hr_bpm` / `spo2_pct` `-1.0` sentinels become NaN at dataset build time
  (`kennel_synth.py:156`) and are median-imputed with **train-set column
  medians** (`train_kennel_model.py:103-104`).

### 3.4 `FEATURE_GROUPS` (`:175-194`)

| Group | Count | Contents |
|---|---|---|
| `pressure` | **13** | 8 FSR means/stds + 5 derived |
| `proximity` | **11** | 6 IR active_frac + `ir_transitions_total` + 4 US |
| `motion` | **18** | 9 acc mean/std/rms + 5 acc derived + 3 gyr std + `gyr_speed_rms` |
| `physiology` | **3** | `imu_temp_mean`, `hr_bpm_mean`, `spo2_pct_mean` |
| **Total** | **45** | exact partition, no overlap |

### 3.5 Trial-level features — 55

```
TRIAL_FEATURE_NAMES = KENNEL_FEATURE_NAMES (45)                       # on the EXPOSURE window
                   + [f"{k}_delta" for k in DELTA_SOURCE_KEYS] (9)
                   + ["post_recovery_delta"]                          # 1
N_TRIAL_FEATURES = 55
```

`DELTA_SOURCE_KEYS` (`:201-212`) — 9 baseline-delta features, Δx = x_exposure −
x_baseline: `total_load_mean`, `us_bottom_mean`, `us_top_mean`, `acc_x_mean`,
`acc_y_mean`, `acc_z_mean`, `acc_jerk_rms`, `gyr_speed_rms`,
`acc_band_sniff_2_5hz`.

`post_recovery_delta` (`:245-254`) — a **ratio**:

```python
exposure_shift = |total_load_mean(expo) − total_load_mean(base)|
post_shift     = |total_load_mean(post) − total_load_mean(base)|
recovery = post_shift / exposure_shift if exposure_shift > 1e-6 else 0.0
```

1.0 = full return to baseline; 0.0 when the post window is absent or the exposure
shift is negligible.

These 10 columns are only computable from a *recorded three-phase trial*, which
is why the live serving path uses the 45-feature set (it has no baseline window)
while the lab's synthetic dataset uses all 55.

### 3.6 Window sizes

| Location | Value | Meaning |
|---|---|---|
| `kennel_streaming.py:30` | `batch_size = 400` | serving window |
| `kennel_features.py:53` | `< 4` ⇒ all-NaN | minimum viable |
| `scripts/train_kennel_model.py:38` | `WINDOW = 400` | "~4 s @ 100 Hz" |
| `scripts/train_kennel_model.py:39` | `STEP = 200` | 50% overlap |
| `kennel_synth.py:86` | `window = int(rate_hz)` = 100 | 1 s stride target |
| `kennel_server.py:175` | `ui_counter % 10 == 0` | UI downsample to 10 Hz |

The 100 Hz assumption behind `WINDOW = 400` is discussed in
[`10_kennel_hardware.md`](10_kennel_hardware.md) §7.1 — **this is the highest
priority fix in the kennel subsystem.**

### 3.7 Tests

`tests/test_hardware_ingest.py::TestFeatureExtraction` (3):
- shape / name order / all-finite
- **determinism** (same frames ⇒ identical features)
- short window ⇒ all-NaN

---

## 4. Server threading model

`api/kennel_server.py:226-237` (lifespan) + `:163-224` (workers).

```python
@asynccontextmanager
async def lifespan(_app):
    nonlocal main_loop
    main_loop = asyncio.get_running_loop()                  # for thread → loop bridging
    app_state["ui_queue"] = asyncio.Queue(maxsize=200)
    source.start()                                          # producer thread
    threading.Thread(target=predictor_worker, daemon=True).start()
    task = asyncio.create_task(raw_stream_broadcaster())
    yield
    task.cancel(); predictor_thread_stop.set(); source.stop()
```

Three concurrent components plus the event loop:

```
   ┌──────────────┐  put    ┌─────────┐  get    ┌────────────────────┐
   │ producer     │ ──────► │ Queue   │ ──────► │ predictor_worker   │
   │ thread       │         │ (4000)  │         │  (single consumer) │
   └──────────────┘         └─────────┘         └─────────┬──────────┘
        ▲                                                  │
        │ TCP or sim                            every 10th │ frame
        │                                                  ▼
        │                              call_soon_threadsafe  asyncio.Queue(200)
        │                                                  │
        │                                                  ▼
        │                                  raw_stream_broadcaster task
        │                                                  │
        └──────────────────  frames ◄─────────────────────┘
                                                           │
                                        every 400 frames ──┤
                                                           ▼
                                                   RF.predict_proba
                                                           │
                                        EMA(0.3) + gate ──┤
                                                           ▼
                                              broadcast_prediction → /ws/diagnostic
```

### 4.1 `predictor_worker` (`:163-214`)

```python
frame = source.get_frame(timeout=0.25)
if frame is None: continue
recorder.add_frame(frame)                                  # 1. Data Lab
app_state["frames_seen"] += 1                              # 2. counter

ui_counter += 1                                            # 3. UI downsample
if main_loop is not None and ui_counter % 10 == 0:
    main_loop.call_soon_threadsafe(app_state["ui_queue"].put_nowait, frame.to_dict())

predictor_buffer.append(frame)                              # 4. windowed inference
if len(predictor_buffer) < settings.batch_size: continue
window = predictor_buffer[:settings.batch_size]
del predictor_buffer[:settings.batch_size]                 # NON-OVERLAPPING
```

Four notes:

1. **The window is non-overlapping.** 400 frames in, 400 consumed, buffer
   cleared. There is no sliding overlap, so at 10 Hz a diagnostic is emitted every
   40 s of wall time, with a hard discontinuity at each boundary. Training uses
   50% overlap (`STEP = 200`) — a **train/serve mismatch in window placement**.
2. **Only 10 Hz reaches the UI**, but `frames_seen` counts every consumed frame.
   The dashboard's "100 Hz Native Sensor Sampling" caption refers to the producer,
   not the wire.
3. **`put_nowait` on a full asyncio queue raises `QueueFull`** inside the
   `call_soon_threadsafe` callback. `raw_stream_broadcaster` never drops or
   bounds, so backpressure is lossy only in the producer→loop direction, and the
   exception is surfaced only by the loop's default handler.
4. **The `timeout=0.25` poll is a busy-wait.** A 250 ms timeout on an empty queue
   means ~4 wakeups/s when idle. Cheap, but not event-driven.

### 4.2 Model loading (`KennelPredictor._load_model`, `:89-136`)

```python
if not path.exists():  status = "untrained"
bundle = joblib.load(path)
model, feature_names, classes, positive_class_label, positive_class_id, metrics, manifest
```

**Positive-class resolution** (`:108-136`) — three-tier, case-insensitive:

1. `model_bundle["positive_class_id"]` if present and in `model.classes_`
2. scan `model_bundle["classes"]` for a label whose lowercase contains `"cancer"`
3. else `None`

Wrapped in a bare `except Exception: return None`. If it returns `None`, the
worker hits `continue` and **no diagnostic is broadcast at all** — `last_prediction`
is not updated, so a browser joining later receives a stale value with no
indication that the pipeline is stalled.

### 4.3 Confidence gate and EMA (`:186-214`)

```python
if predictor.model is None:
    emit({"type": "diagnostic", "status": "untrained",
          "detail": "Collect labelled sessions and run scripts/train_kennel_model.py."})
    continue
proba = predictor._predict_cancer_probability(window)     # None if class missing
if proba is None: continue
ema = proba if app_state["ema"] is None \
      else alpha * proba + (1 - alpha) * app_state["ema"]
status = "ok" if max(ema, 1 - ema) >= settings.confidence_threshold else "uncertain"
app_state["last_prediction"] = {...}
await broadcast_prediction({...})
```

`alpha = 0.3`, threshold `0.6`, both env-configurable. The confidence test is
**two-sided** (`max(p, 1−p) ≥ 0.6`), so `p = 0.1` is "confidently healthy", while
the console's `PredictionCard` colours high/low risk **one-sided** at
`p ≥ 0.5`. Both are defensible; they answer different questions (confidence vs
direction).

---

## 5. WebSocket protocols

### 5.1 `/ws/stream` — raw frames, 10 Hz

Server→client only; the handler calls `manager.connect(...)` then blocks on
`await socket.receive_text()` and never inspects the text. No ping/pong, no
keepalive, no replay.

```jsonc
{
  "ts_ms": 1000, "seq": 7, "state": "SNIFF",
  "fsr": [512.0, 498.0, 602.0, 590.0],      // float[4] FL,FR,RL,RR
  "ir":  [1, 1, 0, 1, 0, 0],                // int[6] 4 bottom + 2 top
  "us":  { "bottom": 41.2, "top": 88.7 },   // cm, -1 = unavailable
  "acc": [-0.12, 9.78, 0.31],              // m/s²
  "gyr":  [0.01, -0.02, 0.0],               // rad/s
  "imu_temp_c": 36.4, "hr_bpm": 92.5, "spo2_pct": 97.2
}
```

**No `"type"` discriminator** on stream frames — they are bare frame objects.
The browser's `Frame` type (`components/KennelDiagram.tsx:12-21`) is a strict
subset: `imu_temp_c`, `hr_bpm`, and `spo2_pct` are received and discarded. A
`"type": "frame"` field would make the two channels self-describing.

### 5.2 `/ws/diagnostic` — windowed results, ~0.25 Hz at 10 Hz

**Emitted on connect: the cached `last_prediction` is replayed immediately**
(`:347-348`). This is how a freshly-loaded page gets a populated risk history
instead of an empty chart.

Three payload variants, all carrying `"type": "diagnostic"`:

**(a) untrained** — no probability field at all:
```json
{"type": "diagnostic", "status": "untrained",
 "detail": "Collect labelled sessions and run scripts/train_kennel_model.py."}
```

**(b) ok / uncertain**:
```json
{"type": "diagnostic", "status": "ok",
 "probability_cancer": 0.7412, "smoothed_probability": 0.6988,
 "confidence_threshold": 0.6}
```

**(c) never sent** — when a model exists but `_predict_cancer_probability`
returns `None`. No diagnostic, no state update.

---

## 6. REST API — `qmlkit.api.kennel_server:app` (port 8000)

| # | Method | Path | Request | Response |
|---|---|---|---|---|
| 1 | GET | `/api/v1/kennel/state` | — | `{stream_source, connected, frames_seen, model_status, recorder{…}, buffer_frames, batch_size}` |
| 2 | POST | `/api/v1/kennel/start` | `StartRecordingRequest` | `{status:"started", recording, samples, elapsed_s, target_s, dog_id, sample_id, label}` · **409** on `RecordingError` |
| 3 | POST | `/api/v1/kennel/stop` | — | `{status:"saved", path}` · **409** if idle |
| 4 | POST | `/api/v1/train` | `TrainRequest` | training result dict · **500** on exception |
| 5 | POST | `/api/v1/benchmark/run` | raw `Body(...)` dict | `{target_cancer, n_qubits, leaderboard[]}` · **500** on exception |
| 6 | GET | `/api/v1/dataset/stats` | — | `{n_sensors, compounds, cancer_types, features_per_sensor}` |
| 7 | POST | `/api/v1/kennel/frame` | raw `Body(...)` dict | `{queued: true}` — **test/utility hook**, pushes a pre-parsed frame straight into `source._enqueue()` |
| 8 | POST | `/api/v1/lab/runs` | `LabRunRequest` | `{job_id, status:"running"}` |
| 9 | GET | `/api/v1/lab/runs` | — | `{jobs: [...]}` newest-first, `progress[-5:]` |
| 10 | GET | `/api/v1/lab/runs/{job_id}` | — | job dict · **404** `"Unknown lab run."` |
| 11 | WS | `/ws/stream` | — | 10 Hz frames |
| 12 | WS | `/ws/diagnostic` | — | ~0.25 Hz results + replay on connect |
| 13 | GET | `/` | — | static console (if `QMLKIT_FRONTEND_DIST` set) |

### 6.1 Request schemas

```python
class StartRecordingRequest(BaseModel):        # :62-66
    dog_id:     str     = Field(min_length=1, examples=["Rex"])
    sample_id:  str     = Field(default="S0", min_length=1)
    label:      str     = Field(min_length=1, description="cancer / control / any custom")
    duration_s: float   = Field(default=20.0, gt=1.0, le=300.0)

class LabRunRequest(BaseModel):                # :69-76
    dataset:      str  = "voc_real"    # voc_real | kennel_synth
    experiment:   str  = "search"      # search | map_ablation | modality_ablation | robustness
    max_samples:  int  = 120   (ge=0)
    vqc_epochs:   int  = 8     (ge=1, le=40)
    n_splits:     int  = 5     (ge=2, le=10)
    n_components: int  = 6     (ge=2, le=12)
    seed:         int  = 42
```

### 6.2 CORS

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

Present **solely** so the static-export Next.js app on a different dev port can
call the API. No `allow_credentials` (so no cookies). Wide open — acceptable for
a localhost research tool, **not** for a deployment. `api/server.py` installs no
CORS at all.

### 6.3 Module-import side effect

```python
app = create_kennel_app()          # :485 — at MODULE IMPORT
app = register_lab_routes(app)
```

`create_kennel_app` builds `settings` and `source` at **factory** time
(`:139, :144`), and `create_stream_source()` performs a **blocking 2.0 s TCP
probe**. So `import qmlkit.api.kennel_server` performs a network probe. The test
suite avoids it by injecting
`KennelSettings(source="simulation", …)` (`tests/test_hardware_ingest.py:159-160`).
Moving the source construction into `lifespan` would remove the import-time I/O.

### 6.4 Port discrepancy

The module docstring says `--port 8001` (`:9`); every real launcher uses **8000**
(`package.json:14`, `qmlkit_desktop.py:57`), which is also the console's default
API base. The docstring is stale.

### 6.5 Static dashboard mount (`:373-379`)

```python
dist_dir = os.environ.get("QMLKIT_FRONTEND_DIST", "")
if dist_dir and Path(dist_dir).is_dir():
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="dashboard")
```

Placed **after** all route decorators, so `app.mount("/")` has the lowest routing
precedence — API routes and WebSockets still resolve. Set by `qmlkit_desktop.py:68`.

---

## 7. Session recording

`hardware/session_recording.py`.

### 7.1 Lifecycle

| Method | Behaviour |
|---|---|
| `start(dog_id, sample_id, label, duration_s=20.0)` | `RecordingError` if already recording |
| `add_frame(frame)` | no-op when idle; appends `frame.to_dict()` + `row["wall_ts"] = time.time()` |
| `progress()` | `min(1.0, elapsed / max(1e-6, duration_s))` |
| `should_finish()` | `elapsed >= duration_s` |
| `stop_and_save()` | `RecordingError` if idle; `RecordingError("No frames captured; nothing to save.")` if 0 rows; **clears meta/rows *before* the existence check** |
| `status()` | `{recording, samples, elapsed_s, target_s, dog_id, sample_id, label}`; `elapsed_s` rounded to 1 dp |

### 7.2 Output path and trial auto-increment

```
<data_dir>/<label>/<dog_id>_<label><trial>.csv       # :92
```

Trial starts at 1 and increments while the path exists (`:90-95`).

**No underscore before the trial number** — `Rex_cancer1.csv`, not
`Rex_cancer_1.csv`. `docs/upgrade/03_server_ingestion.md:46` documents the latter;
see [`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md).

### 7.3 CSV — 25 columns

```
ts_ms, seq, state,
fsr_0, fsr_1, fsr_2, fsr_3,
ir_0, ir_1, ir_2, ir_3, ir_4, ir_5,
us_bottom, us_top,
acc_x, acc_y, acc_z,
gyr_x, gyr_y, gyr_z,
imu_temp_c, hr_bpm, spo2_pct, wall_ts
```

Opened with `newline=""`, `encoding="utf-8"`. FSR → float, IR → int, acc/gyr →
float. The 25-column header is what `load_csv_rows()` (`:122-143`) reads back;
`hr_bpm`/`spo2_pct` use `float(rec.get(...) or -1.0)` so `0.0` and `""` both
collapse to `-1.0`, and `ts_ms`/`seq` go through `int(float(...))` to tolerate
float-formatted CSV.

### 7.4 Metadata loss

`self._meta = {"dog_id", "sample_id", "label"}` is captured at start
(`:51-55`) but **only the label reaches disk** (via the parent directory name).
`sample_id` is discarded. The `RecordingError` on `stop_and_save` clears
meta/rows before the empty-file check (`:77-120`), so a failed save loses the
session metadata as well.

---

## 8. Kennel model training CLI

`scripts/train_kennel_model.py` (168 lines) — the only producer of
`models/kennel_model.joblib`. **Not reachable through the API.**

| Constant | Value | Line |
|---|---|---|
| `WINDOW` | 400 | `:38` |
| `STEP` | 200 (50% overlap) | `:39` |
| RF `n_estimators` | 300 (default), `random_state=42`, `n_jobs=-1` | `:117, :137` |

### 8.1 Pipeline

```
glob("data/kennel/*/*.csv")
  → filename regex ^(?P<dog>.+)_(?P<label>.+?)(?P<trial>\d+)\.csv$
  → label = path.parent.name      (the directory, NOT the regex capture)
  → sliding windows (400 / step 200) → 45 features each
  → np.nanmedian column imputation
  → encode_labels (sorted unique → 0..n-1)
  → cancer_class_id = index of the label whose lower() == "cancer"
      (SystemExit if absent: "Training data must contain a 'cancer' class.")
  → LOSO validation (§8.2)
  → refit on ALL windows
  → dump joblib + JSON report
```

`sys.path.insert(0, .../src)` (`:24`) lets the script run without installing the
package; `sys.stdout.reconfigure(encoding="utf-8")` (`:21-22`) is a Windows
console guard.

**Label encoding is alphabetical**: `encode_labels` (`:42-47`) sorts unique labels
and maps to `0..n-1`. With `{calm, cancer}` that gives `calm→0, cancer→1`; with
`{alert, benign, cancer, control}` the mapping changes. The `positive_class_id`
is resolved separately by name match, so serving is correct — but the *class
ordering* in `model.classes_` is not the semantically obvious one. The
`quantum_augmented_xgb` OOF head and the QSVM probability column both assume
`classes_[1]` is positive, which is **not** guaranteed for the kennel RF (serving
uses the explicit `positive_class_id` lookup, so it is fine).

### 8.2 Leave-one-dog-out validation (`:109-121`)

```python
for dog in unique_dogs:
    test_idx = rows belonging to dog
    train_idx = all others
    if train_idx empty or test_idx empty: continue
    fold_model = RandomForestClassifier(n_estimators, random_state=42, n_jobs=-1)
    fold_model.fit(X[train_idx], y[train_idx])
    fold_acc.append(accuracy_score(y[test_idx], fold_model.predict(X[test_idx])))
    fold_f1.append(f1_score(..., average="macro", zero_division=0))
```

**LOSO is the right protocol here** — a dog's posture and physiology are
individual traits, so a random split would leak subject identity and inflate
scores. Its absence would be a serious methodological flaw; it is present.

Limitations: no probability calibration, no confidence intervals, no nested CV,
and the served artifact is refit on **all** windows rather than being the
LOSO-scored model (standard practice, but worth stating).

### 8.3 Artifacts

`models/kennel_model.joblib`:

```python
{"model": RandomForestClassifier,
 "feature_names": KENNEL_FEATURE_NAMES,          # 45 names
 "classes": {"0": "calm", "1": "cancer"},
 "positive_class_label": "cancer",
 "positive_class_id": 1,
 "metrics": <full report>,
 "manifest": {"version": "0.1.0", "window": 400, "step": 200}}
```

`models/kennel_training_report.json`: `windows, features (first 5 + "..." + last 3),
n_features, class_names, class_distribution, dogs, loso_accuracy_mean,
loso_macro_f1_mean, folds, window, step`.

### 8.4 Current state

**`models/` does not exist**, and `data/kennel/` has one dog across two labels
with no `control/`. So `cancer_class_id` resolves, but LOSO has no usable fold
(single dog ⇒ `train_idx` empty ⇒ skipped ⇒ `loso_accuracy_mean: None`) and
`model_status` is permanently `"untrained"`. The **minimum data to produce a
first artifact** is 2+ dogs with recordings in at least two label directories,
including one named `cancer`.

---

## 9. Test coverage

`tests/test_hardware_ingest.py` — **14 tests**:

| Group | Tests | Coverage |
|---|---|---|
| `TestProtocol` | 5 | valid roundtrip, missing key → `None`, bad state → `None`, malformed JSON → `None`, `frame_to_json` schema |
| `TestSimulationSource` | 1 | frames produced, states ⊆ {IDLE, SNIFF, COOLDOWN} |
| `TestFeatureExtraction` | 3 | shape/order/finiteness, determinism, short window → all-NaN |
| `TestRecordingManager` | 3 | session roundtrip + trial auto-increment, double-start raises, stop-without-start raises |
| `TestKennelServer` | 2 | `TestClient` over `/api/v1/kennel/state`, `/start`, `/stop`, WS `/ws/stream`, and the frame-injection hook |

The canonical `VALID_FRAME` fixture includes `hr_bpm` and `spo2_pct`.

**Not covered:** `TcpKennelSource` (reconnect, partial-line buffering, OSError
path) — the tests use `SimulatedKennelSource`; `KennelPredictor` model loading and
the confidence gate; the EMA state machine; `batch_size`-boundary behaviour.
