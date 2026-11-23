# 10 — Kennel Hardware and Firmware

`firmware/kennel_node/kennel_node.ino` (1,356 lines) + `platformio.ini` +
`src/qmlkit/hardware/protocol.py`.

---

## 1. Bill of materials and pin assignment

Board: **ESP32-WROOM-32 DevKit**, Arduino framework, serial 115200 baud.
Build: `platform = espressif32`, `board = esp32dev`, `framework = arduino`.

### 1.1 Pins (`kennel_node.ino:41-62`)

| Signal | Pin(s) | Mode | Placement |
|---|---|---|---|
| `PIN_FSR[4]` | **34, 35, 36, 39** | `INPUT` (ADC1) | load cells, bottom corners FL / FR / RL / RR |
| `PIN_IR_BOTTOM[4]` | **19, 18, 17, 16** | `INPUT_PULLUP` (active **LOW**) | lower corners, one per FSR |
| `PIN_IR_TOP[2]` | **5, 23** | `INPUT_PULLUP` (active **LOW**) | top-front left / right (head detect) |
| `PIN_US_BOT_TRIG` | **26** | `OUTPUT` | HC-SR04 bottom TRIG |
| `PIN_US_BOT_ECHO` | **27** | `INPUT` | HC-SR04 bottom ECHO |
| `PIN_US_TOP_TRIG` | **14** | `OUTPUT` | HC-SR04 top TRIG |
| `PIN_US_TOP_ECHO` | **13** | `INPUT` | HC-SR04 top ECHO |
| `PIN_I2C_SDA` | **21** | I²C SDA | MPU6050 + MAX30102 |
| `PIN_I2C_SCL` | **22** | I²C SCL @ 400 kHz (`:1031`) | same |
| `PIN_LED` | **2** | `OUTPUT` | onboard status LED |

**13 physical sensors:** 4× FSR, 6× IR proximity, 2× HC-SR04, 1× MPU6050
(6-axis), 1× MAX30102 (pulse oximeter).

GPIOs 34–39 are **input-only and Wi-Fi-safe** — the correct choice for analog
sensing on the ESP32, since GPIOs 6–11 would conflict with the flash.

### 1.2 Wiring notes (`firmware/kennel_node/README.md:20-30`)

- **FSR:** 3.3 V → FSR → ADC pin, with a **10 kΩ** resistor from the ADC pin to
  GND (voltage divider).
- **IR:** OUT → GPIO, powered per module spec; internal pull-ups; detection is
  **pin LOW**.
- **HC-SR04:** if powered at 5 V, divide ECHO (e.g. 10 kΩ / 20 kΩ) for 3.3 V
  input protection. Range timeout ≈ 4 m.
- **MPU6050:** 3.3 V, GND, SDA→21, SCL→22, 44 Hz DLPF.

---

## 2. Sampling rates

| Channel | Constant | Period | Rate |
|---|---|---|---|
| MPU6050 accel + gyro + temp | `IMU_INTERVAL_MS = 10` | 10 ms | **100 Hz** |
| FSR (4) + IR (6) + HC-SR04 (2) | `SLOW_INTERVAL_MS = 50` | 50 ms | **20 Hz** |
| MAX30102 HR / SpO₂ | `HR_INTERVAL_MS = 500` | 500 ms | **2 Hz** |
| JSON telemetry frame | `TX_INTERVAL_MS = 100` | 100 ms | **10 Hz** |
| Serial status mirror | hardcoded `2000` | 2000 ms | 0.5 Hz |
| MAX30102 internal | `setup(..., 100, ...)` | — | 100 SPS |
| I²C bus | `Wire.setClock(400000)` | — | 400 kHz |

**Critical design note.** The IMU is *sampled* at 100 Hz but only the **latest**
sample is transmitted, so the wire carries a 10 Hz decimated stream. Any
frequency-domain feature computed downstream must account for this. See
[`16_limitations_and_risk_register.md`](16_limitations_and_risk_register.md) §3.

### 2.1 PlatformIO dependencies (`platformio.ini`)

```ini
lib_deps =
    adafruit/Adafruit MPU6050@^2.2.6
    adafruit/Adafruit Unified Sensor@^1.1.14
    sparkfun/SparkFun MAX3010x Pulse and Proximity Sensor Library@^1.1.2
build_flags = -DCORE_DEBUG_LEVEL=1
```

Single environment `[env:kennel]`, `src_dir = .` (the `.ino` sits at the firmware
folder root).

---

## 3. State machine

```c
enum KennelState { ST_BOOT, ST_CALIBRATE, ST_IDLE, ST_OCCUPIED, ST_SNIFF, ST_COOLDOWN };
const char* STATE_NAMES[] = { "BOOT","CALIBRATE","IDLE","OCCUPIED","SNIFF","COOLDOWN" };
```

```
                 bodyDetected()
      ┌──────┐  ─────────────►  ┌───────────┐  head held ≥500ms  ┌───────┐
      │ IDLE │                   │ OCCUPIED  │  ────────────────► │ SNIFF │
      └──────┘  ◄─────────────   └───────────┘                    └───────┘
          ▲      load lost >3s        │  load lost >3s                 │ 8 s elapsed
          │                          ▼                                 │ (or body lost
          │                      ┌──────┐                              │  >3 s)
          │                      │ IDLE │  ◄───────────────────────────┤
          │                      └──────┘        10 s cooldown         ▼
          │                                          ┌───────────┐
          └───────────────────────────────────────── │ COOLDOWN  │
                                                     └───────────┘
```

| Constant | Value | Role |
|---|---|---|
| `SNIFF_WINDOW_MS` | 8000 | how long a sniff bout is recorded (8 s) |
| `COOLDOWN_MS` | 10000 | refractory period between bouts (10 s) |
| `HEAD_HOLD_MS` | 500 | head dwell required to start a sniff (0.5 s) |
| `LOAD_LOST_MS` | 3000 | how long the load can be absent before declaring vacancy (3 s) |
| `OCC_LOAD_DELTA` | 400 | summed ADC-count delta that counts as "occupied" |
| `US_BODY_DIST_CM` | 60.0 | body-detection distance threshold |
| `US_HEAD_DIST_CM` | 90.0 | head-detection distance threshold |
| `CALIBRATION_MS` | 2000 | **declared but never referenced** — dead constant |

### 3.1 Detection predicates

```c
bool bodyDetected() {                      // :416-426
    return g_load_delta > OCC_LOAD_DELTA &&
           (isnan(g_us_bottom_cm) || g_us_bottom_cm < US_BODY_DIST_CM);
}

bool headDetected() {                      // :394-409
    return g_ir_top[0] || g_ir_top[1] ||
           (!isnan(g_us_top_cm) && g_us_top_cm > 0 && g_us_top_cm < US_HEAD_DIST_CM);
}
```

`bodyDetected` is a **sensor fusion**, not a single sensor: load *and* distance
must agree. This prevents a dog standing beside the kennel (load from a passing
tail) or an object on the mat (proximity without load) from triggering a session.

### 3.2 LED blink codes (`:565-610`)

| State | on/off (ms) | Reading |
|---|---|---|
| BOOT | 60 / 60 | fast grey blink |
| CALIBRATE | 500 / 0 | solid amber |
| IDLE | 40 / 960 | occasional green |
| OCCUPIED | 120 / 120 | medium blue |
| SNIFF | 30 / 30 | fast purple |
| COOLDOWN | 500 / 500 | slow grey |

---

## 4. Calibration and filtering

### 4.1 FSR baseline zeroing

`zeroBaselines()` (`:360-387`): averages **N = 32** ADC samples with
`delay(5)` between them ⇒ ~160 ms of averaging, **not** the 2000 ms implied by
`CALIBRATION_MS`. The baseline is stored integer-truncated (`acc[i] / N`) in
`g_fsr_baseline[4]`.

```c
g_load_delta = Σ max(0, g_fsr_raw[i] - g_fsr_baseline[i]);    // :244-258
```

`g_load_delta` is **not** transmitted in the JSON frame — it is used only for
state gating. The host receives the raw ADC counts and cannot recompute it, which
is a deliberate choice (the host does not know the calibration) but means the
consumer must establish its own baseline.

### 4.2 Heart rate: EMA + plausibility gate (`:330-336`)

```c
g_beat_interval_ms = 0.7f * g_beat_interval_ms + 0.3f * (float)interval;  // α = 0.3
g_hr_bpm = 60000.0f / g_beat_interval_ms;
if (interval >= 300 && interval <= 2000) { /* accept: 30–200 bpm */ }
```

### 4.3 SpO₂: empirical calibration (`:338-346`)

```c
red_ac = g_red_max - g_red_min;     // AC envelope, reset per beat
ir_ac  = g_ir_max  - g_ir_min;
r = (red_value / red_ac) / (ir_value / ir_ac);
g_spo2_pct = constrain(110.0f - 25.0f * r, 70.0f, 100.0f);
```

This is a **linear ratio-of-ratios** model, not the standard calibration curve
$r = R + a/R$ inverted with empirical coefficients. The coefficients
(110 / 25) are uncalibrated placeholders; a MAX30102 needs device-specific
calibration against a co-oximeter. Treat the SpO₂ channel as a **relative
oximetry index**, not a clinical SpO₂ reading.

### 4.4 No-probe gate (`:315-322`)

```c
if (getIR() < 50000) { g_hr_bpm = -1.0f; g_spo2_pct = -1.0f; /* reset AC trackers */ }
```

Emits the `-1.0` sentinel for both physiology fields, which the Python side
converts to NaN and median-imputes.

### 4.5 Ultrasonic (`:204-229`)

```c
digitalWrite(TRIG, LOW);  delayMicroseconds(3);
digitalWrite(TRIG, HIGH); delayMicroseconds(10);
digitalWrite(TRIG, LOW);
duration_us = pulseIn(ECHO, HIGH, 25000UL);       // 25 ms ≈ 4.3 m at 58 µs/cm
return duration_us == 0 ? NAN : duration_us / 58.0f;
```

`NAN` is serialised as `-1.0` in the frame. The `58.0` divisor is the standard
HC-SR04 speed-of-sound constant for air at ~20 °C.

### 4.6 MPU6050 configuration (`:1055-1065`)

```c
mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
mpu.setGyroRange(MPU6050_RANGE_500_DEG);
mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
```

8 g / ±500 °/s / 44 Hz DLPF. The **500 °/s** range is much wider than dog
micro-movements need (±50 °/s would give ~10× better resolution) and the 44 Hz
DLPF attenuates the 2–8 Hz sniff/tremor band the feature extractor looks for.
Both are tuning opportunities.

### 4.7 MAX30102 configuration (`:1092-1099`)

```c
g_max30102.setup(60,    // LED brightness
                  4,     // sample averaging
                  2,     // LED mode (SpO2: red + IR)
                  100,   // sample rate (SPS)
                  411,   // pulse width (µs)
                  4096); // ADC range
```

---

## 5. Network layer

| Aspect | Value | Line |
|---|---|---|
| WiFi mode | `WIFI_STA` only | `:832` |
| Hostname | `"kennel"` | `:834-836` |
| Credentials | `secrets.h` → `WIFI_SSID` / `WIFI_PASSWORD` | `:829, :839-840` |
| Connect loop | 20,000 ms max, 500 ms poll | `:846-854` |
| mDNS | `MDNS.begin("kennel")` + `addService("qmlkit-kennel", "tcp", 3333)` | `:1129-1135` |
| TCP role | **server** | `:1168` |
| `TCP_PORT` | **3333** | |
| `setNoDelay` | `true` | `:1168` |
| Client slots | **4** (`WiFiClient g_clients[4]`) | `:131` |
| Over-capacity response | `stop()` with "TCP client rejected: slots full." | `:746-753` |
| Framing | newline-delimited JSON via `c.println(buf)` | `:704` |

The device is discoverable as **`kennel.local`** with mDNS service
`_qmlkit-kennel._tcp`, which is what `QMLKIT_KENNEL_ESP32_IP` should point at.

### 5.1 Serial commands

| Char | Action | Line |
|---|---|---|
| `z` | re-zero the FSR baselines | `:777-803` |
| `s` | send an immediate status frame | `:777-803` |

Accepted from any of the 4 connected TCP clients.

---

## 6. Telemetry protocol

### 6.1 Exact format string (`buildFrame`, `:617-686`)

512-byte stack buffer, built with a single `snprintf` (`:626-633`):

```c
"{\"ts_ms\":%lu,\"seq\":%llu,\"state\":\"%s\","
"\"fsr\":[%u,%u,%u,%u],"
"\"ir\":[%d,%d,%d,%d,%d,%d],"
"\"us\":{\"bottom\":%.1f,\"top\":%.1f},"
"\"acc\":[%.3f,%.3f,%.3f],"
"\"gyr\":[%.3f,%.3f,%.3f],"
"\"imu_temp_c\":%.1f,"
"\"hr_bpm\":%.1f,\"spo2_pct\":%.1f}"
```

### 6.2 Field specification

| Field | Type | Units | Format | Source |
|---|---|---|---|---|
| `ts_ms` | int | milliseconds | `%lu` | `millis()`, monotonic |
| `seq` | int (uint64) | counter | `%llu` | `g_seq++` per frame (`:1338`) |
| `state` | string | enum | — | `STATE_NAMES[g_state]` |
| `fsr` | int[4] | **raw ADC counts** (0–4095) | `%u` | `analogRead(34/35/36/39)` |
| `ir` | int[6] | 0/1 | `%d` | `digitalRead(pin) == LOW`; order = 4 bottom then 2 top |
| `us.bottom` | float | **cm** | `%.1f` | HC-SR04 ÷ 58; `-1.0` on no-echo |
| `us.top` | float | **cm** | `%.1f` | same |
| `acc` | float[3] | **m/s²** | `%.3f` | `mpu.getEvent().acceleration` |
| `gyr` | float[3] | **rad/s** | `%.3f` | `mpu.getEvent().gyro` |
| `imu_temp_c` | float | **°C** | `%.1f` | `sensors_event_t.temperature`; `-1.0` on NaN |
| `hr_bpm` | float | **bpm** | `%.1f` | `60000 / EMA(beat interval)`; `-1.0` unavailable |
| `spo2_pct` | float | **% SpO₂** | `%.1f` | `constrain(110 − 25R, 70, 100)`; `-1.0` unavailable |

### 6.3 Example frame

```json
{"ts_ms":123456,"seq":4127,"state":"SNIFF","fsr":[512,498,602,590],
 "ir":[1,1,0,1,0,0],"us":{"bottom":41.2,"top":88.7},
 "acc":[-0.12,9.78,0.31],"gyr":[0.01,-0.02,0.00],
 "imu_temp_c":36.4,"hr_bpm":72.3,"spo2_pct":97.1}
```

### 6.4 Python-side validation (`hardware/protocol.py`)

```python
FRAME_KEYS = {"ts_ms","seq","state","fsr","ir","us","acc","gyr","imu_temp_c"}
VALID_STATES = {"BOOT","CALIBRATE","IDLE","OCCUPIED","SNIFF","COOLDOWN"}
```

`hr_bpm` / `spo2_pct` are **optional** — defaulted to `-1.0` via `.get()` (`:97-98`)
and deliberately excluded from `FRAME_KEYS` so older firmware still validates.

`KennelFrame` dataclass (`:27-58`) uses a **flat** internal representation
(`us_bottom`, `us_top`) and re-nests to `{"bottom": …, "top": …}` in `to_dict()`.

`parse_frame()` (`:61-101`) returns `None` on any of:

1. `JSONDecodeError` / `UnicodeDecodeError`
2. payload not a `dict`
3. any required key missing
4. **length mismatch**: `len(fsr)==4 and len(ir)==6 and len(acc)==3 and len(gyr)==3`
5. `state` not in `VALID_STATES`
6. any `TypeError` / `ValueError` / `AttributeError`

**All failures are silent** — no exception, no log, no counter. The streaming loop
drops non-`None` results (`kennel_streaming.py:126`), so a systematically
malformed stream is indistinguishable from an idle one except via the
`connected` flag. A malformed-frame counter on `/api/v1/kennel/state` would be a
cheap, high-value observability addition.

---

## 7. Firmware defects and doc drift

| # | Issue | Severity | Detail |
|---|---|---|---|
| F1 | `secrets.h` missing from the repo | **Blocker for build** | gitignored (`.gitignore`) and absent ⇒ the sketch does not compile as-is |
| F2 | README overstates capability | Medium | claims NVS/`Preferences` credential storage (keys `ssid`/`pass`, namespace `kennel`) and a SoftAP fallback `QMLKit-Kennel` / `sniff1234` — **neither exists**; `Preferences.h` is included but never used, and there is no `WiFi.softAP` call. Also claims a loop watchdog that is not enabled |
| F3 | README example JSON omits `hr_bpm` / `spo2_pct` | Low | the firmware now emits them |
| F4 | Firmware header comment omits the MAX30102 | Low | `:6-11` lists only FSR/IR/US/MPU6050 |
| F5 | `CALIBRATION_MS = 2000` is dead code | Low | declared `:75`, never referenced; actual zeroing is ~160 ms |
| F6 | Sampling-rate mismatch with the feature extractor | **High** | see §3 below |
| F7 | MAX30102 configured on a GPIO-2 LED pin | Low | the status LED shares GPIO 2 with the onboard LED; if a dev board wires the LED to a data pin this conflicts |

### 7.1 F6 — the 100 Hz ↔ 10 Hz mismatch (High)

`kennel_features.py` assumes the incoming window is sampled at **100 Hz**:

- docstring `:5` — "sampled at ~100 Hz"
- `acc_jerk_rms` (`:79`) — `np.diff(acc) * 100.0` (a 100 Hz finite-difference scale)
- `acc_dom_freq_hz` (`:85`) — `np.fft.rfftfreq(512, d=0.01)`
- `scripts/train_kennel_model.py:38` — `WINDOW = 400  # ~4 s @ 100 Hz`

But the wire rate is **10 Hz** (`TX_INTERVAL_MS = 100`). Consequences on live
hardware:

| Quantity | Assumed (100 Hz) | Actual (10 Hz) | Effect |
|---|---|---|---|
| Physical span of a 400-frame window | 4 s | **40 s** | window is 10× longer in time |
| `acc_jerk_rms` | m/s³ | **m/s¹ per 0.1 s step** | magnitude ~10× off, and the finite-difference is aliased |
| `acc_dom_freq_hz` axis | 0–50 Hz | 0–5 Hz | the 4–8 Hz tremor band is **unreachable**; `acc_band_tremor_4_8hz` is always 0 |
| `acc_band_sniff_2_5hz` | populated | populated (0.2–0.5 Hz band) | accidentally still non-zero, but measuring a different thing |

Mitigating factor: **training and serving both use `WINDOW = 400` frames**
(`train_kennel_model.py:38`, `kennel_streaming.py:30`), so the model is
self-consistent in *frame count*. The committed `data/kennel/*.csv` files have
10 ms `ts_ms` deltas, i.e. genuine 100 Hz data — so a model trained on them and
served on 10 Hz live data **will** suffer a distribution shift. Three clean fixes,
in order of preference:

1. **Derive the sample interval from the frame timestamps** rather than assuming
   it. `dt = median(diff(ts_ms))/1000` inside `extract_window_features` — makes
   every rate-dependent feature correct for any wire rate, and removes the
   assumption entirely.
2. **Buffer on the host** to reconstruct 100 Hz from the 10 Hz stream — not
   possible, the information is gone.
3. **Raise `TX_INTERVAL_MS` to 10 ms** (100 Hz telemetry) — 4× the bandwidth, and
   4 FSR + 6 IR + 2 US + 1 IMU at 100 Hz may exceed the ESP32's serial budget.

Option 1 is a ~5-line change and is the right fix. It is the single
highest-priority code change in the kennel subsystem.
