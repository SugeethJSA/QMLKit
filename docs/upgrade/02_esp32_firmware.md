# Workstream B — ESP32-WROOM Kennel Node Firmware

**Location:** `firmware/kennel_node/`
**Board:** ESP32-WROOM-32 DevKit (Arduino framework; PlatformIO-compatible)

```
firmware/kennel_node/
├── kennel_node.ino        # Single-file Arduino sketch (IDE-compatible)
├── platformio.ini         # PlatformIO build (pio run -e kennel)
├── README.md              # Wiring tables, flashing, protocol reference
└── include/
    └── config.h           # Pin map + tunables shared by both build paths
```

## B1. Pin map (ESP32-WROOM-32 DevKit v1)

| Peripheral | Signal | GPIO | Notes |
|---|---|---|---|
| FSR_FL / FR / RL / RR | analog | 34 / 35 / 36 / 39 | ADC1 input-only pins; 10 kΩ divider to GND |
| IR_BOTTOM_FL/FR/RL/RR | digital in | 19 / 18 / 17 / 16 | Active-LOW typical IR modules |
| IR_TOP_LEFT / RIGHT | digital in | 5 / 23 | Top front corners |
| US_BOTTOM | TRIG/ECHO | 26 / 27 | HC-SR04, 5 V via level shift or 3.3 V-tolerant board |
| US_TOP | TRIG/ECHO | 14 / 13 | |
| MPU6050 (collar node) | SDA/SCL | 21 / 22 | I²C @ 400 kHz, Adafruit MPU6050 lib |
| Status LED | out | 2 | Onboard LED: blink patterns per state |

Avoided strapping pins (0/2-output-only, 12, 15) for peripheral signals.

## B2. Acquisition design

- **IMU (collar):** ~100 Hz into a ring buffer (accel xyz m/s², gyro xyz rad/s,
  die temp). Window tagging on SNIFF events.
- **Slow channels:** FSR/IR/ultrasonic sampled ~20 Hz.
- **Presence gating state machine:**

```
BOOT → CALIBRATE(2 s FSR baseline) → IDLE
IDLE ──(FSR total load > threshold AND bottom-US < body threshold)──► OCCUPIED
OCCUPIED ──(IR head detection at top OR sustained load stability)──► SNIFF_WINDOW(8 s)
SNIFF_WINDOW → COOLDOWN(10 s) → IDLE
Any state ──(load lost > 3 s)──► IDLE
```

## B3. Telemetry protocol (repomono GAIT contract)

TCP **server** on port `3333`; newline-delimited JSON frames batched every
~100 ms. Python server connects as client — exactly like the GAIT reference.

```json
{"ts_ms":123456,"seq":4127,"state":"SNIFF","fsr":[512,498,602,590],
 "ir":[1,1,0,1,0,0],"us":{"bottom":41.2,"top":88.7},
 "acc":[-0.12,9.78,0.31],"gyr":[0.01,-0.02,0.00],"imu_temp_c":36.4}
```

- `ts_ms` = monotonic `millis()` (drift-free alignment; wall clock optional NTP).
- `seq` monotonically increasing for gap detection.
- Serial (115200) mirrors frames for debugging.
- Also accepts single-char commands: `s` = status JSON, `z` = re-zero FSR baseline.

## B4. Robustness

- WiFi creds stored in NVS (`Preferences`), fallback compile-time defaults;
  auto-reconnect with backoff; mDNS `kennel.local`.
- Hardware watchdog; LED blink codes (BOOT=fast, CALIBRATE=solid, IDLE=slow
  heartbeat, OCCUPIED=double-blink, ERROR=SOS pattern).
- Client-safe: multiple TCP clients supported (server broadcasts latest frame).

## B5. Validation

No local PlatformIO/arduino-cli toolchain → static review + wiring README +
protocol documented; runtime bring-up happens on hardware by the user.
