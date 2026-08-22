# Kennel Node Firmware (ESP32-WROOM-32)

Telemetry node for the QMLKit canine olfactory examination kennel. Samples
posture/proximity sensors plus a collar-mounted IMU and streams newline-
delimited JSON over TCP port **3333** — the same contract used by the GAIT
monorepo reference, so the Python server connects *in* as a client.

## Sensor manifest

| Sensor | Count | Placement | ESP32 pins |
|---|---|---|---|
| FSR (force-sensitive resistor) | 4 | Bottom kennel corners: Front-Left, Front-Right, Rear-Left, Rear-Right | GPIO 34, 35, 36, 39 (ADC1) |
| IR proximity (active-LOW modules) | 4 | Lower corners, paired with FSRs | GPIO 19, 18, 17, 16 |
| IR proximity | 2 | Top front Left / Right (head detection) | GPIO 5, 23 |
| HC-SR04 ultrasonic | 1 | Bottom (body presence) | TRIG 26 / ECHO 27 |
| HC-SR04 ultrasonic | 1 | Top (head height) | TRIG 14 / ECHO 13 |
| MPU6050 (collar/harness) | 1 | Body-worn micro-movement capture | SDA 21 / SCL 22 (I²C @400 kHz) |
| Status LED | 1 | Onboard | GPIO 2 |

## Wiring notes

- **FSR:** wire each FSR between 3.3 V and the ADC pin, with a **10 kΩ** resistor
  from the ADC pin to GND (voltage divider). Pins 34–39 are input-only and safe
  with Wi-Fi active.
- **IR modules:** OUT → GPIO; powered at 3.3 V or 5 V per module spec. Inputs use
  internal pull-ups; detection = pin LOW.
- **HC-SR04:** if powering at 5 V, divide ECHO by 2 resistors
  (e.g. 10 kΩ / 20 kΩ) to protect the 3.3 V input. Range timeout ≈ 4 m.
- **MPU6050 (collar):** connect VCC 3.3 V, GND, SDA→21, SCL→22. Keep the I²C run
  short or use a remote collar node later; bandwidth set to 44 Hz anti-aliasing.

## State machine

```
BOOT → CALIBRATE(2 s FSR zeroing) → IDLE ⇄ OCCUPIED → SNIFF(8 s window) → COOLDOWN(10 s)
```

- **OCCUPIED:** summed FSR delta > 400 counts AND bottom-US < 60 cm.
- **SNIFF trigger:** OCCUPIED + head detected (top IR OR top-US < 90 cm)
  sustained 500 ms. Frames tagged `"state":"SNIFF"` during the 8 s window.
- Thresholds are `const` tunables at the top of `kennel_node.ino`.

## Telemetry protocol

TCP **server** on port 3333 (up to 4 simultaneous clients), frames every 100 ms:

```json
{"ts_ms":123456,"seq":4127,"state":"SNIFF","fsr":[512,498,602,590],
 "ir":[1,1,0,1,0,0],"us":{"bottom":41.2,"top":88.7},
 "acc":[-0.12,9.78,0.31],"gyr":[0.01,-0.02,0.00],"imu_temp_c":36.4}
```

- `ts_ms` monotonic `millis()` (drift-free alignment), `seq` gap detection.
- Ultrasonic `-1.0` = no echo.
- Single-char commands from any TCP client or Serial: `z` re-zero FSRs,
  `s` immediate status frame.

## Networking

- STA mode using credentials stored in NVS (keys `ssid`/`pass` in namespace
  `kennel`) — provision once via a sketch edit or `Preferences` tooling.
- Falls back to SoftAP `QMLKit-Kennel` / `sniff1234` when unconfigured or on
  association failure. Server runs identically in both modes.
- Advertises mDNS `kennel.local` (service `_qmlkit-kennel._tcp`).
- Loop watchdog enabled; LED blink codes: fast=BOOT, solid=CALIBRATE,
  heartbeat=IDLE, double=OCCUPIED, rapid=SNIFF, slow-alternate=COOLDOWN.

## Flashing

### Arduino IDE
1. Install boards package **esp32 by Espressif Systems**.
2. Libraries: **Adafruit MPU6050** (+ dependencies Adafruit Unified Sensor,
   Adafruit BusIO).
3. Board: "ESP32 Dev Module", upload `kennel_node.ino`, Serial Monitor 115200.

### PlatformIO
```bash
cd firmware/kennel_node
pio run -e kennel -t upload && pio device monitor
```

> Note: this repo has no CI for firmware; bring-up happens on hardware.
