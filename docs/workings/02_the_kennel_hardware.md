# The Kennel Hardware

## What's inside the kennel

| Sensor | Count | Where | What it tells us |
|---|---|---|---|
| **FSR** (force-sensitive resistor) | 4 | Under each bottom corner | Weight distribution — is the dog leaning toward the sample? Shifting paws? |
| **IR** (infrared proximity) | 6 | 4 near the corners + 2 at top-front left/right | Is a body part in this zone? The top pair catches head position |
| **Ultrasonic** rangefinders | 2 | One low, one high | Exact distance to the dog's body / head |
| **MPU6050** motion sensor | 1 | On the dog's collar | Tiny accelerations and rotations — tremors, sniffing rhythm |
| **MAX30102** | 1 | Collar | Heart rate and blood-oxygen (SpO₂) — did the smell *startle* or *excite*? |
| Status LED | 1 | Onboard | Blink codes so you know what state the kennel is in |

## What the ESP32 does with them

The ESP32 is a small, cheap computer about the size of a matchbox. Every
**100 milliseconds** it bundles the latest readings into one line of text
(called a **frame**) that looks like this:

```json
{"ts_ms":123456,"seq":4127,"state":"SNIFF","fsr":[512,498,602,590],
 "ir":[1,1,0,1,0,0],"us":{"bottom":41.2,"top":88.7},
 "acc":[-0.12,9.78,0.31],"gyr":[0.01,-0.02,0.00],
 "imu_temp_c":36.4,"hr_bpm":72.3,"spo2_pct":97.1}
```

Reading it like a sentence: *"At time 123456, message #4127, the kennel is in
SNIFF state; corner pressures are 512/498/602/590; body zones 1,2,4 are
occupied and the head zones are empty; the dog's chest is 41 cm from the floor
sensor and its head 89 cm from the top sensor; collar acceleration is mostly
downward gravity (−0.12, 9.78, 0.31) with almost no rotation; temperature
36.4 °C, heart rate 72 bpm, oxygen 97 %."*

The collar motion sensor reports much faster (**100 times per second**) so we
don't miss the tiny vibrations of sniffing.

## The state machine (the kennel's brain)

The firmware watches for a simple story:

```
IDLE ──dog walks in (weight + distance)──► OCCUPIED
OCCUPIED ──head near sample for half a second──► SNIFF (8-second recording window)
SNIFF ──window over──► COOLDOWN (10 s rest) ──► IDLE
```

Every frame carries its `state`, so later we know exactly which seconds belong
to "just standing there" versus "actively investigating." That distinction is
what powers the baseline-vs-exposure comparison in the features.

## How it talks to the laptop

The ESP32 runs a tiny server on your WiFi (or creates its own hotspot named
`QMLKit-Kennel` if it can't join yours). Our Python software connects to port
3333 and receives those JSON lines continuously. If no hardware is plugged in,
the software quietly switches to a **simulator** that invents realistic dog
behaviour — so developers can work on trains.

## Wiring & flashing

Full pin-by-pin tables live in [`firmware/kennel_node/README.md`](../../firmware/kennel_node/README.md).
Short version: FSRs use simple voltage dividers into analog pins; IR modules are
plug-and-play digital; ultrasonic modules need two pins each; both collar
sensors share the two I²C wires.
