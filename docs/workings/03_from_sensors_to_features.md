# From Sensors to Features

Models can't drink a firehose. Ten seconds of kennel telemetry at 100 readings
per second is 1,000 frames × ~20 numbers = **20,000 raw values** for one
viewing session. We need to compress that into one tidy row of summary numbers
— **features** — without losing the interesting bits.

## Step 1: Windows

We cut the stream into fixed chunks called windows (about 4 seconds each). For
every window we compute statistics per sensor family.

## Step 2: The window feature vector (~50 numbers)

| Family | Example features | Why it matters |
|---|---|---|
| **Pressure** (FSR) | mean & wiggle of each corner pad, total load, left-vs-right and front-vs-back tilt, how far the "centre of weight" wanders | Posture tells engagement: dogs lean *into* interesting smells |
| **Proximity** (IR + ultrasonic) | which zones fired and how often they flipped, average/min distance to body and head, whether distance is shrinking over time | Approach vs withdrawal behaviour |
| **Motion** (collar IMU) | mean/spread/strength of acceleration on each axis, jerkiness (change of acceleration), the dominant wobble frequency, energy in the 2–5 Hz "sniffing band" and the 4–8 Hz "tremor band", gyro stability | Sniffing has a rhythm; agitation has another |
| **Physiology** | temperature, heart rate, SpO₂ averages | Arousal signature |

One detail worth knowing: missing values (e.g., the pulse sensor couldn't get a
reading) are marked and later filled with the *median from the training data* —
never with values computed from the test data, which would be cheating.

## Step 3: Trial features with before/after comparison

A full screening visit (**trial**) has three acts: **baseline** (dog in kennel,
sample not yet presented), **exposure** (sniffing), and **post** (afterwards).
The paper's key trick: don't just record absolute values — record the
**change**:

> Δheart-rate = exposure heart rate − baseline heart rate

The same for load shift, distances, movement energy, sniffing-band power, and
so on. This cancels out each dog's personal baseline (a sleepy beagle is
different from a jittery terrier) and keeps the *reaction*.

Final trial row = exposure-window features (≈50) + Δ-features (9) + recovery
score (1) ≈ **55–60 numbers describing everything the dog did.**

## Where this happens in code

- `src/qmlkit/hardware/kennel_features.py` — the single source of truth. The
  exact same functions run during training (on recorded sessions) and live
  serving (on streaming data). Using two different code paths would silently
  skew results — a classic ML bug we deliberately designed away.
- `src/qmlkit/lab/kennel_synth.py` builds fake trials with the same pipeline so
  we can test everything before real sessions exist.
