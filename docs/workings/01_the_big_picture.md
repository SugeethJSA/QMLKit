# The Big Picture

## What problem are we solving?

Doctors can already detect many cancers — but usually late, and often only with
expensive or invasive tests. Dogs, remarkably, can smell cancer on someone's
breath *early*, because tumours change the chemistry of the body and that
chemistry leaks into exhaled air as tiny amounts of specific gases (called
**volatile organic compounds**, or VOCs).

The catch with dogs: they can't tell us *which* chemicals they smelled, how
confident they are, or whether they're just distracted by lunch. Traditional
canine screening relies on a trained signal — the dog sits or paws when it
smells something — judged by a human.

## Our idea

Keep the dog as the world's best chemical sensor, but surround the sniffing
moment with electronics that record **everything the dog does**, objectively:

- How does its weight shift on the kennel floor?
- Where is its head? How close does it get? How long does it stay?
- Does its heart rate jump? Does its movement pattern change?
- Are there micro-tremors in its collar while it sniffs?

Then let machine learning find out which of those reaction patterns go together
with cancer-positive samples. The dog provides the nose; the math provides the
readout.

## And where does "quantum" come in?

Two places:

1. **A special quantum feature map (BioZZ).** When we hand data to a quantum
   computer simulator, we must first encode numbers into quantum states. We
   invented an encoding where *strongly related measurements* (say, heart rate
   and movement) create *stronger interactions* between their qubits — the
   quantum version of "these clues belong together." It's described in our IEEE
   manuscript.
2. **Hybrid models.** Instead of betting on either classical OR quantum models,
   the Training Lab tries combinations — e.g., let a small quantum circuit
   produce an extra "opinion" feature, then let XGBoost (a strong classical
   model) make the final call. In our experiments so far, these hybrids win.

## The three moving parts you'll meet in this repo

```
[1] KENNEL HARDWARE          [2] SERVER + MODELS            [3] CONSOLE APP
ESP32 chip + sensors    →    FastAPI ingests telemetry →    Next.js dashboard:
streams readings        →    extracts features per trial→     live gauges,
100×/second             →    trains/tests models         →   predictions,
                        →    (classical, quantum, hybrid)→   recording studio,
                                                             training lab
```

Everything is packaged so it runs from one folder on a Windows laptop next to
the kennel.
