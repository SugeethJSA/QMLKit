# Workstream D — Next.js Live Dashboard

**Location:** `frontend/` (pnpm workspace member, name: `frontend`)
**Stack (repomono parity):** Next.js 16 + React 19 + TypeScript 5.9 +
TailwindCSS v4 (`@tailwindcss/postcss`) + lucide-react icons. No chart lib
dependency — lightweight inline SVG/canvas components to keep the bundle small
and static-export safe.

```
frontend/
├── package.json / tsconfig.json / next.config.ts / postcss.config.mjs
├── eslint.config.mjs
├── app/
│   ├── layout.tsx            # Dark clinical theme shell
│   ├── page.tsx              # Live kennel dashboard
│   ├── diagnostics/page.tsx  # Prediction card + attributions
│   └── collect/page.tsx      # Data Lab (guided recording)
├── components/
│   ├── KennelDiagram.tsx     # Top-view SVG: corners w/ FSR load gauges + IR occupancy
│   ├── ImuWaveforms.tsx      # Rolling accel/gyro traces (canvas)
│   ├── SensorBars.tsx        # Ultrasonic distances + slow channels
│   ├── PredictionCard.tsx    # Class + confidence ring + "uncertain" state
│   ├── PathwayAttribution.tsx# Feature-group attribution bars
│   └── ConnectionBadge.tsx   # hardware / simulation / disconnected
└── lib/
    ├── api.ts                # Base URL auto-detect (?api= override), REST helpers
    └── ws.ts                 # Reconnecting WebSocket client hook
```

## Pages

| Route | Content |
|---|---|
| `/` | Connection badge, live KennelDiagram (FSR corner loads, IR states), ultrasonic bars, collar IMU waveforms, session status strip |
| `/diagnostics` | Latest window prediction: class, confidence ring, probability bars, feature-group attribution, history sparkline |
| `/collect` | Form (dog_id, sample_id, label, duration) → start/stop → progress bar; lists recent sessions from `/api/v1/kennel/state` |

## Behaviour

- `/ws/stream` at ~10 Hz for UI; `/ws/diagnostic` event-driven.
- Auto-reconnect with backoff; API host auto-detects
  (`window.location.hostname`:8000) with `?api=http://host:port` override.
- Graceful "untrained model" banner when diagnostics return
  `{status:"untrained"}`.
- Static-export capable (`output:'export'`) so PyInstaller can embed `out/`.

## Verification

- `pnpm --filter frontend lint`
- `pnpm --filter frontend build`
