# 12 — Frontend Console

`frontend/` — Next.js 16, React 19, Tailwind CSS v4, TypeScript 5.9. Static
export (`output: "export"`), no SSR, no API routes, no state-management library.

---

## 1. Build configuration

### 1.1 `next.config.ts`

```typescript
const nextConfig: NextConfig = { output: "export", images: { unoptimized: true } };
```

Static HTML export. The FastAPI server mounts the resulting `out/` directory at
`/` (`kennel_server.py:373-379`), so one Python process serves the whole console.

### 1.2 Tailwind v4 — CSS-first configuration

`postcss.config.mjs`:

```js
export default { plugins: { "@tailwindcss/postcss": {} } };
```

`app/globals.css:1`:

```css
@import "tailwindcss";
```

**There is no `tailwind.config.js` / `.ts` and no `@config` directive anywhere.**
Tailwind v4's theme customisation is entirely in CSS custom properties, which is
the idiomatic v4 approach. `docs/upgrade/04_nextjs_frontend.md` lists only
`lint`/`build` as verification steps; there is **no `typecheck` script** in
`frontend/package.json`, so type errors are only caught at `next build` time.

### 1.3 `tsconfig.json`

`target ES2017`, `lib [dom, dom.iterable, esnext]`, **`strict: true`**,
`noEmit: true`, `module esnext`, `moduleResolution bundler`,
`jsx react-jsx`, `isolatedModules`, `incremental`, `paths {"@/*": ["./*"]}`,
`include [next-env.d.ts, **/*.ts, **/*.tsx, .next/types/**/*.ts, .next/dev/types/**/*.ts]`.

Note the `@/*` path alias resolves relative to the **frontend root**, not `src/`.

### 1.4 ESLint

`eslint.config.mjs`: flat config from `eslint-config-next/core-web-vitals` +
`eslint-config-next/typescript`; rule override `"@typescript-eslint/no-explicit-any": "off"`;
ignores `node_modules/**`, `.next/**`, `out/**`.

---

## 2. API base URL resolution

`lib/api.ts` derives the base from the current URL, and `lib/ws.ts` upgrades the
scheme:

```ts
// lib/ws.ts:12-14
const url = apiBase().replace(/^http/, "ws") + path;
```

| Page load | Result |
|---|---|
| no query param, dev on :3000 | `http://localhost:3000/api/v1/…` → same-origin, proxied |
| `?api=http://192.168.1.50:8000` | all REST + WS calls target that host |
| `?api=https://…` | `replace(/^http/,"ws")` yields `wss://…` ✔ |

**Known limitation:** the WebSocket connection effect depends only on `[path]`,
while `apiBase()` re-reads `window.location.search` on every render. Changing
`?api=` after mount does **not** re-key the connection, so the override is
effectively read once per mount.

---

## 3. Page inventory

`NAV` in `app/layout.tsx:11-16`:

| Route | File | In nav | Renders |
|---|---|---|---|
| `/` | `app/page.tsx` (201) | ✅ `Activity` "Dashboard" | live kennel dashboard |
| `/diagnostics` | `app/diagnostics/page.tsx` (323) | ✅ `Eye` "Diagnostics" | VOC screening + explainability |
| `/train` | `app/train/page.tsx` (7) | ✅ `Cpu` "Training Studio" + `QML` badge | 7-line wrapper → `<TrainingStudio/>` |
| `/collect` | `app/collect/page.tsx` (303) | ✅ `Database` "Data Lab" | recording session controller |
| `/lab` | `app/lab/page.tsx` (204) | ❌ **unlinked** | hybrid experiment launcher + leaderboard |
| `/404` | Next built-in | — | static export emits `404.html` |

**`/lab` exists but is not in the navigation** — reachable only by typing the
URL. Given that the lab produces the project's headline comparative numbers, it
should be in the nav.

### 3.1 `/` Dashboard

State: `frame: Frame | null`, `count`, `fps`, `state: KennelState | null`.

- `useTelemetrySocket<Frame>("/ws/stream", …)` at `:43-46`; the callback does
  `setFrame(data)` and `setCount(c => c + 1)`.
- FPS: a `useEffect` on `[count]` creates a `setInterval(…, 1000)` **and is
  re-created on every frame** (`:49-56`), so the interval is torn down and re-armed
  on each render. It works, but it is 10 interval re-creations per second. A
  `useRef` for the count and a `[…]`-less effect would be correct.
- Polls `GET /api/v1/kennel/state` every 2000 ms with a bare `catch {}` (`:64`) —
  on failure `state` stays `null` and the UI falls back to `?? "Local Sim"` /
  `?? "Ready"`.
- `accRms = sqrt(mean(acc²))/√3`, `toFixed(2)`; `totalFsr` is computed at `:77-79`
  and **never rendered** (dead variable).

**Layout:** a 4-panel KPI row — Live Stream Status (dot + `{stream_source} · {fps} FPS`),
Total Ingested Frames (with the hardcoded caption "100 Hz Native Sensor Sampling"),
Motion Energy (RMS, in *g*), Model Diagnostic State (linking `/train`) — then a
12-column grid: `lg:col-span-4` = `<KennelDiagram>` + a "Quantum Screening Suite"
nav card; `lg:col-span-8` = `<ImuWaveforms>` + `<SensorBars>`.

### 3.2 `/diagnostics`

State: `diag` (WS), `history: number[]` **seeded with a hardcoded fake series**
`[0.15, 0.18, 0.12, 0.22, 0.19, 0.25, 0.82, 0.88, 0.85]` (`:44`), `selectedPreset`,
`sampleId` (`"SMPL_ONC_8892"`), `patientAge` (62), `smokingStatus` ("Former"),
`isRunning`, `detailedResult`.

- `useTelemetrySocket<Diagnostic>("/ws/diagnostic", …)`; appends
  `probability_cancer` to `history`, capped at 60 (`h.slice(-59)`).
- `getSensorPresetValues(preset)` (`:60-74`) **fabricates 64 floats client-side**
  from sinusoids + `Math.random()`. Presets: `lung_positive = sin(i·0.3)·0.8 + 1.8 + rand·0.15`;
  `healthy_control = sin(i·0.3)·0.2 + 0.4 + rand·0.08`; `breast_positive = cos(i·0.25)·0.7 + 1.6 + rand·0.15`;
  `colorectal_positive` falls to the `else` branch `sin(i·0.2)·0.9 + 1.5 + rand·0.12`.
- `runQuantumDiagnostic()` → `POST /api/v1/predict?deep_explain=true`. Errors
  surface via **`alert(...)`** (`:90`) — the only `alert()` in the app; everything
  else uses inline error panels.
- Renders `<PredictionCard diag>` (WS-driven) + an inline SVG "Risk History (Last
  60 Windows)" polyline with a dashed 50% reference line — **or**, when
  `detailedResult` is set, a result banner + `<BiomarkerWaterfall>` + clinical
  summary.

**Presentation risk:** the risk-history chart is pre-populated with a fabricated
rising series. If a reviewer glances at it before a diagnostic arrives, they will
see a plausible-looking positive result. The seed series should be removed or
visibly marked as placeholder data.

### 3.3 `/train` — `TrainingStudio` (649 lines)

The largest component; entirely self-contained.

`MODELS` (`:79-92`) — 12 entries in three `<optgroup>`s: 5 `Quantum`
(`QSVM_BioZZ`, `QSVM_PauliZZ`, `VQC_StronglyEntangled`, `VQC_RealAmplitudes`,
`QCNN`), 2 `Hybrid` (`Quantum_Kernel_XGB`, `Quantum_Augmented_XGB`), 5
`Classical` (`SVM_RBF`, `SVM_Linear`, `Logistic_Regression`, `Random_Forest`,
`MLP_NeuralNet`, `XGBoost`).

`CANCER_TYPES` (`:94-100`): Lung, Breast, Colorectal, Prostate, Ovarian.

Local defaults (`:104-118`): `modelType "QSVM_BioZZ"`, `targetCancer
"Lung_Cancer"`, `nQubits 6`, `samplesPerClass 50`, `epochs 25`,
`learningRate 0.03`, `cParam 1.0`, `activeTab "single" | "benchmark"`.

`handleTrain()` (`:120-141`) posts `{model_type, target_cancer, n_qubits,
samples_per_class, epochs, learning_rate, c_param, test_size: 0.2}` —
**`test_size` is hardcoded and `random_seed` is never sent** (server default 42).
`handleBenchmark()` posts `{target_cancer, n_samples_per_class, n_qubits}`.

Conditional inputs (`:318`, `:347`): epochs + lr appear only for
`VQC*`/`QCNN`/`MLP_NeuralNet`/`Quantum_Augmented_XGB`; the C slider only for
`QSVM*`/`SVM*`. Slider ranges: qubits 4–12, samples/class 20–150, C 0.1–10.0 —
all **narrower** than the server's bounds (qubits 4–12 match, samples 15–200,
C 0.01–50).

**Results panel:** paradigm-coloured badge (Quantum→purple, Hybrid→amber, else
blue) · Train Time · Inference/Sample · 4 primary metric tiles (ROC-AUC,
Sensitivity, Specificity, Brier) · 4 secondary (Accuracy, Balanced Acc, Precision,
Macro F1) · `<LossChart>` + `<ConfusionMatrix>` · and, when `circuit_profile` is
present, the "NISQ Quantum Hardware Compilation Profile" panel (Active Qubits,
Circuit Depth, 2-Qubit CNOTs, Total Gates, `nisq_verdict` chip).

Empty state (`:553-563`): "No Active Model Trained Yet" with guidance.

**Benchmark tab:** a table of Model / Paradigm / ROC-AUC / Sensitivity /
Specificity / F1 / Brier / Train Time, with the empty state "Click 'Run
Multi-Model Benchmark Suite'…".

**Cosmetic defect:** `Â·` appears literally in the rendered output at `:434-435`
and `:576` — a double-encoded `·` (mojibake from an earlier encoding pass).

### 3.4 `/collect` Data Lab

`LABELS` (`:37-43`) — 5 options: `cancer` "Malignant Target (Cancer Sample)",
`control` "Healthy / Negative Control", `benign` "Benign Tissue Biopsy",
`alert` "Alert / Investigative Posture", `calm` "Resting Physiological Baseline".

State: `dogId` (`"Rex"`), `sampleId` (`"S0"`), `label` (`"cancer"`),
`duration` (20), `error`, `lastSaved`.

- Polls `/api/v1/kennel/state` every **1000 ms** (`:63-67`) with the comment
  "First data paint arrives on the first interval tick" — there is deliberately
  **no immediate first call** (unlike some other pages which have the same
  omission by accident).
- `start()` → `POST /api/v1/kennel/start`; `stop()` → `POST /api/v1/kennel/stop`
  **with an empty JSON body** even though the endpoint takes no body.
- Progress: `min(1, elapsed_s / target_s)`.

Layout: header banner + `/train` link; `lg:col-span-7` form (dog id, trial id,
label `<select>`, duration `min=5 max=300`) and either a Start button or an
active panel (recording dot, `{elapsed}/{target}s`, gradient bar, "Ingested {n}
sensor frames for subject {dog_id}", Stop & Save); `lg:col-span-5` "Ingestion
Telemetry" 2×2 stat grid (Stream Source, Total Frames Seen, Diagnostic State,
Connection) + a "Next Step" card. The Connection tile renders `"Online"`/`"Active"`
and **never shows "Offline"**.

### 3.5 `/lab`

`EXPERIMENTS` (`:26-31`): `search` "Hybrid search (curated presets)",
`map_ablation` "Feature-map ablation (RQ3)", `modality_ablation` "Modality
ablation (RQ2)", `robustness` "Noise robustness (§VII-E)".

State: `jobs`, `dataset` (`"voc_real"`), `experiment` (`"search"`),
`maxSamples` (120), `vqcEpochs` (**6**, vs the server default 8), `nSplits` (5),
`error`.

`launch()` (`:56-68`) hardcodes `n_components: 6, seed: 42` (not user-editable).
Polls `/api/v1/lab/runs` every 2000 ms with a silent `catch { /* backend offline */ }`.

Dataset `<select>` labels: `voc_real` → "Lung VOC (real, 427)", `kennel_synth` →
"Kennel trials (synthetic)".

Renders a runs list — bold `{experiment} · {dataset} · folds={n} · cap={m}`, a
status pill (`done`/`error`/pulsing `running`), the last progress line, error
text — then, for `doneJobs[0]`, a leaderboard `<table>` reading `roc_auc_mean`,
`roc_auc_std`, `accuracy_mean`, `train_time_s_mean`.

Empty state (`:131-134`): "No runs yet. Note: the API executes experiments
in-process — launch times depend on the quantum budget."

**Operational note:** only `experiment: "search"` reports progress, and
`robustness` always errors (see
[`08_experiment_lab.md`](08_experiment_lab.md) §6.4). The page therefore shows
progress lines for one of four experiment types.

---

## 4. Component inventory

| Component | Lines | Props / exports | Renders |
|---|---|---|---|
| `KennelDiagram.tsx` | 100 | **exports `interface Frame`** + default | inline SVG `viewBox="0 0 320 300"`: a 260×260 kennel `<rect rx=16>` at (30,20); a centre `sample` circle r=22 at (160,150); 4 corner pads (FL 60,40 / FR 260,40 / RL 60,260 / RR 260,260) as 52×52 rects with opacity-mapped fill `rgba(99,102,241, 0.15 + 0.85·min(1, v/1024))`; IR dots (red `#ef4444` when > 0) at `+34,+34`; 2 top-front IR at x 70/250, y 34 (amber `#f59e0b`); caption `US bottom … cm · US top … cm` |
| `ImuWaveforms.tsx` | 76 | `{frame}` | `<canvas width={560} height={180}>` redrawn on every `frame` change; 4 series in a `useRef` history, `MAX = 100` (~10 s @ 10 Hz); y-domain `min = min(…,-1)`, `max = max(…,11)`, `pad = 4`; colours ax `#818cf8`, ay `#34d399`, az `#fbbf24`, gx `#f472b6`, `lineWidth 1.5`; legend `["acc X","acc Y","acc Z","gyr X"]` |
| `SensorBars.tsx` | 39 | `{frame}` | 6 gradient bars (`US bottom`/`US top` max 200, `FSR FL/FR/RL/RR` max 1024); `width = min(100, max(0,v)/max·100)`; `linear-gradient(90deg,#4f46e5,#a855f7)`; `duration-100`; readout `.toFixed(0)` |
| `PredictionCard.tsx` | 134 | **exports `interface Diagnostic`** (`type`; `status: "ok"\|"uncertain"\|"untrained"`; `probability_cancer?`; `smoothed_probability?`; `confidence_threshold?`; `detail?`) | 3 states: `null` → "Waiting for telemetry diagnostic window…"; `"untrained"` → amber "Model Awaiting Calibration" + pointer to the Training Studio; otherwise an SVG confidence ring (`viewBox 0 0 140 140`, r=55, `strokeDasharray={`${clamp(v·314,0,314)} 345`}`, `rotate(-90 70 70)`, `duration-700`) with `url(#roseGrad)` when `≥0.5` else `url(#emeraldGrad)`, centre `pct%` + "MALIGNANCY RISK", plus "EMA Smoothed" and "Decision Cutoff" tiles and a "Low Confidence" pill. Footer disclaimer: "Research aid." |
| `BiomarkerWaterfall.tsx` | 114 | **exports `interface Biomarker`** (`compound`; `importance_score`; `clinical_impact`) | "Top Attributed VOC Metabolites (Reverse Hilbert Mapping)" — per-row chip elevated when `clinical_impact` includes "Elevated"/"Risk", score `.toFixed(4)`, bar `min(100, max(8, score/maxScore·100))`; then "Biochemical Pathway Dysregulation" — 2-col grid, only when `pathways` is non-empty. Empty: "No biomarker attribution computed for this window." |
| `LossChart.tsx` | 107 | `{lossHistory, title = "Convergence Loss Trajectory", color = "#a855f7"}` | inline SVG 450×150, `padding = 30`; gradient area `url(#lossGrad)` (colour at 35% → 0%); `polyline strokeWidth 2.5`; per-epoch `circle r=3` with `hover:r-5` and `<title>Epoch {i+1}: Loss = {v.toFixed(4)}</title>`; 3 dashed gridlines; y labels `{max}/{min}`; x labels `Ep 1`/`Ep {n}`. Empty: "Loss trajectory unavailable for analytical closed-form models (e.g. QSVM / SVM)." |
| `ConfusionMatrix.tsx` | 63 | `{matrix: number[][]}` — comment `// [[TN, FP], [FN, TP]]` | 2×2 colour-coded cards: True Negative (emerald, "Healthy"), False Positive (rose, "Type I"), False Negative (amber, "Missed"), True Positive (purple, "Malignant"), each with a count and `{pct}% of cohort`; header "Held-Out Test (N = {total})"; guard `if (!matrix \|\| matrix.length < 2) return null` |
| `TrainingStudio.tsx` | 649 | — | see §3.3 |

### 4.1 Presentation issues in the components

| Component | Issue |
|---|---|
| `KennelDiagram` | the no-frame case is an **early return** to a "Waiting for telemetry…" panel — there is no diagram at all until the first frame, so the dashboard's centrepiece is blank on load |
| `SensorBars` | renders **all-zero bars when `frame` is null** — no placeholder, so it looks like a live reading of zero |
| `ImuWaveforms` | no placeholder; a blank canvas reads as broken |
| `PredictionCard` | the ring colour threshold is one-sided (`p ≥ 0.5`) while the server's `status` is two-sided (`max(p,1−p) ≥ 0.6`) — a confident-healthy frame (`p = 0.1`, `status: "ok"`) is rendered with the rose/malignancy gradient |
| `TrainingStudio` | `Â·` mojibake at `:434-435`, `:576` |

---

## 5. Styling system

`app/globals.css` — 133 lines.

### 5.1 Custom properties (`:3-14`)

```css
--bg-primary:    #07090e
--bg-secondary:  #0d121f        /* declared, never referenced */
--bg-panel:      rgba(17, 24, 39, 0.75)
--border-subtle: rgba(255, 255, 255, 0.08)
--border-active: rgba(129, 140, 248, 0.4)   /* declared, never referenced */
--accent-cyan:    #06b6d4   /* all six --accent-* are declared, never referenced */
--accent-purple:  #a855f7
--accent-indigo:  #6366f1
--accent-emerald: #10b981
--accent-rose:    #f43f5e
```

Six of the eleven tokens are dead. Either adopt them in the component classes or
remove them — dead tokens mislead the next contributor.

### 5.2 Ambient background

```css
body::before {   /* indigo radial glow, 40vw, top/left -10vw, z-index -1 */
  background: radial-gradient(circle, rgba(99,102,241,0.12), transparent 70%); }
body::after {    /* purple radial glow, 45vw, bottom/right -10vw, z-index -1 */
  background: radial-gradient(circle, rgba(168,85,247,0.10), transparent 70%); }
```

Both `pointer-events: none`. This is the "quantum lab" visual identity — two soft
coloured glows behind a near-black surface.

### 5.3 Custom classes

| Class | Definition | Used? |
|---|---|---|
| `.panel` | glassmorphism: `backdrop-filter: blur(16px)`, 16 px radius, `box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37)`, hover border `rgba(255,255,255,0.14)` | ✅ widely |
| `.panel-glow` | `:64-67` | ❌ **never used** |
| `.quantum-badge` | uppercase pill, `#c084fc` | ✅ |
| `.quantum-btn` | `linear-gradient(135deg,#4f46e5,#9333ea)`, hover/active/`:disabled{opacity:.5;cursor:not-allowed}` | ✅ |
| `::-webkit-scrollbar` | 6 px custom track/thumb | ✅ |
| `@keyframes pulseGlow` + `.animate-pulse-glow` | `:126-133` | ❌ **never used** — pages use Tailwind's `animate-pulse` |

### 5.4 Layout shell (`app/layout.tsx:22-82`)

```tsx
<html lang="en" className="dark">
<body className="min-h-screen antialiased bg-[#07090e] text-gray-100 flex flex-col">
  <header className="sticky … backdrop-blur …">…</header>
  <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">…</main>
  <footer>PennyLane · Qiskit · PyTorch · Scikit-Learn</footer>
```

Two things to note:

1. **`className="dark"` is present with no `dark:` variant logic anywhere.** The
   console is dark-only; the class is vestigial (and `darkMode` is not configured
   for Tailwind v4 either).
2. **The footer advertises Qiskit**, which is **not a dependency and has no code
   in the repository** — finding #12 in `docs/reports/code_evaluation_report.md:31`
   was raised about the README badge, and the same claim survives here. The
   frontend package has no Qiskit dependency. This should be corrected before any
   presentation.

---

## 6. State management

**There is none.** Zero Redux / Zustand / Jotai / React Context. All state is
`useState` + `useRef` + `setInterval`.

Consequences:

- **The Dashboard and Diagnostics pages each open their own independent
  WebSocket connection** — two sockets, two server-side broadcasts.
- `useTelemetrySocket` holds its callback in a `useRef` updated in a separate
  effect (`:9-13`) so the connection is **not** torn down when the parent
  re-renders with a new inline closure. This is the correct pattern and is
  explicitly designed for.
- Cross-component data flow is exclusively via the WebSocket and the polled REST
  state endpoint. There is no shared cache, so `/` and `/diagnostics` each
  re-derive the same stream state independently.

### 6.1 Reconnection (`lib/ws.ts`)

```ts
onopen   → setConnected(true); attempt = 0
onmessage→ handlerRef.current(JSON.parse(event.data))   // inside try/catch, silent drop
onclose  → setConnected(false);
           if (!deliberate) { delay = Math.min(8000, 500 * 2 ** attempt++);
                              setTimeout(connect, delay); }
onerror  → socket?.close()   // routes through onclose → triggers backoff
```

**Exponential backoff: 500, 1000, 2000, 4000, 8000, 8000, 8000 … ms, capped at
8 s.** Good. The `try/catch` around `JSON.parse` means a malformed frame is
dropped without logging — acceptable for a telemetry stream.

Cleanup (`:45-49`) sets `closed = true`, clears the timer, and closes the socket.

---

## 7. Build output

`frontend/out/` (gitignored but present on this checkout): `index.html`,
`collect.html`, `diagnostics.html`, `train.html`, `lab.html`, `404.html`,
`_not-found.html`, RSC payload `.txt`/`.html` files, and 14 hashed
`_next/static/chunks/*` assets.

`qmlkit_desktop.py::frontend_dist()` probes, in order:
`<exe>/frontend_out`, `<exe>/frontend/out`, `<root>/frontend_out`,
`<root>/frontend/out`, `<root>/../frontend/out`. **The first existing directory
wins, so an external `frontend_out/` folder overrides the embedded snapshot** —
which makes the packaged console hot-swappable without a rebuild.

On this checkout only `frontend/out` exists (no `frontend_out`, and no `models/`
directory, so the desktop app would report `untrained`).

---

## 8. Frontend review checklist

| # | Item | Severity |
|---|---|---|
| 1 | Footer claims "Qiskit" — no Qiskit dependency or code exists | **High** (factual claim) |
| 2 | `/diagnostics` risk history seeded with a **fabricated** rising series | **High** (could mislead a reviewer) |
| 3 | `/lab` is not in the navigation | Medium |
| 4 | `robustness` experiment always returns `status: "error"` from the API | Medium |
| 5 | `PredictionCard` colour threshold (one-sided 0.5) vs server `status` (two-sided 0.6) | Medium |
| 6 | `TrainingStudio` `Â·` mojibake at `:434-435`, `:576` | Low |
| 7 | 6 dead CSS custom properties, plus `.panel-glow` and `.animate-pulse-glow` | Low |
| 8 | `className="dark"` with no `dark:` variants | Low |
| 9 | Dashboard FPS interval re-created on every frame (10×/s) | Low |
| 10 | `SensorBars` renders zero-bars instead of a placeholder; `KennelDiagram`/`ImuWaveforms` render nothing | Low (polish) |
| 11 | No `typecheck` script; types only verified by `next build` | Low |
| 12 | `?api=` override is not re-read on change | Low |
| 13 | `/collect` Connection tile never renders "Offline" | Low |
| 14 | `docs/upgrade/04_nextjs_frontend.md:23-24` documents `PathwayAttribution.tsx` and `ConnectionBadge.tsx`, which do not exist | Low (doc drift) |
| 15 | Several unused lucide-react imports across pages (`Flame`, `Layers`, `Radio`, `AlertTriangle`, `Gauge`, `Play`, `Download`, `RotateCw`, `Activity`, `CheckCircle2`, `Zap`) | Low |
