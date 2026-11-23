"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  Play,
  Square,
  Timer,
} from "lucide-react";
import { getJson, postJson } from "@/lib/api";

interface KennelState {
  stream_source: string;
  connected: boolean;
  frames_seen: number;
  model_status: string;
  recorder: {
    recording: boolean;
    samples: number;
    elapsed_s: number;
    target_s?: number;
    dog_id?: string;
    label?: string;
  };
}

const LABELS = [
  { id: "cancer", name: "Malignant Target (Cancer Sample)" },
  { id: "control", name: "Healthy / Negative Control" },
  { id: "benign", name: "Benign Tissue Biopsy" },
  { id: "alert", name: "Alert / Investigative Posture" },
  { id: "calm", name: "Resting Physiological Baseline" },
];

export default function CollectPage() {
  const [state, setState] = useState<KennelState | null>(null);
  const [dogId, setDogId] = useState("Rex");
  const [sampleId, setSampleId] = useState("S0");
  const [label, setLabel] = useState("cancer");
  const [duration, setDuration] = useState(20);
  const [error, setError] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  async function refresh() {
    try {
      setState(await getJson<KennelState>("/api/v1/kennel/state"));
      setError(null);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  useEffect(() => {
    // First data paint arrives on the first interval tick.
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, []);

  const recording = state?.recorder?.recording ?? false;
  const progress =
    recording && state?.recorder.target_s
      ? Math.min(1, state.recorder.elapsed_s / state.recorder.target_s)
      : 0;
  const connectionText = !state
    ? error
      ? "Offline"
      : "Connecting..."
    : state.connected
      ? "Online"
      : "Offline";
  const connectionTone = state?.connected ? "chip-good" : state || error ? "chip-bad" : "";

  async function start() {
    try {
      await postJson("/api/v1/kennel/start", {
        dog_id: dogId,
        sample_id: sampleId,
        label,
        duration_s: duration,
      });
      setLastSaved(null);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function stop() {
    try {
      const res = await postJson<{ path?: string }>("/api/v1/kennel/stop", {});
      setLastSaved(res.path ?? null);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex flex-col gap-2">
          <h1 className="section-title">Canine data lab</h1>
          <p className="muted" style={{ margin: 0, maxWidth: 720, fontSize: 14 }}>
            Capture calibrated sensor sessions with synchronized canine telemetry, IMU micro-movements, and VOC
            labels.
          </p>
        </div>
        <Link href="/train" className="btn-quiet" style={{ alignSelf: "flex-start", textDecoration: "none" }}>
          <Cpu size={16} />
          Training studio
          <ArrowRight size={15} />
        </Link>
      </header>

      <hr className="divider" />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Recording controls */}
        <section className="lg:col-span-7">
          <div className="surface flex flex-col gap-5">
            <div className="row">
              <h2 className="section-title flex items-center gap-2">
                <Timer size={18} /> Session parameters
              </h2>
              <span className="chip">Session ingestion</span>
            </div>
            <hr className="divider" />

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="dog-id">
                  Canine subject ID
                </label>
                <input
                  id="dog-id"
                  className="field"
                  value={dogId}
                  onChange={(e) => setDogId(e.target.value)}
                  placeholder="e.g. Rex, Luna"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="sample-id">
                  Specimen / trial ID
                </label>
                <input
                  id="sample-id"
                  className="field"
                  value={sampleId}
                  onChange={(e) => setSampleId(e.target.value)}
                  placeholder="e.g. S0, S1"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="label">
                  Biological class label
                </label>
                <select
                  id="label"
                  className="field"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                >
                  {LABELS.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="duration">
                  Window duration (seconds)
                </label>
                <input
                  id="duration"
                  type="number"
                  min={5}
                  max={300}
                  className="field"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                />
              </div>
            </div>

            <hr className="divider" />

            {!recording ? (
              <button type="button" onClick={start} className="btn-primary w-full">
                <Play size={16} />
                Start Guided Recording Session ({duration}s)
              </button>
            ) : (
              <div className="flex flex-col gap-3">
                <div className="row flex-wrap gap-3">
                  <span className="chip chip-warn">
                    <span
                      style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--warn)" }}
                      aria-hidden="true"
                    />
                    Recording · {state?.recorder.label}
                  </span>
                  <span className="num muted" style={{ fontSize: 13 }}>
                    {state?.recorder.elapsed_s}s / {state?.recorder.target_s}s
                  </span>
                </div>
                <div className="progress">
                  <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
                </div>
                <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                  Ingested{" "}
                  <span className="num" style={{ color: "var(--ink)", fontWeight: 700 }}>
                    {state?.recorder.samples}
                  </span>{" "}
                  sensor frames for subject{" "}
                  <strong style={{ color: "var(--ink)" }}>{state?.recorder.dog_id}</strong>
                </p>
                <button
                  type="button"
                  onClick={stop}
                  className="btn-quiet w-full"
                  style={{ color: "var(--bad)", borderColor: "var(--bad)" }}
                >
                  <Square size={16} />
                  Stop &amp; Save Session
                </button>
              </div>
            )}

            {error && (
              <div className="surface-sm row" style={{ borderLeft: "3px solid var(--bad)", color: "var(--bad)" }} role="alert">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {/* Telemetry and sessions */}
        <aside className="lg:col-span-5 flex flex-col gap-6">
          <div className="surface flex flex-col gap-5">
            <div className="row">
              <h2 className="section-title flex items-center gap-2">
                <Database size={18} /> Ingestion telemetry
              </h2>
              <span className={`chip ${connectionTone}`}>{connectionText}</span>
            </div>
            <hr className="divider" />

            <div className="grid grid-cols-2 gap-x-6 gap-y-5">
              <div className="flex flex-col gap-1">
                <span className="label">Stream source</span>
                <span style={{ fontWeight: 800, overflowWrap: "anywhere" }}>
                  {state?.stream_source ?? "Local Sim"}
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="label">Total frames seen</span>
                <span className="num" style={{ fontWeight: 800 }}>
                  {state?.frames_seen?.toLocaleString() ?? "—"}
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="label">Diagnostic state</span>
                <span style={{ fontWeight: 800, textTransform: "capitalize" }}>
                  {state?.model_status ?? "Checking..."}
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="label">Connection</span>
                <span className={`chip ${connectionTone}`} style={{ alignSelf: "flex-start" }}>
                  {connectionText}
                </span>
              </div>
            </div>

            <p className="muted" style={{ margin: 0, fontSize: 12 }}>
              Kennel state is polled once per second. Local Sim is the simulation fallback used when hardware is
              absent.
            </p>
          </div>

          <div className="surface flex flex-col gap-4">
            <h2 className="section-title">Recent sessions</h2>
            <hr className="divider" />
            {lastSaved ? (
              <div className="flex items-start gap-3" style={{ color: "var(--good)" }}>
                <CheckCircle2 size={18} />
                <div className="flex flex-col gap-1">
                  <span style={{ fontWeight: 800 }}>Last session saved</span>
                  <code className="num" style={{ fontSize: 12, color: "var(--ink)", overflowWrap: "anywhere" }}>
                    {lastSaved}
                  </code>
                </div>
              </div>
            ) : (
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                No sessions saved in this view yet. Start and stop a guided recording to see its saved path here.
              </p>
            )}
          </div>

          <div className="surface-sm flex flex-col gap-2">
            <strong style={{ fontSize: 14 }}>Next step: quantum model training</strong>
            <p className="muted" style={{ margin: 0, fontSize: 13 }}>
              After recording sessions across cancer and control cohorts, invoke in-browser quantum training to
              train a BioZZ QSVM or VQC classifier.
            </p>
            <Link
              href="/train"
              className="inline-flex items-center gap-2"
              style={{ color: "var(--action)", fontWeight: 800, textDecoration: "none", fontSize: 13 }}
            >
              Launch Training Studio
              <ArrowRight size={15} />
            </Link>
          </div>
        </aside>
      </div>
    </div>
  );
}
