"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  Flame,
  Layers,
  Play,
  RotateCw,
  Sparkles,
  Square,
  Timer,
  Zap,
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
  { id: "cancer", name: "Malignant Target (Cancer Sample)", color: "text-rose-400" },
  { id: "control", name: "Healthy / Negative Control", color: "text-emerald-400" },
  { id: "benign", name: "Benign Tissue Biopsy", color: "text-amber-400" },
  { id: "alert", name: "Alert / Investigative Posture", color: "text-purple-400" },
  { id: "calm", name: "Resting Physiological Baseline", color: "text-indigo-400" },
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
    refresh();
    const id = setInterval(refresh, 1000);
    return () => clearInterval(id);
  }, []);

  const recording = state?.recorder?.recording ?? false;
  const progress =
    recording && state?.recorder.target_s
      ? Math.min(1, state.recorder.elapsed_s / state.recorder.target_s)
      : 0;

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
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="panel p-6 border-l-4 border-l-indigo-500 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Database className="w-5 h-5 text-indigo-400" />
              Canine Data Lab &amp; Ground Truth Recorder
            </h2>
            <span className="quantum-badge">Session Ingestion</span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Capture calibrated sensor sessions with synchronized canine telemetry, IMU micro-movements, and VOC labels.
          </p>
        </div>

        <Link
          href="/train"
          className="px-4 py-2 rounded-xl bg-purple-600/30 hover:bg-purple-600/40 border border-purple-500/40 text-purple-200 text-xs font-semibold flex items-center gap-1.5 transition shrink-0"
        >
          <Cpu className="w-3.5 h-3.5 text-purple-400" />
          <span>Go to Training Studio →</span>
        </Link>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Recording Controls (6 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="panel p-6 space-y-5">
            <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2 border-b border-white/5 pb-3">
              <Timer className="w-4 h-4 text-indigo-400" />
              Session Parameters
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300">Canine Subject ID</label>
                <input
                  value={dogId}
                  onChange={(e) => setDogId(e.target.value)}
                  className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none font-mono"
                  placeholder="e.g. Rex, Luna"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300">Specimen / Trial ID</label>
                <input
                  value={sampleId}
                  onChange={(e) => setSampleId(e.target.value)}
                  className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none font-mono"
                  placeholder="e.g. S0, S1"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300">Biological Class Label</label>
                <select
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                >
                  {LABELS.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300">Window Duration (seconds)</label>
                <input
                  type="number"
                  min={5}
                  max={300}
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Record Action Buttons */}
            {!recording ? (
              <button
                onClick={start}
                className="w-full py-3.5 rounded-xl quantum-btn flex items-center justify-center gap-2 text-sm font-bold mt-2"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Start Guided Recording Session ({duration}s)</span>
              </button>
            ) : (
              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-amber-400 font-bold flex items-center gap-1.5 animate-pulse">
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    Recording Active ({state?.recorder.label})
                  </span>
                  <span className="font-mono text-gray-300">
                    {state?.recorder.elapsed_s}s / {state?.recorder.target_s}s
                  </span>
                </div>

                <div className="h-3 bg-[#0b0f19] rounded-full overflow-hidden border border-white/10 p-0.5">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${progress * 100}%`,
                      background: "linear-gradient(90deg, #6366f1, #a855f7, #ec4899)",
                    }}
                  />
                </div>

                <div className="text-xs text-gray-400 text-center font-mono">
                  Ingested {state?.recorder.samples} sensor frames for subject {state?.recorder.dog_id}
                </div>

                <button
                  onClick={stop}
                  className="w-full py-3 rounded-xl bg-rose-600/90 hover:bg-rose-600 text-white font-bold text-sm transition flex items-center justify-center gap-2"
                >
                  <Square className="w-4 h-4 fill-white" />
                  <span>Stop &amp; Save Session</span>
                </button>
              </div>
            )}

            {lastSaved && (
              <div className="p-3 bg-emerald-950/40 border border-emerald-500/40 rounded-xl text-emerald-300 text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Session saved successfully to: <code>{lastSaved}</code></span>
              </div>
            )}

            {error && (
              <div className="p-3 bg-rose-950/40 border border-rose-500/40 rounded-xl text-rose-300 text-xs">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Live Sensor Telemetry & Stats (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="panel p-5 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2 border-b border-white/5 pb-3">
              <Zap className="w-4 h-4 text-emerald-400" />
              Ingestion Telemetry
            </h3>

            <div className="grid grid-cols-2 gap-3 text-center">
              <div className="p-3 bg-[#0b0f19] border border-white/5 rounded-xl">
                <span className="text-[10px] text-gray-400 uppercase block">Stream Source</span>
                <span className="text-sm font-bold text-white mt-1 block font-mono">
                  {state?.stream_source ?? "Local Sim"}
                </span>
              </div>

              <div className="p-3 bg-[#0b0f19] border border-white/5 rounded-xl">
                <span className="text-[10px] text-gray-400 uppercase block">Total Frames Seen</span>
                <span className="text-sm font-bold text-indigo-400 mt-1 block font-mono">
                  {state?.frames_seen?.toLocaleString() ?? "0"}
                </span>
              </div>

              <div className="p-3 bg-[#0b0f19] border border-white/5 rounded-xl">
                <span className="text-[10px] text-gray-400 uppercase block">Diagnostic State</span>
                <span className="text-sm font-bold text-purple-400 mt-1 block capitalize">
                  {state?.model_status ?? "Ready"}
                </span>
              </div>

              <div className="p-3 bg-[#0b0f19] border border-white/5 rounded-xl">
                <span className="text-[10px] text-gray-400 uppercase block">Connection</span>
                <span className="text-sm font-bold text-emerald-400 mt-1 block">
                  {state?.connected ? "Online" : "Active"}
                </span>
              </div>
            </div>

            <div className="p-4 bg-purple-950/20 border border-purple-500/20 rounded-xl space-y-2 text-xs text-gray-300">
              <div className="font-semibold text-purple-300 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Next Step: Quantum Model Training</span>
              </div>
              <p className="leading-relaxed">
                After recording sessions across cancer and control cohorts, invoke in-browser quantum training to train a BioZZ QSVM or VQC classifier.
              </p>
              <Link
                href="/train"
                className="inline-flex items-center gap-1 text-purple-400 hover:text-purple-300 font-semibold mt-1"
              >
                <span>Launch Training Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
