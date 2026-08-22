"use client";

import { useEffect, useState } from "react";
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

const LABELS = ["cancer", "control", "benign", "calm", "alert"];

export default function CollectPage() {
  const [state, setState] = useState<KennelState | null>(null);
  const [dogId, setDogId] = useState("Rex");
  const [sampleId, setSampleId] = useState("S0");
  const [label, setLabel] = useState("control");
  const [duration, setDuration] = useState(20);
  const [error, setError] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  async function refresh() {
    try {
      setState(await getJson<KennelState>("/api/v1/kennel/state"));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    // Poll every second; first paint of data arrives on the first tick.
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
      await postJson("/api/v1/kennel/start", { dog_id: dogId, sample_id: sampleId, label, duration_s: duration });
      setLastSaved(null);
    } catch (e) {
      setError(String(e));
    }
  }

  async function stop() {
    try {
      const res = await postJson<{ path?: string }>("/api/v1/kennel/stop", {});
      setLastSaved(res.path ?? null);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="max-w-xl mx-auto space-y-4">
      <div className="panel p-6 space-y-4">
        <h2 className="text-lg font-semibold">Guided Recording Session</h2>
        <p className="text-sm text-gray-400">
          Position the dog over the center sample, then record a labelled window.
          Sessions are saved to <code>data/kennel/&lt;label&gt;/</code>.
        </p>

        <div className="grid grid-cols-2 gap-3">
          <label className="text-sm">
            <span className="block text-gray-400 mb-1">Dog ID</span>
            <input value={dogId} onChange={(e) => setDogId(e.target.value)}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2" />
          </label>
          <label className="text-sm">
            <span className="block text-gray-400 mb-1">Sample ID</span>
            <input value={sampleId} onChange={(e) => setSampleId(e.target.value)}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2" />
          </label>
          <label className="text-sm">
            <span className="block text-gray-400 mb-1">Label</span>
            <select value={label} onChange={(e) => setLabel(e.target.value)}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2">
              {LABELS.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="block text-gray-400 mb-1">Duration (s)</span>
            <input type="number" min={5} max={300} value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2" />
          </label>
        </div>

        {!recording ? (
          <button onClick={start}
            className="w-full py-3 rounded-xl font-semibold text-white transition"
            style={{ background: "linear-gradient(135deg,#4f46e5,#9333ea)" }}>
            Start recording
          </button>
        ) : (
          <>
            <div className="h-3 bg-[#1f2937] rounded-full overflow-hidden">
              <div className="h-full transition-all"
                style={{ width: `${progress * 100}%`, background: "linear-gradient(90deg,#10b981,#f59e0b,#ef4444)" }} />
            </div>
            <div className="text-xs text-gray-400 text-center">
              Recording “{state?.recorder.label}” for {state?.recorder.dog_id} ·{" "}
              {state?.recorder.samples} frames · {state?.recorder.elapsed_s}s /{" "}
              {state?.recorder.target_s}s
            </div>
            <button onClick={stop}
              className="w-full py-3 rounded-xl font-semibold bg-red-600/80 hover:bg-red-600 text-white">
              Stop &amp; save
            </button>
          </>
        )}

        {lastSaved && (
          <div className="text-xs text-emerald-400">Saved to {lastSaved}</div>
        )}
        {error && <div className="text-xs text-red-400">{error}</div>}
      </div>

      <div className="panel p-4 text-xs text-gray-500">
        Stream source: <b>{state?.stream_source ?? "?"}</b> · model:{" "}
        <b>{state?.model_status ?? "?"}</b> · frames seen:{" "}
        <b>{state?.frames_seen ?? 0}</b>. After collecting sessions run{" "}
        <code>python scripts/train_kennel_model.py</code>.
      </div>
    </div>
  );
}
