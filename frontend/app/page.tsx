"use client";

import { useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import KennelDiagram, { type Frame } from "@/components/KennelDiagram";
import ImuWaveforms from "@/components/ImuWaveforms";
import SensorBars from "@/components/SensorBars";
import { useTelemetrySocket } from "@/lib/ws";
import { getJson } from "@/lib/api";

interface KennelState {
  stream_source: string;
  connected: boolean;
  frames_seen: number;
  model_status: string;
  recorder: {
    recording: boolean;
    samples: number;
    elapsed_s: number;
  };
}

type AttentionTone = "good" | "warn" | "bad";

const LIVE_COPY: Record<string, { title: string; detail: string }> = {
  BOOT: {
    title: "Kennel node is starting up",
    detail: "Waiting for the node to finish booting before sensor readings settle.",
  },
  CALIBRATE: {
    title: "Calibrating sensors",
    detail: "Sensor baselines are being established — readings are not final yet.",
  },
  IDLE: {
    title: "Kennel is idle",
    detail: "No dog is in the kennel right now. The stream is still live.",
  },
  OCCUPIED: {
    title: "Dog is in the kennel",
    detail: "Presence detected. No active sniff window at the moment.",
  },
  SNIFF: {
    title: "Sniff window active",
    detail: "The dog is sampling and micro-movements are being captured.",
  },
  COOLDOWN: {
    title: "Cooling down",
    detail: "The kennel is settling after a sniff window.",
  },
};

function stateChipClass(state: string | null): string {
  switch (state) {
    case "SNIFF":
    case "OCCUPIED":
      return "chip chip-good";
    case "BOOT":
    case "CALIBRATE":
    case "COOLDOWN":
      return "chip chip-warn";
    default:
      return "chip";
  }
}

function toneChipClass(tone: AttentionTone): string {
  if (tone === "good") return "chip chip-good";
  if (tone === "warn") return "chip chip-warn";
  return "chip chip-bad";
}

function toneLabel(tone: AttentionTone): string {
  if (tone === "good") return "On track";
  if (tone === "warn") return "Check";
  return "Action";
}

function StatusRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="row items-center justify-between gap-4 py-2">
      <dt className="muted text-sm">{label}</dt>
      <dd className="text-sm text-right">{children}</dd>
    </div>
  );
}

export default function DashboardPage() {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [count, setCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [state, setState] = useState<KennelState | null>(null);

  // WebSocket telemetry feed
  const connected = useTelemetrySocket<Frame>("/ws/stream", (data) => {
    setFrame(data);
    setCount((c) => c + 1);
  });

  // Calculate FPS over 1s intervals
  useEffect(() => {
    let lastCount = count;
    const interval = setInterval(() => {
      setFps(count - lastCount);
      lastCount = count;
    }, 1000);
    return () => clearInterval(interval);
  }, [count]);

  // Poll state
  useEffect(() => {
    const poll = async () => {
      try {
        const s = await getJson<KennelState>("/api/v1/kennel/state");
        setState(s);
      } catch {}
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  // Derived readings — only meaningful once a frame has actually arrived.
  const accRms = frame?.acc ? Math.sqrt(frame.acc.reduce((sum, val) => sum + val * val, 0) / 3).toFixed(2) : null;
  const totalFsr = frame?.fsr ? frame.fsr.reduce((sum, val) => sum + val, 0).toFixed(1) : null;

  const liveState = frame?.state ?? null;
  const modelStatus = state?.model_status ?? null;
  const recorder = state?.recorder;
  const liveCopy = frame && liveState ? LIVE_COPY[liveState] : undefined;

  const headline =
    liveCopy?.title ??
    (connected ? "Connected — waiting for the first frame" : "Waiting for telemetry");
  const subline =
    liveCopy?.detail ??
    (connected
      ? "The stream is open; telemetry has not arrived yet."
      : "The console is waiting for a live telemetry stream.");

  // Everything here is derived from real connection, frame and REST state.
  const attention: { tone: AttentionTone; text: string }[] = [];
  if (!connected) {
    attention.push({ tone: "bad", text: "Telemetry link is down. The console is retrying the stream connection." });
  } else if (!frame) {
    attention.push({ tone: "warn", text: "Stream is connected. Waiting for the first telemetry frame." });
  }
  if (modelStatus === "untrained") {
    attention.push({
      tone: "warn",
      text: "Model is untrained. Diagnostics will report untrained until a model artifact is trained.",
    });
  }
  if (recorder?.recording) {
    attention.push({ tone: "good", text: `Recording in progress — ${recorder.samples} samples captured.` });
  }
  if (attention.length === 0) {
    attention.push({ tone: "good", text: "Stream healthy. Nothing needs attention right now." });
  }

  const waiting = <span className="muted">Waiting for telemetry</span>;

  return (
    <div className="stack gap-6">
      {/* Lead panel: the live signal plus what needs attention, in one place. */}
      <section className="surface p-6" aria-labelledby="lead-title">
        <div className="row items-start justify-between gap-4 flex-wrap">
          <div className="stack gap-1">
            <h1 id="lead-title" className="section-title">Live kennel</h1>
            <p className="muted text-sm">One shared view of the kennel stream for operators and reviewers.</p>
          </div>
          <Link href="/diagnostics" className="btn-primary row items-center gap-2">
            Open diagnostics
            <ArrowRight aria-hidden="true" className="h-4 w-4" />
          </Link>
        </div>

        <div className="divider my-5" />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Live signal */}
          <div className="lg:col-span-7 stack gap-4">
            <div className="row items-center gap-2 flex-wrap">
              <span className={stateChipClass(liveState)}>{liveState ?? "Waiting"}</span>
              <span className={connected ? "chip chip-good" : "chip chip-bad"}>
                {connected ? "Stream connected" : "Stream disconnected"}
              </span>
            </div>

            <div className="stack gap-1">
              <p className="text-lg font-semibold">{headline}</p>
              <p className="muted text-sm">{subline}</p>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="surface-sm p-4 stack gap-1">
                <span className="label">Motion · RMS</span>
                <span className="num text-xl">
                  {accRms ?? "—"}
                  {accRms && <span className="muted text-sm"> g</span>}
                </span>
                <span className="muted text-xs">{accRms ? "Collar accelerometer" : "Waiting for telemetry"}</span>
              </div>
              <div className="surface-sm p-4 stack gap-1">
                <span className="label">Pad load · total</span>
                <span className="num text-xl">{totalFsr ?? "—"}</span>
                <span className="muted text-xs">{totalFsr ? "Sum of four corner pads" : "Waiting for telemetry"}</span>
              </div>
            </div>
          </div>

          {/* What needs attention */}
          <div className="lg:col-span-5 stack gap-3">
            <div className="row items-center justify-between gap-3">
              <h2 className="section-title">Needs attention</h2>
              <span className="chip">
                <span className="num">{attention.length}</span> {attention.length === 1 ? "item" : "items"}
              </span>
            </div>

            <ul className="stack">
              {attention.map((item, index) => (
                <li key={index}>
                  {index > 0 && <div className="divider" />}
                  <div className="row items-start gap-3 py-2">
                    <span className={toneChipClass(item.tone)}>{toneLabel(item.tone)}</span>
                    <span className="text-sm">{item.text}</span>
                  </div>
                </li>
              ))}
            </ul>

            <Link href="/diagnostics" className="btn-quiet row items-center gap-2 self-start">
              Review diagnostics
              <ArrowRight aria-hidden="true" className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Compact supporting status list */}
      <section className="surface p-5" aria-labelledby="status-title">
        <h2 id="status-title" className="section-title mb-3">Stream status</h2>
        <dl className="stack">
          <StatusRow label="Source">
            {state?.stream_source ?? waiting}
          </StatusRow>
          <div className="divider" />
          <StatusRow label="Link">
            <span className={connected ? "chip chip-good" : "chip chip-bad"}>
              {connected ? "Connected" : "Disconnected"}
            </span>
          </StatusRow>
          <div className="divider" />
          <StatusRow label="Frames received">
            {count > 0 ? <span className="num">{count.toLocaleString()}</span> : waiting}
          </StatusRow>
          <div className="divider" />
          <StatusRow label="Feed rate">
            {frame ? <span className="num">{fps}</span> : waiting}
            {frame && <span className="muted"> fps</span>}
          </StatusRow>
          <div className="divider" />
          <StatusRow label="Model state">
            {modelStatus ? (
              <span className={`chip ${modelStatus === "ready" ? "chip-good" : "chip-warn"}`}>{modelStatus}</span>
            ) : (
              waiting
            )}
          </StatusRow>
          <div className="divider" />
          <StatusRow label="Recorder">
            {recorder?.recording ? (
              <span className="chip chip-good">
                Recording · <span className="num">{recorder.samples}</span> samples
              </span>
            ) : state ? (
              <span className="muted">Idle</span>
            ) : (
              waiting
            )}
          </StatusRow>
        </dl>
      </section>

      {/* Live visualizers */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5">
          <KennelDiagram frame={frame} />
        </div>
        <div className="lg:col-span-7 stack gap-6">
          <ImuWaveforms frame={frame} />
          <SensorBars frame={frame} />
        </div>
      </div>

      <p className="muted text-xs">
        Research aid · QMLKit is a Problem Statement 26139 prototype. Screening outputs are experimental and this is
        not a validated clinical device.
      </p>
    </div>
  );
}
