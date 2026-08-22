"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Cpu,
  Eye,
  Flame,
  Layers,
  Radio,
  RefreshCw,
  Sparkles,
  Wifi,
  Zap,
} from "lucide-react";
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

  // Compute RMS Acceleration
  const accRms = frame?.acc
    ? Math.sqrt(frame.acc.reduce((sum, val) => sum + val * val, 0) / 3).toFixed(2)
    : "0.00";

  // Compute FSR total pressure
  const totalFsr = frame?.fsr
    ? frame.fsr.reduce((sum, val) => sum + val, 0).toFixed(1)
    : "0.0";

  return (
    <div className="space-y-6">
      {/* Top Telemetry Summary Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="panel p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-gray-400 uppercase tracking-wider block font-semibold">
              Live Stream Status
            </span>
            <div className="flex items-center gap-2 mt-1">
              <span className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-rose-500"}`} />
              <span className="font-bold text-sm text-white">
                {connected ? "Connected" : "Disconnected"}
              </span>
            </div>
            <span className="text-[10px] text-gray-400 mt-0.5 block font-mono">
              {state?.stream_source ?? "Local Sim"} · {fps} FPS
            </span>
          </div>
          <div className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
            <Wifi className="w-4 h-4 text-purple-400" />
          </div>
        </div>

        <div className="panel p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-gray-400 uppercase tracking-wider block font-semibold">
              Total Ingested Frames
            </span>
            <span className="text-xl font-bold text-white font-mono mt-1 block">
              {count.toLocaleString()}
            </span>
            <span className="text-[10px] text-indigo-400 mt-0.5 block">
              100 Hz Native Sensor Sampling
            </span>
          </div>
          <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
        </div>

        <div className="panel p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-gray-400 uppercase tracking-wider block font-semibold">
              Motion Energy (RMS)
            </span>
            <span className="text-xl font-bold text-white font-mono mt-1 block">
              {accRms} <span className="text-xs font-normal text-gray-400">g</span>
            </span>
            <span className="text-[10px] text-emerald-400 mt-0.5 block">
              Micro-movement steady
            </span>
          </div>
          <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <Zap className="w-4 h-4 text-emerald-400" />
          </div>
        </div>

        <div className="panel p-4 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-gray-400 uppercase tracking-wider block font-semibold">
              Model Diagnostic State
            </span>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="font-bold text-sm text-purple-300 capitalize">
                {state?.model_status ?? "Ready"}
              </span>
            </div>
            <Link href="/train" className="text-[10px] text-purple-400 hover:text-purple-300 underline mt-0.5 block">
              Train / Tune Hyperparameters →
            </Link>
          </div>
          <div className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-purple-400" />
          </div>
        </div>
      </div>

      {/* Main Visualizer Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Kennel Diagram (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <KennelDiagram frame={frame} />

          {/* Quick Action Navigation Card */}
          <div className="panel p-5 space-y-3 bg-gradient-to-br from-[#111827] to-[#1e1b4b]/40 border-purple-500/20">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <h4 className="text-sm font-bold text-white">Quantum Screening Suite</h4>
            </div>
            <p className="text-xs text-gray-300">
              Run quantum kernel Hilbert space classification on canine olfactory VOCs and collar micro-movements.
            </p>
            <div className="grid grid-cols-2 gap-2 pt-1">
              <Link
                href="/diagnostics"
                className="px-3 py-2 rounded-xl bg-purple-600/30 hover:bg-purple-600/40 border border-purple-500/40 text-purple-200 text-xs font-semibold text-center transition flex items-center justify-center gap-1"
              >
                <span>Diagnostics</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
              <Link
                href="/train"
                className="px-3 py-2 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/40 border border-indigo-500/40 text-indigo-200 text-xs font-semibold text-center transition flex items-center justify-center gap-1"
              >
                <span>Train Lab</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
          </div>
        </div>

        {/* Right: Waveforms and Sensor Reactivity (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          <ImuWaveforms frame={frame} />
          <SensorBars frame={frame} />
        </div>
      </div>
    </div>
  );
}
