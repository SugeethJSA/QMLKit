"use client";

import { useState } from "react";
import KennelDiagram, { type Frame } from "@/components/KennelDiagram";
import ImuWaveforms from "@/components/ImuWaveforms";
import SensorBars from "@/components/SensorBars";
import { useTelemetrySocket } from "@/lib/ws";

export default function DashboardPage() {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [count, setCount] = useState(0);
  const connected = useTelemetrySocket<Frame>("/ws/stream", (data) => {
    setFrame(data);
    setCount((c) => c + 1);
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span
          className={`w-3 h-3 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`}
        />
        <span className="text-sm text-gray-400">
          {connected ? "Live telemetry" : "Disconnected"} · {count} frames received
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <KennelDiagram frame={frame} />
        <div className="lg:col-span-2 space-y-4">
          <ImuWaveforms frame={frame} />
          <SensorBars frame={frame} />
        </div>
      </div>
    </div>
  );
}
