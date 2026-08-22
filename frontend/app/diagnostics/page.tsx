"use client";

import { useState } from "react";
import PredictionCard, { type Diagnostic } from "@/components/PredictionCard";
import { useTelemetrySocket } from "@/lib/ws";

export default function DiagnosticsPage() {
  const [diag, setDiag] = useState<Diagnostic | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const connected = useTelemetrySocket<Diagnostic>("/ws/diagnostic", (d) => {
    setDiag(d);
    if (typeof d.probability_cancer === "number") {
      setHistory((h) => [...h.slice(-59), d.probability_cancer!]);
    }
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 text-sm text-gray-400">
        <span className={`w-3 h-3 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`} />
        {connected ? "Connected to diagnostic stream" : "Disconnected"}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PredictionCard diag={diag} />

        <div className="panel p-6">
          <h3 className="font-semibold mb-4">Risk history (last 60 windows)</h3>
          <svg viewBox="0 0 300 120" className="w-full">
            <line x1={0} y1={60} x2={300} y2={60} stroke="#374151" strokeDasharray="4 4" />
            <text x={4} y={14} fontSize={9} fill="#9ca3af">100%</text>
            <text x={4} y={114} fontSize={9} fill="#9ca3af">0%</text>
            {(() => {
              const n = Math.max(history.length, 1);
              const pts = history
                .map((v, i) => `${(i / (n - 1 || 1)) * 296 + 2},${118 - v * 116}`)
                .join(" ");
              return (
                <polyline points={pts} fill="none" stroke="#a855f7" strokeWidth={2} />
              );
            })()}
          </svg>
          <p className="text-xs text-gray-500 mt-2">
            One point per completed window (~{`batch_size`} frames). Dashed line =
            decision boundary.
          </p>
        </div>
      </div>

      <div className="panel p-4 text-xs text-gray-500">
        Kennel diagnostics are a research screening aid derived from body-worn IMU
        micro-movements. They are not veterinary diagnoses.
      </div>
    </div>
  );
}
