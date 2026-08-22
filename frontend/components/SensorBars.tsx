"use client";

import type { Frame } from "./KennelDiagram";

export default function SensorBars({ frame }: { frame: Frame | null }) {
  const rows: { label: string; value: number; unit: string; max: number }[] = [
    { label: "US bottom", value: frame?.us?.bottom ?? 0, unit: "cm", max: 200 },
    { label: "US top", value: frame?.us?.top ?? 0, unit: "cm", max: 200 },
    { label: "FSR FL", value: frame?.fsr?.[0] ?? 0, unit: "", max: 1024 },
    { label: "FSR FR", value: frame?.fsr?.[1] ?? 0, unit: "", max: 1024 },
    { label: "FSR RL", value: frame?.fsr?.[2] ?? 0, unit: "", max: 1024 },
    { label: "FSR RR", value: frame?.fsr?.[3] ?? 0, unit: "", max: 1024 },
  ];
  return (
    <div className="panel p-4">
      <h3 className="font-semibold mb-3">Slow channels</h3>
      <div className="space-y-2">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2 text-xs">
            <span className="w-20 text-gray-400">{r.label}</span>
            <div className="flex-1 h-3 bg-[#1f2937] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-100"
                style={{
                  width: `${Math.min(100, (Math.max(0, r.value) / r.max) * 100)}%`,
                  background: "linear-gradient(90deg,#4f46e5,#a855f7)",
                }}
              />
            </div>
            <span className="w-16 text-right tabular-nums text-gray-300">
              {(frame ? r.value : 0).toFixed(0)}
              {r.unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
