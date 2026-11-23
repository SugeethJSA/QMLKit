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
    <div className="surface p-5 stack gap-3">
      <div className="stack gap-1">
        <h3 className="section-title">Slow channels</h3>
        <p className="muted text-sm">Ultrasonic distance and corner-pad load, as reported by the node.</p>
      </div>
      {!frame ? (
        <div className="stack gap-1">
          <p className="muted text-sm">Waiting for telemetry…</p>
          <p className="muted text-xs">Channel readings appear with the first frame.</p>
        </div>
      ) : (
        <div className="stack gap-2" role="list" aria-label="Slow channel readings">
          {rows.map((r) => (
            <div key={r.label} role="listitem" className="row items-center gap-3 text-sm">
              <span className="label w-20">{r.label}</span>
              <div
                className="flex-1 h-2"
                style={{ backgroundColor: "#e8e1d4", borderRadius: 9999, overflow: "hidden" }}
                aria-hidden="true"
              >
                <div
                  className="h-full"
                  style={{
                    width: `${Math.min(100, (Math.max(0, r.value) / r.max) * 100)}%`,
                    backgroundColor: "#2f5fd2",
                    borderRadius: 9999,
                  }}
                />
              </div>
              <span className="num w-16 text-right">
                {r.value.toFixed(0)}
                {r.unit}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
