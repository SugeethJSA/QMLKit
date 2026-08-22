"use client";

/** Top-view kennel schematic: FSR corner loads + IR occupancy + ultrasonic. */

const CORNERS = [
  { key: "FL", x: 60, y: 40, fsr: 0, ir: 0 },
  { key: "FR", x: 260, y: 40, fsr: 1, ir: 1 },
  { key: "RL", x: 60, y: 260, fsr: 2, ir: 2 },
  { key: "RR", x: 260, y: 260, fsr: 3, ir: 3 },
] as const;

export interface Frame {
  ts_ms: number;
  seq: number;
  state: string;
  fsr: number[];
  ir: number[];
  us: { bottom: number; top: number };
  acc: number[];
  gyr: number[];
}

const STATE_COLORS: Record<string, string> = {
  BOOT: "#6b7280",
  CALIBRATE: "#f59e0b",
  IDLE: "#10b981",
  OCCUPIED: "#3b82f6",
  SNIFF: "#a855f7",
  COOLDOWN: "#6b7280",
};

function loadColor(v: number): string {
  const t = Math.min(1, v / 1024);
  return `rgba(99, 102, 241, ${0.15 + 0.85 * t})`;
}

export default function KennelDiagram({ frame }: { frame: Frame | null }) {
  if (!frame) {
    return (
      <div className="panel h-[320px] flex items-center justify-center text-gray-500">
        Waiting for telemetry…
      </div>
    );
  }
  const state = frame.state ?? "IDLE";
  return (
    <div className="panel p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold">Kennel Top View</h3>
        <span
          className="text-xs px-2 py-1 rounded-full font-medium"
          style={{ background: STATE_COLORS[state] ?? "#6b7280" }}
        >
          {state}
        </span>
      </div>
      <svg viewBox="0 0 320 300" className="w-full">
        {/* kennel walls */}
        <rect x={30} y={20} width={260} height={260} rx={16} fill="#0b0f19" stroke="#374151" strokeWidth={3} />
        {/* center sample */}
        <circle cx={160} cy={150} r={22} fill="#111827" stroke="#818cf8" strokeWidth={2} />
        <text x={160} y={155} textAnchor="middle" fontSize={11} fill="#c7d2fe">sample</text>

        {CORNERS.map((c) => (
          <g key={c.key}>
            {/* FSR pad */}
            <rect x={c.x - 26} y={c.y - 26} width={52} height={52} rx={10}
              fill={loadColor(frame.fsr?.[c.fsr] ?? 0)} stroke="#4b5563" />
            <text x={c.x} y={c.y - 8} textAnchor="middle" fontSize={11} fill="#d1d5db">{c.key}</text>
            <text x={c.x} y={c.y + 8} textAnchor="middle" fontSize={13} fontWeight="bold" fill="#fff">
              {Math.round(frame.fsr?.[c.fsr] ?? 0)}
            </text>
            {/* IR dot */}
            <circle cx={c.x + 34} cy={c.y + 34} r={6}
              fill={(frame.ir?.[c.ir] ?? 0) > 0 ? "#ef4444" : "#374151"} />
          </g>
        ))}

        {/* top-front IR sensors */}
        {[0, 1].map((i) => (
          <g key={`top-${i}`}>
            <circle cx={i === 0 ? 70 : 250} cy={34} r={7}
              fill={(frame.ir?.[4 + i] ?? 0) > 0 ? "#f59e0b" : "#374151"} />
            <text x={i === 0 ? 70 : 250} y={14} textAnchor="middle" fontSize={10} fill="#9ca3af">
              IR top {i === 0 ? "L" : "R"}
            </text>
          </g>
        ))}

        {/* ultrasonic readouts */}
        <text x={160} y={286} textAnchor="middle" fontSize={12} fill="#9ca3af">
          US bottom {frame.us?.bottom?.toFixed?.(0) ?? "-"} cm · US top {frame.us?.top?.toFixed?.(0) ?? "-"} cm
        </text>
      </svg>
      <p className="text-xs text-gray-500 mt-2">
        Pad opacity = FSR pressure · red dots = IR triggered · amber = head detected
      </p>
    </div>
  );
}
