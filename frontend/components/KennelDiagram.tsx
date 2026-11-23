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
  BOOT: "#6c675f",
  CALIBRATE: "#a9681f",
  IDLE: "#2f7d5b",
  OCCUPIED: "#2f5fd2",
  SNIFF: "#2f7d5b",
  COOLDOWN: "#6c675f",
};

function stateChipClass(state: string): string {
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

function loadColor(v: number): string {
  const t = Math.min(1, v / 1024);
  return `rgba(47, 95, 210, ${0.12 + 0.78 * t})`;
}

export default function KennelDiagram({ frame }: { frame: Frame | null }) {
  if (!frame) {
    return (
      <div className="surface p-5 stack gap-1">
        <h3 className="section-title">Kennel top view</h3>
        <p className="muted text-sm">Waiting for telemetry…</p>
        <p className="muted text-xs">The schematic appears as soon as the first frame arrives.</p>
      </div>
    );
  }
  const state = frame.state ?? "IDLE";
  return (
    <div className="surface p-5">
      <div className="row items-center justify-between gap-3 mb-3">
        <h3 className="section-title">Kennel top view</h3>
        <span className={stateChipClass(state)}>{state}</span>
      </div>
      <svg viewBox="0 0 320 300" className="w-full" role="img" aria-label={`Kennel schematic, current state ${state}`}>
        {/* kennel floor — warm ground with a hairline wall */}
        <rect x={30} y={20} width={260} height={260} rx={16} fill="#f6f1e8" stroke="#e8e1d4" strokeWidth={2} />

        {/* center sample point — the ring carries the meaningful state colour */}
        <circle cx={160} cy={150} r={22} fill="#ffffff" stroke={STATE_COLORS[state] ?? "#6c675f"} strokeWidth={2} />
        <text x={160} y={155} textAnchor="middle" fontSize={11} fill="#6c675f">sample</text>

        {CORNERS.map((c) => (
          <g key={c.key}>
            {/* FSR pad — opacity tracks pressure */}
            <rect
              x={c.x - 26}
              y={c.y - 26}
              width={52}
              height={52}
              rx={10}
              fill={loadColor(frame.fsr?.[c.fsr] ?? 0)}
              stroke="#e8e1d4"
            />
            <text x={c.x} y={c.y - 8} textAnchor="middle" fontSize={11} fill="#6c675f">{c.key}</text>
            <text x={c.x} y={c.y + 8} textAnchor="middle" fontSize={13} fontWeight="bold" fill="#22201d">
              {Math.round(frame.fsr?.[c.fsr] ?? 0)}
            </text>
            {/* IR dot */}
            <circle
              cx={c.x + 34}
              cy={c.y + 34}
              r={6}
              fill={(frame.ir?.[c.ir] ?? 0) > 0 ? "#b23a2f" : "#e8e1d4"}
            />
          </g>
        ))}

        {/* front IR sensors */}
        {[0, 1].map((i) => (
          <g key={`top-${i}`}>
            <circle
              cx={i === 0 ? 70 : 250}
              cy={34}
              r={7}
              fill={(frame.ir?.[4 + i] ?? 0) > 0 ? "#a9681f" : "#e8e1d4"}
            />
            <text x={i === 0 ? 70 : 250} y={14} textAnchor="middle" fontSize={10} fill="#6c675f">
              IR top {i === 0 ? "L" : "R"}
            </text>
          </g>
        ))}

        {/* ultrasonic readouts */}
        <text x={160} y={286} textAnchor="middle" fontSize={12} fill="#6c675f">
          US bottom {frame.us?.bottom?.toFixed?.(0) ?? "-"} cm · US top {frame.us?.top?.toFixed?.(0) ?? "-"} cm
        </text>
      </svg>
      <p className="muted text-xs mt-2">
        Pad shade = FSR load · red dot = IR triggered · amber = head detected
      </p>
    </div>
  );
}
