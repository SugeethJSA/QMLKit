"use client";

export interface Diagnostic {
  type: string;
  status: "ok" | "uncertain" | "untrained";
  probability_cancer?: number;
  smoothed_probability?: number;
  confidence_threshold?: number;
  detail?: string;
}

function ConfidenceRing({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const angle = Math.min(360, Math.max(0, value * 360));
  const color = value >= 0.5 ? "#ef4444" : "#10b981";
  return (
    <svg viewBox="0 0 120 120" className="w-40 h-40 mx-auto">
      <circle cx={60} cy={60} r={50} fill="none" stroke="#1f2937" strokeWidth={12} />
      <circle
        cx={60}
        cy={60}
        r={50}
        fill="none"
        stroke={color}
        strokeWidth={12}
        strokeLinecap="round"
        strokeDasharray={`${angle} 360`}
        transform="rotate(-90 60 60)"
      />
      <text x={60} y={58} textAnchor="middle" fontSize={22} fontWeight="bold" fill="#fff">
        {pct}%
      </text>
      <text x={60} y={78} textAnchor="middle" fontSize={10} fill="#9ca3af">
        risk index
      </text>
    </svg>
  );
}

export default function PredictionCard({ diag }: { diag: Diagnostic | null }) {
  if (!diag) {
    return (
      <div className="panel p-6 text-center text-gray-500">
        Waiting for the first diagnostic window…
      </div>
    );
  }

  if (diag.status === "untrained") {
    return (
      <div className="panel p-6">
        <h3 className="font-semibold mb-2 text-amber-400">Model not trained yet</h3>
        <p className="text-sm text-gray-400">{diag.detail}</p>
        <p className="text-xs text-gray-500 mt-3">
          Use the Data Lab to collect labelled sessions, then run{" "}
          <code className="bg-[#1f2937] px-1 rounded">python scripts/train_kennel_model.py</code>.
        </p>
      </div>
    );
  }

  const prob = diag.probability_cancer ?? 0;
  const uncertain = diag.status === "uncertain";

  return (
    <div className="panel p-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold">Micro-Movement Screening</h3>
        {uncertain && (
          <span className="text-xs bg-amber-500/20 text-amber-300 px-2 py-1 rounded-full">
            low confidence
          </span>
        )}
      </div>
      <ConfidenceRing value={prob} />
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="bg-[#0b0f19] border border-[#374151] rounded-lg p-3 text-center">
          <div className="text-gray-400 text-xs">EMA smoothed</div>
          <div className="text-lg font-bold tabular-nums">
            {((diag.smoothed_probability ?? prob) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-[#0b0f19] border border-[#374151] rounded-lg p-3 text-center">
          <div className="text-gray-400 text-xs">Threshold</div>
          <div className="text-lg font-bold tabular-nums">
            {((diag.confidence_threshold ?? 0.6) * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      <p className="mt-4 text-xs text-gray-500">
        Screening signal derived from collar micro-movements only — not a medical
        diagnosis.
      </p>
    </div>
  );
}
