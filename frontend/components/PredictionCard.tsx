"use client";

import { AlertTriangle, CheckCircle2, Cpu, ShieldAlert, Sparkles } from "lucide-react";

export interface Diagnostic {
  type: string;
  status: "ok" | "uncertain" | "untrained";
  probability_cancer?: number;
  smoothed_probability?: number;
  confidence_threshold?: number;
  detail?: string;
}

function ModernConfidenceGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const strokeDash = Math.min(314, Math.max(0, value * 314));
  const isHighRisk = value >= 0.5;
  const strokeColor = isHighRisk ? "url(#roseGrad)" : "url(#emeraldGrad)";

  return (
    <div className="relative flex flex-col items-center justify-center">
      <svg viewBox="0 0 140 140" className="w-44 h-44 drop-shadow-lg">
        <defs>
          <linearGradient id="roseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
        </defs>

        {/* Outer subtle ring */}
        <circle cx={70} cy={70} r={55} fill="none" stroke="rgba(255, 255, 255, 0.06)" strokeWidth={10} />

        {/* Animated Active Arc */}
        <circle
          cx={70}
          cy={70}
          r={55}
          fill="none"
          stroke={strokeColor}
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={`${strokeDash} 345`}
          transform="rotate(-90 70 70)"
          className="transition-all duration-700 ease-out"
        />

        {/* Center Labels */}
        <text x={70} y={66} textAnchor="middle" fontSize={26} fontWeight="800" fill="#ffffff" fontFamily="monospace">
          {pct}%
        </text>
        <text x={70} y={84} textAnchor="middle" fontSize={9} fontWeight="600" fill="#9ca3af" letterSpacing="0.08em">
          MALIGNANCY RISK
        </text>
      </svg>
    </div>
  );
}

export default function PredictionCard({ diag }: { diag: Diagnostic | null }) {
  if (!diag) {
    return (
      <div className="panel p-8 text-center text-gray-500 flex flex-col items-center justify-center space-y-2">
        <Sparkles className="w-6 h-6 text-purple-400/50 animate-pulse" />
        <span className="text-xs">Waiting for telemetry diagnostic window...</span>
      </div>
    );
  }

  if (diag.status === "untrained") {
    return (
      <div className="panel p-6 space-y-3 border-amber-500/30">
        <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span>Model Awaiting Calibration</span>
        </div>
        <p className="text-xs text-gray-400 leading-relaxed">{diag.detail}</p>
        <p className="text-[11px] text-gray-500">
          Navigate to the <strong className="text-purple-400">Training Studio</strong> to train a Quantum or Classical classifier directly from the browser.
        </p>
      </div>
    );
  }

  const prob = diag.probability_cancer ?? 0;
  const uncertain = diag.status === "uncertain";
  const isHighRisk = prob >= 0.5;

  return (
    <div className={`panel p-6 space-y-4 border-l-4 ${isHighRisk ? "border-l-rose-500" : "border-l-emerald-500"}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isHighRisk ? (
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          )}
          <h3 className="text-sm font-bold text-white">Canine Micro-Movement Screening</h3>
        </div>

        {uncertain && (
          <span className="text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded-full font-semibold">
            Low Confidence
          </span>
        )}
      </div>

      <ModernConfidenceGauge value={prob} />

      <div className="grid grid-cols-2 gap-3 text-center">
        <div className="bg-[#0b0f19] border border-white/5 rounded-xl p-3">
          <span className="text-gray-400 text-[10px] uppercase tracking-wider block">EMA Smoothed</span>
          <span className="text-base font-bold text-white font-mono mt-0.5 block">
            {((diag.smoothed_probability ?? prob) * 100).toFixed(1)}%
          </span>
        </div>

        <div className="bg-[#0b0f19] border border-white/5 rounded-xl p-3">
          <span className="text-gray-400 text-[10px] uppercase tracking-wider block">Decision Cutoff</span>
          <span className="text-base font-bold text-purple-400 font-mono mt-0.5 block">
            {((diag.confidence_threshold ?? 0.5) * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <p className="text-[11px] text-gray-500 text-center">
        Screening signal derived from bio-electronic &amp; collar micro-movements · Research aid.
      </p>
    </div>
  );
}
