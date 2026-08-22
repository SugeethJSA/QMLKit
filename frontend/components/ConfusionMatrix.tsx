"use client";

import React from "react";

interface ConfusionMatrixProps {
  matrix: number[][]; // [[TN, FP], [FN, TP]]
}

export default function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  if (!matrix || matrix.length < 2) {
    return null;
  }

  const tn = matrix[0][0] || 0;
  const fp = matrix[0][1] || 0;
  const fn = matrix[1][0] || 0;
  const tp = matrix[1][1] || 0;
  const total = tn + fp + fn + tp || 1;

  const tnPct = ((tn / total) * 100).toFixed(1);
  const fpPct = ((fp / total) * 100).toFixed(1);
  const fnPct = ((fn / total) * 100).toFixed(1);
  const tpPct = ((tp / total) * 100).toFixed(1);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span className="font-semibold text-gray-300">Confusion Matrix Heatmap</span>
        <span className="text-gray-400">Held-Out Test (N = {total})</span>
      </div>

      <div className="grid grid-cols-2 gap-2 p-3 bg-[#07090e]/80 border border-white/5 rounded-xl text-center font-mono">
        {/* True Negative */}
        <div className="p-3 rounded-lg border border-emerald-500/30 bg-emerald-950/20 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase tracking-wider text-emerald-400 font-sans font-semibold">True Negative (Healthy)</span>
          <span className="text-xl font-bold text-emerald-300 mt-1">{tn}</span>
          <span className="text-[11px] text-emerald-500/80 mt-0.5">{tnPct}% of cohort</span>
        </div>

        {/* False Positive */}
        <div className="p-3 rounded-lg border border-rose-500/20 bg-rose-950/10 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase tracking-wider text-rose-400 font-sans font-semibold">False Positive (Type I)</span>
          <span className="text-xl font-bold text-rose-300 mt-1">{fp}</span>
          <span className="text-[11px] text-rose-400/80 mt-0.5">{fpPct}% of cohort</span>
        </div>

        {/* False Negative */}
        <div className="p-3 rounded-lg border border-amber-500/20 bg-amber-950/10 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase tracking-wider text-amber-400 font-sans font-semibold">False Negative (Missed)</span>
          <span className="text-xl font-bold text-amber-300 mt-1">{fn}</span>
          <span className="text-[11px] text-amber-400/80 mt-0.5">{fnPct}% of cohort</span>
        </div>

        {/* True Positive */}
        <div className="p-3 rounded-lg border border-purple-500/30 bg-purple-950/20 flex flex-col items-center justify-center">
          <span className="text-[10px] uppercase tracking-wider text-purple-400 font-sans font-semibold">True Positive (Malignant)</span>
          <span className="text-xl font-bold text-purple-300 mt-1">{tp}</span>
          <span className="text-[11px] text-purple-400/80 mt-0.5">{tpPct}% of cohort</span>
        </div>
      </div>
    </div>
  );
}
