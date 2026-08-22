"use client";

import React from "react";

interface LossChartProps {
  lossHistory: number[];
  title?: string;
  color?: string;
}

export default function LossChart({
  lossHistory,
  title = "Convergence Loss Trajectory",
  color = "#a855f7",
}: LossChartProps) {
  if (!lossHistory || lossHistory.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-6 text-center text-gray-500 border border-dashed border-gray-800 rounded-xl h-48">
        <span className="text-sm">Loss trajectory unavailable for analytical closed-form models (e.g. QSVM / SVM).</span>
      </div>
    );
  }

  const minLoss = Math.min(...lossHistory);
  const maxLoss = Math.max(...lossHistory);
  const range = maxLoss - minLoss || 1;

  const width = 450;
  const height = 150;
  const padding = 30;

  const points = lossHistory.map((val, idx) => {
    const x = padding + (idx / Math.max(1, lossHistory.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((val - minLoss) / range) * (height - 2 * padding);
    return `${x},${y}`;
  });

  const polylineStr = points.join(" ");
  const areaPoints = `${padding},${height - padding} ${polylineStr} ${width - padding},${height - padding}`;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span className="font-semibold text-gray-300">{title}</span>
        <span>
          Initial: <strong className="text-gray-200">{lossHistory[0].toFixed(4)}</strong> → Final:{" "}
          <strong className="text-purple-400">{lossHistory[lossHistory.length - 1].toFixed(4)}</strong>
        </span>
      </div>

      <div className="relative bg-[#07090e]/80 border border-white/5 rounded-xl p-2">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-40 overflow-visible">
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.35" />
              <stop offset="100%" stopColor={color} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Horizontal Grid lines */}
          <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.1)" />

          {/* Area fill */}
          <polygon points={areaPoints} fill="url(#lossGrad)" />

          {/* Line */}
          <polyline points={polylineStr} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

          {/* Data Points */}
          {lossHistory.map((val, idx) => {
            const x = padding + (idx / Math.max(1, lossHistory.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((val - minLoss) / range) * (height - 2 * padding);
            return (
              <circle
                key={idx}
                cx={x}
                cy={y}
                r="3"
                className="fill-purple-400 stroke-[#07090e] hover:r-5 transition-all cursor-pointer"
              >
                <title>{`Epoch ${idx + 1}: Loss = ${val.toFixed(4)}`}</title>
              </circle>
            );
          })}

          {/* Y Axis Labels */}
          <text x={padding - 6} y={padding + 4} textAnchor="end" fontSize="9" fill="#9ca3af">
            {maxLoss.toFixed(3)}
          </text>
          <text x={padding - 6} y={height - padding + 4} textAnchor="end" fontSize="9" fill="#9ca3af">
            {minLoss.toFixed(3)}
          </text>

          {/* X Axis Labels */}
          <text x={padding} y={height - 10} textAnchor="middle" fontSize="9" fill="#6b7280">
            Ep 1
          </text>
          <text x={width - padding} y={height - 10} textAnchor="middle" fontSize="9" fill="#6b7280">
            Ep {lossHistory.length}
          </text>
        </svg>
      </div>
    </div>
  );
}
