"use client";

interface LossChartProps {
  lossHistory: number[];
  title?: string;
  color?: string;
}

export default function LossChart({
  lossHistory,
  title = "Convergence loss",
  color = "#2f5fd2",
}: LossChartProps) {
  if (!lossHistory || lossHistory.length === 0) {
    return (
      <div className="surface-sm muted flex min-h-40 items-center justify-center p-6 text-center text-sm">
        Loss trajectory unavailable for analytical closed-form models (e.g. QSVM / SVM).
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
    return { x, y, val };
  });

  return (
    <div className="stack">
      <div className="row text-[13px]">
        <span className="font-bold">{title}</span>
        <span className="muted num">
          First <strong className="text-[var(--ink)]">{lossHistory[0].toFixed(4)}</strong> · Last{" "}
          <strong style={{ color }}>{lossHistory[lossHistory.length - 1].toFixed(4)}</strong>
        </span>
      </div>

      <div className="surface-sm overflow-hidden">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-40 w-full" role="img" aria-label={title}>
          <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="var(--line)" strokeDasharray="3 3" />
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="var(--line)" strokeDasharray="3 3" />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--line-strong)" />

          <polygon
            points={`${padding},${height - padding} ${points.map((p) => `${p.x},${p.y}`).join(" ")} ${width - padding},${height - padding}`}
            fill={color}
            opacity="0.1"
          />
          <polyline
            points={points.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke={color}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {points.map((p, idx) => (
            <circle key={idx} cx={p.x} cy={p.y} r="3" fill={color}>
              <title>{`Epoch ${idx + 1}: loss ${p.val.toFixed(4)}`}</title>
            </circle>
          ))}

          <text x={padding - 6} y={padding + 4} textAnchor="end" fontSize="9" fill="var(--muted)">
            {maxLoss.toFixed(3)}
          </text>
          <text x={padding - 6} y={height - padding + 4} textAnchor="end" fontSize="9" fill="var(--muted)">
            {minLoss.toFixed(3)}
          </text>
          <text x={padding} y={height - 8} textAnchor="middle" fontSize="9" fill="var(--muted)">
            Epoch 1
          </text>
          <text x={width - padding} y={height - 8} textAnchor="middle" fontSize="9" fill="var(--muted)">
            Epoch {lossHistory.length}
          </text>
        </svg>
      </div>
    </div>
  );
}
