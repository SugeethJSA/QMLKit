"use client";

interface ConfusionMatrixProps {
  matrix: number[][]; // [[TN, FP], [FN, TP]]
}

export default function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  if (!matrix || matrix.length < 2) return null;

  const tn = matrix[0][0] || 0;
  const fp = matrix[0][1] || 0;
  const fn = matrix[1][0] || 0;
  const tp = matrix[1][1] || 0;
  const total = tn + fp + fn + tp || 1;
  const pct = (n: number) => `${((n / total) * 100).toFixed(1)}% of cohort`;

  const cells = [
    { label: "True negative · Healthy", value: tn, hint: pct(tn), className: "chip-good" },
    { label: "False positive · Type I", value: fp, hint: pct(fp), className: "chip-warn" },
    { label: "False negative · Missed", value: fn, hint: pct(fn), className: "chip-warn" },
    { label: "True positive · Malignant", value: tp, hint: pct(tp), className: "" },
  ];

  return (
    <div className="stack">
      <div className="row text-[13px]">
        <span className="font-bold">Confusion matrix</span>
        <span className="muted num">Held-out test · N = {total}</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {cells.map((cell) => (
          <div key={cell.label} className={`surface-sm flex flex-col gap-1 ${cell.className}`}>
            <span className="label !text-current !opacity-80">{cell.label}</span>
            <span className="num text-2xl font-extrabold">{cell.value}</span>
            <span className="num text-xs opacity-75">{cell.hint}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
