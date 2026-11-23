"use client";

import { Dna } from "lucide-react";

export interface Biomarker {
  compound: string;
  importance_score: number;
  clinical_impact: string;
}

interface BiomarkerWaterfallProps {
  biomarkers: Biomarker[];
  pathways?: Record<string, number>;
}

/* Solid palette values (no gradients, no glow). */
const COLOR = {
  good: "#2f7d5b",
  bad: "#b23a2f",
  action: "#2f5fd2",
  line: "#e8e1d4",
};

export default function BiomarkerWaterfall({
  biomarkers,
  pathways,
}: BiomarkerWaterfallProps) {
  if (!biomarkers || biomarkers.length === 0) {
    return (
      <div className="surface-sm row items-center" style={{ border: "1px dashed #e8e1d4" }}>
        <p className="muted">No biomarker attribution computed for this window.</p>
      </div>
    );
  }

  const maxScore = Math.max(...biomarkers.map((b) => b.importance_score), 0.1);

  return (
    <div className="stack gap-5">
      <div className="stack gap-3">
        <div className="row items-center justify-between gap-3">
          <h2 className="section-title">Attributed VOC metabolites</h2>
          <span className="label">Relative importance</span>
        </div>

        <div className="stack gap-2">
          {biomarkers.map((b, idx) => {
            const isElevated = /elevated|risk/i.test(b.clinical_impact);
            const tone = isElevated ? COLOR.bad : COLOR.good;
            const widthPct = Math.min(100, Math.max(8, (b.importance_score / maxScore) * 100));

            return (
              <div key={idx} className="surface-sm stack gap-2">
                <div className="row items-center justify-between gap-3">
                  <div className="row items-center gap-2">
                    <span style={{ color: "#22201d", fontWeight: 600 }}>{b.compound}</span>
                    <span className={`chip ${isElevated ? "chip-bad" : "chip-good"}`}>
                      {b.clinical_impact}
                    </span>
                  </div>
                  <span className="num" style={{ color: tone }}>
                    {b.importance_score.toFixed(4)}
                  </span>
                </div>
                <div
                  className="progress"
                  role="progressbar"
                  aria-label={`${b.compound} importance`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(widthPct)}
                >
                  <div
                    className="progress-fill"
                    style={{ width: `${widthPct}%`, backgroundColor: tone }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {pathways && Object.keys(pathways).length > 0 ? (
        <div className="stack gap-3" style={{ borderTop: `1px solid ${COLOR.line}`, paddingTop: 16 }}>
          <div className="row items-center gap-2">
            <Dna width={16} height={16} style={{ color: COLOR.action }} aria-hidden="true" />
            <h2 className="section-title">Biochemical pathway dysregulation</h2>
          </div>

          <div className="stack">
            {Object.entries(pathways).map(([pathway, pct], i) => {
              const name = pathway.replace(/_/g, " ");
              const value = Math.min(100, Math.max(0, pct));
              return (
                <div
                  key={i}
                  className="row items-center gap-4"
                  style={{ padding: "10px 0", borderBottom: `1px solid ${COLOR.line}` }}
                >
                  <span className="label" style={{ flex: 1 }} title={name}>
                    {name}
                  </span>
                  <span className="num" style={{ color: COLOR.action, width: 44, textAlign: "right" }}>
                    {pct}%
                  </span>
                  <div className="w-16 sm:w-32">
                    <div
                      className="progress"
                      role="progressbar"
                      aria-label={`${name} share`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={value}
                    >
                      <div
                        className="progress-fill"
                        style={{ width: `${value}%`, backgroundColor: COLOR.action }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
