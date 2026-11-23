"use client";

import Link from "next/link";
import { AlertTriangle } from "lucide-react";

export interface Diagnostic {
  type: string;
  status: "ok" | "uncertain" | "untrained";
  probability_cancer?: number;
  smoothed_probability?: number;
  confidence_threshold?: number;
  detail?: string;
}

/* Solid palette values (no gradients, no glow) — kept local so the card stays
   correct even before the global semantic layer is applied. */
const COLOR = {
  good: "#2f7d5b",
  bad: "#b23a2f",
  action: "#2f5fd2",
  warn: "#a9681f",
  goodTint: "#f4f8f6",
  badTint: "#fbf3f1",
  actionTint: "#f4f7fd",
  warnTint: "#fbf6ee",
};

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex-1 stack gap-1">
      <span className="label">{label}</span>
      <span className="num" style={tone ? { color: tone } : undefined}>
        {value}
      </span>
    </div>
  );
}

export default function PredictionCard({ diag }: { diag: Diagnostic | null }) {
  if (!diag) {
    return (
      <div className="surface stack gap-3">
        <h2 className="section-title">Screening state</h2>
        <div className="surface-sm row items-center" style={{ border: "1px dashed #e8e1d4" }}>
          <p className="muted">
            Waiting for a diagnostic window from the kennel stream. Live results appear here automatically.
          </p>
        </div>
      </div>
    );
  }

  if (diag.status === "untrained") {
    return (
      <div
        className="surface stack gap-3"
        style={{ border: `1px solid ${COLOR.warn}`, backgroundColor: COLOR.warnTint }}
      >
        <div className="row items-center gap-2">
          <AlertTriangle width={18} height={18} style={{ color: COLOR.warn }} aria-hidden="true" />
          <h2 className="section-title">No trained model yet</h2>
        </div>
        <div>
          <span className="chip chip-warn">Untrained</span>
        </div>
        {diag.detail ? <p className="muted">{diag.detail}</p> : null}
        <p className="muted">
          This is the expected state before a classifier artifact is saved. Train a model in the Training
          Studio, then return here for a live screening signal.
        </p>
        <div>
          <Link href="/train" className="btn-quiet">
            Open Training Studio
          </Link>
        </div>
        <p className="muted" style={{ fontSize: 12 }}>
          Research aid only — not a clinical result.
        </p>
      </div>
    );
  }

  const prob = diag.probability_cancer ?? 0;
  const uncertain = diag.status === "uncertain";
  const elevated = prob >= 0.5;
  const tone = uncertain ? COLOR.action : elevated ? COLOR.bad : COLOR.good;
  const tint = uncertain ? COLOR.actionTint : elevated ? COLOR.badTint : COLOR.goodTint;
  const pct = (prob * 100).toFixed(1);
  const stateChip = uncertain ? "chip-warn" : elevated ? "chip-bad" : "chip-good";
  const stateLabel = uncertain ? "Uncertain" : elevated ? "Elevated" : "Within range";

  return (
    <div className="surface stack gap-4" style={{ border: `1px solid ${tone}`, backgroundColor: tint }}>
      <div className="row items-center justify-between gap-3">
        <h2 className="section-title">Canine micro-movement screening</h2>
        <span className={`chip ${stateChip}`}>{stateLabel}</span>
      </div>

      <div className="stack gap-1">
        <span className="num" style={{ color: tone, fontSize: 44, lineHeight: 1 }}>
          {pct}%
        </span>
        <span className="label">Malignancy risk index</span>
      </div>

      <div
        className="progress"
        role="progressbar"
        aria-label="Malignancy risk index"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(pct)}
      >
        <div
          className="progress-fill"
          style={{
            width: `${Math.min(100, Math.max(0, prob * 100))}%`,
            backgroundColor: tone,
          }}
        />
      </div>

      <div className="row items-center gap-3">
        <Stat label="EMA smoothed" value={`${((diag.smoothed_probability ?? prob) * 100).toFixed(1)}%`} />
        <div
          aria-hidden="true"
          style={{ width: 1, alignSelf: "stretch", backgroundColor: "#e8e1d4" }}
        />
        <Stat
          label="Decision cutoff"
          value={`${((diag.confidence_threshold ?? 0.5) * 100).toFixed(0)}%`}
          tone={COLOR.action}
        />
      </div>

      {diag.detail ? <p className="muted">{diag.detail}</p> : null}
      <p className="muted" style={{ fontSize: 12 }}>
        Research aid — signal derived from bio-electronic and collar micro-movement telemetry, not a
        clinical diagnosis.
      </p>
    </div>
  );
}
