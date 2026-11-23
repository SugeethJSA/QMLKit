"use client";

import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Eye,
  FileText,
  FlaskConical,
  Play,
  RotateCw,
} from "lucide-react";
import PredictionCard, { type Diagnostic } from "@/components/PredictionCard";
import BiomarkerWaterfall, { type Biomarker } from "@/components/BiomarkerWaterfall";
import { useTelemetrySocket } from "@/lib/ws";
import { postJson } from "@/lib/api";

interface DetailedPrediction {
  sample_id: string;
  prediction: string;
  cancer_probability: number;
  quantum_confidence_pct: number;
  top_biomarkers: Biomarker[];
  biochemical_pathways: Record<string, number>;
  clinical_summary: string;
}

const PRESETS = [
  { id: "lung_positive", name: "Lung Cancer (Hexanal / Benzaldehyde High)", type: "Malignant" },
  { id: "healthy_control", name: "Healthy Control (Physiological Normal)", type: "Healthy" },
  { id: "breast_positive", name: "Breast Cancer (Heptanal / Ketone Surge)", type: "Malignant" },
  { id: "colorectal_positive", name: "Colorectal Cancer (DMDS / Aromatic High)", type: "Malignant" },
];

const COLOR = {
  good: "#2f7d5b",
  bad: "#b23a2f",
  action: "#2f5fd2",
  warn: "#a9681f",
  line: "#e8e1d4",
  goodTint: "#f4f8f6",
  badTint: "#fbf3f1",
};

export default function DiagnosticsPage() {
  const [diag, setDiag] = useState<Diagnostic | null>(null);
  // Honest default: no seeded history. Points only arrive from real messages/responses.
  const [history, setHistory] = useState<number[]>([]);
  const [selectedPreset, setSelectedPreset] = useState("lung_positive");
  const [sampleId, setSampleId] = useState("SMPL_ONC_8892");
  const [patientAge, setPatientAge] = useState(62);
  const [smokingStatus, setSmokingStatus] = useState("Former");
  const [isRunning, setIsRunning] = useState(false);
  const [detailedResult, setDetailedResult] = useState<DetailedPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Live WebSocket diagnostic listener
  const connected = useTelemetrySocket<Diagnostic>("/ws/diagnostic", (d) => {
    setDiag(d);
    if (typeof d.probability_cancer === "number") {
      setHistory((h) => [...h.slice(-59), d.probability_cancer!]);
    }
  });

  function getSensorPresetValues(preset: string): number[] {
    const vals: number[] = [];
    for (let i = 0; i < 64; i++) {
      if (preset === "lung_positive") {
        vals.push(Math.sin(i * 0.3) * 0.8 + 1.8 + Math.random() * 0.15);
      } else if (preset === "healthy_control") {
        vals.push(Math.sin(i * 0.3) * 0.2 + 0.4 + Math.random() * 0.08);
      } else if (preset === "breast_positive") {
        vals.push(Math.cos(i * 0.25) * 0.7 + 1.6 + Math.random() * 0.15);
      } else {
        vals.push(Math.sin(i * 0.2) * 0.9 + 1.5 + Math.random() * 0.12);
      }
    }
    return vals;
  }

  async function runQuantumDiagnostic() {
    setIsRunning(true);
    setError(null);
    try {
      const sensorValues = getSensorPresetValues(selectedPreset);
      const res = await postJson<DetailedPrediction>("/api/v1/predict?deep_explain=true", {
        sample_id: sampleId,
        patient_age: patientAge,
        patient_sex: "M",
        smoking_status: smokingStatus,
        sensor_readings: sensorValues,
      });
      setDetailedResult(res);
      setHistory((h) => [...h.slice(-59), res.cancer_probability]);
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      setError(
        `The screening request could not be completed (${reason}). Check that the API host is reachable — append ?api=http://host:port to this URL when the console is opened from another machine — then try again.`
      );
    } finally {
      setIsRunning(false);
    }
  }

  const riskPct = detailedResult
    ? Math.min(100, Math.max(0, detailedResult.cancer_probability * 100))
    : 0;
  const highRisk = detailedResult ? detailedResult.cancer_probability >= 0.5 : false;
  const resultTone = highRisk ? COLOR.bad : COLOR.good;
  const resultTint = highRisk ? COLOR.badTint : COLOR.goodTint;
  const chartPoints =
    history.length >= 2
      ? history
          .map((v, i) => `${(i / (history.length - 1)) * 296 + 2},${118 - v * 116}`)
          .join(" ")
      : "";

  return (
    <div className="stack gap-6">
      <div className="surface row items-center justify-between gap-4">
        <div className="stack gap-1">
          <div className="row items-center gap-2">
            <Eye width={20} height={20} style={{ color: COLOR.action }} aria-hidden="true" />
            <h1 className="section-title">Oncological VOC screening &amp; explainability</h1>
          </div>
          <p className="muted">
            BioZZ quantum-kernel attribution over volatile organic biomarker vectors. Research aid — not a
            validated clinical device.
          </p>
        </div>
        <div className="row items-center gap-2">
          <span className={`chip ${connected ? "chip-good" : ""}`}>
            {connected ? "Live stream connected" : "On-demand mode"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls and specimen input */}
        <div className="lg:col-span-4 stack gap-4">
          <div className="surface stack gap-4">
            <h2 className="section-title row items-center gap-2">
              <FlaskConical width={18} height={18} style={{ color: COLOR.action }} aria-hidden="true" />
              Specimen input
            </h2>

            <div className="stack gap-1">
              <label className="label" htmlFor="sample-id">
                Specimen identifier
              </label>
              <input
                id="sample-id"
                type="text"
                value={sampleId}
                onChange={(e) => setSampleId(e.target.value)}
                className="field num"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="stack gap-1">
                <label className="label" htmlFor="patient-age">
                  Patient age
                </label>
                <input
                  id="patient-age"
                  type="number"
                  value={patientAge}
                  onChange={(e) => setPatientAge(parseInt(e.target.value))}
                  className="field num"
                />
              </div>

              <div className="stack gap-1">
                <label className="label" htmlFor="smoking-status">
                  Smoking status
                </label>
                <select
                  id="smoking-status"
                  value={smokingStatus}
                  onChange={(e) => setSmokingStatus(e.target.value)}
                  className="field"
                >
                  <option value="Former">Former smoker</option>
                  <option value="Current">Current smoker</option>
                  <option value="Never">Never smoker</option>
                </select>
              </div>
            </div>

            <div className="stack gap-1">
              <div className="row items-center justify-between gap-2">
                <label className="label" htmlFor="sensor-preset">
                  Sensor preset
                </label>
                <span className="chip chip-warn">Synthetic preset</span>
              </div>
              <select
                id="sensor-preset"
                value={selectedPreset}
                onChange={(e) => setSelectedPreset(e.target.value)}
                className="field"
              >
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <p className="muted" style={{ fontSize: 12 }}>
                Preset vectors are generated for demonstration. They are not readings from a kennel node.
              </p>
            </div>

            <button
              type="button"
              onClick={runQuantumDiagnostic}
              disabled={isRunning}
              className="btn-primary row items-center justify-center gap-2"
            >
              {isRunning ? (
                <>
                  <RotateCw width={16} height={16} aria-hidden="true" />
                  <span>Evaluating…</span>
                </>
              ) : (
                <>
                  <Play width={16} height={16} aria-hidden="true" />
                  <span>Run screening</span>
                </>
              )}
            </button>
          </div>

          <div className="surface-sm stack gap-2">
            <div className="row items-center gap-2">
              <Cpu width={16} height={16} style={{ color: COLOR.action }} aria-hidden="true" />
              <h2 className="label" style={{ fontWeight: 600 }}>
                How attribution works
              </h2>
            </div>
            <p className="muted">
              Sensor transients are encoded as rotation angles and entangling gates, then projected into a
              higher-dimensional feature space. Attribution ranks which input dimensions moved the model
              output.
            </p>
          </div>

          {error ? (
            <div
              className="surface stack gap-2"
              role="alert"
              style={{ border: `1px solid ${COLOR.bad}`, backgroundColor: COLOR.badTint }}
            >
              <div className="row items-center gap-2">
                <AlertTriangle width={18} height={18} style={{ color: COLOR.bad }} aria-hidden="true" />
                <h2 className="label" style={{ color: COLOR.bad, fontWeight: 600 }}>
                  Screening could not run
                </h2>
              </div>
              <p className="muted">{error}</p>
              <button type="button" className="btn-quiet" style={{ alignSelf: "flex-start" }} onClick={runQuantumDiagnostic}>
                Try again
              </button>
            </div>
          ) : null}
        </div>

        {/* Results */}
        <div className="lg:col-span-8 stack gap-6">
          {detailedResult ? (
            <div className="stack gap-6">
              <div
                className="surface stack gap-4"
                style={{ border: `1px solid ${resultTone}`, backgroundColor: resultTint }}
              >
                <div className="row items-center justify-between gap-3">
                  <span className={`chip ${highRisk ? "chip-bad" : "chip-good"}`}>
                    {detailedResult.prediction}
                  </span>
                  <span className="muted num">ID {detailedResult.sample_id}</span>
                </div>

                <div className="row items-end justify-between gap-4">
                  <div className="stack gap-1">
                    <span className="num" style={{ color: resultTone, fontSize: 40, lineHeight: 1 }}>
                      {riskPct.toFixed(1)}%
                    </span>
                    <span className="label">Malignancy risk index</span>
                  </div>
                  <div className="stack gap-1">
                    <span className="num" style={{ color: COLOR.action }}>
                      {detailedResult.quantum_confidence_pct}%
                    </span>
                    <span className="label">Quantum confidence</span>
                  </div>
                </div>

                <div
                  className="progress"
                  role="progressbar"
                  aria-label="Malignancy risk index"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Number(riskPct.toFixed(1))}
                >
                  <div
                    className="progress-fill"
                    style={{ width: `${riskPct}%`, backgroundColor: resultTone }}
                  />
                </div>

                <p className="muted" style={{ fontSize: 12 }}>
                  Computed from a synthetic preset request. Research aid — not a clinical result.
                </p>
              </div>

              <div className="surface">
                <BiomarkerWaterfall
                  biomarkers={detailedResult.top_biomarkers}
                  pathways={detailedResult.biochemical_pathways}
                />
              </div>

              <div className="surface stack gap-2">
                <div className="row items-center gap-2">
                  <FileText width={16} height={16} style={{ color: COLOR.action }} aria-hidden="true" />
                  <h2 className="label" style={{ fontWeight: 600 }}>
                    Clinical review summary
                  </h2>
                </div>
                <p className="muted">{detailedResult.clinical_summary}</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <PredictionCard diag={diag} />

              <div className="surface stack gap-3">
                <div className="row items-center gap-2">
                  <Activity width={16} height={16} style={{ color: COLOR.action }} aria-hidden="true" />
                  <h2 className="section-title">Risk history</h2>
                </div>

                {history.length >= 2 ? (
                  <>
                    <svg
                      viewBox="0 0 300 120"
                      className="w-full"
                      role="img"
                      aria-label="Rolling malignancy risk across the most recent diagnostic windows"
                    >
                      <line x1={0} y1={60} x2={300} y2={60} stroke={COLOR.line} strokeDasharray="4 4" />
                      <text x={4} y={14} fontSize={9} fill="#6c675f">
                        100%
                      </text>
                      <text x={4} y={64} fontSize={8} fill="#6c675f">
                        Cutoff 50%
                      </text>
                      <text x={4} y={114} fontSize={9} fill="#6c675f">
                        0%
                      </text>
                      <polyline points={chartPoints} fill="none" stroke={COLOR.action} strokeWidth={2.5} />
                    </svg>
                    <p className="muted" style={{ fontSize: 12 }}>
                      Last {history.length} windows. Dashed line marks the 50% cutoff.
                    </p>
                  </>
                ) : (
                  <div className="surface-sm row items-center" style={{ border: "1px dashed #e8e1d4" }}>
                    <p className="muted">
                      No diagnostic windows recorded yet. Run a screening or wait for the live stream; points
                      appear here as results arrive.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
