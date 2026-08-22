"use client";

import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Dna,
  Download,
  Eye,
  FileText,
  FlaskConical,
  Gauge,
  Play,
  RotateCw,
  Sparkles,
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

export default function DiagnosticsPage() {
  const [diag, setDiag] = useState<Diagnostic | null>(null);
  const [history, setHistory] = useState<number[]>([0.15, 0.18, 0.12, 0.22, 0.19, 0.25, 0.82, 0.88, 0.85]);
  const [selectedPreset, setSelectedPreset] = useState("lung_positive");
  const [sampleId, setSampleId] = useState("SMPL_ONC_8892");
  const [patientAge, setPatientAge] = useState(62);
  const [smokingStatus, setSmokingStatus] = useState("Former");
  const [isRunning, setIsRunning] = useState(false);
  const [detailedResult, setDetailedResult] = useState<DetailedPrediction | null>(null);

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
      alert("Diagnostic execution error: " + err);
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="panel p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-l-4 border-l-purple-500">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Eye className="w-5 h-5 text-purple-400" />
              Oncological VOC Screening &amp; Explainability
            </h2>
            <span className="quantum-badge">Quantum SHAP</span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            BioZZ Quantum Kernel evaluation of volatile organic biomarkers with reverse Hilbert space attribution.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-emerald-400" : "bg-gray-500"}`} />
          <span className="text-xs text-gray-400">
            {connected ? "Live Stream Synced" : "On-Demand Diagnostic Mode"}
          </span>
        </div>
      </div>

      {/* Main Grid: Left Controls & Sample Ingestion | Right Diagnostics & Waterfall */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Sample Selector & Ingestion Form (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="panel p-5 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2 border-b border-white/5 pb-3">
              <FlaskConical className="w-4 h-4 text-purple-400" />
              Specimen Ingestion
            </h3>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-300">Specimen Identifier</label>
              <input
                type="text"
                value={sampleId}
                onChange={(e) => setSampleId(e.target.value)}
                className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none font-mono"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300">Patient Age</label>
                <input
                  type="number"
                  value={patientAge}
                  onChange={(e) => setPatientAge(parseInt(e.target.value))}
                  className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300">Smoking Status</label>
                <select
                  value={smokingStatus}
                  onChange={(e) => setSmokingStatus(e.target.value)}
                  className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
                >
                  <option value="Former">Former Smoker</option>
                  <option value="Current">Current Smoker</option>
                  <option value="Never">Never Smoker</option>
                </select>
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-300">Canine Olfactory Sensor Preset</label>
              <select
                value={selectedPreset}
                onChange={(e) => setSelectedPreset(e.target.value)}
                className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
              >
                {PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={runQuantumDiagnostic}
              disabled={isRunning}
              className="w-full py-3 rounded-xl quantum-btn flex items-center justify-center gap-2 text-sm mt-2"
            >
              {isRunning ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>Evaluating Quantum Kernel...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Run Quantum Diagnostic</span>
                </>
              )}
            </button>
          </div>

          {/* Quick Info Panel */}
          <div className="panel p-4 text-xs text-gray-400 space-y-2 border-dashed">
            <div className="flex items-center gap-2 text-purple-300 font-semibold">
              <Cpu className="w-3.5 h-3.5" />
              <span>BioZZ Quantum Feature Mapping</span>
            </div>
            <p className="leading-relaxed">
              Patient sensor transient vectors are encoded via angle rotations and multi-qubit entangling gates (Rz phase shifts), projecting into a 2^n high-dimensional Hilbert space.
            </p>
          </div>
        </div>

        {/* Right Column: Prediction, Waterfall & Risk History (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {detailedResult ? (
            <div className="space-y-6">
              {/* Top Result Banner */}
              <div
                className={`panel p-6 border-l-4 ${
                  detailedResult.cancer_probability >= 0.5 ? "border-l-rose-500" : "border-l-emerald-500"
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase ${
                          detailedResult.cancer_probability >= 0.5
                            ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                            : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                        }`}
                      >
                        {detailedResult.prediction}
                      </span>
                      <span className="text-xs text-gray-400 font-mono">ID: {detailedResult.sample_id}</span>
                    </div>
                    <h3 className="text-2xl font-extrabold text-white mt-1">
                      {(detailedResult.cancer_probability * 100).toFixed(1)}% Malignancy Risk Index
                    </h3>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <span className="text-[10px] text-gray-400 uppercase block">Quantum Confidence</span>
                      <span className="text-lg font-bold text-purple-400 font-mono">
                        {detailedResult.quantum_confidence_pct}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="w-full h-2.5 bg-gray-800 rounded-full overflow-hidden mt-4">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${detailedResult.cancer_probability * 100}%`,
                      background:
                        detailedResult.cancer_probability >= 0.5
                          ? "linear-gradient(90deg, #f59e0b, #ef4444)"
                          : "linear-gradient(90deg, #10b981, #06b6d4)",
                    }}
                  />
                </div>
              </div>

              {/* Biomarker Waterfall Attribution */}
              <div className="panel p-6">
                <BiomarkerWaterfall
                  biomarkers={detailedResult.top_biomarkers}
                  pathways={detailedResult.biochemical_pathways}
                />
              </div>

              {/* Oncologist Summary */}
              <div className="panel p-5 bg-indigo-950/20 border border-indigo-500/30 space-y-2">
                <div className="flex items-center gap-2 text-xs font-bold uppercase text-indigo-300">
                  <FileText className="w-4 h-4 text-indigo-400" />
                  Clinical Oncologist Evaluation Summary
                </div>
                <p className="text-xs text-gray-300 leading-relaxed font-sans">
                  {detailedResult.clinical_summary}
                </p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <PredictionCard diag={diag} />

              <div className="panel p-6 space-y-3">
                <h3 className="text-sm font-bold text-gray-300 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-purple-400" />
                  Risk History (Last 60 Windows)
                </h3>
                <svg viewBox="0 0 300 120" className="w-full h-36">
                  <line x1={0} y1={60} x2={300} y2={60} stroke="#374151" strokeDasharray="4 4" />
                  <text x={4} y={14} fontSize={9} fill="#9ca3af">
                    100%
                  </text>
                  <text x={4} y={64} fontSize={8} fill="#6b7280">
                    Threshold (50%)
                  </text>
                  <text x={4} y={114} fontSize={9} fill="#9ca3af">
                    0%
                  </text>
                  {(() => {
                    const n = Math.max(history.length, 1);
                    const pts = history
                      .map((v, i) => `${(i / (n - 1 || 1)) * 296 + 2},${118 - v * 116}`)
                      .join(" ");
                    return <polyline points={pts} fill="none" stroke="#a855f7" strokeWidth={2.5} />;
                  })()}
                </svg>
                <p className="text-[11px] text-gray-500">
                  Real-time continuous rolling risk index window. Dashed line denotes 50% malignancy cutoff.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
