"use client";

import { useState, type ReactNode } from "react";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Cpu,
  Play,
  RotateCw,
  Sliders,
} from "lucide-react";
import { postJson } from "@/lib/api";
import LossChart from "@/components/LossChart";
import ConfusionMatrix from "@/components/ConfusionMatrix";

interface TrainResult {
  model_name: string;
  paradigm: string;
  status: string;
  train_time_sec: number;
  inference_time_ms: number;
  total_elapsed_sec: number;
  loss_history: number[];
  metrics: {
    accuracy: number;
    balanced_accuracy: number;
    sensitivity_recall: number;
    specificity: number;
    precision_ppv: number;
    negative_predictive_val: number;
    f1_macro: number;
    roc_auc: number;
    brier_score: number;
  };
  confusion_matrix: number[][];
  circuit_profile?: {
    n_qubits: number;
    circuit_depth: number;
    total_gates: number;
    two_qubit_cnot_gates: number;
    nisq_verdict: string;
  };
  dataset_summary: {
    train_samples: number;
    test_samples: number;
    target_cancer: string;
    n_sensors: number;
    n_features_raw: number;
    n_qubits: number;
  };
}

interface BenchmarkResult {
  target_cancer: string;
  n_qubits: number;
  leaderboard: Array<{
    model_name: string;
    paradigm: string;
    accuracy: number;
    balanced_accuracy: number;
    sensitivity_recall: number;
    specificity: number;
    precision_ppv: number;
    f1_macro: number;
    roc_auc: number;
    brier_score: number;
    train_time_sec: number;
    inference_time_ms: number;
  }>;
}

const MODELS = [
  { id: "QSVM_BioZZ", name: "QSVM (BioZZ Covariance)", paradigm: "Quantum", desc: "NISQ Kernel matrix with empirical chemical entanglement" },
  { id: "QSVM_PauliZZ", name: "QSVM (Pauli ZZ Map)", paradigm: "Quantum", desc: "Second-order Pauli phase interaction embedding" },
  { id: "VQC_StronglyEntangled", name: "VQC (Strongly Entangled)", paradigm: "Quantum", desc: "Variational circuit with cyclic multi-qubit CNOTs" },
  { id: "VQC_RealAmplitudes", name: "VQC (Real Amplitudes)", paradigm: "Quantum", desc: "Ry rotation layers with nearest-neighbor entanglement" },
  { id: "QCNN", name: "QCNN (CQSV-Net)", paradigm: "Quantum", desc: "Hierarchical quantum convolution and quantum pooling" },
  { id: "Quantum_Kernel_XGB", name: "Quantum-Kernel XGBoost (CG-ZZ+XGB)", paradigm: "Hybrid", desc: "CG-ZZ/BioZZ quantum kernel features fused with XGBoost - quantum-first hybrid" },
  { id: "Quantum_Augmented_XGB", name: "Quantum-Augmented XGBoost (VQC+XGB)", paradigm: "Hybrid", desc: "VQC BioZZ opinion as extra feature for XGBoost - variational hybrid" },
  { id: "SVM_RBF", name: "SVM (RBF Kernel)", paradigm: "Classical", desc: "Classical non-linear radial basis function baseline" },
  { id: "SVM_Linear", name: "Linear SVM", paradigm: "Classical", desc: "Standard classical linear separating hyperplane" },
  { id: "Random_Forest", name: "Random Forest", paradigm: "Classical", desc: "Ensemble of 150 randomized decision trees" },
  { id: "XGBoost", name: "XGBoost / GBDT", paradigm: "Classical", desc: "Gradient boosted decision tree trees" },
  { id: "MLP_NeuralNet", name: "Deep MLP Neural Net", paradigm: "Classical", desc: "Multi-layer perceptron (64-32 ReLU layers)" },
];

const CANCER_TYPES = [
  { id: "Lung_Cancer", name: "Lung Cancer (Hexanal & Benzaldehyde)" },
  { id: "Breast_Cancer", name: "Breast Cancer (Heptanal & Ketones)" },
  { id: "Colorectal_Cancer", name: "Colorectal Cancer (DMDS & Aromatics)" },
  { id: "Prostate_Cancer", name: "Prostate Cancer (Urine Volatiles)" },
  { id: "Ovarian_Cancer", name: "Ovarian Cancer (Nonanal & Acetophenone)" },
];

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div className="label">{label}</div>
      <div
        className="num"
        style={{ fontSize: 26, fontWeight: 800, lineHeight: 1.15, marginTop: 6, color: color ?? "var(--ink)" }}
      >
        {value}
      </div>
    </div>
  );
}

function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <div
      className="surface flex flex-col items-center justify-center gap-4"
      style={{ minHeight: 360, textAlign: "center" }}
    >
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: "50%",
          border: "1px solid var(--line)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--muted)",
        }}
      >
        {icon}
      </div>
      <div className="flex flex-col gap-2" style={{ maxWidth: 460 }}>
        <h3 className="section-title">{title}</h3>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>
          {children}
        </p>
      </div>
    </div>
  );
}

export default function TrainingStudio() {
  // Config state
  const [modelType, setModelType] = useState("QSVM_BioZZ");
  const [targetCancer, setTargetCancer] = useState("Lung_Cancer");
  const [nQubits, setNQubits] = useState(6);
  const [samplesPerClass, setSamplesPerClass] = useState(50);
  const [epochs, setEpochs] = useState(25);
  const [learningRate, setLearningRate] = useState(0.03);
  const [cParam, setCParam] = useState(1.0);

  // Execution state
  const [isTraining, setIsTraining] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const [activeTab, setActiveTab] = useState<"single" | "benchmark">("single");

  async function handleTrain() {
    setIsTraining(true);
    setError(null);
    try {
      const res = await postJson<TrainResult>("/api/v1/train", {
        model_type: modelType,
        target_cancer: targetCancer,
        n_qubits: nQubits,
        samples_per_class: samplesPerClass,
        epochs: epochs,
        learning_rate: learningRate,
        c_param: cParam,
        test_size: 0.2,
      });
      setTrainResult(res);
      setActiveTab("single");
    } catch (err: any) {
      setError(err?.message || "Training job failed to complete.");
    } finally {
      setIsTraining(false);
    }
  }

  async function handleBenchmark() {
    setIsBenchmarking(true);
    setError(null);
    try {
      const res = await postJson<BenchmarkResult>("/api/v1/benchmark/run", {
        target_cancer: targetCancer,
        n_samples_per_class: samplesPerClass,
        n_qubits: nQubits,
      });
      setBenchmarkResult(res);
      setActiveTab("benchmark");
    } catch (err: any) {
      setError(err?.message || "Benchmark failed to execute.");
    } finally {
      setIsBenchmarking(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex flex-col gap-2">
          <h1 className="section-title">Quantum ML training studio &amp; benchmarking lab</h1>
          <p className="muted" style={{ margin: 0, maxWidth: 720, fontSize: 14 }}>
            Configure hyperparameters, execute quantum circuit training directly from your browser, and evaluate
            leak-free generalization against classical baselines.
          </p>
        </div>
        <span className="chip" style={{ alignSelf: "flex-start" }}>
          NISQ Accelerated
        </span>
      </header>

      <hr className="divider" />

      <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="Training results view">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "single"}
          className={`tab ${activeTab === "single" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("single")}
        >
          Model Trainer
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "benchmark"}
          className={`tab ${activeTab === "benchmark" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("benchmark")}
        >
          Benchmark Leaderboard
        </button>
      </div>

      {error && (
        <div className="surface-sm row" style={{ borderLeft: "3px solid var(--bad)", color: "var(--bad)" }} role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls */}
        <aside className="lg:col-span-4">
          <div className="surface flex flex-col gap-5">
            <div className="row">
              <h2 className="section-title">Hyperparameters</h2>
              <Sliders size={18} aria-hidden="true" />
            </div>
            <hr className="divider" />

            <div className="flex flex-col gap-1.5">
              <label className="label" htmlFor="model-type">
                Model architecture
              </label>
              <select
                id="model-type"
                className="field"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
              >
                <optgroup label="Quantum Machine Learning">
                  {MODELS.filter((m) => m.paradigm === "Quantum").map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Hybrid Quantum-Classical">
                  {MODELS.filter((m) => m.paradigm === "Hybrid").map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Classical Baselines">
                  {MODELS.filter((m) => m.paradigm === "Classical").map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </optgroup>
              </select>
              <p className="muted" style={{ margin: 0, fontSize: 12 }}>
                {MODELS.find((m) => m.id === modelType)?.desc}
              </p>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="label" htmlFor="target-cancer">
                Target cancer cohort
              </label>
              <select
                id="target-cancer"
                className="field"
                value={targetCancer}
                onChange={(e) => setTargetCancer(e.target.value)}
              >
                {CANCER_TYPES.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-2">
              <div className="row">
                <label className="label" htmlFor="n-qubits">
                  Register qubits (n)
                </label>
                <span className="num" style={{ fontWeight: 800, color: "var(--action)" }}>
                  {nQubits} qubits
                </span>
              </div>
              <input
                id="n-qubits"
                type="range"
                min="4"
                max="12"
                step="1"
                value={nQubits}
                onChange={(e) => setNQubits(parseInt(e.target.value))}
                className="field"
              />
              <div className="row muted" style={{ fontSize: 11 }}>
                <span>4 qubits (fast)</span>
                <span>8 (standard)</span>
                <span>12 (deep)</span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="row">
                <label className="label" htmlFor="samples-per-class">
                  Cohort size per class
                </label>
                <span className="num" style={{ fontWeight: 800, color: "var(--action)" }}>
                  {samplesPerClass} samples
                </span>
              </div>
              <input
                id="samples-per-class"
                type="range"
                min="20"
                max="150"
                step="10"
                value={samplesPerClass}
                onChange={(e) => setSamplesPerClass(parseInt(e.target.value))}
                className="field"
              />
              <div className="row muted" style={{ fontSize: 11 }}>
                <span>20 (preview)</span>
                <span>80 (balanced)</span>
                <span>150 (deep)</span>
              </div>
            </div>

            {(modelType.startsWith("VQC") ||
              modelType === "QCNN" ||
              modelType === "MLP_NeuralNet" ||
              modelType === "Quantum_Augmented_XGB") && (
              <div className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-1.5">
                  <label className="label" htmlFor="epochs">
                    Epochs
                  </label>
                  <input
                    id="epochs"
                    type="number"
                    min="5"
                    max="100"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    className="field"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="label" htmlFor="learning-rate">
                    Learning rate
                  </label>
                  <input
                    id="learning-rate"
                    type="number"
                    step="0.01"
                    min="0.001"
                    max="0.2"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="field"
                  />
                </div>
              </div>
            )}

            {(modelType.startsWith("QSVM") || modelType.startsWith("SVM")) && (
              <div className="flex flex-col gap-2">
                <div className="row">
                  <label className="label" htmlFor="c-param">
                    Regularization (C)
                  </label>
                  <span className="num" style={{ fontWeight: 800, color: "var(--action)" }}>
                    {cParam.toFixed(2)}
                  </span>
                </div>
                <input
                  id="c-param"
                  type="range"
                  min="0.1"
                  max="10.0"
                  step="0.1"
                  value={cParam}
                  onChange={(e) => setCParam(parseFloat(e.target.value))}
                  className="field"
                />
              </div>
            )}

            <hr className="divider" />

            <div className="flex flex-col gap-2">
              <button
                type="button"
                onClick={handleTrain}
                disabled={isTraining || isBenchmarking}
                className="btn-primary w-full"
              >
                {isTraining ? (
                  <>
                    <RotateCw size={16} className="animate-spin" />
                    Compiling Quantum Circuit...
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    Invoke Training from Browser
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={handleBenchmark}
                disabled={isTraining || isBenchmarking}
                className="btn-quiet w-full"
              >
                {isBenchmarking ? (
                  <>
                    <RotateCw size={16} className="animate-spin" />
                    Evaluating All 12 Baselines...
                  </>
                ) : (
                  <>
                    <BarChart3 size={16} />
                    Run Multi-Model Benchmark Suite
                  </>
                )}
              </button>
            </div>
          </div>
        </aside>

        {/* Results */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          {activeTab === "single" ? (
            trainResult ? (
              <div className="flex flex-col gap-6">
                <div className="surface flex flex-col gap-5">
                  <div className="row flex-wrap gap-4">
                    <div className="flex flex-col gap-2">
                      <div className="flex flex-wrap items-center gap-3">
                        <h2 className="section-title">{trainResult.model_name}</h2>
                        <span className="chip">{trainResult.paradigm}</span>
                      </div>
                      <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                        Target:{" "}
                        <strong style={{ color: "var(--ink)" }}>
                          {trainResult.dataset_summary.target_cancer}
                        </strong>{" "}
                        · Train: {trainResult.dataset_summary.train_samples} samples · Test:{" "}
                        {trainResult.dataset_summary.test_samples} samples
                      </p>
                    </div>
                    <span className="chip chip-good">
                      <CheckCircle2 size={14} /> Trained
                    </span>
                  </div>

                  <hr className="divider" />
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
                    <Stat label="ROC-AUC score" value={pct(trainResult.metrics.roc_auc)} color="var(--action)" />
                    <Stat
                      label="Sensitivity (recall)"
                      value={pct(trainResult.metrics.sensitivity_recall)}
                      color="var(--good)"
                    />
                    <Stat label="Specificity" value={pct(trainResult.metrics.specificity)} />
                    <Stat
                      label="Brier calibration"
                      value={trainResult.metrics.brier_score.toFixed(4)}
                      color="var(--muted)"
                    />
                  </div>

                  <hr className="divider" />
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
                    <Stat label="Accuracy" value={pct(trainResult.metrics.accuracy)} />
                    <Stat label="Balanced acc" value={pct(trainResult.metrics.balanced_accuracy)} />
                    <Stat label="Precision (PPV)" value={pct(trainResult.metrics.precision_ppv)} />
                    <Stat label="Macro F1" value={trainResult.metrics.f1_macro.toFixed(3)} />
                  </div>

                  <hr className="divider" />
                  <div className="row flex-wrap gap-4 muted" style={{ fontSize: 13 }}>
                    <span>
                      Train time{" "}
                      <span className="num" style={{ color: "var(--ink)", fontWeight: 700 }}>
                        {trainResult.train_time_sec}s
                      </span>
                    </span>
                    <span>
                      Inference / sample{" "}
                      <span className="num" style={{ color: "var(--ink)", fontWeight: 700 }}>
                        {trainResult.inference_time_ms}ms
                      </span>
                    </span>
                    <span>
                      Total elapsed{" "}
                      <span className="num" style={{ color: "var(--ink)", fontWeight: 700 }}>
                        {trainResult.total_elapsed_sec}s
                      </span>
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="surface">
                    <LossChart lossHistory={trainResult.loss_history} color="#2f5fd2" />
                  </div>
                  <div className="surface">
                    <ConfusionMatrix matrix={trainResult.confusion_matrix} />
                  </div>
                </div>

                {trainResult.circuit_profile && (
                  <div className="surface flex flex-col gap-4">
                    <div className="row flex-wrap gap-3">
                      <h3 className="section-title flex items-center gap-2">
                        <Cpu size={18} /> NISQ hardware compilation profile
                      </h3>
                      <span className="chip chip-warn">{trainResult.circuit_profile.nisq_verdict}</span>
                    </div>
                    <hr className="divider" />
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
                      <Stat label="Active qubits" value={String(trainResult.circuit_profile.n_qubits)} />
                      <Stat
                        label="Circuit depth"
                        value={`${trainResult.circuit_profile.circuit_depth} gates`}
                      />
                      <Stat
                        label="2-qubit CNOTs"
                        value={String(trainResult.circuit_profile.two_qubit_cnot_gates)}
                      />
                      <Stat label="Total gates" value={String(trainResult.circuit_profile.total_gates)} />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <EmptyState icon={<Cpu size={24} />} title="No model trained yet">
                Select an architecture on the left (for example QSVM BioZZ or VQC), then choose Invoke Training
                from Browser to train and evaluate it on the held-out cohort.
              </EmptyState>
            )
          ) : benchmarkResult ? (
            <div className="surface flex flex-col gap-5">
              <div className="row flex-wrap gap-3">
                <div className="flex flex-col gap-2">
                  <h2 className="section-title flex items-center gap-2">
                    <BarChart3 size={18} /> Comparative leaderboard
                  </h2>
                  <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                    Target:{" "}
                    <strong style={{ color: "var(--ink)" }}>{benchmarkResult.target_cancer}</strong> · Evaluated on
                    a strictly held-out test cohort
                  </p>
                </div>
                <span className="chip">12 models</span>
              </div>

              <hr className="divider" />

              <div className="overflow-x-auto">
                <table className="w-full" style={{ minWidth: 760, fontSize: 13 }}>
                  <thead>
                    <tr>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        Model
                      </th>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        Paradigm
                      </th>
                      <th className="label" style={{ textAlign: "right", paddingBottom: 10 }}>
                        ROC-AUC
                      </th>
                      <th className="label" style={{ textAlign: "right", paddingBottom: 10 }}>
                        Sensitivity
                      </th>
                      <th className="label" style={{ textAlign: "right", paddingBottom: 10 }}>
                        Specificity
                      </th>
                      <th className="label" style={{ textAlign: "right", paddingBottom: 10 }}>
                        F1 score
                      </th>
                      <th className="label" style={{ textAlign: "right", paddingBottom: 10 }}>
                        Brier loss
                      </th>
                      <th className="label" style={{ textAlign: "right", paddingBottom: 10 }}>
                        Train time
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResult.leaderboard.map((row, idx) => (
                      <tr key={idx} style={{ borderTop: "1px solid var(--line)" }}>
                        <td style={{ padding: "10px 8px 10px 0", fontWeight: 700 }}>{row.model_name}</td>
                        <td style={{ padding: "10px 8px" }}>
                          <span className="chip">{row.paradigm}</span>
                        </td>
                        <td
                          className="num"
                          style={{ textAlign: "right", padding: "10px 8px", fontWeight: 700, color: "var(--action)" }}
                        >
                          {pct(row.roc_auc)}
                        </td>
                        <td
                          className="num"
                          style={{ textAlign: "right", padding: "10px 8px", color: "var(--good)" }}
                        >
                          {pct(row.sensitivity_recall)}
                        </td>
                        <td className="num" style={{ textAlign: "right", padding: "10px 8px" }}>
                          {pct(row.specificity)}
                        </td>
                        <td className="num" style={{ textAlign: "right", padding: "10px 8px" }}>
                          {row.f1_macro.toFixed(3)}
                        </td>
                        <td
                          className="num"
                          style={{ textAlign: "right", padding: "10px 8px", color: "var(--muted)" }}
                        >
                          {row.brier_score.toFixed(4)}
                        </td>
                        <td
                          className="num"
                          style={{ textAlign: "right", padding: "10px 0 10px 8px", color: "var(--muted)" }}
                        >
                          {row.train_time_sec}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <EmptyState icon={<BarChart3 size={24} />} title="No benchmark results yet">
              Choose Run Multi-Model Benchmark Suite to evaluate all quantum and classical models on the same
              held-out partitions.
            </EmptyState>
          )}
        </section>
      </div>
    </div>
  );
}
