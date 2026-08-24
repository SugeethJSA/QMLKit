"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Cpu,
  Flame,
  Gauge,
  Layers,
  Play,
  RotateCw,
  Sliders,
  Sparkles,
  Zap,
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
    <div className="space-y-6">
      {/* Top Header Banner */}
      <div className="panel p-6 border-l-4 border-l-purple-500 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-400" />
              Quantum ML Training Studio &amp; Benchmarking Lab
            </h2>
            <span className="quantum-badge">NISQ Accelerated</span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Configure hyperparameters, execute quantum circuit training directly from your browser, and evaluate leak-free generalization against classical baselines.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab("single")}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition ${
              activeTab === "single"
                ? "bg-purple-600/30 text-purple-300 border border-purple-500/50"
                : "bg-white/5 text-gray-400 hover:bg-white/10"
            }`}
          >
            Model Trainer
          </button>
          <button
            onClick={() => setActiveTab("benchmark")}
            className={`px-4 py-2 rounded-lg text-xs font-semibold transition ${
              activeTab === "benchmark"
                ? "bg-indigo-600/30 text-indigo-300 border border-indigo-500/50"
                : "bg-white/5 text-gray-400 hover:bg-white/10"
            }`}
          >
            Benchmark Leaderboard
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/50 text-rose-300 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid: Left Controls | Right Live Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Hyperparameters & Trigger (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <h3 className="text-sm font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-purple-400" />
                Hyperparameters
              </h3>
              <span className="text-[11px] text-gray-500 font-mono">PS-26139</span>
            </div>

            {/* Model Architecture Selector */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-300">Model Architecture</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-gray-200 focus:border-purple-500 focus:outline-none"
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
              <p className="text-[11px] text-gray-500">
                {MODELS.find((m) => m.id === modelType)?.desc}
              </p>
            </div>

            {/* Target Cancer Indication */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-gray-300">Target Cancer Cohort</label>
              <select
                value={targetCancer}
                onChange={(e) => setTargetCancer(e.target.value)}
                className="w-full bg-[#0b0f19] border border-white/10 rounded-xl px-3 py-2 text-sm text-gray-200 focus:border-purple-500 focus:outline-none"
              >
                {CANCER_TYPES.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Qubits Slider (Quantum Register) */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-300">Quantum Register Qubits (n)</span>
                <span className="font-mono text-purple-400 font-bold">{nQubits} qubits</span>
              </div>
              <input
                type="range"
                min="4"
                max="12"
                step="1"
                value={nQubits}
                onChange={(e) => setNQubits(parseInt(e.target.value))}
                className="w-full accent-purple-500 bg-gray-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-gray-500">
                <span>4 Qubits (Fast)</span>
                <span>8 Qubits (Standard)</span>
                <span>12 Qubits (Deep)</span>
              </div>
            </div>

            {/* Samples Per Class */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-300">Cohort Size (per class)</span>
                <span className="font-mono text-indigo-400 font-bold">{samplesPerClass} samples</span>
              </div>
              <input
                type="range"
                min="20"
                max="150"
                step="10"
                value={samplesPerClass}
                onChange={(e) => setSamplesPerClass(parseInt(e.target.value))}
                className="w-full accent-indigo-500 bg-gray-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-gray-500">
                <span>20 (Fast preview)</span>
                <span>80 (Balanced)</span>
                <span>150 (Deep clinical)</span>
              </div>
            </div>

            {/* Epochs & Learning Rate (For VQC/QCNN/MLP + Quantum-Augmented hybrid) */}
            {(modelType.startsWith("VQC") || modelType === "QCNN" || modelType === "MLP_NeuralNet" || modelType === "Quantum_Augmented_XGB") && (
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-white/5">
                <div className="space-y-1">
                  <label className="text-xs text-gray-400">Epochs</label>
                  <input
                    type="number"
                    min="5"
                    max="100"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    className="w-full bg-[#0b0f19] border border-white/10 rounded-lg px-2.5 py-1.5 text-sm text-gray-200"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-gray-400">Learning Rate</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.001"
                    max="0.2"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full bg-[#0b0f19] border border-white/10 rounded-lg px-2.5 py-1.5 text-sm text-gray-200"
                  />
                </div>
              </div>
            )}

            {/* C Regularization (For QSVM / SVM) */}
            {(modelType.startsWith("QSVM") || modelType.startsWith("SVM")) && (
              <div className="space-y-1 pt-2 border-t border-white/5">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-300">Regularization Param (C)</span>
                  <span className="font-mono text-purple-400 font-bold">{cParam.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="10.0"
                  step="0.1"
                  value={cParam}
                  onChange={(e) => setCParam(parseFloat(e.target.value))}
                  className="w-full accent-purple-500 bg-gray-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}

            {/* Action Buttons */}
            <div className="pt-3 space-y-2">
              <button
                onClick={handleTrain}
                disabled={isTraining || isBenchmarking}
                className="w-full py-3 rounded-xl quantum-btn flex items-center justify-center gap-2 text-sm"
              >
                {isTraining ? (
                  <>
                    <RotateCw className="w-4 h-4 animate-spin" />
                    <span>Compiling Quantum Circuit...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white" />
                    <span>Invoke Training from Browser</span>
                  </>
                )}
              </button>

              <button
                onClick={handleBenchmark}
                disabled={isTraining || isBenchmarking}
                className="w-full py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 text-xs font-semibold transition flex items-center justify-center gap-2"
              >
                {isBenchmarking ? (
                  <>
                    <RotateCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                    <span>Evaluating All 12 Baselines...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Run Multi-Model Benchmark Suite</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Live Results Display (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {activeTab === "single" ? (
            trainResult ? (
              <div className="space-y-4">
                {/* Result Overview Header Card */}
                <div className="panel p-5 space-y-4 border-l-4 border-l-emerald-500">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center">
                        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-base font-bold text-white">{trainResult.model_name}</h3>
                          <span
                            className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${
                              trainResult.paradigm === "Quantum"
                                ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                                : trainResult.paradigm === "Hybrid"
                                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                                  : "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                            }`}
                          >
                            {trainResult.paradigm}
                          </span>
                        </div>
                        <p className="text-xs text-gray-400 mt-0.5">
                          Target: <strong className="text-gray-200">{trainResult.dataset_summary.target_cancer}</strong> Â· Train:{" "}
                          {trainResult.dataset_summary.train_samples} samples Â· Test: {trainResult.dataset_summary.test_samples} samples
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <div className="px-3 py-1.5 bg-[#0b0f19] border border-white/10 rounded-lg text-center">
                        <span className="text-gray-500 block text-[10px]">Train Time</span>
                        <span className="font-mono text-purple-300 font-bold">{trainResult.train_time_sec}s</span>
                      </div>
                      <div className="px-3 py-1.5 bg-[#0b0f19] border border-white/10 rounded-lg text-center">
                        <span className="text-gray-500 block text-[10px]">Inference/Sample</span>
                        <span className="font-mono text-indigo-300 font-bold">{trainResult.inference_time_ms}ms</span>
                      </div>
                    </div>
                  </div>

                  {/* 8 Primary Diagnostic Metrics Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-[#0b0f19]/80 border border-white/5 rounded-xl p-3 text-center">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider block">ROC-AUC Score</span>
                      <span className="text-2xl font-extrabold text-purple-400 font-mono">
                        {(trainResult.metrics.roc_auc * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="bg-[#0b0f19]/80 border border-white/5 rounded-xl p-3 text-center">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider block">Sensitivity (Recall)</span>
                      <span className="text-2xl font-extrabold text-emerald-400 font-mono">
                        {(trainResult.metrics.sensitivity_recall * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="bg-[#0b0f19]/80 border border-white/5 rounded-xl p-3 text-center">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider block">Specificity</span>
                      <span className="text-2xl font-extrabold text-indigo-400 font-mono">
                        {(trainResult.metrics.specificity * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="bg-[#0b0f19]/80 border border-white/5 rounded-xl p-3 text-center">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider block">Brier Calibration</span>
                      <span className="text-2xl font-extrabold text-rose-400 font-mono">
                        {trainResult.metrics.brier_score.toFixed(4)}
                      </span>
                    </div>
                  </div>

                  {/* Secondary Metrics Bar */}
                  <div className="grid grid-cols-4 gap-2 pt-2 border-t border-white/5 text-center text-xs">
                    <div>
                      <span className="text-gray-500 block text-[10px]">Accuracy</span>
                      <span className="font-mono font-bold text-gray-200">{(trainResult.metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-[10px]">Balanced Acc</span>
                      <span className="font-mono font-bold text-gray-200">{(trainResult.metrics.balanced_accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-[10px]">Precision (PPV)</span>
                      <span className="font-mono font-bold text-gray-200">{(trainResult.metrics.precision_ppv * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-[10px]">Macro F1</span>
                      <span className="font-mono font-bold text-gray-200">{trainResult.metrics.f1_macro.toFixed(3)}</span>
                    </div>
                  </div>
                </div>

                {/* Charts & Diagnostics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Left: Loss Curve */}
                  <div className="panel p-5">
                    <LossChart lossHistory={trainResult.loss_history} />
                  </div>

                  {/* Right: Confusion Matrix */}
                  <div className="panel p-5">
                    <ConfusionMatrix matrix={trainResult.confusion_matrix} />
                  </div>
                </div>

                {/* NISQ Quantum Circuit Hardware Profile (if Quantum) */}
                {trainResult.circuit_profile && (
                  <div className="panel p-5 space-y-3 border border-purple-500/20">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-bold text-purple-300 flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-purple-400" />
                        NISQ Quantum Hardware Compilation Profile
                      </h4>
                      <span className="text-[11px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-200 font-semibold">
                        {trainResult.circuit_profile.nisq_verdict}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-2.5 bg-[#0b0f19] border border-white/5 rounded-lg">
                        <span className="text-[10px] text-gray-400 uppercase block">Active Qubits</span>
                        <span className="text-lg font-bold text-white font-mono">{trainResult.circuit_profile.n_qubits}</span>
                      </div>
                      <div className="p-2.5 bg-[#0b0f19] border border-white/5 rounded-lg">
                        <span className="text-[10px] text-gray-400 uppercase block">Circuit Depth</span>
                        <span className="text-lg font-bold text-indigo-300 font-mono">{trainResult.circuit_profile.circuit_depth} gates</span>
                      </div>
                      <div className="p-2.5 bg-[#0b0f19] border border-white/5 rounded-lg">
                        <span className="text-[10px] text-gray-400 uppercase block">2-Qubit CNOTs</span>
                        <span className="text-lg font-bold text-purple-300 font-mono">{trainResult.circuit_profile.two_qubit_cnot_gates}</span>
                      </div>
                      <div className="p-2.5 bg-[#0b0f19] border border-white/5 rounded-lg">
                        <span className="text-[10px] text-gray-400 uppercase block">Total Quantum Gates</span>
                        <span className="text-lg font-bold text-pink-300 font-mono">{trainResult.circuit_profile.total_gates}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Empty Placeholder State */
              <div className="panel p-12 text-center flex flex-col items-center justify-center space-y-4 min-h-[450px]">
                <div className="w-16 h-16 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                  <Cpu className="w-8 h-8 text-purple-400" />
                </div>
                <div className="max-w-md space-y-1">
                  <h3 className="text-base font-bold text-white">No Active Model Trained Yet</h3>
                  <p className="text-xs text-gray-400">
                    Select an architecture on the left (e.g. <strong>QSVM BioZZ</strong> or <strong>VQC</strong>) and click <em>&quot;Invoke Training from Browser&quot;</em> to simulate and evaluate in real-time.
                  </p>
                </div>
              </div>
            )
          ) : (
            /* Benchmark Leaderboard View */
            benchmarkResult ? (
              <div className="panel p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-white/5 pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-indigo-400" />
                      Comparative Leaderboard: Quantum vs Classical Baselines
                    </h3>
                    <p className="text-xs text-gray-400 mt-0.5">
                      Target: <strong className="text-gray-200">{benchmarkResult.target_cancer}</strong> Â· Evaluated on strictly held-out test cohort
                    </p>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="text-[11px] text-gray-400 uppercase bg-white/5 border-b border-white/10">
                      <tr>
                        <th className="py-2.5 px-3">Model</th>
                        <th className="py-2.5 px-2">Paradigm</th>
                        <th className="py-2.5 px-2 text-right">ROC-AUC</th>
                        <th className="py-2.5 px-2 text-right">Sensitivity</th>
                        <th className="py-2.5 px-2 text-right">Specificity</th>
                        <th className="py-2.5 px-2 text-right">F1-Score</th>
                        <th className="py-2.5 px-2 text-right">Brier Loss</th>
                        <th className="py-2.5 px-3 text-right">Train Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {benchmarkResult.leaderboard.map((row, idx) => (
                        <tr key={idx} className="hover:bg-white/5 transition">
                          <td className="py-2.5 px-3 font-semibold text-gray-200">{row.model_name}</td>
                          <td className="py-2.5 px-2">
                            <span
                              className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                                row.paradigm === "Quantum"
                                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                                  : row.paradigm === "Hybrid"
                                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                                    : "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                              }`}
                            >
                              {row.paradigm}
                            </span>
                          </td>
                          <td className="py-2.5 px-2 text-right font-mono font-bold text-purple-400">
                            {(row.roc_auc * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-2 text-right font-mono text-emerald-400">
                            {(row.sensitivity_recall * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-2 text-right font-mono text-indigo-400">
                            {(row.specificity * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-2 text-right font-mono text-gray-300">
                            {row.f1_macro.toFixed(3)}
                          </td>
                          <td className="py-2.5 px-2 text-right font-mono text-rose-400">
                            {row.brier_score.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-gray-400">
                            {row.train_time_sec}s
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="panel p-12 text-center flex flex-col items-center justify-center space-y-4 min-h-[450px]">
                <BarChart3 className="w-10 h-10 text-indigo-400" />
                <p className="text-xs text-gray-400">
                  Click <em>&quot;Run Multi-Model Benchmark Suite&quot;</em> to evaluate all quantum &amp; classical models simultaneously.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
