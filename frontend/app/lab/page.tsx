"use client";

import { useEffect, useState } from "react";
import { getJson, postJson } from "@/lib/api";

interface LabRunRequest {
  dataset: string;
  experiment: string;
  max_samples: number;
  vqc_epochs: number;
  n_splits: number;
  n_components: number;
  seed: number;
}

interface LabJob {
  job_id: string;
  status: "running" | "done" | "error";
  request: LabRunRequest;
  started_at: number;
  progress?: string[];
  error?: string;
  result?: { run_dir: string; leaderboard: Record<string, unknown>[] };
}

const EXPERIMENTS = [
  { value: "search", label: "Hybrid search (curated presets)" },
  { value: "map_ablation", label: "Feature-map ablation (RQ3)" },
  { value: "modality_ablation", label: "Modality ablation (RQ2)" },
  { value: "robustness", label: "Noise robustness (§VII-E)" },
];

export default function LabPage() {
  const [jobs, setJobs] = useState<LabJob[]>([]);
  const [dataset, setDataset] = useState("voc_real");
  const [experiment, setExperiment] = useState("search");
  const [maxSamples, setMaxSamples] = useState(120);
  const [vqcEpochs, setVqcEpochs] = useState(6);
  const [nSplits, setNSplits] = useState(5);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const res = await getJson<{ jobs: LabJob[] }>("/api/v1/lab/runs");
      setJobs(res.jobs);
    } catch {
      /* backend offline */
    }
  }

  useEffect(() => {
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, []);

  async function launch() {
    try {
      await postJson("/api/v1/lab/runs", {
        dataset, experiment, max_samples: maxSamples,
        vqc_epochs: vqcEpochs, n_splits: nSplits,
        n_components: 6, seed: 42,
      });
      setError(null);
      void refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  const doneJobs = jobs.filter((j) => j.status === "done" && j.result);

  return (
    <div className="space-y-4">
      <div className="panel p-6 max-w-3xl">
        <h2 className="text-lg font-semibold mb-1">Hybrid Training Lab</h2>
        <p className="text-sm text-gray-400 mb-4">
          Composable quantum × classical pipelines: reduction → embedding (Angle /
          ZZ / CW-ZZ ± permuted-correlation control) → head, ranked by stratified
          k-fold ROC-AUC on identical partitions.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
          <label>
            <span className="block text-gray-400 mb-1">Dataset</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2">
              <option value="voc_real">Lung VOC (real, 427)</option>
              <option value="kennel_synth">Kennel trials (synthetic)</option>
            </select>
          </label>
          <label className="md:col-span-2">
            <span className="block text-gray-400 mb-1">Experiment</span>
            <select value={experiment} onChange={(e) => setExperiment(e.target.value)}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2">
              {EXPERIMENTS.map((x) => (
                <option key={x.value} value={x.value}>{x.label}</option>
              ))}
            </select>
          </label>
          <label>
            <span className="block text-gray-400 mb-1">Max samples</span>
            <input type="number" min={40} value={maxSamples}
              onChange={(e) => setMaxSamples(Number(e.target.value))}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2" />
          </label>
          <label>
            <span className="block text-gray-400 mb-1">VQC epochs</span>
            <input type="number" min={1} max={30} value={vqcEpochs}
              onChange={(e) => setVqcEpochs(Number(e.target.value))}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2" />
          </label>
          <label>
            <span className="block text-gray-400 mb-1">CV folds</span>
            <input type="number" min={2} max={10} value={nSplits}
              onChange={(e) => setNSplits(Number(e.target.value))}
              className="w-full bg-[#0b0f19] border border-[#374151] rounded-lg px-3 py-2" />
          </label>
        </div>

        <button onClick={launch}
          className="mt-4 w-full py-3 rounded-xl font-semibold text-white"
          style={{ background: "linear-gradient(135deg,#4f46e5,#9333ea)" }}>
          Launch experiment
        </button>
        {error && <div className="text-xs text-red-400 mt-2">{error}</div>}
      </div>

      <div className="panel p-4 max-w-3xl">
        <h3 className="font-semibold mb-2">Runs</h3>
        {jobs.length === 0 && (
          <p className="text-xs text-gray-500">
            No runs yet. Note: the API executes experiments in-process — launch
            times depend on the quantum budget.
          </p>
        )}
        <div className="space-y-2">
          {jobs.map((j) => (
            <div key={j.job_id} className="border border-[#374151] rounded-lg p-3 text-sm">
              <div className="flex items-center justify-between">
                <span>
                  <b>{j.request.experiment}</b> · {j.request.dataset} · folds=
                  {j.request.n_splits} · cap={j.request.max_samples}
                </span>
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    j.status === "done"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : j.status === "error"
                        ? "bg-red-500/20 text-red-300"
                        : "bg-indigo-500/20 text-indigo-300 animate-pulse"
                  }`}
                >
                  {j.status}
                </span>
              </div>
              {j.status === "running" && j.progress && j.progress.length > 0 && (
                <p className="text-xs text-gray-400 mt-1">{j.progress[j.progress.length - 1]}</p>
              )}
              {j.error && <p className="text-xs text-red-400 mt-1">{j.error}</p>}
            </div>
          ))}
        </div>
      </div>

      {doneJobs[0]?.result && (
        <div className="panel p-4 overflow-x-auto">
          <h3 className="font-semibold mb-2">
            Latest leaderboard <span className="text-xs text-gray-500">({doneJobs[0].result.run_dir})</span>
          </h3>
          <table className="text-xs w-full">
            <thead>
              <tr className="text-gray-400 text-left">
                <th className="py-1 pr-4">Config</th>
                <th className="py-1 pr-4">ROC-AUC</th>
                <th className="py-1 pr-4">±std</th>
                <th className="py-1 pr-4">Accuracy</th>
                <th className="py-1">Train s</th>
              </tr>
            </thead>
            <tbody>
              {(doneJobs[0].result.leaderboard as Record<string, number | string>[]).map((row) => (
                <tr key={String(row.config)} className="border-t border-[#1f2937]">
                  <td className="py-1 pr-4">{String(row.config)}</td>
                  <td className="py-1 pr-4 tabular-nums">
                    {Number(row.roc_auc_mean ?? 0).toFixed(3)}
                  </td>
                  <td className="py-1 pr-4 tabular-nums">
                    {Number(row.roc_auc_std ?? 0).toFixed(3)}
                  </td>
                  <td className="py-1 pr-4 tabular-nums">
                    {Number(row.accuracy_mean ?? 0).toFixed(3)}
                  </td>
                  <td className="py-1 tabular-nums">
                    {Number(row.train_time_s_mean ?? 0).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
