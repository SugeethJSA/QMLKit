"use client";

import { useEffect, useState } from "react";
import { AlertCircle, FlaskConical, Play } from "lucide-react";
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
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <h1 className="section-title">Hybrid training lab</h1>
        <p className="muted" style={{ margin: 0, maxWidth: 760, fontSize: 14 }}>
          Composable quantum × classical pipelines: reduction → embedding (Angle / ZZ / CW-ZZ ± permuted-correlation
          control) → head, ranked by stratified k-fold ROC-AUC on identical partitions.
        </p>
      </header>

      <hr className="divider" />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Configuration */}
        <section className="lg:col-span-5">
          <div className="surface flex flex-col gap-5">
            <div className="row">
              <h2 className="section-title flex items-center gap-2">
                <FlaskConical size={18} /> Configure a run
              </h2>
              <span className="chip">PS-26139</span>
            </div>
            <hr className="divider" />

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="lab-dataset">
                  Dataset
                </label>
                <select
                  id="lab-dataset"
                  className="field"
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="voc_real">Lung VOC (real, 427)</option>
                  <option value="kennel_synth">Kennel trials (synthetic)</option>
                </select>
              </div>
              <div className="flex flex-col gap-1.5 md:col-span-2">
                <label className="label" htmlFor="lab-experiment">
                  Experiment
                </label>
                <select
                  id="lab-experiment"
                  className="field"
                  value={experiment}
                  onChange={(e) => setExperiment(e.target.value)}
                >
                  {EXPERIMENTS.map((x) => (
                    <option key={x.value} value={x.value}>
                      {x.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="lab-max-samples">
                  Max samples
                </label>
                <input
                  id="lab-max-samples"
                  type="number"
                  min={40}
                  className="field"
                  value={maxSamples}
                  onChange={(e) => setMaxSamples(Number(e.target.value))}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="lab-vqc-epochs">
                  VQC epochs
                </label>
                <input
                  id="lab-vqc-epochs"
                  type="number"
                  min={1}
                  max={30}
                  className="field"
                  value={vqcEpochs}
                  onChange={(e) => setVqcEpochs(Number(e.target.value))}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="label" htmlFor="lab-n-splits">
                  CV folds
                </label>
                <input
                  id="lab-n-splits"
                  type="number"
                  min={2}
                  max={10}
                  className="field"
                  value={nSplits}
                  onChange={(e) => setNSplits(Number(e.target.value))}
                />
              </div>
            </div>

            <hr className="divider" />

            <button type="button" onClick={launch} className="btn-primary w-full">
              <Play size={16} />
              Launch experiment
            </button>

            {error && (
              <div
                className="surface-sm row"
                style={{ borderLeft: "3px solid var(--bad)", color: "var(--bad)" }}
                role="alert"
              >
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {/* Runs and results */}
        <section className="lg:col-span-7 flex flex-col gap-6">
          <div className="surface flex flex-col gap-4">
            <div className="row">
              <h2 className="section-title">Runs</h2>
              <span className="chip num">{jobs.length}</span>
            </div>
            <hr className="divider" />

            {jobs.length === 0 ? (
              <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                No runs yet. The API executes experiments in-process, so launch times depend on the quantum budget.
              </p>
            ) : (
              <div className="flex flex-col gap-3">
                {jobs.map((j) => (
                  <div
                    key={j.job_id}
                    className="surface-sm flex flex-col gap-2"
                    style={j.status === "error" ? { borderLeft: "3px solid var(--bad)" } : undefined}
                  >
                    <div className="row flex-wrap gap-3">
                      <span style={{ fontWeight: 700 }}>
                        {j.request.experiment} · {j.request.dataset} · folds={j.request.n_splits} · cap=
                        {j.request.max_samples}
                      </span>
                      <span
                        className={`chip ${
                          j.status === "done"
                            ? "chip-good"
                            : j.status === "error"
                              ? "chip-bad"
                              : "chip-warn"
                        }`}
                      >
                        {j.status}
                      </span>
                    </div>
                    {j.status === "running" && j.progress && j.progress.length > 0 && (
                      <p className="muted" style={{ margin: 0, fontSize: 12 }}>
                        {j.progress[j.progress.length - 1]}
                      </p>
                    )}
                    {j.error && (
                      <p style={{ margin: 0, fontSize: 12, color: "var(--bad)" }}>{j.error}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {doneJobs[0]?.result ? (
            <div className="surface flex flex-col gap-4">
              <div className="row flex-wrap gap-3">
                <h2 className="section-title">Latest leaderboard</h2>
                <code className="muted num" style={{ fontSize: 12 }}>
                  {doneJobs[0].result.run_dir}
                </code>
              </div>
              <hr className="divider" />
              <div className="overflow-x-auto">
                <table className="w-full" style={{ fontSize: 13 }}>
                  <thead>
                    <tr>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        Config
                      </th>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        ROC-AUC
                      </th>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        ±std
                      </th>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        Accuracy
                      </th>
                      <th className="label" style={{ textAlign: "left", paddingBottom: 10 }}>
                        Train s
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {(doneJobs[0].result.leaderboard as Record<string, number | string>[]).map((row) => (
                      <tr key={String(row.config)} style={{ borderTop: "1px solid var(--line)" }}>
                        <td style={{ padding: "9px 16px 9px 0", fontWeight: 700 }}>{String(row.config)}</td>
                        <td className="num" style={{ padding: "9px 16px 9px 0", fontWeight: 700, color: "var(--action)" }}>
                          {Number(row.roc_auc_mean ?? 0).toFixed(3)}
                        </td>
                        <td className="num muted" style={{ padding: "9px 16px 9px 0" }}>
                          {Number(row.roc_auc_std ?? 0).toFixed(3)}
                        </td>
                        <td className="num" style={{ padding: "9px 16px 9px 0" }}>
                          {Number(row.accuracy_mean ?? 0).toFixed(3)}
                        </td>
                        <td className="num muted" style={{ padding: "9px 0" }}>
                          {Number(row.train_time_s_mean ?? 0).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="surface" style={{ minHeight: 120 }}>
              <div className="flex flex-col gap-2">
                <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                  No leaderboard to show yet. A completed run will publish its ranked configurations here.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
