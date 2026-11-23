# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Three primary audiences, confirmed together:

- **Researchers and evaluators** — investigate hybrid quantum/classical screening performance, VOC biomarker attribution, and live kennel telemetry.
- **Kennel operators** — run live hardware sessions, record labelled canine trials, and monitor sensor/stream health.
- **Clinical and lab reviewers** — inspect individual screening outputs and biomarker attributions.

## Product Purpose

QMLKit is an integrated screening platform: ESP32 kennel-node telemetry, leak-free benchmarking of quantum versus classical models on real VOC biomarker data, and a Next.js console that makes the whole loop observable. The console exposes live kennel state, on-demand explainable diagnostics, guided session recording, model training/tuning, and hybrid experiment runs.

Purpose: support early-disease-detection research under Problem Statement 26139. Success, inferred from the repository: a user can trace a screening result back to real data, a labelled kennel session, and a documented model configuration rather than an opaque demo.

## Positioning

Canine-biomimetic olfactory VOC sensing combined with NISQ-compatible quantum feature maps (BioZZ covariance-weighted ZZ), QSVM fidelity kernels and VQC classifiers, plus an instrumented examination kennel that captures canine micro-movement during sniffing. Models are benchmarked leak-free against classical baselines on a real VOC dataset. The user confirmed this mechanism and comparison story as durable positioning for the console.

## Operating Context

- ESP32 kennel node streams TCP JSON (~100 Hz) from 4 corner load cells, 6 IR proximity sensors, 2 ultrasonic sensors, and a collar-mounted MPU6050. A simulation fallback supplies telemetry when hardware is absent.
- FastAPI ingests telemetry and serves the console; WebSocket `/ws/stream` drives live UI (~10 Hz) and `/ws/diagnostic` is event-driven. REST helpers live under `/api/v1/...`.
- Workflows: monitor the live dashboard, run an on-demand VOC diagnostic with explainability, train/tune models in the Training Studio, record labelled sessions in the Data Lab, and launch hybrid experiments in the lab page.
- The console is a static export (`output: "export"`) served by the backend, with a `?api=http://host:port` host override. A PyInstaller desktop bundle can embed the exported snapshot.
- Diagnostics report `untrained` until a model artifact exists; this is a normal, expected state.

## Capabilities and Constraints

- Web platform: Next.js 16, React 19, TypeScript 5.9, Tailwind CSS v4; static export, no SSR and no app API routes. The existing codebase answers the stack question.
- The 427-sample real lung VOC dataset and its documented caveats are the primary evidence.
- Kennel telemetry, labelled sessions, and the quantum-vs-classical benchmarking story must be preserved.
- Claim boundary: a prototype / PS-26139 demo. Ambitious claims are acceptable only when framed as experimental; no validated clinical product is implied.
- No fabricated evidence. Seeded fake result series, invented preset readings presented as real, and capability claims unsupported by code (for example a Qiskit dependency that does not exist) must not be presented as real output.
- User constraint, verbatim: "No false code should be there." Code and UI must not present placeholder, stub, or invented behaviour as real functionality; synthetic demo inputs must be visibly labelled as such.

## Brand Commitments

- Name: QMLKit. The PS-26139 problem-statement identity is part of the product's framing.
- Voice is technical/research-oriented; the console frames itself as a quantum screening laboratory. No additional binding aesthetic direction was given.

## Evidence on Hand

- `docs/Lung_Cancer_VOC_Dataset_427.md` — real VOC dataset.
- `docs/reports/dataset_integration_report.md` — dataset schemas, mappings, caveats.
- `docs/reports/real_data_benchmark_report.md` — quantum vs classical results on real data.
- `docs/init/` — original design corpus; `docs/upgrade/` and `docs/technical-approach/` — integration and implementation documentation.
- Live kennel telemetry and simulation fallback as runtime evidence.
- Absences that future work must not fabricate: no trained model artifact in this checkout (diagnostics report `untrained`), no Qiskit dependency or code, no validated clinical evidence, testimonials, or deployment claims.

## Product Principles

1. Evidence before presentation: real VOC data and real telemetry drive results; synthetic inputs are labelled as synthetic.
2. Honest system state: untrained, uncertain, and disconnected states are first-class, not hidden.
3. End-to-end traceability: session capture → training and benchmarking → diagnostic → biomarker attribution.
4. Research-first framing: every claim is experimental and tied to PS-26139.
5. One shared live truth for operators, researchers, and reviewers.
