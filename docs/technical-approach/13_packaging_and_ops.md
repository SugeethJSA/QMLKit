# 13 — Packaging, Tooling and Operations

Everything needed to install, run, test, package, and release QMLKit.

---

## 1. Dependency matrix

### 1.1 Python — `pyproject.toml` core

| Package | Constraint | Role |
|---|---|---|
| `numpy` | ≥1.24.0 | numerics |
| `scipy` | ≥1.10.0 | numerics |
| `pandas` | ≥2.0.0 | dataframes, markdown-table handling |
| **`pennylane`** | **≥0.40.0** | quantum circuits, QNodes, `AdamOptimizer` |
| `scikit-learn` | ≥1.4.0 | SVC, PCA, KernelPCA, MI, calibration, metrics, RF, MLP |
| `torch` | ≥2.2.0 | autoencoder, `Temporal1DCNN` |
| `fastapi` | ≥0.110.0 | HTTP + WebSocket services |
| `uvicorn` | ≥0.28.0 | ASGI server |
| `pydantic` | ≥2.6.0 | config models, request schemas |
| `matplotlib` | ≥3.8.0 | benchmark plots |
| `seaborn` | ≥0.13.0 | benchmark plots |

### 1.2 Python — `dev` extra

| Package | Constraint | Role |
|---|---|---|
| `pytest` | ≥8.0.0 | tests |
| `ruff` | ≥0.3.0 | lint |
| `httpx` | ≥0.27.0 | `TestClient` (FastAPI) — added as finding #8's fix |
| `xgboost` | ≥2.0.0 | XGBoost heads and baselines |
| `shap` | ≥0.45.0 | `QuantumKernelSHAP` |

**Critical:** `xgboost` and `shap` are in the **dev extra, not core**. Both are
guarded imports with behavioural fallbacks:

| Missing package | Fallback | Consequence |
|---|---|---|
| `xgboost` | `GradientBoostingClassifier` / `RandomForestClassifier` | leaderboard row label changes; the hybrid "XGB" arms become gradient boosting — **different model, different numbers** |
| `shap` | `shap_explainer = None` | `?deep_explain=true` silently degrades to the mean-delta approximation |

The PyInstaller spec installs `python -m pip install -e .` (core only) and does
**not** hidden-import `xgboost` or `shap` — so the packaged desktop app will use
the fallbacks even on a machine that has them. See §3.3.

### 1.3 `requirements.txt` — three defects

| Line | Content | Problem |
|---|---|---|
| 47 | `httpx2` | **typo** — should be `httpx>=0.27.0`. `httpx2` is a different, unrelated PyPI package. `pip install -r requirements.txt` installs the wrong thing |
| 42 | `jinja2>=3.1.0` | not in `pyproject.toml`; not imported anywhere in the codebase (FastAPI's default templates need it, but no Jinja templates are used) |
| — | no `xgboost` pin alignment | `requirements.txt` lists `xgboost>=2.0.0` as "Optional" but the file is a flat install — it is not actually optional here |

`requirements.txt` should be regenerated from `pyproject.toml` (e.g.
`pip install -e ".[dev]"`), or at minimum have `httpx2` → `httpx`.

### 1.4 JavaScript

Root (`package.json`): `concurrently ^10.0.5`.
Frontend (`frontend/package.json`): `next ^16.3.1`, `react ^19.2.8`,
`react-dom ^19.2.8`, `lucide-react ^1.33.0`.
Dev: `typescript ^5.9.3`, `tailwindcss ^4.3.3`, `@tailwindcss/postcss ^4.3.3`,
`eslint ^9.39.5`, `eslint-config-next ^16.3.1`, `@types/node ^26.2.0`,
`@types/react ^19.2.18`, `@types/react-dom ^19.2.4`, `@eslint/eslintrc ^3.3.1`.

**Qiskit is not a dependency and there is no Qiskit code.** Yet `README.md`
carries a Qiskit-style framing and `frontend/app/layout.tsx:82` renders a footer
reading "PennyLane · Qiskit · PyTorch · Scikit-Learn". This is finding #12 in
`docs/reports/code_evaluation_report.md:31`, still open. **Fix before presenting.**

---

## 2. Development commands

### 2.1 Setup

```bash
python -m pip install -e ".[dev]"      # backend
pnpm install                            # workspace
```

Prerequisites: Python 3.11+, Node 18+, pnpm 9+.

### 2.2 Root `package.json` scripts

| Script | Command |
|---|---|
| `dev` | `concurrently -n "BACKEND,FRONTEND" -c "magenta,cyan" "pnpm run backend:dev" "pnpm run frontend:dev"` |
| `start` | alias for `dev` |
| `frontend:dev` | `pnpm --filter frontend dev` |
| `frontend:build` | `pnpm --filter frontend build` |
| `frontend:start` | `pnpm --filter frontend start` |
| `frontend:lint` | `pnpm --filter frontend lint` |
| `backend:install` | `python -m pip install -e ".[dev]"` |
| `backend:dev` | `python -m uvicorn --app-dir src qmlkit.api.kennel_server:app --host 0.0.0.0 --port 8000 --reload` |
| `backend:benchmark` | `python scripts/run_benchmark.py --source markdown` |
| `desktop:build` | `powershell -ExecutionPolicy Bypass -File scripts/build_desktop.ps1` |
| `desktop:start` | `powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1` |

`pnpm dev` gives backend :8000 + frontend :3000 concurrently. The console
auto-detects the API host; override with `?api=http://host:port`.

### 2.3 Python CLIs

| Script | Purpose |
|---|---|
| `scripts/run_benchmark.py` | single-split quantum vs classical benchmark + CSV/PNG/metadata |
| `scripts/run_hybrid_search.py` | CV lab: search / map_ablation / modality_ablation / robustness |
| `scripts/train_kennel_model.py` | kennel RandomForest + LOSO → `models/kennel_model.joblib` |
| `scripts/build_ecg_tables.py` | `docs/*.md` → `data/ecg/*.csv` |
| `scripts/explain_patient.py` | per-patient explanation report |
| `scripts/generate_voc_data.py` | synthetic cohort export |
| `scripts/launch_portal.py` | launches the screening portal |
| `qmlkit_desktop.py` | desktop launcher (uvicorn + pywebview) |

`run_benchmark.py` and `run_hybrid_search.py` both `sys.path.insert(0, ../src)`
so they run without installing; both also `sys.stdout.reconfigure(encoding="utf-8")`
for Windows console safety.

### 2.4 Quality gates

```bash
python -m pytest -q
python -m ruff check src scripts tests packaging qmlkit_desktop.py
pnpm --filter frontend lint
pnpm --filter frontend build
```

Ruff config (`pyproject.toml:47-61`): `line-length = 120`,
`target-version = "py311"`, `extend-exclude = ["frontend", "docs"]`,
`select = ["E4","E7","E9","F","I","B"]` (real errors, pyflakes, import hygiene,
bug-prone patterns), per-file-ignore `"scripts/*" = ["F541"]`.

**Gaps in the gate set:**

| Missing gate | Consequence |
|---|---|
| **No CI workflow runs tests or lint on PRs/pushes** | `.github/workflows/ci.yml` triggers only on `version.properties` changes to `main`, or `workflow_dispatch`. Nothing validates a normal PR |
| No `mypy` / `pyright` | the codebase has type hints but no static type checking |
| No frontend `typecheck` script | TS errors surface only at `next build` |
| No coverage measurement | 59 tests, no coverage report |
| No bandit / pip-audit / dependency scanning | — |

### 2.5 Test inventory

`pyproject.toml:63-67`: `testpaths = ["tests"]`, `python_files = ["test_*.py"]`.
`conftest.py` inserts `src` on `sys.path`.

| File | Tests | Focus |
|---|---|---|
| `test_lab_pipeline.py` | **19** | `HybridPipeline`, CV harness, all 4 experiment types, registry, kennel synth, ensembles, `default_presets` |
| `test_hardware_ingest.py` | **14** | protocol parse/validate, simulation source, 45-feature extraction, recording manager, FastAPI + WS |
| `test_dataset_loader.py` | **13** | markdown parser, Lung VOC loader (incl. the real bundled file), balanced subsample, CSV, dog ECG |
| `test_data_pipeline.py` | 4 | selector consistency, VOC cohort shapes, leak-free split, angle bounds + inverse transform |
| `test_classical_baselines.py` | 3 | all baselines fit/predict, `Temporal1DCNN`, `compute_clinical_metrics` |
| `test_quantum_circuits.py` | 3 | BioZZ Gram properties (symmetry/unit-diagonal/[0,1]), QSVM, VQC training |
| `test_e2e_workflow.py` | 2 | `BenchmarkSuite` end-to-end, FastAPI health/portal/predict |
| `test_explainability.py` | 1 | `BiomarkerAttributionEngine` |
| **Total** | **59** | |

`docs/reports/code_evaluation_report.md:46` says "40 passed" — that was true
before `test_lab_pipeline.py` (19 tests) was added. The report is stale.

**Notable untested modules:** `quantum/qrc.py` (`QuantumReservoirClassifier`),
`evaluation/hardware_profiler.py` (all 3 `profile_*` +
`evaluate_shot_noise_robustness`), `explainability/quantum_shap.py`,
`api/training_service.py`, `api/kennel_server.py` lab routes, `qmlkit_desktop.py`,
`preprocessor.save/load`, `selector.save/load`, `evaluation`'s metric-fallback
branches, `lab/registry.py::finalise` content.

---

## 3. Desktop packaging

### 3.1 `qmlkit_desktop.py` — the launcher

```python
def resource_root() -> Path:
    if sys.frozen:
        if hasattr(sys, "_MEIPASS"): return Path(sys._MEIPASS).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def frontend_dist() -> Path | None:                 # first existing wins
    candidates = [exe_dir/"frontend_out", exe_dir/"frontend"/"out",       # frozen only
                  root/"frontend_out",    root/"frontend"/"out", root.parent/"frontend"/"out"]
```

`main()` (`:55-108`):

1. `--port` (default 8000), `--browser` flag.
2. Inserts `root/src` on `sys.path` when running from source.
3. `os.environ.setdefault("QMLKIT_FRONTEND_DIST", str(frontend_dist() or ""))`.
4. `uvicorn.Config("qmlkit.api.kennel_server:app", host="127.0.0.1", port=…)`,
   `factory=False`; runs `server.run` in a **daemon thread**.
5. Waits up to **60 × 0.25 s = 15 s** for `server.started`.
6. `webview.create_window("QMLKit Kennel Console", url, width=1280, height=860)`
   then `webview.start()`; on `ImportError` falls back to `webbrowser.open(url)`.
7. `server.should_exit = True`.

**Hot-swap property:** because `frontend_dist()` prefers an external
`frontend_out/` over the embedded snapshot, a developer can drop a rebuilt
console next to the `.exe` and it takes effect on the next launch, with no
PyInstaller rebuild. This is a deliberate and well-executed choice.

**Bound to `127.0.0.1`** — localhost only, which is the right default for a
local research tool and does mitigate the open CORS policy in practice.

### 3.2 `scripts/build_desktop.ps1` — 3-stage local build

```
[1/3] pnpm --filter frontend build                    (skipped if frontend/ absent)
[2/3] pip show pyinstaller → install if missing
      python -c "import qmlkit" → pip install -e ".[dev]" if it fails
[3/3] python -m PyInstaller packaging/qmlkit_desktop.spec
        --noconfirm --distpath dist --workpath build/pyinstaller
      → dist/QMLKitConsole/QMLKitConsole.exe
```

Then **optionally** an Inno Setup installer: probes
`%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe`, `%ProgramFiles%\…`,
`%LOCALAPPDATA%\Programs\…`; if found **and** `packaging/installer/qmlkit.iss`
exists, parses `version` from `version.properties` (fallback `"0.1.0"`), makes
`dist-artifacts`, and runs
`& $iscc "/DMyAppVersion=$ver" "packaging/installer/qmlkit.iss"`.

### 3.3 `packaging/qmlkit_desktop.spec`

```python
frontend_out = ROOT / "frontend" / "out"
datas = [(".../frontend/out", "frontend_out")] if frontend_out.is_dir() else []

Analysis([ROOT/"qmlkit_desktop.py"], pathex=[ROOT/"src"], datas=datas,
    hiddenimports=[uvicorn.logging, uvicorn.loops(.auto), uvicorn.protocols(.http/.http.auto/
                   .websockets/.websockets.auto), uvicorn.lifespan(.on),
                   qmlkit.api.kennel_server, qmlkit.api.server, torch, pennylane],
    excludes=["tkinter"])

EXE(..., exclude_binaries=True, name="QMLKitConsole", console=False, upx=False)
COLLECT(..., name="QMLKitConsole")            # ONEDIR

if sys.platform == "darwin":
    BUNDLE(coll, name="QMLKitConsole.app", bundle_identifier="com.qmlkit.console",
           info_plist={CFBundleName "QMLKit Console", CFBundleDisplayName "QMLKit Console",
                       CFBundleVersion "0.1.0", CFBundlePackageType "APPL",
                       CFBundleSignature "????", NSHighResolutionCapable "True"})
```

**Onedir + windowed** — the right combination: onedir avoids PyInstaller's
onefile extraction latency and temp-dir antivirus false positives, and
`console=False` produces a native window (comment: *"logs go to
`<app dir>/logs/qmlkit.log`"* — note that logging is configured via
`logging.basicConfig` in `qmlkit_desktop.py:21` writing to **stderr**, which is
`None` in a windowed app, so this log-file claim is not actually implemented).

**Missing hidden imports:** `xgboost`, `shap`, `lightgbm`, `pywebview`. The
consequences:

| Missing | Effect in the packaged app |
|---|---|
| `xgboost` | hybrid "XGB" heads fall back to `GradientBoostingClassifier`; the classical `XGBoost` baseline row becomes `Gradient_Boosting`. **Leaderboard differs from source mode.** |
| `shap` | `deep_explain=true` silently degrades to mean-delta |
| `pywebview` | always falls back to `webbrowser.open` — the "native window" is never shown, so the app opens a browser tab instead. `pywebview` is not in `pyproject.toml` **at all** |

Adding `pywebview` to a `desktop` extra and all three to `hiddenimports` is a
5-minute fix that makes the packaged app match the source-mode behaviour.

### 3.4 `packaging/installer/qmlkit.iss` (Inno Setup)

```
AppId={{E8B83742-1279-4B52-9216-B549A4C071D2}
MyAppName     "QMLKit Console"
MyAppPublisher "QMLKit Development Team"
MyAppURL      "https://github.com/SugeethJSA/QMLKit"
MyAppExeName  "QMLKitConsole.exe"

DefaultDirName={autopf}\{#MyAppName}
Compression=lzma2/max; SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest; PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\..\dist-artifacts
OutputBaseFilename=QMLKitConsole-v{#MyAppVersion}-windows-x64-installer
```

Files `..\..\dist\QMLKitConsole\*` → `{app}` with
`ignoreversion recursesubdirs createallsubdirs`; optional unchecked `desktopicon`
task; postinstall run entry.

`PrivilegesRequired=lowest` + `OverridesAllowed=dialog` means a non-admin user
installs to a user-writable location — appropriate for a research tool that
should not require elevation.

### 3.5 `packaging/linux/`

| File | Purpose |
|---|---|
| `build_deb.sh` | native Debian package |
| `install.sh` | self-installing tarball helper |
| `uninstall.sh` | removal helper |
| `qmlkit-console.desktop` | App Launcher `.desktop` entry |

---

## 4. CI/CD — `.github/workflows/ci.yml` (585 lines)

### 4.1 Trigger

```yaml
on:
  push:
    branches: [main]
    paths: ["version.properties"]
  workflow_dispatch:
```

**Release-on-version-bump.** Nothing runs on a pull request or on an ordinary
push. There is no test job, no lint job, no build-verification job.

### 4.2 Jobs

| Job | OS | Output |
|---|---|---|
| `prep-release` | ubuntu-latest | parses `version`, gates on the file actually changing, checks tag existence, builds the release body |
| `build-binaries` (matrix) | windows-latest / macos-latest / ubuntu-latest | PyInstaller bundle + platform installer |
| `publish-release` | ubuntu-latest | GitHub Release via `softprops/action-gh-release@v2`, `make_latest: true` |

### 4.3 `prep-release` logic

1. **Gate** — on push, verify `version.properties` appears in
   `git diff --name-only before..sha`; handles the initial-push
   (`before == 000…0`) and missing-history cases by falling back to `HEAD~1`.
2. **Version parse** — a fairly elaborate tolerant parser: tries
   `^\s*[^#]*version\s*[:=]`, falls back to the first non-comment line, strips a
   leading `v`, warns (but proceeds) if the result is not semver-shaped.
3. **Bump check** — compares against `git show HEAD~1:version.properties`.
4. **Tag check** — `git tag --list` plus `git ls-remote --tags origin`;
   idempotent skip if the tag exists and the version was not bumped.
5. **Release notes** — if `releasenotes.properties` changed (or exists and is
   non-empty on manual dispatch), use it; the embedded Python parser strips
   single-`#` properties comments while **preserving `##` markdown headers**,
   then tries a `key: value` / `key=value` form before falling back to treating
   the whole file as markdown. Otherwise it tries the GitHub
   `generate-notes` API, then falls back to a `git log` diff listing with a
   compare URL.

The version-parsing robustness is unusual and deliberate (comment: *"attempting
to handle whatever format a user might drop in"*) — it will not fail the
pipeline on a malformed `version.properties` unless nothing at all can be
parsed.

### 4.4 `build-binaries` per platform

Common: Node 20 + pnpm 9 → `pnpm install` → `pnpm --filter frontend build` →
Python 3.11 with `cache: "pip"` → `pip install PyInstaller` + `pip install -e .`
→ PyInstaller.

| Platform | Installer | Also |
|---|---|---|
| Windows | Inno Setup `.exe` via `choco install innosetup` + `ISCC.exe /DMyAppVersion=$ver` | portable `Compress-Archive` zip |
| macOS | `.dmg` via `hdiutil create -volname "QMLKit Console" … -format UDZO` with an `/Applications` symlink for drag-and-drop | `.app` zip |
| Linux | `.deb` via `bash packaging/linux/build_deb.sh "$VERSION" …` | self-installing `.tar.gz` with `install.sh`/`uninstall.sh` |

All upload via `actions/upload-artifact@v4` with `retention-days: 1`.

**Note:** `pip install -e .` installs **core only** — so `xgboost` and `shap` are
absent in the CI-built binaries, confirming §1.2: the released desktop app runs
with the classical fallbacks. If the released app's numbers are quoted anywhere,
this must be disclosed.

### 4.5 `publish-release`

Downloads the notes and all `binary-*` artifacts (merged), previews both, then
creates the release with `generate_release_notes: false` (the body always comes
from the prepared file) and `make_latest: true`. Finally writes a job summary
table classifying each asset by filename suffix (installer / portable / dmg /
app.zip / deb / tar.gz).

### 4.6 CI review checklist

| # | Gap | Impact |
|---|---|---|
| 1 | **No test or lint job on PRs** | regressions can land unreviewed by any gate |
| 2 | `pip install -e .` (core only) in the build matrix | released binaries lack xgboost/shap ⇒ different leaderboard than source |
| 3 | `pywebview` never installed | "native window" always degrades to a browser tab |
| 4 | `build-binaries` is gated on a `version.properties` bump | no CI signal that the build still works on ordinary changes |
| 5 | `macos-latest` ⇒ arm64 only | no Intel-Mac artifact |
| 6 | No SBOM, no dependency scanning, no signed artifacts | supply-chain posture is minimal |
| 7 | The `prep-release` version parser is very permissive | a typo'd version publishes a bad tag |

---

## 5. Configuration

### 5.1 Pydantic config models (`src/qmlkit/config.py`)

`SensorArrayConfig`, `VOCBiomarkerConfig`, `QuantumModelConfig`,
`BenchmarkConfig`, and the aggregate `QMLKitAppConfig`. The models are declared
and validated but **`QMLKitAppConfig` is never instantiated anywhere** — the
running system is configured by CLI flags, environment variables, and
`KennelSettings`, not by these models. They are documentation-as-code for the
design space, plus the source of `VOCBiomarkerConfig.compounds` used by the
explainability layer.

### 5.2 Environment variables (the real runtime configuration)

| Variable | Default | Consumer |
|---|---|---|
| `QMLKIT_KENNEL_ESP32_IP` | `192.168.4.1` | `KennelSettings.esp32_host` |
| `QMLKIT_KENNEL_ESP32_PORT` | `3333` | `KennelSettings.esp32_port` |
| `QMLKIT_KENNEL_STREAM_SOURCE` | `auto` | `KennelSettings.source` |
| `QMLKIT_KENNEL_BATCH_SIZE` | `400` | inference window |
| `QMLKIT_KENNEL_CONFIDENCE_THRESHOLD` | `0.6` | ok/uncertain gate |
| `QMLKIT_KENNEL_SMOOTHING_ALPHA` | `0.3` | EMA |
| `QMLKIT_KENNEL_DATA_DIR` | `data/kennel` | recording output |
| `QMLKIT_FRONTEND_DIST` | *(unset)* | static console mount |

`QMLKIT_KENNEL_ESP32_IP` should be set to the board's DHCP address or
`kennel.local` — the `192.168.4.1` default assumes a SoftAP the firmware does not
start.

### 5.3 Determinism

`set_seed(seed)` seeds `random`, `numpy`, `torch`, CUDA, and `PYTHONHASHSEED`.
See [`03_preprocessing_and_reduction.md`](03_preprocessing_and_reduction.md) §7
for the important caveat that most quantum components carry their own **fixed
internal seed 42**, so `--seed` does not change their randomness.

---

## 6. Operations runbook

### 6.1 First run

```bash
python -m pip install -e ".[dev]"
pnpm install
pnpm dev                                  # backend :8000 + frontend :3000
```

Open `http://localhost:3000`. Expect:

- `/` — kennel diagram populating from the **simulator** (24 s state cycle:
  8 s IDLE → 8 s SNIFF → 8 s COOLDOWN), IMU waveforms scrolling, sensor bars
  moving, "Total Ingested Frames" incrementing at 10 Hz.
- `/diagnostics` — `PredictionCard` in the **"Model Awaiting Calibration"**
  amber state (no `models/kennel_model.joblib`).
- `/train` — empty state "No Active Model Trained Yet"; the QSVM BioZZ run
  completes in ~1 s on a synthetic cohort.
- `/collect` — Stream Source "Local Sim".
- `/lab` — launch `search` with `vqc-epochs: 3` to get a leaderboard in ~2 min
  instead of ~10.

### 6.2 Bringing up real hardware

1. Create `firmware/kennel_node/secrets.h` (gitignored, **required**):
   ```c
   #define WIFI_SSID     "your-ssid"
   #define WIFI_PASSWORD "your-password"
   ```
2. Wire per `firmware/kennel_node/README.md:20-30`; flash with
   `pio run -t upload -e kennel`; open `pio device monitor` (115200).
3. Confirm the board answers `socket.create_connection(("<ip>", 3333), timeout=2)`.
4. **Start the backend *after* the board is up** — the `auto` probe is one-shot.
   Or set `QMLKIT_KENNEL_STREAM_SOURCE=hardware` to skip the probe entirely.
5. Verify `GET /api/v1/kennel/state` reports
   `"stream_source": "hardware", "connected": true`.

### 6.3 Producing a kennel model

Prerequisites: **≥2 dogs**, recordings under `data/kennel/cancer/` and
`data/kennel/<other-label>/`. The trainer `SystemExit`s if no label lowercases to
`"cancer"`.

```bash
# 1. record sessions via the console /collect page (or curl)
# 2. train
python scripts/train_kennel_model.py --data-dir data/kennel --out-dir models --n-estimators 300
```

Verify `models/kennel_training_report.json` has non-null
`loso_accuracy_mean` — that confirms at least one usable LOSO fold. Then restart
the backend; `/api/v1/kennel/state` should report `model_status: "ready"`.

### 6.4 Reproducing the published results

```bash
# real-data single-split benchmark (docs/reports/real_data_benchmark_report.md)
python scripts/run_benchmark.py --source markdown --voc-task cancer_vs_control \
    --max-samples 120 --vqc-epochs 8 --n-qubits 6 --output-dir outputs/benchmark_real

# 5-fold hybrid leaderboard (docs/reports/hybrid_search_report.md)
python scripts/run_hybrid_search.py --dataset voc_real --experiment search \
    --max-samples 120 --vqc-epochs 6 --n-splits 5

# feature-map ablation (RQ3)
python scripts/run_hybrid_search.py --dataset voc_real --experiment map_ablation

# noise robustness
python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness
```

**Expected wall times** (from the committed logs, 6 qubits): search ~10 min
(dominated by `Stacked-LR` at 135.7 s and the VQC arms); map_ablation ~1.5 min;
robustness ~2 min; benchmark ~6 min (VQC 343.5 s dominates).

### 6.5 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `stream_source: "simulation"` with hardware connected | one-shot probe ran before the board booted | restart the backend, or set `QMLKIT_KENNEL_STREAM_SOURCE=hardware` |
| `model_status: "untrained"` | no `models/kennel_model.joblib` | record ≥2 dogs across ≥2 labels, then run the trainer |
| `loso_accuracy_mean: null` in the report | only one dog in the corpus | add recordings for a second dog |
| Dashboard shows zero bars, no diagram | no frame received yet | check `connected` on `/api/v1/kennel/state` |
| `acc_band_tremor_4_8hz` always 0 | 10 Hz wire rate ⇒ the FFT axis tops out at 5 Hz | see [`10_kennel_hardware.md`](10_kennel_hardware.md) §7.1 |
| `/lab` robustness run shows `error: "'leaderboard'"` | the API reads a key `run_robustness` does not return | use the CLI, or fix `kennel_server.py:449` |
| Packaged app opens a browser tab, not a window | `pywebview` is not a dependency and not hidden-imported | add it to a `desktop` extra + `hiddenimports` |
| Leaderboard differs between source and the packaged app | CI installs core only ⇒ no `xgboost`/`shap` | install `.[dev]` in the build matrix |
| `pip install -r requirements.txt` installs the wrong httpx | `httpx2` typo | use `pip install -e ".[dev]"` |
