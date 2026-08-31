# Workstream E — Packaging (repomono methodology)

Replicated from `C:\Users\sugee\Github\repomono` root + GAIT project:

## E1. pnpm workspace root

`package.json` scripts (concurrently-orchestrated):

| Script | Action |
|---|---|
| `pnpm dev` | backend (uvicorn) + frontend (next dev) concurrently |
| `backend:install` | pip install -r requirements.txt |
| `backend:dev` | `python -m uvicorn --app-dir src qmlkit.api.server:app --port 8000 --reload` |
| `frontend:dev/build/start/lint` | pnpm --filter frontend … |
| `desktop:build` | powershell scripts/build_desktop.ps1 |
| `desktop:start` | powershell scripts/start_app.ps1 |

`pnpm-workspace.yaml`: `packages: ["frontend"]`.

## E2. Desktop bundle

```
scripts/
├── build_desktop.ps1   # frontend static export → PyInstaller onedir bundle
└── start_app.ps1       # launch exe; fallback to python server + browser
packaging/
└── qmlkit_desktop.spec # PyInstaller: windowed, onedir, bundles src/qmlkit +
                         models/ + embedded frontend/out snapshot
qmlkit_desktop.py        # Entry point: uvicorn server in-process (pywebview
                         window if available, else browser mode)
```

Runtime layout preference (GAIT rule): external `frontend_out/` next to the exe
overrides the embedded snapshot → UI hot-swappable without repackaging.
Logs → `<app dir>/logs/qmlkit.log`. Optional Inno Setup installer script when
ISCC.exe present.

## E3. Release flow (Multi-platform runnable binaries)

`.github/workflows/ci.yml` builds and packages runnable binaries across 3 matrix targets whenever `version.properties` is bumped on `main` (or on `workflow_dispatch`):

- **Windows x64** (`windows-latest`): `QMLKitConsole-v<tag>-windows-x64.zip`
- **macOS Apple Silicon arm64** (`macos-latest`): `QMLKitConsole-v<tag>-macos-arm64.tar.gz` & `.zip`
- **macOS Intel x64** (`macos-13`): `QMLKitConsole-v<tag>-macos-x64.tar.gz` & `.zip`
- **Linux x64** (`ubuntu-latest`): `QMLKitConsole-v<tag>-linux-x64.tar.gz` & `.zip`

All compiled bundles are automatically attached as downloadable assets to the published GitHub Release with custom or smart auto-generated changelog notes.
