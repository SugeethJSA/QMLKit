#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "==> [1/3] Building frontend static export"
if [ -d "frontend" ]; then
    pnpm --filter frontend build
else
    echo "    no frontend dir - skipping"
fi

echo "==> [2/3] Ensuring python deps"
python3 -m pip show pyinstaller >/dev/null 2>&1 || python3 -m pip install pyinstaller
if ! python3 -c "import qmlkit" >/dev/null 2>&1; then
    echo "    installing qmlkit (editable)"
    python3 -m pip install -e ".[dev]"
fi

echo "==> [3/3] PyInstaller bundle"
python3 -m PyInstaller packaging/qmlkit_desktop.spec --noconfirm --distpath dist --workpath build/pyinstaller

echo ""
echo "[OK] Bundle ready in dist/QMLKitConsole/"
