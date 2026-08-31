#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

BINARY="dist/QMLKitConsole/QMLKitConsole"
if [ -f "$BINARY" ]; then
    echo "Starting packaged app: $BINARY"
    chmod +x "$BINARY"
    exec "$BINARY" "$@"
fi

echo "Packaged binary not found - falling back to source mode."
if [ ! -d "frontend/out" ]; then
    pnpm --filter frontend build
fi

exec python3 qmlkit_desktop.py --port 8000 "$@"
