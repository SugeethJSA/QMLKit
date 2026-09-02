#!/usr/bin/env bash
# Uninstaller for QMLKit Console
set -euo pipefail

INSTALL_DIR="${HOME}/.local/share/qmlkit"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"

echo "==> Uninstalling QMLKit Console..."
rm -rf "${INSTALL_DIR}"
rm -f "${BIN_DIR}/qmlkit-console"
rm -f "${DESKTOP_DIR}/qmlkit-console.desktop"

echo "[OK] QMLKit Console has been uninstalled."
