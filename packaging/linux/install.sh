#!/usr/bin/env bash
# Universal Linux Installer for QMLKit Console
set -euo pipefail

INSTALL_DIR="${HOME}/.local/share/qmlkit"
BIN_DIR="${HOME}/.local/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing QMLKit Console to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}" "${BIN_DIR}" "${DESKTOP_DIR}"

# Copy application files
cp -r "${SCRIPT_DIR}"/* "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/QMLKitConsole"

# Create symlink in ~/.local/bin
ln -sf "${INSTALL_DIR}/QMLKitConsole" "${BIN_DIR}/qmlkit-console"

# Create desktop launcher
cat << EOF > "${DESKTOP_DIR}/qmlkit-console.desktop"
[Desktop Entry]
Version=1.0
Type=Application
Name=QMLKit Console
GenericName=Hybrid Quantum ML Olfactory VOC Platform
Comment=Hybrid Quantum ML Olfactory VOC Sensing & Kennel Diagnostics
Exec=${INSTALL_DIR}/QMLKitConsole
Terminal=false
Categories=Science;MedicalSoftware;Education;Development;
EOF

chmod +x "${DESKTOP_DIR}/qmlkit-console.desktop"

echo ""
echo "[OK] Installation successful!"
echo "You can launch QMLKit Console from your application menu or run: qmlkit-console"
