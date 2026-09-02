#!/usr/bin/env bash
# Build Debian/Ubuntu package (.deb) for QMLKit Console
set -euo pipefail

VERSION="${1:-0.1.0}"
VERSION="${VERSION#v}" # strip leading v if present
DIST_DIR="${2:-dist/QMLKitConsole}"
OUTPUT_DIR="${3:-dist-artifacts}"

mkdir -p "$OUTPUT_DIR"
TMP_DIR="$(mktemp -d)"
DEB_ROOT="${TMP_DIR}/qmlkit-console_${VERSION}_amd64"

mkdir -p "$DEB_ROOT/opt/qmlkit"
mkdir -p "$DEB_ROOT/usr/bin"
mkdir -p "$DEB_ROOT/usr/share/applications"
mkdir -p "$DEB_ROOT/DEBIAN"

# Copy binary bundle to /opt/qmlkit
cp -r "$DIST_DIR"/* "$DEB_ROOT/opt/qmlkit/"
chmod +x "$DEB_ROOT/opt/qmlkit/QMLKitConsole"

# Create symlink in /usr/bin
ln -sf /opt/qmlkit/QMLKitConsole "$DEB_ROOT/usr/bin/qmlkit-console"

# Copy desktop file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/qmlkit-console.desktop" "$DEB_ROOT/usr/share/applications/"

# Create control file
cat << EOF > "$DEB_ROOT/DEBIAN/control"
Package: qmlkit-console
Version: ${VERSION}
Section: science
Priority: optional
Architecture: amd64
Maintainer: QMLKit Development Team <dev@qmlkit.org>
Description: Hybrid Quantum ML Platform for Olfactory VOC Sensing
 QMLKit provides real-time canine telemetry ingestion, biomedical feature
 extraction, quantum machine learning (QSVM, VQC, QCNN), and diagnostic
 benchmarking with a Next.js console interface.
EOF

dpkg-deb --build --root-owner-group "$DEB_ROOT" "$OUTPUT_DIR/QMLKitConsole-v${VERSION}-linux-x64.deb"
rm -rf "$TMP_DIR"
echo "[OK] Built Debian package: $OUTPUT_DIR/QMLKitConsole-v${VERSION}-linux-x64.deb"
