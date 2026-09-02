#!/usr/bin/env bash
# Build the QMLKit desktop bundle and native installer for Unix (macOS / Linux).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

VERSION="0.1.0"
if [ -f "version.properties" ]; then
    VERSION=$(grep -iE '^\s*version\s*=' version.properties | sed -E 's/^\s*version\s*=\s*//' | tr -d '\r' || echo "0.1.0")
fi

echo "==> [1/4] Building frontend static export"
if [ -d "frontend" ]; then
    pnpm --filter frontend build
else
    echo "    no frontend dir - skipping"
fi

echo "==> [2/4] Ensuring python deps"
python3 -m pip show pyinstaller >/dev/null 2>&1 || python3 -m pip install pyinstaller
if ! python3 -c "import qmlkit" >/dev/null 2>&1; then
    echo "    installing qmlkit (editable)"
    python3 -m pip install -e ".[dev]"
fi

echo "==> [3/4] PyInstaller bundle"
python3 -m PyInstaller packaging/qmlkit_desktop.spec --noconfirm --distpath dist --workpath build/pyinstaller

echo ""
echo "==> [4/4] Building native installer"
mkdir -p dist-artifacts

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: build .dmg with /Applications link
    if [ -d "dist/QMLKitConsole.app" ] && command -v hdiutil >/dev/null 2>&1; then
        echo "Building macOS .dmg installer..."
        DMG_TMP="$(mktemp -d)"
        cp -R "dist/QMLKitConsole.app" "$DMG_TMP/"
        ln -s /Applications "$DMG_TMP/Applications"
        hdiutil create -volname "QMLKit Console" -srcfolder "$DMG_TMP" -ov -format UDZO "dist-artifacts/QMLKitConsole-v${VERSION}-macos-arm64.dmg"
        rm -rf "$DMG_TMP"
        echo "[OK] macOS DMG installer created: dist-artifacts/QMLKitConsole-v${VERSION}-macos-arm64.dmg"
    fi
else
    # Linux: build .deb package and self-installer bundle
    if command -v dpkg-deb >/dev/null 2>&1; then
        echo "Building Debian package (.deb)..."
        bash packaging/linux/build_deb.sh "$VERSION" "dist/QMLKitConsole" "dist-artifacts"
    fi

    # Package self-installing tarball
    cp packaging/linux/install.sh packaging/linux/uninstall.sh dist/QMLKitConsole/
    chmod +x dist/QMLKitConsole/install.sh dist/QMLKitConsole/uninstall.sh dist/QMLKitConsole/QMLKitConsole
    tar -czf "dist-artifacts/QMLKitConsole-v${VERSION}-linux-x64.tar.gz" -C dist QMLKitConsole
    echo "[OK] Linux bundle created: dist-artifacts/QMLKitConsole-v${VERSION}-linux-x64.tar.gz"
fi

echo ""
echo "[OK] All artifacts ready under dist-artifacts/"
