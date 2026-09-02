# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the QMLKit desktop bundle (repomono GAIT methodology).
# Build:  pyinstaller packaging/qmlkit_desktop.spec --noconfirm
# Output: dist/QMLKitConsole/QMLKitConsole[.exe] (onedir, windowed) and dist/QMLKitConsole.app on macOS

import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

# Embedded frontend snapshot (external frontend_out/ overrides at runtime).
frontend_out = ROOT / "frontend" / "out"
datas = []
if frontend_out.is_dir():
    datas.append((str(frontend_out), "frontend_out"))

a = Analysis(
    [str(ROOT / "qmlkit_desktop.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "qmlkit.api.kennel_server",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "seaborn", "torch", "pennylane"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QMLKitConsole",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # native window app; logs go to <app dir>/logs/qmlkit.log
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="QMLKitConsole",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="QMLKitConsole.app",
        bundle_identifier="com.qmlkit.console",
        info_plist={
            "CFBundleName": "QMLKit Console",
            "CFBundleDisplayName": "QMLKit Console",
            "CFBundleIdentifier": "com.qmlkit.console",
            "CFBundleVersion": "0.1.0",
            "CFBundlePackageType": "APPL",
            "CFBundleSignature": "????",
            "NSHighResolutionCapable": "True",
        },
    )
