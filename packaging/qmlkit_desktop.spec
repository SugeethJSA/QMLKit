# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the QMLKit desktop bundle (repomono GAIT methodology).
# Build:  pyinstaller packaging/qmlkit_desktop.spec --noconfirm
# Output: dist/QMLKitConsole/QMLKitConsole.exe  (onedir, windowed)

import os
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

a = Analysis(
    [str(ROOT / "qmlkit_desktop.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        # Embedded frontend snapshot (external frontend_out/ overrides at runtime).
        *((str(ROOT / "frontend" / "out"), "frontend_out"),)
        if (ROOT / "frontend" / "out").is_dir()
        else [],
    ],
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
