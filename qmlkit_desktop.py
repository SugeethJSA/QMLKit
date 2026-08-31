"""QMLKit desktop launcher (repomono GAIT pattern).

Runs the kennel FastAPI server in-process under uvicorn, then opens the
dashboard either in a native WebView2 window (pywebview) or the default
browser.

Usage:
  python qmlkit_desktop.py [--port 8000] [--browser] [--app-dir src]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qmlkit-desktop")


def resource_root() -> Path:
    """Directory containing bundled assets (works from source and PyInstaller)."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def frontend_dist() -> Path | None:
    """Prefer an external hot-swappable folder, else the embedded snapshot."""
    root = resource_root()
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([
            exe_dir / "frontend_out",
            exe_dir / "frontend" / "out",
        ])
    candidates.extend([
        root / "frontend_out",
        root / "frontend" / "out",
        root.parent / "frontend" / "out",
    ])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="QMLKit desktop launcher")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--browser", action="store_true", help="Open in system browser instead of native window")
    args = parser.parse_args()

    root = resource_root()

    # Ensure src/ imports work when running from source.
    src_dir = root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    os.environ.setdefault("QMLKIT_FRONTEND_DIST", str(frontend_dist() or ""))

    import uvicorn

    from qmlkit.api.kennel_server import create_kennel_app  # noqa: F401  (app factory)

    config = uvicorn.Config(
        "qmlkit.api.kennel_server:app",
        host="127.0.0.1",
        port=args.port,
        log_level="info",
        factory=False,
    )
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/"
    # Wait briefly for startup.
    for _ in range(60):
        if server.started:
            break
        time.sleep(0.25)

    dist = frontend_dist()
    if dist:
        logger.info("Serving dashboard snapshot from %s", dist)

    try:
        import webview  # type: ignore

        webview.create_window("QMLKit Kennel Console", url, width=1280, height=860)
        logger.info("Opening native window -> %s", url)
        webview.start()
    except ImportError:
        import webbrowser

        logger.info("pywebview not installed; opening browser -> %s", url)
        webbrowser.open(url)

    logger.info("Shutting down.")
    server.should_exit = True


if __name__ == "__main__":
    main()
