"""CLI script to launch the FastAPI interactive diagnostic portal."""

import os
import sys
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from qmlkit.api.server import create_app


def main():
    app = create_app()
    print("\n🚀 Starting QMLKit Diagnostic Portal...")
    print("📍 Open your browser at: http://127.0.0.1:8000")
    print("📖 OpenAPI documentation: http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
