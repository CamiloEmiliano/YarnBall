#!/usr/bin/env python3
"""Local CLI launcher for the YarnBall FastAPI Workbench Backend (ADR-00017)."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for the local workbench server."""
    parser = argparse.ArgumentParser(
        description="Launch the local read-only YarnBall FastAPI Workbench Backend (127.0.0.1:8050)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Interface address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="TCP port to serve on (default: 8050)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the local readiness URL in the default browser after startup",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload on source edits (local development only)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parses CLI flags and launches Uvicorn bound cleanly to local host."""
    args = build_parser().parse_args(argv)
    url = f"http://{args.host}:{args.port}/"

    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "explorer.workbench_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
