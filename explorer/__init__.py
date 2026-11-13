"""Local FastAPI Knowledge-Graph & Contagion Workbench Backend (ADR-00017)."""

from explorer.workbench_api import app, create_workbench_app

__all__ = ["app", "create_workbench_app"]

