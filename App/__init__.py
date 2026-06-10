"""PyQt application package for SmartSense."""

from __future__ import annotations


def __getattr__(name: str):
    if name in ("DashboardWindow", "run"):
        from .dashboard import DashboardWindow, run  # noqa: PLC0415
        globals()["DashboardWindow"] = DashboardWindow
        globals()["run"] = run
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DashboardWindow", "run"]
