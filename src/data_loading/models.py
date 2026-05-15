from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class TrajectoryMeasurement:
    """Describes one paired X/Y vibration measurement."""

    joint: str
    measurement_date: str
    station: str
    trajectory: str
    x_path: Path
    y_path: Path


@dataclass(frozen=True)
class VibrationPreview:
    """Human-readable validation payload shown before processing."""

    measurement: TrajectoryMeasurement
    x_columns: List[str]
    y_columns: List[str]
    selected_columns: Dict[str, str]
    preview_rows: List[Dict[str, str]]
