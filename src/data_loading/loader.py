from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import TrajectoryMeasurement, VibrationPreview


class VibrationDataLoader:
    """Discovers and validates paired X/Y vibration measurements."""

    X_SIGNAL_TOKEN = "Analog raw input X"
    Y_SIGNAL_TOKEN = "Analog raw input Y"

    def __init__(self, data_root: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        if data_root is None:
            env_root = os.environ.get("SMARTSENSE_DATA_ROOT")
            if env_root:
                candidate = Path(env_root)
                data_root = candidate if candidate.is_absolute() else base_dir / candidate
        self.data_root = data_root or self._resolve_data_root(base_dir)

    @staticmethod
    def _resolve_data_root(base_dir: Path) -> Path:
        for candidate in (base_dir / "Data", base_dir / "data"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Data directory not found. Expected either 'Data' or 'data' in the project root."
        )

    def discover_measurements(self) -> List[TrajectoryMeasurement]:
        measurements: List[TrajectoryMeasurement] = []
        grouped_files: Dict[Path, Dict[str, Path]] = {}

        for csv_path in self.data_root.rglob("*.csv"):
            name = csv_path.name.lower()
            parent = csv_path.parent
            slot = grouped_files.setdefault(parent, {})
            if "axisx" in name:
                slot["x"] = csv_path
            elif "axisy" in name:
                slot["y"] = csv_path

        for directory in sorted(grouped_files):
            pair = grouped_files[directory]
            if "x" not in pair or "y" not in pair:
                continue

            relative_parts = directory.relative_to(self.data_root).parts
            if len(relative_parts) < 4:
                continue

            measurements.append(
                TrajectoryMeasurement(
                    joint=relative_parts[0],
                    measurement_date=relative_parts[1],
                    station=relative_parts[2],
                    trajectory=relative_parts[3],
                    x_path=pair["x"],
                    y_path=pair["y"],
                )
            )

        return measurements

    def build_preview(
        self, measurement: TrajectoryMeasurement, preview_rows: int = 5
    ) -> VibrationPreview:
        x_columns, x_records = self._read_csv_rows(measurement.x_path, limit=preview_rows)
        y_columns, y_records = self._read_csv_rows(measurement.y_path, limit=preview_rows)

        x_signal = self._find_signal_column(x_columns, self.X_SIGNAL_TOKEN)
        y_signal = self._find_signal_column(y_columns, self.Y_SIGNAL_TOKEN)

        preview: List[Dict[str, str]] = []
        for x_row, y_row in zip(x_records, y_records):
            preview.append(
                {
                    "Timestamp X": x_row.get("Timestamp", ""),
                    "Timestamp Y": y_row.get("Timestamp", ""),
                    x_signal: x_row.get(x_signal, ""),
                    y_signal: y_row.get(y_signal, ""),
                }
            )

        return VibrationPreview(
            measurement=measurement,
            x_columns=x_columns,
            y_columns=y_columns,
            selected_columns={"x": x_signal, "y": y_signal},
            preview_rows=preview,
        )

    def load_selected_signals(
        self, measurement: TrajectoryMeasurement
    ) -> Tuple[List[Dict[str, str]], str, str]:
        x_columns, x_rows = self._read_csv_rows(measurement.x_path)
        y_columns, y_rows = self._read_csv_rows(measurement.y_path)

        x_signal = self._find_signal_column(x_columns, self.X_SIGNAL_TOKEN)
        y_signal = self._find_signal_column(y_columns, self.Y_SIGNAL_TOKEN)

        paired_rows: List[Dict[str, str]] = []
        for x_row, y_row in zip(x_rows, y_rows):
            paired_rows.append(
                {
                    "timestamp_x": x_row.get("Timestamp", ""),
                    "timestamp_y": y_row.get("Timestamp", ""),
                    "analog_raw_input_x": x_row.get(x_signal, ""),
                    "analog_raw_input_y": y_row.get(y_signal, ""),
                }
            )

        return paired_rows, x_signal, y_signal

    def _find_signal_column(self, columns: Iterable[str], token: str) -> str:
        for column in columns:
            if token in column:
                return column
        raise ValueError(f"Required vibration column containing '{token}' was not found.")

    def _read_csv_rows(
        self, file_path: Path, limit: Optional[int] = None
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            content = handle.read()

        delimiter = self._detect_delimiter(content)
        cleaned_lines = self._clean_csv_lines(content)
        reader = csv.DictReader(io.StringIO("".join(cleaned_lines)), delimiter=delimiter)
        columns = [column for column in (reader.fieldnames or []) if column]
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({key: value for key, value in row.items() if key})
            if limit is not None and len(rows) >= limit:
                break
        return columns, rows

    @staticmethod
    def _detect_delimiter(content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("sep="):
                return stripped.split("=", 1)[1]
            return ";" if stripped.count(";") > stripped.count(",") else ","
        return ","

    @staticmethod
    def _clean_csv_lines(content: str) -> List[str]:
        cleaned: List[str] = []
        for line in content.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or stripped.startswith("sep="):
                continue
            cleaned.append(line)
        return cleaned
