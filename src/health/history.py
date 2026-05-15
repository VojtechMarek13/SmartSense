from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from src.history.operating_hours import HistoricalConditionPoint, OperatingHoursRegistry


@dataclass(frozen=True)
class HistoricalThresholdProfile:
    """Joint-specific threshold calibration from dated operating-hour history."""

    warning_score: float
    critical_score: float
    warning_hours: float | None
    critical_hours: float | None
    history_points: List[HistoricalConditionPoint]


class HistoricalThresholdCalibrator:
    """Derives warning/critical limits from historical joint severity progression."""

    def __init__(self, registry: OperatingHoursRegistry | None = None) -> None:
        self.registry = registry or OperatingHoursRegistry()

    def build_profile(
        self,
        joint: str,
        dated_scores: Dict[str, Sequence[float]],
    ) -> HistoricalThresholdProfile | None:
        points: List[HistoricalConditionPoint] = []
        for measurement_date, scores in sorted(dated_scores.items(), key=lambda item: self._sort_key(item[0])):
            record = self.registry.get_record(joint, measurement_date)
            if record is None or not scores:
                continue
            points.append(
                HistoricalConditionPoint(
                    joint=joint,
                    measurement_date=record.measurement_date,
                    operating_hours=record.operating_hours,
                    severity_score=float(np.mean(np.asarray(scores, dtype=np.float64))),
                )
            )

        if len(points) < 2:
            return None

        warning_score = 35.0
        critical_score = 70.0
        warning_hours, critical_hours = self._derive_hour_limits(points)

        return HistoricalThresholdProfile(
            warning_score=warning_score,
            critical_score=critical_score,
            warning_hours=warning_hours,
            critical_hours=critical_hours,
            history_points=points,
        )

    @staticmethod
    def _sort_key(measurement_date: str) -> tuple[int, int, int]:
        day, month, year = measurement_date.split(".")
        return int(year), int(month), int(day)

    @staticmethod
    def _derive_hour_limits(
        points: Sequence[HistoricalConditionPoint],
    ) -> tuple[float | None, float | None]:
        if len(points) < 2:
            return None, None

        hours = np.asarray([point.operating_hours for point in points], dtype=np.float64)
        increments = np.diff(hours)
        median_increment = float(np.median(increments))

        if len(hours) >= 3:
            warning_hours = float((hours[-2] + hours[-1]) / 2.0)
        else:
            warning_hours = float(hours[-1])

        critical_hours = float(hours[-1] + median_increment)
        return warning_hours, critical_hours
