from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np


DATE_FORMAT = "%d.%m.%Y"


RAW_OPERATING_HOURS: Dict[str, Dict[str, float]] = {
    "02.02.2026": {
        "Joint 1": 921.9,
        "Joint 2": 918.7,
        "Joint 3": 915.7,
        "Joint 4": 702.7,
        "Joint 5": 917.7,
    },
    "19.02.2026": {
        "Joint 1": 1031.9,
        "Joint 2": 1028.7,
        "Joint 3": 1025.7,
        "Joint 4": 812.7,
        "Joint 5": 1027.7,
    },
    "12.03.2026": {
        "Joint 1": 1167.4,
        "Joint 2": 1164.2,
        "Joint 3": 1161.2,
        "Joint 4": 948.2,
        "Joint 5": 1163.2,
    },
}


@dataclass(frozen=True)
class OperatingHourRecord:
    joint: str
    measurement_date: str
    operating_hours: float


@dataclass(frozen=True)
class HistoricalConditionPoint:
    joint: str
    measurement_date: str
    operating_hours: float
    severity_score: float


class OperatingHoursRegistry:
    """Provides operating-hour context for each joint/date pair."""

    def __init__(self) -> None:
        self._records = RAW_OPERATING_HOURS

    def get_record(self, joint: str, measurement_date: str) -> Optional[OperatingHourRecord]:
        normalized_date = self._normalize_date(measurement_date)
        hours = self._records.get(normalized_date, {}).get(joint)
        if hours is None:
            return None
        return OperatingHourRecord(joint=joint, measurement_date=normalized_date, operating_hours=hours)

    def fit_severity_trend(
        self,
        points: Sequence[HistoricalConditionPoint],
    ) -> tuple[float, float] | None:
        if len(points) < 2:
            return None

        x = np.asarray([point.operating_hours for point in points], dtype=np.float64)
        y = np.asarray([point.severity_score for point in points], dtype=np.float64)
        slope, intercept = np.polyfit(x, y, deg=1)
        return float(slope), float(intercept)

    @staticmethod
    def _normalize_date(value: str) -> str:
        parsed = datetime.strptime(value, DATE_FORMAT)
        return parsed.strftime(DATE_FORMAT)
