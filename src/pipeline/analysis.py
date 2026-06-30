from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from src.data_loading.loader import VibrationDataLoader
from src.data_loading.models import TrajectoryMeasurement
from src.health.history import HistoricalThresholdCalibrator
from src.health.index import HealthAssessment, JointHealthAnalyzer
from src.history.operating_hours import OperatingHoursRegistry
from src.pipeline.history import build_joint_history_profile
from src.processing.features import TimeSeriesFeatureSet, VibrationFeatureExtractor


@dataclass(frozen=True)
class DashboardAnalysis:
    """Payload consumed by the GUI and examples."""

    measurement: TrajectoryMeasurement
    raw_timestamps: List[datetime]
    raw_signal_x: np.ndarray
    raw_signal_y: np.ndarray
    feature_sets: List[TimeSeriesFeatureSet]
    health_assessment: HealthAssessment
    sampling_rate_hz: float
    selected_x_column: str
    selected_y_column: str


class SmartSenseAnalysisPipeline:
    """Orchestrates data loading, feature extraction, health scoring, and prediction."""

    def __init__(
        self,
        loader: Optional[VibrationDataLoader] = None,
        extractor: Optional[VibrationFeatureExtractor] = None,
        analyzer: Optional[JointHealthAnalyzer] = None,
    ) -> None:
        self.loader = loader or VibrationDataLoader()
        self.extractor = extractor or VibrationFeatureExtractor()
        self.analyzer = analyzer or JointHealthAnalyzer()
        self.operating_hours_registry = OperatingHoursRegistry()
        self.threshold_calibrator = HistoricalThresholdCalibrator(self.operating_hours_registry)
        self._feature_cache: Dict[str, List[TimeSeriesFeatureSet]] = {}
        self._measurements_cache: List[TrajectoryMeasurement] | None = None

    def list_measurements(self) -> List[TrajectoryMeasurement]:
        if self._measurements_cache is None:
            self._measurements_cache = self.loader.discover_measurements()
        return self._measurements_cache

    def analyze_measurement(self, measurement: TrajectoryMeasurement) -> DashboardAnalysis:
        paired_rows, selected_x_column, selected_y_column = self.loader.load_selected_signals(measurement)
        raw_timestamps = [self.extractor._parse_timestamp(row["timestamp_x"]) for row in paired_rows]
        raw_signal_x = np.asarray(
            [self.extractor._parse_measurement_value(row["analog_raw_input_x"]) for row in paired_rows],
            dtype=np.float64,
        )
        raw_signal_y = np.asarray(
            [self.extractor._parse_measurement_value(row["analog_raw_input_y"]) for row in paired_rows],
            dtype=np.float64,
        )

        feature_sets = self.extractor.extract_features(paired_rows)
        self._feature_cache[str(measurement.x_path)] = feature_sets
        _, sampling_rate_hz = self.extractor.build_windows(paired_rows)
        related_measurements = [
            candidate for candidate in self.list_measurements() if candidate.joint == measurement.joint
        ]
        for related_measurement in related_measurements:
            cache_key = str(related_measurement.x_path)
            if cache_key not in self._feature_cache:
                related_rows, _, _ = self.loader.load_selected_signals(related_measurement)
                self._feature_cache[cache_key] = self.extractor.extract_features(related_rows)

        reference_feature_sets = self._get_reference_feature_sets(measurement)
        history_profile = build_joint_history_profile(
            joint=measurement.joint,
            measurements=related_measurements,
            feature_cache=self._feature_cache,
            calibrator=self.threshold_calibrator,
        )
        operating_record = self.operating_hours_registry.get_record(
            measurement.joint,
            measurement.measurement_date,
        )
        current_operating_hours = None if operating_record is None else operating_record.operating_hours
        health_assessment = self.analyzer.assess(
            feature_sets,
            reference_feature_sets=reference_feature_sets,
            historical_threshold_profile=history_profile,
            current_operating_hours=current_operating_hours,
        )

        return DashboardAnalysis(
            measurement=measurement,
            raw_timestamps=raw_timestamps,
            raw_signal_x=raw_signal_x,
            raw_signal_y=raw_signal_y,
            feature_sets=feature_sets,
            health_assessment=health_assessment,
            sampling_rate_hz=sampling_rate_hz,
            selected_x_column=selected_x_column,
            selected_y_column=selected_y_column,
        )

    def analyze_uploaded_signal(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_col: str,
        y_col: str,
        label: str,
        sampling_rate_hz: float = 1000.0,
    ) -> DashboardAnalysis:
        """Analyze an arbitrary X/Y signal without historical calibration."""
        from pathlib import Path

        n = min(len(x_values), len(y_values))
        x_values = x_values[:n]
        y_values = y_values[:n]

        t0 = datetime(2026, 1, 1)
        dt = timedelta(microseconds=max(1, int(1_000_000 / sampling_rate_hz)))
        timestamps = [t0 + dt * i for i in range(n)]
        ts_fmt = "%Y %m %d %H:%M:%S:%f"
        ts_str = [t.strftime(ts_fmt) for t in timestamps]

        paired_rows = [
            {
                "timestamp_x": ts_str[i],
                "timestamp_y": ts_str[i],
                "analog_raw_input_x": str(float(x_values[i])),
                "analog_raw_input_y": str(float(y_values[i])),
            }
            for i in range(n)
        ]

        feature_sets = self.extractor.extract_features(paired_rows)
        if not feature_sets:
            raise ValueError(
                f"Not enough data for feature extraction — got {n} samples, need at least one full window."
            )

        health_assessment = self.analyzer.assess(
            feature_sets,
            reference_feature_sets=None,
            historical_threshold_profile=None,
            current_operating_hours=None,
        )

        synthetic_measurement = TrajectoryMeasurement(
            joint=label,
            measurement_date="uploaded",
            station="Uploaded",
            trajectory="Custom",
            x_path=Path(""),
            y_path=Path(""),
        )

        return DashboardAnalysis(
            measurement=synthetic_measurement,
            raw_timestamps=timestamps,
            raw_signal_x=x_values,
            raw_signal_y=y_values,
            feature_sets=feature_sets,
            health_assessment=health_assessment,
            sampling_rate_hz=sampling_rate_hz,
            selected_x_column=x_col,
            selected_y_column=y_col,
        )

    def _get_reference_feature_sets(
        self,
        measurement: TrajectoryMeasurement,
    ) -> Optional[List[TimeSeriesFeatureSet]]:
        candidates = [
            candidate
            for candidate in self.list_measurements()
            if candidate.joint == measurement.joint and candidate.trajectory == measurement.trajectory
        ]
        if not candidates:
            return None

        def operating_hours_key(candidate: TrajectoryMeasurement) -> float:
            record = self.operating_hours_registry.get_record(candidate.joint, candidate.measurement_date)
            if record is None:
                return float("inf")
            return record.operating_hours

        reference_measurement = min(candidates, key=operating_hours_key)
        cache_key = str(reference_measurement.x_path)
        if cache_key not in self._feature_cache:
            related_rows, _, _ = self.loader.load_selected_signals(reference_measurement)
            self._feature_cache[cache_key] = self.extractor.extract_features(related_rows)
        return self._feature_cache[cache_key]
