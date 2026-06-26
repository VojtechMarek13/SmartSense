from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np

from src.health.history import HistoricalThresholdProfile
from src.processing.features import TimeSeriesFeatureSet


@dataclass(frozen=True)
class HealthThresholds:
    """Threshold proposal derived from harmonic-drive vibrodiagnostics heuristics."""

    normal_max_score: float = 35.0
    warning_max_score: float = 70.0
    rms_warning_ratio: float = 1.35
    rms_critical_ratio: float = 1.60
    crest_warning_ratio: float = 1.20
    crest_critical_ratio: float = 1.45
    spectral_warning_ratio: float = 1.30
    spectral_critical_ratio: float = 1.75


@dataclass(frozen=True)
class JointHealthTimelinePoint:
    """Health metrics for one feature window."""

    timestamp: datetime
    rms_vector: float
    rms_ratio: float
    crest_ratio: float
    spectral_ratio: float
    trend_ratio: float
    health_index: float
    state: str


@dataclass(frozen=True)
class HealthAssessment:
    """Aggregated condition assessment for a trajectory."""

    timeline: List[JointHealthTimelinePoint]
    thresholds: HealthThresholds
    baseline_rms: float
    baseline_crest: float
    baseline_spectral: float
    current_health_index: float
    current_state: str
    age_score: float
    moving_average: np.ndarray
    regression_slope_per_hour: float
    regression_intercept: float
    estimated_hours_to_critical: Optional[float]
    hours_to_critical_lower_90: Optional[float]
    hours_to_critical_upper_90: Optional[float]
    estimated_operating_hours_to_critical: Optional[float]
    recommendation: str
    historical_threshold_profile: Optional[HistoricalThresholdProfile]


class JointHealthAnalyzer:
    """Builds a composite Joint Health Index from vibrodiagnostic features."""

    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        baseline_fraction: float = 0.2,
        moving_average_window: int = 5,
    ) -> None:
        self.thresholds = thresholds or HealthThresholds()
        self.baseline_fraction = baseline_fraction
        self.moving_average_window = max(2, moving_average_window)

    def assess(
        self,
        feature_sets: Sequence[TimeSeriesFeatureSet],
        reference_feature_sets: Optional[Sequence[TimeSeriesFeatureSet]] = None,
        historical_threshold_profile: Optional[HistoricalThresholdProfile] = None,
        current_operating_hours: Optional[float] = None,
    ) -> HealthAssessment:
        if not feature_sets:
            raise ValueError("Feature sets are required for health assessment.")

        baseline_source = list(reference_feature_sets) if reference_feature_sets else list(feature_sets)
        baseline_count = max(3, int(len(baseline_source) * self.baseline_fraction))
        baseline_count = min(baseline_count, len(baseline_source))
        baseline_slice = baseline_source[:baseline_count]

        baseline_rms = float(np.median([item.rms_vector for item in baseline_slice]))
        baseline_crest = float(
            np.median(
                [
                    (item.crest_factor_x + item.crest_factor_y) / 2.0
                    for item in baseline_slice
                ]
            )
        )
        baseline_spectral = float(
            np.median([self._spectral_indicator(item) for item in baseline_slice])
        )

        rms_ratios = np.asarray(
            [item.rms_vector / max(baseline_rms, 1e-6) for item in feature_sets],
            dtype=np.float64,
        )
        crest_ratios = np.asarray(
            [
                ((item.crest_factor_x + item.crest_factor_y) / 2.0) / max(baseline_crest, 1e-6)
                for item in feature_sets
            ],
            dtype=np.float64,
        )
        spectral_ratios = np.asarray(
            [self._spectral_indicator(item) / max(baseline_spectral, 1e-6) for item in feature_sets],
            dtype=np.float64,
        )

        moving_average = self._moving_average(rms_ratios, self.moving_average_window)
        trend_ratios = self._trend_ratios(moving_average)

        rms_scores = self._severity_score(
            rms_ratios,
            self.thresholds.rms_warning_ratio,
            self.thresholds.rms_critical_ratio,
        )
        crest_scores = self._severity_score(
            crest_ratios,
            self.thresholds.crest_warning_ratio,
            self.thresholds.crest_critical_ratio,
        )
        spectral_scores = self._severity_score(
            spectral_ratios,
            self.thresholds.spectral_warning_ratio,
            self.thresholds.spectral_critical_ratio,
        )
        trend_scores = self._severity_score(trend_ratios, 1.05, 1.20)
        age_score = self._age_score(current_operating_hours, historical_threshold_profile)

        health_index = np.clip(
            0.40 * rms_scores
            + 0.15 * crest_scores
            + 0.15 * spectral_scores
            + 0.10 * trend_scores
            + 0.20 * age_score,
            0.0,
            100.0,
        )

        effective_warning_score = self.thresholds.normal_max_score
        effective_critical_score = self.thresholds.warning_max_score

        timeline: List[JointHealthTimelinePoint] = []
        for index, feature_set in enumerate(feature_sets):
            state = self._classify_state(
                health_index[index],
                rms_ratios[index],
                crest_ratios[index],
                spectral_ratios[index],
                effective_warning_score,
                effective_critical_score,
            )
            timeline.append(
                JointHealthTimelinePoint(
                    timestamp=feature_set.end_time,
                    rms_vector=feature_set.rms_vector,
                    rms_ratio=float(rms_ratios[index]),
                    crest_ratio=float(crest_ratios[index]),
                    spectral_ratio=float(spectral_ratios[index]),
                    trend_ratio=float(trend_ratios[index]),
                    health_index=float(health_index[index]),
                    state=state,
                )
            )

        regression_slope_per_hour, regression_intercept, _trend_cov = self._fit_health_trend(timeline)
        estimated_hours_to_critical, hours_lower_90, hours_upper_90 = self._estimate_time_to_critical(
            timeline,
            regression_slope_per_hour,
            regression_intercept,
            effective_critical_score,
            _trend_cov,
        )
        estimated_operating_hours_to_critical = self._estimate_operating_hours_to_critical(
            current_operating_hours=current_operating_hours,
            critical_score=effective_critical_score,
            historical_threshold_profile=historical_threshold_profile,
        )
        current_health_index = float(np.percentile(health_index, 75))
        current_state = self._classify_score(current_health_index)

        return HealthAssessment(
            timeline=timeline,
            thresholds=self.thresholds,
            baseline_rms=baseline_rms,
            baseline_crest=baseline_crest,
            baseline_spectral=baseline_spectral,
            current_health_index=current_health_index,
            current_state=current_state,
            age_score=age_score,
            moving_average=moving_average,
            regression_slope_per_hour=regression_slope_per_hour,
            regression_intercept=regression_intercept,
            estimated_hours_to_critical=estimated_hours_to_critical,
            hours_to_critical_lower_90=hours_lower_90,
            hours_to_critical_upper_90=hours_upper_90,
            estimated_operating_hours_to_critical=estimated_operating_hours_to_critical,
            recommendation=self._recommendation(
                current_state,
                estimated_hours_to_critical,
                estimated_operating_hours_to_critical,
            ),
            historical_threshold_profile=historical_threshold_profile,
        )

    @staticmethod
    def _spectral_indicator(feature_set: TimeSeriesFeatureSet) -> float:
        low_energy = feature_set.band_energies_x.get("low", 0.0) + feature_set.band_energies_y.get("low", 0.0)
        high_energy = feature_set.band_energies_x.get("high", 0.0) + feature_set.band_energies_y.get("high", 0.0)
        return float(high_energy / max(low_energy, 1e-6))

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window:
            return values.copy()
        kernel = np.ones(window, dtype=np.float64) / window
        leading = np.full(window - 1, values[0], dtype=np.float64)
        padded = np.concatenate([leading, values])
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _trend_ratios(moving_average: np.ndarray) -> np.ndarray:
        baseline = max(float(np.median(moving_average[: max(3, min(len(moving_average), 5))])), 1e-6)
        return moving_average / baseline

    @staticmethod
    def _severity_score(values: np.ndarray, warning_limit: float, critical_limit: float) -> np.ndarray:
        scores = np.zeros_like(values, dtype=np.float64)

        warning_zone = (values > 1.0) & (values <= warning_limit)
        scores[warning_zone] = 35.0 * (values[warning_zone] - 1.0) / max(warning_limit - 1.0, 1e-6)

        critical_zone = (values > warning_limit) & (values <= critical_limit)
        scores[critical_zone] = 35.0 + 35.0 * (
            (values[critical_zone] - warning_limit) / max(critical_limit - warning_limit, 1e-6)
        )

        overload_zone = values > critical_limit
        scores[overload_zone] = 70.0 + 30.0 * np.clip(
            (values[overload_zone] - critical_limit) / max(critical_limit, 1e-6),
            0.0,
            1.0,
        )
        return scores

    def _classify_state(
        self,
        health_index: float,
        rms_ratio: float,
        crest_ratio: float,
        spectral_ratio: float,
        warning_score: float,
        critical_score: float,
    ) -> str:
        if health_index >= critical_score:
            return "CRITICAL"

        if health_index >= warning_score:
            return "WARNING"

        return "NORMAL"

    def _classify_score(self, health_index: float) -> str:
        if health_index >= self.thresholds.warning_max_score:
            return "CRITICAL"
        if health_index >= self.thresholds.normal_max_score:
            return "WARNING"
        return "NORMAL"

    def _fit_health_trend(
        self, timeline: Sequence[JointHealthTimelinePoint]
    ) -> tuple[float, float, Optional[np.ndarray]]:
        if len(timeline) < 2:
            return 0.0, timeline[0].health_index, None

        start_time = timeline[0].timestamp
        elapsed_hours = np.asarray(
            [max((p.timestamp - start_time).total_seconds() / 3600.0, 0.0) for p in timeline],
            dtype=np.float64,
        )
        health_values = np.asarray([p.health_index for p in timeline], dtype=np.float64)

        if len(timeline) >= 3:
            try:
                coeffs, cov = np.polyfit(elapsed_hours, health_values, deg=1, cov=True)
                if np.all(np.isfinite(cov)):
                    return float(coeffs[0]), float(coeffs[1]), cov
            except (np.linalg.LinAlgError, ValueError):
                pass

        slope, intercept = np.polyfit(elapsed_hours, health_values, deg=1)
        return float(slope), float(intercept), None

    def _estimate_time_to_critical(
        self,
        timeline: Sequence[JointHealthTimelinePoint],
        slope_per_hour: float,
        intercept: float,
        critical_score: float,
        cov: Optional[np.ndarray] = None,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if slope_per_hour <= 0:
            return None, None, None

        current_hours = max(
            (timeline[-1].timestamp - timeline[0].timestamp).total_seconds() / 3600.0, 0.0
        )
        predicted_crossing_hours = (critical_score - intercept) / slope_per_hour
        remaining = predicted_crossing_hours - current_hours

        if remaining <= 0:
            return 0.0, None, None

        mean_hours = float(remaining)

        if cov is None:
            return mean_hours, None, None

        # Delta method: x_c = (y_c - b) / a  →  r = x_c - x_current
        # ∂r/∂a = -x_c / a,   ∂r/∂b = -1 / a
        a = slope_per_hour
        x_c = predicted_crossing_hours
        var_r = (
            (x_c / a) ** 2 * cov[0, 0]
            + (1.0 / a) ** 2 * cov[1, 1]
            - 2.0 * (x_c / a) * (1.0 / a) * cov[0, 1]
        )

        if not np.isfinite(var_r) or var_r < 0:
            return mean_hours, None, None

        std_r = float(np.sqrt(var_r))
        z90 = 1.645
        lower = float(max(0.0, remaining - z90 * std_r))
        upper = float(remaining + z90 * std_r)
        return mean_hours, lower, upper

    @staticmethod
    def _estimate_operating_hours_to_critical(
        current_operating_hours: Optional[float],
        critical_score: float,
        historical_threshold_profile: Optional[HistoricalThresholdProfile],
    ) -> Optional[float]:
        if current_operating_hours is None or historical_threshold_profile is None:
            return None
        critical_hours = historical_threshold_profile.critical_hours
        if critical_hours is None:
            return None
        remaining = critical_hours - current_operating_hours
        return float(remaining) if remaining > 0 else 0.0

    @staticmethod
    def _age_score(
        current_operating_hours: Optional[float],
        historical_threshold_profile: Optional[HistoricalThresholdProfile],
    ) -> float:
        if (
            current_operating_hours is None
            or historical_threshold_profile is None
            or not historical_threshold_profile.history_points
            or historical_threshold_profile.critical_hours is None
        ):
            return 0.0

        oldest_hours = historical_threshold_profile.history_points[0].operating_hours
        critical_hours = historical_threshold_profile.critical_hours
        span = max(critical_hours - oldest_hours, 1e-6)
        progress = (current_operating_hours - oldest_hours) / span
        return float(np.clip(progress * 100.0, 0.0, 100.0))

    @staticmethod
    def _recommendation(
        state: str,
        estimated_hours_to_critical: Optional[float],
        estimated_operating_hours_to_critical: Optional[float],
    ) -> str:
        if state == "CRITICAL":
            return "Critical condition detected. Inspect the harmonic drive immediately and plan maintenance now."
        if state == "WARNING":
            if estimated_operating_hours_to_critical is not None:
                return (
                    "Maintenance recommended within approximately "
                    f"{estimated_operating_hours_to_critical:.1f} operating hours."
                )
            if estimated_hours_to_critical is None:
                return "Warning trend detected. Increase monitoring frequency and schedule maintenance review."
            return f"Maintenance recommended within approximately {estimated_hours_to_critical:.1f} hours."
        return "Joint is operating within the normal vibration envelope. Continue routine monitoring."
