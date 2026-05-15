from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from src.data_loading.models import TrajectoryMeasurement
from src.health.history import HistoricalThresholdCalibrator, HistoricalThresholdProfile
from src.processing.features import TimeSeriesFeatureSet


def summarize_feature_severity(feature_sets: list[TimeSeriesFeatureSet]) -> float:
    if not feature_sets:
        return 0.0

    rms_vector = np.asarray([item.rms_vector for item in feature_sets], dtype=np.float64)
    crest = np.asarray(
        [(item.crest_factor_x + item.crest_factor_y) / 2.0 for item in feature_sets],
        dtype=np.float64,
    )
    high_energy = np.asarray(
        [
            item.band_energies_x.get("high", 0.0) + item.band_energies_y.get("high", 0.0)
            for item in feature_sets
        ],
        dtype=np.float64,
    )

    rms_level = float(np.percentile(rms_vector, 90))
    crest_level = float(np.percentile(crest, 90))
    spectral_level = float(np.percentile(high_energy, 90))

    return float(
        0.55 * rms_level
        + 0.20 * crest_level
        + 0.25 * np.log1p(max(spectral_level, 0.0))
    )


def build_joint_history_profile(
    joint: str,
    measurements: list[TrajectoryMeasurement],
    feature_cache: Dict[str, list[TimeSeriesFeatureSet]],
    calibrator: Optional[HistoricalThresholdCalibrator] = None,
) -> Optional[HistoricalThresholdProfile]:
    calibrator = calibrator or HistoricalThresholdCalibrator()
    dated_scores: Dict[str, List[float]] = defaultdict(list)

    for measurement in measurements:
        if measurement.joint != joint:
            continue
        cache_key = str(measurement.x_path)
        feature_sets = feature_cache.get(cache_key)
        if not feature_sets:
            continue
        dated_scores[measurement.measurement_date].append(summarize_feature_severity(feature_sets))

    return calibrator.build_profile(joint, dated_scores)
