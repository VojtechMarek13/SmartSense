from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


TIMESTAMP_FORMAT = "%Y %m %d %H:%M:%S:%f"


@dataclass(frozen=True)
class FrequencyBand:
    """Inclusive low/high frequency band used for spectral summaries."""

    name: str
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class WindowedSignal:
    """One sliding-window segment of paired X/Y vibration signals."""

    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    x_signal: np.ndarray
    y_signal: np.ndarray


@dataclass(frozen=True)
class TimeSeriesFeatureSet:
    """Vibrodiagnostic features for one sliding window."""

    start_time: datetime
    end_time: datetime
    sampling_rate_hz: float
    rms_x: float
    rms_y: float
    rms_vector: float
    peak_x: float
    peak_y: float
    crest_factor_x: float
    crest_factor_y: float
    dominant_frequency_x_hz: float
    dominant_frequency_y_hz: float
    spectral_centroid_x_hz: float
    spectral_centroid_y_hz: float
    band_energies_x: Dict[str, float]
    band_energies_y: Dict[str, float]


class VibrationFeatureExtractor:
    """Transforms validated raw vibration signals into analysis-ready features."""

    DEFAULT_BANDS: Tuple[FrequencyBand, ...] = (
        FrequencyBand("low", 0.0, 50.0),
        FrequencyBand("mid", 50.0, 200.0),
        FrequencyBand("high", 200.0, 1000.0),
    )

    def __init__(
        self,
        window_size: int = 2048,
        step_size: int = 1024,
        detrend: bool = True,
        frequency_bands: Optional[Sequence[FrequencyBand]] = None,
    ) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1 sample.")
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0.")

        self.window_size = window_size
        self.step_size = step_size
        self.detrend = detrend
        self.frequency_bands = tuple(frequency_bands or self.DEFAULT_BANDS)

    def build_windows(self, paired_rows: Sequence[Dict[str, str]]) -> Tuple[List[WindowedSignal], float]:
        timestamps_x = [self._parse_timestamp(row["timestamp_x"]) for row in paired_rows]
        timestamps_y = [self._parse_timestamp(row["timestamp_y"]) for row in paired_rows]

        x_signal = np.asarray(
            [self._parse_measurement_value(row["analog_raw_input_x"]) for row in paired_rows],
            dtype=np.float64,
        )
        y_signal = np.asarray(
            [self._parse_measurement_value(row["analog_raw_input_y"]) for row in paired_rows],
            dtype=np.float64,
        )

        sampling_rate_hz = self._estimate_sampling_rate(timestamps_x, timestamps_y)
        windows: List[WindowedSignal] = []

        last_start = len(paired_rows) - self.window_size
        for start in range(0, last_start + 1, self.step_size):
            end = start + self.window_size
            x_window = x_signal[start:end]
            y_window = y_signal[start:end]

            if self.detrend:
                x_window = x_window - np.mean(x_window)
                y_window = y_window - np.mean(y_window)

            windows.append(
                WindowedSignal(
                    start_index=start,
                    end_index=end,
                    start_time=timestamps_x[start],
                    end_time=timestamps_x[end - 1],
                    duration_seconds=(timestamps_x[end - 1] - timestamps_x[start]).total_seconds(),
                    x_signal=x_window,
                    y_signal=y_window,
                )
            )

        return windows, sampling_rate_hz

    def extract_features(self, paired_rows: Sequence[Dict[str, str]]) -> List[TimeSeriesFeatureSet]:
        windows, sampling_rate_hz = self.build_windows(paired_rows)
        features: List[TimeSeriesFeatureSet] = []

        for window in windows:
            x_spectrum_freqs, x_spectrum = self._compute_fft(window.x_signal, sampling_rate_hz)
            y_spectrum_freqs, y_spectrum = self._compute_fft(window.y_signal, sampling_rate_hz)

            rms_x = self._rms(window.x_signal)
            rms_y = self._rms(window.y_signal)
            peak_x = float(np.max(np.abs(window.x_signal)))
            peak_y = float(np.max(np.abs(window.y_signal)))

            features.append(
                TimeSeriesFeatureSet(
                    start_time=window.start_time,
                    end_time=window.end_time,
                    sampling_rate_hz=sampling_rate_hz,
                    rms_x=rms_x,
                    rms_y=rms_y,
                    rms_vector=float(np.sqrt(rms_x ** 2 + rms_y ** 2)),
                    peak_x=peak_x,
                    peak_y=peak_y,
                    crest_factor_x=self._safe_divide(peak_x, rms_x),
                    crest_factor_y=self._safe_divide(peak_y, rms_y),
                    dominant_frequency_x_hz=self._dominant_frequency(x_spectrum_freqs, x_spectrum),
                    dominant_frequency_y_hz=self._dominant_frequency(y_spectrum_freqs, y_spectrum),
                    spectral_centroid_x_hz=self._spectral_centroid(x_spectrum_freqs, x_spectrum),
                    spectral_centroid_y_hz=self._spectral_centroid(y_spectrum_freqs, y_spectrum),
                    band_energies_x=self._band_energies(x_spectrum_freqs, x_spectrum),
                    band_energies_y=self._band_energies(y_spectrum_freqs, y_spectrum),
                )
            )

        return features

    def features_to_matrix(
        self, feature_sets: Sequence[TimeSeriesFeatureSet]
    ) -> Tuple[np.ndarray, List[str]]:
        feature_names = [
            "rms_x",
            "rms_y",
            "rms_vector",
            "peak_x",
            "peak_y",
            "crest_factor_x",
            "crest_factor_y",
            "dominant_frequency_x_hz",
            "dominant_frequency_y_hz",
            "spectral_centroid_x_hz",
            "spectral_centroid_y_hz",
        ]

        for band in self.frequency_bands:
            feature_names.append(f"band_energy_x_{band.name}")
            feature_names.append(f"band_energy_y_{band.name}")

        matrix = np.zeros((len(feature_sets), len(feature_names)), dtype=np.float32)
        for row_index, feature_set in enumerate(feature_sets):
            values = [
                feature_set.rms_x,
                feature_set.rms_y,
                feature_set.rms_vector,
                feature_set.peak_x,
                feature_set.peak_y,
                feature_set.crest_factor_x,
                feature_set.crest_factor_y,
                feature_set.dominant_frequency_x_hz,
                feature_set.dominant_frequency_y_hz,
                feature_set.spectral_centroid_x_hz,
                feature_set.spectral_centroid_y_hz,
            ]
            for band in self.frequency_bands:
                values.append(feature_set.band_energies_x.get(band.name, 0.0))
                values.append(feature_set.band_energies_y.get(band.name, 0.0))
            matrix[row_index, :] = np.asarray(values, dtype=np.float32)

        return matrix, feature_names

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.strptime(value, TIMESTAMP_FORMAT)

    @staticmethod
    def _parse_measurement_value(value: str) -> float:
        normalized = value.strip().replace(",", ".")
        if not normalized:
            return 0.0
        return float(normalized)

    @staticmethod
    def _estimate_sampling_rate(
        timestamps_x: Sequence[datetime], timestamps_y: Sequence[datetime]
    ) -> float:
        deltas = []
        for sequence in (timestamps_x, timestamps_y):
            for current, previous in zip(sequence[1:], sequence[:-1]):
                delta = (current - previous).total_seconds()
                if delta > 0:
                    deltas.append(delta)

        if not deltas:
            raise ValueError("Unable to estimate sampling rate from timestamps.")

        median_delta = float(np.median(np.asarray(deltas, dtype=np.float64)))
        return 1.0 / median_delta

    @staticmethod
    def _rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(signal))))

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    @staticmethod
    def _compute_fft(signal: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        window = np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal * window))
        frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate_hz)
        return frequencies, spectrum

    @staticmethod
    def _dominant_frequency(frequencies: np.ndarray, amplitudes: np.ndarray) -> float:
        if len(frequencies) <= 1:
            return 0.0
        index = int(np.argmax(amplitudes[1:]) + 1)
        return float(frequencies[index])

    @staticmethod
    def _spectral_centroid(frequencies: np.ndarray, amplitudes: np.ndarray) -> float:
        total_energy = float(np.sum(amplitudes))
        if total_energy <= 0:
            return 0.0
        return float(np.sum(frequencies * amplitudes) / total_energy)

    def _band_energies(self, frequencies: np.ndarray, amplitudes: np.ndarray) -> Dict[str, float]:
        energies: Dict[str, float] = {}
        for band in self.frequency_bands:
            mask = (frequencies >= band.low_hz) & (frequencies <= band.high_hz)
            band_amplitudes = amplitudes[mask]
            energies[band.name] = float(np.sum(np.square(band_amplitudes)))
        return energies
