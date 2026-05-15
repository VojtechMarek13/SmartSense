from __future__ import annotations

import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
mpl_config_dir = project_root / ".mplconfig"
mpl_config_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.pipeline.analysis import DashboardAnalysis, SmartSenseAnalysisPipeline


STATE_COLORS = {
    "NORMAL": "#23B26D",
    "WARNING": "#E8A317",
    "CRITICAL": "#D8434B",
}


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        figure = Figure(figsize=(5, 4), dpi=100, facecolor="#F5F7F4")
        self.axes = figure.add_subplot(111)
        super().__init__(figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class HealthLight(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(92, 92)
        self.setObjectName("healthLight")

    def set_state(self, state: str) -> None:
        color = STATE_COLORS.get(state, "#64748B")
        self.setStyleSheet(
            f"""
            QFrame#healthLight {{
                background: {color};
                border-radius: 46px;
                border: 6px solid rgba(255, 255, 255, 0.65);
            }}
            """
        )


class DashboardWindow(QMainWindow):
    def __init__(self, pipeline: SmartSenseAnalysisPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.measurements = self.pipeline.list_measurements()
        self.analysis: DashboardAnalysis | None = None

        self.setWindowTitle("SmartSense Predictive Maintenance Dashboard")
        self.resize(1500, 920)
        self.setMinimumSize(1280, 820)
        self._build_ui()

        if not self.measurements:
            raise RuntimeError("No vibration datasets were found under the project data directory.")

        self._populate_measurements()
        self._load_current_measurement()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(18)

        header = self._build_header()
        layout.addWidget(header)

        body = QGridLayout()
        body.setHorizontalSpacing(18)
        body.setVerticalSpacing(18)
        layout.addLayout(body, stretch=1)

        self.signal_canvas = MplCanvas()
        self.trend_canvas = MplCanvas()

        self.signal_card = self._wrap_card("Signal View", self.signal_canvas)
        self.trend_card = self._wrap_card("Health Trend", self.trend_canvas)
        body.addWidget(self.signal_card, 0, 0, 1, 2)
        body.addWidget(self.trend_card, 0, 2, 1, 2)

        self.health_light = HealthLight()
        self.state_label = QLabel("UNKNOWN")
        self.state_label.setObjectName("metricHeading")
        self.state_value = QLabel("JHI: --")
        self.state_value.setObjectName("metricValue")
        self.recommendation_label = QLabel("Waiting for analysis")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setObjectName("metricText")

        self.eta_value = QLabel("--")
        self.eta_value.setObjectName("metricValue")
        self.operating_eta_value = QLabel("--")
        self.operating_eta_value.setObjectName("metricValue")
        self.rms_value = QLabel("--")
        self.rms_value.setObjectName("metricValue")
        self.freq_value = QLabel("--")
        self.freq_value.setObjectName("metricValue")
        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        self.meta_label.setObjectName("metricText")

        body.addWidget(self._build_health_card(), 1, 0, 1, 2)
        body.addWidget(self._build_prediction_card(), 1, 2, 1, 1)
        body.addWidget(self._build_measurement_card(), 1, 3, 1, 1)

        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #E6F3EE, stop:1 #F8F3E9);
            }
            QWidget {
                color: #18332B;
                font-family: "Segoe UI";
                font-size: 12px;
            }
            QFrame[card="true"] {
                background: rgba(255, 255, 255, 0.88);
                border-radius: 24px;
                border: 1px solid rgba(24, 51, 43, 0.08);
            }
            QLabel#title {
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #58756B;
                font-size: 13px;
            }
            QLabel#cardTitle {
                font-size: 17px;
                font-weight: 700;
            }
            QLabel#metricHeading {
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#metricValue {
                font-size: 26px;
                font-weight: 800;
            }
            QLabel#metricText {
                color: #4C655D;
                font-size: 13px;
            }
            QComboBox, QPushButton {
                min-height: 40px;
                border-radius: 14px;
                padding: 6px 12px;
                border: 1px solid rgba(24, 51, 43, 0.12);
                background: white;
            }
            QPushButton {
                background: #1A9C74;
                color: white;
                font-weight: 700;
            }
            """
        )

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setProperty("card", True)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 18, 22, 18)

        title_box = QVBoxLayout()
        title = QLabel("SmartSense")
        title.setObjectName("title")
        subtitle = QLabel(
            "Harmonic-drive vibrodiagnostics dashboard with RMS trend, Joint Health Index, and maintenance prediction."
        )
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        control_box = QHBoxLayout()
        control_box.setSpacing(12)
        self.measurement_selector = QComboBox()
        self.measurement_selector.currentIndexChanged.connect(self._load_current_measurement)
        refresh_button = QPushButton("Refresh Analysis")
        refresh_button.clicked.connect(self._load_current_measurement)
        control_box.addWidget(self.measurement_selector)
        control_box.addWidget(refresh_button)

        layout.addLayout(title_box, stretch=2)
        layout.addLayout(control_box, stretch=1)
        return header

    def _wrap_card(self, title: str, widget: QWidget) -> QWidget:
        card = QFrame()
        card.setProperty("card", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        card_title = QLabel(title)
        card_title.setObjectName("cardTitle")
        layout.addWidget(card_title)
        layout.addWidget(widget, stretch=1)
        return card

    def _build_health_card(self) -> QWidget:
        card = QFrame()
        card.setProperty("card", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Joint Health Indicator")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        top = QHBoxLayout()
        text_box = QVBoxLayout()
        text_box.addWidget(self.state_label)
        text_box.addWidget(self.state_value)
        text_box.addWidget(self.recommendation_label)
        top.addWidget(self.health_light)
        top.addLayout(text_box, stretch=1)
        layout.addLayout(top)
        return card

    def _build_prediction_card(self) -> QWidget:
        card = QFrame()
        card.setProperty("card", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Prediction Panel")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        for label, widget in (
            ("Time to critical threshold", self.eta_value),
            ("Remaining operating hours", self.operating_eta_value),
            ("Current RMS vector", self.rms_value),
            ("Dominant frequency", self.freq_value),
        ):
            heading = QLabel(label)
            heading.setObjectName("metricHeading")
            layout.addWidget(heading)
            layout.addWidget(widget)

        return card

    def _build_measurement_card(self) -> QWidget:
        card = QFrame()
        card.setProperty("card", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Dataset Context")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        layout.addWidget(self.meta_label)
        layout.addStretch(1)
        return card

    def _populate_measurements(self) -> None:
        self.measurement_selector.blockSignals(True)
        self.measurement_selector.clear()
        for measurement in self.measurements:
            label = (
                f"{measurement.joint} | {measurement.measurement_date} | "
                f"{measurement.station} | {measurement.trajectory}"
            )
            self.measurement_selector.addItem(label)
        self.measurement_selector.blockSignals(False)

    def _load_current_measurement(self) -> None:
        if not self.measurements:
            return

        measurement = self.measurements[self.measurement_selector.currentIndex()]
        try:
            self.analysis = self.pipeline.analyze_measurement(measurement)
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", str(exc))
            return

        self._update_summary()
        self._draw_signal_plot()
        self._draw_trend_plot()

    def _update_summary(self) -> None:
        if self.analysis is None:
            return

        assessment = self.analysis.health_assessment
        last_feature = self.analysis.feature_sets[-1]

        self.health_light.set_state(assessment.current_state)
        self.state_label.setText(assessment.current_state)
        self.state_value.setText(f"JHI: {assessment.current_health_index:.1f}")
        self.recommendation_label.setText(assessment.recommendation)

        if assessment.estimated_hours_to_critical is None:
            self.eta_value.setText("Stable / no crossing")
        else:
            self.eta_value.setText(f"{assessment.estimated_hours_to_critical:.1f} h")
        if assessment.estimated_operating_hours_to_critical is None:
            self.operating_eta_value.setText("Historical estimate unavailable")
        else:
            self.operating_eta_value.setText(f"{assessment.estimated_operating_hours_to_critical:.1f} h")

        self.rms_value.setText(f"{assessment.timeline[-1].rms_vector:.2f} mg")
        dominant_frequency = (
            last_feature.dominant_frequency_x_hz + last_feature.dominant_frequency_y_hz
        ) / 2.0
        self.freq_value.setText(f"{dominant_frequency:.1f} Hz")

        measurement = self.analysis.measurement
        self.meta_label.setText(
            "\n".join(
                [
                    f"Joint: {measurement.joint}",
                    f"Date: {measurement.measurement_date}",
                    f"Station: {measurement.station}",
                    f"Trajectory: {measurement.trajectory}",
                    f"Sampling rate: {self.analysis.sampling_rate_hz:.1f} Hz",
                    f"Signals: {self.analysis.selected_x_column} / {self.analysis.selected_y_column}",
                    (
                        "Operating hours to warning/critical: "
                        f"{assessment.historical_threshold_profile.warning_hours:.1f} / "
                        f"{assessment.historical_threshold_profile.critical_hours:.1f}"
                    )
                    if assessment.historical_threshold_profile
                    and assessment.historical_threshold_profile.warning_hours is not None
                    and assessment.historical_threshold_profile.critical_hours is not None
                    else "Operating-hour thresholds: not enough historical data",
                ]
            )
        )

    def _draw_signal_plot(self) -> None:
        if self.analysis is None:
            return

        self.signal_canvas.figure.clear()
        ax = self.signal_canvas.figure.add_subplot(111)
        self.signal_canvas.axes = ax

        timestamps = self.analysis.raw_timestamps
        x_signal = self.analysis.raw_signal_x
        y_signal = self.analysis.raw_signal_y

        sample_count = min(3000, len(timestamps))
        indices = np.linspace(0, len(timestamps) - 1, sample_count, dtype=int)
        x_axis = np.arange(sample_count)

        ax.plot(x_axis, x_signal[indices], color="#0F8B8D", linewidth=1.3, label="Raw X")
        ax.plot(x_axis, y_signal[indices], color="#F28F3B", linewidth=1.0, alpha=0.9, label="Raw Y")

        window_points = min(1024, len(x_signal))
        if window_points > 16:
            processed = x_signal[:window_points] - np.mean(x_signal[:window_points])
            processed_x = np.linspace(0, sample_count * (window_points / len(x_signal)), window_points)
            ax.plot(processed_x, processed, color="#284B63", linewidth=1.0, label="Processed X")

        ax.set_title("Raw and Processed Vibration Signal")
        ax.set_xlabel("Sample index (downsampled view)")
        ax.set_ylabel("Acceleration [mg]")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right")
        self.signal_canvas.draw_idle()

    def _draw_trend_plot(self) -> None:
        if self.analysis is None:
            return

        self.trend_canvas.figure.clear()
        ax = self.trend_canvas.figure.add_subplot(111)
        ax2 = ax.twinx()
        self.trend_canvas.axes = ax

        assessment = self.analysis.health_assessment
        x_axis = np.arange(len(assessment.timeline))
        rms_values = np.asarray([point.rms_vector for point in assessment.timeline], dtype=np.float64)
        health_values = np.asarray([point.health_index for point in assessment.timeline], dtype=np.float64)

        line1 = ax.plot(x_axis, rms_values, color="#2D7DD2", linewidth=1.8, label="RMS vector [mg]")[0]
        line2 = ax.plot(
            x_axis,
            assessment.moving_average * assessment.baseline_rms,
            color="#35A675",
            linewidth=1.4,
            linestyle="--",
            label="Moving average RMS estimate",
        )[0]
        line3 = ax2.plot(x_axis, health_values, color="#D8434B", linewidth=2.1, label="Joint Health Index")[0]

        warning_score = (
            assessment.historical_threshold_profile.warning_score
            if assessment.historical_threshold_profile is not None
            else assessment.thresholds.normal_max_score
        )
        critical_score = (
            assessment.historical_threshold_profile.critical_score
            if assessment.historical_threshold_profile is not None
            else assessment.thresholds.warning_max_score
        )
        ax2.axhline(warning_score, color="#E8A317", linestyle=":", linewidth=1.2)
        ax2.axhline(critical_score, color="#D8434B", linestyle=":", linewidth=1.2)

        ax.set_title("RMS and Health Trend")
        ax.set_xlabel("Sliding window")
        ax.set_ylabel("RMS vector [mg]")
        ax2.set_ylabel("Joint Health Index")
        ax.grid(alpha=0.18)
        ax2.set_ylim(0, 100)
        ax.legend([line1, line2, line3], [line1.get_label(), line2.get_label(), line3.get_label()], loc="upper left")
        self.trend_canvas.draw_idle()


def run() -> int:
    app = QApplication.instance() or QApplication([])
    pipeline = SmartSenseAnalysisPipeline()
    window = DashboardWindow(pipeline)
    window.show()
    return app.exec()
