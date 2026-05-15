from __future__ import annotations

from src.data_loading.loader import VibrationDataLoader
from src.processing.features import VibrationFeatureExtractor


def main() -> None:
    loader = VibrationDataLoader()
    measurements = loader.discover_measurements()

    if not measurements:
        raise SystemExit("No paired X/Y vibration measurements were found in the data directory.")

    measurement = measurements[0]
    preview = loader.build_preview(measurement)

    print("SmartSense vibration data validation")
    print(f"Joint: {measurement.joint}")
    print(f"Date: {measurement.measurement_date}")
    print(f"Station: {measurement.station}")
    print(f"Trajectory: {measurement.trajectory}")
    print(f"X file: {measurement.x_path}")
    print(f"Y file: {measurement.y_path}")
    print()

    print("X columns:")
    for column in preview.x_columns:
        print(f"  - {column}")
    print()

    print("Y columns:")
    for column in preview.y_columns:
        print(f"  - {column}")
    print()

    print("Selected vibration columns:")
    print(f"  - {preview.selected_columns['x']}")
    print(f"  - {preview.selected_columns['y']}")
    print()

    print("Preview rows:")
    for row in preview.preview_rows:
        print(row)
    print()

    confirmation = input(
        "Please confirm that these are the correct vibration signals. [y/N]: "
    ).strip().lower()

    if confirmation != "y":
        raise SystemExit("Confirmation not received. Processing must not continue.")

    print("Confirmation received. The dataset is validated for the next processing stage.")

    extractor = VibrationFeatureExtractor(window_size=2048, step_size=1024)
    paired_rows, _, _ = loader.load_selected_signals(measurement)
    feature_sets = extractor.extract_features(paired_rows)

    print(f"Generated {len(feature_sets)} sliding-window feature rows.")
    if feature_sets:
        first = feature_sets[0]
        print(
            "First window summary:",
            {
                "rms_x": round(first.rms_x, 4),
                "rms_y": round(first.rms_y, 4),
                "rms_vector": round(first.rms_vector, 4),
                "peak_x": round(first.peak_x, 4),
                "peak_y": round(first.peak_y, 4),
                "crest_factor_x": round(first.crest_factor_x, 4),
                "crest_factor_y": round(first.crest_factor_y, 4),
                "dominant_frequency_x_hz": round(first.dominant_frequency_x_hz, 4),
                "dominant_frequency_y_hz": round(first.dominant_frequency_y_hz, 4),
            },
        )


if __name__ == "__main__":
    main()
