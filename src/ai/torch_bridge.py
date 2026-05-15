from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from src.processing.features import TimeSeriesFeatureSet, VibrationFeatureExtractor


def build_torch_training_data(
    feature_sets: Sequence[TimeSeriesFeatureSet],
    extractor: VibrationFeatureExtractor,
):
    """Converts engineered features into tensors for later PyTorch models."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install 'torch' before using AI model preparation."
        ) from exc

    feature_matrix, feature_names = extractor.features_to_matrix(feature_sets)
    tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    return tensor, feature_names
