"""Detector field classification and frame orientation helpers."""
from __future__ import annotations

import numpy as np

from PyHyperScattering import smi_defaults as smid


def detector_for_field(field: str) -> str | None:
    """Classify a detector field name as ``'saxs'`` / ``'waxs'`` / ``None``."""
    return smid.classify_detector_field(field)


def orient_frame(arr: np.ndarray, field: str) -> np.ndarray:
    """Re-orient detector frames for display via the canonical PyHyper transform."""
    detector = detector_for_field(field)
    if detector is None:
        return arr
    return smid.orient_frame_for_display(arr, detector)
