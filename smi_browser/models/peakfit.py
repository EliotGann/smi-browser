"""Per-frame peak fitting — thin re-export shim over :mod:`smi_tiled.derived.peakfit`.

The implementation now lives in ``smi-tiled`` so that peak fits flow through
:class:`smi_tiled.integrator.CombinedReductionResult` and into the upload-tiled
catalog with provenance.  This module preserves the historical import path so
existing browser code and tests keep working unchanged.
"""
from __future__ import annotations

from smi_tiled.derived.peakfit import (
    FIT_PARAMS,
    MIN_R2,
    MIN_SNR,
    PeakDef,
    apply_peak_fits,
    fit_peak_across_frames,
)

__all__ = [
    "FIT_PARAMS",
    "MIN_R2",
    "MIN_SNR",
    "PeakDef",
    "apply_peak_fits",
    "fit_peak_across_frames",
]
