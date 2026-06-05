"""Per-frame peak fitting — thin re-export shim over :mod:`smi_tiled.derived.peakfit`.

The implementation now lives in ``smi-tiled`` so that peak fits flow through
:class:`smi_tiled.integrator.CombinedReductionResult` and into the upload-tiled
catalog with provenance.  This module preserves the historical import path so
existing browser code and tests keep working unchanged.

Browser-only naming helpers (:func:`peak_q_center`, :func:`peak_display_label`,
:func:`peak_slug`) live here too so UI, exports, and HDF5 writes share one
canonical formatting of a peak's identity.
"""
from __future__ import annotations

import re
from typing import Any

from smi_tiled.derived.peakfit import (
    FIT_PARAMS,
    MIN_R2,
    MIN_SNR,
    PeakDef,
    apply_peak_fits,
    fit_peak_across_frames,
)


# ---------------------------------------------------------------------------
# Naming helpers — canonical "this peak" label/slug used everywhere.
# ---------------------------------------------------------------------------

#: Decimal places used to render the q-centre in display labels and slugs.
#: Three decimals matches the precision the Peak Map UI uses for box-select
#: emitted q-ranges and is enough to discriminate peaks at SAXS resolution.
_Q_DECIMALS = 3

# Characters that survive into a filesystem/H5-safe slug.  Anything else
# collapses to a single underscore.  Keep ``.`` so the rendered q-centre
# stays human-legible (``q1.234``) instead of ``q1_234``.
_SLUG_KEEP_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def peak_q_center(peak: "PeakDef | dict | Any") -> float:
    """Return the centre q-value of the *drawn* band (``(q_min+q_max)/2``).

    Used for display/slug formatting.  This is intentionally NOT the fitted
    centre — fitted centres vary per frame, while the drawn band is the
    stable identity the user originally selected.

    Accepts a :class:`PeakDef`, a dict (``{"q_min": ..., "q_max": ...}``),
    or any object exposing ``q_min``/``q_max`` attributes — the export
    layer hands us dicts, the UI hands us :class:`PeakDef` instances.
    """
    if isinstance(peak, dict):
        q_min = float(peak.get("q_min", 0.0))
        q_max = float(peak.get("q_max", 0.0))
    else:
        q_min = float(getattr(peak, "q_min", 0.0))
        q_max = float(getattr(peak, "q_max", 0.0))
    return (q_min + q_max) / 2.0


def _peak_name(peak: "PeakDef | dict | Any") -> str:
    if isinstance(peak, dict):
        return str(peak.get("name") or "").strip()
    return str(getattr(peak, "name", "") or "").strip()


def peak_display_label(peak: "PeakDef | dict | Any") -> str:
    """Human-readable label, e.g. ``"alpha (q=1.234)"``.

    Falls back to ``"(q=1.234)"`` if the peak has no name set.
    """
    name = _peak_name(peak)
    qs = f"{peak_q_center(peak):.{_Q_DECIMALS}f}"
    if not name:
        return f"(q={qs})"
    return f"{name} (q={qs})"


def peak_slug(peak: "PeakDef | dict | Any") -> str:
    """Filesystem/H5-safe identifier, e.g. ``"alpha_q1.234"``.

    Always includes the q-centre so two peaks with the same user-given name
    (e.g. duplicated by accident) still produce distinct filenames and HDF5
    group keys.  When the user didn't set a name, the slug is just
    ``"q1.234"``.
    """
    name = _peak_name(peak)
    qs = f"{peak_q_center(peak):.{_Q_DECIMALS}f}"
    if not name:
        return f"q{qs}"
    safe = _SLUG_KEEP_RE.sub("_", name).strip("_")
    if not safe:
        return f"q{qs}"
    return f"{safe}_q{qs}"


__all__ = [
    "FIT_PARAMS",
    "MIN_R2",
    "MIN_SNR",
    "PeakDef",
    "apply_peak_fits",
    "fit_peak_across_frames",
    "peak_display_label",
    "peak_q_center",
    "peak_slug",
]
