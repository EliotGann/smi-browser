"""Reconstruct ``CombinedReductionResult`` / ``GIReductionResult`` shapes
from the disk cache so the Process tab can redisplay a previously-reduced
scan without re-running the reduction.

The disk cache stores numpy arrays + JSON-safe attrs; this module wraps
them in stub objects whose attribute surface matches what the Process-tab
display code reads from a live ``smi_tiled`` reduction result.

What is NOT reconstructible:

* per-detector ``q_chi_frames`` (lazy zarr-backed datasets) — large stacks
  live in a separate zarr store managed by smi-tiled and are not duplicated
  into the per-scan cache.  The Process tab reflects this by hiding the
  per-frame 2D toggle on cached reloads (the merged 2D and per-frame I(q)
  are still available).
* ``timing`` — recorded only by the live reducer.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .reduction_params import attach_attrs, dumps_params, flat_attrs, loads_params


# Connection / storage hints + already-discriminated keys we do NOT diff.
_PARAM_DIFF_SKIP = frozenset({
    "uid",
    "tiled_uri",
    "catalog",
    "image_cache_path",
    "cache_geometry",
    "progress",
    # peak_fits is a list of PeakDef instances on the live side; on the
    # cached side only the per-peak /peakfit results survive — the input
    # spec doesn't roundtrip through h5py attrs cleanly, so we ignore it
    # here.  Re-fitting peaks is its own user action.
    "peak_fits",
    # Geometry is the discriminator we use to pick the cached branch.
    "geometry",
    # GI provenance strings stashed for display only.
    "gi_scan_motor",
    "gi_alpha_i_source",
    "gi_sample_name",
})


_MISSING = object()


def _values_equal(a: Any, b: Any) -> bool:
    """Equality tolerant of ``NaN``/float-fuzz and tuple-vs-list mismatches."""
    if a is _MISSING or b is _MISSING:
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(a - b) < 1e-12
    return a == b


def params_for_diff(params: dict) -> dict:
    """Project a params dict onto the keys/types comparable with cached attrs.

    Mirrors the ``safe_params`` filter the writer applies, so widget-current
    params and what was cached can be compared shape-for-shape.
    """
    out: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k in _PARAM_DIFF_SKIP:
            continue
        if isinstance(v, (str, int, float, bool, type(None), list, tuple)):
            out[k] = v
    return out


def proc_params_differ(current: dict, cached: dict) -> list[str]:
    """Return sorted keys whose value differs between the two param dicts.

    Used by the Process tab to flag a cache hit as stale relative to the
    current widget configuration.  Empty list means the cache is in sync.
    """
    a = params_for_diff(current)
    b = params_for_diff(cached)
    keys = set(a) | set(b)
    diffs: list[str] = []
    for k in keys:
        if not _values_equal(a.get(k, _MISSING), b.get(k, _MISSING)):
            diffs.append(k)
    return sorted(diffs)


# ---------------------------------------------------------------------------
# Result stubs
# ---------------------------------------------------------------------------

class CachedResult:
    """Stand-in for a transmission ``CombinedReductionResult`` reconstructed
    from the disk cache.  Exposes the attributes Process-tab display code
    reads (``merged_iq``, ``merged_qchi``, ``per_frame_iq``, ``peak_fits``,
    ``saxs``/``waxs``, ``geometry``, ``uid``); per-detector lazy
    ``q_chi_frames`` are not cached, so per-frame 2D maps fall back to the
    merged image.
    """

    def __init__(
        self,
        uid: str,
        merged_iq,
        *,
        merged_qchi=None,
        per_frame_iq=None,
        peak_fits=None,
        geometry: str = "transmission",
        reduction_parameters: dict[str, Any] | None = None,
    ):
        self.uid = uid
        self.merged_iq = merged_iq
        self.merged_qchi = merged_qchi
        self.per_frame_iq = per_frame_iq
        self.peak_fits = peak_fits
        self.reduction_parameters = reduction_parameters
        # No lazy per-frame zarr stacks on the cached path.
        self.saxs = None
        self.waxs = None
        self.timing = None
        self.geometry = geometry


class CachedGiResult:
    """Stand-in for a ``GIReductionResult`` reconstructed from the disk cache.

    Mirrors the smi-tiled dataclass attributes that the Process tab reads:
    ``frames``, ``summed``, ``qxy_grid``, ``qz_grid``, ``alpha_i_deg``, plus
    provenance strings (``scan_motor``, ``alpha_i_source``, ``sample_name``).
    """

    def __init__(
        self,
        uid: str,
        *,
        frames,
        summed,
        qxy_grid,
        qz_grid,
        alpha_i_deg,
        scan_motor: str = "",
        scan_motor_values=None,
        alpha_i_source: str = "cached",
        sample_name: str = "",
        reduction_parameters: dict[str, Any] | None = None,
    ):
        self.uid = uid
        self.frames = list(frames) if frames is not None else []
        self.summed = summed
        self.qxy_grid = qxy_grid
        self.qz_grid = qz_grid
        self.alpha_i_deg = alpha_i_deg
        self.scan_motor = scan_motor
        self.scan_motor_values = scan_motor_values
        self.alpha_i_source = alpha_i_source
        self.sample_name = sample_name
        self.q_chi_frames = None
        self.summed_ds = None
        self.timing = None
        self.line_cuts = None
        self.peak_fits = None
        self.geometry = "grazing"
        self.reduction_parameters = reduction_parameters


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def peak_fits_from_cache(cache) -> "Any | None":
    """Rebuild an ``apply_peak_fits``-shaped ``(peak, frame)`` ``xr.Dataset``
    from the per-peak entries persisted in the cache, or ``None`` if no peak
    fits are stored.  Output layout matches what smi-tiled emits, so an
    exporter that reads ``result.peak_fits`` works on the cached path too.
    """
    import xarray as xr

    entries = cache.read_peakfit_full()
    if not entries:
        return None

    # Order peaks by name for stability.
    entries = sorted(entries, key=lambda e: e.get("attrs", {}).get("name", ""))
    n_peaks = len(entries)

    first = entries[0]["arrays"]
    n_frames = int(np.asarray(first.get("amplitude", [])).shape[0])

    def _stack(field: str, dtype, fill) -> np.ndarray:
        arr = np.full((n_peaks, n_frames), fill, dtype=dtype)
        for i, e in enumerate(entries):
            v = np.asarray(e["arrays"].get(field, []))
            if v.shape[:1] == (n_frames,):
                arr[i] = v
        return arr

    names = [str(e["attrs"].get("name", "")) for e in entries]
    return xr.Dataset(
        {
            "amplitude": (("peak", "frame"), _stack("amplitude", float, np.nan)),
            "center":    (("peak", "frame"), _stack("center", float, np.nan)),
            "fwhm":      (("peak", "frame"), _stack("fwhm", float, np.nan)),
            "area":      (("peak", "frame"), _stack("area", float, np.nan)),
            "success":   (("peak", "frame"), _stack("success", bool, False)),
        },
        coords={
            "peak": np.array(names, dtype=object),
            "frame": np.arange(n_frames),
        },
        attrs={"peaks": [dict(e["attrs"]) for e in entries]},
    )


def build_cached_result(cache):
    """Build a ``CachedResult`` / ``CachedGiResult`` from a ``ScanCache``.

    Returns ``(result_or_None, params_dict)``.  ``result_or_None`` is None
    if no usable reduction was cached.  ``cache`` is anything that exposes
    ``read_reduction()`` and ``read_peakfit_full()`` — typically a
    :class:`smi_browser.cache.ScanCache`.
    """
    import xarray as xr

    cached = cache.read_reduction()
    if cached is None:
        return None, {}
    arrays = cached.get("arrays", {})
    params = cached.get("params", {})
    geometry = params.get("geometry", "transmission")
    reduction_parameters = loads_params(params.get("smi_reduction_parameters"))

    if geometry == "grazing":
        frames = arrays.get("gi_frames")
        if frames is None or np.asarray(frames).size == 0:
            return None, params
        gi = CachedGiResult(
            uid=str(getattr(cache, "uid", "")),
            frames=[np.asarray(f) for f in frames],
            summed=arrays.get("gi_summed"),
            qxy_grid=arrays.get("gi_qxy"),
            qz_grid=arrays.get("gi_qz"),
            alpha_i_deg=arrays.get("gi_alpha_i_deg"),
            scan_motor_values=arrays.get("gi_scan_motor_values"),
            scan_motor=str(params.get("gi_scan_motor", "")),
            alpha_i_source=str(params.get("gi_alpha_i_source", "cached")),
            sample_name=str(params.get("gi_sample_name", "")),
            reduction_parameters=reduction_parameters,
        )
        return gi, params

    # Transmission
    q = arrays.get("iq_q")
    I = arrays.get("iq_I")
    if q is None or I is None:
        return None, params

    iq_vars: dict[str, tuple] = {"I": (("q",), I)}
    for opt in ("saxs_I", "waxs_I", "counts"):
        v = arrays.get(f"iq_{opt}")
        if v is not None:
            iq_vars[opt] = (("q",), v)
    merged_iq = xr.Dataset(iq_vars, coords={"q": q})
    attach_attrs(merged_iq, reduction_parameters)

    merged_qchi = None
    qchi_intensity = arrays.get("qchi_intensity")
    if qchi_intensity is not None:
        coords: dict[str, Any] = {}
        if "qchi_q" in arrays:
            coords["q"] = arrays["qchi_q"]
        if "qchi_chi" in arrays:
            coords["chi"] = arrays["qchi_chi"]
        merged_qchi = xr.Dataset(
            {"intensity": (("q", "chi"), qchi_intensity)},
            coords=coords,
        )
        attach_attrs(merged_qchi, reduction_parameters)

    per_frame_iq = None
    pf_I = arrays.get("pf_iq_I")
    pf_q = arrays.get("pf_iq_q")
    if pf_I is not None and pf_q is not None:
        pf_vars: dict[str, tuple] = {"I": (("frame", "q"), pf_I)}
        for opt in ("saxs_I", "waxs_I"):
            v = arrays.get(f"pf_iq_{opt}")
            if v is not None:
                pf_vars[opt] = (("frame", "q"), v)
        per_frame_iq = xr.Dataset(
            pf_vars,
            coords={"q": pf_q, "frame": np.arange(pf_I.shape[0])},
        )
        attach_attrs(per_frame_iq, reduction_parameters)

    peak_fits = peak_fits_from_cache(cache)

    return (
        CachedResult(
            uid=str(getattr(cache, "uid", "")),
            merged_iq=merged_iq,
            merged_qchi=merged_qchi,
            per_frame_iq=per_frame_iq,
            peak_fits=peak_fits,
            geometry=geometry,
            reduction_parameters=reduction_parameters,
        ),
        params,
    )


def reduction_params_to_attrs(params: dict[str, Any] | None) -> dict[str, Any]:
    """Return cache-safe attrs for resolved reduction parameters."""
    if not params:
        return {}
    return {
        "smi_reduction_parameters_schema": "smi_tiled.resolved_reduction_parameters.v1",
        "smi_reduction_parameters": dumps_params(params),
        **flat_attrs(params),
    }
