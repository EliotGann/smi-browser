"""Tests for ``smi_browser.models.cached_result``.

Covers the disk-cache → result-stub reconstruction used to redisplay the
Process tab without re-running a reduction, plus the param-drift helper
that flags a cached hit as stale relative to the current widget settings.
"""
from __future__ import annotations

import numpy as np
import pytest

from smi_browser import cache as cache_mod
from smi_browser.cache import ScanCache
from smi_browser.models.cached_result import (
    CachedGiResult,
    CachedResult,
    build_cached_result,
    params_for_diff,
    peak_fits_from_cache,
    proc_params_differ,
)


def _resolved_params():
    return {
        "schema": "smi_tiled.resolved_reduction_parameters.v1",
        "saxs": {
            "sample_detector_distance_mm": 5020.0,
            "beam_center_row_px": 1165.2,
            "beam_center_col_px": 746.4,
            "energy_kev": 13.5,
        },
        "waxs": {
            "sample_detector_distance_mm": 278.0,
            "beam_center_row_px": 217.0,
            "beam_center_col_px": 314.5,
            "energy_kev": 13.5,
        },
    }


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMI_BROWSER_CACHE_DIR", str(tmp_path))
    cache_mod._LOCK_TABLE.clear()
    yield tmp_path


# ---------------------------------------------------------------------------
# Param-diff helpers
# ---------------------------------------------------------------------------

def test_params_for_diff_strips_storage_hints_and_non_primitives():
    """Connection / storage hints + non-primitive values must drop out so we
    only compare reduction-affecting widget settings."""
    out = params_for_diff({
        "uid": "abc",                 # excluded — connection
        "tiled_uri": "https://x",     # excluded
        "catalog": "raw",             # excluded
        "image_cache_path": "/tmp",   # excluded
        "cache_geometry": True,       # excluded
        "progress": object(),         # not a primitive
        "peak_fits": [object()],      # excluded explicitly
        "geometry": "transmission",   # excluded — discriminator
        "n_q": 1000,
        "n_chi": 360,
        "saxs_beam_delta_px": (0.0, 1.0),
        "dezinger_threshold": None,
    })
    assert out == {
        "n_q": 1000,
        "n_chi": 360,
        "saxs_beam_delta_px": (0.0, 1.0),
        "dezinger_threshold": None,
    }


def test_proc_params_differ_no_difference():
    a = {"n_q": 1000, "n_chi": 360, "uid": "abc"}
    b = {"n_q": 1000, "n_chi": 360, "uid": "different"}
    # ``uid`` is in the skip set, so the two dicts are equivalent.
    assert proc_params_differ(a, b) == []


def test_proc_params_differ_reports_changed_and_added_keys():
    """Mismatched values AND keys present on only one side both surface."""
    cur = {"n_q": 2000, "n_chi": 360, "saxs_q_cutoff": 0.5}
    cached = {"n_q": 1000, "n_chi": 360}  # saxs_q_cutoff missing
    diffs = proc_params_differ(cur, cached)
    assert diffs == ["n_q", "saxs_q_cutoff"]


def test_proc_params_differ_handles_floats_tuples_nan():
    """Float fuzz, tuple/list interchangeability, and NaN equality must
    not produce false-positive diffs (otherwise the user is constantly
    nudged to re-process)."""
    a = {
        "x": 1.0 + 1e-15,
        "ts": (1, 2, 3),
        "missing": float("nan"),
    }
    b = {
        "x": 1.0,
        "ts": [1, 2, 3],
        "missing": float("nan"),
    }
    assert proc_params_differ(a, b) == []


# ---------------------------------------------------------------------------
# peak_fits_from_cache: rebuild (peak, frame) Dataset from /peakfit/<hash>
# ---------------------------------------------------------------------------

def test_peak_fits_from_cache_none_when_empty():
    c = ScanCache("uid-pf-none")
    assert peak_fits_from_cache(c) is None


def test_peak_fits_from_cache_rebuilds_dataset_shape():
    """The reconstruction must match smi-tiled's apply_peak_fits output:
    dims ``(peak, frame)``, vars ``amplitude/center/fwhm/area/success``,
    and ``attrs["peaks"]`` carrying provenance dicts in the same order."""
    c = ScanCache("uid-pf-rebuild")
    c.write_peakfit(
        (1.1, 1.4, "gaussian", "linear", "linked", 2.0),
        {
            "amplitude": np.array([1.0, 2.0, 3.0]),
            "center":    np.array([5.0, 5.1, 5.2]),
            "fwhm":      np.array([0.3, 0.3, 0.3]),
            "area":      np.array([1.5, 1.5, 1.5]),
            "success":   np.array([True, True, False]),
        },
        attrs={"name": "alpha", "q_min": 1.1, "q_max": 1.4,
               "model": "gaussian", "baseline": "linear",
               "link": "linked", "bg_factor": 2.0},
    )
    c.write_peakfit(
        (2.1, 2.4, "lorentzian", "none", "independent", 3.0),
        {
            "amplitude": np.array([10.0, 20.0, 30.0]),
            "center":    np.array([6.0, 6.0, 6.0]),
            "fwhm":      np.array([0.4, 0.4, 0.4]),
            "area":      np.array([2.5, 2.5, 2.5]),
            "success":   np.array([True, True, True]),
        },
        attrs={"name": "beta", "q_min": 2.1, "q_max": 2.4,
               "model": "lorentzian", "baseline": "none",
               "link": "independent", "bg_factor": 3.0},
    )

    ds = peak_fits_from_cache(c)
    assert ds is not None
    assert dict(ds.sizes) == {"peak": 2, "frame": 3}
    assert list(ds.data_vars) == ["amplitude", "center", "fwhm", "area", "success"]
    # Sorted by name for stability — alpha first, beta second.
    np.testing.assert_array_equal(ds["peak"].values.astype(str), ["alpha", "beta"])
    np.testing.assert_allclose(ds["amplitude"].values,
                               [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    assert ds["success"].dtype == bool
    np.testing.assert_array_equal(
        ds["success"].values, [[True, True, False], [True, True, True]],
    )
    # Provenance roundtrips through attrs.
    assert ds.attrs["peaks"][0]["name"] == "alpha"
    assert ds.attrs["peaks"][1]["model"] == "lorentzian"


# ---------------------------------------------------------------------------
# build_cached_result: full transmission round-trip
# ---------------------------------------------------------------------------

def _write_transmission_arrays(c: ScanCache, *, n_q=8, n_frames=4, n_chi=6,
                               with_per_detector=True, with_pf=True,
                               with_qchi=True):
    """Write a synthetic reduction blob mirroring _cache_reduction_result."""
    q = np.linspace(0.01, 5.0, n_q)
    arrays: dict[str, np.ndarray] = {
        "iq_q": q,
        "iq_I": np.linspace(1.0, 10.0, n_q),
    }
    if with_per_detector:
        arrays["iq_saxs_I"] = np.linspace(0.0, 5.0, n_q)
        arrays["iq_waxs_I"] = np.linspace(5.0, 0.0, n_q)
        arrays["iq_counts"] = np.full(n_q, 100.0)
    if with_qchi:
        arrays["qchi_q"] = q
        arrays["qchi_chi"] = np.linspace(-90, 90, n_chi)
        arrays["qchi_intensity"] = np.random.rand(n_q, n_chi)
    if with_pf:
        arrays["pf_iq_q"] = q
        arrays["pf_iq_I"] = np.random.rand(n_frames, n_q)
        arrays["pf_iq_saxs_I"] = np.random.rand(n_frames, n_q)
        arrays["pf_iq_waxs_I"] = np.random.rand(n_frames, n_q)
    return arrays


def test_build_cached_result_returns_none_when_no_reduction():
    c = ScanCache("uid-empty-cache")
    result, params = build_cached_result(c)
    assert result is None
    assert params == {}


def test_build_cached_result_transmission_full_shape():
    """All transmission keys present → CachedResult exposes merged_iq with
    per-detector vars, merged_qchi, per_frame_iq with per-detector vars,
    and the geometry attribute."""
    c = ScanCache("uid-trans-full")
    arrays = _write_transmission_arrays(c)
    c.write_reduction(arrays, {"geometry": "transmission", "n_q": 8})

    result, params = build_cached_result(c)
    assert isinstance(result, CachedResult)
    assert result.geometry == "transmission"
    assert result.uid == "uid-trans-full"
    assert params["geometry"] == "transmission"
    assert int(params["n_q"]) == 8

    # merged_iq: data_vars I + per-detector overlay traces + counts
    assert set(result.merged_iq.data_vars) == {"I", "saxs_I", "waxs_I", "counts"}
    np.testing.assert_allclose(result.merged_iq["q"].values, arrays["iq_q"])
    np.testing.assert_allclose(result.merged_iq["I"].values, arrays["iq_I"])

    # merged_qchi: 2D intensity on (q, chi)
    assert result.merged_qchi is not None
    assert "intensity" in result.merged_qchi
    assert result.merged_qchi["intensity"].dims == ("q", "chi")
    np.testing.assert_allclose(result.merged_qchi["intensity"].values,
                               arrays["qchi_intensity"])

    # per_frame_iq: (frame, q) plus per-detector overlay traces
    assert result.per_frame_iq is not None
    assert set(result.per_frame_iq.data_vars) == {"I", "saxs_I", "waxs_I"}
    assert result.per_frame_iq["I"].dims == ("frame", "q")

    # No q_chi_frames stub (lazy zarr; caller falls back to merged map)
    assert result.saxs is None
    assert result.waxs is None


def test_build_cached_result_restores_resolved_parameter_attrs():
    from smi_browser.models.reduction_params import dumps_params

    c = ScanCache("uid-trans-params")
    arrays = _write_transmission_arrays(c)
    resolved = _resolved_params()
    c.write_reduction(
        arrays,
        {
            "geometry": "transmission",
            "smi_reduction_parameters": dumps_params(resolved),
            "saxs_sdd_mm": 5020.0,
            "waxs_sdd_mm": 278.0,
        },
    )

    result, _ = build_cached_result(c)

    assert result.reduction_parameters["saxs"]["sample_detector_distance_mm"] == 5020.0
    assert result.merged_iq.attrs["saxs_sdd_mm"] == pytest.approx(5020.0)
    assert result.merged_qchi.attrs["waxs_sdd_mm"] == pytest.approx(278.0)


def test_build_cached_result_transmission_minimal_no_per_detector_or_pf():
    """Older cache files without per-detector or per-frame keys still load —
    they just expose less."""
    c = ScanCache("uid-trans-min")
    arrays = _write_transmission_arrays(
        c, with_per_detector=False, with_pf=False, with_qchi=False,
    )
    c.write_reduction(arrays, {"geometry": "transmission"})

    result, _ = build_cached_result(c)
    assert isinstance(result, CachedResult)
    assert set(result.merged_iq.data_vars) == {"I"}
    assert result.merged_qchi is None
    assert result.per_frame_iq is None


def test_build_cached_result_transmission_drops_when_iq_missing():
    """Without iq_q / iq_I we can't draw anything useful — bail out."""
    c = ScanCache("uid-trans-bad")
    c.write_reduction({"qchi_intensity": np.zeros((4, 4))},
                      {"geometry": "transmission"})
    result, params = build_cached_result(c)
    assert result is None
    assert params["geometry"] == "transmission"


def test_build_cached_result_transmission_attaches_peak_fits():
    """Peak fits cached in /peakfit must be reattached as a (peak, frame)
    Dataset on the rehydrated CachedResult, so exporters and Peak Map work
    on the cached path."""
    c = ScanCache("uid-trans-peaks")
    arrays = _write_transmission_arrays(c, n_frames=3)
    c.write_reduction(arrays, {"geometry": "transmission"})
    c.write_peakfit(
        (1.1, 1.4, "gaussian", "none", "linked", 2.0),
        {
            "amplitude": np.array([1.0, 2.0, 3.0]),
            "center":    np.array([5.0, 5.0, 5.0]),
            "fwhm":      np.array([0.3, 0.3, 0.3]),
            "area":      np.array([1.5, 1.5, 1.5]),
            "success":   np.ones(3, dtype=bool),
        },
        attrs={"name": "alpha", "q_min": 1.1, "q_max": 1.4,
               "model": "gaussian", "baseline": "none",
               "link": "linked", "bg_factor": 2.0},
    )

    result, _ = build_cached_result(c)
    assert result.peak_fits is not None
    assert result.peak_fits.sizes == {"peak": 1, "frame": 3}


# ---------------------------------------------------------------------------
# build_cached_result: GI round-trip
# ---------------------------------------------------------------------------

def test_build_cached_result_grazing_full_shape():
    """GI arrays + provenance attrs round-trip into CachedGiResult so the
    Process tab can redraw qxy/qz maps without re-reducing."""
    c = ScanCache("uid-gi-full")
    n_fr, n_qxy, n_qz = 3, 5, 4
    qxy = np.linspace(-1, 1, n_qxy)
    qz = np.linspace(0, 2, n_qz)
    frames = np.random.rand(n_fr, n_qxy, n_qz)
    summed = frames.mean(axis=0)
    alpha_i = np.array([0.10, 0.20, 0.30])
    smv = np.array([1.0, 2.0, 3.0])
    c.write_reduction(
        {
            "gi_frames": frames,
            "gi_qxy": qxy,
            "gi_qz": qz,
            "gi_summed": summed,
            "gi_alpha_i_deg": alpha_i,
            "gi_scan_motor_values": smv,
        },
        {
            "geometry": "grazing",
            "gi_scan_motor": "piezo_th",
            "gi_alpha_i_source": "cached_test",
            "gi_sample_name": "Si-cal",
        },
    )

    result, params = build_cached_result(c)
    assert isinstance(result, CachedGiResult)
    assert result.geometry == "grazing"
    assert len(result.frames) == n_fr
    np.testing.assert_allclose(result.qxy_grid, qxy)
    np.testing.assert_allclose(result.qz_grid, qz)
    np.testing.assert_allclose(result.summed, summed)
    np.testing.assert_allclose(result.alpha_i_deg, alpha_i)
    assert result.scan_motor == "piezo_th"
    assert result.alpha_i_source == "cached_test"
    assert result.sample_name == "Si-cal"
    assert params["geometry"] == "grazing"


def test_build_cached_result_grazing_drops_when_no_frames():
    """A GI cache with the discriminator flag but no frames is unusable."""
    c = ScanCache("uid-gi-empty")
    c.write_reduction(
        {"gi_frames": np.zeros((0,))},
        {"geometry": "grazing"},
    )
    result, params = build_cached_result(c)
    assert result is None
    assert params["geometry"] == "grazing"
