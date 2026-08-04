"""Tests for the export layer: granular HDF5 sections + peak-result PNGs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smi_browser.export import (
    _save_dataset_h5,
    _save_peak_pngs,
    export_scan,
)


# --- helpers ---------------------------------------------------------------

def _peak_results(n=6):
    return {
        "amplitude": np.linspace(1.0, 2.0, n),
        "center": np.full(n, 2.3),
        "fwhm": np.full(n, 0.1),
        "area": np.linspace(0.5, 1.5, n),
        "success": np.ones(n, dtype=bool),
    }


def _peak_fit(name="p1", q_min=2.1, q_max=2.45):
    return {
        "name": name, "q_min": q_min, "q_max": q_max,
        "model": "gaussian", "baseline": "linear", "link": "linked",
        "bg_factor": 2.0, "results": _peak_results(),
    }


def _fake_transmission_result():
    import xarray as xr
    q = np.linspace(0.1, 5.0, 10)
    chi = np.linspace(-90.0, 90.0, 8)
    merged_iq = xr.Dataset({"I": ("q", np.ones(10))}, coords={"q": q})
    merged_qchi = xr.Dataset(
        {"intensity": (("q", "chi"), np.ones((10, 8)))},
        coords={"q": q, "chi": chi},
    )
    return SimpleNamespace(
        geometry="transmission", uid="u-test",
        merged_iq=merged_iq, merged_qchi=merged_qchi,
        per_frame_iq=None, saxs=None, waxs=None,
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


def _full_transmission_result(n_frames: int = 4):
    """Realistic transmission result mirroring smi-tiled output shape.

    Includes all 6 ``merged_qchi`` data vars, native per-detector ``iq``
    Datasets, per-frame q-chi stacks (with ``counts``), provenance attrs
    (``incident_angle_deg``, ``scan_info``).
    """
    import xarray as xr
    q = np.linspace(0.1, 5.0, 10)
    chi = np.linspace(-90.0, 90.0, 8)
    saxs_q = np.linspace(0.05, 1.5, 12)  # native SAXS grid
    waxs_q = np.linspace(0.5, 5.5, 14)   # native WAXS grid
    merged_iq = xr.Dataset(
        {
            "I": ("q", np.ones(10)),
            "counts": ("q", np.ones(10) * 100),
            "saxs_I": ("q", np.ones(10)),
            "waxs_I": ("q", np.ones(10)),
        },
        coords={"q": q},
    )
    merged_qchi = xr.Dataset(
        {
            "intensity": (("q", "chi"), np.ones((10, 8))),
            "counts": (("q", "chi"), np.ones((10, 8)) * 100),
            "saxs_intensity": (("q", "chi"), np.ones((10, 8)) * 0.5),
            "saxs_counts": (("q", "chi"), np.ones((10, 8)) * 50),
            "waxs_intensity": (("q", "chi"), np.ones((10, 8)) * 0.5),
            "waxs_counts": (("q", "chi"), np.ones((10, 8)) * 50),
        },
        coords={"q": q, "chi": chi},
    )
    saxs_iq = xr.Dataset(
        {"I": ("q", np.ones(12)), "counts": ("q", np.ones(12) * 30)},
        coords={"q": saxs_q},
    )
    waxs_iq = xr.Dataset(
        {"I": ("q", np.ones(14)), "counts": ("q", np.ones(14) * 70)},
        coords={"q": waxs_q},
    )
    saxs_qchi_frames = xr.Dataset(
        {
            "intensity": (
                ("frame", "q", "chi"),
                np.random.RandomState(0).rand(n_frames, 10, 8),
            ),
            "counts": (
                ("frame", "q", "chi"),
                np.ones((n_frames, 10, 8)) * 25,
            ),
        },
        coords={"frame": np.arange(n_frames), "q": q, "chi": chi},
    )
    waxs_qchi_frames = xr.Dataset(
        {
            "intensity": (
                ("frame", "q", "chi"),
                np.random.RandomState(1).rand(n_frames, 10, 8),
            ),
            "counts": (
                ("frame", "q", "chi"),
                np.ones((n_frames, 10, 8)) * 25,
            ),
        },
        coords={"frame": np.arange(n_frames), "q": q, "chi": chi},
    )
    return SimpleNamespace(
        geometry="transmission", uid="u-test",
        merged_iq=merged_iq, merged_qchi=merged_qchi,
        per_frame_iq=None,
        saxs={"iq": saxs_iq, "q_chi_frames": saxs_qchi_frames},
        waxs={"iq": waxs_iq, "q_chi_frames": waxs_qchi_frames},
        incident_angle_deg=0.12,
        scan_info={"sample_name": "demo", "n_frames": n_frames,
                   "step_candidates": ["stage_x", "stage_y"]},
        reduction_parameters=_resolved_params(),
    )


def _full_gi_result(n_frames: int = 3):
    """Realistic GI result with q_chi_frames (qxy/qz coords) + provenance."""
    import xarray as xr
    qxy = np.linspace(-1.0, 1.0, 6)
    qz = np.linspace(0.0, 2.0, 7)
    frame_maps = [
        np.random.RandomState(i).rand(qxy.size, qz.size) for i in range(n_frames)
    ]
    summed = np.mean(np.stack(frame_maps), axis=0)
    q_chi_frames = xr.Dataset(
        {"intensity": (("frame", "qxy", "qz"), np.stack(frame_maps))},
        coords={"frame": np.arange(n_frames), "qxy": qxy, "qz": qz},
    )
    return SimpleNamespace(
        uid="u-gi", sample_name="film",
        scan_motor="piezo_th",
        scan_motor_values=np.linspace(0.1, 0.3, n_frames),
        alpha_i_deg=np.full(n_frames, 0.2),
        alpha_i_source="motor",
        qxy_grid=qxy, qz_grid=qz,
        frames=frame_maps, summed=summed,
        q_chi_frames=q_chi_frames, summed_ds=None,
    )


def _groups(path):
    import h5py
    with h5py.File(path, "r") as f:
        return set(f.keys())


# --- HDF5 section gating ---------------------------------------------------

def test_h5_sections_restrict_groups(tmp_path):
    import pandas as pd
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        primary_df=pd.DataFrame({"T": [1.0, 2.0]}),
        baseline_df=pd.DataFrame({"P": [3.0]}),
        raw_metadata={"start": {"scan_id": 7}},
        sections={"primary", "metadata"},
    )
    groups = _groups(path)
    assert "primary" in groups
    assert "metadata" in groups
    assert "baseline" not in groups  # not in selected sections


def test_h5_qchi_excluded_by_default_section(tmp_path):
    import h5py
    result = _fake_transmission_result()
    path = tmp_path / "r.h5"
    # processed_iq on, processed_qchi OFF — the heavy q-χ map must be absent.
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path,
        sections={"processed_iq"},
    )
    with h5py.File(path, "r") as f:
        assert "transmission/merged_iq" in f
        assert "transmission/merged_qchi" not in f

    path2 = tmp_path / "r2.h5"
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path2,
        sections={"processed_iq", "processed_qchi"},
    )
    with h5py.File(path2, "r") as f:
        assert "transmission/merged_qchi" in f


def test_h5_peakfit_section(tmp_path):
    import h5py
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        sections={"peakfit"}, peak_fits=[_peak_fit()],
    )
    with h5py.File(path, "r") as f:
        assert "peakfit" in f
        sub = list(f["peakfit"].keys())
        assert len(sub) == 1
        # Group key uses the slug-prefixed form so an external script can walk
        # /peakfit/ entries without consulting attrs.
        assert sub[0] == "peak_p1_q2.275"
        g = f["peakfit"][sub[0]]
        assert "area" in g and "success" in g
        assert g.attrs["name"] == "p1"
        assert float(g.attrs["q_min"]) == pytest.approx(2.1)


def test_h5_sections_none_writes_all(tmp_path):
    import pandas as pd, h5py
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        primary_df=pd.DataFrame({"T": [1.0]}),
        baseline_df=pd.DataFrame({"P": [2.0]}),
        raw_metadata={"start": {"scan_id": 1}},
        sections=None,  # backward-compat: everything
    )
    with h5py.File(path, "r") as f:
        assert {"primary", "baseline", "metadata"} <= set(f.keys())


# --- Peak-result PNGs ------------------------------------------------------

def test_save_peak_pngs_one_per_peak_with_qrange(tmp_path):
    """PNG names use the slug ``<name>_q<center>``: peak identity is fully
    captured in one filename component, and two peaks with the same name
    but different q-centres still produce distinct files."""
    peaks = [_peak_fit("p1", 2.1, 2.45), _peak_fit("p2", 3.0, 3.4)]
    paths = _save_peak_pngs(peaks, axis=None, param="area",
                            scan_dir=tmp_path, prefix="")
    names = sorted(p.name for p in paths)
    # q-centre = (q_min + q_max) / 2, formatted to 3 decimals.
    assert names == [
        "peak_p1_q2.275_area.png",
        "peak_p2_q3.200_area.png",
    ]
    assert all((tmp_path / n).stat().st_size > 0 for n in names)


def test_save_peak_pngs_dedupes_collisions(tmp_path):
    # Same name + q-range twice → second file gets a counter suffix.
    peaks = [_peak_fit("p1", 2.1, 2.45), _peak_fit("p1", 2.1, 2.45)]
    paths = _save_peak_pngs(peaks, axis=None, param="area",
                            scan_dir=tmp_path, prefix="")
    assert len(paths) == 2
    assert len({p.name for p in paths}) == 2


# --- export_scan end-to-end ------------------------------------------------

def test_export_scan_peaks_and_sections(tmp_path):
    import h5py
    out, files = export_scan(
        out_dir=tmp_path, uid="abcd1234efgh",
        primary_df=None, raw_metadata={"start": {"scan_id": 5}},
        formats={"h5", "png_peaks"},
        h5_sections={"metadata", "peakfit"},
        peak_fits=[_peak_fit()],
        peak_param="amplitude",
        subdir_template="", basename_template="",
    )
    assert "result.h5" in files
    assert any(f.startswith("peak_p1_q2.275_amplitude") for f in files)
    with h5py.File(out / "result.h5", "r") as f:
        assert "peakfit" in f and "metadata" in f


def test_export_scan_no_peaks_skips_png(tmp_path):
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=None,
        subdir_template="", basename_template="",
    )
    assert not any("peak_" in f for f in files)


# --- Composite (RGB-additive) PNG export ----------------------------------

def _comp_config_active(channels):
    return {"active": True, "channels": channels, "pct_lo": 2.0, "pct_hi": 99.0}


def test_export_scan_composite_active_writes_one_png(tmp_path):
    """When the composite config is active, ``png_peaks`` produces a single
    ``peak_composite.png`` instead of per-peak PNGs."""
    peaks = [_peak_fit("p1", 2.1, 2.45), _peak_fit("p2", 3.0, 3.4)]
    from smi_browser.models.peakfit import peak_slug
    channels = [
        {"include": True, "label": "p1", "color": "#ff0000", "gain": 1.0,
         "log": False, "kind": "peak", "source": peak_slug(peaks[0])},
        {"include": True, "label": "p2", "color": "#00ff00", "gain": 1.0,
         "log": False, "kind": "peak", "source": peak_slug(peaks[1])},
    ]
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=peaks,
        peak_composite_config=_comp_config_active(channels),
        subdir_template="", basename_template="",
    )
    assert "peak_composite.png" in files
    # Per-peak PNGs are *not* written when the composite is active.
    assert not any(f.startswith("peak_p1_q") for f in files)
    assert not any(f.startswith("peak_p2_q") for f in files)
    assert (tmp_path / "peak_composite.png").stat().st_size > 0


def test_export_scan_composite_inactive_keeps_per_peak(tmp_path):
    """``active=False`` falls back to the per-peak PNG path."""
    peaks = [_peak_fit("p1", 2.1, 2.45)]
    cfg = {"active": False, "channels": [], "pct_lo": 2.0, "pct_hi": 99.0}
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=peaks,
        peak_composite_config=cfg,
        subdir_template="", basename_template="",
    )
    assert any(f.startswith("peak_p1_q2.275_") for f in files)
    assert "peak_composite.png" not in files


def test_export_scan_composite_with_primary_channel(tmp_path):
    """A ``kind='primary'`` channel uses values from ``primary_scalars``."""
    peaks = [_peak_fit("p1", 2.1, 2.45)]
    from smi_browser.models.peakfit import peak_slug
    n = peaks[0]["results"]["area"].size
    channels = [
        {"include": True, "label": "p1", "color": "#ff0000", "gain": 1.0,
         "log": False, "kind": "peak", "source": peak_slug(peaks[0])},
        {"include": True, "label": "T", "color": "#0000ff", "gain": 1.0,
         "log": False, "kind": "primary", "source": "primary:temperature"},
    ]
    primary_scalars = {"temperature": np.linspace(20.0, 100.0, n)}
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=peaks,
        peak_composite_config=_comp_config_active(channels),
        primary_scalars=primary_scalars,
        subdir_template="", basename_template="",
    )
    assert "peak_composite.png" in files
    assert (tmp_path / "peak_composite.png").stat().st_size > 0


def test_export_scan_composite_excluded_channel_dropped(tmp_path):
    """Channels with ``include=False`` are dropped before rendering."""
    peaks = [_peak_fit("p1", 2.1, 2.45), _peak_fit("p2", 3.0, 3.4)]
    from smi_browser.models.peakfit import peak_slug
    channels = [
        {"include": True, "label": "p1", "color": "#ff0000", "gain": 1.0,
         "log": False, "kind": "peak", "source": peak_slug(peaks[0])},
        {"include": False, "label": "p2", "color": "#00ff00", "gain": 1.0,
         "log": False, "kind": "peak", "source": peak_slug(peaks[1])},
    ]
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=peaks,
        peak_composite_config=_comp_config_active(channels),
        subdir_template="", basename_template="",
    )
    # Single PNG written (one active channel still composes).
    assert "peak_composite.png" in files


def test_export_scan_composite_no_resolvable_channels_writes_nothing(tmp_path):
    """When *all* channels are unresolvable (unknown peak slugs, no primary
    data), the composite PNG is silently skipped."""
    peaks = [_peak_fit("p1", 2.1, 2.45)]
    channels = [
        {"include": True, "label": "ghost", "color": "#ffffff", "gain": 1.0,
         "log": False, "kind": "peak", "source": "no_such_slug_q9.999"},
    ]
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=peaks,
        peak_composite_config=_comp_config_active(channels),
        subdir_template="", basename_template="",
    )
    assert "peak_composite.png" not in files
    # And no per-peak PNGs either, because the composite branch was taken.
    assert not any(f.startswith("peak_p1_q") for f in files)


def test_resolve_composite_channels_inactive_returns_empty():
    from smi_browser.export import _resolve_composite_channels
    peaks = [_peak_fit("p1", 2.1, 2.45)]
    assert _resolve_composite_channels({"active": False, "channels": []},
                                       peaks, None) == []
    assert _resolve_composite_channels(None, peaks, None) == []


# --- Derived peak-area columns in /primary --------------------------------

def test_h5_primary_includes_derived_peak_area_columns(tmp_path):
    """Per-peak ``area`` arrays are written as ``peak_<slug>_area`` datasets
    inside ``/primary`` with ``derived=True`` and q-range attrs."""
    import h5py
    import pandas as pd
    n = 6
    primary = pd.DataFrame({
        "temperature": np.linspace(20.0, 100.0, n),
        "scan_id": np.arange(n),
    })
    peaks = [_peak_fit("p1", 2.1, 2.45), _peak_fit("p2", 3.0, 3.4)]
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        primary_df=primary,
        sections={"primary"},
        peak_fits=peaks,
    )
    with h5py.File(path, "r") as f:
        assert "primary" in f
        # Raw columns preserved.
        assert "temperature" in f["primary"]
        assert "scan_id" in f["primary"]
        # Derived peak columns added with the slug naming.
        assert "peak_p1_q2.275_area" in f["primary"]
        assert "peak_p2_q3.200_area" in f["primary"]
        ds = f["primary"]["peak_p1_q2.275_area"]
        assert ds.attrs["derived"] == np.True_ or bool(ds.attrs["derived"])
        assert ds.attrs["peak_name"].decode() == "p1" \
            if isinstance(ds.attrs["peak_name"], bytes) \
            else ds.attrs["peak_name"] == "p1"
        assert float(ds.attrs["q_min"]) == pytest.approx(2.1)
        assert float(ds.attrs["q_max"]) == pytest.approx(2.45)
        assert ds.shape == (n,)


def test_h5_primary_derived_columns_skipped_when_section_off(tmp_path):
    """No derived peak columns are written when the ``primary`` section is off."""
    import h5py
    import pandas as pd
    primary = pd.DataFrame({"temperature": np.linspace(20.0, 100.0, 6)})
    peaks = [_peak_fit("p1", 2.1, 2.45)]
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        primary_df=primary,
        sections={"metadata"},  # primary section disabled
        peak_fits=peaks,
    )
    with h5py.File(path, "r") as f:
        assert "primary" not in f


def test_h5_primary_derived_columns_length_mismatch_skipped(tmp_path):
    """A peak whose ``area`` length doesn't match primary rows is skipped."""
    import h5py
    import pandas as pd
    primary = pd.DataFrame({"temperature": np.linspace(20.0, 100.0, 6)})
    bad_peak = _peak_fit("p1", 2.1, 2.45)
    bad_peak["results"] = {"area": np.zeros(3)}  # wrong length
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        primary_df=primary,
        sections={"primary"},
        peak_fits=[bad_peak],
    )
    with h5py.File(path, "r") as f:
        assert "temperature" in f["primary"]
        assert not any(k.startswith("peak_") for k in f["primary"].keys())


def test_h5_primary_derived_columns_no_clobber(tmp_path):
    """A user-supplied primary column with the derived name is *not* overwritten."""
    import h5py
    import pandas as pd
    n = 6
    sentinel = np.full(n, 999.0)
    primary = pd.DataFrame({
        "temperature": np.linspace(20.0, 100.0, n),
        "peak_p1_q2.275_area": sentinel,  # user-supplied column with same name
    })
    peaks = [_peak_fit("p1", 2.1, 2.45)]
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, None, [], None, None, None, "", "", {}, path,
        primary_df=primary,
        sections={"primary"},
        peak_fits=peaks,
    )
    with h5py.File(path, "r") as f:
        ds = f["primary"]["peak_p1_q2.275_area"]
        # User data preserved; no `derived` attr because we kept the original.
        assert np.allclose(ds[:], sentinel)
        assert "derived" not in ds.attrs


# --- Realistic-shape regression tests for HDF5 export gaps ----------------

def test_h5_merged_qchi_writes_all_data_vars(tmp_path):
    """``merged_qchi`` writes all 6 vars (intensity, counts, saxs/waxs split)."""
    import h5py
    result = _full_transmission_result()
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path,
        sections={"processed_qchi"},
    )
    expected = {
        "intensity", "counts",
        "saxs_intensity", "saxs_counts",
        "waxs_intensity", "waxs_counts",
        "q", "chi",
    }
    with h5py.File(path, "r") as f:
        assert expected <= set(f["transmission/merged_qchi"].keys())


def test_h5_per_detector_native_iq_present(tmp_path):
    """Per-detector native-grid I(q) (saxs_iq, waxs_iq) is written."""
    import h5py
    result = _full_transmission_result()
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path,
        sections={"processed_iq"},
    )
    with h5py.File(path, "r") as f:
        assert "transmission/saxs_iq/I" in f
        assert "transmission/saxs_iq/counts" in f
        assert "transmission/saxs_iq/q" in f
        assert "transmission/waxs_iq/I" in f
        # Native grid lengths differ from merged grid (12, 14 vs 10).
        assert f["transmission/saxs_iq/q"].shape == (12,)
        assert f["transmission/waxs_iq/q"].shape == (14,)


def test_h5_per_frame_qchi_writes_counts(tmp_path):
    """Per-frame q-chi stack now writes both intensity AND counts."""
    import h5py
    result = _full_transmission_result(n_frames=4)
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path,
        sections={"processed_qchi"},
    )
    with h5py.File(path, "r") as f:
        for det in ("saxs", "waxs"):
            g = f[f"transmission/per_frame_qchi/{det}"]
            assert "intensity" in g
            assert "counts" in g
            assert g["intensity"].shape == (4, 10, 8)
            assert g["counts"].shape == (4, 10, 8)
            # Counts should reflect the fixture value, not silently zero.
            assert np.all(g["counts"][...] == 25)


def test_h5_transmission_provenance_attrs(tmp_path):
    """incident_angle_deg + scan_info land as attrs on /transmission."""
    import h5py, json
    result = _full_transmission_result()
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path,
        sections={"processed_iq"},
    )
    with h5py.File(path, "r") as f:
        attrs = f["transmission"].attrs
        assert float(attrs["incident_angle_deg"]) == pytest.approx(0.12)
        info = json.loads(attrs["scan_info"])
        assert info["sample_name"] == "demo"
        assert info["n_frames"] == 4


def test_h5_transmission_resolved_reduction_parameters(tmp_path):
    import h5py, json
    result = _full_transmission_result()
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        result, None, [], None, None, None, "", "", {}, path,
        sections={"processed_iq", "processed_qchi"},
    )
    with h5py.File(path, "r") as f:
        attrs = f["transmission"].attrs
        assert attrs["smi_reduction_parameters_schema"] == (
            "smi_tiled.resolved_reduction_parameters.v1"
        )
        decoded = json.loads(attrs["smi_reduction_parameters"])
        assert decoded["saxs"]["sample_detector_distance_mm"] == pytest.approx(5020.0)
        assert float(attrs["saxs_sdd_mm"]) == pytest.approx(5020.0)
        assert float(attrs["waxs_beam_center_col_px"]) == pytest.approx(314.5)
        assert float(f["transmission/merged_iq"].attrs["saxs_sdd_mm"]) == pytest.approx(5020.0)
        assert float(f["transmission/merged_qchi"].attrs["waxs_sdd_mm"]) == pytest.approx(278.0)


def test_h5_gi_per_frame_qxy_qz_written(tmp_path):
    """GI per-frame stack lands at gi/per_frame_qxy_qz with qxy/qz coords."""
    import h5py
    gi = _full_gi_result(n_frames=3)
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, gi, [], None, None, None, "", "", {}, path,
        sections={"processed_qchi"},
    )
    with h5py.File(path, "r") as f:
        g = f["gi/per_frame_qxy_qz"]
        assert "intensity" in g
        assert "qxy" in g and "qz" in g
        # GI stack has only intensity (no counts) per smi-tiled.
        assert "counts" not in g
        assert g["intensity"].shape == (3, 6, 7)
        assert g.attrs["n_frames"] == 3
        # Legacy gi/frames/frame_NNNN must be gone.
        assert "gi/frames" not in f


def test_h5_gi_provenance(tmp_path):
    """scan_motor as attr, scan_motor_values as dataset on /gi."""
    import h5py
    gi = _full_gi_result(n_frames=3)
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        None, gi, [], None, None, None, "", "", {}, path,
        sections={"processed_iq"},
    )
    with h5py.File(path, "r") as f:
        assert f["gi"].attrs["scan_motor"] == "piezo_th"
        assert f["gi/scan_motor_values"].shape == (3,)
        assert np.allclose(
            f["gi/scan_motor_values"][...], np.linspace(0.1, 0.3, 3),
        )


def test_h5_qchi_gate_off_skips_per_frame(tmp_path):
    """processed_qchi off → no per_frame_qchi (transmission) or per_frame_qxy_qz (GI)."""
    import h5py
    result = _full_transmission_result()
    gi = _full_gi_result()
    path = tmp_path / "r.h5"
    _save_dataset_h5(
        result, gi, [], None, None, None, "", "", {}, path,
        sections={"processed_iq"},
    )
    with h5py.File(path, "r") as f:
        assert "transmission/per_frame_qchi" not in f
        assert "gi/per_frame_qxy_qz" not in f
        # But native I(q), summed, etc. should still be present.
        assert "transmission/saxs_iq" in f
        assert "gi/summed" in f
