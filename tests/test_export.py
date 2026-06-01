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
    peaks = [_peak_fit("p1", 2.1, 2.45), _peak_fit("p2", 3.0, 3.4)]
    paths = _save_peak_pngs(peaks, axis=None, param="area",
                            scan_dir=tmp_path, prefix="")
    names = sorted(p.name for p in paths)
    assert names == [
        "peak_p1_q2.100-2.450_area.png",
        "peak_p2_q3.000-3.400_area.png",
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
    assert any(f.startswith("peak_p1_q2.100-2.450_amplitude") for f in files)
    with h5py.File(out / "result.h5", "r") as f:
        assert "peakfit" in f and "metadata" in f


def test_export_scan_no_peaks_skips_png(tmp_path):
    _, files = export_scan(
        out_dir=tmp_path, uid="u1",
        formats={"png_peaks"}, peak_fits=None,
        subdir_template="", basename_template="",
    )
    assert not any("peak_" in f for f in files)
