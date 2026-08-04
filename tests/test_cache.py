"""Tests for the per-scan disk cache (`smi_browser.cache`)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from smi_browser import cache as cache_mod
from smi_browser.cache import (
    ScanCache,
    clear_disk_cache,
    disk_cache_info,
    get_or_fetch_image_stack,
    get_or_fetch_scalars,
    peak_defs_path,
    write_peak_defs,
)


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect the cache root to a fresh tmp dir for every test."""
    monkeypatch.setenv("SMI_BROWSER_CACHE_DIR", str(tmp_path))
    # Reset any cached locks held against the previous root.
    cache_mod._LOCK_TABLE.clear()
    yield tmp_path


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

def test_scalars_roundtrip(isolated_cache_dir):
    c = ScanCache("uid-abc-123")
    assert c.read_scalars("primary") is None

    data = {
        "sample_temperature": np.array([300.1, 300.2, 300.3]),
        "waxs_arc": np.array([0.0, 30.0, 60.0]),
    }
    c.write_scalars("primary", data)
    assert c.path.exists()

    out = c.read_scalars("primary")
    assert out is not None
    assert set(out.keys()) == set(data.keys())
    np.testing.assert_allclose(out["sample_temperature"], data["sample_temperature"])
    np.testing.assert_allclose(out["waxs_arc"], data["waxs_arc"])


def test_scalars_overwrite(isolated_cache_dir):
    c = ScanCache("uid-overwrite")
    c.write_scalars("primary", {"a": np.array([1.0, 2.0])})
    c.write_scalars("primary", {"a": np.array([9.0, 8.0, 7.0])})
    out = c.read_scalars("primary")
    np.testing.assert_allclose(out["a"], [9.0, 8.0, 7.0])


def test_scalars_object_dtype_coerced(isolated_cache_dir):
    c = ScanCache("uid-obj")
    # object dtype that can't be cast to float → stored as strings
    c.write_scalars("primary", {"label": np.array(["a", "b", "c"], dtype=object)})
    out = c.read_scalars("primary")
    assert out is not None
    assert list(out["label"].astype(str)) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def test_image_stack_roundtrip(isolated_cache_dir):
    c = ScanCache("uid-img")
    stack = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    c.write_image_stack("pil1M_image", stack)
    assert c.has_image_field("pil1M_image")
    out = c.read_image_stack("pil1M_image")
    np.testing.assert_array_equal(out, stack)


def test_image_stack_per_stream_no_collision(isolated_cache_dir):
    """Same detector field in two streams must not clobber each other."""
    c = ScanCache("uid-arc")
    waxs_arc0 = np.zeros((2, 3, 3), dtype=np.float32)
    waxs_arc20 = np.ones((2, 3, 3), dtype=np.float32)
    # 'primary' uses the legacy images/<field> path; arc20 is namespaced.
    c.write_image_stack("pil900KW_image", waxs_arc0, stream="primary")
    c.write_image_stack("pil900KW_image", waxs_arc20, stream="arc20")

    np.testing.assert_array_equal(
        c.read_image_stack("pil900KW_image", stream="primary"), waxs_arc0)
    np.testing.assert_array_equal(
        c.read_image_stack("pil900KW_image", stream="arc20"), waxs_arc20)
    assert c.has_image_field("pil900KW_image", stream="primary")
    assert c.has_image_field("pil900KW_image", stream="arc20")
    # A stream we never wrote is absent.
    assert c.read_image_stack("pil900KW_image", stream="arc0") is None
    assert not c.has_image_field("pil900KW_image", stream="arc0")


def test_image_primary_uses_legacy_path(isolated_cache_dir):
    """primary writes must remain readable at the bare images/<field> path
    so caches written before per-stream support still load."""
    import h5py
    c = ScanCache("uid-legacy")
    stack = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    c.write_image_stack("pil2M_image", stack, stream="primary")
    with h5py.File(c.path, "r") as f:
        assert "images/pil2M_image" in f
        assert "images/primary" not in f  # no extra nesting for primary



# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

def test_reduction_roundtrip_with_params(isolated_cache_dir):
    c = ScanCache("uid-red")
    arrays = {
        "q": np.linspace(0.01, 5.0, 50),
        "I": np.random.rand(50),
    }
    params = {"n_q": 50, "geometry": "transmission", "extras": None}
    c.write_reduction(arrays, params)
    blob = c.read_reduction()
    assert blob is not None
    np.testing.assert_allclose(blob["arrays"]["q"], arrays["q"])
    np.testing.assert_allclose(blob["arrays"]["I"], arrays["I"])
    assert int(blob["params"]["n_q"]) == 50
    assert str(blob["params"]["geometry"]) == "transmission"


def test_reduction_roundtrip_with_resolved_parameters(isolated_cache_dir):
    import json

    c = ScanCache("uid-red-params")
    resolved = {
        "schema": "smi_tiled.resolved_reduction_parameters.v1",
        "saxs": {
            "sample_detector_distance_mm": 5020.0,
            "beam_center_row_px": 1165.2,
            "beam_center_col_px": 746.4,
        },
        "waxs": {
            "sample_detector_distance_mm": 278.0,
            "beam_center_row_px": 217.0,
            "beam_center_col_px": 314.5,
        },
    }
    c.write_reduction(
        {"q": np.array([1.0]), "I": np.array([2.0])},
        {"geometry": "transmission", "smi_reduction_parameters": resolved},
    )
    blob = c.read_reduction()

    params = blob["params"]
    decoded = json.loads(params["smi_reduction_parameters"])
    assert decoded["saxs"]["sample_detector_distance_mm"] == 5020.0


def test_lazy_per_frame_fetch_only_loads_requested(isolated_cache_dir):
    from smi_browser.cache import get_or_fetch_image_frame

    calls = []

    def fetch_one(i):
        calls.append(i)
        return np.full((3, 4), i, dtype="int32")

    uid, field, n = "lazy", "det_image", 100
    f = get_or_fetch_image_frame(uid, field, 42, fetch_one_fn=fetch_one, n_frames=n)
    assert f[0, 0] == 42 and calls == [42]

    # Re-view is served from disk — no refetch.
    calls.clear()
    f = get_or_fetch_image_frame(uid, field, 42, fetch_one_fn=fetch_one, n_frames=n)
    assert f[0, 0] == 42 and calls == []

    # A different (unfilled) frame fetches only that frame.
    calls.clear()
    f = get_or_fetch_image_frame(uid, field, 88, fetch_one_fn=fetch_one, n_frames=n)
    assert f[0, 0] == 88 and calls == [88]

    # Dataset is pre-sized for all frames but only the two viewed are filled.
    import h5py
    with h5py.File(ScanCache(uid).path, "r") as h:
        assert h[f"images/{field}"].shape == (n, 3, 4)
        assert int(h[f"images_filled/{field}"][...].sum()) == 2


def test_lazy_per_frame_is_stream_isolated(isolated_cache_dir):
    """Per-frame caching keeps streams in separate datasets + fill masks."""
    from smi_browser.cache import get_or_fetch_image_frame
    import h5py

    uid, field, n = "lazy-arc", "pil900KW_image", 10

    a = get_or_fetch_image_frame(
        uid, field, 3, fetch_one_fn=lambda i: np.full((2, 2), 10 + i, "int32"),
        n_frames=n, stream="arc0")
    b = get_or_fetch_image_frame(
        uid, field, 3, fetch_one_fn=lambda i: np.full((2, 2), 90 + i, "int32"),
        n_frames=n, stream="arc20")
    assert a[0, 0] == 13 and b[0, 0] == 93  # no cross-stream clobber

    # Re-view of arc0 returns arc0's value from disk (not arc20's).
    a2 = get_or_fetch_image_frame(
        uid, field, 3, fetch_one_fn=lambda i: (_ for _ in ()).throw(
            AssertionError("should not refetch")),
        n_frames=n, stream="arc0")
    assert a2[0, 0] == 13

    with h5py.File(ScanCache(uid).path, "r") as h:
        # arc0/arc20 are both namespaced (neither is 'primary'), so the bare
        # legacy path is NOT used.
        assert "images/pil900KW_image" not in h
        assert "images/arc0/pil900KW_image" in h
        assert "images/arc20/pil900KW_image" in h
        assert "images_filled/arc0/pil900KW_image" in h
        assert "images_filled/arc20/pil900KW_image" in h



def test_lazy_unknown_length_serves_uncached(isolated_cache_dir):
    from smi_browser.cache import get_or_fetch_image_frame
    calls = []

    def fetch_one(i):
        calls.append(i)
        return np.full((2, 2), i, dtype="int32")

    # n_frames unknown → single frame served, nothing written to disk.
    f = get_or_fetch_image_frame("u2", "det", 5, fetch_one_fn=fetch_one, n_frames=0)
    assert f[0, 0] == 5 and calls == [5]
    assert not ScanCache("u2").has_image_field("det")


def test_legacy_full_stack_still_served(isolated_cache_dir):
    """A full-stack write (no fill mask) serves every frame as filled."""
    from smi_browser.cache import get_or_fetch_image_frame
    uid, field = "legacy", "det_image"
    stack = np.arange(5 * 2 * 2, dtype="int32").reshape(5, 2, 2)
    ScanCache(uid).write_image_stack(field, stack)
    # No fetch_one_fn needed — frame comes straight from the cached stack.
    f = get_or_fetch_image_frame(uid, field, 3, fetch_one_fn=lambda i: None, n_frames=5)
    np.testing.assert_array_equal(f, stack[3])


def test_read_reduction_datasets_subset(isolated_cache_dir):
    c = ScanCache("uid-subset")
    c.write_reduction({
        "pf_iq_I": np.zeros((4, 6)),
        "pf_iq_q": np.arange(6.0),
        "qchi_intensity": np.ones((6, 3)),  # large array we should NOT load
    }, {"n_q": 6})
    got = c.read_reduction_datasets(["pf_iq_I", "pf_iq_q"])
    assert set(got) == {"pf_iq_I", "pf_iq_q"}
    assert got["pf_iq_I"].shape == (4, 6)
    # Missing keys are simply absent; no error.
    assert c.read_reduction_datasets(["nope"]) == {}


def test_read_reduction_datasets_no_reduction(isolated_cache_dir):
    assert ScanCache("uid-empty").read_reduction_datasets(["pf_iq_I"]) is None


def test_reduction_overwrites(isolated_cache_dir):
    c = ScanCache("uid-red2")
    c.write_reduction({"x": np.array([1.0, 2.0])}, {"n_q": 10})
    c.write_reduction({"x": np.array([9.0])}, {"n_q": 99})
    blob = c.read_reduction()
    np.testing.assert_allclose(blob["arrays"]["x"], [9.0])
    assert int(blob["params"]["n_q"]) == 99


# ---------------------------------------------------------------------------
# Peak fits + peak definitions
# ---------------------------------------------------------------------------

def _peakfit_arrays(n=4):
    return {
        "amplitude": np.arange(n, dtype=float),
        "center": np.full(n, 5.0),
        "fwhm": np.full(n, 0.3),
        "area": np.arange(n, dtype=float) * 2.0,
        "success": np.array([True, False, True, True][:n]),
    }


def test_peakfit_roundtrip(isolated_cache_dir):
    c = ScanCache("uid-pf")
    key = (3.0, 7.0, "gaussian", "linear", "linked", 2.0)
    c.write_peakfit(key, _peakfit_arrays(),
                    attrs={"name": "p1", "model": "gaussian"})
    idx = c.read_peakfit_index()
    assert len(idx) == 1
    got_key, res = idx[0]
    assert tuple(got_key) == key
    np.testing.assert_allclose(res["amplitude"], [0.0, 1.0, 2.0, 3.0])
    assert res["success"].dtype == bool
    assert list(res["success"]) == [True, False, True, True]


def test_peakfit_overwrites_same_key(isolated_cache_dir):
    c = ScanCache("uid-pf2")
    key = (3.0, 7.0, "gaussian", "none", "independent", 2.0)
    c.write_peakfit(key, {"amplitude": np.array([1.0])})
    c.write_peakfit(key, {"amplitude": np.array([9.0])})
    idx = c.read_peakfit_index()
    assert len(idx) == 1
    np.testing.assert_allclose(idx[0][1]["amplitude"], [9.0])


def test_reduction_clears_peakfit(isolated_cache_dir):
    c = ScanCache("uid-pf3")
    c.write_reduction({"pf_iq_I": np.zeros((2, 3))}, {})
    c.write_peakfit((1.0, 2.0, "gaussian", "none", "linked", 2.0),
                    {"amplitude": np.array([1.0, 2.0])})
    assert len(c.read_peakfit_index()) == 1
    # Re-processing overwrites pf_iq → stale fits must be dropped.
    c.write_reduction({"pf_iq_I": np.ones((2, 3))}, {})
    assert c.read_peakfit_index() == []


def test_clear_peakfit(isolated_cache_dir):
    c = ScanCache("uid-pf4")
    c.write_peakfit((1.0, 2.0, "gaussian", "none", "linked", 2.0),
                    {"amplitude": np.array([1.0])})
    c.clear_peakfit()
    assert c.read_peakfit_index() == []


def _make_peak_fits_dataset():
    """Synthesize an apply_peak_fits-shaped xr.Dataset (peak, frame)."""
    import xarray as xr

    n_peaks, n_frames = 2, 5
    return xr.Dataset(
        {
            "amplitude": (("peak", "frame"), np.arange(n_peaks * n_frames,
                                                       dtype=float).reshape(n_peaks, n_frames)),
            "center":    (("peak", "frame"), np.full((n_peaks, n_frames), 5.0)),
            "fwhm":      (("peak", "frame"), np.full((n_peaks, n_frames), 0.3)),
            "area":      (("peak", "frame"), np.full((n_peaks, n_frames), 1.5)),
            "success":   (("peak", "frame"), np.ones((n_peaks, n_frames), dtype=bool)),
        },
        coords={
            "peak": np.array(["alpha", "beta"], dtype=object),
            "frame": np.arange(n_frames),
        },
        attrs={"peaks": [
            {"name": "alpha", "q_min": 1.1, "q_max": 1.4, "model": "gaussian",
             "baseline": "linear", "link": "linked", "bg_factor": 2.0},
            {"name": "beta",  "q_min": 2.1, "q_max": 2.4, "model": "lorentzian",
             "baseline": "none",   "link": "independent", "bg_factor": 3.0},
        ]},
    )


def test_write_peakfit_dataset_splits_per_peak(isolated_cache_dir):
    """Smi-tiled apply_peak_fits returns a (peak, frame) dataset; verify the
    cache splits it into per-peak entries that round-trip via the existing
    read_peakfit_full reader (i.e., the format export_scan consumes).
    """
    c = ScanCache("uid-pf-ds")
    n = c.write_peakfit_dataset(_make_peak_fits_dataset())
    assert n == 2

    entries = c.read_peakfit_full()
    assert len(entries) == 2
    # Sort by attr name so test is order-independent.
    by_name = {e["attrs"]["name"]: e for e in entries}
    assert set(by_name) == {"alpha", "beta"}

    a = by_name["alpha"]
    np.testing.assert_allclose(a["arrays"]["amplitude"], [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(a["arrays"]["center"], 5.0)
    assert a["attrs"]["model"] == "gaussian"
    assert a["attrs"]["q_min"] == 1.1
    assert a["arrays"]["success"].dtype == bool

    b = by_name["beta"]
    np.testing.assert_allclose(b["arrays"]["amplitude"], [5.0, 6.0, 7.0, 8.0, 9.0])
    assert b["attrs"]["model"] == "lorentzian"
    assert b["attrs"]["link"] == "independent"
    assert b["attrs"]["bg_factor"] == 3.0


def test_write_peakfit_dataset_handles_none_and_empty(isolated_cache_dir):
    """Defensive: ScanCache must be a no-op when there are no peaks to write."""
    c = ScanCache("uid-pf-empty")
    assert c.write_peakfit_dataset(None) == 0
    assert c.read_peakfit_index() == []

    import xarray as xr
    empty = xr.Dataset(
        {"amplitude": (("peak", "frame"), np.zeros((0, 0)))},
        coords={"peak": np.array([], dtype=object), "frame": np.arange(0)},
    )
    assert c.write_peakfit_dataset(empty) == 0
    assert c.read_peakfit_index() == []


def test_write_peakfit_dataset_invalidated_by_reduction(isolated_cache_dir):
    """write_reduction wipes /peakfit; the dataset writer must run after it."""
    c = ScanCache("uid-pf-inv")
    c.write_peakfit_dataset(_make_peak_fits_dataset())
    assert len(c.read_peakfit_index()) == 2
    # Re-reduction nukes peak fits (they're keyed off the prior pf_iq).
    c.write_reduction({"pf_iq_I": np.zeros((2, 3))}, {})
    assert c.read_peakfit_index() == []


def test_peak_defs_roundtrip(isolated_cache_dir):
    from smi_browser.cache import read_peak_defs, write_peak_defs
    assert read_peak_defs() == []
    defs = [{"name": "p1", "q_min": 2.1, "q_max": 2.5, "model": "gaussian",
             "baseline": "linear", "link": "linked", "bg": 2.0}]
    write_peak_defs(defs)
    got = read_peak_defs()
    assert got == defs


# ---------------------------------------------------------------------------
# Fetch wrappers
# ---------------------------------------------------------------------------

def test_get_or_fetch_scalars_caches(isolated_cache_dir):
    calls = {"n": 0}
    data = {"T": np.array([1.0, 2.0, 3.0])}

    def fetch():
        calls["n"] += 1
        return data

    out1 = get_or_fetch_scalars("uid-fetch", "primary", fetch)
    out2 = get_or_fetch_scalars("uid-fetch", "primary", fetch)
    np.testing.assert_allclose(out1["T"], data["T"])
    np.testing.assert_allclose(out2["T"], data["T"])
    assert calls["n"] == 1, "second call should hit cache"


def test_get_or_fetch_scalars_handles_empty(isolated_cache_dir):
    """An empty fetch result must not write a sentinel that masks future fetches."""
    calls = {"n": 0}
    payload = {"T": np.array([1.0])}

    def fetch():
        calls["n"] += 1
        return payload if calls["n"] > 1 else {}

    out1 = get_or_fetch_scalars("uid-empty", "primary", fetch)
    assert out1 == {}
    out2 = get_or_fetch_scalars("uid-empty", "primary", fetch)
    np.testing.assert_allclose(out2["T"], payload["T"])
    assert calls["n"] == 2


def test_get_or_fetch_image_stack_caches(isolated_cache_dir):
    calls = {"n": 0}
    stack = np.ones((3, 4, 5), dtype=np.float32)

    def fetch():
        calls["n"] += 1
        return stack

    out1 = get_or_fetch_image_stack("uid-img-fetch", "pil1M_image", fetch)
    out2 = get_or_fetch_image_stack("uid-img-fetch", "pil1M_image", fetch)
    np.testing.assert_array_equal(out1, stack)
    np.testing.assert_array_equal(out2, stack)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def test_eviction_drops_oldest_when_over_cap(isolated_cache_dir, monkeypatch):
    rng = np.random.default_rng(0)

    # Write the first scan, measure its on-disk size, then set the cap
    # just above that so a second equally-sized scan is guaranteed to
    # exceed the budget (independent of gzip's actual compression ratio).
    c1 = ScanCache("uid-old")
    c1.write_scalars("primary", {"a": rng.random(40_000)})
    old_path = c1.path
    assert old_path.exists()
    size1 = old_path.stat().st_size
    cap_gb = (size1 * 1.5) / (1024 ** 3)
    monkeypatch.setenv("SMI_BROWSER_CACHE_MAX_GB", f"{cap_gb:.12f}")

    # Backdate so c1 is the eviction victim.
    os.utime(old_path, (1000.0, 1000.0))

    c2 = ScanCache("uid-new")
    c2.write_scalars("primary", {"a": rng.random(40_000)})

    # Newer file must still exist; the older one should have been evicted.
    assert c2.path.exists()
    assert not old_path.exists(), "oldest cache file should be evicted"


# ---------------------------------------------------------------------------
# Disk cache management
# ---------------------------------------------------------------------------

def test_disk_cache_info_breaks_down_cache_entries(isolated_cache_dir):
    c = ScanCache("uid-info")
    c.write_scalars("primary", {"a": np.arange(5)})
    qchi = Path(isolated_cache_dir) / "uid-info_qchi"
    qchi.mkdir()
    (qchi / "chunk.bin").write_bytes(b"1234")
    write_peak_defs([{"name": "peak", "q_min": 1.0, "q_max": 2.0}])
    (Path(isolated_cache_dir) / "misc.tmp").write_bytes(b"xx")

    info = disk_cache_info()

    assert info["root"] == str(isolated_cache_dir)
    assert info["total_bytes"] >= c.path.stat().st_size + 6
    assert info["h5_files"] == 1
    assert info["h5_bytes"] == c.path.stat().st_size
    assert info["qchi_dirs"] == 1
    assert info["qchi_bytes"] == 4
    assert info["peak_defs_bytes"] == peak_defs_path().stat().st_size
    assert info["other_entries"] == 1
    assert info["other_bytes"] == 2


def test_clear_disk_cache_preserves_peak_defs_by_default(isolated_cache_dir):
    c = ScanCache("uid-clear")
    c.write_scalars("primary", {"a": np.arange(5)})
    qchi = Path(isolated_cache_dir) / "uid-clear_qchi"
    qchi.mkdir()
    (qchi / "chunk.bin").write_bytes(b"1234")
    write_peak_defs([{"name": "peak", "q_min": 1.0, "q_max": 2.0}])

    stats = clear_disk_cache()

    assert stats["errors"] == []
    assert stats["deleted_entries"] == 2
    assert stats["preserved_entries"] == 1
    assert not c.path.exists()
    assert not qchi.exists()
    assert peak_defs_path().exists()


def test_clear_disk_cache_can_delete_peak_defs(isolated_cache_dir):
    write_peak_defs([{"name": "peak", "q_min": 1.0, "q_max": 2.0}])

    stats = clear_disk_cache(include_peak_defs=True)

    assert stats["errors"] == []
    assert stats["deleted_entries"] == 1
    assert not peak_defs_path().exists()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_cache_path_sanitises_uid(isolated_cache_dir):
    p = cache_mod.cache_path("weird/uid:with*chars")
    assert p.parent == Path(isolated_cache_dir)
    assert "/" not in p.name and ":" not in p.name and "*" not in p.name


def test_cache_path_rejects_empty():
    with pytest.raises(ValueError):
        cache_mod.cache_path("")
