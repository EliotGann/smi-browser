"""Tests for the per-scan disk cache (`smi_browser.cache`)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from smi_browser import cache as cache_mod
from smi_browser.cache import (
    ScanCache,
    get_or_fetch_image_stack,
    get_or_fetch_scalars,
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
# Path resolution
# ---------------------------------------------------------------------------

def test_cache_path_sanitises_uid(isolated_cache_dir):
    p = cache_mod.cache_path("weird/uid:with*chars")
    assert p.parent == Path(isolated_cache_dir)
    assert "/" not in p.name and ":" not in p.name and "*" not in p.name


def test_cache_path_rejects_empty():
    with pytest.raises(ValueError):
        cache_mod.cache_path("")
