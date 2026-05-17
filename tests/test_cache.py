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


def test_reduction_overwrites(isolated_cache_dir):
    c = ScanCache("uid-red2")
    c.write_reduction({"x": np.array([1.0, 2.0])}, {"n_q": 10})
    c.write_reduction({"x": np.array([9.0])}, {"n_q": 99})
    blob = c.read_reduction()
    np.testing.assert_allclose(blob["arrays"]["x"], [9.0])
    assert int(blob["params"]["n_q"]) == 99


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
