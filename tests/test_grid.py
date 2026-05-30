"""Tests for smi_browser.models.grid — empirical grid detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smi_browser.models.grid import (
    GriddedView,
    gridded_view,
    pick_default_axes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raster_xyz(nx: int, ny: int, *, snake: bool = False, z_fn=None):
    """Synthesise a raster scan: outer y, inner x.

    Returns ``(x, y, z)`` flattened arrays.  ``z_fn(xc, yc)`` defaults to
    a smooth ``xc * yc`` so missing-cell tests can detect filled vs not.
    """
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 2, ny)
    x_list, y_list, z_list = [], [], []
    z_fn = z_fn or (lambda xc, yc: xc * 10 + yc)
    for j, yc in enumerate(ys):
        order = xs if (j % 2 == 0 or not snake) else xs[::-1]
        for xc in order:
            x_list.append(xc)
            y_list.append(yc)
            z_list.append(z_fn(xc, yc))
    return np.asarray(x_list), np.asarray(y_list), np.asarray(z_list)


# ---------------------------------------------------------------------------
# gridded_view — happy paths
# ---------------------------------------------------------------------------

def test_gridded_view_raster_2d():
    x, y, z = _raster_xyz(21, 151)
    gv = gridded_view(x, y, z)
    assert isinstance(gv, GriddedView)
    assert gv.image.shape == (151, 21)
    assert gv.n_filled == 21 * 151
    assert gv.max_collision == 1
    assert gv.fill_fraction == 1.0
    # x is the inner axis (fast); centres should match np.linspace endpoints
    assert np.allclose(gv.x_centres[[0, -1]], [0.0, 1.0])
    assert np.allclose(gv.y_centres[[0, -1]], [0.0, 2.0])


def test_gridded_view_snake_unscrambles_via_position():
    """Snake scan: cells are still placed by their (x,y), not by scan index."""
    x, y, z = _raster_xyz(21, 151, snake=True)
    gv = gridded_view(x, y, z)
    # Cell (iy=10, ix=5) should hold f(xs[5], ys[10]) regardless of snake.
    xs = np.linspace(0, 1, 21)
    ys = np.linspace(0, 2, 151)
    assert gv.image[10, 5] == pytest.approx(xs[5] * 10 + ys[10])


def test_gridded_view_partial_grid_nan_filled():
    x, y, z = _raster_xyz(10, 8)
    # Drop the last 5 points (incomplete scan)
    x, y, z = x[:-5], y[:-5], z[:-5]
    gv = gridded_view(x, y, z)
    assert gv is not None
    assert gv.n_filled == len(x)
    assert gv.n_total == 10 * 8
    assert np.isnan(gv.image[-1, -1])
    assert gv.fill_fraction < 1.0


def test_gridded_view_handles_floating_jitter():
    """Motor positions repeating with sub-tolerance jitter should still cluster."""
    rng = np.random.default_rng(0)
    x_base = np.linspace(0, 1, 12)
    y_base = np.linspace(0, 2, 9)
    x_list, y_list, z_list = [], [], []
    for yc in y_base:
        for xc in x_base:
            x_list.append(xc + rng.normal(scale=1e-9))
            y_list.append(yc + rng.normal(scale=1e-9))
            z_list.append(xc * 10 + yc)
    gv = gridded_view(np.array(x_list), np.array(y_list), np.array(z_list))
    assert gv is not None
    assert gv.image.shape == (9, 12)
    assert gv.max_collision == 1


def test_gridded_view_collision_averaging():
    """Multiple data points per cell average together; max_collision reports it."""
    x = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    z = np.array([1.0, 3.0, 5.0, 10.0, 20.0, 30.0])
    gv = gridded_view(x, y, z, min_axis_size=2)
    assert gv is not None
    assert gv.image.shape == (2, 2)
    assert gv.image[0, 0] == pytest.approx(3.0)
    assert gv.image[1, 1] == pytest.approx(20.0)
    assert np.isnan(gv.image[0, 1])
    assert gv.max_collision == 3


def test_gridded_view_drops_nan_z_per_point():
    x, y, z = _raster_xyz(5, 5)
    z = z.astype(float)
    z[0] = np.nan
    gv = gridded_view(x, y, z)
    assert gv is not None
    # Cell that *only* had the NaN point should now be empty.
    assert np.isnan(gv.image[0, 0])
    assert gv.counts[0, 0] == 0
    # Neighbours unaffected.
    assert not np.isnan(gv.image[0, 1])


# ---------------------------------------------------------------------------
# gridded_view — rejections
# ---------------------------------------------------------------------------

def test_gridded_view_list_scan_returns_none():
    """A truly arbitrary list scan with no grid structure → None."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=200)
    y = rng.uniform(0, 1, size=200)
    z = rng.normal(size=200)
    gv = gridded_view(x, y, z)
    assert gv is None


def test_gridded_view_single_axis_returns_none():
    """1-D scan (Y constant) — degenerate grid."""
    x = np.linspace(0, 1, 50)
    y = np.zeros_like(x)
    z = np.sin(x)
    gv = gridded_view(x, y, z)
    assert gv is None


def test_gridded_view_empty_returns_none():
    assert gridded_view([], [], []) is None


def test_gridded_view_too_sparse_returns_none():
    """nx*ny ≫ N → should reject (not enough fill)."""
    # 5 points along the (x = y) diagonal of a hypothetical 5×5 grid:
    # nx=ny=5, but only 5 cells filled out of 25 → too sparse with default
    # max_sparseness=4.0 (25/5 = 5.0 > 4.0).
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = x + y
    gv = gridded_view(x, y, z)
    assert gv is None


def test_gridded_view_validates_shapes():
    with pytest.raises(ValueError):
        gridded_view([0, 1], [0], [0])


# ---------------------------------------------------------------------------
# pick_default_axes
# ---------------------------------------------------------------------------

def _make_df(columns: list[str], n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({c: np.arange(n, dtype=float) + i for i, c in enumerate(columns)})


def test_pick_default_axes_from_motors_outer_first():
    """start.motors = [outer, inner] → x=inner, y=outer."""
    df = _make_df(["piezo_y", "piezo_x", "pin_diode", "saxs_sum"])
    start = {"motors": ["piezo_y", "piezo_x"]}
    x, y, z = pick_default_axes(df, start)
    assert x == "piezo_x"
    assert y == "piezo_y"
    assert z == "pin_diode"


def test_pick_default_axes_from_hints_when_motors_missing():
    df = _make_df(["temperature", "exposure", "monitor", "I0"])
    start = {"hints": {"dimensions": [[["temperature"], "primary"],
                                       [["exposure"], "primary"]]}}
    x, y, z = pick_default_axes(df, start)
    assert x == "exposure"
    assert y == "temperature"
    assert z == "monitor"


def test_pick_default_axes_falls_back_to_first_numerics():
    """No start metadata → first two numeric cols as X/Y."""
    df = _make_df(["alpha", "beta", "gamma"])
    x, y, z = pick_default_axes(df, None)
    assert x == "alpha"
    assert y == "beta"
    assert z == "gamma"


def test_pick_default_axes_prefers_priority_z_names():
    """Z prefers intensity-like names even when listed later."""
    df = _make_df(["m1", "m2", "tail_motor", "saxs_sum"])
    start = {"motors": ["m1", "m2"]}
    x, y, z = pick_default_axes(df, start)
    assert z == "saxs_sum"


def test_pick_default_axes_handles_motor_not_in_df():
    """Motor names that aren't in the dataframe are silently skipped."""
    df = _make_df(["temperature", "pressure", "I0"])
    start = {"motors": ["energy", "temperature"]}
    x, y, z = pick_default_axes(df, start)
    # "energy" isn't a column → falls through to "temperature" for x,
    # then "pressure" for y (next numeric col).
    assert x == "temperature"
    assert y == "pressure"
    assert z == "I0"


def test_pick_default_axes_empty_df_returns_nones():
    x, y, z = pick_default_axes(pd.DataFrame(), {"motors": ["a", "b"]})
    assert (x, y, z) == (None, None, None)
