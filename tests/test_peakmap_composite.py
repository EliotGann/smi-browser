"""Tests for the additive RGB peak/primary composite figure builder.

Cover the pure compositor (percentile clipping, normalization, RGB add,
grid detection / 1-D fallback, length mismatch handling) and the
RGB→RGBA-uint32 packing used by the Bokeh renderer.  The matplotlib /
Bokeh renderers themselves are exercised lightly to catch import or
shape errors — they're built on the pure compositor and don't need
pixel-by-pixel asserting.
"""
from __future__ import annotations

import numpy as np
import pytest

from smi_browser.figures.peakmap_composite import (
    DEFAULT_COLOR_CYCLE,
    DEFAULT_PCT_HI,
    DEFAULT_PCT_LO,
    channel_scale,
    color_to_hex,
    compose_rgb,
    default_color_for,
    normalize_channel,
    rgb_to_rgba_uint32,
)


# ---------------------------------------------------------------------------
# channel_scale: percentile clipping (matches reference script semantics)
# ---------------------------------------------------------------------------

def test_channel_scale_linear_default_percentiles():
    """The default 2nd / 99th percentiles trim outliers without mode-shifting
    the bulk of the data."""
    rng = np.random.default_rng(0)
    v = rng.uniform(10.0, 20.0, size=1_000)
    v[0] = 1e6  # outlier — must NOT push vmax up to ~1e6
    vmin, vmax = channel_scale(v)
    assert 9.5 <= vmin <= 11.0
    assert 19.5 <= vmax <= 20.5  # well below the outlier


def test_channel_scale_log_mode_drops_non_positive_values():
    """With ``log=True`` only strictly-positive values are considered."""
    v = np.array([np.nan, 0.0, -1.0, 1e-3, 1.0, 100.0, 1e6])
    vmin, vmax = channel_scale(v, log=True, pct_lo=0, pct_hi=100)
    # Min/max are computed in log10 space.
    assert vmin == pytest.approx(np.log10(1e-3))
    assert vmax == pytest.approx(np.log10(1e6))


def test_channel_scale_all_nan_returns_safe_fallback():
    """An all-NaN channel must not raise — the legend still wants a range."""
    v = np.full(10, np.nan)
    assert channel_scale(v) == (0.0, 1.0)
    assert channel_scale(v, log=True) == (1.0, 10.0)


def test_channel_scale_constant_values_returns_nonzero_width():
    """A flat channel would otherwise produce vmin == vmax → divide by zero
    in normalize_channel.  We bump vmax above vmin instead."""
    v = np.full(10, 4.2)
    vmin, vmax = channel_scale(v)
    assert vmax > vmin


# ---------------------------------------------------------------------------
# normalize_channel
# ---------------------------------------------------------------------------

def test_normalize_channel_linear_maps_to_unit_range():
    v = np.array([0.0, 5.0, 10.0])
    out = normalize_channel(v, vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_normalize_channel_clips_outside_vmin_vmax():
    v = np.array([-5.0, 0.0, 5.0, 10.0, 20.0])
    out = normalize_channel(v, vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_normalize_channel_nan_becomes_zero():
    v = np.array([np.nan, 1.0, np.nan])
    out = normalize_channel(v, vmin=0.0, vmax=2.0)
    np.testing.assert_allclose(out, [0.0, 0.5, 0.0])


def test_normalize_channel_log_drops_non_positive_to_zero():
    v = np.array([0.0, -1.0, 1.0, 10.0, 100.0])
    out = normalize_channel(
        v, vmin=np.log10(1.0), vmax=np.log10(100.0), log=True,
    )
    # 0 and -1 → NaN after log → clipped to 0; 1 → 0; 10 → 0.5; 100 → 1
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.5, 1.0])


def test_normalize_channel_zero_width_does_not_divide_by_zero():
    v = np.array([1.0, 2.0, 3.0])
    out = normalize_channel(v, vmin=2.0, vmax=2.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# compose_rgb: 2-D grid path
# ---------------------------------------------------------------------------

def _grid_xy(nx=5, ny=4):
    """Return flattened (x, y) for a regular nx*ny grid."""
    xs = np.linspace(0.0, nx - 1, nx)
    ys = np.linspace(0.0, ny - 1, ny)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def test_compose_rgb_2d_pure_red_channel_matches_normalized_grid():
    """A single red-only channel produces an image whose red plane equals
    the normalized grid and whose G/B planes are zero — sanity check on
    grid alignment, channel scaling, and additive compositing."""
    x, y = _grid_xy(nx=5, ny=4)
    z = np.linspace(1.0, 20.0, 20)
    res = compose_rgb(
        channels=[{"id": "r", "label": "r", "values": z,
                   "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False}],
        x=x, y=y, pct_lo=0, pct_hi=100,
    )
    assert res.gridded is True
    assert res.rgb.shape == (4, 5, 3)
    assert res.rgb[..., 1].max() == 0.0  # G never gets red's signal
    assert res.rgb[..., 2].max() == 0.0  # B same
    # Red plane is monotonically increasing along the row-major axis since
    # z is.  Brightest cell at the end (z = 20) hits 1.0 after scaling.
    assert res.rgb[..., 0].max() == pytest.approx(1.0)


def test_compose_rgb_2d_three_channels_add_to_white_at_their_max():
    """At the brightest cell of each channel, the additive composite hits
    full intensity — and pure-RGB primaries don't bleed across."""
    x, y = _grid_xy(nx=4, ny=3)
    n = x.size
    # Three "single-pixel hot" channels at three different cells.
    cells = [0, 5, 10]
    z_r = np.zeros(n); z_r[cells[0]] = 1.0
    z_g = np.zeros(n); z_g[cells[1]] = 1.0
    z_b = np.zeros(n); z_b[cells[2]] = 1.0
    res = compose_rgb(
        channels=[
            {"id": "r", "label": "r", "values": z_r,
             "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False},
            {"id": "g", "label": "g", "values": z_g,
             "color": (0.0, 1.0, 0.0), "gain": 1.0, "log": False},
            {"id": "b", "label": "b", "values": z_b,
             "color": (0.0, 0.0, 1.0), "gain": 1.0, "log": False},
        ],
        x=x, y=y, pct_lo=0, pct_hi=100,
    )
    assert res.gridded is True
    # Each channel's max cell has full intensity in only that colour plane.
    flat = res.rgb.reshape(-1, 3)
    assert flat[cells[0], 0] == pytest.approx(1.0)
    assert flat[cells[0], 1] == 0.0 and flat[cells[0], 2] == 0.0
    assert flat[cells[1], 1] == pytest.approx(1.0)
    assert flat[cells[1], 0] == 0.0 and flat[cells[1], 2] == 0.0
    assert flat[cells[2], 2] == pytest.approx(1.0)


def test_compose_rgb_gain_applies_after_normalize_and_clips_at_one():
    """A gain > 1.0 brightens the channel but must still clip at 1.0 so the
    composite stays in displayable range."""
    x, y = _grid_xy(nx=4, ny=3)
    z = np.linspace(0.0, 1.0, x.size)  # already in [0, 1]
    res = compose_rgb(
        channels=[{"id": "r", "label": "r", "values": z,
                   "color": (1.0, 0.0, 0.0), "gain": 5.0, "log": False}],
        x=x, y=y, pct_lo=0, pct_hi=100,
    )
    # Without clipping the brightest cell would be 5.0; with clipping, 1.0.
    assert res.rgb[..., 0].max() == pytest.approx(1.0)
    # Cells originally normalized to >= 0.2 all clip to 1.0 with gain=5.
    assert (res.rgb[..., 0] >= 0.999).sum() >= (z >= 0.2).sum()


def test_compose_rgb_records_per_channel_scales_for_legend():
    """``channel_scales`` is what the legend reads — it must contain every
    channel even when the channel got dropped (e.g. length mismatch)."""
    x, y = _grid_xy(nx=4, ny=3)
    n = x.size
    res = compose_rgb(
        channels=[
            {"id": "ok", "label": "ok", "values": np.linspace(0, 1, n),
             "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False},
            {"id": "wrong_length", "label": "wl",
             "values": np.zeros(n + 5),
             "color": (0.0, 1.0, 0.0), "gain": 1.0, "log": False},
        ],
        x=x, y=y, pct_lo=0, pct_hi=100,
    )
    assert "ok" in res.channel_scales
    assert "wrong_length" in res.skipped
    # The wrong-length channel had no scale computed (it was skipped before
    # the scale step).  The kept-channel scale must reflect the data range.
    ok_lo, ok_hi = res.channel_scales["ok"]
    assert ok_lo == pytest.approx(0.0)
    assert ok_hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compose_rgb: 1-D fallback
# ---------------------------------------------------------------------------

def test_compose_rgb_1d_fallback_when_no_y_axis():
    """Without a Y axis, the compositor returns a single-row RGB image and
    sets ``gridded=False`` so the caller can swap layouts."""
    n = 20
    x = np.linspace(0, 1, n)
    z = np.linspace(0.0, 1.0, n)
    res = compose_rgb(
        channels=[{"id": "r", "label": "r", "values": z,
                   "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False}],
        x=x, y=None, pct_lo=0, pct_hi=100,
    )
    assert res.gridded is False
    assert res.rgb.shape == (1, n, 3)
    # x order is preserved (sorted ascending) so the brightest pixel is at the right.
    assert np.argmax(res.rgb[0, :, 0]) == n - 1


def test_compose_rgb_falls_back_to_1d_when_grid_detection_fails():
    """An irregular X/Y point cloud (not on a grid) must still render — the
    1-D fallback is the safety net."""
    rng = np.random.default_rng(0)
    n = 30
    x = rng.uniform(0, 1, n)
    y = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    res = compose_rgb(
        channels=[{"id": "r", "label": "r", "values": z,
                   "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False}],
        x=x, y=y, pct_lo=0, pct_hi=100,
    )
    assert res.gridded is False
    assert res.rgb.shape == (1, n, 3)


# ---------------------------------------------------------------------------
# Length-mismatch / no-data resilience
# ---------------------------------------------------------------------------

def test_compose_rgb_no_usable_channels_returns_empty_image_not_raise():
    """When *every* channel is unusable, return a 1×1 black image so the
    caller can render a "no data" placeholder without special-casing."""
    n = 10
    x = np.linspace(0, 1, n)
    res = compose_rgb(
        channels=[{"id": "wrong", "label": "wrong",
                   "values": np.zeros(n + 1),
                   "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False}],
        x=x, y=None,
    )
    assert res.rgb.shape == (1, 1, 3)
    assert res.rgb.sum() == 0.0
    assert "wrong" in res.skipped


def test_compose_rgb_log_channel_yields_log_scale_in_metadata():
    """A log-scaled channel records (vmin, vmax) in log10 space so the
    legend annotation matches what the eye sees in the picture."""
    x, y = _grid_xy(nx=4, ny=3)
    n = x.size
    z = np.geomspace(1.0, 1e4, n)
    res = compose_rgb(
        channels=[{"id": "saxs", "label": "saxs", "values": z,
                   "color": (1.0, 1.0, 1.0), "gain": 1.0, "log": True}],
        x=x, y=y, pct_lo=0, pct_hi=100,
    )
    lo, hi = res.channel_scales["saxs"]
    assert lo == pytest.approx(0.0)        # log10(1) = 0
    assert hi == pytest.approx(4.0)        # log10(1e4) = 4


# ---------------------------------------------------------------------------
# Bokeh RGBA packing
# ---------------------------------------------------------------------------

def test_rgb_to_rgba_uint32_packs_red_correctly():
    """Pack a single fully-red pixel and verify the byte layout.

    Bokeh expects little-endian RGBA: low byte = R, high byte = A.
    """
    rgb = np.zeros((1, 1, 3), dtype=np.float32)
    rgb[0, 0] = (1.0, 0.0, 0.0)
    out = rgb_to_rgba_uint32(rgb)
    assert out.shape == (1, 1)
    word = int(out[0, 0])
    # alpha = 255 (high byte)
    assert (word >> 24) & 0xFF == 255
    assert (word >> 16) & 0xFF == 0       # B
    assert (word >> 8) & 0xFF == 0        # G
    assert word & 0xFF == 255             # R


def test_rgb_to_rgba_uint32_packs_green_and_blue_in_correct_bytes():
    rgb = np.zeros((1, 2, 3), dtype=np.float32)
    rgb[0, 0] = (0.0, 1.0, 0.0)  # green
    rgb[0, 1] = (0.0, 0.0, 1.0)  # blue
    out = rgb_to_rgba_uint32(rgb)
    word_g = int(out[0, 0])
    word_b = int(out[0, 1])
    assert (word_g >> 8) & 0xFF == 255    # G
    assert word_g & 0xFF == 0
    assert (word_b >> 16) & 0xFF == 255   # B
    assert word_b & 0xFF == 0


# ---------------------------------------------------------------------------
# Default colour cycle / hex roundtrip
# ---------------------------------------------------------------------------

def test_default_color_for_cycles_after_six():
    """The 7th channel reuses the 1st colour — caller is responsible for
    ensuring there are at most six default-coloured channels at once."""
    assert default_color_for(0) == DEFAULT_COLOR_CYCLE[0]
    assert default_color_for(6) == DEFAULT_COLOR_CYCLE[0]


def test_color_to_hex_roundtrips_through_rgb_tuple():
    assert color_to_hex((1.0, 0.0, 0.0)) == "#ff0000"
    assert color_to_hex((0.0, 1.0, 0.0)) == "#00ff00"
    assert color_to_hex((0.0, 0.0, 1.0)) == "#0000ff"
    # 8-bit input also accepted (matches what a ColorPicker may emit).
    assert color_to_hex((255, 128, 0)) == "#ff8000"


# ---------------------------------------------------------------------------
# Renderer smoke tests (don't assert pixels — just that they build)
# ---------------------------------------------------------------------------

def test_build_matplotlib_figure_smoke():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from smi_browser.figures.peakmap_composite import build_matplotlib_figure

    x, y = _grid_xy(nx=4, ny=3)
    n = x.size
    channels = [
        {"id": "alpha", "label": "alpha (q=1.250)",
         "values": np.linspace(0, 1, n), "color": (1.0, 0.0, 0.0),
         "gain": 1.0, "log": False},
        {"id": "primary:total", "label": "total",
         "values": np.linspace(1, 1000, n),
         "color": (1.0, 1.0, 1.0), "gain": 1.0, "log": True},
    ]
    res = compose_rgb(channels, x=x, y=y)
    fig = build_matplotlib_figure(
        res, channels, x_label="piezo_x", y_label="piezo_y", title="test",
    )
    try:
        # Two axes (image + legend), legend has one Patch per channel.
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_build_bokeh_figure_smoke():
    """Smoke-test the Bokeh builder so an import error or shape mismatch
    surfaces in CI without needing a running Panel server."""
    from smi_browser.figures.peakmap_composite import build_bokeh_figure

    x, y = _grid_xy(nx=4, ny=3)
    n = x.size
    channels = [{"id": "r", "label": "r",
                 "values": np.linspace(0, 1, n),
                 "color": (1.0, 0.0, 0.0), "gain": 1.0, "log": False}]
    res = compose_rgb(channels, x=x, y=y)
    p = build_bokeh_figure(res, channels, x_label="x", y_label="y", title="t")
    # Bokeh figures always have at least the renderers we attached.
    assert any(getattr(r, "glyph", None).__class__.__name__ == "ImageRGBA"
               for r in p.renderers)
