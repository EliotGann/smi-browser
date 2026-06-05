"""RGB-additive peak/primary composite figure (a.k.a. "fluorescence overlay").

Each channel (a peak's fitted ``area`` map, or a per-frame primary scalar)
is independently percentile-normalized to ``[0, 1]``, multiplied by an RGB
colour, and added on a black background.  The resulting composite captures
several maps in one frame and is the same compositing model used by the
user's stand-alone ``make_ezra_overlay.py`` script.

Two renderers share one pure compositor:

* :func:`compose_rgb` builds the ``(H, W, 3)`` float RGB image plus the
  axes — Panel/Bokeh-free, unit-tested.
* :func:`build_matplotlib_figure` produces the matplotlib figure used by
  the PNG exporter (WYSIWYG with the user's reference script).
* :func:`build_bokeh_figure` produces the interactive Bokeh figure used by
  the Peak Map tab (zoom/pan, hover-on-extent).

Channel spec (one dict per channel):

    {
        "id":     str,                # unique key — peak slug or "primary:<col>"
        "label":  str,                # legend-display label
        "values": np.ndarray (n,),    # 1-D per-frame metric
        "color":  (r, g, b) in [0, 1],
        "gain":   float,              # post-normalize multiplier
        "log":    bool,               # log10 before normalizing (positives only)
        "kind":   "peak" | "primary", # for legend grouping (optional)
    }

A 2-D scan is detected via :func:`smi_browser.models.grid.gridded_view` on
the first channel that produces a usable grid; failure falls back to a
1-D line plot in the same colour scheme.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from smi_browser.models.grid import gridded_view


# Default percentile clip range, matching the user's reference script.
DEFAULT_PCT_LO = 2.0
DEFAULT_PCT_HI = 99.0


# ---------------------------------------------------------------------------
# Pure normalization / compositing
# ---------------------------------------------------------------------------

def channel_scale(
    values: np.ndarray,
    pct_lo: float = DEFAULT_PCT_LO,
    pct_hi: float = DEFAULT_PCT_HI,
    log: bool = False,
) -> tuple[float, float]:
    """Return ``(vmin, vmax)`` for a channel from finite-value percentiles.

    When ``log`` is set, only strictly-positive values are considered and
    the percentile is computed in log10 space — matching the reference
    script's ``--saxs-log`` behaviour.  An all-NaN / all-zero channel
    returns a fallback range (``(0, 1)`` linear, ``(1, 10)`` log) so the
    downstream normalize doesn't divide by zero.
    """
    a = np.asarray(values, dtype=float).ravel()
    if log:
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return (1.0, 10.0)
        a = np.log10(a)
    else:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (0.0, 1.0)
    vmin = float(np.percentile(a, pct_lo))
    vmax = float(np.percentile(a, pct_hi))
    if vmin == vmax:
        # Avoid a zero-width range; spread vmax above so normalize still works.
        vmax = vmin + 1.0
    return (vmin, vmax)


def normalize_channel(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    log: bool = False,
) -> np.ndarray:
    """Map ``values`` onto ``[0, 1]`` using ``(vmin, vmax)``, NaN→0.

    Optional ``log10`` is applied first (negatives/zeros become NaN, then 0
    after the final mask).  ``vmax == vmin`` falls back to a divide-by-1
    via ``max(..., 1e-30)`` so the function is total.
    """
    a = np.asarray(values, dtype=float)
    if log:
        # ``np.where`` evaluates both branches; suppress the harmless
        # log10-of-zero/negative warning rather than emitting it on every
        # render.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(np.isfinite(a) & (a > 0), np.log10(a), np.nan)
    a = (a - vmin) / max(vmax - vmin, 1e-30)
    a = np.where(np.isfinite(a), a, 0.0)
    return np.clip(a, 0.0, 1.0)


@dataclass(frozen=True)
class CompositeResult:
    """Output of :func:`compose_rgb`.

    Attributes
    ----------
    rgb : np.ndarray
        ``(H, W, 3)`` float32 RGB in ``[0, 1]``.  Empty (zero) array when
        no channel produced data.
    xs, ys : np.ndarray
        Axis cell-centres for ``rgb`` (always ascending).  ``xs.size == W``
        and ``ys.size == H``.
    gridded : bool
        True when a regular 2-D grid was detected; False means the caller
        should fall back to a 1-D line plot.
    channel_scales : dict[str, tuple[float, float]]
        ``{channel_id: (vmin, vmax)}`` for every input channel that had any
        finite data.  Used to render the legend's ``[lo .. hi]`` annotation.
    skipped : list[str]
        ``channel_id`` values that were silently excluded — usually because
        their length didn't match the chosen (x, y) axes.
    """
    rgb: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    gridded: bool
    channel_scales: dict
    skipped: list


def compose_rgb(
    channels: Sequence[dict],
    x: np.ndarray,
    y: np.ndarray | None = None,
    pct_lo: float = DEFAULT_PCT_LO,
    pct_hi: float = DEFAULT_PCT_HI,
) -> CompositeResult:
    """Build the additive RGB composite from ``channels`` over ``(x, y)``.

    Channels with mismatched length, all-NaN data, or zero-gain are silently
    excluded from the image but kept in ``channel_scales`` so the legend can
    still annotate them.  The 2-D path uses :func:`gridded_view`; if the
    points don't form a regular grid (e.g. true 1-D scan, or two
    irregularly-spaced motors), ``rgb`` is returned as a synthetic
    ``(1, n)`` row so callers always get a renderable shape, with
    ``gridded=False`` to flag the fallback.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.size)

    # First pass: per-channel scales (independent of the grid).  Done
    # eagerly so the legend can show ranges even when a channel is dropped.
    scales: dict[str, tuple[float, float]] = {}
    skipped: list[str] = []
    usable: list[dict] = []
    for ch in channels:
        cid = str(ch.get("id", ""))
        vals = np.asarray(ch.get("values"), dtype=float)
        if vals.size != n:
            skipped.append(cid)
            continue
        if not np.isfinite(vals).any():
            scales[cid] = (0.0, 1.0)
            continue
        scales[cid] = channel_scale(
            vals, pct_lo=pct_lo, pct_hi=pct_hi, log=bool(ch.get("log")),
        )
        usable.append(ch)

    if not usable:
        return CompositeResult(
            rgb=np.zeros((1, 1, 3), dtype=np.float32),
            xs=np.array([0.0]), ys=np.array([0.0]),
            gridded=False, channel_scales=scales, skipped=skipped,
        )

    # 2-D path: pick the first channel that successfully grids; reuse its
    # axes for every other channel so they line up cell-for-cell.
    grid = None
    if y is not None and np.asarray(y, dtype=float).size == n:
        y_arr = np.asarray(y, dtype=float)
        for ch in usable:
            grid = gridded_view(x, y_arr, np.asarray(ch["values"], dtype=float))
            if grid is not None:
                break

    if grid is not None:
        H, W = grid.image.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for ch in usable:
            cid = str(ch.get("id", ""))
            color = _to_rgb(ch.get("color"))
            gain = float(ch.get("gain", 1.0) or 1.0)
            log = bool(ch.get("log"))
            # Re-grid this channel onto the same x/y centres as the chosen
            # reference grid so the composite is cell-aligned.
            ch_grid = gridded_view(
                x, y_arr,
                np.asarray(ch["values"], dtype=float),
            )
            if ch_grid is None or ch_grid.image.shape != (H, W):
                skipped.append(cid)
                continue
            norm = normalize_channel(ch_grid.image, *scales[cid], log=log)
            for c in range(3):
                rgb[..., c] += norm * color[c] * gain
        np.clip(rgb, 0.0, 1.0, out=rgb)
        return CompositeResult(
            rgb=rgb,
            xs=grid.x_centres, ys=grid.y_centres,
            gridded=True, channel_scales=scales, skipped=skipped,
        )

    # 1-D fallback: synthesise a single-row "image" so the matplotlib /
    # bokeh paths can both ``imshow`` it.  Sort by x for legibility.
    order = np.argsort(x)
    xs = x[order]
    rgb = np.zeros((1, xs.size, 3), dtype=np.float32)
    for ch in usable:
        cid = str(ch.get("id", ""))
        color = _to_rgb(ch.get("color"))
        gain = float(ch.get("gain", 1.0) or 1.0)
        log = bool(ch.get("log"))
        vals = np.asarray(ch["values"], dtype=float)[order]
        norm = normalize_channel(vals, *scales[cid], log=log)
        for c in range(3):
            rgb[0, :, c] += norm * color[c] * gain
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return CompositeResult(
        rgb=rgb, xs=xs, ys=np.array([0.0]),
        gridded=False, channel_scales=scales, skipped=skipped,
    )


def _to_rgb(color) -> tuple[float, float, float]:
    """Coerce a colour spec to a ``(r, g, b)`` tuple of floats in ``[0, 1]``.

    Accepts ``"#rrggbb"``, ``(r, g, b)`` in ``[0, 1]``, or ``(r, g, b)`` in
    ``[0, 255]``.  Unrecognised values fall back to white.
    """
    if isinstance(color, str):
        s = color.strip().lstrip("#")
        if len(s) == 6:
            try:
                r = int(s[0:2], 16) / 255.0
                g = int(s[2:4], 16) / 255.0
                b = int(s[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass
        return (1.0, 1.0, 1.0)
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        r, g, b = (float(color[0]), float(color[1]), float(color[2]))
        if max(r, g, b) > 1.0:
            return (r / 255.0, g / 255.0, b / 255.0)
        return (r, g, b)
    return (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Default colour cycle
# ---------------------------------------------------------------------------

#: Channel-colour cycle used when the user hasn't picked a colour.  Matches
#: the reference script's "fluorescence" palette: red, green, blue, yellow,
#: magenta, cyan — all primary additive colours so combined channels add to
#: white.
DEFAULT_COLOR_CYCLE: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),    # red
    (0.0, 1.0, 0.0),    # green
    (0.0, 0.4, 1.0),    # blue (slightly cyan-shifted, like the script)
    (1.0, 0.85, 0.0),   # yellow
    (1.0, 0.0, 1.0),    # magenta
    (0.0, 1.0, 1.0),    # cyan
)


def default_color_for(index: int) -> tuple[float, float, float]:
    """Return the default colour for the ``index``-th channel (cycles)."""
    return DEFAULT_COLOR_CYCLE[index % len(DEFAULT_COLOR_CYCLE)]


def color_to_hex(color) -> str:
    """Convert any RGB spec to a ``"#rrggbb"`` string for Tabulator/ColorPicker."""
    r, g, b = _to_rgb(color)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255)),
    )


# ---------------------------------------------------------------------------
# Matplotlib renderer (PNG export, WYSIWYG with the reference script)
# ---------------------------------------------------------------------------

def build_matplotlib_figure(
    composite: CompositeResult,
    channels: Sequence[dict],
    *,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (13.0, 7.5),
):
    """Render a composite to a matplotlib ``Figure``.

    Layout matches the reference script: the RGB image fills the left
    panel, a colour-swatch legend with the per-channel ``[vmin .. vmax]``
    ranges sits on the right.  Caller owns the figure (closes it).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (ax, ax_leg) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw=dict(width_ratios=[5, 1]),
    )

    rgb = composite.rgb
    xs, ys = composite.xs, composite.ys
    if rgb.shape[0] == 1 and not composite.gridded:
        # 1-D fallback: stretch the row vertically a bit so it's visible.
        extent = (float(xs.min()), float(xs.max()), 0.0, 1.0)
        ax.imshow(rgb, origin="lower", extent=extent, aspect="auto",
                  interpolation="nearest")
        ax.set_yticks([])
    else:
        extent = (float(xs.min()), float(xs.max()),
                  float(ys.min()), float(ys.max()))
        ax.imshow(rgb, origin="lower", extent=extent, aspect="auto",
                  interpolation="nearest")
        if y_label:
            ax.set_ylabel(y_label)

    if x_label:
        ax.set_xlabel(x_label)
    if title:
        ax.set_title(title, fontsize=10)

    handles = []
    skipped = set(composite.skipped)
    for ch in channels:
        cid = str(ch.get("id", ""))
        label = str(ch.get("label", cid))
        color = _to_rgb(ch.get("color"))
        log_tag = " [log10]" if ch.get("log") else ""
        scale = composite.channel_scales.get(cid)
        if scale is not None:
            text = f"{label}{log_tag}\n  [{scale[0]:.3g} .. {scale[1]:.3g}]"
        else:
            text = f"{label}{log_tag}\n  (no data)"
        if cid in skipped:
            text += "  (excluded)"
        handles.append(Patch(facecolor=color, edgecolor="0.3", label=text))
    ax_leg.legend(handles=handles, loc="center left", fontsize=8,
                  frameon=False, handlelength=2.0, labelspacing=1.2)
    ax_leg.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bokeh renderer (interactive Peak Map tab)
# ---------------------------------------------------------------------------

def rgb_to_rgba_uint32(rgb: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Convert ``(H, W, 3)`` float RGB → ``(H, W)`` uint32 RGBA for Bokeh.

    Bokeh's ``image_rgba`` expects little-endian RGBA packed into a single
    uint32 per pixel.  Values are clipped to ``[0, 1]`` then scaled to
    ``[0, 255]``.
    """
    a = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0) * 255.0
    a = a.astype(np.uint32)
    H, W, _ = a.shape
    out = np.empty((H, W), dtype=np.uint32)
    out[:] = (
        (alpha & 0xFF) << 24
        | (a[..., 2] & 0xFF) << 16   # B
        | (a[..., 1] & 0xFF) << 8    # G
        | (a[..., 0] & 0xFF)         # R
    )
    return out


def build_bokeh_figure(
    composite: CompositeResult,
    channels: Sequence[dict],
    *,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    width: int = 720,
    height: int = 470,
):
    """Render a composite to a Bokeh figure with a colour-swatch legend.

    The image uses ``image_rgba`` (one packed uint32 per cell) so the
    additive composite renders exactly as the matplotlib version.  Hover
    on the image surfaces the cell's ``(x, y)`` in data coordinates.
    """
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure as _bk_figure

    rgb = composite.rgb
    xs, ys = composite.xs, composite.ys
    rgba = rgb_to_rgba_uint32(rgb)

    if not composite.gridded and xs.size > 1:
        # 1-D fallback: pad to a thin band for visibility.
        y0, y1 = 0.0, 1.0
    else:
        y0, y1 = float(ys.min()), float(ys.max())

    x0, x1 = float(xs.min()), float(xs.max())
    dw = max(x1 - x0, 1e-12)
    dh = max(y1 - y0, 1e-12)

    p = _bk_figure(
        width=width, height=height, title=title,
        x_axis_label=x_label or None, y_axis_label=y_label or None,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_drag="box_zoom", active_scroll="wheel_zoom",
        background_fill_color="black",
        border_fill_color="black",
        outline_line_color="#888888",
    )
    if title:
        p.title.text_color = "#dddddd"
    p.xaxis.axis_label_text_color = "#dddddd"
    p.yaxis.axis_label_text_color = "#dddddd"
    p.xaxis.major_label_text_color = "#dddddd"
    p.yaxis.major_label_text_color = "#dddddd"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    src = ColumnDataSource(data={
        "image": [rgba],
        "x": [x0], "y": [y0],
        "dw": [dw], "dh": [dh],
    })
    p.image_rgba(image="image", x="x", y="y", dw="dw", dh="dh", source=src)

    # Hover surfaces (x, y) in data coords; the channel breakdown is left
    # to the legend table because individual channel intensities aren't
    # recoverable from the composited pixel.
    hover = HoverTool(
        tooltips=[("x", "$x{0.000}"), ("y", "$y{0.000}")],
        attachment="right",
    )
    p.add_tools(hover)
    return p
