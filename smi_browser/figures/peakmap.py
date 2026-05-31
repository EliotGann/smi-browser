"""Bokeh builders for the Peak Map sub-tab.

Two figures:

* :func:`build_iq_heatmap` — a frame×q heatmap of the per-frame I(q) stack with
  an overlaid mean curve (on a secondary y-axis) and a shaded band per peak
  definition.  A box-select tool (x-only) lets the caller capture a drag to
  seed a new peak range (the caller attaches a ``SelectionGeometry`` handler).
  The image is column-decimated for display so thousands of frames stay fluid.

* :func:`build_peak_map` — the output map: a 1-D ``X → Z`` line when no Y axis
  is chosen, otherwise the 2-D ``X × Y → Z`` map delegated to
  :func:`smi_browser.figures.primary_2d.build_primary_2d`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from smi_browser.figures.primary_2d import (
    AVAILABLE_CMAPS,
    DEFAULT_CMAP,
    build_primary_2d,
)

__all__ = [
    "build_iq_heatmap",
    "build_peak_map",
    "band_source_data",
    "AVAILABLE_CMAPS",
    "DEFAULT_CMAP",
]

_PEAK_BAND_FILL = "#2ca02c"


def band_source_data(peaks, n_frames: int) -> dict:
    """Column data for the peak-range band glyphs (one vertical band/peak)."""
    left, right, top, bottom, name = [], [], [], [], []
    for pk in peaks:
        left.append(float(pk.q_min))
        right.append(float(pk.q_max))
        top.append(float(n_frames))
        bottom.append(0.0)
        name.append(str(pk.name))
    return dict(left=left, right=right, top=top, bottom=bottom, name=name)


def _percentile_range(values: np.ndarray, log: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 99.5))
    if lo == hi:
        hi = lo + (abs(lo) * 1e-6 if lo != 0 else 1.0)
    if log:
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.000001)
    return (lo, hi)


def build_iq_heatmap(
    q: Sequence[float],
    iq: np.ndarray,
    *,
    peaks: Sequence = (),
    cmap: str = "Turbo256",
    log_color: bool = True,
    display_max_cols: int = 1200,
    height: int = 320,
):
    """Frame×q heatmap of the per-frame I(q) stack.

    Parameters
    ----------
    q : 1-D q axis, shape ``(n_q,)``.
    iq : per-frame intensities, shape ``(n_frames, n_q)``.
    peaks : iterable of objects with ``q_min`` / ``q_max`` / ``name`` (PeakDef)
        — drawn as shaded bands.
    display_max_cols : decimate q columns to at most this many for rendering.

    Returns
    -------
    ``(figure, band_source)`` where ``band_source`` is the ColumnDataSource
    backing the peak bands.  The caller updates ``band_source.data`` (via
    :func:`band_source_data`) when peaks change, so adding a peak never rebuilds
    the figure — preserving the zoom and the active box-select tool.
    """
    from bokeh.models import (
        BoxSelectTool,
        ColorBar,
        ColumnDataSource,
        HoverTool,
        LinearColorMapper,
        LogColorMapper,
        Range1d,
    )
    from bokeh.plotting import figure as bk_figure

    q = np.asarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)
    if iq.ndim == 1:
        iq = iq[None, :]
    n_frames, n_q = iq.shape

    if q.size != n_q or n_q == 0 or n_frames == 0:
        p = bk_figure(height=height, sizing_mode="stretch_width",
                      title="No per-frame I(q) available")
        return p, ColumnDataSource(band_source_data((), 0))

    # Column-decimate for display only (keeps the glyph light).
    if n_q > display_max_cols:
        step = int(np.ceil(n_q / display_max_cols))
        col = np.arange(0, n_q, step)
    else:
        col = np.arange(n_q)
    q_disp = q[col]
    img = iq[:, col]

    x0, x1 = float(q[0]), float(q[-1])
    p = bk_figure(
        title="Per-frame I(q)  ·  drag (box-select) to add a peak range",
        height=height, sizing_mode="stretch_width",
        x_range=(x0, x1), y_range=(0, n_frames),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    cls = LogColorMapper if log_color else LinearColorMapper
    lo, hi = _percentile_range(img, log=log_color)
    mapper = cls(palette=cmap, low=lo, high=hi)
    renderer = p.image(
        image=[img.astype(np.float64, copy=False)],
        x=x0, y=0, dw=x1 - x0, dh=n_frames,
        color_mapper=mapper,
    )
    p.add_tools(HoverTool(renderers=[renderer], tooltips=[
        ("q", "$x{0.000}"),
        ("frame", "$y{0}"),
        ("I", "@image{0.000e}"),
    ]))
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, width=10,
                          title="I"), "right")
    p.xaxis.axis_label = "q (nm⁻¹)"
    p.yaxis.axis_label = "frame"

    # Box-select restricted to the q (x) direction, and made the default drag
    # tool so the user can sweep peak ranges without re-selecting it each time.
    box = BoxSelectTool(dimensions="width")
    p.add_tools(box)
    p.toolbar.active_drag = box

    # Mean curve on a secondary y-axis so its shape lines up in q.
    mean_iq = np.nanmean(iq, axis=0)
    finite = np.isfinite(mean_iq)
    if finite.any():
        mlo = float(np.nanmin(mean_iq[finite]))
        mhi = float(np.nanmax(mean_iq[finite]))
        if mhi <= mlo:
            mhi = mlo + 1.0
        p.extra_y_ranges = {"mean": Range1d(start=mlo, end=mhi * 1.05)}
        from bokeh.models import LinearAxis
        p.add_layout(LinearAxis(y_range_name="mean", axis_label="mean I"),
                     "left")
        p.line(q[finite], mean_iq[finite], y_range_name="mean",
               line_color="white", line_width=1.5, alpha=0.85)

    # Peak range bands — CDS-driven so the caller can update them in place.
    band_source = ColumnDataSource(band_source_data(peaks, n_frames))
    p.quad(left="left", right="right", top="top", bottom="bottom",
           source=band_source, fill_color=_PEAK_BAND_FILL, fill_alpha=0.15,
           line_color=_PEAK_BAND_FILL, line_alpha=0.6, level="overlay")
    p.text(x="left", y="top", text="name", source=band_source,
           text_font_size="8pt", text_color=_PEAK_BAND_FILL,
           x_offset=2, y_offset=2, level="overlay")
    return p, band_source


def build_peak_map(
    x: Sequence[float],
    y: Sequence[float] | None,
    z: Sequence[float],
    *,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    cmap: str = DEFAULT_CMAP,
    log_color: bool = False,
    aspect: str = "fill",
    height: int = 460,
):
    """Build the output map.

    Returns ``(figure, status)``.  When ``y`` is ``None`` a 1-D ``X → Z`` line
    plot is built; otherwise the 2-D map is delegated to ``build_primary_2d``.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    if y is not None:
        out = build_primary_2d(
            x, np.asarray(y, dtype=float), z,
            x_label=x_label, y_label=y_label, z_label=z_label,
            cmap=cmap, log_color=log_color, aspect=aspect, height=height,
        )
        return out.figure, out.status

    # --- 1-D X → Z ---
    from bokeh.models import HoverTool
    from bokeh.plotting import figure as bk_figure

    finite = np.isfinite(x) & np.isfinite(z)
    if not finite.any():
        return None, "*No finite points to plot.*"
    xf, zf = x[finite], z[finite]
    order = np.argsort(xf)
    xf, zf = xf[order], zf[order]

    p = bk_figure(
        title=f"{z_label} vs {x_label}",
        height=height, sizing_mode="stretch_width",
        y_axis_type="log" if log_color else "linear",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.line(xf, zf, line_color="#1f77b4", line_width=1.5)
    r = p.scatter(xf, zf, size=6, fill_color="#1f77b4", line_color=None)
    p.add_tools(HoverTool(renderers=[r], tooltips=[
        (x_label, "$x{0.000}"),
        (z_label, "$y{0.000e}"),
    ]))
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = z_label
    status = f"1D: {z_label} vs {x_label}  ·  {xf.size} points"
    return p, status
