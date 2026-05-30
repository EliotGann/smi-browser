"""Bokeh figure builder for the Primary-tab 2D plot.

The caller has already chosen which dataframe columns drive X, Y, Z and
has handed off the corresponding 1-D arrays.  This module decides
whether to render as a colour-mapped image (regular grid) or as a
colour-mapped scatter (everything else), and returns a Bokeh figure
plus a short status string for the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from smi_browser.models.grid import GriddedView, gridded_view

__all__ = [
    "Primary2DPlot",
    "build_primary_2d",
    "AVAILABLE_CMAPS",
    "DEFAULT_CMAP",
]

#: Bokeh palette names exposed in the UI.
AVAILABLE_CMAPS: tuple[str, ...] = ("Viridis256", "Turbo256", "Cividis256",
                                    "Greys256", "Plasma256", "Inferno256")
DEFAULT_CMAP = "Viridis256"


@dataclass(frozen=True)
class Primary2DPlot:
    figure: "object"     # bokeh.plotting.figure
    status: str
    is_grid: bool
    grid: GriddedView | None = None


def _percentile_range(values: np.ndarray, log: bool) -> tuple[float, float]:
    """Robust ``(low, high)`` for the colour mapper, ignoring NaN/inf."""
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 99.5))
    if lo == hi:
        # Avoid zero-span mapper.
        hi = lo + (abs(lo) * 1e-6 if lo != 0 else 1.0)
    if log:
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.000001)
    return (lo, hi)


def _build_mapper(values: np.ndarray, palette: str, log: bool):
    from bokeh.models import LinearColorMapper, LogColorMapper

    lo, hi = _percentile_range(values, log=log)
    cls = LogColorMapper if log else LinearColorMapper
    return cls(palette=palette, low=lo, high=hi)


def _grid_status(gv: GriddedView, x_label: str, y_label: str) -> str:
    ny, nx = gv.image.shape
    pct = gv.fill_fraction * 100
    base = f"{nx}×{ny} grid on ({x_label}, {y_label}), {pct:.0f}% filled"
    if gv.max_collision > 1:
        base += (f"  ·  ⚠ {gv.max_collision} pts/cell max — "
                 f"likely a third varying axis; values are averaged")
    return base


def build_primary_2d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    *,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "z",
    cmap: str = DEFAULT_CMAP,
    log_color: bool = False,
    aspect: str = "fill",
    title: str | None = None,
    height: int = 460,
    tol_frac: float = 1e-4,
    max_sparseness: float = 4.0,
) -> Primary2DPlot:
    """Render an (x, y, z) triple as image or scatter, picking automatically.

    Parameters
    ----------
    x, y, z
        Equal-length 1-D numeric arrays.
    x_label, y_label, z_label
        Axis labels (typically the original dataframe column names).
    cmap
        Bokeh palette name; see :data:`AVAILABLE_CMAPS`.
    log_color
        Use a logarithmic colour mapper (only over the positive subset).
    aspect : {'fill', 'equal'}
        ``'fill'`` (default) lets the plot stretch to the available
        width.  ``'equal'`` forces ``match_aspect=True`` so motor units
        stay isotropic.
    title
        Plot title.  ``None`` → ``"{z_label} on ({x_label}, {y_label})"``.
    height
        Plot height in pixels.  Width is driven by ``sizing_mode``.
    tol_frac, max_sparseness
        Forwarded to :func:`gridded_view`.

    Returns
    -------
    Primary2DPlot
        Bundle of figure + diagnostic status string + the grid view (or
        ``None`` for the scatter fallback).
    """
    from bokeh.models import ColorBar, HoverTool
    from bokeh.plotting import figure as bk_figure

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if not (x_arr.shape == y_arr.shape == z_arr.shape):
        raise ValueError("x, y, z must share the same shape")
    if x_arr.size == 0:
        return Primary2DPlot(figure=None, status="*No data to plot.*",
                             is_grid=False, grid=None)

    if title is None:
        title = f"{z_label} on ({x_label}, {y_label})"

    gv = gridded_view(
        x_arr, y_arr, z_arr,
        tol_frac=tol_frac,
        max_sparseness=max_sparseness,
    )

    fig_kw = dict(
        title=title, height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        sizing_mode="stretch_width",
    )
    if aspect == "equal":
        fig_kw["match_aspect"] = True

    if gv is not None:
        # --- Gridded image path ---
        x_centres = gv.x_centres
        y_centres = gv.y_centres
        # Bokeh image() places the array with corner at (x, y) and
        # extents (dw, dh).  We want pixel centres at x_centres / y_centres
        # so the visible extent is centred on each cell.
        dx = (x_centres[-1] - x_centres[0]) / max(len(x_centres) - 1, 1)
        dy = (y_centres[-1] - y_centres[0]) / max(len(y_centres) - 1, 1)
        x0 = float(x_centres[0] - dx / 2.0)
        y0 = float(y_centres[0] - dy / 2.0)
        x1 = float(x_centres[-1] + dx / 2.0)
        y1 = float(y_centres[-1] + dy / 2.0)
        p = bk_figure(
            x_range=(x0, x1), y_range=(y0, y1),
            **fig_kw,
        )
        mapper = _build_mapper(gv.image, cmap, log_color)
        # Replace NaNs with a colour-mapped sentinel so they render as
        # transparent.  Bokeh's LogColorMapper drops NaN/inf to the
        # ``nan_color`` slot (default transparent), which is what we want.
        display = gv.image.astype(np.float64, copy=True)
        renderer = p.image(
            image=[display], x=x0, y=y0,
            dw=x1 - x0, dh=y1 - y0,
            color_mapper=mapper,
        )
        # Hover: show cell index + value.  Bokeh's image hover gives us
        # x/y in data coords; we map them to nearest cell ourselves.
        p.add_tools(HoverTool(renderers=[renderer], tooltips=[
            (x_label, "@x{0.000}"),
            (y_label, "@y{0.000}"),
            (z_label, "@image{0.000e}"),
        ]))
        bar = ColorBar(color_mapper=mapper, label_standoff=8, width=12,
                       title=z_label)
        p.add_layout(bar, "right")
        p.xaxis.axis_label = x_label
        p.yaxis.axis_label = y_label
        return Primary2DPlot(
            figure=p,
            status=_grid_status(gv, x_label, y_label),
            is_grid=True,
            grid=gv,
        )

    # --- Scatter fallback ---
    from bokeh.models import ColumnDataSource

    # Only keep finite triples for scatter.
    finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
    xf = x_arr[finite]
    yf = y_arr[finite]
    zf = z_arr[finite]
    if xf.size == 0:
        return Primary2DPlot(figure=None,
                             status="*All points were non-finite.*",
                             is_grid=False, grid=None)

    p = bk_figure(**fig_kw)
    mapper = _build_mapper(zf, cmap, log_color)
    src = ColumnDataSource(data={"x": xf, "y": yf, "z": zf})
    renderer = p.scatter(
        x="x", y="y", source=src,
        size=6,
        fill_color={"field": "z", "transform": mapper},
        line_color=None, fill_alpha=0.85,
    )
    p.add_tools(HoverTool(renderers=[renderer], tooltips=[
        (x_label, "@x{0.000}"),
        (y_label, "@y{0.000}"),
        (z_label, "@z{0.000e}"),
    ]))
    bar = ColorBar(color_mapper=mapper, label_standoff=8, width=12,
                   title=z_label)
    p.add_layout(bar, "right")
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    return Primary2DPlot(
        figure=p,
        status=f"scatter, {xf.size} points on ({x_label}, {y_label})",
        is_grid=False,
        grid=None,
    )
