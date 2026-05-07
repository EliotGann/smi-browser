"""Bokeh image figure builder and in-place updater for the Explore tab."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def thumbnail_figure(arr: np.ndarray, title: str, mask_visible: bool = False):
    """Build an interactive Bokeh image figure with mask/draw/edit overlays.

    Parameters
    ----------
    arr : 2-D array
        Image frame to render.
    title : str
        Figure title.
    mask_visible : bool
        Initial visibility of the static-mask overlay.

    Returns
    -------
    figure, source, mapper, extras : tuple
        ``extras`` is a dict with keys: mask_source, new_mask_source,
        mask_renderer, new_mask_renderer, draw_tool, edit_tool,
        image_height, image_width, dyn_source, dyn_renderer.
    """
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import (
        ColorBar, ColumnDataSource, CustomJS,
        LinearColorMapper, LogColorMapper,
        PolyDrawTool, PolyEditTool,
    )

    h, w = arr.shape
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 1)), 1e-3)
        vhi = float(np.percentile(finite, 99.5))
        vhi = max(vhi, vlo + 1.0)
        mapper = LogColorMapper(palette="Turbo256", low=vlo, high=vhi)
    else:
        lo = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0
        hi = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1
        mapper = LinearColorMapper(palette="Greys256", low=lo, high=hi)

    pw = 600
    ph = 600

    display = np.where(np.isfinite(arr), arr, 0).astype(np.float32)
    source = ColumnDataSource(data=dict(image=[display], x=[0], y=[0], dw=[w], dh=[h]))

    p = bk_figure(
        title=title, width=pw, height=ph,
        x_range=(0, w), y_range=(0, h),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
        sizing_mode="stretch_both",
    )
    p.image(image="image", x="x", y="y", dw="dw", dh="dh",
            color_mapper=mapper, source=source)
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, width=12), "right")
    p.xaxis.axis_label = "col (px)"
    p.yaxis.axis_label = "row (px)"

    # ----- Polygon mask overlay -----
    mask_source = ColumnDataSource(
        data=dict(xs=[], ys=[], name=[], kind=[],
                  fill_color=[], line_color=[]),
    )
    mask_renderer = p.patches(
        xs="xs", ys="ys",
        fill_color="fill_color", fill_alpha=0.25,
        line_color="line_color", line_width=2,
        source=mask_source,
    )
    # ----- New polygons overlay -----
    new_mask_source = ColumnDataSource(data=dict(xs=[], ys=[]))
    new_mask_renderer = p.patches(
        xs="xs", ys="ys",
        fill_color="#00e5ff", fill_alpha=0.30,
        line_color="#0066ff", line_width=2,
        source=new_mask_source,
    )
    vertex_source = ColumnDataSource(data=dict(x=[], y=[]))
    vertex_renderer = p.scatter(
        x="x", y="y", source=vertex_source,
        size=8, color="white", line_color="black", line_width=1,
    )
    draw_tool = PolyDrawTool(renderers=[new_mask_renderer], num_objects=200)
    edit_tool = PolyEditTool(renderers=[mask_renderer, new_mask_renderer],
                             vertex_renderer=vertex_renderer)
    p.add_tools(draw_tool, edit_tool)

    # ----- Tap-position debug readout -----
    tap_cb = CustomJS(
        args=dict(),
        code="""
        const x = cb_obj.x;
        const y = cb_obj.y;
        const ev = (cb_obj.event_name || 'tap');
        console.log('[mask-debug]', ev, 'x=', x.toFixed(1), 'y=', y.toFixed(1));
        """,
    )
    p.js_on_event("tap", tap_cb)
    p.js_on_event("doubletap", tap_cb)

    # ----- Dynamic-mask overlay -----
    empty_rgba = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint32)
    dyn_source = ColumnDataSource(
        data=dict(image=[empty_rgba], x=[0], y=[0], dw=[w], dh=[h]),
    )
    dyn_renderer = p.image_rgba(
        image="image", x="x", y="y", dw="dw", dh="dh", source=dyn_source,
    )
    dyn_renderer.visible = False

    mask_renderer.visible = mask_visible

    extras = {
        "mask_source": mask_source,
        "new_mask_source": new_mask_source,
        "mask_renderer": mask_renderer,
        "new_mask_renderer": new_mask_renderer,
        "draw_tool": draw_tool,
        "edit_tool": edit_tool,
        "image_height": h,
        "image_width": w,
        "dyn_source": dyn_source,
        "dyn_renderer": dyn_renderer,
    }
    return p, source, mapper, extras


def update_image_data(
    fig, source, mapper, arr: np.ndarray, title: str,
    cached_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Update an existing Bokeh image figure in-place.

    Returns the new image shape ``(h, w)`` for cache bookkeeping.
    """
    from bokeh.models import LogColorMapper, LinearColorMapper

    h, w = arr.shape
    display = np.where(np.isfinite(arr), arr, 0).astype(np.float32)

    if cached_shape != (h, w):
        fig.x_range.start = 0
        fig.x_range.end = w
        fig.y_range.start = 0
        fig.y_range.end = h

    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 1)), 1e-3)
        vhi = float(np.percentile(finite, 99.5))
        vhi = max(vhi, vlo + 1.0)
        mapper.low = vlo
        mapper.high = vhi
    else:
        lo = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0
        hi = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1
        mapper.low = lo
        mapper.high = hi

    source.data = dict(image=[display], x=[0], y=[0], dw=[w], dh=[h])
    fig.title.text = title
    return h, w
