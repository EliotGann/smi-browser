"""2D result plotting — transmission q-vs-chi and GI qxy-vs-qz."""
from __future__ import annotations

import numpy as np


def plot_2d_transmission(result, frame_idx=None):
    """Build a Bokeh figure for q-vs-chi.

    Returns ``(figure, x_arr, y_arr, display, x_label, y_label, title)``.
    The caller is responsible for attaching cut overlays and assigning
    to the widget.
    """
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar

    qchi = result.merged_qchi
    if frame_idx is not None and "frame" in qchi.dims:
        img = qchi["intensity"].isel(frame=frame_idx).values
        title = f"q vs χ — frame {frame_idx}"
    else:
        img = qchi["intensity"].values
        title = "q vs χ (merged)"
    q = qchi["q"].values if "q" in qchi.coords else np.arange(img.shape[-1])
    chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(img.shape[0])

    if img.shape == (len(q), len(chi)):
        img = img.T

    display = np.where(np.isfinite(img), img, 0).astype(np.float32)
    finite = img[np.isfinite(img) & (img > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 2)), 1e-6)
        vhi = float(np.percentile(finite, 99.5))
        mapper = LogColorMapper(palette="Turbo256", low=vlo, high=max(vhi, vlo * 2))
    else:
        mapper = LinearColorMapper(
            palette="Greys256",
            low=float(np.nanmin(display)),
            high=max(float(np.nanmax(display)), 1),
        )

    q0, q1 = float(q.min()), float(q.max())
    c0, c1 = float(chi.min()), float(chi.max())
    p = bk_figure(
        title=title, height=500,
        sizing_mode="stretch_width",
        x_range=(q0, q1), y_range=(c0, c1),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.image(image=[display], x=q0, y=c0, dw=q1 - q0, dh=c1 - c0, color_mapper=mapper)
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, width=12), "right")
    p.xaxis.axis_label = "q (nm⁻¹)"
    p.yaxis.axis_label = "χ (°)"

    return p, q, chi, display, "q (nm⁻¹)", "χ (°)", title


def plot_2d_gi(gi_result, frame_idx=None):
    """Build a Bokeh figure for qxy-vs-qz.

    Returns ``(figure, x_arr, y_arr, display, x_label, y_label, title)``.
    """
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar

    if frame_idx is not None and frame_idx < len(gi_result.frames):
        img = gi_result.frames[frame_idx]
        ai = gi_result.alpha_i_deg[frame_idx]
        title = f"qxy vs qz — frame {frame_idx} (α_i={ai:.3f}°)"
    else:
        img = gi_result.summed
        title = "qxy vs qz (summed)"

    qxy = gi_result.qxy_grid
    qz = gi_result.qz_grid

    display = np.where(np.isfinite(img), img, 0).astype(np.float32)
    finite = img[np.isfinite(img) & (img > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 2)), 1e-6)
        vhi = float(np.percentile(finite, 99.5))
        mapper = LogColorMapper(palette="Viridis256", low=vlo, high=max(vhi, vlo * 2))
    else:
        mapper = LinearColorMapper(
            palette="Greys256",
            low=float(np.nanmin(display)),
            high=max(float(np.nanmax(display)), 1),
        )

    x0, x1 = float(qxy.min()), float(qxy.max())
    y0, y1 = float(qz.min()), float(qz.max())
    p = bk_figure(
        title=title, height=500,
        sizing_mode="stretch_width",
        x_range=(x0, x1), y_range=(y0, y1),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.image(image=[display.T], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, width=12), "right")
    p.xaxis.axis_label = "q_xy (nm⁻¹)"
    p.yaxis.axis_label = "q_z (nm⁻¹)"

    return p, qxy, qz, display.T, "q_xy (nm⁻¹)", "q_z (nm⁻¹)", title
