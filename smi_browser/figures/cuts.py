"""Cross-section (cuts) helpers — projecting cuts to/from Bokeh sources.

The numerical cross-section math (:func:`compute_cross_section`) now lives in
:mod:`smi_tiled.derived.linecuts`; this module re-exports it via a thin
wrapper so existing browser code keeps importing from ``smi_browser.figures.cuts``.
The Bokeh-glyph helpers below stay in the browser because they are UI-specific.
"""
from __future__ import annotations

import numpy as np

_CUT_FILL = {"h": "#1f77b4", "v": "#d62728"}
_CUT_LINE = {"h": "#0a3a6e", "v": "#7a1414"}


def cuts_to_source_data(cuts: list[dict], x, y) -> dict:
    """Project a persisted cuts list into Bokeh Rect glyph columns.

    Parameters
    ----------
    cuts : list of dicts with keys kind, center, width
    x, y : 1-D arrays (q / chi or qxy / qz axes)
    """
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return dict(x=[], y=[], width=[], height=[],
                    kind=[], fill_color=[], line_color=[])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    xw = xmax - xmin
    yh = ymax - ymin
    cx, cy, w, h, kinds, fc, lc = [], [], [], [], [], [], []
    for cut in cuts:
        k = cut["kind"]
        if k == "h":
            cx.append(xc)
            cy.append(float(cut["center"]))
            w.append(xw)
            h.append(float(cut["width"]))
        else:
            cx.append(float(cut["center"]))
            cy.append(yc)
            w.append(float(cut["width"]))
            h.append(yh)
        kinds.append(k)
        fc.append(_CUT_FILL[k])
        lc.append(_CUT_LINE[k])
    return dict(x=cx, y=cy, width=w, height=h,
                kind=kinds, fill_color=fc, line_color=lc)


def source_data_to_cuts(data: dict, x, y) -> list[dict]:
    """Inverse of ``cuts_to_source_data`` — also classifies newly drawn
    boxes by aspect ratio against the plot extents."""
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return []
    xspan = float(np.max(x) - np.min(x)) or 1.0
    yspan = float(np.max(y) - np.min(y)) or 1.0
    cuts = []
    kinds = list(data.get("kind", []))
    xs = list(data.get("x", []))
    ys = list(data.get("y", []))
    ws = list(data.get("width", []))
    hs = list(data.get("height", []))
    for i, (cx, cy, w, h) in enumerate(zip(xs, ys, ws, hs)):
        k = kinds[i] if i < len(kinds) and kinds[i] else None
        if not k:
            k = "h" if (w / xspan) >= (h / yspan) else "v"
        if k == "h":
            cuts.append({"kind": "h",
                         "center": float(cy),
                         "width": float(abs(h))})
        else:
            cuts.append({"kind": "v",
                         "center": float(cx),
                         "width": float(abs(w))})
    return cuts


def compute_cross_section(cut: dict, x, y, image,
                          x_label: str = "", y_label: str = ""):
    """See :func:`smi_tiled.derived.linecuts.compute_cross_section`.

    Re-exported here so existing browser imports keep resolving while the
    canonical implementation lives in ``smi-tiled``.
    """
    from smi_tiled.derived.linecuts import compute_cross_section as _impl
    return _impl(cut, x, y, image, x_label=x_label, y_label=y_label)


def format_cut_label(i: int, cut: dict) -> str:
    """Label string for a cut."""
    arrow = "─" if cut["kind"] == "h" else "│"
    return f"{arrow} #{i + 1}: c={cut['center']:.3g}, Δ={cut['width']:.3g}"