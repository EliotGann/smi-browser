"""Cross-section (cuts) helpers — projecting cuts to/from Bokeh sources."""
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
    """Compute a 1-D cross section through a 2-D image.

    Returns ``(axis, intensity, axis_label)`` or ``None``.
    """
    if x is None or y is None or image is None:
        return None
    c = float(cut["center"])
    w = float(cut["width"]) or 0.0
    half = max(w / 2.0, 0.0)
    if cut["kind"] == "h":
        mask = (y >= c - half) & (y <= c + half)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(y - c)))
            section = image[idx, :].astype(float)
        else:
            section = np.nanmean(image[mask, :], axis=0)
        return x, section, x_label
    mask = (x >= c - half) & (x <= c + half)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(x - c)))
        section = image[:, idx].astype(float)
    else:
        section = np.nanmean(image[:, mask], axis=1)
    return y, section, y_label


def format_cut_label(i: int, cut: dict) -> str:
    """Label string for a cut."""
    arrow = "─" if cut["kind"] == "h" else "│"
    return f"{arrow} #{i + 1}: c={cut['center']:.3g}, Δ={cut['width']:.3g}"
