"""Polygon mask projection helpers for Bokeh overlay ↔ smi-tiled mask dicts."""
from __future__ import annotations

from smi_tiled import defaults as smid


def classify_detector_field(field: str) -> str | None:
    """Classify a detector field name as ``'saxs'`` / ``'waxs'`` / ``None``."""
    return smid.classify_detector_field(field)


def default_mask_path_for(detector: str):
    """Resolve the bundled default mask path for a detector."""
    return (smid.default_waxs_mask_path() if detector == "waxs"
            else smid.default_saxs_mask_path())


def normalized_mask_to_xs_ys(
    mask: dict,
    field: str | None = None,
    raw_shape: tuple[int, int] | None = None,
) -> tuple[list, list, list, list]:
    """Project a normalized mask dict into Bokeh patches columns.

    Input is the dict returned by ``smid.load_mask_polygons`` (always shaped
    ``{image_shape, static_regions, beamstops}`` in raw detector indexing).
    When ``field`` and ``raw_shape`` are given, vertices are transformed
    via ``smid.orient_polygon_xy`` so the polygons overlay the *displayed*
    image.
    """
    xs, ys, names, kinds = [], [], [], []
    detector = classify_detector_field(field) if field else None

    def _xy(col_raw: float, row_raw: float) -> tuple[float, float]:
        if detector is not None and raw_shape is not None:
            return smid.orient_polygon_xy(col_raw, row_raw, detector, raw_shape)
        return float(col_raw), float(row_raw)

    for kind, bucket in (("static", "static_regions"), ("beamstop", "beamstops")):
        for name, verts in (mask.get(bucket) or {}).items():
            if not verts:
                continue
            xl, yl = [], []
            for v in verts:
                x, y = _xy(v[0], v[1])
                xl.append(x)
                yl.append(y)
            xs.append(xl)
            ys.append(yl)
            names.append(str(name))
            kinds.append(kind)
    return xs, ys, names, kinds


def xs_ys_to_normalized_mask(
    xs, ys, names, kinds,
    field: str | None = None,
    raw_shape: tuple[int, int] | None = None,
) -> dict:
    """Inverse projection — build a normalized mask dict from Bokeh columns."""
    out: dict = {
        "image_shape": list(raw_shape) if raw_shape else None,
        "static_regions": {},
        "beamstops": {},
    }
    detector = classify_detector_field(field) if field else None
    counters = {"static": 0, "beamstop": 0}
    for px, py, name, kind in zip(xs, ys, names, kinds):
        if not px or not py:
            continue
        verts = []
        for x, y in zip(px, py):
            if detector is not None and raw_shape is not None:
                col, row = smid.orient_polygon_xy_inverse(
                    x, y, detector, raw_shape)
            else:
                col, row = float(x), float(y)
            verts.append([col, row])
        if not name:
            counters[kind] += 1
            name = f"{kind}_{counters[kind]}"
        bucket = "static_regions" if kind == "static" else "beamstops"
        out[bucket][name] = verts
    return out
