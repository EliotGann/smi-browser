"""
smi_app.py — SMI Tiled Browser with PyHyperScattering integration.

Launch with:
    pixi run panel serve smi_app.py --show

Features
--------
- Full-text search across the SMI tiled catalog (max 100 results per page)
- Metadata-only browse: sample_name, plan_name, institution, n_steps,
  detectors (SAXS / WAXS), exit_status — no array I/O during search
- Click a row → lazy detail tabs (metadata, primary scalars, baseline, images)
- Process tab runs PyHyperScattering reduce_smi_combined with tunable params
- Scan Collection: gather processed results, detect varying parameters,
  compare I(q) overlays, stack into xarray Datasets for parameter sweeps

Scan Collection Scheme
----------------------
*Within-scan* variation (e.g., waxs_arc angle per frame) is already handled by
reduce_smi_combined, which integrates all frames into merged q-chi and I(q).

*Between-scan* variation (sample, temperature, exposure, ...) is tracked by the
ScanCollection.  It auto-detects which metadata fields differ across scans
(``varying_parameters``), enables I(q) overlay comparisons, and can stack
merged results into a multi-dimensional xarray Dataset (``stack_iq``) for
downstream analysis.
"""

from __future__ import annotations

import datetime
import gc
import logging
import threading
import time
from typing import Any

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import panel as pn

import tiled_browser as tb
from batch_processor import BatchProcessor
from live_stream import LiveStreamManager
from smi_browser import nsls2api
from smi_browser.cache import (
    ScanCache, cache_path, get_or_fetch_scalars, get_or_fetch_image_frame,
    prune_lock_table,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

pn.extension("tabulator", sizing_mode="stretch_width", notifications=True)

# ---------------------------------------------------------------------------
# Bokeh compat patch: BokehJS 3.9 sends visual properties as
# {"type": "value", "value": X} but the Python-side set_from_json expects
# a plain scalar.  Unwrap silently to avoid noisy ValueErrors on every
# PATCH-DOC round-trip (line_width, line_alpha, etc.).
# ---------------------------------------------------------------------------
from bokeh.core.property.descriptors import PropertyDescriptor as _PD

_orig_set_from_json = _PD.set_from_json

def _patched_set_from_json(self, obj, value, *, setter=None):
    if isinstance(value, dict) and value.get("type") == "value" and "value" in value:
        value = value["value"]
    return _orig_set_from_json(self, obj, value, setter=setter)

_PD.set_from_json = _patched_set_from_json
# ---------------------------------------------------------------------------

# UI modules must be imported AFTER pn.extension() so that Tabulator widgets
# created inside wire() register the JS extension correctly.
from smi_browser.ui import collection as _coll_mod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAGE_SIZE = 25

# Canonical defaults & helpers are owned by PyHyperScattering.smi_defaults.
# Importing it triggers PyHyperScattering/__init__.py once (which pulls in
# the heavy integrators), but we need LOADER_DEFAULTS at widget-construct
# time anyway, so eat the cost here rather than mirroring constants.
from PyHyperScattering import smi_defaults as smid
from PyHyperScattering.SMISWAXSIntegrator import (
    clear_geometry_cache,
    geometry_cache_info,
)

DEFAULT_TILED_URI = smid.DEFAULT_TILED_URI
DEFAULT_CATALOG = smid.DEFAULT_CATALOG
DEFAULT_SAXS_MASK_NAME = smid.DEFAULT_SAXS_MASK_NAME
DEFAULT_WAXS_MASK_NAME = smid.DEFAULT_WAXS_MASK_NAME

# Detector-name classification (kept as module aliases for legacy usage).
SAXS_DETECTOR_NAMES = smid.SAXS_DETECTOR_NAMES
WAXS_DETECTOR_NAMES = smid.WAXS_DETECTOR_NAMES

# Loader-side calibrated defaults (frozen dataclass exposed by PyHyper).
_LD = smid.LOADER_DEFAULTS
DEFAULT_SAXS_ROW_DELTA = _LD.saxs_row_delta_px
DEFAULT_SAXS_COL_DELTA = _LD.saxs_col_delta_px
DEFAULT_WAXS_ROW_DELTA = _LD.waxs_row_delta_px
DEFAULT_WAXS_COL_DELTA = _LD.waxs_col_delta_px
DEFAULT_SAXS_DIST_DELTA = _LD.saxs_distance_delta_mm

# Processing defaults  (UI-side; these mirror the upstream PyHyper defaults
# so the widgets show meaningful numbers even before any override.  When a
# widget value still equals its default, _on_process passes None so the
# upstream loader supplies its own calibrated default.)
DEFAULT_N_Q = 2000          # PyHyper default is 1000; smi-browser used 2000
DEFAULT_N_CHI = 360
DEFAULT_SAXS_MASK = ""      # empty → use bundled default from PyHyper
DEFAULT_WAXS_MASK = ""
DEFAULT_DEZINGER = 3000.0
DEFAULT_DEZINGER_KERNEL = 5
DEFAULT_PIXEL_SPLITTING = 1
DEFAULT_INCIDENT_ANGLE = 0.0
DEFAULT_THETA_OFFSET = -0.5
DEFAULT_N_QXY = 500
DEFAULT_N_QZ = 500
DEFAULT_SOLID_ANGLE = True
DEFAULT_SAXS_AGBH_RING_ORDER = 5
DEFAULT_SAXS_Q_MARGIN_FRACTION = 0.01
DEFAULT_WAXS_BEAM_COL_PER_ARC_DEG = 0.08
DEFAULT_BEAMSTOP_MAX_ABS_ARC_DEG = 15.0
DEFAULT_DYN_SHADOW_BEAM_VISIBLE_DEG = 14.5
DEFAULT_DYN_SHADOW_CLEAR_EDGE_DEG = 18.0
DEFAULT_DYN_APER_AGBH_RING_ORDER = 5
DEFAULT_DYN_APER_Q_MARGIN_FRACTION = 0.01
DEFAULT_WAXS_ENERGY_KEV = 16.1
DEFAULT_WAXS_SAMPLE_DIST_MM = 273.0
DEFAULT_WAXS_PIXEL_SIZE_MM = 0.172
DEFAULT_WAXS_BEAM_CENTER_ROW = 217.0
DEFAULT_WAXS_BEAM_CENTER_COL = 319.0
DEFAULT_WAXS_THETA_ZERO_DEG = 0.0
DEFAULT_WAXS_SAMPLE_OFFSET_X_MM = 0.0
DEFAULT_WAXS_SAMPLE_OFFSET_Z_MM = 0.0
DEFAULT_WAXS_Q_HORIZONTAL_SIGN = -1.0
DEFAULT_WAXS_Q_VERTICAL_SIGN = -1.0
DEFAULT_WAXS_ROTATION_K = 3

# Common metadata keys for Like search (user can also type custom keys)
COMMON_SEARCH_KEYS = [
    "sample_name",
    "plan_name",
    "data_session",
    "proposal.first_name",
    "proposal.last_name",
    "project_name",
    "institution",
    "scan_id",
    "uid",
    "detectors",
]

RESULT_COLS = [
    "scan_id", "n_steps", "sample_name", "plan_name",
    "data_session", "detectors", "exit_status", "time", "uid",
]

_EMPTY_DF = pd.DataFrame(columns=RESULT_COLS)


# ---------------------------------------------------------------------------
# Enhanced run summary  (metadata-only, adds detector + institution fields)
# ---------------------------------------------------------------------------

def enhanced_summary(run) -> dict:
    """
    Build a lightweight summary from a tiled run node.
    Only reads .metadata — zero array I/O.  Extends tb.run_summary with
    detector classification and institution info.
    """
    md = run.metadata
    start = md.get("start", {})
    stop = md.get("stop", {})

    t0 = start.get("time")
    time_str = (
        datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M")
        if t0 else "?"
    )

    # Detectors
    det_list = start.get("detectors", [])
    if isinstance(det_list, str):
        det_list = [det_list]
    det_lower = {d.lower() for d in det_list}
    has_saxs = bool(det_lower & SAXS_DETECTOR_NAMES)
    has_waxs = bool(det_lower & WAXS_DETECTOR_NAMES)
    if has_saxs and has_waxs:
        det_str = "SAXS+WAXS"
    elif has_saxs:
        det_str = "SAXS"
    elif has_waxs:
        det_str = "WAXS"
    else:
        det_str = ", ".join(det_list) if det_list else "?"

    # Steps
    num_events = stop.get("num_events", {})
    if isinstance(num_events, dict):
        n_steps = num_events.get("primary", "?")
    else:
        n_steps = num_events

    # Institution / data session
    institution = start.get(
        "institution",
        start.get("data_session", start.get("proposal_id", "?")),
    )

    exit_status = stop.get("exit_status", "?")

    return {
        "uid":          start.get("uid", "?"),
        "scan_id":      start.get("scan_id", "?"),
        "time":         time_str,
        "plan_name":    start.get("plan_name", "?"),
        "sample_name":  start.get(
            "sample_name",
            start.get("sample", start.get("Sample", "?")),
        ),
        "exit_status":  exit_status,
        "n_steps":      n_steps,
        "institution":  str(institution),
        "detectors":    det_str,
        "has_saxs":     has_saxs,
        "has_waxs":     has_waxs,
        "detector_list": det_list,
    }


from smi_browser.models.collection import ScanCollection


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_cat = None          # migration catalog (for data access / processing)
_collection = ScanCollection()

_state = {
    "unified_filters": [],  # list of (type, key, value) tuples
    "page":         0,
    "page_size":    PAGE_SIZE,
    "total":        0,
    "selected_uid": None,
}

_detail_cache = {
    "uid":              None,
    "run":              None,
    "summary":          None,
    "primary_loaded":   False,
    "baseline_loaded":  False,
    "images_loaded":    False,
    "primary_info":     None,
    "primary_dataset":  None,
}

_last_result = {"result": None, "params": None}


def _get_cat():
    """Return the migration catalog (used for data access / processing)."""
    global _cat
    if _cat is None:
        t0 = time.time()
        _cat = tb.connect(DEFAULT_TILED_URI, DEFAULT_CATALOG)
        log.info("tiled connect (migration) took %.2fs", time.time() - t0)
    return _cat


def _n_pages() -> int:
    return max(1, (_state["total"] + _state["page_size"] - 1) // _state["page_size"])


# ---------------------------------------------------------------------------
# Search / data helpers
# ---------------------------------------------------------------------------

SEARCH_TYPES = ["Anywhere", "Text in field", "Contains", "Exact"]
SEARCH_TYPE_MAP = {"Anywhere": "anywhere", "Text in field": "like", "Contains": "contains", "Exact": "exact"}


def _selected_cycle_value() -> str | None:
    """Return selected cycle, or None when cycle filtering is disabled."""
    cycle_widget = globals().get("w_proposal_cycle")
    if cycle_widget is None:
        return None
    cycle = str(cycle_widget.value or "").strip()
    if not cycle or cycle in {"All cycles", "(loading…)", "(unavailable)"}:
        return None
    return cycle


def _with_cycle_filter(
    unified_filters: list[tuple[str, str, str]] | None,
) -> list[tuple[str, str, str]]:
    """Append an implicit exact cycle filter unless one is already present."""
    filters = list(unified_filters or [])
    cycle = _selected_cycle_value()
    if not cycle:
        return filters

    for ftype, key, _value in filters:
        if ftype.strip().lower() != "exact":
            continue
        if key.strip().lower() in {"cycle", "start.cycle"}:
            return filters

    filters.append(("exact", "cycle", cycle))
    return filters


def _collect_unified_filters() -> list[tuple[str, str, str]]:
    """Read filter rows and return (type, key, value) tuples."""
    filters = []
    for row in _filter_rows:
        ftype = SEARCH_TYPE_MAP.get(row["type"].value, "like")
        key = row["key"].value.strip() if ftype != "anywhere" else ""
        val = row["val"].value.strip()
        if val:
            filters.append((ftype, key, val))
    return filters


def _fetch_page() -> pd.DataFrame:
    """Metadata-only fetch via fast REST API (single HTTP round-trip)."""
    unified = _with_cycle_filter(_state["unified_filters"])
    offset = _state["page"] * _state["page_size"]
    limit = _state["page_size"]

    summaries, total = tb.fetch_page_fast(
        _get_cat(), unified_filters=unified,
        offset=offset, limit=limit,
    )
    # Update total from REST response (avoids a separate count query)
    _state["total"] = total

    # Persist unfiltered count for next startup
    if total > 0 and not unified:
        tb.save_cached_count(total)

    if not summaries:
        return _EMPTY_DF.copy()

    df = pd.DataFrame(summaries)
    for col in RESULT_COLS:
        if col not in df.columns:
            df[col] = "?"
    return df[RESULT_COLS].fillna("?")


def _count_total() -> int:
    # Now handled inside _fetch_page via REST meta.count
    # This is kept as fallback but should rarely be called separately
    return _state.get("total", 0)


def _scalars_to_dataframe(scalar_data: dict) -> pd.DataFrame:
    """Convert a {field: ndarray} dict of scalars into a DataFrame."""
    if not scalar_data:
        return pd.DataFrame()

    columns = {}
    max_len = 0
    for key, arr in scalar_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = np.array([arr.item()])
        if arr.ndim != 1:
            continue
        columns[key] = arr
        max_len = max(max_len, len(arr))

    if not columns:
        return pd.DataFrame()

    data = {}
    for key, arr in columns.items():
        if len(arr) < max_len:
            padded = np.full(
                max_len, np.nan,
                dtype=float if np.issubdtype(arr.dtype, np.number) else object,
            )
            padded[:len(arr)] = arr
            data[key] = padded
        else:
            data[key] = arr
    return pd.DataFrame(data)


def _scalar_stream_to_frame(run, stream: str, *, uid: str | None = None,
                            dataset=None) -> pd.DataFrame:
    """Read scalar fields from a stream into a DataFrame.

    If ``uid`` is provided, the disk cache (``smi_browser.cache.ScanCache``)
    is consulted first and populated on a miss so subsequent reads — within
    this session and across restarts — avoid the tiled round-trip.
    """
    if uid:
        scalar_data = get_or_fetch_scalars(
            uid, stream,
            lambda: tb.fetch_scalars(run, stream, _dataset=dataset),
        )
    else:
        scalar_data = tb.fetch_scalars(run, stream, _dataset=dataset)
    return _scalars_to_dataframe(scalar_data)


def _config_to_dataframe(run) -> pd.DataFrame:
    """Build a single-row DataFrame from the primary stream configuration."""
    config_data = tb.fetch_primary_config(run)
    if not config_data:
        return pd.DataFrame()
    # Build single-row DataFrame: each config field is a column
    return pd.DataFrame([config_data])


def _detector_for_field(field: str) -> str | None:
    """Classify a detector field name as ``'saxs'`` / ``'waxs'`` / ``None``."""
    return smid.classify_detector_field(field)


def _orient_frame(arr: np.ndarray, field: str) -> np.ndarray:
    """Re-orient detector frames for display via the canonical PyHyper transform."""
    detector = _detector_for_field(field)
    if detector is None:
        return arr
    return smid.orient_frame_for_display(arr, detector)


# ---------------------------------------------------------------------------
# Polygon mask helpers (overlay + edit on the Explore image preview)
#
# All schema parsing and orientation math lives in PyHyperScattering's
# smi_defaults module.  The browser keeps only the thin projection between
# the *normalized* mask dict (PyHyper's canonical shape) and Bokeh's
# (xs, ys, names, kinds) ColumnDataSource columns.
# ---------------------------------------------------------------------------


def _normalized_mask_to_xs_ys(
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
    detector = _detector_for_field(field) if field else None

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


def _xs_ys_to_normalized_mask(
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
    detector = _detector_for_field(field) if field else None
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


def _default_mask_path_for(detector: str):
    """Resolve the bundled default mask path for a detector."""
    return (smid.default_waxs_mask_path() if detector == "waxs"
            else smid.default_saxs_mask_path())


def _normalize_mask_path(path_str: str | None) -> str | None:
    """Expand user/env vars and make custom mask paths absolute."""
    if not path_str:
        return None
    raw = str(path_str).strip()
    if not raw:
        return None

    import os
    from pathlib import Path

    p = Path(os.path.expandvars(raw)).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p)


def _thumbnail_figure(arr, title):
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import (
        ColorBar, ColumnDataSource, LinearColorMapper, LogColorMapper,
        PolyDrawTool, PolyEditTool,
    )

    # Ensure 2-D — squeeze singleton dims, take first sub-frame if needed
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = arr[0]

    h, w = arr.shape
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 1)), 1e-3)
        vhi = float(np.percentile(finite, 99.5))
        vhi = max(vhi, vlo + 1.0)
    else:
        vlo = float(np.nanmin(arr)) if np.any(np.isfinite(arr)) else 0
        vhi = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1
    palette = w_cs_cmap.value or "Turbo256"
    use_log = bool(w_cs_log.value)
    # If locked and we have a previous mapper, reuse its low/high
    prev_mapper = _image_cache.get("mapper")
    if w_cs_lock.value and prev_mapper is not None:
        try:
            vlo = float(prev_mapper.low)
            vhi = float(prev_mapper.high)
        except Exception:
            pass
    if use_log:
        mapper = LogColorMapper(palette=palette, low=max(vlo, 1e-9),
                                high=max(vhi, vlo * 1.0001))
    else:
        mapper = LinearColorMapper(palette=palette, low=vlo, high=vhi)

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

    # ----- Polygon mask overlay (always present; populated/hidden by callers)
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
    # ----- Separate "new polygons" overlay --------------------------------
    # PolyDrawTool draws into its own dedicated source so:
    #   (a) new polygons get a visually distinct colour (cyan), and
    #   (b) the tool isn't confused by the many pre-loaded static-mask
    #       polygons (which carry extra columns and can prevent draw
    #       activation in some Bokeh versions).
    new_mask_source = ColumnDataSource(data=dict(xs=[], ys=[]))
    new_mask_renderer = p.patches(
        xs="xs", ys="ys",
        fill_color="#00e5ff", fill_alpha=0.30,
        line_color="#0066ff", line_width=2,
        source=new_mask_source,
    )
    # Vertex source for PolyEditTool (shows draggable handles in edit mode)
    vertex_source = ColumnDataSource(data=dict(x=[], y=[]))
    vertex_renderer = p.scatter(
        x="x", y="y", source=vertex_source,
        size=8, color="white", line_color="black", line_width=1,
    )
    # PolyDrawTool only sees the *new* renderer; PolyEditTool can edit both.
    draw_tool = PolyDrawTool(renderers=[new_mask_renderer], num_objects=200)
    edit_tool = PolyEditTool(renderers=[mask_renderer, new_mask_renderer],
                             vertex_renderer=vertex_renderer)
    p.add_tools(draw_tool, edit_tool)

    # ----- Tap-position debug readout -------------------------------------
    # Reports the (x, y) of every tap on the figure, so we can verify the
    # PolyDrawTool is actually being reached by tap events.
    from bokeh.models import CustomJS

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

    # ----- Dynamic-mask overlay (rasterised PyHyper mask, optional) -------
    # Render as an RGBA image; alpha=0 for valid pixels, semi-transparent
    # red for invalid pixels.  Populated by _render_dynamic_mask().
    # image_rgba requires a 2D uint32 array (or a uint8 (h,w,4) view as uint32).
    empty_rgba = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint32)
    dyn_source = ColumnDataSource(
        data=dict(image=[empty_rgba], x=[0], y=[0], dw=[w], dh=[h]),
    )
    dyn_renderer = p.image_rgba(
        image="image", x="x", y="y", dw="dw", dh="dh", source=dyn_source,
    )
    dyn_renderer.visible = False

    # ----- Line-draw overlay (alignment tool) -----------------------------
    # A separate MultiLine renderer + PolyDrawTool with num_objects unbounded.
    # Each "line" is a polyline; we treat the first and last vertex of each
    # polyline as the line endpoints for stats / profile computation.
    line_source = ColumnDataSource(data=dict(xs=[], ys=[]))
    line_renderer = p.multi_line(
        xs="xs", ys="ys", source=line_source,
        line_color="#ff00aa", line_width=max(1.5, float(w_align_width.value)),
        line_alpha=0.9,
    )
    line_draw_tool = PolyDrawTool(renderers=[line_renderer], num_objects=20)
    p.add_tools(line_draw_tool)
    line_renderer.visible = bool(w_align_enable.value)

    # Watch the line source for changes (server-side callback)
    try:
        line_source.on_change("data", lambda attr, old, new: _update_line_analysis())
    except Exception as exc:
        log.warning("could not attach line_source change callback: %s", exc)

    _image_cache["line_source"] = line_source
    _image_cache["line_renderer"] = line_renderer
    _image_cache["line_draw_tool"] = line_draw_tool

    # Stash on image cache so the Explore controls can read/write it
    _image_cache["mask_source"] = mask_source
    _image_cache["new_mask_source"] = new_mask_source
    _image_cache["mask_renderer"] = mask_renderer
    _image_cache["new_mask_renderer"] = new_mask_renderer
    _image_cache["draw_tool"] = draw_tool
    _image_cache["edit_tool"] = edit_tool
    _image_cache["image_height"] = h
    _image_cache["image_width"] = w
    _image_cache["dyn_source"] = dyn_source
    _image_cache["dyn_renderer"] = dyn_renderer

    # Initial visibility follows the Show-mask checkbox state
    mask_renderer.visible = bool(w_mask_show.value)

    return p, source, mapper


def _update_image_in_place(arr, title):
    """Update the existing image figure in-place, preserving zoom/pan state."""
    from bokeh.models import LogColorMapper, LinearColorMapper

    fig = _image_cache.get("figure")
    source = _image_cache.get("source")
    mapper = _image_cache.get("mapper")

    if fig is None or source is None:
        # No existing figure — create fresh
        fig, source, mapper = _thumbnail_figure(arr, title)
        _image_cache["figure"] = fig
        _image_cache["source"] = source
        _image_cache["mapper"] = mapper
        _image_cache["fig_image_shape"] = tuple(arr.shape)
        w_image_thumb.object = fig
        # Sync color-scale widgets to the initial mapper range
        try:
            _cs_sync_widgets_to_range(float(mapper.low), float(mapper.high))
        except Exception:
            pass
        # Build initial histogram
        try:
            _build_histogram(arr)
        except Exception:
            pass
        # Auto-load the default mask if Show-mask is enabled (handles the
        # initial render and detector switches).  Defer to the next tick
        # so Bokeh has finished syncing the freshly-attached figure before
        # we push polygon data into its mask source.
        if w_mask_show.value:
            def _deferred_reload():
                try:
                    _on_mask_reload(None)
                except Exception as exc:
                    log.warning("auto mask reload failed: %s", exc)
            try:
                doc = pn.state.curdoc
                if doc is not None:
                    doc.add_next_tick_callback(_deferred_reload)
                else:
                    _deferred_reload()
            except Exception:
                _deferred_reload()
        return

    # Ensure 2-D
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = arr[0]

    h, w = arr.shape
    display = np.where(np.isfinite(arr), arr, 0).astype(np.float32)

    # Reset axis ranges only when the *underlying* image dimensions change
    # (e.g. switching detector/scan).  Comparing against fig.x_range.end
    # would also fire after the user zooms, clobbering their zoom state.
    cached_shape = _image_cache.get("fig_image_shape")
    if cached_shape != (h, w):
        fig.x_range.start = 0
        fig.x_range.end = w
        fig.y_range.start = 0
        fig.y_range.end = h
        _image_cache["fig_image_shape"] = (h, w)

    # Update color mapper range (respect lock toggle)
    if not w_cs_lock.value:
        vlo, vhi = _cs_finite_range(arr)
        _cs_apply_to_mapper(vlo, vhi)
        _cs_sync_widgets_to_range(vlo, vhi)

    # Update image data (keeps zoom/pan state)
    source.data = dict(image=[display], x=[0], y=[0], dw=[w], dh=[h])
    fig.title.text = title

    # Rebuild histogram + refresh any line profile from new frame data
    try:
        _build_histogram(arr)
    except Exception as exc:
        log.warning("histogram refresh failed: %s", exc)
    try:
        _update_line_analysis()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Widgets — Search  (unified stackable filters)
# ---------------------------------------------------------------------------

# Dynamic filter rows: each is a dict {type, key, val, suggest, remove}
_filter_rows: list[dict] = []
w_filter_column = pn.Column(sizing_mode="stretch_width")

# Cancellation flag — set by Reset to abort in-flight queries
_cancel = threading.Event()

# ---------------------------------------------------------------------------
# Filter state persistence across websocket reconnections
# ---------------------------------------------------------------------------
_initializing = True  # Suppress search triggers during initial startup

_FILTER_CACHE_KEY = "smi_browser_saved_filters"
_PAGE_CACHE_KEY = "smi_browser_saved_page"
_CYCLE_CACHE_KEY = "smi_browser_saved_cycle"
_DATASESSION_CACHE_KEY = "smi_browser_saved_datasession"
_PROJECT_CACHE_KEY = "smi_browser_saved_project"


def _save_filter_state():
    """Persist the current filters + page to pn.state.cache so they survive reconnects."""
    filters = _collect_unified_filters()
    pn.state.cache[_FILTER_CACHE_KEY] = filters
    pn.state.cache[_PAGE_CACHE_KEY] = _state["page"]


def _load_saved_filters() -> list[tuple[str, str, str]] | None:
    """Return saved filters from cache, or None if nothing was saved."""
    return pn.state.cache.get(_FILTER_CACHE_KEY)


def _load_saved_page() -> int:
    """Return saved page number from cache, defaulting to 0."""
    return pn.state.cache.get(_PAGE_CACHE_KEY, 0)


def _save_proposal_state():
    """Persist cycle/data-session/project selections for auto-reload."""
    pn.state.cache[_CYCLE_CACHE_KEY] = w_proposal_cycle.value
    pn.state.cache[_DATASESSION_CACHE_KEY] = w_proposal_select.value
    pn.state.cache[_PROJECT_CACHE_KEY] = w_proposal_project.value


def _load_saved_cycle() -> str | None:
    return pn.state.cache.get(_CYCLE_CACHE_KEY)


def _load_saved_datasession() -> str | None:
    return pn.state.cache.get(_DATASESSION_CACHE_KEY)


def _load_saved_project() -> str | None:
    return pn.state.cache.get(_PROJECT_CACHE_KEY)

# ---------------------------------------------------------------------------
# Debounced live count — lightweight count query while typing
# ---------------------------------------------------------------------------

_live_count_timer: threading.Timer | None = None
_LIVE_COUNT_DEBOUNCE_S = 0.4  # seconds after last keystroke before firing


def _collect_unified_filters_from_input() -> list[tuple[str, str, str]]:
    """Like _collect_unified_filters but reads value_input (live typing)."""
    filters = []
    for row in _filter_rows:
        ftype = SEARCH_TYPE_MAP.get(row["type"].value, "like")
        key = row["key"].value.strip() if ftype != "anywhere" else ""
        # Use value_input if the user is actively typing; fall back to
        # committed value (useful after Enter when value_input resets).
        val = row["val"].value_input or row["val"].value or ""
        val = val.strip()
        if val:
            filters.append((ftype, key, val))
    return filters


def _live_count_fire():
    """Execute a lightweight count-only query and update the status line."""
    if _cancel.is_set():
        return
    unified = _collect_unified_filters_from_input()
    # Don't fire if nothing is typed
    if not unified:
        return
    try:
        total = tb.count_fast(
            _get_cat(), unified_filters=_with_cycle_filter(unified),
        )
    except Exception:
        return  # silently ignore errors during live count
    if _cancel.is_set():
        return
    # Schedule the widget update on the Bokeh document thread to avoid
    # cross-thread issues with Panel/Bokeh state management.
    msg = f"~**{total} scan{'s' if total != 1 else ''}** matching (press Enter to load)"
    try:
        doc = pn.state.curdoc
        if doc is not None:
            doc.add_next_tick_callback(lambda: setattr(w_status, "object", msg))
        else:
            w_status.object = msg
    except Exception:
        w_status.object = msg


def _schedule_live_count(*_events):
    """Debounce: reset the timer on each keystroke, fire after the delay."""
    global _live_count_timer
    if _live_count_timer is not None:
        _live_count_timer.cancel()
    _live_count_timer = threading.Timer(_LIVE_COUNT_DEBOUNCE_S, _live_count_fire)
    _live_count_timer.daemon = True
    _live_count_timer.start()


def _make_filter_row(ftype: str = "Text in field", key: str = "", val: str = "") -> dict:
    """Create one unified filter row: type selector + key + value + suggest + remove."""
    w_type = pn.widgets.Select(
        name="Match", options=SEARCH_TYPES, value=ftype, width=90,
    )
    w_key = pn.widgets.AutocompleteInput(
        name="Key",
        options=COMMON_SEARCH_KEYS,
        restrict=False,
        placeholder="key…",
        value=key,
        min_characters=0,
        width=110,
    )
    w_val = pn.widgets.AutocompleteInput(
        name="Value", placeholder="search…",
        options=[],
        restrict=False,
        value=val, width=110,
        min_characters=0,
    )
    w_suggest = pn.widgets.Button(
        name="▼", button_type="light", width=28, height=28,
        description="Suggest values",
    )
    w_rm = pn.widgets.Button(name="✕", button_type="danger", width=28, height=28)

    # Hide/show key field based on type
    def _on_type_change(event):
        is_anywhere = event.new == "Anywhere"
        w_key.visible = not is_anywhere
        w_suggest.visible = not is_anywhere
    w_type.param.watch(_on_type_change, "value")
    # Apply initial visibility
    if ftype == "Anywhere":
        w_key.visible = False
        w_suggest.visible = False

    row_dict = {"type": w_type, "key": w_key, "val": w_val,
                "suggest": w_suggest, "remove": w_rm}

    def _remove(_event, rd=row_dict):
        if rd in _filter_rows:
            _filter_rows.remove(rd)
            _rebuild_filter_column()

    def _suggest_cb(_event, rd=row_dict):
        _cancel.clear()
        k = rd["key"].value.strip()
        if not k:
            w_status.object = "**Suggest:** enter a key first"
            return
        # Build unified filters from all *other* rows
        other_filters = []
        for r in _filter_rows:
            if r is rd:
                continue
            ft = SEARCH_TYPE_MAP.get(r["type"].value, "like")
            rk = r["key"].value.strip() if ft != "anywhere" else ""
            rv = r["val"].value.strip()
            if rv and (ft == "anywhere" or rk):
                other_filters.append((ft, rk, rv))
        w_status.object = f"*Fetching distinct values for **{k}**…*"
        try:
            vals = tb.distinct_values(
                _get_cat(),
                key=k,
                unified_filters=_with_cycle_filter(other_filters or None),
                counts=True,
                size_limit=tb.DISTINCT_SIZE_LIMIT,
            )
        except Exception as exc:
            if _cancel.is_set():
                return
            w_status.object = f"**Suggest error:** `{exc}`"
            return
        if _cancel.is_set():
            return
        if vals is None:
            w_status.object = (
                "**Suggest:** too many results or timed out – "
                "add more filters to narrow down first"
            )
            return
        suggestions = [str(v["value"]) for v in vals]
        rd["val"].options = suggestions
        n = len(suggestions)
        w_status.object = (
            f"**{n} distinct value{'s' if n != 1 else ''}** for **{k}** – "
            "type to filter suggestions"
        )

    w_rm.on_click(_remove)
    w_suggest.on_click(_suggest_cb)

    # Enter-to-search: AutocompleteInput fires 'value' on Enter/commit.
    def _on_enter(event):
        _do_search(0)
    w_val.param.watch(_on_enter, "value")

    # Live count while typing: fires lightweight count query after debounce.
    w_val.param.watch(_schedule_live_count, "value_input")
    w_key.param.watch(_schedule_live_count, "value_input")

    return row_dict


def _rebuild_filter_column():
    """Re-render the filter column from the current _filter_rows list."""
    rows = []
    for rd in _filter_rows:
        rows.append(
            pn.Column(
                pn.Row(rd["type"], rd["key"], margin=(0, 0)),
                pn.Row(rd["val"],
                       pn.Column(pn.Spacer(height=15), rd["remove"]),
                       margin=(0, 0)),
                margin=(0, 0, 5, 0),
            )
        )
    w_filter_column.objects = rows


def _add_filter(_event=None, ftype: str = "Text in field", key: str = "", val: str = ""):
    rd = _make_filter_row(ftype, key, val)
    _filter_rows.append(rd)
    _rebuild_filter_column()


w_btn_add_filter = pn.widgets.Button(name="+ Add filter", button_type="default", width=110)
w_btn_add_filter.on_click(_add_filter)

w_btn_search = pn.widgets.Button(name="Search", button_type="primary", width=80)
w_btn_reset = pn.widgets.Button(name="Reset", button_type="warning", width=70)

w_status = pn.pane.Markdown("*Connecting to catalog…*", width=700)
w_search_spinner = pn.indicators.LoadingSpinner(value=True, size=20, visible=True)
w_btn_first = pn.widgets.Button(name="⏮", width=40, button_type="light")
w_btn_prev = pn.widgets.Button(name="◀", width=40, button_type="light")
w_btn_next = pn.widgets.Button(name="▶", width=40, button_type="light")
w_btn_last = pn.widgets.Button(name="⏭", width=40, button_type="light")
w_page_info = pn.pane.Markdown("–/–", width=80)

w_table = pn.widgets.Tabulator(
    value=_EMPTY_DF.copy(),
    pagination=None,
    selectable=1,
    show_index=False,
    sizing_mode="stretch_width",
    height=600,
    configuration={"rowHeight": 22, "layout": "fitColumns",
                    "selectableRows": True},
    hidden_columns=["uid"],
    widths={"scan_id": 55, "n_steps": 40, "sample_name": 110, "plan_name": 60,
            "data_session": 65, "detectors": 75, "exit_status": 50, "time": 80},
    text_align="left",
    header_align="left",
    disabled=True,
    stylesheets=[
        ":host .tabulator {font-size: 10px;}",
        ":host .tabulator .tabulator-header {font-size: 10px;}",
        ":host .tabulator .tabulator-row {cursor: pointer;}",
    ],
)

# Start with a fresh empty filter row — initial view is always unfiltered.
# Saved filters are available in pn.state.cache and applied when the user
# explicitly selects a proposal or adds filters.
_add_filter()

# ---------------------------------------------------------------------------
# Widgets — Detail panel
# ---------------------------------------------------------------------------

w_detail_title = pn.pane.Markdown("### Select a scan", width=550)

w_meta_json = pn.pane.JSON(
    object={},
    depth=2,
    theme="light",
    sizing_mode="stretch_both",
    margin=(5, 5),
)
w_primary_table = pn.widgets.Tabulator(
    value=pd.DataFrame(), show_index=False,
    sizing_mode="stretch_width", height=280,
    configuration={"layout": "fitColumns", "rowHeight": 22},
)
w_primary_status = pn.pane.Markdown("*Click tab to load.*")
w_primary_spinner = pn.indicators.LoadingSpinner(value=False, size=20, visible=False)
w_primary_x = pn.widgets.Select(
    name="X axis", options=[], width=180,
)
w_primary_y = pn.widgets.MultiChoice(
    name="Y axis (select columns)", options=[], width=400, max_items=6,
)
w_primary_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=300)
w_primary_sort_btn = pn.widgets.Button(
    name="Sort plot by table", button_type="default", width=140,
)
w_primary_fit = pn.widgets.Select(
    name="Fit", options=["None", "Gaussian", "Knife edge"], value="None", width=120,
)
w_primary_fit_btn = pn.widgets.Button(
    name="Fit", button_type="primary", width=60,
)
w_primary_fit_result = pn.pane.Markdown("", sizing_mode="stretch_width")

w_baseline_table = pn.widgets.Tabulator(
    value=pd.DataFrame(columns=["source", "field", "before", "after"]),
    show_index=False, sizing_mode="stretch_both",
    configuration={"layout": "fitColumns", "rowHeight": 22},
    header_filters={
        "field": {"type": "input", "func": "like", "placeholder": "filter…"},
        "source": {"type": "input", "func": "like", "placeholder": "filter…"},
    },
)
w_baseline_status = pn.pane.Markdown("*Click tab to load.*")

# Images tab — frame slider for browsing raw detector images
w_image_thumb = pn.pane.Bokeh(object=None, sizing_mode="stretch_both", min_height=400)
w_image_status = pn.pane.Markdown("")
w_image_spinner = pn.indicators.LoadingSpinner(value=False, size=20, visible=False)
w_image_slider = pn.widgets.IntSlider(
    name="Frame", start=0, end=1, value=0, step=1, sizing_mode="stretch_width",
)
w_image_field = pn.widgets.Select(
    name="Detector", options=[], width=200,
)
# Linked 1D primary plot for the Explore tab
w_explore_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=250)
# Container that hides itself when the line plot has no data, so we don't
# leave a tall empty band between the slider and the mask card.
w_explore_plot_container = pn.Column(w_explore_plot, sizing_mode="stretch_width", visible=False)


def _sync_explore_plot_visibility(*_):
    w_explore_plot_container.visible = w_explore_plot.object is not None


w_explore_plot.param.watch(_sync_explore_plot_visibility, "object")
w_explore_x = pn.widgets.Select(name="X", options=[], width=120)
w_explore_y = pn.widgets.MultiChoice(name="Y", options=[], width=300, max_items=6)

# Mask overlay / edit controls (Explore tab)
w_mask_show = pn.widgets.Checkbox(
    name="Show static mask", value=True, width=130,
)
w_mask_dynamic = pn.widgets.Checkbox(
    name="Show dynamic mask", value=False, width=150,
)
w_mask_edit = pn.widgets.Checkbox(
    name="Edit mode", value=False, width=100,
)
w_mask_path = pn.widgets.TextInput(
    name="Save / load path", value="", width=320,
    placeholder="e.g. ~/my_pil2M_mask.json",
)
w_btn_mask_reload = pn.widgets.Button(
    name="↻ Reload default", button_type="light", width=130,
    description="Reload the bundled PyHyper default mask for this detector.",
)
w_btn_mask_save = pn.widgets.Button(
    name="💾 Save", button_type="primary", width=80,
    description="Save the current overlay polygons to the path on the left.",
)
w_btn_mask_use = pn.widgets.Button(
    name="↪ Use in Process", button_type="success", width=140,
    description="Copy the saved-mask path into the Process tab's mask field.",
)
w_mask_status = pn.pane.Markdown("", width=600)

# ---------------------------------------------------------------------------
# Plotting tools — Color scale controls (Explore tab)
# ---------------------------------------------------------------------------
CS_PALETTES = [
    "Turbo256", "Viridis256", "Plasma256", "Inferno256",
    "Magma256", "Cividis256", "Greys256",
]
w_cs_cmap = pn.widgets.Select(
    name="Colormap", value="Turbo256", options=CS_PALETTES, width=130,
)
w_cs_log = pn.widgets.Checkbox(name="Log scale", value=True, width=100)
w_cs_lock = pn.widgets.Checkbox(
    name="Lock range across frames", value=False, width=200,
)
w_cs_range = pn.widgets.RangeSlider(
    name="Intensity range", start=0.0, end=1.0, value=(0.0, 1.0),
    step=0.01, sizing_mode="stretch_width", format="0.000",
)
w_cs_min = pn.widgets.FloatInput(name="Min", value=0.0, width=110)
w_cs_max = pn.widgets.FloatInput(name="Max", value=1.0, width=110)
w_cs_status = pn.pane.Markdown("", sizing_mode="stretch_width")
w_cs_hist = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=120)

# ---------------------------------------------------------------------------
# Plotting tools — Alignment / line-profile controls (Explore tab)
# ---------------------------------------------------------------------------
w_align_enable = pn.widgets.Checkbox(
    name="Enable line draw", value=False, width=140,
)
w_align_width = pn.widgets.IntInput(
    name="Width (px)", value=1, start=1, end=100, step=1, width=80,
)
w_btn_align_clear = pn.widgets.Button(
    name="✕ Clear lines", button_type="light", width=110,
)
w_align_stats = pn.pane.Markdown(
    "*Toggle 'Enable line draw' then click two points on the image to draw a line.*",
    sizing_mode="stretch_width",
)
w_align_profile = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=180)

_image_cache = {"field": None, "n_frames": 0, "dataset": None, "fields": [],
                "figure": None, "source": None, "mapper": None,
                "mask_source": None, "mask_renderer": None,
                "draw_tool": None, "edit_tool": None,
                "mask_image_shape": None,
                # Color-scale state
                "cs_suspend": False,
                # Alignment / line tools
                "line_source": None, "line_renderer": None, "line_draw_tool": None}


# ----- Color scale helpers -----

def _cs_finite_range(arr: np.ndarray) -> tuple[float, float]:
    """Return (lo, hi) percentile bounds from finite positive values."""
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if not finite.size:
        return 1e-3, 1.0
    lo = max(float(np.percentile(finite, 1)), 1e-6)
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        hi = lo * 10
    return lo, hi


def _cs_sync_widgets_to_range(lo: float, hi: float) -> None:
    """Update min/max/range widgets to reflect new bounds without firing callbacks."""
    use_log = bool(w_cs_log.value)
    if use_log:
        slider_lo = float(np.log10(max(lo, 1e-12)))
        slider_hi = float(np.log10(max(hi, lo * 1.0001)))
        pad_lo, pad_hi = slider_lo - 0.5, slider_hi + 0.5
    else:
        pad_lo, pad_hi = lo - abs(lo) * 0.5 - 1e-6, hi + abs(hi) * 0.5 + 1e-6
        slider_lo, slider_hi = lo, hi
    _image_cache["cs_suspend"] = True
    try:
        w_cs_range.start = pad_lo
        w_cs_range.end = pad_hi
        w_cs_range.value = (slider_lo, slider_hi)
        w_cs_min.value = float(lo)
        w_cs_max.value = float(hi)
    finally:
        _image_cache["cs_suspend"] = False


def _cs_apply_to_mapper(lo: float, hi: float) -> None:
    """Apply (lo, hi) to the active color mapper, clamping for log."""
    mapper = _image_cache.get("mapper")
    if mapper is None:
        return
    if w_cs_log.value:
        lo = max(lo, 1e-9)
        hi = max(hi, lo * 1.0001)
    elif hi <= lo:
        hi = lo + abs(lo) * 0.01 + 1e-9
    try:
        mapper.low = lo
        mapper.high = hi
    except Exception as exc:
        log.warning("color-scale mapper update failed: %s", exc)


def _build_histogram(arr: np.ndarray) -> None:
    """Build/refresh the histogram pane from the current frame data."""
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import Span, ColumnDataSource

    if arr is None or arr.size == 0:
        w_cs_hist.object = None
        return
    use_log = bool(w_cs_log.value)
    flat = arr[np.isfinite(arr)]
    if use_log:
        flat = flat[flat > 0]
    if flat.size == 0:
        w_cs_hist.object = None
        return

    # Sample for performance on large detectors
    if flat.size > 200000:
        idx = np.random.choice(flat.size, 200000, replace=False)
        flat = flat[idx]

    if use_log:
        # Log-spaced bins in intensity space, displayed in log10
        lo_val = float(np.min(flat))
        hi_val = float(np.max(flat))
        edges_lin = np.logspace(np.log10(lo_val), np.log10(hi_val), 80)
        hist, edges_lin = np.histogram(flat, bins=edges_lin)
        # Convert edges to log10 for display
        edges = np.log10(edges_lin)
        x_axis_label = "log10(intensity)"
    else:
        data = flat
        edges = np.linspace(float(np.min(data)), float(np.max(data)), 80)
        hist, edges = np.histogram(data, bins=edges)
        x_axis_label = "intensity"

    # Use log-scaled counts for the y-axis (avoids single dominant bin)
    hist_display = np.where(hist > 0, hist, 0.1).astype(float)

    src = ColumnDataSource(data=dict(
        left=edges[:-1], right=edges[1:],
        top=hist_display, bottom=np.full_like(hist_display, 0.1),
    ))
    p = bk_figure(
        height=120, sizing_mode="stretch_width",
        tools="", toolbar_location=None,
        x_axis_label=x_axis_label, y_axis_label="count",
        y_axis_type="log",
    )
    p.quad(left="left", right="right", top="top", bottom="bottom",
           source=src, fill_color="#5b9bd5", line_color="white", alpha=0.85)
    p.yaxis.minor_tick_line_color = None
    p.xaxis.minor_tick_line_color = None

    # Add lo/hi Spans (in slider's coordinate system)
    lo_val, hi_val = w_cs_range.value
    lo_span = Span(location=float(lo_val), dimension="height",
                   line_color="red", line_width=2, line_dash="solid")
    hi_span = Span(location=float(hi_val), dimension="height",
                   line_color="red", line_width=2, line_dash="solid")
    p.add_layout(lo_span)
    p.add_layout(hi_span)

    _image_cache["hist_lo_span"] = lo_span
    _image_cache["hist_hi_span"] = hi_span
    w_cs_hist.object = p


def _update_hist_spans() -> None:
    """Move the histogram low/high markers to current slider values."""
    lo_span = _image_cache.get("hist_lo_span")
    hi_span = _image_cache.get("hist_hi_span")
    lo, hi = w_cs_range.value
    if lo_span is not None:
        lo_span.location = float(lo)
    if hi_span is not None:
        hi_span.location = float(hi)


# ----- Color scale callbacks -----

def _on_cs_cmap(event):
    mapper = _image_cache.get("mapper")
    if mapper is None:
        return
    try:
        mapper.palette = event.new
    except Exception as exc:
        log.warning("colormap change failed: %s", exc)


def _on_cs_log(_event=None):
    """Toggling log requires rebuilding the mapper (Log vs Linear class)."""
    # Force a figure rebuild on next render
    _image_cache["figure"] = None
    _image_cache["source"] = None
    _image_cache["mapper"] = None
    _image_cache["fig_image_shape"] = None
    field = _image_cache.get("field")
    idx = w_image_slider.value
    if field is not None:
        _render_image_frame(field, idx)


def _on_cs_range(event):
    if _image_cache.get("cs_suspend"):
        return
    lo, hi = event.new
    if w_cs_log.value:
        lo_v, hi_v = 10 ** float(lo), 10 ** float(hi)
    else:
        lo_v, hi_v = float(lo), float(hi)
    _image_cache["cs_suspend"] = True
    try:
        w_cs_min.value = lo_v
        w_cs_max.value = hi_v
    finally:
        _image_cache["cs_suspend"] = False
    _cs_apply_to_mapper(lo_v, hi_v)
    _update_hist_spans()


def _on_cs_minmax(_event=None):
    if _image_cache.get("cs_suspend"):
        return
    lo_v = float(w_cs_min.value)
    hi_v = float(w_cs_max.value)
    if hi_v <= lo_v:
        return
    if w_cs_log.value:
        slider_lo = float(np.log10(max(lo_v, 1e-12)))
        slider_hi = float(np.log10(max(hi_v, lo_v * 1.0001)))
    else:
        slider_lo, slider_hi = lo_v, hi_v
    _image_cache["cs_suspend"] = True
    try:
        # Expand slider bounds if needed
        if slider_lo < w_cs_range.start:
            w_cs_range.start = slider_lo - 0.5
        if slider_hi > w_cs_range.end:
            w_cs_range.end = slider_hi + 0.5
        w_cs_range.value = (slider_lo, slider_hi)
    finally:
        _image_cache["cs_suspend"] = False
    _cs_apply_to_mapper(lo_v, hi_v)
    _update_hist_spans()


w_cs_cmap.param.watch(_on_cs_cmap, "value")
w_cs_log.param.watch(_on_cs_log, "value")
w_cs_range.param.watch(_on_cs_range, "value")
w_cs_min.param.watch(_on_cs_minmax, "value")
w_cs_max.param.watch(_on_cs_minmax, "value")


# ----- Line / alignment helpers -----

def _line_profile(arr: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                  n: int = 200, width: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sample a 1D profile along the line from (x0,y0) to (x1,y1).

    Parameters
    ----------
    width : int
        Number of pixels to average perpendicular to the line direction.
        If > 1, multiple parallel lines are sampled and the mean taken.

    Returns (distance_px, intensity).  Uses bilinear interpolation.
    """
    h, w = arr.shape
    n = max(2, int(n))
    width = max(1, int(width))

    dx_line = x1 - x0
    dy_line = y1 - y0
    length = np.hypot(dx_line, dy_line)
    if length < 1e-6:
        return np.zeros(n), np.zeros(n)

    # Unit perpendicular vector
    perp_x = -dy_line / length
    perp_y = dx_line / length

    # Offsets centered on 0 for the width band
    offsets = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width)

    # Sample along the line for each offset, then average
    all_vals = []
    for off in offsets:
        xs = np.linspace(x0 + off * perp_x, x1 + off * perp_x, n)
        ys = np.linspace(y0 + off * perp_y, y1 + off * perp_y, n)
        # Clamp to bounds (leave margin for bilinear)
        xs_c = np.clip(xs, 0, w - 2.001)
        ys_c = np.clip(ys, 0, h - 2.001)
        xi = xs_c.astype(int)
        yi = ys_c.astype(int)
        fx = xs_c - xi
        fy = ys_c - yi
        # Bilinear sample
        a = arr[yi, xi]
        b = arr[yi, np.minimum(xi + 1, w - 1)]
        c = arr[np.minimum(yi + 1, h - 1), xi]
        d = arr[np.minimum(yi + 1, h - 1), np.minimum(xi + 1, w - 1)]
        vals = (
            a * (1 - fx) * (1 - fy)
            + b * fx * (1 - fy)
            + c * (1 - fx) * fy
            + d * fx * fy
        )
        all_vals.append(vals)

    avg_vals = np.nanmean(all_vals, axis=0)
    # Distance along the center line
    center_xs = np.linspace(x0, x1, n)
    center_ys = np.linspace(y0, y1, n)
    dist = np.sqrt((center_xs - x0) ** 2 + (center_ys - y0) ** 2)
    return dist, avg_vals


def _line_stats_text(xs: list, ys: list) -> tuple[str, list]:
    """Format stats markdown for all drawn lines; return (md, line_list).

    Each line is represented as (x0, y0, x1, y1) using the first and last
    vertex of its xs/ys polyline.
    """
    lines = []
    rows = []
    for i, (xx, yy) in enumerate(zip(xs, ys)):
        if not xx or len(xx) < 2:
            continue
        x0, y0 = float(xx[0]), float(yy[0])
        x1, y1 = float(xx[-1]), float(yy[-1])
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        lines.append((x0, y0, x1, y1))
        rows.append(
            f"**Line {i + 1}** — "
            f"start=({x0:.1f}, {y0:.1f}), end=({x1:.1f}, {y1:.1f}), "
            f"length={length:.2f} px, angle={angle:.2f}°"
        )
    if not rows:
        return ("*Toggle 'Enable line draw' then click two points on the image "
                "to add a line. Hold shift+drag to add more. "
                "Click a line + Backspace to delete.*"), lines
    return "  \n".join(rows), lines


def _update_line_analysis(*_events) -> None:
    """Recompute line stats and profile plot from the line CDS."""
    from bokeh.plotting import figure as bk_figure

    line_src = _image_cache.get("line_source")
    if line_src is None:
        return
    xs_list = list(line_src.data.get("xs", []))
    ys_list = list(line_src.data.get("ys", []))
    md, lines = _line_stats_text(xs_list, ys_list)
    w_align_stats.object = md

    # Update line renderer width to match the width widget
    line_renderer = _image_cache.get("line_renderer")
    line_width_px = max(1, int(w_align_width.value))
    if line_renderer is not None:
        try:
            line_renderer.glyph.line_width = max(1.5, float(line_width_px))
        except Exception:
            pass

    if not lines:
        w_align_profile.object = None
        return

    # Get current displayed image array
    src = _image_cache.get("source")
    if src is None:
        return
    image_list = src.data.get("image")
    if not image_list:
        return
    arr = np.asarray(image_list[0])

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#17becf"]
    fig_kwargs = dict(
        height=180, sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_axis_label="distance (px)", y_axis_label="intensity",
    )
    if w_cs_log.value:
        fig_kwargs["y_axis_type"] = "log"
    p = bk_figure(**fig_kwargs)
    for i, (x0, y0, x1, y1) in enumerate(lines):
        try:
            dist, vals = _line_profile(arr, x0, y0, x1, y1, width=line_width_px)
        except Exception as exc:
            log.warning("line profile failed: %s", exc)
            continue
        c = palette[i % len(palette)]
        lbl = f"Line {i + 1}"
        if line_width_px > 1:
            lbl += f" (w={line_width_px})"
        p.line(dist, vals, line_color=c, line_width=1.5,
               legend_label=lbl)
    if lines:
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "8pt"
    w_align_profile.object = p


def _on_align_enable(event):
    """Show or hide the line-draw renderer + tool."""
    renderer = _image_cache.get("line_renderer")
    tool = _image_cache.get("line_draw_tool")
    if renderer is None or tool is None:
        return
    on = bool(event.new)
    renderer.visible = on
    # Toggle active tool
    fig = _image_cache.get("figure")
    if fig is not None:
        try:
            fig.toolbar.active_tap = tool if on else None
            fig.toolbar.active_drag = tool if on else None
        except Exception:
            pass


def _on_align_clear(_event=None):
    line_src = _image_cache.get("line_source")
    if line_src is None:
        return
    line_src.data = dict(xs=[], ys=[])
    _update_line_analysis()


w_align_enable.param.watch(_on_align_enable, "value")
w_btn_align_clear.on_click(_on_align_clear)
w_align_width.param.watch(lambda *_: _update_line_analysis(), "value")


def _cached_fetch_frame(run, field: str, frame_idx: int, ds=None) -> np.ndarray | None:
    """Fetch a single image frame, using the disk cache when a uid is known."""
    uid = _state.get("selected_uid")
    if uid:
        arr = get_or_fetch_image_frame(
            uid, field, frame_idx,
            lambda: tb.fetch_all_frames(run, "primary", field),
        )
    else:
        arr = tb.fetch_frame(run, "primary", field, frame_idx=frame_idx, _dataset=ds)
    return _coerce_to_2d_frame(arr)


def _coerce_to_2d_frame(arr) -> np.ndarray | None:
    """Reduce a detector frame to a 2-D array for display.

    Some streams return per-frame arrays with extra leading axes (e.g. a
    multi-exposure detector field shaped ``(N, K, H, W)`` indexes to
    ``(K, H, W)``).  Squeeze singleton dims; if still 3-D+, take the
    first sub-frame along each extra axis so downstream image widgets
    always see ``(H, W)``.
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim <= 2:
        return a
    a = np.squeeze(a)
    while a.ndim > 2:
        a = a[0]
    return a


def _render_image_frame(field, idx):
    """Fetch, orient, and render a single image frame (preserves zoom/pan)."""
    run = _ensure_run()
    if run is None:
        return
    ds = _image_cache.get("dataset")
    frame = _cached_fetch_frame(run, field, idx, ds)
    if frame is not None:
        # Capture raw detector shape *before* orientation so the polygon
        # transform knows the original (rows, cols).
        _image_cache["raw_shape"] = tuple(frame.shape)
        frame = _orient_frame(frame, field)
        _update_image_in_place(frame, f"primary/{field} frame {idx}")
        w_image_status.object = f"**primary/{field}** — frame {idx}"
        # Refresh the dynamic mask overlay if enabled
        if w_mask_dynamic.value:
            _render_dynamic_mask(idx)


def _on_image_slider(event):
    """Render the selected frame when slider changes, sync explore cursor."""
    field = _image_cache.get("field")
    if not field:
        return
    _render_image_frame(field, event.new)
    _update_explore_cursor(event.new)


def _on_image_field(event):
    """Switch detector field and reset slider."""
    field = event.new
    if not field:
        return
    _image_cache["field"] = field
    # Reset persistent figure so a new one is created for the new detector
    # (different orientation / dimensions / mask).
    _image_cache["figure"] = None
    _image_cache["source"] = None
    _image_cache["mapper"] = None
    _image_cache["fig_image_shape"] = None
    info = _detail_cache.get("primary_info")
    if info:
        shape = info["fields"].get(field, ())
        n = shape[0] if len(shape) >= 3 else 1
        _image_cache["n_frames"] = n
        w_image_slider.value = 0
        w_image_slider.end = max(0, n - 1)
    _render_image_frame(field, 0)


w_image_slider.param.watch(_on_image_slider, "value")
w_image_field.param.watch(_on_image_field, "value")


# ---------------------------------------------------------------------------
# Grid (multi-frame) tab — facet grid of all frames for one detector,
# with linked axes, colormap / log / range controls, and intensity hover.
# ---------------------------------------------------------------------------

MV_PALETTES = [
    "Turbo256", "Viridis256", "Plasma256", "Inferno256",
    "Magma256", "Cividis256", "Greys256",
]
MV_MAX_FRAMES = 64  # safety cap so we don't spawn hundreds of figures

w_mv_field = pn.widgets.Select(name="Detector", options=[], width=200)
w_mv_cmap = pn.widgets.Select(
    name="Colormap", value="Turbo256", options=MV_PALETTES, width=140,
)
w_mv_log = pn.widgets.Checkbox(name="Log color scale", value=True, width=140)
w_mv_range = pn.widgets.RangeSlider(
    name="Intensity (log10)", start=-2.0, end=6.0, value=(-1.0, 4.0),
    step=0.05, sizing_mode="stretch_width", format="0.00",
)
w_mv_status = pn.pane.Markdown("*Click tab to load.*")
w_mv_spinner = pn.indicators.LoadingSpinner(value=False, size=20, visible=False)
w_mv_label = pn.widgets.Select(
    name="Frame label", options=["(frame #)"], value="(frame #)", width=180,
)
w_mv_grid = pn.pane.Bokeh(object=None, sizing_mode="stretch_both", min_height=500)

# Pagination controls for scans with more frames than MV_MAX_FRAMES
w_mv_prev = pn.widgets.Button(name="\u25C0 Prev", width=80, button_type="default", disabled=True)
w_mv_next = pn.widgets.Button(name="Next \u25B6", width=80, button_type="default", disabled=True)
w_mv_page_status = pn.pane.Markdown("", width=180)

_multiview_cache: dict = {
    "uid": None, "field": None, "n_frames": 0,
    "total_frames": 0, "page": 0,
    "frames": None, "renderers": None, "mapper": None,
    "log": None, "data_lo": None, "data_hi": None,
    "suspend_range_cb": False,
    "loading": False,
}


def _mv_grid_dims(n: int) -> tuple[int, int]:
    """Pick (rows, cols) so cols/rows ≈ 2 (approx 2:1 grid aspect)."""
    if n <= 0:
        return 1, 1
    rows = max(1, int(round(np.sqrt(n / 2.0))))
    cols = int(np.ceil(n / rows))
    # Make sure it actually fits
    while rows * cols < n:
        cols += 1
    return rows, cols


def _mv_compute_data_range(frames):
    """Return (lo, hi) finite-positive percentile bounds across all frames."""
    lo, hi = None, None
    for arr in frames:
        finite = arr[np.isfinite(arr) & (arr > 0)]
        if not finite.size:
            continue
        flo = float(np.percentile(finite, 1))
        fhi = float(np.percentile(finite, 99.5))
        if lo is None or flo < lo:
            lo = flo
        if hi is None or fhi > hi:
            hi = fhi
    if lo is None:
        # Fallback: use full finite range (including zero/negative)
        all_finite = np.concatenate(
            [arr[np.isfinite(arr)].ravel() for arr in frames if np.any(np.isfinite(arr))]
        ) if frames else np.array([])
        if all_finite.size:
            lo = max(float(np.percentile(all_finite, 1)), 1e-6)
            hi = float(np.percentile(all_finite, 99.5))
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi
        return 1e-3, 1.0
    lo = max(lo, 1e-6)
    if hi <= lo:
        hi = lo * 10
    return lo, hi


def _build_multiview_grid(frames, field):
    """Build the Bokeh gridplot of all frames with linked axes."""
    from bokeh.plotting import figure as bk_figure
    from bokeh.layouts import gridplot
    from bokeh.models import (
        ColorBar, ColumnDataSource, HoverTool,
        LinearColorMapper, LogColorMapper,
    )

    if not frames:
        w_mv_grid.object = None
        return

    # Build per-frame labels from primary scalar column if selected
    frame_offset = _multiview_cache.get("page", 0) * MV_MAX_FRAMES
    label_col = w_mv_label.value
    frame_labels: list[str] = []
    if label_col and label_col != "(frame #)":
        df = w_primary_table.value
        if df is not None and label_col in df.columns:
            vals = df[label_col].values
            for i in range(len(frames)):
                abs_i = frame_offset + i
                if abs_i < len(vals):
                    v = vals[abs_i]
                    try:
                        frame_labels.append(f"{label_col}={float(v):.4g}")
                    except (ValueError, TypeError):
                        frame_labels.append(f"{label_col}={v}")
                else:
                    frame_labels.append(f"frame {abs_i}")
    if not frame_labels:
        frame_labels = [f"frame {frame_offset + i}" for i in range(len(frames))]

    # Per-frame display arrays + global data range
    displays = []
    for a in frames:
        a = np.asarray(a)
        if a.ndim > 2:
            a = np.squeeze(a)
            while a.ndim > 2:
                a = a[0]
        if a.ndim < 2:
            # Skip degenerate frames (e.g. 1-D after squeeze)
            continue
        displays.append(np.where(np.isfinite(a), a, 0).astype(np.float32))
    if not displays:
        w_mv_grid.object = None
        w_mv_status.object = "*No renderable frames (all degenerate or empty).*"
        return
    data_lo, data_hi = _mv_compute_data_range(displays)
    _multiview_cache["data_lo"] = data_lo
    _multiview_cache["data_hi"] = data_hi

    # Sync the range slider bounds to the data range (in log10 space)
    log10_lo = float(np.log10(data_lo))
    log10_hi = float(np.log10(data_hi))
    # Suspend the slider callback while we reconfigure it
    _multiview_cache["suspend_range_cb"] = True
    try:
        w_mv_range.start = log10_lo - 0.5
        w_mv_range.end = log10_hi + 0.5
        # Always snap to data range — prevents stale ranges from a previous
        # scan making the grid appear blank.
        w_mv_range.value = (log10_lo, log10_hi)
    finally:
        _multiview_cache["suspend_range_cb"] = False

    lo_val = 10 ** w_mv_range.value[0]
    hi_val = 10 ** w_mv_range.value[1]
    palette = w_mv_cmap.value
    use_log = bool(w_mv_log.value)
    if use_log:
        mapper = LogColorMapper(palette=palette,
                                low=max(lo_val, 1e-9),
                                high=max(hi_val, lo_val * 1.1),
                                nan_color="gray")
    else:
        mapper = LinearColorMapper(palette=palette, low=lo_val, high=hi_val,
                                   nan_color="gray")

    # For log color scale, replace zeros/negatives with mapper.low so they
    # render as the lowest palette color instead of transparent.
    if use_log:
        clip_lo = float(mapper.low)
        displays = [np.where(d > 0, d, clip_lo) for d in displays]

    rows, cols = _mv_grid_dims(len(displays))
    figs = []
    renderers = []
    shared_x = None
    shared_y = None
    for i, disp in enumerate(displays):
        h, w = disp.shape
        kwargs = dict(
            width=300,
            height=300,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            match_aspect=True,
            sizing_mode="stretch_both",
            min_height=180,
        )
        if shared_x is not None:
            kwargs["x_range"] = shared_x
            kwargs["y_range"] = shared_y
        else:
            kwargs["x_range"] = (0, w)
            kwargs["y_range"] = (0, h)
        p = bk_figure(title=frame_labels[i], **kwargs)
        if shared_x is None:
            shared_x = p.x_range
            shared_y = p.y_range
        src = ColumnDataSource(
            data=dict(image=[disp], x=[0], y=[0], dw=[w], dh=[h]),
        )
        r = p.image(image="image", x="x", y="y", dw="dw", dh="dh",
                    color_mapper=mapper, source=src)
        hover = HoverTool(
            renderers=[r],
            tooltips=[
                ("label", frame_labels[i]),
                ("frame", str(frame_offset + i)),
                ("(col, row)", "($x{0}, $y{0})"),
                ("intensity", "@image{0.000}"),
            ],
            point_policy="follow_mouse",
        )
        p.add_tools(hover)
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.title.text_font_size = "9pt"
        # Add colorbar to the rightmost column, top row only — keeps grid tidy
        if i == cols - 1 or (i == len(displays) - 1 and i < cols):
            p.add_layout(
                ColorBar(color_mapper=mapper, label_standoff=6, width=10),
                "right",
            )
        figs.append(p)
        renderers.append(r)

    # Pad with None so gridplot fills the rectangle cleanly
    grid_cells = [figs[r * cols:(r + 1) * cols] for r in range(rows)]
    for row in grid_cells:
        while len(row) < cols:
            row.append(None)

    grid = gridplot(grid_cells, sizing_mode="stretch_both",
                    toolbar_location="above", merge_tools=True)
    w_mv_grid.object = grid
    _multiview_cache["renderers"] = renderers
    _multiview_cache["mapper"] = mapper
    _multiview_cache["log"] = use_log
    _multiview_cache["field"] = field


def _load_multiview():
    """Fetch all frames for the current detector and build the grid."""
    run = _ensure_run()
    if run is None:
        return
    w_mv_status.object = "*Loading…*"

    # Re-use primary stream info (loaded by Explore / Primary tabs too)
    info = _detail_cache.get("primary_info")
    if info is None and "primary" in tb.stream_names(run):
        info = tb.stream_info_for(run, "primary")
        _detail_cache["primary_info"] = info
        _detail_cache["primary_dataset"] = info.get("dataset")

    if not info or not info["images"]:
        w_mv_status.object = "*No image fields found.*"
        w_mv_grid.object = None
        return

    # Ensure primary scalars are loaded so the label dropdown has options
    if not _detail_cache.get("primary_loaded"):
        _load_primary()
    df = w_primary_table.value
    label_options = ["(frame #)"]
    if df is not None and not df.empty:
        label_options += [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    image_fields = list(info["images"])

    # Guard: suppress _on_mv_field / _on_mv_label callbacks while we
    # reconfigure widget options/values to avoid nested Bokeh doc writes.
    _multiview_cache["loading"] = True
    try:
        # Update label dropdown — skip if unchanged to avoid spurious events
        if list(w_mv_label.options) != label_options:
            w_mv_label.options = label_options
        if w_mv_label.value not in label_options:
            w_mv_label.value = "(frame #)"

        # Update detector dropdown — set directly (no clear-then-reset)
        prev = w_mv_field.value
        if list(w_mv_field.options) != image_fields:
            w_mv_field.options = image_fields
        field = prev if prev in image_fields else image_fields[0]
        if w_mv_field.value != field:
            w_mv_field.value = field
    finally:
        _multiview_cache["loading"] = False

    _fetch_and_build_multiview(field)


def _fetch_and_build_multiview(field: str, *, page: int = 0):
    """Fetch a page of frames for `field` and (re)build the grid."""
    run = _ensure_run()
    if not run or not field:
        return
    info = _detail_cache.get("primary_info")
    if not info:
        return
    shape = info["fields"].get(field, ())
    total_frames = shape[0] if len(shape) >= 3 else 1

    # Pagination
    start = page * MV_MAX_FRAMES
    end = min(start + MV_MAX_FRAMES, total_frames)
    if start >= total_frames:
        page = 0
        start = 0
        end = min(MV_MAX_FRAMES, total_frames)
    page_count = max(1, int(np.ceil(total_frames / MV_MAX_FRAMES)))

    w_mv_spinner.value = True
    w_mv_spinner.visible = True
    w_mv_status.object = f"*Loading frames {start}–{end - 1} of {total_frames}…*"
    t0 = time.perf_counter()
    ds = _detail_cache.get("primary_dataset")
    frames = []
    uid = _state.get("selected_uid")
    for i in range(start, end):
        if uid:
            arr = get_or_fetch_image_frame(
                uid, field, i,
                lambda: tb.fetch_all_frames(run, "primary", field),
            )
        else:
            arr = tb.fetch_frame(run, "primary", field, frame_idx=i, _dataset=ds)
        arr = _coerce_to_2d_frame(arr)
        if arr is None:
            continue
        frames.append(_orient_frame(arr, field))
    _multiview_cache["frames"] = frames
    _multiview_cache["n_frames"] = len(frames)
    _multiview_cache["total_frames"] = total_frames
    _multiview_cache["page"] = page
    _multiview_cache["uid"] = _state.get("selected_uid")

    # Diagnostic: log shape & data statistics for debugging blank grids
    if frames:
        shapes = set(f.shape for f in frames)
        sample = frames[0]
        log.info(
            "multiview: %d frames loaded, shapes=%s, sample min=%.3g max=%.3g "
            "finite_pos=%d/%d",
            len(frames), shapes,
            float(np.nanmin(sample)), float(np.nanmax(sample)),
            int(np.sum(np.isfinite(sample) & (sample > 0))), sample.size,
        )

    # Update pagination buttons
    w_mv_prev.disabled = (page <= 0)
    w_mv_next.disabled = (end >= total_frames)
    if page_count > 1:
        w_mv_page_status.object = f"Page **{page + 1}** / {page_count}"
    else:
        w_mv_page_status.object = ""

    try:
        _build_multiview_grid(frames, field)
    except Exception as exc:
        log.exception("multiview grid build failed")
        w_mv_grid.object = None
        w_mv_spinner.value = False
        w_mv_spinner.visible = False
        w_mv_status.object = f"**Grid build error:** `{exc}`"
        return
    dt_ms = (time.perf_counter() - t0) * 1000
    w_mv_spinner.value = False
    w_mv_spinner.visible = False
    w_mv_status.object = (
        f"**primary/{field}** — frames {start}–{end - 1} of {total_frames} ({dt_ms:.0f} ms)"
    )


def _on_mv_field(event):
    if _multiview_cache.get("loading"):
        return
    field = event.new
    if not field:
        return
    _fetch_and_build_multiview(field)


def _on_mv_cmap(event):
    mapper = _multiview_cache.get("mapper")
    if mapper is None:
        return
    try:
        mapper.palette = event.new
    except Exception as exc:
        log.warning("multiview palette update failed: %s", exc)


def _on_mv_log(event):
    # Switching between Linear / Log mapper requires a rebuild
    field = _multiview_cache.get("field")
    frames = _multiview_cache.get("frames")
    if not field or not frames:
        return
    _build_multiview_grid(frames, field)


def _on_mv_range(event):
    if _multiview_cache.get("suspend_range_cb"):
        return
    mapper = _multiview_cache.get("mapper")
    if mapper is None:
        return
    lo, hi = event.new
    lo_v = 10 ** float(lo)
    hi_v = 10 ** float(hi)
    if hi_v <= lo_v:
        hi_v = lo_v * 1.0001
    try:
        mapper.low = max(lo_v, 1e-9) if isinstance(mapper.low, (int, float)) else lo_v
        mapper.high = hi_v
    except Exception as exc:
        log.warning("multiview range update failed: %s", exc)


w_mv_field.param.watch(_on_mv_field, "value")
w_mv_cmap.param.watch(_on_mv_cmap, "value")
w_mv_log.param.watch(_on_mv_log, "value")
w_mv_range.param.watch(_on_mv_range, "value")


def _on_mv_label(event):
    """Rebuild grid with updated frame labels when the label column changes."""
    if _multiview_cache.get("loading"):
        return
    field = _multiview_cache.get("field")
    frames = _multiview_cache.get("frames")
    if field and frames:
        _build_multiview_grid(frames, field)


w_mv_label.param.watch(_on_mv_label, "value")


def _on_mv_prev(_event=None):
    """Go to the previous page of frames."""
    page = _multiview_cache.get("page", 0)
    field = _multiview_cache.get("field")
    if page > 0 and field:
        _fetch_and_build_multiview(field, page=page - 1)


def _on_mv_next(_event=None):
    """Go to the next page of frames."""
    page = _multiview_cache.get("page", 0)
    field = _multiview_cache.get("field")
    total = _multiview_cache.get("total_frames", 0)
    if (page + 1) * MV_MAX_FRAMES < total and field:
        _fetch_and_build_multiview(field, page=page + 1)


w_mv_prev.on_click(_on_mv_prev)
w_mv_next.on_click(_on_mv_next)


# ---------------------------------------------------------------------------
# Mask-overlay callbacks (Explore tab)
# ---------------------------------------------------------------------------

def _current_detector_kind() -> str:
    field = w_image_field.value
    return _detector_for_field(field) or "saxs" if field else "saxs"


def _apply_mask_to_overlay(mask_dict: dict, *, source_label: str = ""):
    """Push polygons from a normalized mask dict into the live overlay source.

    The on-disk polygon coordinates are in *raw* detector indexing.  We
    transform them to match the displayed (oriented) image via
    ``_normalized_mask_to_xs_ys(field, raw_shape)``.
    """
    src = _image_cache.get("mask_source")
    if src is None:
        w_mask_status.object = (
            "*Load a frame first — the overlay attaches to the image figure.*"
        )
        return

    field = _image_cache.get("field")
    raw_shape = _image_cache.get("raw_shape")
    if raw_shape is None:
        # Fall back to the mask file's own declaration so the overlay at
        # least has *some* coordinate system (may be slightly off).
        shape = mask_dict.get("image_shape")
        raw_shape = tuple(shape) if shape else None

    xs, ys, names, kinds = _normalized_mask_to_xs_ys(
        mask_dict, field=field, raw_shape=raw_shape,
    )
    fill_colors = ["#888888" if k == "static" else "#ff5555" for k in kinds]
    line_colors = ["#222222" if k == "static" else "#aa0000" for k in kinds]
    src.data = dict(xs=xs, ys=ys, name=names, kind=kinds,
                    fill_color=fill_colors, line_color=line_colors)
    _image_cache["mask_image_shape"] = mask_dict.get("image_shape")
    suffix = f" ({source_label})" if source_label else ""
    w_mask_status.object = f"*Loaded {len(xs)} polygon(s){suffix}.*"


def _on_mask_show(event):
    """Toggle visibility of the mask overlay."""
    renderer = _image_cache.get("mask_renderer")
    if renderer is None:
        if event.new:
            w_mask_status.object = "*Load a frame first.*"
        return
    renderer.visible = bool(event.new)
    if event.new and not renderer.data_source.data["xs"]:
        # First show — auto-load default for current detector
        _on_mask_reload(None)


# ---------------------------------------------------------------------------
# Dynamic mask overlay (rasterised PyHyper mask, per-frame)
# ---------------------------------------------------------------------------


def _render_dynamic_mask(idx: int | None = None):
    """Compute & push the per-frame dynamic mask into the overlay source."""
    src = _image_cache.get("dyn_source")
    renderer = _image_cache.get("dyn_renderer")
    if src is None or renderer is None:
        return
    if not w_mask_dynamic.value:
        renderer.visible = False
        return

    field = _image_cache.get("field")
    if not field:
        renderer.visible = False
        return
    detector = _detector_for_field(field)
    if detector is None:
        renderer.visible = False
        return
    raw_shape = _image_cache.get("raw_shape")
    if raw_shape is None:
        renderer.visible = False
        return
    if idx is None:
        idx = int(w_image_slider.value or 0)

    run = _ensure_run()
    if run is None:
        renderer.visible = False
        return

    try:
        from PyHyperScattering.SMISWAXSIntegrator import mask_for_frame
        mask = mask_for_frame(
            run, idx, detector,
            raw_shape=raw_shape,
            orient_for_display=True,
        )
    except Exception as exc:
        log.warning("dynamic mask build failed (%s): %s", detector, exc)
        renderer.visible = False
        return

    h, w = mask.shape
    # Build RGBA as (h,w,4) uint8, then view as uint32 for Bokeh.
    rgba8 = np.zeros((h, w, 4), dtype=np.uint8)
    invalid = ~mask
    rgba8[invalid, 0] = 255   # R
    rgba8[invalid, 3] = 110   # A (~43%)
    rgba = np.ascontiguousarray(rgba8).view(dtype=np.uint32).reshape(h, w)

    src.data = dict(image=[rgba], x=[0], y=[0], dw=[w], dh=[h])
    renderer.visible = True


def _on_mask_dynamic(event):
    """Toggle visibility of the dynamic mask overlay."""
    if event.new:
        _render_dynamic_mask()
        if _image_cache.get("dyn_renderer") is None:
            w_mask_status.object = "*Load a frame first.*"
    else:
        renderer = _image_cache.get("dyn_renderer")
        if renderer is not None:
            renderer.visible = False


def _on_mask_edit(event):
    """Activate the PolyDraw tool so the user can draw new polygons.

    PolyDraw is a *tap* tool (tap to add vertices, double-tap to finish);
    PolyEdit is also added so users can switch via the toolbar icon.
    """
    fig = _image_cache.get("figure")
    draw_tool = _image_cache.get("draw_tool")
    edit_tool = _image_cache.get("edit_tool")
    if fig is None or draw_tool is None:
        w_mask_status.object = "*Load a frame first.*"
        return
    if event.new:
        renderer = _image_cache.get("mask_renderer")
        if renderer is not None and not renderer.visible:
            w_mask_show.value = True
        try:
            # PolyDraw is a tap tool — don't touch active_drag (that would
            # disable panning).  Activating it via active_tap arms the tool
            # immediately so the user can start clicking.
            fig.toolbar.active_tap = draw_tool
        except Exception as exc:
            log.warning("could not activate draw tool: %s", exc)
        w_mask_status.object = (
            "*Edit mode on. **Draw a new polygon** (cyan): **long-press** "
            "on the image to place the first vertex, then **single-tap** "
            "to add each next vertex, and **long-press** again to "
            "finish (Esc cancels). New polygons go into a separate cyan "
            "layer; on Save they are appended to the static-mask file as "
            "additional `static_regions` — i.e. they **add to** (do not "
            "replace) the existing grey static polygons, and at "
            "processing time the static mask is combined with the "
            "**dynamic mask** (red overlay). **Edit existing static "
            "polygons**: click the PolyEdit icon (square-with-handles) "
            "in the toolbar, tap a grey or cyan polygon, then drag its "
            "vertices. The **dynamic mask is not editable here** — it "
            "is recomputed per-frame by PyHyper from the live beamstop "
            "position, so its shape can only be changed indirectly "
            "(e.g. by adjusting beam-centre Δ in the Process tab). "
            "Open the browser console (F12) to see `[mask-debug]` tap "
            "coordinates.*"
        )
    else:
        try:
            fig.toolbar.active_tap = "auto"
        except Exception:
            pass
        w_mask_status.object = "*Edit mode off.*"


def _on_mask_reload(_event):
    """Reload the bundled PyHyper default mask for the current detector."""
    detector = _current_detector_kind()
    path = _default_mask_path_for(detector)
    try:
        mask = smid.load_mask_polygons(path)
    except Exception as exc:
        log.warning("mask load failed (%s): %s", path, exc)
        mask = None
    if not mask or not (mask.get("static_regions") or mask.get("beamstops")):
        w_mask_status.object = f"*No default mask available for {detector.upper()}.*"
        return
    _apply_mask_to_overlay(mask, source_label=f"default {detector.upper()}")


def _on_mask_save(_event):
    """Save the current overlay polygons to JSON at w_mask_path."""
    src = _image_cache.get("mask_source")
    new_src = _image_cache.get("new_mask_source")
    if src is None:
        w_mask_status.object = "*Load a frame first.*"
        return
    path_str = (w_mask_path.value or "").strip()
    if not path_str:
        w_mask_status.object = "*Enter a save path first.*"
        return

    from pathlib import Path
    out_path = Path(path_str).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    field = _image_cache.get("field")
    raw_shape = _image_cache.get("raw_shape")
    if raw_shape is None:
        shape = _image_cache.get("mask_image_shape")
        raw_shape = tuple(shape) if shape else None

    # Combine the loaded static-mask polygons with any newly-drawn ones.
    data = src.data
    xs = list(data["xs"])
    ys = list(data["ys"])
    names = list(data["name"])
    kinds = list(data["kind"])
    n_existing = len(xs)
    if new_src is not None:
        new_data = new_src.data
        for i, (px, py) in enumerate(zip(new_data["xs"], new_data["ys"])):
            xs.append(list(px))
            ys.append(list(py))
            names.append(f"user_{i + 1}")
            kinds.append("static")  # treat user-drawn as static-mask regions
    n_new = len(xs) - n_existing

    out_dict = _xs_ys_to_normalized_mask(
        xs, ys, names, kinds,
        field=field, raw_shape=raw_shape,
    )
    try:
        smid.save_mask_polygons(out_dict, out_path)
    except Exception as exc:
        log.exception("mask save failed")
        w_mask_status.object = f"**Save failed:** `{exc}`"
        return
    w_mask_status.object = (
        f"*Saved {len(xs)} polygon(s) "
        f"({n_existing} existing + {n_new} new) → `{out_path}`.*"
    )


def _on_mask_use(_event):
    """Copy the mask path into the corresponding Process-tab mask field."""
    path_str = (w_mask_path.value or "").strip()
    if not path_str:
        w_mask_status.object = "*Save first, then click Use in Process.*"
        return
    detector = _current_detector_kind()
    if detector == "saxs":
        w_proc_saxs_mask.value = path_str
    else:
        w_proc_waxs_mask.value = path_str
    w_mask_status.object = (
        f"*Set Process tab {detector.upper()} mask path to `{path_str}`.*"
    )


w_mask_show.param.watch(_on_mask_show, "value")
w_mask_dynamic.param.watch(_on_mask_dynamic, "value")
w_mask_edit.param.watch(_on_mask_edit, "value")
w_btn_mask_reload.on_click(_on_mask_reload)
w_btn_mask_save.on_click(_on_mask_save)
w_btn_mask_use.on_click(_on_mask_use)


# ---------------------------------------------------------------------------
# Widgets — Processing
# ---------------------------------------------------------------------------

w_proc_geometry = pn.widgets.Select(
    name="Geometry",
    options=["transmission", "grazing"],
    value="transmission",
    width=130,
)

# --- Output grid ---
w_proc_nq = pn.widgets.IntInput(
    name="n_q", value=3000, start=100, end=10000, step=100, width=90,
)
w_proc_nchi = pn.widgets.IntInput(
    name="n_χ", value=DEFAULT_N_CHI, start=36, end=720, step=36, width=90,
)
w_proc_nqxy = pn.widgets.IntInput(
    name="n_qxy", value=DEFAULT_N_QXY, start=100, end=2000, step=50, width=90,
)
w_proc_nqz = pn.widgets.IntInput(
    name="n_qz", value=DEFAULT_N_QZ, start=100, end=2000, step=50, width=90,
)
w_trans_row = pn.Row(w_proc_nq, w_proc_nchi)
w_gi_grid_row = pn.Row(w_proc_nqxy, w_proc_nqz)

# --- Masks ---
w_proc_saxs_mask = pn.widgets.TextInput(
    name="SAXS mask",
    value=DEFAULT_SAXS_MASK,
    placeholder=f"(default: {DEFAULT_SAXS_MASK_NAME})",
    width=320,
)
w_proc_waxs_mask = pn.widgets.TextInput(
    name="WAXS mask",
    value="~/smi/waxs_hotspots.json",
    placeholder=f"(default: {DEFAULT_WAXS_MASK_NAME})",
    width=320,
)

# --- SAXS Q-range / aperture ---
w_proc_saxs_q_cutoff = pn.widgets.FloatInput(
    name="q cutoff (nm⁻¹)", value=0.0, step=0.01, start=0.0, width=120,
)
w_proc_saxs_agbh_ring = pn.widgets.IntInput(
    name="AgBh ring order", value=DEFAULT_SAXS_AGBH_RING_ORDER,
    start=1, end=20, width=100,
)
w_proc_saxs_q_margin = pn.widgets.FloatInput(
    name="q margin frac", value=DEFAULT_SAXS_Q_MARGIN_FRACTION,
    step=0.005, start=0.0, end=0.5, width=110,
)

# --- Geometry corrections ---
w_proc_saxs_row_delta = pn.widgets.FloatInput(
    name="SAXS Δrow", value=DEFAULT_SAXS_ROW_DELTA, step=0.50, width=80,
)
w_proc_saxs_col_delta = pn.widgets.FloatInput(
    name="SAXS Δcol", value=DEFAULT_SAXS_COL_DELTA, step=0.50, width=80,
)
w_proc_dist_delta = pn.widgets.FloatInput(
    name="SAXS Δdist (mm)", value=DEFAULT_SAXS_DIST_DELTA, step=1.00, width=110,
)
w_proc_waxs_row_delta = pn.widgets.FloatInput(
    name="WAXS Δrow", value=DEFAULT_WAXS_ROW_DELTA, step=0.50, width=80,
)
w_proc_waxs_col_delta = pn.widgets.FloatInput(
    name="WAXS Δcol", value=0, step=0.50, width=80,
)
w_proc_waxs_col_per_arc = pn.widgets.FloatInput(
    name="WAXS col/arc°", value=DEFAULT_WAXS_BEAM_COL_PER_ARC_DEG,
    step=0.01, width=100,
)
w_saxs_geom_section = pn.Column(
    pn.pane.Markdown("**SAXS beam-centre Δ (px)**"),
    pn.Row(w_proc_saxs_row_delta, w_proc_saxs_col_delta, w_proc_dist_delta),
)

# --- Hot-pixel rejection ---
w_proc_dezinger = pn.widgets.FloatInput(
    name="Dezinger σ", value=DEFAULT_DEZINGER, step=100.00, width=100,
)
w_proc_dezinger_kernel = pn.widgets.IntInput(
    name="Kernel size", value=DEFAULT_DEZINGER_KERNEL,
    start=3, end=21, step=2, width=90,
)
w_proc_pixel_splitting = pn.widgets.IntInput(
    name="Pixel splitting", value=3,
    start=1, end=8, step=1, width=100,
)

# --- Intensity corrections ---
w_proc_solid_angle = pn.widgets.Checkbox(
    name="Solid-angle correction", value=DEFAULT_SOLID_ANGLE, width=180,
)

# --- GI-specific ---
w_proc_incident_angle = pn.widgets.FloatInput(
    name="α_i (°)", value=DEFAULT_INCIDENT_ANGLE, step=0.01, width=90,
)
w_proc_incident_angle_auto = pn.widgets.Checkbox(
    name="Auto α_i", value=True, width=80,
)
w_proc_theta_offset = pn.widgets.FloatInput(
    name="θ offset (°)", value=DEFAULT_THETA_OFFSET, step=0.01, width=90,
)
w_proc_beamstop_max_arc = pn.widgets.FloatInput(
    name="Beamstop |arc| max (°)", value=DEFAULT_BEAMSTOP_MAX_ABS_ARC_DEG,
    step=0.50, width=150,
)

# --- Backend / display options ---
w_proc_saxs_rotate = pn.widgets.Checkbox(
    name="SAXS rotate CW 90°", value=False, width=160,
)
w_proc_waxs_flip = pn.widgets.Checkbox(
    name="WAXS flip horizontal", value=False, width=170,
)
w_proc_waxs_qx_shift = pn.widgets.FloatInput(
    name="WAXS Δqx (nm⁻¹)", value=0.0, step=0.01, width=120,
)
w_proc_waxs_qy_shift = pn.widgets.FloatInput(
    name="WAXS Δqy (nm⁻¹)", value=0.0, step=0.01, width=120,
)

# --- Dynamic SAXS masking ---
w_proc_dynamic_mask = pn.widgets.Checkbox(
    name="Enable dynamic SAXS mask", value=False, width=200,
)
w_proc_dyn_shadow_enabled = pn.widgets.Checkbox(
    name="WAXS shadow", value=True, width=120,
)
w_proc_dyn_shadow_beam_deg = pn.widgets.FloatInput(
    name="Beam visible (°)", value=DEFAULT_DYN_SHADOW_BEAM_VISIBLE_DEG,
    step=0.50, width=120,
)
w_proc_dyn_shadow_clear_deg = pn.widgets.FloatInput(
    name="Clear edge (°)", value=DEFAULT_DYN_SHADOW_CLEAR_EDGE_DEG,
    step=0.50, width=120,
)
w_proc_dyn_aper_enabled = pn.widgets.Checkbox(
    name="Aperture mask", value=True, width=120,
)
w_proc_dyn_aper_agbh_ring = pn.widgets.IntInput(
    name="AgBh ring", value=DEFAULT_DYN_APER_AGBH_RING_ORDER,
    start=1, end=20, width=90,
)
w_proc_dyn_aper_q_margin = pn.widgets.FloatInput(
    name="q margin frac", value=DEFAULT_DYN_APER_Q_MARGIN_FRACTION,
    step=0.005, start=0.0, end=0.5, width=110,
)
w_proc_dyn_aper_q_cutoff = pn.widgets.FloatInput(
    name="q cutoff (nm⁻¹)", value=0.0, step=0.01, start=0.0, width=120,
)

# --- Advanced WAXS calibration ---
w_proc_waxs_energy = pn.widgets.FloatInput(
    name="Energy (keV)", value=DEFAULT_WAXS_ENERGY_KEV, step=0.01, width=110,
)
w_proc_waxs_dist = pn.widgets.FloatInput(
    name="Distance (mm)", value=DEFAULT_WAXS_SAMPLE_DIST_MM, step=1.00, width=120,
)
w_proc_waxs_pixel = pn.widgets.FloatInput(
    name="Pixel size (mm)", value=DEFAULT_WAXS_PIXEL_SIZE_MM,
    step=0.001, width=120,
)
w_proc_waxs_beam_row = pn.widgets.FloatInput(
    name="Beam row (px)", value=DEFAULT_WAXS_BEAM_CENTER_ROW, step=0.50, width=110,
)
w_proc_waxs_beam_col = pn.widgets.FloatInput(
    name="Beam col (px)", value=DEFAULT_WAXS_BEAM_CENTER_COL, step=0.50, width=110,
)
w_proc_waxs_panel_cols = pn.widgets.TextInput(
    name="Panel col ranges", value="(0,206),(206,413),(413,619)", width=280,
)
w_proc_waxs_panel_offsets = pn.widgets.TextInput(
    name="Panel offsets (°)", value="-7.0, 0.0, 7.0", width=200,
)
w_proc_waxs_panel_row_shifts = pn.widgets.TextInput(
    name="Panel row shifts (px)", value="0.0, 0.0, 0.0", width=200,
)
w_proc_waxs_panel_col_shifts = pn.widgets.TextInput(
    name="Panel col shifts (px)", value="0.0, 0.0, 0.0", width=200,
)
w_proc_waxs_panel_delta = pn.widgets.TextInput(
    name="Panel Δ tilt (°)", value="0.0, 0.0, 0.0", width=200,
)
w_proc_waxs_theta_zero = pn.widgets.FloatInput(
    name="θ₀ arc (°)", value=DEFAULT_WAXS_THETA_ZERO_DEG, step=0.01, width=100,
)
w_proc_waxs_offset_x = pn.widgets.FloatInput(
    name="Sample Δx (mm)", value=DEFAULT_WAXS_SAMPLE_OFFSET_X_MM,
    step=0.01, width=110,
)
w_proc_waxs_offset_z = pn.widgets.FloatInput(
    name="Sample Δz (mm)", value=DEFAULT_WAXS_SAMPLE_OFFSET_Z_MM,
    step=0.01, width=110,
)
w_proc_waxs_col_arc_cal = pn.widgets.FloatInput(
    name="col/arc° (cal)", value=0.0, step=0.01, width=100,
)
w_proc_waxs_qh_sign = pn.widgets.FloatInput(
    name="q_h sign", value=DEFAULT_WAXS_Q_HORIZONTAL_SIGN, step=1.00, width=80,
)
w_proc_waxs_qv_sign = pn.widgets.FloatInput(
    name="q_v sign", value=DEFAULT_WAXS_Q_VERTICAL_SIGN, step=1.00, width=80,
)
w_proc_waxs_rot_k = pn.widgets.IntInput(
    name="rot90 k", value=DEFAULT_WAXS_ROTATION_K, start=0, end=3, width=80,
)

# --- Advanced WAXS masking ---
w_proc_waxs_bsx_ref = pn.widgets.FloatInput(
    name="BSX ref (mm)", value=0.0, step=0.01, width=110,
)

# ---------------------------------------------------------------------------
# Default-value tracking — grey-out widgets at default, highlight changed
# ---------------------------------------------------------------------------

_PARAM_DEFAULT_CSS = """
:host(.param-default) { opacity: 0.45; }
:host(.param-changed) {
  opacity: 1.0;
  border-left: 3px solid #1f77b4;
  padding-left: 2px;
}
"""

# Registry: card-label → [(widget, default_value), ...]
_CARD_PARAM_REGISTRY: dict[str, list[tuple[Any, Any]]] = {
    "grid": [
        (w_proc_nq, DEFAULT_N_Q),
        (w_proc_nchi, DEFAULT_N_CHI),
        (w_proc_nqxy, DEFAULT_N_QXY),
        (w_proc_nqz, DEFAULT_N_QZ),
    ],
    "masks": [
        (w_proc_saxs_mask, DEFAULT_SAXS_MASK),
        (w_proc_waxs_mask, DEFAULT_WAXS_MASK),
    ],
    "saxs_qrange": [
        (w_proc_saxs_q_cutoff, 0.0),
        (w_proc_saxs_agbh_ring, DEFAULT_SAXS_AGBH_RING_ORDER),
        (w_proc_saxs_q_margin, DEFAULT_SAXS_Q_MARGIN_FRACTION),
    ],
    "geometry": [
        (w_proc_saxs_row_delta, DEFAULT_SAXS_ROW_DELTA),
        (w_proc_saxs_col_delta, DEFAULT_SAXS_COL_DELTA),
        (w_proc_dist_delta, DEFAULT_SAXS_DIST_DELTA),
        (w_proc_waxs_row_delta, DEFAULT_WAXS_ROW_DELTA),
        (w_proc_waxs_col_delta, DEFAULT_WAXS_COL_DELTA),
        (w_proc_waxs_col_per_arc, DEFAULT_WAXS_BEAM_COL_PER_ARC_DEG),
    ],
    "dezinger": [
        (w_proc_dezinger, DEFAULT_DEZINGER),
        (w_proc_dezinger_kernel, DEFAULT_DEZINGER_KERNEL),
        (w_proc_pixel_splitting, DEFAULT_PIXEL_SPLITTING),
    ],
    "intensity": [
        (w_proc_solid_angle, DEFAULT_SOLID_ANGLE),
    ],
    "gi": [
        (w_proc_incident_angle, DEFAULT_INCIDENT_ANGLE),
        (w_proc_incident_angle_auto, True),
        (w_proc_theta_offset, DEFAULT_THETA_OFFSET),
        (w_proc_beamstop_max_arc, DEFAULT_BEAMSTOP_MAX_ABS_ARC_DEG),
    ],
    "backend": [
        (w_proc_saxs_rotate, False),
        (w_proc_waxs_flip, False),
        (w_proc_waxs_qx_shift, 0.0),
        (w_proc_waxs_qy_shift, 0.0),
    ],
    "dynamic_mask": [
        (w_proc_dynamic_mask, False),
        (w_proc_dyn_shadow_enabled, True),
        (w_proc_dyn_shadow_beam_deg, DEFAULT_DYN_SHADOW_BEAM_VISIBLE_DEG),
        (w_proc_dyn_shadow_clear_deg, DEFAULT_DYN_SHADOW_CLEAR_EDGE_DEG),
        (w_proc_dyn_aper_enabled, True),
        (w_proc_dyn_aper_agbh_ring, DEFAULT_DYN_APER_AGBH_RING_ORDER),
        (w_proc_dyn_aper_q_margin, DEFAULT_DYN_APER_Q_MARGIN_FRACTION),
        (w_proc_dyn_aper_q_cutoff, 0.0),
    ],
    "waxs_cal": [
        (w_proc_waxs_energy, DEFAULT_WAXS_ENERGY_KEV),
        (w_proc_waxs_dist, DEFAULT_WAXS_SAMPLE_DIST_MM),
        (w_proc_waxs_pixel, DEFAULT_WAXS_PIXEL_SIZE_MM),
        (w_proc_waxs_beam_row, DEFAULT_WAXS_BEAM_CENTER_ROW),
        (w_proc_waxs_beam_col, DEFAULT_WAXS_BEAM_CENTER_COL),
        (w_proc_waxs_panel_cols, "(0,206),(206,413),(413,619)"),
        (w_proc_waxs_panel_offsets, "-7.0, 0.0, 7.0"),
        (w_proc_waxs_panel_row_shifts, "0.0, 0.0, 0.0"),
        (w_proc_waxs_panel_col_shifts, "0.0, 0.0, 0.0"),
        (w_proc_waxs_panel_delta, "0.0, 0.0, 0.0"),
        (w_proc_waxs_theta_zero, DEFAULT_WAXS_THETA_ZERO_DEG),
        (w_proc_waxs_offset_x, DEFAULT_WAXS_SAMPLE_OFFSET_X_MM),
        (w_proc_waxs_offset_z, DEFAULT_WAXS_SAMPLE_OFFSET_Z_MM),
        (w_proc_waxs_col_arc_cal, 0.0),
        (w_proc_waxs_qh_sign, DEFAULT_WAXS_Q_HORIZONTAL_SIGN),
        (w_proc_waxs_qv_sign, DEFAULT_WAXS_Q_VERTICAL_SIGN),
        (w_proc_waxs_rot_k, DEFAULT_WAXS_ROTATION_K),
    ],
    "waxs_mask_adv": [
        (w_proc_waxs_bsx_ref, 0.0),
    ],
}


def _is_at_default(widget, default_val) -> bool:
    """Compare widget value to its default, with tolerance for floats."""
    val = widget.value
    if isinstance(val, float) and isinstance(default_val, (int, float)):
        return abs(val - float(default_val)) < 1e-9
    return val == default_val


def _apply_param_style(widget, default_val) -> None:
    """Toggle param-default / param-changed CSS class on *widget*."""
    css = widget.css_classes or []
    css = [c for c in css if c not in ("param-default", "param-changed")]
    if _is_at_default(widget, default_val):
        css.append("param-default")
    else:
        css.append("param-changed")
    widget.css_classes = css


def _make_param_watcher(widget, default_val):
    """Return a callback that re-styles *widget* when its value changes."""
    def _cb(event):
        _apply_param_style(widget, default_val)
    return _cb


# Inject stylesheet and wire watchers on every tracked widget
for _card_key, _entries in _CARD_PARAM_REGISTRY.items():
    for _w, _dv in _entries:
        if not hasattr(_w, "stylesheets") or _w.stylesheets is None:
            _w.stylesheets = [_PARAM_DEFAULT_CSS]
        else:
            _w.stylesheets = list(_w.stylesheets) + [_PARAM_DEFAULT_CSS]
        _apply_param_style(_w, _dv)
        _w.param.watch(_make_param_watcher(_w, _dv), "value")


def _make_restore_defaults_cb(card_key: str):
    """Return a callback that resets all widgets in *card_key* to defaults."""
    def _cb(event):
        for widget, default_val in _CARD_PARAM_REGISTRY[card_key]:
            widget.value = default_val
    return _cb


_card_restore_buttons: dict[str, pn.widgets.Button] = {}
for _card_key in _CARD_PARAM_REGISTRY:
    _btn = pn.widgets.Button(
        name="↩ Defaults", button_type="light", width=90,
        margin=(0, 0, 5, 0),
    )
    _btn.on_click(_make_restore_defaults_cb(_card_key))
    _card_restore_buttons[_card_key] = _btn


# ---------------------------------------------------------------------------
# Parameter cards — named so _on_geometry_change can toggle visibility
# ---------------------------------------------------------------------------

w_card_grid = pn.Card(
    _card_restore_buttons["grid"],
    w_trans_row,
    w_gi_grid_row,
    title="Output grid",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_masks = pn.Card(
    _card_restore_buttons["masks"],
    pn.Row(w_proc_saxs_mask, w_proc_waxs_mask),
    pn.pane.Markdown(
        "*Leave blank to use the bundled PyHyperScattering defaults.  "
        "Use the **Explore → mask editor** to draw, edit, and save masks "
        "interactively, then click **↪ Use in Process** to set the path here.*",
    ),
    title="Masks",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_saxs_qrange = pn.Card(
    _card_restore_buttons["saxs_qrange"],
    pn.Row(w_proc_saxs_q_cutoff, w_proc_saxs_agbh_ring, w_proc_saxs_q_margin),
    pn.pane.Markdown(
        "*q cutoff = 0 → auto from silver behenate calibration.  "
        "Ring order and margin fraction control the AgBh auto-calculation.*",
    ),
    title="SAXS Q-range / aperture",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_geometry = pn.Card(
    _card_restore_buttons["geometry"],
    w_saxs_geom_section,
    pn.pane.Markdown("**WAXS beam-centre Δ (px)**"),
    pn.Row(w_proc_waxs_row_delta, w_proc_waxs_col_delta, w_proc_waxs_col_per_arc),
    pn.pane.Markdown(
        "*Beam-centre deltas are added to values from metadata.  "
        "\"WAXS col/arc°\" compensates column drift per degree of "
        "waxs_arc (GI default: 0.08).*",
    ),
    title="Geometry corrections",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_dezinger = pn.Card(
    _card_restore_buttons["dezinger"],
    pn.Row(w_proc_dezinger, w_proc_dezinger_kernel, w_proc_pixel_splitting),
    pn.pane.Markdown(
        "*σ = 0 disables hot-pixel rejection.  Kernel is the median-filter "
        "window size (odd integer).  GI default σ is 30 000.  "
        "Pixel splitting N×N sub-divides each pixel for fractional "
        "binning (1 = no splitting, 2–4 typical).*",
    ),
    title="Hot-pixel rejection / pixel splitting",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_intensity = pn.Card(
    _card_restore_buttons["intensity"],
    w_proc_solid_angle,
    title="Intensity corrections",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_gi = pn.Card(
    _card_restore_buttons["gi"],
    pn.Row(
        w_proc_incident_angle, w_proc_incident_angle_auto,
        w_proc_theta_offset,
    ),
    pn.Row(w_proc_beamstop_max_arc),
    pn.pane.Markdown(
        "*Auto α_i: detect incident angle from sample name or motor "
        "positions.  θ offset is added to (stage_th + piezo_th) during "
        "auto-detection.  Beamstop mask is applied for |arc| ≤ max.*",
    ),
    title="Grazing-incidence parameters",
    collapsed=False, sizing_mode="stretch_width",
)

w_card_backend = pn.Card(
    _card_restore_buttons["backend"],
    pn.Row(w_proc_saxs_rotate, w_proc_waxs_flip),
    pn.Row(w_proc_waxs_qx_shift, w_proc_waxs_qy_shift),
    pn.pane.Markdown(
        "*Display orientation and global q-space shift for the WAXS "
        "detector.*",
    ),
    title="Backend / display options",
    collapsed=True, sizing_mode="stretch_width",
)

w_card_dynamic_mask = pn.Card(
    _card_restore_buttons["dynamic_mask"],
    w_proc_dynamic_mask,
    pn.pane.Markdown("**WAXS shadow on SAXS**"),
    pn.Row(
        w_proc_dyn_shadow_enabled, w_proc_dyn_shadow_beam_deg,
        w_proc_dyn_shadow_clear_deg,
    ),
    pn.pane.Markdown("**Aperture mask**"),
    pn.Row(
        w_proc_dyn_aper_enabled, w_proc_dyn_aper_agbh_ring,
        w_proc_dyn_aper_q_margin, w_proc_dyn_aper_q_cutoff,
    ),
    pn.pane.Markdown(
        "*Per-frame WAXS-shadow and aperture masking on the SAXS "
        "detector.  q cutoff = 0 → auto from AgBh ring.*",
    ),
    title="Dynamic SAXS masking",
    collapsed=True, sizing_mode="stretch_width",
)

w_card_waxs_cal = pn.Card(
    _card_restore_buttons["waxs_cal"],
    pn.Row(w_proc_waxs_energy, w_proc_waxs_dist, w_proc_waxs_pixel),
    pn.Row(w_proc_waxs_beam_row, w_proc_waxs_beam_col),
    w_proc_waxs_panel_cols,
    pn.Row(w_proc_waxs_panel_offsets, w_proc_waxs_panel_delta),
    pn.Row(w_proc_waxs_panel_row_shifts, w_proc_waxs_panel_col_shifts),
    pn.Row(
        w_proc_waxs_theta_zero, w_proc_waxs_offset_x, w_proc_waxs_offset_z,
    ),
    pn.Row(
        w_proc_waxs_col_arc_cal, w_proc_waxs_qh_sign,
        w_proc_waxs_qv_sign, w_proc_waxs_rot_k,
    ),
    pn.pane.Markdown(
        "*Override WAXSCalibration defaults.  Only changed values are sent — "
        "leave unchanged to use PyHyper's calibration.  In GI mode these are "
        "passed as `waxs_cal_overrides`.*",
    ),
    title="Advanced WAXS calibration",
    collapsed=True, sizing_mode="stretch_width",
)

w_card_waxs_mask_adv = pn.Card(
    _card_restore_buttons["waxs_mask_adv"],
    pn.Row(w_proc_waxs_bsx_ref),
    pn.pane.Markdown(
        "*BSX ref = 0 → auto-derived from metadata.  "
        "Beamstop max |arc| is set in the GI parameters card above.*",
    ),
    title="Advanced WAXS masking",
    collapsed=True, sizing_mode="stretch_width",
)


def _on_geometry_change(event):
    """Show/hide GI vs transmission params."""
    is_gi = event.new == "grazing"
    # Grid rows
    w_gi_grid_row.visible = is_gi
    w_trans_row.visible = not is_gi
    # SAXS-specific (hidden in GI)
    w_proc_saxs_mask.visible = not is_gi
    w_saxs_geom_section.visible = not is_gi
    # Transmission-only cards
    w_card_saxs_qrange.visible = not is_gi
    w_card_intensity.visible = not is_gi
    w_card_backend.visible = not is_gi
    w_card_dynamic_mask.visible = not is_gi
    # GI-only card
    w_card_gi.visible = is_gi


w_proc_geometry.param.watch(_on_geometry_change, "value")
# Initial visibility
w_gi_grid_row.visible = False
w_card_gi.visible = False

w_btn_process = pn.widgets.Button(
    name="⚙ Process", button_type="success", width=110,
)
w_btn_add_collection = pn.widgets.Button(
    name="+ Add to Collection", button_type="primary", width=150, disabled=True,
)
w_proc_status = pn.pane.Markdown("*Select a scan and click Process.*")
w_proc_spinner = pn.indicators.LoadingSpinner(value=False, size=40, visible=False)
w_proc_iq_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=400)
w_proc_2d_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=500)
w_proc_frame_slider = pn.widgets.IntSlider(
    name="Frame", start=0, end=1, value=0, step=1, width=400,
)

# Display mode: merged (summed 2D + averaged I(q)) vs per-frame (slider + individual curves)
w_proc_iq_mode = pn.widgets.RadioButtonGroup(
    name="Display mode", options=["merged", "per-frame"], value="merged", width=200,
)
w_proc_iq_label = pn.widgets.Select(
    name="Frame label", options=["(frame #)"], value="(frame #)", width=180,
)

# Plot style selector — controls line cuts, 1D I(q), and collection plots
_PLOT_STYLES = ["markers", "line", "both"]
w_plot_style = pn.widgets.Select(
    name="Plot style", options=_PLOT_STYLES, value="markers", width=120,
)

_proc_result_cache = {"result": None, "gi_result": None}

# Guard: suppress _update_proc_2d while _on_process is building its own plot
_processing_guard = {"active": False}

# ---------------------------------------------------------------------------
# Geometry cache monitoring / control
# ---------------------------------------------------------------------------

w_cache_enabled = pn.widgets.Checkbox(
    name="Cache geometry between reductions", value=True, width=250,
)
w_cache_info = pn.pane.Markdown("*Click Refresh to view geometry cache status.*")
w_btn_cache_refresh = pn.widgets.Button(
    name="🔄 Refresh", button_type="default", width=100,
)
w_btn_cache_clear = pn.widgets.Button(
    name="🗑 Clear cache", button_type="warning", width=120,
)


def _refresh_cache_info(_event=None):
    """Update the cache info display."""
    info = geometry_cache_info()
    n = info.get("size", 0)
    mb = info.get("estimated_mb", 0.0)
    keys = info.get("keys", [])
    lines = [
        f"**Cached geometries:** {n}",
        f"**Estimated memory:** {mb:.1f} MB",
    ]
    if keys:
        lines.append("")
        lines.append("**Entries:**")
        for k in keys:
            lines.append(f"- `{k}`")
    w_cache_info.object = "\n".join(lines)


def _clear_cache(_event=None):
    """Clear the geometry cache and refresh the display."""
    clear_geometry_cache()
    _refresh_cache_info()
    pn.state.notifications.info("Geometry cache cleared.")


w_btn_cache_refresh.on_click(_refresh_cache_info)
w_btn_cache_clear.on_click(_clear_cache)

# ---------------------------------------------------------------------------
# Cross-sections — interactive horizontal/vertical slices on the 2D map
# ---------------------------------------------------------------------------
#
# Persisted across scans so the same set of cuts is automatically re-applied
# each time a new scan is processed.  Each entry is:
#   {"kind": "h"|"v", "center": float, "width": float}
# For an h-cut, ``center`` is in y-units (chi or qz) and ``width`` is the
# slice extent along y.  For a v-cut, both are in x-units (q or qxy).
_persisted_cuts: list[dict] = []

# Cache of the data currently rendered in the Process tab's 2D plot, used
# to recompute cross sections when cuts move/resize or the frame changes.
_proc_2d_cache: dict[str, Any] = {
    "x": None, "y": None, "image": None,
    "x_label": "", "y_label": "",
    "title": "",
    "cuts_source": None,
    "cut_renderer": None,
}

# Recursion guard for the cuts ColumnDataSource.on_change callback —
# avoids feedback loops when we snap rectangles back to canonical extents.
_cuts_guard = {"in_progress": False}

_CUT_FILL = {"h": "#1f77b4", "v": "#d62728"}
_CUT_LINE = {"h": "#0a3a6e", "v": "#7a1414"}


def _cuts_to_source_data(cuts: list[dict]) -> dict:
    """Project the persisted cuts list into Bokeh Rect glyph columns."""
    cache = _proc_2d_cache
    x = cache["x"]
    y = cache["y"]
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


def _source_data_to_cuts(data: dict) -> list[dict]:
    """Inverse of ``_cuts_to_source_data`` — also classifies any newly drawn
    boxes (which lack a ``kind``) by aspect ratio against the plot extents."""
    cache = _proc_2d_cache
    x = cache["x"]
    y = cache["y"]
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
            # New box drawn via shift-drag in the toolbar — classify by
            # how it compares to the data span on each axis.
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


def _compute_cross_section_from_arrays(cut: dict, x, y, img):
    """Compute a 1-D cross-section from explicit arrays.

    Returns ``(axis, intensity, axis_label_key)`` or ``None``.
    ``axis_label_key`` is 'x' for h-cuts and 'y' for v-cuts.
    """
    if x is None or y is None or img is None:
        return None
    c = float(cut["center"])
    w = float(cut["width"]) or 0.0
    half = max(w / 2.0, 0.0)
    if cut["kind"] == "h":
        mask = (y >= c - half) & (y <= c + half)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(y - c)))
            section = img[idx, :].astype(float)
        else:
            section = np.nanmean(img[mask, :], axis=0)
        return x, section, "x"
    mask = (x >= c - half) & (x <= c + half)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(x - c)))
        section = img[:, idx].astype(float)
    else:
        section = np.nanmean(img[:, mask], axis=1)
    return y, section, "y"


def _compute_cross_section(cut: dict):
    """Return ``(axis, intensity, axis_label)`` for one cut, or ``None``."""
    cache = _proc_2d_cache
    x = cache["x"]
    y = cache["y"]
    img = cache["image"]
    out = _compute_cross_section_from_arrays(cut, x, y, img)
    if out is None:
        return None
    axis, section, key = out
    return axis, section, cache[f"{key}_label"]


def _get_all_frame_images():
    """Return a list of (x, y, image) tuples for each frame, or empty list.

    Uses the cached processing result to retrieve all per-frame 2D arrays.
    """
    gi = _proc_result_cache.get("gi_result")
    trans = _proc_result_cache.get("result")
    if gi is not None:
        qxy = np.asarray(gi.qxy_grid)
        qz = np.asarray(gi.qz_grid)
        frames = []
        for f in gi.frames:
            img = np.where(np.isfinite(f), f, np.nan).astype(np.float64)
            # GI images are (n_qxy, n_qz), need (n_qz, n_qxy) for cuts
            if img.shape == (len(qxy), len(qz)):
                img = img.T
            frames.append((qxy, qz, img))
        return frames
    if trans is not None:
        qchi = getattr(trans, "merged_qchi", None)
        if qchi is not None and "frame" in qchi.dims:
            q = qchi["q"].values if "q" in qchi.coords else np.arange(qchi["intensity"].shape[-1])
            chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(qchi["intensity"].shape[-2] if qchi["intensity"].ndim > 2 else qchi["intensity"].shape[0])
            n_frames = qchi.sizes["frame"]
            frames = []
            for i in range(n_frames):
                img_raw = qchi["intensity"].isel(frame=i).values
                # Ensure (n_chi, n_q)
                if img_raw.shape == (len(q), len(chi)):
                    img_raw = img_raw.T
                img = np.where(np.isfinite(img_raw), img_raw, np.nan).astype(np.float64)
                frames.append((q, chi, img))
            return frames
        # Fallback: use q_chi_frames from saxs/waxs result
        pf = _get_per_frame_qchi(trans)
        if pf is not None:
            q, chi, frames_3d = pf
            return [(q, chi, frames_3d[i]) for i in range(frames_3d.shape[0])]
    return []


def _format_cut_label(i: int, cut: dict) -> str:
    arrow = "─" if cut["kind"] == "h" else "│"
    return f"{arrow} #{i + 1}: c={cut['center']:.3g}, Δ={cut['width']:.3g}"


def _refresh_cuts_table():
    rows = [{
        "#": i + 1,
        "kind": "horizontal" if c["kind"] == "h" else "vertical",
        "center": round(c["center"], 6),
        "width": round(c["width"], 6),
    } for i, c in enumerate(_persisted_cuts)]
    cols = ["#", "kind", "center", "width"]
    w_cuts_table.value = (pd.DataFrame(rows, columns=cols)
                          if rows else pd.DataFrame(columns=cols))


def _add_trace(fig, x, y, *, color, width=1.2, alpha=1.0, legend_label=None, size=4):
    """Add a trace to a Bokeh figure respecting the current plot style setting."""
    style = w_plot_style.value
    kw = dict(legend_label=legend_label) if legend_label else {}
    if style in ("line", "both"):
        fig.line(x, y, line_color=color, line_width=width, alpha=alpha, **kw)
    if style in ("markers", "both"):
        fig.scatter(x, y, color=color, size=size, alpha=alpha, **kw)


def _render_cuts_plot():
    """Redraw cross-section plots: separate axes for h and v cuts.

    In per-frame mode, overlays all frames for each cut with color-coded
    labels (like the per-frame I(q) plot).
    """
    from bokeh.plotting import figure as bk_figure
    from bokeh.layouts import column as bk_column

    if not _persisted_cuts or _proc_2d_cache["image"] is None:
        w_proc_cuts_plot.object = None
        return

    h_cuts = [(i, c) for i, c in enumerate(_persisted_cuts) if c["kind"] == "h"]
    v_cuts = [(i, c) for i, c in enumerate(_persisted_cuts) if c["kind"] == "v"]

    x_log = w_cuts_log_x.value
    y_log = w_cuts_log_y.value
    x_type = "log" if x_log else "linear"
    y_type = "log" if y_log else "linear"

    # Determine if we should show per-frame overlaid cuts
    per_frame_mode = (w_proc_iq_mode.value == "per-frame")
    all_frames = []
    frame_labels = []
    if per_frame_mode:
        all_frames = _get_all_frame_images()
        if len(all_frames) > 1:
            frame_labels = _get_frame_labels()
        else:
            per_frame_mode = False  # Only 1 frame, fall back to single

    plots = []

    if h_cuts:
        for cut_idx, cut in h_cuts:
            p_h = bk_figure(
                title=f"Horizontal cut #{cut_idx + 1} \u2014 I(q)",
                height=300,
                sizing_mode="stretch_width",
                x_axis_type=x_type, y_axis_type=y_type,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll="wheel_zoom",
            )
            plotted = False
            if per_frame_mode:
                # Overlay all frames for this cut
                from bokeh.palettes import Category10, Turbo256
                n_frames = len(all_frames)
                if n_frames <= 10:
                    colors = Category10[max(3, n_frames)][:n_frames]
                else:
                    step = max(1, len(Turbo256) // n_frames)
                    colors = [Turbo256[i * step % len(Turbo256)] for i in range(n_frames)]
                for fi, (fx, fy, fimg) in enumerate(all_frames):
                    out = _compute_cross_section_from_arrays(cut, fx, fy, fimg)
                    if out is None:
                        continue
                    axis, section, _key = out
                    finite = np.isfinite(section) & ((section > 0) if y_log else np.ones_like(section, dtype=bool))
                    if not np.any(finite):
                        continue
                    lbl = frame_labels[fi] if fi < len(frame_labels) else f"frame {fi}"
                    _add_trace(p_h, axis[finite], section[finite],
                               color=colors[fi], width=1.0, alpha=0.8,
                               legend_label=lbl)
                    plotted = True
            else:
                # Single-frame (current cache image)
                out = _compute_cross_section(cut)
                if out is not None:
                    axis, section, axis_label = out
                    finite = np.isfinite(section) & ((section > 0) if y_log else np.ones_like(section, dtype=bool))
                    if np.any(finite):
                        _add_trace(p_h, axis[finite], section[finite],
                                   color=_CUT_FILL["h"], width=1.4, alpha=0.9,
                                   legend_label=_format_cut_label(cut_idx, cut))
                        plotted = True
            if plotted:
                p_h.xaxis.axis_label = _proc_2d_cache["x_label"]
                p_h.yaxis.axis_label = "I"
                p_h.legend.click_policy = "hide"
                p_h.legend.label_text_font_size = "8pt"
                if per_frame_mode and len(all_frames) > 20:
                    p_h.legend.visible = False
                plots.append(p_h)

    if v_cuts:
        for cut_idx, cut in v_cuts:
            p_v = bk_figure(
                title=f"Vertical cut #{cut_idx + 1} \u2014 I(\u03c7)",
                height=300,
                sizing_mode="stretch_width",
                x_axis_type=x_type, y_axis_type=y_type,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll="wheel_zoom",
            )
            plotted = False
            if per_frame_mode:
                from bokeh.palettes import Category10, Turbo256
                n_frames = len(all_frames)
                if n_frames <= 10:
                    colors = Category10[max(3, n_frames)][:n_frames]
                else:
                    step = max(1, len(Turbo256) // n_frames)
                    colors = [Turbo256[i * step % len(Turbo256)] for i in range(n_frames)]
                for fi, (fx, fy, fimg) in enumerate(all_frames):
                    out = _compute_cross_section_from_arrays(cut, fx, fy, fimg)
                    if out is None:
                        continue
                    axis, section, _key = out
                    finite = np.isfinite(section) & ((section > 0) if y_log else np.ones_like(section, dtype=bool))
                    if not np.any(finite):
                        continue
                    lbl = frame_labels[fi] if fi < len(frame_labels) else f"frame {fi}"
                    _add_trace(p_v, axis[finite], section[finite],
                               color=colors[fi], width=1.0, alpha=0.8,
                               legend_label=lbl)
                    plotted = True
            else:
                out = _compute_cross_section(cut)
                if out is not None:
                    axis, section, axis_label = out
                    finite = np.isfinite(section) & ((section > 0) if y_log else np.ones_like(section, dtype=bool))
                    if np.any(finite):
                        _add_trace(p_v, axis[finite], section[finite],
                                   color=_CUT_FILL["v"], width=1.4, alpha=0.9,
                                   legend_label=_format_cut_label(cut_idx, cut))
                        plotted = True
            if plotted:
                p_v.xaxis.axis_label = _proc_2d_cache["y_label"]
                p_v.yaxis.axis_label = "I"
                p_v.legend.click_policy = "hide"
                p_v.legend.label_text_font_size = "8pt"
                if per_frame_mode and len(all_frames) > 20:
                    p_v.legend.visible = False
                plots.append(p_v)

    if not plots:
        w_proc_cuts_plot.object = None
        return
    w_proc_cuts_plot.object = bk_column(*plots, sizing_mode="stretch_width")


def _on_cuts_data_change(attr, old, new):
    """Bokeh callback: source.data mutated by user (drag/resize/draw/delete)."""
    if _cuts_guard["in_progress"]:
        return
    cuts = _source_data_to_cuts(new)
    _persisted_cuts.clear()
    _persisted_cuts.extend(cuts)
    # Snap perpendicular extents back to full plot range.
    snapped = _cuts_to_source_data(cuts)
    if snapped != dict(new):
        _cuts_guard["in_progress"] = True
        try:
            src = _proc_2d_cache.get("cuts_source")
            if src is not None:
                src.data = snapped
        finally:
            _cuts_guard["in_progress"] = False
    _refresh_cuts_table()
    _render_cuts_plot()


def _attach_cuts_to_figure(p, x, y, image, x_label, y_label, title=""):
    """Cache the figure's data and wire the cross-section overlay + tool."""
    from bokeh.models import BoxEditTool, ColumnDataSource

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    img_arr = np.asarray(image)
    # Normalize to (n_y, n_x) so _compute_cross_section can index correctly.
    if img_arr.shape == (len(x_arr), len(y_arr)):
        img_arr = img_arr.T
    _proc_2d_cache.update(
        x=x_arr, y=y_arr,
        image=img_arr,
        x_label=x_label, y_label=y_label, title=title,
    )
    src = ColumnDataSource(data=_cuts_to_source_data(_persisted_cuts))
    rect = p.rect(
        x="x", y="y", width="width", height="height",
        fill_color="fill_color", fill_alpha=0.18,
        line_color="line_color", line_width=2, line_dash="dashed",
        source=src,
    )
    edit_tool = BoxEditTool(renderers=[rect], num_objects=20)
    p.add_tools(edit_tool)
    src.on_change("data", _on_cuts_data_change)
    _proc_2d_cache["cuts_source"] = src
    _proc_2d_cache["cut_renderer"] = rect
    # Refresh the 1D pane so the lines reflect the new image (e.g. after
    # frame slider movement) even when cut centres are unchanged.
    try:
        _render_cuts_plot()
    except Exception as exc:
        log.warning("_render_cuts_plot in _attach_cuts: %s", exc)


def _add_cut(kind: str):
    """Create a new cut centred on the plot with a default ~5% slice width."""
    cache = _proc_2d_cache
    x = cache["x"]
    y = cache["y"]
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        w_proc_status.object = "*Process a scan first, then add cuts.*"
        return
    if kind == "h":
        ymin, ymax = float(np.min(y)), float(np.max(y))
        center = (ymin + ymax) / 2.0
        width = (ymax - ymin) * 0.05 or 1.0
    else:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        center = (xmin + xmax) / 2.0
        width = (xmax - xmin) * 0.05 or 1.0
    _persisted_cuts.append({"kind": kind, "center": center, "width": width})
    src = cache.get("cuts_source")
    if src is not None:
        _cuts_guard["in_progress"] = True
        try:
            src.data = _cuts_to_source_data(_persisted_cuts)
        finally:
            _cuts_guard["in_progress"] = False
    _refresh_cuts_table()
    _render_cuts_plot()


def _on_add_hcut(_event):
    _add_cut("h")


def _on_add_vcut(_event):
    _add_cut("v")


def _on_clear_cuts(_event):
    _persisted_cuts.clear()
    src = _proc_2d_cache.get("cuts_source")
    if src is not None:
        _cuts_guard["in_progress"] = True
        try:
            src.data = _cuts_to_source_data(_persisted_cuts)
        finally:
            _cuts_guard["in_progress"] = False
    _refresh_cuts_table()
    _render_cuts_plot()


# Cross-section widgets
w_btn_add_hcut = pn.widgets.Button(
    name="+ Horizontal cut", button_type="primary", width=140,
    description="Add a horizontal slice at the centre of the 2D plot. "
                "Drag/resize the dashed rectangle to change centre & width.",
)
w_btn_add_vcut = pn.widgets.Button(
    name="+ Vertical cut", button_type="primary", width=130,
    description="Add a vertical slice at the centre of the 2D plot.",
)
w_btn_clear_cuts = pn.widgets.Button(
    name="Clear all cuts", button_type="warning", width=120,
)
w_cuts_log_x = pn.widgets.Checkbox(name="Log x", value=False, width=70)
w_cuts_log_y = pn.widgets.Checkbox(name="Log y", value=False, width=70)


def _on_cuts_log_change(*_events):
    _render_cuts_plot()


w_cuts_log_x.param.watch(_on_cuts_log_change, "value")
w_cuts_log_y.param.watch(_on_cuts_log_change, "value")


def _on_plot_style_change(*_events):
    _render_cuts_plot()


w_plot_style.param.watch(_on_plot_style_change, "value")

w_cuts_table = pn.widgets.Tabulator(
    value=pd.DataFrame(columns=["#", "kind", "center", "width"]),
    show_index=False, sizing_mode="stretch_width", height=160,
    configuration={"layout": "fitColumns", "rowHeight": 22},
)
w_proc_cuts_plot = pn.pane.Bokeh(
    object=None, sizing_mode="stretch_width", height=650,
)

w_btn_add_hcut.on_click(_on_add_hcut)
w_btn_add_vcut.on_click(_on_add_vcut)
w_btn_clear_cuts.on_click(_on_clear_cuts)


# ---------------------------------------------------------------------------
# Widgets — Collection (delegated to smi_browser.ui.collection)
# ---------------------------------------------------------------------------

_coll_ns = _coll_mod.wire(_collection, plot_style_widget=w_plot_style)

w_coll_table = _coll_ns.coll_table
w_btn_coll_remove = _coll_ns.btn_remove
w_coll_label = _coll_ns.label_select
w_coll_compare_plot = _coll_ns.compare_plot


# ---------------------------------------------------------------------------
# Helpers — 2D result plotting
# ---------------------------------------------------------------------------

# Cache for per-frame q-chi regridding (expensive scipy interpolation)
_per_frame_qchi_cache: dict[str, Any] = {"uid": None, "data": None}


def _get_per_frame_qchi(result):
    """Extract per-frame q-chi data from result.saxs/waxs['q_chi_frames'].

    Returns (q, chi, frames_3d) where frames_3d has shape (n_frames, n_chi, n_q),
    or None if no per-frame data is available.  Results are cached by UID.
    """
    uid = getattr(result, "uid", None)
    if uid and _per_frame_qchi_cache["uid"] == uid and _per_frame_qchi_cache["data"] is not None:
        return _per_frame_qchi_cache["data"]
    saxs = getattr(result, "saxs", None)
    waxs = getattr(result, "waxs", None)
    saxs_frames = saxs.get("q_chi_frames") if saxs else None
    waxs_frames = waxs.get("q_chi_frames") if waxs else None

    if saxs_frames is None and waxs_frames is None:
        return None

    # Use the merged_qchi grid for q/chi (same grid used during processing)
    qchi = result.merged_qchi
    if qchi is None:
        return None
    q_grid = qchi["q"].values if "q" in qchi.coords else None
    chi_grid = qchi["chi"].values if "chi" in qchi.coords else None
    if q_grid is None or chi_grid is None:
        return None

    n_q = len(q_grid)
    n_chi = len(chi_grid)

    # Determine number of frames
    if saxs_frames is not None and waxs_frames is not None:
        n_frames = max(saxs_frames.sizes.get("frame", 1),
                       waxs_frames.sizes.get("frame", 1))
    elif saxs_frames is not None:
        n_frames = saxs_frames.sizes.get("frame", 1)
    else:
        n_frames = waxs_frames.sizes.get("frame", 1)

    if n_frames <= 1:
        return None

    from scipy.interpolate import RegularGridInterpolator

    def _regrid_frame(src_q, src_chi, data):
        """Regrid a single frame (q, chi) -> (n_q, n_chi) on the merged grid."""
        finite_data = np.where(np.isfinite(data), data, 0.0)
        interp = RegularGridInterpolator(
            (src_q, src_chi), finite_data,
            method="nearest", bounds_error=False, fill_value=np.nan,
        )
        tq, tc = np.meshgrid(q_grid, chi_grid, indexing="ij")
        return interp((tq, tc))

    frames_3d = np.full((n_frames, n_chi, n_q), np.nan, dtype=np.float64)

    for fi in range(n_frames):
        s_I = None
        s_N = None
        w_I = None
        w_N = None

        if saxs_frames is not None and fi < saxs_frames.sizes.get("frame", 0):
            sq = np.asarray(saxs_frames["q"].values, dtype=float)
            schi = np.asarray(saxs_frames["chi"].values, dtype=float)
            s_I_raw = saxs_frames["intensity"].isel(frame=fi).values.astype(float)
            s_N_raw = saxs_frames["counts"].isel(frame=fi).values.astype(float)
            s_I = _regrid_frame(sq, schi, s_I_raw)
            s_N = _regrid_frame(sq, schi, np.where(np.isfinite(s_N_raw), s_N_raw, 0.0))

        if waxs_frames is not None and fi < waxs_frames.sizes.get("frame", 0):
            wq = np.asarray(waxs_frames["q"].values, dtype=float)
            wchi = np.asarray(waxs_frames["chi"].values, dtype=float)
            w_I_raw = waxs_frames["intensity"].isel(frame=fi).values.astype(float)
            w_N_raw = waxs_frames["counts"].isel(frame=fi).values.astype(float)
            w_I = _regrid_frame(wq, wchi, w_I_raw)
            w_N = _regrid_frame(wq, wchi, np.where(np.isfinite(w_N_raw), w_N_raw, 0.0))

        # Count-weighted merge (same logic as merge_q_chi_weighted)
        if s_I is not None and w_I is not None:
            total_N = s_N + w_N
            with np.errstate(divide="ignore", invalid="ignore"):
                merged = np.where(
                    total_N > 0,
                    (np.nan_to_num(s_I, nan=0.0) * s_N
                     + np.nan_to_num(w_I, nan=0.0) * w_N) / total_N,
                    np.nan,
                )
        elif s_I is not None:
            merged = s_I
        elif w_I is not None:
            merged = w_I
        else:
            continue

        # merged is (n_q, n_chi) — transpose to (n_chi, n_q)
        frames_3d[fi] = merged.T

    out = (q_grid, chi_grid, frames_3d)
    if uid:
        _per_frame_qchi_cache["uid"] = uid
        _per_frame_qchi_cache["data"] = out
    return out


def _plot_2d_transmission(result, frame_idx=None):
    """Plot q-vs-chi as an interactive Bokeh figure."""
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar

    qchi = result.merged_qchi

    # Per-frame display: try merged_qchi frame dim first, then saxs/waxs q_chi_frames
    if frame_idx is not None:
        if "frame" in qchi.dims:
            img = qchi["intensity"].isel(frame=frame_idx).values
            q = qchi["q"].values if "q" in qchi.coords else np.arange(img.shape[-1])
            chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(img.shape[0])
        else:
            pf = _get_per_frame_qchi(result)
            if pf is not None:
                q, chi, frames_3d = pf
                img = frames_3d[frame_idx]  # already (n_chi, n_q)
            else:
                img = qchi["intensity"].values
                q = qchi["q"].values if "q" in qchi.coords else np.arange(img.shape[-1])
                chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(img.shape[0])
                frame_idx = None  # fallback to merged
        title = f"q vs χ — frame {frame_idx}" if frame_idx is not None else "q vs χ (merged)"
    else:
        img = qchi["intensity"].values
        q = qchi["q"].values if "q" in qchi.coords else np.arange(img.shape[-1])
        chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(img.shape[0])
        title = "q vs χ (merged)"

    # Ensure img has shape (n_chi, n_q) so Bokeh renders chi on y, q on x.
    if img.shape == (len(q), len(chi)):
        img = img.T

    display = np.where(np.isfinite(img), img, 0).astype(np.float32)
    # Keep NaNs for cuts so nanmean excludes masked pixels properly
    cuts_image = np.where(np.isfinite(img), img, np.nan).astype(np.float64)
    finite = img[np.isfinite(img) & (img > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 2)), 1e-6)
        vhi = float(np.percentile(finite, 99.5))
        mapper = LogColorMapper(palette="Turbo256", low=vlo, high=max(vhi, vlo * 2))
    else:
        mapper = LinearColorMapper(palette="Greys256", low=float(np.nanmin(display)), high=max(float(np.nanmax(display)), 1))

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
    _attach_cuts_to_figure(p, q, chi, cuts_image,
                           x_label="q (nm⁻¹)", y_label="χ (°)", title=title)
    return p


def _plot_2d_gi(gi_result, frame_idx=None):
    """Plot qxy-vs-qz as an interactive Bokeh figure."""
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
    # Keep NaNs for cuts so nanmean excludes masked pixels properly
    cuts_image = np.where(np.isfinite(img), img, np.nan).astype(np.float64)
    finite = img[np.isfinite(img) & (img > 0)]
    if finite.size:
        vlo = max(float(np.percentile(finite, 2)), 1e-6)
        vhi = float(np.percentile(finite, 99.5))
        mapper = LogColorMapper(palette="Viridis256", low=vlo, high=max(vhi, vlo * 2))
    else:
        mapper = LinearColorMapper(palette="Greys256", low=float(np.nanmin(display)), high=max(float(np.nanmax(display)), 1))

    # qxy/qz are 1-D grids; image is (n_qxy, n_qz), displayed transposed
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
    _attach_cuts_to_figure(p, qxy, qz, cuts_image.T,
                           x_label="q_xy (nm⁻¹)", y_label="q_z (nm⁻¹)",
                           title=title)
    return p


def _update_proc_2d(event):
    """Redraw the 2D map when the frame slider changes."""
    if _processing_guard["active"]:
        return  # _on_process will set the 2D plot itself
    gi = _proc_result_cache.get("gi_result")
    trans = _proc_result_cache.get("result")
    idx = event.new
    try:
        if gi is not None:
            w_proc_2d_plot.object = _plot_2d_gi(gi, frame_idx=idx)
        elif trans is not None:
            w_proc_2d_plot.object = _plot_2d_transmission(trans, frame_idx=idx)
        # Update frame slider label
        _update_frame_slider_label()
    except Exception as exc:
        log.warning("2D plot update failed: %s", exc)


w_proc_frame_slider.param.watch(_update_proc_2d, "value")


def _get_frame_labels():
    """Build a list of frame labels using primary scalars and the selected label column."""
    label_col = w_proc_iq_label.value
    result = _proc_result_cache.get("result")
    if result is None:
        return []
    # Determine number of frames
    pf_iq = getattr(result, "per_frame_iq", None)
    qchi = getattr(result, "merged_qchi", None)
    if pf_iq is not None and "frame" in pf_iq.dims:
        n_frames = pf_iq.sizes["frame"]
    elif qchi is not None and "frame" in qchi.dims:
        n_frames = qchi.sizes["frame"]
    else:
        return ["merged"]
    if not label_col or label_col == "(frame #)":
        return [f"frame {i}" for i in range(n_frames)]
    # Prefer bundled primary scalars from per_frame_iq
    vals = None
    if pf_iq is not None and label_col in pf_iq.data_vars:
        vals = pf_iq[label_col].values
    else:
        # Fallback to primary table
        df = w_primary_table.value
        if df is not None and label_col in df.columns:
            vals = df[label_col].values
    if vals is not None:
        labels = []
        for i in range(n_frames):
            if i < len(vals):
                try:
                    labels.append(f"{label_col}={float(vals[i]):.4g}")
                except (ValueError, TypeError):
                    labels.append(f"{label_col}={vals[i]}")
            else:
                labels.append(f"frame {i}")
        return labels
    return [f"frame {i}" for i in range(n_frames)]


def _build_proc_iq_plot():
    """Build the I(q) Bokeh figure in either merged or per-frame mode."""
    from bokeh.plotting import figure as bk_figure

    result = _proc_result_cache.get("result")
    if result is None or not hasattr(result, "merged_iq"):
        w_proc_iq_plot.object = None
        return

    iq = result.merged_iq
    mode = w_proc_iq_mode.value
    uid = _state.get("selected_uid") or ""

    if mode == "per-frame" and hasattr(result, "per_frame_iq") and result.per_frame_iq is not None:
        # Per-frame I(q) from PyHyper (preferred path)
        pf_iq = result.per_frame_iq
        q = pf_iq["q"].values
        frame_labels = _get_frame_labels()
        n_frames = pf_iq.sizes.get("frame", 1)

        # Use a perceptually-spaced colormap for many frames
        from bokeh.palettes import Category10, Turbo256
        if n_frames <= 10:
            colors = Category10[max(3, n_frames)][:n_frames]
        else:
            step = max(1, len(Turbo256) // n_frames)
            colors = [Turbo256[i * step % len(Turbo256)] for i in range(n_frames)]

        p = bk_figure(
            title=f"{uid[:8]} — per-frame I(q)",
            width=1000, height=400,
            x_axis_type="log", y_axis_type="log",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        merged_iq = getattr(result, "merged_iq", None)
        det_mode = _iq_detector_mode(merged_iq) if merged_iq is not None else "both"
        y_key = "I"
        if det_mode == "saxs_only" and "saxs_I" in pf_iq:
            y_key = "saxs_I"
        elif det_mode == "waxs_only" and "waxs_I" in pf_iq:
            y_key = "waxs_I"

        for i in range(n_frames):
            I_frame = pf_iq[y_key].isel(frame=i).values
            mask = np.isfinite(I_frame) & (I_frame > 0)
            if mask.any():
                lbl = frame_labels[i] if i < len(frame_labels) else f"frame {i}"
                _add_trace(p, q[mask], I_frame[mask], color=colors[i],
                           width=0.9, alpha=0.8, legend_label=lbl)
        p.xaxis.axis_label = "q (nm⁻¹)"
        p.yaxis.axis_label = "I(q)"
        if n_frames <= 20:
            p.legend.click_policy = "hide"
            p.legend.label_text_font_size = "8pt"
        else:
            p.legend.visible = False
        w_proc_iq_plot.object = p

    elif mode == "per-frame":
        # Fallback: per-frame from merged_qchi by integrating over chi
        qchi = getattr(result, "merged_qchi", None)
        if qchi is None or "frame" not in qchi.dims:
            # No per-frame data available — show merged with a note
            _build_merged_iq_plot(result, uid, note=" (no per-frame data)")
            return
        q = qchi["q"].values if "q" in qchi.coords else np.arange(qchi["intensity"].shape[-1])
        n_frames = qchi.sizes["frame"]
        frame_labels = _get_frame_labels()

        from bokeh.palettes import Category10, Turbo256
        if n_frames <= 10:
            colors = Category10[max(3, n_frames)][:n_frames]
        else:
            step = max(1, len(Turbo256) // n_frames)
            colors = [Turbo256[i * step % len(Turbo256)] for i in range(n_frames)]

        p = bk_figure(
            title=f"{uid[:8]} — per-frame I(q) (χ-integrated)",
            width=1000, height=400,
            x_axis_type="log", y_axis_type="log",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        for i in range(n_frames):
            img_frame = qchi["intensity"].isel(frame=i).values
            # Integrate over chi axis (axis 0 after ensuring shape is (chi, q))
            if img_frame.shape == (len(q), len(qchi["chi"].values)):
                img_frame = img_frame.T
            I_frame = np.nanmean(img_frame, axis=0)
            mask = np.isfinite(I_frame) & (I_frame > 0)
            if mask.any():
                lbl = frame_labels[i] if i < len(frame_labels) else f"frame {i}"
                _add_trace(p, q[mask], I_frame[mask], color=colors[i],
                           width=0.9, alpha=0.8, legend_label=lbl)
        p.xaxis.axis_label = "q (nm⁻¹)"
        p.yaxis.axis_label = "I(q)"
        if n_frames <= 20:
            p.legend.click_policy = "hide"
            p.legend.label_text_font_size = "8pt"
        else:
            p.legend.visible = False
        w_proc_iq_plot.object = p

    else:
        # Merged mode (default)
        _build_merged_iq_plot(result, uid)


def _build_merged_iq_plot(result, uid, note=""):
    """Render the standard merged I(q) plot."""
    from bokeh.plotting import figure as bk_figure

    iq = result.merged_iq
    q = iq["q"].values
    I = iq["I"].values

    p = bk_figure(
        title=f"{uid[:8]} — merged I(q){note}", width=1000, height=400,
        x_axis_type="log", y_axis_type="log",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )

    det_mode = _iq_detector_mode(iq)

    mask = np.isfinite(I) & (I > 0)
    if det_mode != "saxs_only" and det_mode != "waxs_only" and mask.any():
        _add_trace(p, q[mask], I[mask], color="black", width=1.2, legend_label="merged")
    if "saxs_I" in iq:
        sI = iq["saxs_I"].values
        sm = np.isfinite(sI) & (sI > 0)
        if sm.any():
            width = 1.2 if det_mode == "saxs_only" else 0.8
            alpha = 0.9 if det_mode == "saxs_only" else 0.6
            _add_trace(p, q[sm], sI[sm], color="blue", width=width, alpha=alpha, legend_label="SAXS")
    if "waxs_I" in iq:
        wI = iq["waxs_I"].values
        wm = np.isfinite(wI) & (wI > 0)
        if wm.any():
            width = 1.2 if det_mode == "waxs_only" else 0.8
            alpha = 0.9 if det_mode == "waxs_only" else 0.6
            _add_trace(p, q[wm], wI[wm], color="red", width=width, alpha=alpha, legend_label="WAXS")
    p.xaxis.axis_label = "q (nm⁻¹)"
    p.yaxis.axis_label = "I(q)"
    p.legend.click_policy = "hide"
    w_proc_iq_plot.object = p


def _iq_detector_mode(iq) -> str:
    """Classify I(q) availability as both/saxs_only/waxs_only/none."""
    has_saxs = False
    has_waxs = False
    if iq is None:
        return "none"
    if "saxs_I" in iq:
        s = iq["saxs_I"].values
        has_saxs = bool(np.isfinite(s).any() and (s > 0).any())
    if "waxs_I" in iq:
        w = iq["waxs_I"].values
        has_waxs = bool(np.isfinite(w).any() and (w > 0).any())
    if has_saxs and has_waxs:
        return "both"
    if has_saxs:
        return "saxs_only"
    if has_waxs:
        return "waxs_only"
    return "none"


def _on_proc_iq_mode_change(event):
    """Redraw both 2D map and I(q) when the user switches merged ↔ per-frame."""
    mode = event.new if hasattr(event, "new") else w_proc_iq_mode.value
    result = _proc_result_cache.get("result")
    gi = _proc_result_cache.get("gi_result")

    if mode == "per-frame":
        # Show frame slider and display selected frame
        if gi is not None:
            n_fr = len(gi.frames)
            w_proc_frame_slider.end = max(0, n_fr - 1)
            w_proc_frame_slider.visible = n_fr > 1
            idx = w_proc_frame_slider.value
            w_proc_2d_plot.object = _plot_2d_gi(gi, frame_idx=idx)
        elif result is not None:
            # Detect per-frame 2D: merged_qchi with frame dim, or q_chi_frames
            qchi = getattr(result, "merged_qchi", None)
            has_qchi_frames = qchi is not None and "frame" in qchi.dims
            if has_qchi_frames:
                n_fr = qchi.sizes["frame"]
            else:
                pf = _get_per_frame_qchi(result)
                n_fr = pf[2].shape[0] if pf is not None else 0
            if n_fr > 1:
                w_proc_frame_slider.end = max(0, n_fr - 1)
                w_proc_frame_slider.visible = True
                idx = w_proc_frame_slider.value
                w_proc_2d_plot.object = _plot_2d_transmission(result, frame_idx=idx)
            else:
                w_proc_frame_slider.visible = False
    else:
        # Merged: hide slider, show summed/merged 2D
        w_proc_frame_slider.visible = False
        if gi is not None:
            w_proc_2d_plot.object = _plot_2d_gi(gi)
        elif result is not None:
            w_proc_2d_plot.object = _plot_2d_transmission(result)

    _build_proc_iq_plot()
    _update_frame_slider_label()
    _render_cuts_plot()


def _on_proc_iq_label_change(event):
    """Redraw I(q) when the frame label column changes (only in per-frame mode)."""
    if w_proc_iq_mode.value == "per-frame":
        _build_proc_iq_plot()
        _render_cuts_plot()
    # Also update frame slider label on the 2D plot
    _update_frame_slider_label()


def _update_frame_slider_label():
    """Update the frame slider name to include primary axis label."""
    labels = _get_frame_labels()
    idx = w_proc_frame_slider.value
    if labels and idx < len(labels):
        w_proc_frame_slider.name = f"Frame — {labels[idx]}"
    else:
        w_proc_frame_slider.name = "Frame"


w_proc_iq_mode.param.watch(_on_proc_iq_mode_change, "value")
w_proc_iq_label.param.watch(_on_proc_iq_label_change, "value")


# ---------------------------------------------------------------------------
# Callbacks — Search & Browse
# ---------------------------------------------------------------------------

def _refresh_pagination():
    n_pg = max(1, _n_pages())
    pg = _state["page"]
    w_page_info.object = f"**{pg + 1}/{n_pg}**"
    w_btn_first.disabled = (pg == 0)
    w_btn_prev.disabled = (pg == 0)
    w_btn_next.disabled = (pg >= n_pg - 1)
    w_btn_last.disabled = (pg >= n_pg - 1)


def _do_search(page=0):
    _cancel.clear()
    _state["unified_filters"] = _collect_unified_filters()
    _state["page"] = page

    w_status.object = "*Searching…*"
    w_search_spinner.value = True
    w_search_spinner.visible = True

    if _cancel.is_set():
        w_search_spinner.value = False
        w_search_spinner.visible = False
        return

    try:
        df = _fetch_page()  # gets count + page data via REST
    except Exception as exc:
        w_search_spinner.value = False
        w_search_spinner.visible = False
        if _cancel.is_set():
            return
        w_status.object = f"**Error:** `{exc}`"
        return

    w_search_spinner.value = False
    w_search_spinner.visible = False

    if _cancel.is_set():
        return

    total = _state["total"]
    n_pg = _n_pages()

    if total == 0:
        w_table.value = _EMPTY_DF.copy()
        w_status.object = "**0 scans**"
        _refresh_pagination()
        return

    w_table.value = df
    w_status.object = (
        f"**{total} scan{'s' if total != 1 else ''}** — "
        f"page {page + 1}/{n_pg}"
    )
    _refresh_pagination()
    # Collapse search and update summary
    w_filter_summary.object = _filter_summary_text()
    search_card.collapsed = True
    # Persist filter state so it survives websocket reconnects
    _save_filter_state()


def _go_to_page(page):
    _state["page"] = page
    try:
        df = _fetch_page()
    except Exception as exc:
        w_status.object = f"**Page error:** `{exc}`"
        return
    w_table.value = df
    n_pg = _n_pages()
    w_status.object = f"**{_state['total']} scans** — page {page + 1}/{n_pg}"
    _refresh_pagination()
    pn.state.cache[_PAGE_CACHE_KEY] = page


def _on_reset(_event=None):
    _cancel.set()  # abort any in-flight queries immediately
    _filter_rows.clear()
    _add_filter()  # start with one empty row
    _state.update(unified_filters=[], page=0, total=0, selected_uid=None)
    w_table.value = _EMPTY_DF.copy()
    w_status.object = "*Ready*"
    _refresh_pagination()
    _reset_detail()
    w_filter_summary.object = _filter_summary_text()
    search_card.collapsed = False  # expand so user can start a new search
    # Clear saved filter state
    pn.state.cache.pop(_FILTER_CACHE_KEY, None)
    pn.state.cache.pop(_PAGE_CACHE_KEY, None)


def _reset_detail(preserve_figure=False):
    w_detail_title.object = "### Select a scan"
    w_meta_json.object = {}
    w_primary_table.value = pd.DataFrame()
    w_primary_status.object = "*Click tab to load.*"
    # Don't clear x/y options/values — they persist across scans
    w_primary_plot.object = None
    w_baseline_table.value = pd.DataFrame(columns=["source", "field", "before", "after"])
    w_baseline_status.object = "*Click tab to load.*"
    if not preserve_figure:
        # Full reset — destroy the Bokeh figure and all overlays
        w_image_thumb.object = None
        _image_cache.update(
            figure=None, source=None, mapper=None,
            fig_image_shape=None,
            mask_source=None, mask_renderer=None,
            new_mask_source=None, new_mask_renderer=None,
            draw_tool=None, edit_tool=None,
            mask_image_shape=None,
            dyn_source=None, dyn_renderer=None,
        )
    w_image_status.object = ""
    # Clear image cache field BEFORE touching slider so _on_image_slider
    # sees no field and returns early — avoids Bokeh model mutations that
    # can fail with _pending_writes errors when the document lock context
    # doesn't match (e.g. cascading from param.watch callbacks).
    _image_cache.update(field=None, n_frames=0, dataset=None, fields=[],
                        raw_shape=None)
    w_image_slider.value = 0
    w_image_slider.end = 1
    # Don't clear image_field options — preserve detector selection
    w_explore_plot.object = None
    # Grid (multi-view) tab — drop the figure so a new scan rebuilds fresh
    w_mv_grid.object = None
    w_mv_status.object = "*Click tab to load.*"
    _multiview_cache.update(
        uid=None, field=None, n_frames=0, total_frames=0, page=0,
        frames=None, renderers=None, mapper=None, log=None,
        data_lo=None, data_hi=None,
        loading=False,
    )
    w_mv_prev.disabled = True
    w_mv_next.disabled = True
    w_mv_page_status.object = ""
    w_proc_status.object = "*Select a scan and click Process.*"
    w_proc_spinner.value = False
    w_proc_spinner.visible = False
    # Clear the result cache BEFORE touching the frame slider so that
    # _update_proc_2d (triggered by the slider value change) finds no result
    # and exits early instead of rebuilding a stale 2D plot.
    _proc_result_cache.update(result=None, gi_result=None)
    w_proc_iq_plot.object = None
    w_proc_2d_plot.object = None
    w_proc_frame_slider.value = 0
    w_proc_frame_slider.end = 1
    w_proc_frame_slider.visible = False
    w_btn_add_collection.disabled = True
    _detail_cache.update(
        uid=None, run=None, summary=None,
        primary_loaded=False, baseline_loaded=False, images_loaded=False,
        primary_info=None, primary_dataset=None,
    )


def _ensure_run():
    uid = _state.get("selected_uid")
    if not uid:
        return None
    if _detail_cache["uid"] == uid and _detail_cache["run"] is not None:
        return _detail_cache["run"]
    run = _get_cat()[uid]
    _detail_cache.update(
        uid=uid, run=run, summary=None,
        primary_loaded=False, baseline_loaded=False, images_loaded=False,
        primary_info=None, primary_dataset=None,
    )
    return run


def _load_metadata(uid):
    t0 = time.perf_counter()
    run = _ensure_run()
    summary = enhanced_summary(run)
    _detail_cache["summary"] = summary

    # Show full raw metadata as collapsible JSON
    raw_md = dict(run.metadata)
    # Convert non-serialisable values to strings
    import json as _json

    def _sanitize(obj, _key=None):
        if isinstance(obj, dict):
            return {k: _sanitize(v, _key=k) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        # Convert epoch timestamps to human-readable
        if _key == "time" and isinstance(obj, (int, float)) and 1e9 < obj < 2e10:
            import datetime as _dt
            return _dt.datetime.fromtimestamp(obj).strftime("%Y-%m-%d %H:%M:%S")
        try:
            _json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    w_meta_json.object = _sanitize(raw_md)

    sid = summary.get("scan_id", "?")
    sample = summary.get("sample_name", "?")
    det = summary.get("detectors", "?")
    dt_ms = (time.perf_counter() - t0) * 1000
    w_detail_title.object = (
        f"### {sid} — {sample} [{det}] ({dt_ms:.0f} ms)"
    )


def _load_primary():
    if _detail_cache["primary_loaded"]:
        return
    run = _ensure_run()
    if run is None:
        return
    if "primary" not in tb.stream_names(run):
        w_primary_status.object = "*No primary stream.*"
        _detail_cache["primary_loaded"] = True
        return
    t0 = time.perf_counter()
    w_primary_status.object = "*Loading…*"
    w_primary_spinner.value = True
    w_primary_spinner.visible = True
    # Single read: get dataset, extract scalars from it
    info = tb.stream_info_for(run, "primary")
    ds = info.get("dataset")
    uid = _state.get("selected_uid")
    if uid:
        scalar_data = get_or_fetch_scalars(
            uid, "primary",
            lambda: tb.fetch_scalars(run, "primary", _dataset=ds),
        )
    else:
        scalar_data = tb.fetch_scalars(run, "primary", _dataset=ds)
    df = _scalars_to_dataframe(scalar_data)
    dt_ms = (time.perf_counter() - t0) * 1000
    w_primary_table.value = df
    w_primary_spinner.value = False
    w_primary_spinner.visible = False
    w_primary_status.object = (
        f"**{len(df)} rows, {len(df.columns)} fields** ({dt_ms:.0f} ms)"
    )
    # Populate plot selectors with numeric columns (preserve previous selection if possible)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    prev_x = w_primary_x.value
    prev_y = list(w_primary_y.value or [])
    w_primary_x.options = numeric_cols
    w_primary_y.options = numeric_cols
    # Keep previous selections if they exist in the new column set
    if prev_x in numeric_cols:
        w_primary_x.value = prev_x
    else:
        w_primary_x.value = numeric_cols[0] if numeric_cols else None
    valid_y = [y for y in prev_y if y in numeric_cols]
    w_primary_y.value = valid_y if valid_y else []
    # Also sync explore selectors
    prev_ex = w_explore_x.value
    prev_ey = list(w_explore_y.value or [])
    w_explore_x.options = numeric_cols
    w_explore_y.options = numeric_cols
    if prev_ex in numeric_cols:
        w_explore_x.value = prev_ex
    else:
        w_explore_x.value = numeric_cols[0] if numeric_cols else None
    valid_ey = [y for y in prev_ey if y in numeric_cols]
    w_explore_y.value = valid_ey if valid_ey else []
    # Cache primary info for image tab
    _detail_cache["primary_info"] = info
    _detail_cache["primary_dataset"] = ds
    _detail_cache["primary_loaded"] = True
    # Update export frame-label options
    _refresh_export_labels()
    _refresh_export_resolved_path()


def _load_baseline():
    if _detail_cache["baseline_loaded"]:
        return
    run = _ensure_run()
    if run is None:
        return
    t0 = time.perf_counter()
    w_baseline_status.object = "*Loading…*"

    rows = []

    try:
        # --- Baseline stream scalars ---
        has_baseline = "baseline" in tb.stream_names(run)
        if has_baseline:
            uid = _state.get("selected_uid")
            if uid:
                scalar_data = get_or_fetch_scalars(
                    uid, "baseline",
                    lambda: tb.fetch_scalars(run, "baseline"),
                )
            else:
                scalar_data = tb.fetch_scalars(run, "baseline")
            for key, arr in sorted(scalar_data.items()):
                arr = np.asarray(arr).flatten()
                if arr.size >= 2:
                    rows.append({"source": "baseline", "field": key,
                                 "before": str(arr[0]), "after": str(arr[-1])})
                elif arr.size == 1:
                    rows.append({"source": "baseline", "field": key,
                                 "before": str(arr[0]), "after": ""})
                else:
                    rows.append({"source": "baseline", "field": key,
                                 "before": str(arr.tolist()), "after": ""})

        # --- Primary stream configuration data ---
        try:
            config_data = tb.fetch_primary_config(run)
            for key, val in sorted(config_data.items()):
                rows.append({"source": "config", "field": key,
                             "before": str(val), "after": ""})
        except Exception as exc:
            log.warning("Config fetch failed: %s", exc)

    except Exception as exc:
        log.exception("Baseline/config load failed")
        w_baseline_status.object = f"**Error:** `{exc}`"
        _detail_cache["baseline_loaded"] = True
        return

    df = pd.DataFrame(rows, columns=["source", "field", "before", "after"])
    dt_ms = (time.perf_counter() - t0) * 1000
    w_baseline_table.value = df
    n_baseline = sum(1 for r in rows if r["source"] == "baseline")
    n_config = sum(1 for r in rows if r["source"] == "config")
    parts = []
    if n_baseline:
        parts.append(f"{n_baseline} baseline")
    if n_config:
        parts.append(f"{n_config} config")
    if parts:
        w_baseline_status.object = f"**{' + '.join(parts)} fields** ({dt_ms:.0f} ms)"
    else:
        w_baseline_status.object = "*No baseline or config data.*"
    _detail_cache["baseline_loaded"] = True


def _load_images():
    if _detail_cache["images_loaded"]:
        return
    run = _ensure_run()
    if run is None:
        return
    t0 = time.perf_counter()
    w_image_status.object = "*Loading thumbnail…*"
    w_image_spinner.value = True
    w_image_spinner.visible = True

    # Re-use primary info if already loaded; otherwise get it with one read
    info = _detail_cache.get("primary_info")
    if info is None and "primary" in tb.stream_names(run):
        info = tb.stream_info_for(run, "primary")
        _detail_cache["primary_info"] = info
        _detail_cache["primary_dataset"] = info.get("dataset")

    if info and info["images"]:
        image_fields = info["images"]
        ds = _detail_cache.get("primary_dataset")

        # Populate field selector — preserve previous detector if available.
        # Force a re-push of options by clearing first; otherwise Panel may
        # skip the update when the option set differs only by additions
        # (e.g. switching from a WAXS-only scan to a SAXS+WAXS one).
        prev_field = _image_cache.get("field")
        if list(w_image_field.options) != list(image_fields):
            w_image_field.options = []
            w_image_field.options = list(image_fields)
        if prev_field in image_fields:
            field = prev_field
        else:
            field = image_fields[0]
        # If the value is unchanged, _on_image_field won't fire — we still
        # need to refresh the displayed image for the new scan, so call it
        # explicitly below regardless.
        value_changed = w_image_field.value != field
        w_image_field.value = field

        shape = info["fields"].get(field, ())
        n_frames = shape[0] if len(shape) >= 3 else 1

        # Cache for slider use
        _image_cache["field"] = field
        _image_cache["n_frames"] = n_frames
        _image_cache["dataset"] = ds
        _image_cache["fields"] = image_fields

        # Configure slider — set value before end to avoid spurious
        # callbacks when the old value exceeds the new range.
        w_image_slider.value = 0
        w_image_slider.end = max(0, n_frames - 1)

        # Show first frame
        frame = _cached_fetch_frame(run, field, 0, ds)
        if frame is not None:
            _image_cache["raw_shape"] = tuple(frame.shape)
            frame = _orient_frame(frame, field)
            _update_image_in_place(frame, f"primary/{field} frame 0")
            dt_ms = (time.perf_counter() - t0) * 1000
            w_image_spinner.value = False
            w_image_spinner.visible = False
            w_image_status.object = (
                f"**primary/{field}** — {n_frames} frames ({dt_ms:.0f} ms)"
            )
            # Refresh dynamic mask if it was already enabled by the user
            if w_mask_dynamic.value:
                _render_dynamic_mask(0)
            # Build linked explore plot if primary data is available
            if _detail_cache.get("primary_loaded"):
                _build_explore_plot()
            _detail_cache["images_loaded"] = True
            return

    w_image_spinner.value = False
    w_image_spinner.visible = False
    w_image_status.object = "*No image fields found.*"
    _detail_cache["images_loaded"] = True


def _on_row_select(event):
    sel = w_table.selection
    if not sel:
        return
    df = w_table.value
    if sel[0] >= len(df):
        return
    uid = df.iloc[sel[0]]["uid"]
    if not uid or uid == "?":
        return
    # Preserve current tab and reload
    active_tab = w_detail_tabs.active
    _state["selected_uid"] = uid
    _reset_detail(preserve_figure=True)
    _state["selected_uid"] = uid  # re-set after _reset_detail clears it
    try:
        _load_metadata(uid)
        # Set tab (may not fire watch if same value), then force-load content
        w_detail_tabs.active = active_tab
        _load_active_tab(active_tab)
    except Exception as exc:
        w_detail_title.object = f"**Error:** `{exc}`"


def _load_active_tab(active):
    """Load data for the given tab index."""
    if active == 1:
        _load_primary()
    elif active == 2:
        _load_baseline()
    elif active == 3:
        _load_primary()
        _load_images()
    elif active == 4:
        _load_multiview()
    elif active == 5:
        pass  # Process tab — no auto-load
    elif active == 6:
        _load_primary()
        _refresh_export_labels()
        _refresh_export_resolved_path()


def _on_detail_tab(event):
    if not _state.get("selected_uid"):
        return
    try:
        _load_active_tab(event.new)
    except Exception as exc:
        log.exception("Detail tab load error")
        # Surface the error on the relevant status widget so the user sees it
        if event.new == 4:
            w_mv_status.object = f"**Error loading grid:** `{exc}`"
            w_mv_spinner.value = False
            w_mv_spinner.visible = False


def _update_primary_plot(*_events, use_table_order: bool = False):
    """Redraw the primary scatter plot as interactive Bokeh."""
    from bokeh.plotting import figure as bk_figure

    df = w_primary_table.value
    if use_table_order:
        try:
            df = w_primary_table.current_view
        except Exception:
            pass
    x_col = w_primary_x.value
    y_cols = w_primary_y.value
    if df is None or df.empty or not x_col or not y_cols:
        w_primary_plot.object = None
        w_primary_fit_result.object = ""
        return
    try:
        colors = ["black", "blue", "red", "green", "orange", "purple"]
        p = bk_figure(
            title=f"{x_col} vs {', '.join(y_cols)}", height=280,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            sizing_mode="stretch_width",
        )
        x = df[x_col].values
        for i, y_col in enumerate(y_cols):
            y = df[y_col].values
            c = colors[i % len(colors)]
            p.line(x, y, line_color=c, line_width=1.2, legend_label=y_col)
            p.scatter(x, y, color=c, size=4, legend_label=y_col)
        p.xaxis.axis_label = x_col
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "8pt"
        w_primary_plot.object = p
    except Exception as exc:
        log.warning("Primary plot failed: %s", exc)
        w_primary_plot.object = None


def _on_primary_sort_btn(_event=None):
    """Re-plot using the current table sort/filter order."""
    _update_primary_plot(use_table_order=True)


w_primary_sort_btn.on_click(_on_primary_sort_btn)


def _fit_gaussian(x, y):
    """Fit a Gaussian to (x, y) data. Returns dict of fit stats or None."""
    from scipy.optimize import curve_fit

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None

    # Initial guesses
    peak_idx = int(np.argmax(y))
    amp_guess = float(y[peak_idx])
    mu_guess = float(x[peak_idx])
    # Estimate sigma from half-max width
    half_max = amp_guess / 2
    above = np.where(y >= half_max)[0]
    if len(above) >= 2:
        sigma_guess = (x[above[-1]] - x[above[0]]) / 2.355
    else:
        sigma_guess = (x[-1] - x[0]) / 6
    sigma_guess = max(sigma_guess, 1e-12)
    offset_guess = float(np.percentile(y, 5))

    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    try:
        popt, _ = curve_fit(
            gaussian, x, y,
            p0=[amp_guess - offset_guess, mu_guess, sigma_guess, offset_guess],
            maxfev=5000,
        )
        amp, mu, sigma, offset = popt
        fwhm = abs(sigma) * 2.3548
        # Center of mass
        y_shifted = y - y.min()
        com = float(np.sum(x * y_shifted) / np.sum(y_shifted)) if np.sum(y_shifted) > 0 else mu
        # Highest point
        highest_x = float(x[peak_idx])
        highest_y = float(y[peak_idx])
        fit_y = gaussian(x, *popt)
        return {
            "type": "Gaussian",
            "center": mu,
            "fwhm": fwhm,
            "amplitude": amp + offset,
            "com": com,
            "highest_x": highest_x,
            "highest_y": highest_y,
            "fit_x": x,
            "fit_y": fit_y,
        }
    except Exception:
        return None


def _fit_knife_edge(x, y):
    """Fit an error-function (knife edge) to (x, y). Returns dict or None."""
    from scipy.optimize import curve_fit
    from scipy.special import erf

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None

    # Sort by x for consistent fitting
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Initial guesses
    y_min, y_max = float(y.min()), float(y.max())
    amp_guess = (y_max - y_min) / 2
    offset_guess = (y_max + y_min) / 2
    # Edge center: where y crosses midpoint
    mid = (y_min + y_max) / 2
    cross_idx = int(np.argmin(np.abs(y - mid)))
    mu_guess = float(x[cross_idx])
    sigma_guess = (x[-1] - x[0]) / 10
    sigma_guess = max(abs(sigma_guess), 1e-12)

    def knife_edge(x, amp, mu, sigma, offset):
        return amp * erf((x - mu) / (sigma * np.sqrt(2))) + offset

    try:
        popt, _ = curve_fit(
            knife_edge, x, y,
            p0=[amp_guess, mu_guess, sigma_guess, offset_guess],
            maxfev=5000,
        )
        amp, mu, sigma, offset = popt
        fwhm = abs(sigma) * 2.3548  # width of the derivative Gaussian
        # Center of mass of derivative (= edge location)
        dy = np.gradient(y, x)
        dy_abs = np.abs(dy)
        com = float(np.sum(x * dy_abs) / np.sum(dy_abs)) if np.sum(dy_abs) > 0 else mu
        # Highest point of derivative
        peak_idx = int(np.argmax(dy_abs))
        highest_x = float(x[peak_idx])
        highest_y = float(dy[peak_idx])
        fit_y = knife_edge(x, *popt)
        return {
            "type": "Knife edge",
            "center": mu,
            "fwhm": fwhm,
            "amplitude": amp,
            "com": com,
            "highest_x": highest_x,
            "highest_y": highest_y,
            "fit_x": x,
            "fit_y": fit_y,
        }
    except Exception:
        return None


def _on_primary_fit(_event=None):
    """Run the selected fit on all plotted Y columns and overlay results."""
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import Legend, LegendItem

    fit_type = w_primary_fit.value
    if fit_type == "None":
        w_primary_fit_result.object = ""
        _update_primary_plot()
        return

    df = w_primary_table.value
    try:
        df = w_primary_table.current_view
    except Exception:
        pass
    x_col = w_primary_x.value
    y_cols = w_primary_y.value
    if df is None or df.empty or not x_col or not y_cols:
        return

    colors = ["black", "blue", "red", "green", "orange", "purple"]
    x = df[x_col].values.astype(float)

    p = bk_figure(
        title=f"{x_col} vs {', '.join(y_cols)} — {fit_type} fit", height=280,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        sizing_mode="stretch_width",
    )

    results_md = []
    for i, y_col in enumerate(y_cols):
        y = df[y_col].values.astype(float)
        c = colors[i % len(colors)]
        p.scatter(x, y, color=c, size=4, legend_label=y_col)
        p.line(x, y, line_color=c, line_width=1, line_alpha=0.4)

        # Perform fit
        if fit_type == "Gaussian":
            result = _fit_gaussian(x.copy(), y.copy())
        else:
            result = _fit_knife_edge(x.copy(), y.copy())

        if result is not None:
            from bokeh.models import BoxAnnotation, Span

            # Overlay fit curve
            p.line(
                result["fit_x"], result["fit_y"],
                line_color=c, line_width=2.5, line_dash="dashed",
                legend_label=f"{y_col} fit",
            )
            # Vertical line at fit center
            center_span = Span(
                location=result["center"], dimension="height",
                line_color=c, line_width=1.5, line_dash="solid",
                line_alpha=0.8,
            )
            p.add_layout(center_span)
            # Vertical line at peak position (dotted, slightly thinner)
            if abs(result["highest_x"] - result["center"]) > result["fwhm"] * 0.05:
                peak_span = Span(
                    location=result["highest_x"], dimension="height",
                    line_color=c, line_width=1.2, line_dash="dotted",
                    line_alpha=0.7,
                )
                p.add_layout(peak_span)
            # FWHM shaded band centered on fit center
            fwhm_lo = result["center"] - result["fwhm"] / 2
            fwhm_hi = result["center"] + result["fwhm"] / 2
            fwhm_box = BoxAnnotation(
                left=fwhm_lo, right=fwhm_hi,
                fill_color=c, fill_alpha=0.07,
                line_color=c, line_alpha=0.3, line_width=1, line_dash="dashed",
            )
            p.add_layout(fwhm_box)
            # Build stats string
            stats = (
                f"**{y_col}** ({result['type']}): "
                f"center={result['center']:.5g}, "
                f"FWHM={result['fwhm']:.5g}, "
                f"COM={result['com']:.5g}, "
                f"peak=({result['highest_x']:.5g}, {result['highest_y']:.5g})"
            )
            results_md.append(stats)
        else:
            results_md.append(f"**{y_col}**: fit failed")

    p.xaxis.axis_label = x_col
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    w_primary_plot.object = p
    w_primary_fit_result.object = "  \n".join(results_md)


w_primary_fit_btn.on_click(_on_primary_fit)


# --- Linked explore plot (1D primary + cursor synced with image slider) ---

_EXPLORE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_explore_cursor_source = None  # ColumnDataSource for vertical cursor line


def _build_explore_plot():
    """Build / rebuild the Bokeh 1D line plot with a vertical cursor."""
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import ColumnDataSource, Span, Label

    global _explore_cursor_source

    df = w_primary_table.value
    x_col = w_explore_x.value
    y_cols = w_explore_y.value
    if df is None or df.empty or not x_col or not y_cols:
        w_explore_plot.object = None
        return

    x = df[x_col].values
    p = bk_figure(
        sizing_mode="stretch_both",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    y_data = {}
    for i, y_col in enumerate(y_cols):
        y = df[y_col].values
        y_data[y_col] = y
        c = _EXPLORE_COLORS[i % len(_EXPLORE_COLORS)]
        p.line(x, y, line_color=c, line_width=1.2, legend_label=y_col)
        p.scatter(x, y, color=c, size=4, legend_label=y_col)
    p.xaxis.axis_label = x_col
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"

    # Add vertical cursor at frame position
    idx = w_image_slider.value
    x_val = float(x[idx]) if idx < len(x) else float(x[0]) if len(x) else 0
    cursor = Span(location=x_val, dimension="height",
                  line_color="red", line_width=2, line_alpha=0.7, line_dash="dashed")
    p.add_layout(cursor)

    # Label showing cursor values
    label_parts = [f"frame {idx}"]
    for yc in y_cols:
        if idx < len(y_data[yc]):
            label_parts.append(f"{yc}={y_data[yc][idx]:.4g}")
    cursor_label = Label(
        x=5, x_units="screen", y=5, y_units="screen",
        text="  ".join(label_parts),
        text_font_size="9pt", text_color="red",
        background_fill_color="white", background_fill_alpha=0.8,
    )
    p.add_layout(cursor_label)

    _explore_cursor_source = {
        "span": cursor, "label": cursor_label,
        "x_values": x, "y_data": y_data, "y_cols": y_cols,
    }

    w_explore_plot.object = p


def _update_explore_cursor(idx):
    """Move the vertical cursor and update the value label."""
    if _explore_cursor_source is None:
        return
    span = _explore_cursor_source.get("span")
    label = _explore_cursor_source.get("label")
    x_arr = _explore_cursor_source.get("x_values")
    y_data = _explore_cursor_source.get("y_data", {})
    y_cols = _explore_cursor_source.get("y_cols", [])
    if span is not None and x_arr is not None and idx < len(x_arr):
        x_val = float(x_arr[idx])
        span.location = x_val
        if label is not None:
            parts = [f"frame {idx}"]
            for yc in y_cols:
                yarr = y_data.get(yc)
                if yarr is not None and idx < len(yarr):
                    parts.append(f"{yc}={yarr[idx]:.4g}")
            label.text = "  ".join(parts)


def _on_explore_xy(*_events):
    _build_explore_plot()


w_explore_x.param.watch(_on_explore_xy, "value")
w_explore_y.param.watch(_on_explore_xy, "value")


w_primary_x.param.watch(_update_primary_plot, "value")
w_primary_y.param.watch(_update_primary_plot, "value")
w_primary_table.param.watch(
    lambda *_e: _update_primary_plot(use_table_order=True), "sorters",
)


# Wire search events
w_btn_search.on_click(lambda e: _do_search(0))
w_btn_reset.on_click(_on_reset)
w_btn_first.on_click(lambda e: _go_to_page(0))
w_btn_prev.on_click(lambda e: _go_to_page(max(0, _state["page"] - 1)))
w_btn_next.on_click(lambda e: _go_to_page(min(_n_pages() - 1, _state["page"] + 1)))
w_btn_last.on_click(lambda e: _go_to_page(_n_pages() - 1))

w_table.param.watch(_on_row_select, "selection")


# ---------------------------------------------------------------------------
# Callbacks — Processing
# ---------------------------------------------------------------------------

def _on_process(event):
    uid = _state.get("selected_uid")
    if not uid:
        pn.state.notifications.warning("No scan selected.")
        return

    geometry = w_proc_geometry.value
    w_proc_status.object = f"*Processing `{uid[:12]}…` ({geometry})*"
    w_proc_spinner.value = True
    w_proc_spinner.visible = True
    w_btn_process.disabled = True
    _proc_result_cache.update(result=None, gi_result=None)
    _per_frame_qchi_cache.update(uid=None, data=None)
    _processing_guard["active"] = True

    try:
        t0 = time.perf_counter()

        # Use _build_proc_params() for both GI and transmission so the
        # printed command and actual kwargs always stay in sync.
        reduce_fn, call_params, _geom_label = _build_proc_params(uid)

        if geometry == "grazing":
            gi_params = call_params

            _args_str = ", ".join(f"{k}={v!r}" for k, v in gi_params.items())
            print(f"\n>>> reduce_smi_gi({_args_str})\n")

            gi_result = reduce_fn(**gi_params)
            dt = time.perf_counter() - t0

            _cache_reduction_result(uid, gi_result, geometry, gi_params)

            _proc_result_cache["gi_result"] = gi_result
            _last_result["result"] = gi_result
            _last_result["params"] = gi_params

            # Configure frame slider for GI result
            n_fr = len(gi_result.frames)
            w_proc_frame_slider.end = max(0, n_fr - 1)
            w_proc_frame_slider.value = 0
            w_proc_iq_mode.visible = n_fr > 1

            # Display 2D map respecting mode toggle
            if w_proc_iq_mode.value == "per-frame" and n_fr > 1:
                w_proc_frame_slider.visible = True
                w_proc_2d_plot.object = _plot_2d_gi(gi_result, frame_idx=0)
            else:
                w_proc_frame_slider.visible = False
                w_proc_2d_plot.object = _plot_2d_gi(gi_result)

            # No merged I(q) for GI — clear the I(q) plot
            w_proc_iq_plot.object = None

            timing = gi_result.timing or {}
            timing_str = ", ".join(f"{k}: {v:.1f}s" for k, v in timing.items())
            w_proc_status.object = (
                f"**Done** in {dt:.1f}s — GI-WAXS, "
                f"{n_fr} frames, α_i: {gi_result.alpha_i_source}\n\n"
                f"_{timing_str}_"
            )
            w_btn_add_collection.disabled = True  # GI not in collection yet

        else:
            # Transmission mode
            params = call_params

            _args_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            print(f"\n>>> reduce_smi_combined({_args_str})\n")

            result = reduce_fn(**params)
            dt = time.perf_counter() - t0

            _cache_reduction_result(uid, result, geometry, params)

            _proc_result_cache["result"] = result
            _last_result["result"] = result
            _last_result["params"] = params

            # Detect whether per-frame data is available from any source
            pf_iq = getattr(result, "per_frame_iq", None)
            has_perframe = (
                (pf_iq is not None and "frame" in pf_iq.dims and pf_iq.sizes["frame"] > 1)
            )

            # 2D q-chi map
            try:
                qchi = result.merged_qchi
                has_qchi_frames = "frame" in qchi.dims
                if has_qchi_frames:
                    n_fr = qchi.sizes["frame"]
                else:
                    # Check for per-frame data in saxs/waxs q_chi_frames
                    pf_qchi = _get_per_frame_qchi(result)
                    if pf_qchi is not None:
                        n_fr = pf_qchi[2].shape[0]
                        has_qchi_frames = True  # we have per-frame 2D data
                    else:
                        n_fr = 0
                if has_qchi_frames and n_fr > 1:
                    w_proc_frame_slider.end = max(0, n_fr - 1)
                    w_proc_frame_slider.value = 0
                # Show per-frame controls if frames exist in either source
                w_proc_iq_mode.visible = has_qchi_frames or has_perframe
                if not w_proc_iq_mode.visible:
                    w_proc_iq_mode.value = "merged"
                # Display 2D map respecting the current mode toggle
                if w_proc_iq_mode.value == "per-frame" and has_qchi_frames and n_fr > 1:
                    w_proc_frame_slider.visible = True
                    w_proc_2d_plot.object = _plot_2d_transmission(
                        result, frame_idx=w_proc_frame_slider.value)
                else:
                    w_proc_frame_slider.visible = False
                    w_proc_2d_plot.object = _plot_2d_transmission(result)
            except Exception as exc:
                log.exception("2D q-chi plot failed")
                w_proc_2d_plot.object = None
                w_proc_frame_slider.visible = False
                # Still show per-frame I(q) controls if per_frame_iq exists
                w_proc_iq_mode.visible = has_perframe
                if not has_perframe:
                    w_proc_iq_mode.value = "merged"

            # Populate frame label selector from per_frame_iq bundled
            # primary scalars (preferred) plus primary table (fallback)
            label_options = ["(frame #)"]
            if pf_iq is not None:
                # 1D vars in per_frame_iq are primary scalars (dims == (frame,))
                iq_vars = {"I", "saxs_I", "waxs_I"}  # skip I(q) variables
                label_options += [
                    v for v in pf_iq.data_vars
                    if v not in iq_vars and pf_iq[v].ndim == 1
                ]
            # Supplement with primary table columns not already listed
            df = w_primary_table.value
            if df is not None and not df.empty:
                existing = set(label_options)
                label_options += [
                    c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c not in existing
                ]
            if list(w_proc_iq_label.options) != label_options:
                w_proc_iq_label.options = label_options
            w_proc_iq_label.visible = w_proc_iq_mode.visible

            # I(q) plot — merged or per-frame depending on current toggle
            _build_proc_iq_plot()
            _update_frame_slider_label()

            timing = result.timing or {}
            timing_str = ", ".join(f"{k}: {v:.1f}s" for k, v in timing.items())
            w_proc_status.object = (
                f"**Done** in {dt:.1f}s — {result.geometry}\n\n"
                f"_{timing_str}_"
            )
            w_btn_add_collection.disabled = False

    except Exception as exc:
        log.exception("Processing failed")
        w_proc_status.object = f"**Error:** `{exc}`"
        w_proc_iq_plot.object = None
        w_proc_2d_plot.object = None
    else:
        # Auto-export if enabled
        if w_export_auto.value and uid:
            try:
                out = _do_export_single(uid)
                if out:
                    scan_dir, files = out
                    log.info("Auto-exported %d files to %s", len(files), scan_dir)
            except Exception:
                log.exception("Auto-export failed for %s", uid[:8])
    finally:
        _processing_guard["active"] = False
        w_btn_process.disabled = False
        w_proc_spinner.value = False
        w_proc_spinner.visible = False


def _on_add_to_collection(event):
    result = _last_result.get("result")
    if result is None:
        return
    summary = _detail_cache.get("summary") or {}
    params = _last_result.get("params") or {}
    # Bundle primary/baseline/config/raw metadata with the processed scan
    primary_df = w_primary_table.value if _detail_cache.get("primary_loaded") else None
    # Fetch baseline as raw scalar DataFrame (the UI table is transposed
    # into field/before/after rows, which isn't suitable for numeric lookups).
    baseline_df = None
    config_df = None
    run = _ensure_run()
    if _detail_cache.get("baseline_loaded"):
        if run and "baseline" in tb.stream_names(run):
            try:
                baseline_df = _scalar_stream_to_frame(run, "baseline")
            except Exception:
                pass
    if run:
        try:
            config_df = _config_to_dataframe(run)
        except Exception:
            pass
    raw_metadata = w_meta_json.object if w_meta_json.object else None
    _collection.add(
        result, summary, params,
        primary_df=primary_df,
        baseline_df=baseline_df,
        config_df=config_df,
        raw_metadata=raw_metadata,
    )
    _refresh_collection()
    _open_collection_panel()  # pop open the floating panel
    pn.state.notifications.success(
        f"Added {result.uid[:8]} to collection ({len(_collection)} scans)"
    )


w_btn_process.on_click(_on_process)
w_btn_add_collection.on_click(_on_add_to_collection)


# ---------------------------------------------------------------------------
# Callbacks — Collection (delegated to smi_browser.ui.collection)
# ---------------------------------------------------------------------------

w_coll_summary = _coll_ns.summary_md


def _refresh_collection():
    _coll_ns.refresh()


# ---------------------------------------------------------------------------
# Batch processing — queue many scans from the current search results
# ---------------------------------------------------------------------------
#
# A BatchProcessor (see batch_processor.py) runs PyHyperScattering reductions
# on a background worker thread, leaving the UI interactive.  Status updates
# are routed back onto the Bokeh document thread via add_next_tick_callback,
# which is the only safe way to mutate widgets from a non-UI thread.
#
# The reduction parameters are read from the Process tab widgets at job time
# (so the user configures them once in Parameters, then queues a batch).
# Already-collected uids can be skipped.  Hard caps prevent runaway queues
# when search results contain hundreds of scans.

w_batch_status = pn.pane.Markdown(
    "*Idle — queue scans from the current search results to process them in "
    "the background.*",
    margin=(0, 5),
)
w_batch_progress = pn.indicators.Progress(
    name="Batch progress", value=0, max=1, width=400, visible=False,
)
w_batch_table = pn.widgets.Tabulator(
    pd.DataFrame(columns=["uid_short", "label", "state", "duration_s", "error"]),
    height=320, layout="fit_data_stretch", show_index=False, disabled=True,
    sizing_mode="stretch_width",
)

_BATCH_ROW_COLORS = {
    "running": "background-color: #d4edda",   # green
    "error":   "background-color: #f8d7da",   # red
    "done":    "background-color: #f0f0f0; color: #888",   # light grey
    "skipped": "background-color: #f0f0f0; color: #888",
    "cancelled": "background-color: #fff3cd; color: #888", # pale yellow
    "queued":  "",
}
w_batch_max_workers = pn.widgets.IntInput(
    name="Workers", value=1, start=1, end=16, width=90,
)
w_batch_low_memory_mode = pn.widgets.Checkbox(
    name="Low-memory mode (force 1 worker)", value=True,
)
w_batch_skip_existing = pn.widgets.Checkbox(
    name="Skip uids already in collection", value=True,
)
w_batch_skip_processed = pn.widgets.Checkbox(
    name="Skip uids in processed history", value=True,
)
w_batch_add_to_collection = pn.widgets.Checkbox(
    name="Add processed scans to collection", value=True,
)
w_batch_max_jobs = pn.widgets.IntInput(
    name="Max jobs", value=PAGE_SIZE, start=1, end=BatchProcessor.MAX_QUEUE,
    width=110,
)
w_batch_log_rows = pn.widgets.IntInput(
    name="Log rows", value=200, start=25, end=2000, width=110,
)
w_btn_batch_queue = pn.widgets.Button(
    name="Queue scans", button_type="primary",
)
w_btn_batch_cancel = pn.widgets.Button(
    name="Cancel", button_type="warning", disabled=True,
)
w_btn_batch_clear = pn.widgets.Button(
    name="Clear log", button_type="light", disabled=True,
)
w_btn_batch_clear_processed = pn.widgets.Button(
    name="Clear processed list", button_type="light",
)
w_batch_processed_status = pn.pane.Markdown(
    "*Processed history: 0 uids*",
    margin=(0, 5),
)
w_batch_memory_status = pn.pane.Markdown(
    "*Batch memory: RSS 0.0 MB, peak 0.0 MB*",
    margin=(0, 5),
)

_batch_state: dict[str, Any] = {
    "doc": None,
    "processor": None,
    "processed_uids": set(),
    "processed_lock": threading.Lock(),
    "rss_peak_mb": 0.0,
}


def _batch_rss_mb() -> float | None:
    """Return process RSS in MB, or None if unavailable."""
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss_kb) / 1024.0
    except Exception:
        pass
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        return float(rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _batch_memory_status_text() -> str:
    """Build memory telemetry line for the Batch UI."""
    rss = _batch_rss_mb()
    if rss is None:
        return "*Batch memory: unavailable*"
    peak = max(float(_batch_state.get("rss_peak_mb", 0.0)), rss)
    _batch_state["rss_peak_mb"] = peak
    return f"*Batch memory: RSS {rss:.1f} MB, peak {peak:.1f} MB*"


def _batch_effective_workers() -> int:
    requested = max(1, int(w_batch_max_workers.value or 1))
    if w_batch_low_memory_mode.value:
        return 1
    return requested


def _batch_processed_contains(uid: str) -> bool:
    with _batch_state["processed_lock"]:
        return uid in _batch_state["processed_uids"]


def _batch_processed_add(uid: str) -> None:
    with _batch_state["processed_lock"]:
        _batch_state["processed_uids"].add(uid)


def _batch_processed_count() -> int:
    with _batch_state["processed_lock"]:
        return len(_batch_state["processed_uids"])


def _batch_processed_clear() -> int:
    with _batch_state["processed_lock"]:
        n = len(_batch_state["processed_uids"])
        _batch_state["processed_uids"].clear()
    return n


def _batch_cleanup_memory() -> None:
    """Release temporary references created during one batch job."""
    try:
        plt.close("all")
    except Exception:
        pass
    gc.collect()


def _batch_trim_allocator_now() -> None:
    """Best-effort request to return free heap pages to the OS."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _batch_post_job_cleanup() -> None:
    """Aggressive cleanup after each batch job to limit RSS growth."""
    if w_batch_low_memory_mode.value:
        try:
            clear_geometry_cache()
        except Exception:
            pass
    _batch_cleanup_memory()
    prune_lock_table()
    _batch_trim_allocator_now()


def _try_parse_tuple(d: dict, key: str, text: str, default: tuple) -> None:
    """Parse comma-separated string to float tuple; add to *d* if ≠ *default*."""
    try:
        vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
        if vals and vals != default:
            d[key] = vals
    except (ValueError, TypeError):
        pass


def _try_parse_panel_cols(d: dict, text: str) -> None:
    """Parse panel col ranges '(0,206),(206,413),(413,619)' to tuple of tuples."""
    import re
    default = ((0, 206), (206, 413), (413, 619))
    try:
        pairs = re.findall(r'\((\d+)\s*,\s*(\d+)\)', text)
        if pairs:
            result = tuple((int(a), int(b)) for a, b in pairs)
            if result != default:
                d["panel_col_ranges"] = result
    except (ValueError, TypeError):
        pass


def _build_waxs_overrides() -> dict[str, Any]:
    """Collect WAXS calibration + masking overrides from widgets."""
    waxs_kw: dict[str, Any] = {}
    if w_proc_waxs_energy.value != DEFAULT_WAXS_ENERGY_KEV:
        waxs_kw["energy_kev"] = w_proc_waxs_energy.value
    if w_proc_waxs_dist.value != DEFAULT_WAXS_SAMPLE_DIST_MM:
        waxs_kw["sample_distance_mm"] = w_proc_waxs_dist.value
    if w_proc_waxs_pixel.value != DEFAULT_WAXS_PIXEL_SIZE_MM:
        waxs_kw["pixel_size_mm"] = w_proc_waxs_pixel.value
    if w_proc_waxs_beam_row.value != DEFAULT_WAXS_BEAM_CENTER_ROW:
        waxs_kw["beam_center_row"] = w_proc_waxs_beam_row.value
    if w_proc_waxs_beam_col.value != DEFAULT_WAXS_BEAM_CENTER_COL:
        waxs_kw["beam_center_col"] = w_proc_waxs_beam_col.value
    _try_parse_panel_cols(waxs_kw, w_proc_waxs_panel_cols.value)
    _try_parse_tuple(
        waxs_kw, "panel_offsets_deg",
        w_proc_waxs_panel_offsets.value, (-7.0, 0.0, 7.0),
    )
    _try_parse_tuple(
        waxs_kw, "panel_row_shifts",
        w_proc_waxs_panel_row_shifts.value, (0.0, 0.0, 0.0),
    )
    _try_parse_tuple(
        waxs_kw, "panel_col_shifts",
        w_proc_waxs_panel_col_shifts.value, (0.0, 0.0, 0.0),
    )
    _try_parse_tuple(
        waxs_kw, "panel_delta_deg",
        w_proc_waxs_panel_delta.value, (0.0, 0.0, 0.0),
    )
    if w_proc_waxs_theta_zero.value != DEFAULT_WAXS_THETA_ZERO_DEG:
        waxs_kw["theta_zero_deg"] = w_proc_waxs_theta_zero.value
    if w_proc_waxs_offset_x.value != DEFAULT_WAXS_SAMPLE_OFFSET_X_MM:
        waxs_kw["sample_offset_x_mm"] = w_proc_waxs_offset_x.value
    if w_proc_waxs_offset_z.value != DEFAULT_WAXS_SAMPLE_OFFSET_Z_MM:
        waxs_kw["sample_offset_z_mm"] = w_proc_waxs_offset_z.value
    if w_proc_waxs_col_arc_cal.value != 0.0:
        waxs_kw["beam_col_per_arc_deg"] = w_proc_waxs_col_arc_cal.value
    if w_proc_waxs_qh_sign.value != DEFAULT_WAXS_Q_HORIZONTAL_SIGN:
        waxs_kw["q_horizontal_sign"] = w_proc_waxs_qh_sign.value
    if w_proc_waxs_qv_sign.value != DEFAULT_WAXS_Q_VERTICAL_SIGN:
        waxs_kw["q_vertical_sign"] = w_proc_waxs_qv_sign.value
    if w_proc_waxs_rot_k.value != DEFAULT_WAXS_ROTATION_K:
        waxs_kw["rotation_k"] = w_proc_waxs_rot_k.value
    # WAXS masking overrides
    if w_proc_waxs_bsx_ref.value != 0.0:
        waxs_kw["waxs_bsx_ref"] = w_proc_waxs_bsx_ref.value
    if w_proc_beamstop_max_arc.value != DEFAULT_BEAMSTOP_MAX_ABS_ARC_DEG:
        waxs_kw["beamstop_max_abs_arc_deg"] = w_proc_beamstop_max_arc.value
    return waxs_kw


def _build_proc_params(uid: str) -> tuple:
    """Snapshot current Process-tab widget values into reduction params.

    Mirrors the param-construction in :func:`_on_process` so a batch job
    runs with whatever the user has configured in the Parameters sub-tab.
    Returns ``(callable, params_dict, geometry_label)``.
    """
    geometry = w_proc_geometry.value
    waxs_kw = _build_waxs_overrides()
    saxs_mask_path = _normalize_mask_path(w_proc_saxs_mask.value)
    waxs_mask_path = _normalize_mask_path(w_proc_waxs_mask.value)

    if geometry == "grazing":
        from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_gi

        gi_params: dict[str, Any] = dict(
            uid=uid,
            tiled_uri=DEFAULT_TILED_URI,
            catalog=DEFAULT_CATALOG,
            waxs_mask_path=waxs_mask_path,
        )
        if w_proc_nqxy.value != DEFAULT_N_QXY:
            gi_params["n_qxy"] = w_proc_nqxy.value
        if w_proc_nqz.value != DEFAULT_N_QZ:
            gi_params["n_qz"] = w_proc_nqz.value
        if w_proc_theta_offset.value != DEFAULT_THETA_OFFSET:
            gi_params["theta_offset"] = w_proc_theta_offset.value
        if w_proc_dezinger.value != DEFAULT_DEZINGER:
            gi_params["dezinger_threshold"] = (
                w_proc_dezinger.value if w_proc_dezinger.value > 0 else None
            )
        if w_proc_dezinger_kernel.value != DEFAULT_DEZINGER_KERNEL:
            gi_params["dezinger_kernel"] = w_proc_dezinger_kernel.value
        if w_proc_pixel_splitting.value != DEFAULT_PIXEL_SPLITTING:
            gi_params["pixel_splitting"] = w_proc_pixel_splitting.value
        if not w_proc_incident_angle_auto.value:
            gi_params["incident_angle_deg"] = w_proc_incident_angle.value
        if w_proc_beamstop_max_arc.value != DEFAULT_BEAMSTOP_MAX_ABS_ARC_DEG:
            gi_params["beamstop_max_abs_arc_deg"] = w_proc_beamstop_max_arc.value
        if w_proc_waxs_col_per_arc.value != DEFAULT_WAXS_BEAM_COL_PER_ARC_DEG:
            gi_params["waxs_beam_col_per_arc_deg"] = w_proc_waxs_col_per_arc.value
        if waxs_kw:
            gi_params["waxs_cal_overrides"] = waxs_kw
        # Supply pre-cached images if available
        _cp = cache_path(uid)
        if _cp.exists():
            gi_params["image_cache_path"] = str(_cp)
        return reduce_smi_gi, gi_params, geometry

    from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined

    params: dict[str, Any] = dict(
        uid=uid,
        tiled_uri=DEFAULT_TILED_URI,
        catalog=DEFAULT_CATALOG,
        solid_angle_correction=w_proc_solid_angle.value,
        geometry=geometry,
        saxs_mask_path=saxs_mask_path,
        waxs_mask_path=waxs_mask_path,
        cache_geometry=(w_cache_enabled.value and not w_batch_low_memory_mode.value),
    )
    # Supply pre-cached images if available
    _cp = cache_path(uid)
    if _cp.exists():
        params["image_cache_path"] = str(_cp)
    if w_proc_nq.value != DEFAULT_N_Q:
        params["n_q"] = w_proc_nq.value
    else:
        params["n_q"] = DEFAULT_N_Q
    if w_proc_nchi.value != DEFAULT_N_CHI:
        params["n_chi"] = w_proc_nchi.value

    # SAXS beam-centre deltas
    saxs_row_changed = w_proc_saxs_row_delta.value != DEFAULT_SAXS_ROW_DELTA
    saxs_col_changed = w_proc_saxs_col_delta.value != DEFAULT_SAXS_COL_DELTA
    if saxs_row_changed or saxs_col_changed:
        params["saxs_beam_delta_px"] = (
            w_proc_saxs_row_delta.value, w_proc_saxs_col_delta.value,
        )
    # WAXS beam-centre deltas
    waxs_row_changed = w_proc_waxs_row_delta.value != DEFAULT_WAXS_ROW_DELTA
    waxs_col_changed = w_proc_waxs_col_delta.value != DEFAULT_WAXS_COL_DELTA
    if waxs_row_changed or waxs_col_changed:
        params["waxs_beam_delta_px"] = (
            w_proc_waxs_row_delta.value, w_proc_waxs_col_delta.value,
        )
    if w_proc_dist_delta.value != DEFAULT_SAXS_DIST_DELTA:
        params["saxs_distance_delta_mm"] = w_proc_dist_delta.value
    if w_proc_waxs_col_per_arc.value != DEFAULT_WAXS_BEAM_COL_PER_ARC_DEG:
        params["waxs_beam_col_per_arc_deg"] = w_proc_waxs_col_per_arc.value

    # Dezinger
    if w_proc_dezinger.value != DEFAULT_DEZINGER:
        params["dezinger_threshold"] = (
            w_proc_dezinger.value if w_proc_dezinger.value > 0 else None
        )
    if w_proc_dezinger_kernel.value != DEFAULT_DEZINGER_KERNEL:
        params["dezinger_kernel"] = w_proc_dezinger_kernel.value
    if w_proc_pixel_splitting.value != DEFAULT_PIXEL_SPLITTING:
        params["pixel_splitting"] = w_proc_pixel_splitting.value

    # SAXS Q-range / aperture
    if w_proc_saxs_q_cutoff.value > 0:
        params["saxs_q_cutoff"] = w_proc_saxs_q_cutoff.value
    if w_proc_saxs_agbh_ring.value != DEFAULT_SAXS_AGBH_RING_ORDER:
        params["saxs_agbh_ring_order"] = w_proc_saxs_agbh_ring.value
    if w_proc_saxs_q_margin.value != DEFAULT_SAXS_Q_MARGIN_FRACTION:
        params["saxs_q_margin_fraction"] = w_proc_saxs_q_margin.value

    # Backend options
    backend_opts: dict[str, Any] = {}
    if w_proc_saxs_rotate.value:
        backend_opts["saxs_rotate_cw_90"] = True
    if w_proc_waxs_flip.value:
        backend_opts["waxs_flip_horizontal"] = True
    if w_proc_waxs_qx_shift.value != 0.0:
        backend_opts["waxs_qx_shift_nm"] = w_proc_waxs_qx_shift.value
    if w_proc_waxs_qy_shift.value != 0.0:
        backend_opts["waxs_qy_shift_nm"] = w_proc_waxs_qy_shift.value
    if backend_opts:
        params["backend_options"] = backend_opts

    # Dynamic SAXS masking
    if w_proc_dynamic_mask.value:
        saxs_kw: dict[str, Any] = {"dynamic_saxs_mask": True}
        dyn_kwargs: dict[str, Any] = {}
        shadow: dict[str, Any] = {}
        if not w_proc_dyn_shadow_enabled.value:
            shadow["enabled"] = False
        if w_proc_dyn_shadow_beam_deg.value != DEFAULT_DYN_SHADOW_BEAM_VISIBLE_DEG:
            shadow["beam_visible_deg"] = w_proc_dyn_shadow_beam_deg.value
        if w_proc_dyn_shadow_clear_deg.value != DEFAULT_DYN_SHADOW_CLEAR_EDGE_DEG:
            shadow["clear_edge_deg"] = w_proc_dyn_shadow_clear_deg.value
        if shadow:
            dyn_kwargs["waxs_shadow"] = shadow
        aperture: dict[str, Any] = {}
        if not w_proc_dyn_aper_enabled.value:
            aperture["enabled"] = False
        if w_proc_dyn_aper_agbh_ring.value != DEFAULT_DYN_APER_AGBH_RING_ORDER:
            aperture["agbh_ring_order"] = w_proc_dyn_aper_agbh_ring.value
        if w_proc_dyn_aper_q_margin.value != DEFAULT_DYN_APER_Q_MARGIN_FRACTION:
            aperture["q_margin_fraction"] = w_proc_dyn_aper_q_margin.value
        if w_proc_dyn_aper_q_cutoff.value > 0:
            aperture["q_cutoff"] = w_proc_dyn_aper_q_cutoff.value
        if aperture:
            dyn_kwargs["aperture"] = aperture
        if dyn_kwargs:
            saxs_kw["dynamic_saxs_kwargs"] = dyn_kwargs
        params["saxs_kwargs"] = saxs_kw

    # WAXS kwargs (calibration + masking overrides)
    if waxs_kw:
        params["waxs_kwargs"] = waxs_kw

    return reduce_smi_combined, params, geometry


def _cache_reduction_result(uid: str, result, geometry: str, params: dict) -> None:
    """Extract arrays from a reduction result and write to the disk cache."""
    try:
        cache = ScanCache(uid)
        arrays: dict[str, np.ndarray] = {}

        if geometry == "grazing":
            # GI result: frames list (each is a 2D array)
            frames = getattr(result, "frames", None)
            if frames is not None and len(frames) > 0:
                arrays["gi_frames"] = np.asarray(frames)
        else:
            # Transmission: merged_qchi and merged_iq are xarray Datasets
            qchi = getattr(result, "merged_qchi", None)
            if qchi is not None:
                intensity = qchi["intensity"].values if "intensity" in qchi else None
                if intensity is not None:
                    arrays["qchi_intensity"] = intensity
                if "q" in qchi.coords:
                    arrays["qchi_q"] = qchi["q"].values
                if "chi" in qchi.coords:
                    arrays["qchi_chi"] = qchi["chi"].values

            iq = getattr(result, "merged_iq", None)
            if iq is not None:
                if "q" in iq.coords:
                    arrays["iq_q"] = iq["q"].values
                if "I" in iq:
                    arrays["iq_I"] = iq["I"].values

            # Per-frame I(q) if available
            pf_iq = getattr(result, "per_frame_iq", None)
            if pf_iq is not None and "I" in pf_iq:
                arrays["pf_iq_I"] = pf_iq["I"].values
                if "q" in pf_iq.coords:
                    arrays["pf_iq_q"] = pf_iq["q"].values

        # Filter out params that aren't JSON-safe for attr storage
        safe_params = {
            k: v for k, v in params.items()
            if isinstance(v, (str, int, float, bool, type(None), list, tuple))
        }
        safe_params["geometry"] = geometry

        if arrays:
            cache.write_reduction(arrays, safe_params)
            log.info("cache: wrote reduction for %s (%d arrays)", uid[:8], len(arrays))
    except Exception:
        log.exception("cache: failed to write reduction for %s", uid[:8])


# ---------------------------------------------------------------------------
# Subprocess-isolated batch worker
# ---------------------------------------------------------------------------

def _batch_subprocess_target(uid: str, export_config: dict, conn) -> None:
    """Run one reduction + cache + export in an isolated subprocess.

    This function is the target for ``multiprocessing.Process(target=...)``.
    It inherits the parent's imports via fork but allocates all heavy arrays
    in its OWN address space.  When it exits, the OS reclaims everything —
    no heap fragmentation, no module-level caches lingering.
    """
    import threading as _thr

    # After fork, threading locks inherited from parent may be in a bad
    # state.  Replace the matplotlib serialisation lock with a fresh one
    # so export_scan doesn't deadlock.
    try:
        import smi_browser.export as _export_mod
        _export_mod._MPL_LOCK = _thr.Lock()
    except Exception:
        pass

    # Force a fresh tiled connection (parent's HTTP sockets are not
    # usable in the child after fork).
    global _cat
    _cat = None

    try:
        run_fn, params, geometry = _build_proc_params(uid)
        result = run_fn(**params)

        _cache_reduction_result(uid, result, geometry, params)

        # Fetch scalars for export / summary
        primary_df = baseline_df = config_df = raw_md = None
        try:
            run = _get_cat()[uid]
            primary_df = _scalar_stream_to_frame(run, "primary", uid=uid)
            baseline_df = _scalar_stream_to_frame(run, "baseline", uid=uid)
            config_df = _config_to_dataframe(run)
            raw_md = dict(run.metadata)
            del run
        except Exception:
            pass

        # Build lightweight summary
        start_md = (raw_md or {}).get("start", {})
        summary = {
            "uid": uid,
            "sample_name": start_md.get(
                "sample_name",
                start_md.get("sample", start_md.get("Sample", "?")),
            ),
            "plan_name": start_md.get("plan_name", "?"),
            "scan_id": start_md.get("scan_id", "?"),
        }

        # Auto-export while result is alive
        if export_config.get("auto_export"):
            out_dir = export_config.get("export_dir")
            if out_dir:
                export_scan(
                    out_dir=out_dir,
                    uid=uid,
                    result=result,
                    params=params,
                    primary_df=primary_df,
                    baseline_df=baseline_df,
                    config_df=config_df,
                    raw_metadata=raw_md,
                    formats=export_config.get("formats", []),
                    subdir_template=export_config.get("subdir_template", ""),
                    basename_template=export_config.get("basename_template", ""),
                    frame_label_col=export_config.get("frame_label_col"),
                )

        conn.send(("ok", summary))
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _snapshot_export_config() -> dict:
    """Snapshot current export-related widget values into a plain dict.

    This is called in the parent process before forking so the child has
    all the settings it needs without touching Panel widgets (which may
    not survive fork cleanly in future Panel versions).
    """
    return {
        "auto_export": bool(w_export_auto.value),
        "export_dir": _resolve_export_dir() or "",
        "formats": _export_formats(),
        "subdir_template": w_export_subdir.value or "",
        "basename_template": w_export_basename.value or "",
        "frame_label_col": _get_frame_label_cols() or None,
    }


def _batch_process_fn(uid: str):
    """BatchProcessor.process_fn — runs reduction in an isolated subprocess.

    Each scan's heavy work (reduce + cache + export) runs in a forked child
    process.  When the child exits, ALL memory it allocated is returned to
    the OS unconditionally — no heap fragmentation, no module-level caches.
    This guarantees constant memory usage regardless of how many scans are
    processed.
    """
    import multiprocessing as mp

    export_config = _snapshot_export_config()

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_batch_subprocess_target,
        args=(uid, export_config, child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()  # parent doesn't write to child's end

    # Wait for result from child
    try:
        status, data = parent_conn.recv()
    except EOFError:
        proc.join(timeout=10)
        raise RuntimeError(
            f"Batch subprocess for {uid[:8]} died unexpectedly "
            f"(exit code {proc.exitcode})"
        )
    finally:
        parent_conn.close()

    proc.join(timeout=300)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise RuntimeError(f"Batch subprocess for {uid[:8]} timed out")

    if proc.exitcode and proc.exitcode != 0:
        raise RuntimeError(
            f"Batch subprocess for {uid[:8]} exited with code {proc.exitcode}"
        )

    if status == "error":
        raise RuntimeError(data)

    summary = data
    return None, summary, {}


def _batch_skip(uid: str) -> bool:
    if w_batch_skip_processed.value and _batch_processed_contains(uid):
        return True
    if w_batch_skip_existing.value and uid in _collection:
        return True
    return False


def _batch_dispatch(snap: dict) -> None:
    """status_cb: marshal UI updates onto the Bokeh document thread."""

    def _apply():
        try:
            states = snap["states"]
            total = snap["total"]
            done = (
                states["done"] + states["error"]
                + states["skipped"] + states["cancelled"]
            )
            running = snap["running"]

            w_batch_progress.max = max(total, 1)
            w_batch_progress.value = done
            w_batch_progress.visible = total > 0

            label = "running" if running else (
                "cancelling" if snap["cancel_requested"] else "idle"
            )
            w_batch_status.object = (
                f"**Batch {label}** — {done}/{total} processed "
                f"(done={states['done']}, error={states['error']}, "
                f"skipped={states['skipped']}, cancelled={states['cancelled']}, "
                f"queued={states['queued']}, running={states['running']})"
            )

            # Build table from full job list
            jobs_view = snap["jobs"]
            rows = []
            for j in jobs_view:
                dur = j.get("duration_s")
                rows.append({
                    "uid_short": (j["uid"] or "")[:8],
                    "label": j.get("label", ""),
                    "state": j["state"],
                    "duration_s": f"{dur:.1f}" if dur else "",
                    "error": j.get("error", ""),
                })
            new_df = pd.DataFrame(
                rows,
                columns=["uid_short", "label", "state", "duration_s", "error"],
            )
            # Always do a full replace — patching is slow for large tables
            # and caused delayed/stale UI.
            w_batch_table.value = new_df

            # Apply row colours based on job state.
            def _color_batch_rows(row):
                css = _BATCH_ROW_COLORS.get(row["state"], "")
                return [css] * len(row)
            w_batch_table.style.apply(_color_batch_rows, axis=1)

            w_btn_batch_queue.disabled = running
            w_btn_batch_cancel.disabled = not running
            w_btn_batch_clear.disabled = running or total == 0
            w_btn_batch_clear_processed.disabled = running
            w_batch_processed_status.object = (
                f"*Processed history: {_batch_processed_count()} uids*"
            )
            w_batch_memory_status.object = _batch_memory_status_text()

            # Keep the collection panel in sync as jobs land.
            if w_batch_add_to_collection.value:
                _refresh_collection()
        except Exception:
            log.exception("batch: UI render failed")

    doc = _batch_state.get("doc")
    if doc is None:
        _apply()
        return
    try:
        doc.add_next_tick_callback(_apply)
    except Exception:
        log.exception("batch: add_next_tick_callback failed")


class _CachedResult:
    """Minimal stub with .uid and .merged_iq for collection display."""

    def __init__(self, uid: str, merged_iq, geometry: str = "transmission"):
        self.uid = uid
        self.merged_iq = merged_iq
        self.merged_qchi = None
        self.per_frame_iq = None
        self.timing = None
        self.geometry = geometry


def _batch_add_to_collection_from_cache(uid: str, summary: dict) -> None:
    """Reload cached I(q) arrays and add a lightweight result to collection."""
    import xarray as xr

    try:
        cache = ScanCache(uid)
        cached = cache.read_reduction()
        if cached is None:
            return
        arrays = cached.get("arrays", {})
        params = cached.get("params", {})
        q = arrays.get("iq_q")
        I = arrays.get("iq_I")
        if q is None or I is None:
            return
        # Build minimal xarray Dataset matching what collection expects
        merged_iq = xr.Dataset(
            {"I": xr.DataArray(I, dims=["q"], coords={"q": q})},
        )
        geometry = params.get("geometry", "transmission")
        stub = _CachedResult(uid, merged_iq, geometry=geometry)
        _collection.add(stub, summary)
    except Exception:
        log.debug("batch: collection add from cache failed for %s", uid[:8])


def _batch_add_fn(result, summary, params):
    """Lightweight bookkeeping after a batch job completes.

    Heavy reduction runs in a subprocess.  This function (in the parent)
    records the uid and optionally reloads the cached I(q) into the
    collection for live display (only when collection add is enabled,
    which is auto-disabled for batches > 50 scans).
    """
    uid = summary.get("uid") if isinstance(summary, dict) else None
    try:
        if uid:
            _batch_processed_add(uid)
            # Reload from disk cache into collection (lightweight — only
            # the merged I(q) arrays, not the full result).
            if w_batch_add_to_collection.value:
                _batch_add_to_collection_from_cache(uid, summary)
    finally:
        result = None
        summary = None
        params = None


def _prefetch_image_caches(uids: list[str]) -> None:
    """Ensure an HDF5 cache file exists for each UID.

    When ``reduce_smi_combined`` receives an ``image_cache_path`` pointing
    to an existing file, it skips the expensive ``populate_cache`` step
    (which re-fetches images from Tiled a second time just to write them).
    Creating the empty file is enough — the reduction still loads images
    from Tiled once, but avoids the redundant second fetch+write.
    """
    import h5py

    for uid in uids:
        cp = cache_path(uid)
        if not cp.exists():
            cp.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(cp, "w"):
                pass  # empty file is sufficient
            log.debug("batch: created empty cache for %s", uid[:8])


def _ensure_batch_processor() -> BatchProcessor:
    bp = _batch_state.get("processor")
    workers = _batch_effective_workers()
    # Re-create when worker count changes or the previous run finished
    # (BatchProcessor is single-shot in the sense that its worker threads
    # exit when the queue drains; restarting cleanly = new instance).
    if bp is None or bp._max_workers != workers or not bp.is_running:
        if bp is not None and bp.is_running:
            return bp
        bp = BatchProcessor(
            process_fn=_batch_process_fn,
            add_fn=_batch_add_fn,
            status_cb=_batch_dispatch,
            skip_fn=_batch_skip,
            max_workers=workers,
        )
        _batch_state["processor"] = bp
    return bp


def _on_batch_queue(event):
    total = _state.get("total", 0)
    if total == 0:
        pn.state.notifications.warning("No search results to queue.")
        return
    # Capture the current Bokeh document on the UI thread for cross-thread
    # dispatch from worker threads.
    try:
        _batch_state["doc"] = pn.state.curdoc
    except Exception:
        _batch_state["doc"] = None
    _batch_state["rss_peak_mb"] = 0.0

    if w_batch_low_memory_mode.value and (w_batch_max_workers.value or 1) > 1:
        pn.state.notifications.info(
            "Low-memory mode is on; batch workers are forced to 1."
        )

    max_jobs = max(1, int(w_batch_max_jobs.value or 25))
    skip_existing = w_batch_skip_existing.value
    skip_processed = w_batch_skip_processed.value

    # Use UIDs from the current search table (already fetched for display)
    # then page forward through more results if needed.  This guarantees
    # the same search filters and ordering the user sees.
    items: list[tuple[str, str]] = []
    unified = _with_cycle_filter(_state.get("unified_filters", []))
    page_size = _state.get("page_size", PAGE_SIZE)

    # Start from UIDs already visible in the table
    df = w_table.value
    if df is not None and len(df) > 0 and "uid" in df.columns:
        for uid in df["uid"].tolist():
            if not uid:
                continue
            if skip_processed and _batch_processed_contains(uid):
                continue
            if skip_existing and uid in _collection:
                continue
            items.append((uid, ""))
            if len(items) >= max_jobs:
                break

    # If we need more, page through additional results
    offset = page_size  # skip current page (already processed above)
    while len(items) < max_jobs and offset < total:
        page_summaries, _ = tb.fetch_page_fast(
            _get_cat(), unified_filters=unified or None,
            offset=offset, limit=page_size,
        )
        if not page_summaries:
            break
        for s in page_summaries:
            uid = s.get("uid", "")
            if not uid:
                continue
            if skip_processed and _batch_processed_contains(uid):
                continue
            if skip_existing and uid in _collection:
                continue
            items.append((uid, ""))
            if len(items) >= max_jobs:
                break
        offset += page_size

    if not items:
        pn.state.notifications.info("Nothing to queue (all already processed).")
        return

    _batch_state["summaries"] = {}  # populated lazily during processing

    # Auto-disable collection add for large batches (memory grows per scan
    # stored in the collection).
    _BATCH_COLLECTION_MAX = 50
    if len(items) > _BATCH_COLLECTION_MAX and w_batch_add_to_collection.value:
        w_batch_add_to_collection.value = False
        pn.state.notifications.warning(
            f"Collection add disabled — batch has >{_BATCH_COLLECTION_MAX} "
            f"scans. Re-enable manually for small batches."
        )

    # Ensure empty cache files exist so reduce_smi_combined skips the
    # redundant populate_cache (second Tiled fetch) for each scan.
    _prefetch_image_caches([uid for uid, _label in items])

    bp = _ensure_batch_processor()
    n = bp.enqueue(items)
    if n == 0:
        pn.state.notifications.info("Nothing to queue (all already tracked).")
        return
    bp.start()
    pn.state.notifications.success(
        f"Queued {n} scan{'s' if n != 1 else ''} for batch processing."
    )


def _on_batch_cancel(event):
    bp = _batch_state.get("processor")
    if bp is None:
        return
    bp.cancel()
    pn.state.notifications.info(
        "Cancellation requested; the running job will finish."
    )


def _on_batch_clear(event):
    bp = _batch_state.get("processor")
    if bp is None:
        return
    bp.clear_terminal()


def _on_batch_clear_processed(event):
    n = _batch_processed_clear()
    w_batch_processed_status.object = "*Processed history: 0 uids*"
    pn.state.notifications.info(
        f"Cleared processed history ({n} uid{'s' if n != 1 else ''})."
    )


w_btn_batch_queue.on_click(_on_batch_queue)
w_btn_batch_cancel.on_click(_on_batch_cancel)
w_btn_batch_clear.on_click(_on_batch_clear)
w_btn_batch_clear_processed.on_click(_on_batch_clear_processed)


batch_panel = pn.Column(
    pn.pane.Markdown(
        "**Batch process scans from the current search results.** "
        "Each job uses the parameters configured in the *Parameters* "
        "sub-tab above.  Reductions run on a background thread so the "
        "interface stays interactive.  You can optionally keep results out "
        "of the Scan Collection to reduce memory usage; processed uid "
        "history can be used to skip repeats.",
    ),
    pn.Row(
        w_btn_batch_queue,
        w_btn_batch_cancel,
        w_btn_batch_clear,
        w_btn_batch_clear_processed,
    ),
    pn.Row(w_batch_max_jobs, w_batch_max_workers),
    pn.Row(w_batch_low_memory_mode),
    pn.Row(
        w_batch_skip_existing,
        w_batch_skip_processed,
        w_batch_add_to_collection,
    ),
    w_batch_processed_status,
    w_batch_status,
    w_batch_progress,
    w_batch_table,
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------------
# Export tab — configurable output formats and destinations
# ---------------------------------------------------------------------------

from smi_browser.export import export_scan, resolve_output_dir

# --- Widgets ---

w_export_dir = pn.widgets.TextInput(
    name="Relative path (within proposal directory)",
    value="projects/{project_name}/analysis",
    placeholder="projects/{project_name}/analysis",
    width=500,
)
w_export_resolved_path = pn.pane.Markdown("", sizing_mode="stretch_width")
w_export_subdir = pn.widgets.TextInput(
    name="Subdirectory template",
    value="",
    placeholder="e.g. {uid_short}_{sample_name}",
    width=300,
)
w_export_basename = pn.widgets.TextInput(
    name="Base filename template",
    value="{sample_name}_{scan_id}",
    width=300,
)
w_export_frame_label_1 = pn.widgets.Select(
    name="Frame label 1",
    options=["(none)"],
    value="(none)",
    width=180,
)
w_export_frame_label_2 = pn.widgets.Select(
    name="Frame label 2",
    options=["(none)"],
    value="(none)",
    width=180,
)
w_export_frame_label_3 = pn.widgets.Select(
    name="Frame label 3",
    options=["(none)"],
    value="(none)",
    width=180,
)

# Format checkboxes
w_export_h5 = pn.widgets.Checkbox(name="HDF5 result", value=True)
w_export_png_2d = pn.widgets.Checkbox(name="PNG 2D map", value=True)
w_export_png_iq = pn.widgets.Checkbox(name="PNG I(q)", value=True)
w_export_png_linecuts = pn.widgets.Checkbox(name="PNG linecuts", value=True)
w_export_csv_iq = pn.widgets.Checkbox(name="CSV I(q) + per-frame", value=True)
w_export_csv_scalars = pn.widgets.Checkbox(name="CSV primary scalars", value=False)
w_export_csv_baseline = pn.widgets.Checkbox(name="CSV baseline", value=False)
w_export_metadata = pn.widgets.Checkbox(name="Metadata JSON", value=False)

w_export_auto = pn.widgets.Checkbox(
    name="Auto-export after processing",
    value=False,
)

w_btn_export_current = pn.widgets.Button(
    name="Export current scan", button_type="primary", width=180,
)
w_btn_export_collection = pn.widgets.Button(
    name="Export entire collection", button_type="success", width=180,
)
w_btn_download_current = pn.widgets.Button(
    name="⬇ Download current scan", button_type="light", width=180,
)
w_btn_download_collection = pn.widgets.Button(
    name="⬇ Download collection", button_type="light", width=180,
)
w_export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
w_export_spinner = pn.indicators.LoadingSpinner(
    value=False, visible=False, size=20,
)


def _refresh_export_labels():
    """Populate the frame-label dropdowns from the current scan's primary scalars."""
    options = ["(none)"]
    df = w_primary_table.value
    if df is not None and not df.empty:
        import pandas as pd
        options += [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
    for w in (w_export_frame_label_1, w_export_frame_label_2, w_export_frame_label_3):
        prev = w.value
        w.options = options
        if prev in options:
            w.value = prev
        else:
            w.value = "(none)"


def _refresh_export_resolved_path():
    """Show the auto-resolved output path in the Export tab."""
    ds = w_proposal_select.value
    proj = w_proposal_project.value
    rel = w_export_dir.value.strip() or "projects/{project_name}/analysis"
    scan_cycle = None

    uid = _state.get("selected_uid")
    if uid:
        try:
            run = _get_cat()[uid]
            md = dict(run.metadata)
            scan_cycle = md.get("start", {}).get("cycle")
        except Exception:
            scan_cycle = None

    # Try proposal dropdown first
    if ds and not ds.startswith("("):
        resolved = resolve_output_dir(ds, proj, cycle=scan_cycle, relative_path=rel)
        if resolved:
            w_export_resolved_path.object = f"*Resolved: `{resolved}`*"
            return

    # Fallback: use scan metadata
    if uid:
        try:
            run = _get_cat()[uid]
            md = dict(run.metadata)
            scan_ds = md.get("start", {}).get("data_session")
            scan_cycle = md.get("start", {}).get("cycle")
            if scan_ds:
                resolved = resolve_output_dir(
                    scan_ds, proj, cycle=scan_cycle, relative_path=rel,
                )
                if resolved:
                    w_export_resolved_path.object = (
                        f"*Resolved (from scan): `{resolved}`*"
                    )
                    return
        except Exception:
            pass

    w_export_resolved_path.object = "*Cannot resolve — select a proposal or scan.*"


def _get_frame_label_cols() -> list[str]:
    """Return the list of selected frame label columns (filtering out '(none)')."""
    cols = []
    for w in (w_export_frame_label_1, w_export_frame_label_2, w_export_frame_label_3):
        v = w.value
        if v and v != "(none)":
            cols.append(v)
    return cols


def _export_formats() -> set[str]:
    """Gather selected export format keys from checkboxes."""
    fmts: set[str] = set()
    if w_export_h5.value:
        fmts.add("h5")
    if w_export_png_2d.value:
        fmts.add("png_2d")
    if w_export_png_iq.value:
        fmts.add("png_iq")
    if w_export_png_linecuts.value:
        fmts.add("png_linecuts")
    if w_export_csv_iq.value:
        fmts.add("csv_iq")
    if w_export_csv_scalars.value:
        fmts.add("csv_scalars")
    if w_export_csv_baseline.value:
        fmts.add("csv_baseline")
    if w_export_metadata.value:
        fmts.add("metadata_txt")
    return fmts


def _resolve_export_dir() -> Path | None:
    """Get export output directory rooted in the proposal directory."""
    ds = w_proposal_select.value
    proj = w_proposal_project.value
    rel = w_export_dir.value.strip() or "projects/{project_name}/analysis"
    scan_cycle = None

    # If project is "(all)", try to get the actual project name from scan metadata
    if not proj or proj == "(all)":
        uid = _state.get("selected_uid")
        if uid:
            try:
                run = _get_cat()[uid]
                md = dict(run.metadata)
                proj = md.get("start", {}).get("project_name") or proj
                scan_cycle = md.get("start", {}).get("cycle")
            except Exception:
                pass
    else:
        uid = _state.get("selected_uid")
        if uid:
            try:
                run = _get_cat()[uid]
                md = dict(run.metadata)
                scan_cycle = md.get("start", {}).get("cycle")
            except Exception:
                pass

    # If proposal dropdown has a valid data session, use it directly
    if ds and not ds.startswith("("):
        resolved = resolve_output_dir(ds, proj, cycle=scan_cycle, relative_path=rel)
        if resolved:
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved

    # Fallback: extract data_session from the currently selected scan's metadata
    uid = _state.get("selected_uid")
    if uid:
        try:
            run = _get_cat()[uid]
            md = dict(run.metadata)
            scan_ds = md.get("start", {}).get("data_session")
            scan_cycle = md.get("start", {}).get("cycle")
            if scan_ds:
                resolved = resolve_output_dir(
                    scan_ds, proj, cycle=scan_cycle, relative_path=rel,
                )
                if resolved:
                    resolved.mkdir(parents=True, exist_ok=True)
                    return resolved
        except Exception:
            pass

    return None


def _do_export_single(uid: str) -> tuple[Path, list[str]] | None:
    """Export one scan using current settings.  Returns (dir, files) or None."""
    out_dir = _resolve_export_dir()
    if out_dir is None:
        pn.state.notifications.error(
            "Cannot resolve export directory. Set it manually or select a proposal."
        )
        return None

    result = _proc_result_cache.get("result")
    gi_result = _proc_result_cache.get("gi_result")
    params = _last_result.get("params") or {}

    # Gather 2D cache
    proc_2d = None
    if _proc_2d_cache.get("image") is not None:
        proc_2d = {
            "x": _proc_2d_cache.get("x"),
            "y": _proc_2d_cache.get("y"),
            "image": _proc_2d_cache.get("image"),
            "x_label": _proc_2d_cache.get("x_label", ""),
            "y_label": _proc_2d_cache.get("y_label", ""),
            "title": _proc_2d_cache.get("title", ""),
        }

    # Gather cuts
    cuts = list(_persisted_cuts)

    # Primary/baseline/config
    primary_df = w_primary_table.value if _detail_cache.get("primary_loaded") else None
    baseline_df = None
    config_df = None
    if _detail_cache.get("baseline_loaded"):
        run = _ensure_run()
        if run and "baseline" in tb.stream_names(run):
            try:
                baseline_df = _scalar_stream_to_frame(run, "baseline", uid=uid)
            except Exception:
                pass
        if run:
            try:
                config_df = _config_to_dataframe(run)
            except Exception:
                pass

    # Raw metadata
    raw_md = None
    try:
        run = _ensure_run()
        if run:
            raw_md = dict(run.metadata)
    except Exception:
        pass

    # Raw detector images
    raw_images = None
    if "h5" in _export_formats():
        raw_images = {}
        image_fields = _image_cache.get("fields") or []
        run = _ensure_run()
        if run and image_fields:
            for field in image_fields:
                try:
                    from smi_browser.cache import cache_path as _cache_path
                    import h5py as _h5
                    cp = _cache_path(uid)
                    if cp.exists():
                        with _h5.File(cp, "r") as cf:
                            if f"images/{field}" in cf:
                                raw_images[field] = cf[f"images/{field}"][:]
                                continue
                    # Fallback: fetch from tiled
                    raw_images[field] = tb.fetch_all_frames(run, "primary", field)
                except Exception:
                    log.debug("Could not fetch raw images for field %s", field)

    # Frame labels from primary scalars
    frame_labels = None
    label_cols = _get_frame_label_cols()
    if label_cols and primary_df is not None and not primary_df.empty:
        parts = []
        for col in label_cols:
            if col in primary_df.columns:
                parts.append(primary_df[col].astype(str))
        if parts:
            import pandas as pd
            frame_labels = pd.concat(parts, axis=1).apply(
                lambda row: " | ".join(row), axis=1,
            ).tolist()

    return export_scan(
        out_dir=out_dir,
        uid=uid,
        result=result,
        gi_result=gi_result,
        cuts=cuts,
        proc_2d_cache=proc_2d,
        params=params,
        primary_df=primary_df,
        baseline_df=baseline_df,
        config_df=config_df,
        raw_metadata=raw_md,
        raw_images=raw_images if raw_images else None,
        frame_labels=frame_labels,
        formats=_export_formats(),
        subdir_template=w_export_subdir.value,
        basename_template=w_export_basename.value,
        frame_label_col=_get_frame_label_cols() or None,
    )


def _on_export_current(event):
    uid = _state.get("selected_uid")
    if not uid:
        pn.state.notifications.warning("No scan selected.")
        return
    w_export_spinner.value = True
    w_export_spinner.visible = True
    w_export_status.object = "*Exporting…*"
    try:
        out = _do_export_single(uid)
        if out:
            scan_dir, files = out
            w_export_status.object = (
                f"**Exported** {len(files)} items → `{scan_dir}`\n\n"
                + "\n".join(f"- {f}" for f in files)
            )
    except Exception as exc:
        log.exception("Export failed")
        w_export_status.object = f"**Export error:** `{exc}`"
    finally:
        w_export_spinner.value = False
        w_export_spinner.visible = False


def _on_export_collection(event):
    if len(_collection) == 0:
        pn.state.notifications.warning("Collection is empty.")
        return
    out_dir = _resolve_export_dir()
    if out_dir is None:
        pn.state.notifications.error(
            "Cannot resolve export directory. Set it manually or select a proposal."
        )
        return
    w_export_spinner.value = True
    w_export_spinner.visible = True
    w_export_status.object = f"*Exporting {len(_collection)} scans…*"
    total_files = 0
    errors = 0
    try:
        for coll_uid in _collection.uids:
            coll_result = _collection.get_result(coll_uid)
            coll_params = _collection._processing.get(coll_uid, {})
            coll_primary = _collection.get_primary_df(coll_uid)
            coll_baseline = _collection.get_baseline_df(coll_uid)
            coll_config = _collection.get_config_df(coll_uid)
            coll_raw_md = _collection.get_raw_metadata(coll_uid)

            try:
                _, files = export_scan(
                    out_dir=out_dir,
                    uid=coll_uid,
                    result=coll_result,
                    params=coll_params,
                    primary_df=coll_primary,
                    baseline_df=coll_baseline,
                    config_df=coll_config,
                    raw_metadata=coll_raw_md,
                    formats=_export_formats(),
                    subdir_template=w_export_subdir.value,
                    basename_template=w_export_basename.value,
                    frame_label_col=_get_frame_label_cols() or None,
                )
                total_files += len(files)
            except Exception:
                log.exception("Export failed for %s", coll_uid[:8])
                errors += 1

        status = f"**Exported** {len(_collection)} scans ({total_files} files) → `{out_dir}`"
        if errors:
            status += f"\n\n⚠️ {errors} scan(s) had errors."
        w_export_status.object = status
    except Exception as exc:
        log.exception("Collection export failed")
        w_export_status.object = f"**Export error:** `{exc}`"
    finally:
        w_export_spinner.value = False
        w_export_spinner.visible = False


w_btn_export_current.on_click(_on_export_current)
w_btn_export_collection.on_click(_on_export_collection)


def _on_download_current(event):
    """Export current scan to a temp dir, zip it, and serve via browser download."""
    import io
    import tempfile
    import zipfile

    uid = _state.get("selected_uid")
    if not uid:
        pn.state.notifications.warning("No scan selected.")
        return
    w_export_spinner.value = True
    w_export_spinner.visible = True
    w_export_status.object = "*Preparing download…*"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            out = _do_export_single_to_dir(uid, Path(tmpdir))
            if not out:
                w_export_status.object = "**Download failed:** could not generate export."
                return
            scan_dir, files = out
            # Create zip in memory
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    fpath = scan_dir / f
                    if fpath.exists():
                        zf.write(fpath, arcname=f)
            buf.seek(0)
            # Determine filename
            uid_short = uid[:8]
            summary = _detail_cache.get("summary") or {}
            sample = summary.get("sample_name", "scan")
            zip_name = f"{sample}_{uid_short}.zip"
            # Serve download via Panel
            from bokeh.models.callbacks import CustomJS
            import base64
            b64 = base64.b64encode(buf.read()).decode()
            # Use pn.state.execute to trigger browser-side download
            js = f"""
            var link = document.createElement('a');
            link.href = 'data:application/zip;base64,{b64}';
            link.download = '{zip_name}';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            pn.state.execute(js)
            w_export_status.object = f"**Download started:** `{zip_name}` ({len(files)} files)"
    except Exception as exc:
        log.exception("Download failed")
        w_export_status.object = f"**Download error:** `{exc}`"
    finally:
        w_export_spinner.value = False
        w_export_spinner.visible = False


def _do_export_single_to_dir(uid: str, out_dir) -> tuple | None:
    """Like _do_export_single but writes to a specified directory."""
    from pathlib import Path

    result = _proc_result_cache.get("result")
    gi_result = _proc_result_cache.get("gi_result")
    params = _last_result.get("params") or {}

    # Gather 2D cache
    proc_2d = None
    if _proc_2d_cache.get("image") is not None:
        proc_2d = {
            "x": _proc_2d_cache.get("x"),
            "y": _proc_2d_cache.get("y"),
            "image": _proc_2d_cache.get("image"),
            "x_label": _proc_2d_cache.get("x_label", ""),
            "y_label": _proc_2d_cache.get("y_label", ""),
            "title": _proc_2d_cache.get("title", ""),
        }

    cuts = list(_persisted_cuts)

    # Primary/baseline/config
    primary_df = w_primary_table.value if _detail_cache.get("primary_loaded") else None
    baseline_df = None
    config_df = None
    run = _ensure_run()
    if _detail_cache.get("baseline_loaded") and run:
        if "baseline" in tb.stream_names(run):
            try:
                baseline_df = _scalar_stream_to_frame(run, "baseline", uid=uid)
            except Exception:
                pass
        try:
            config_df = _config_to_dataframe(run)
        except Exception:
            pass

    # Raw metadata
    raw_md = None
    try:
        if run:
            raw_md = dict(run.metadata)
    except Exception:
        pass

    # Frame labels
    frame_labels = None
    label_cols = _get_frame_label_cols()
    if label_cols and primary_df is not None and not primary_df.empty:
        parts = []
        for col in label_cols:
            if col in primary_df.columns:
                parts.append(primary_df[col].astype(str))
        if parts:
            frame_labels = pd.concat(parts, axis=1).apply(
                lambda row: " | ".join(row), axis=1,
            ).tolist()

    return export_scan(
        out_dir=out_dir,
        uid=uid,
        result=result,
        gi_result=gi_result,
        cuts=cuts,
        proc_2d_cache=proc_2d,
        params=params,
        primary_df=primary_df,
        baseline_df=baseline_df,
        config_df=config_df,
        raw_metadata=raw_md,
        raw_images=None,  # skip raw images for download (too large)
        frame_labels=frame_labels,
        formats=_export_formats(),
        subdir_template="",  # flat — no subdir in zip
        basename_template=w_export_basename.value,
        frame_label_col=_get_frame_label_cols() or None,
    )


w_btn_download_current.on_click(_on_download_current)


def _on_download_collection(event):
    """Export all scans in the collection to a zip and serve via browser download."""
    import io
    import tempfile
    import zipfile

    if len(_collection) == 0:
        pn.state.notifications.warning("Collection is empty.")
        return
    w_export_spinner.value = True
    w_export_spinner.visible = True
    w_export_status.object = f"*Preparing download of {len(_collection)} scans…*"
    try:
        buf = io.BytesIO()
        total_files = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for coll_uid in _collection.uids:
                    coll_result = _collection.get_result(coll_uid)
                    coll_params = _collection._processing.get(coll_uid, {})
                    coll_primary = _collection.get_primary_df(coll_uid)
                    coll_baseline = _collection.get_baseline_df(coll_uid)
                    coll_config = _collection.get_config_df(coll_uid)
                    coll_raw_md = _collection.get_raw_metadata(coll_uid)

                    uid_short = coll_uid[:8]
                    scan_dir = Path(tmpdir) / uid_short
                    scan_dir.mkdir(exist_ok=True)

                    try:
                        _, files = export_scan(
                            out_dir=Path(tmpdir),
                            uid=coll_uid,
                            result=coll_result,
                            params=coll_params,
                            primary_df=coll_primary,
                            baseline_df=coll_baseline,
                            config_df=coll_config,
                            raw_metadata=coll_raw_md,
                            raw_images=None,  # skip raw images for download
                            formats=_export_formats(),
                            subdir_template=w_export_subdir.value or "{uid_short}",
                            basename_template=w_export_basename.value,
                            frame_label_col=_get_frame_label_cols() or None,
                        )
                        # Add files to zip under a scan subdirectory
                        actual_subdir = w_export_subdir.value or "{uid_short}"
                        try:
                            sub_name = actual_subdir.format(
                                uid=coll_uid, uid_short=uid_short,
                                scan_id="", sample_name="",
                            )
                        except Exception:
                            sub_name = uid_short
                        export_dir = Path(tmpdir) / sub_name
                        for f in files:
                            fpath = export_dir / f
                            if fpath.exists():
                                zf.write(fpath, arcname=f"{sub_name}/{f}")
                                total_files += 1
                    except Exception as exc:
                        log.warning("Download: export failed for %s: %s",
                                    coll_uid[:8], exc)

        buf.seek(0)
        zip_name = f"collection_{len(_collection)}scans.zip"
        import base64
        b64 = base64.b64encode(buf.read()).decode()
        js = f"""
        var link = document.createElement('a');
        link.href = 'data:application/zip;base64,{b64}';
        link.download = '{zip_name}';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        """
        pn.state.execute(js)
        w_export_status.object = (
            f"**Download started:** `{zip_name}` "
            f"({len(_collection)} scans, {total_files} files)"
        )
    except Exception as exc:
        log.exception("Collection download failed")
        w_export_status.object = f"**Download error:** `{exc}`"
    finally:
        w_export_spinner.value = False
        w_export_spinner.visible = False


w_btn_download_collection.on_click(_on_download_collection)

export_panel = pn.Column(
    pn.pane.Markdown(
        "**Export processed results** to disk in multiple formats. "
        "Output is always within the proposal directory. "
        "Use `{project_name}` in the relative path to include the selected project."
    ),
    pn.pane.Alert(
        "⚠️ **Permissions note:** Writing to the proposal directory requires that "
        "this app is running under your own user credentials. If you are using a "
        "shared deployment, use the **Download** button instead — it sends files "
        "through your browser, which saves them with your own permissions.",
        alert_type="warning",
    ),
    pn.Row(w_export_dir, sizing_mode="stretch_width"),
    w_export_resolved_path,
    pn.Row(w_export_subdir, w_export_basename),
    pn.pane.Markdown(
        "*Subdirectory template:* puts each scan in its own folder. "
        "Leave blank to put all files in one directory.  \n"
        "*Base filename template:* prepended to all output filenames "
        "(e.g. `{sample_name}_{scan_id}` → `myfilm_12345_iq_merged.csv`).  \n"
        "*Placeholders:* `{uid}`, `{uid_short}`, `{scan_id}`, `{sample_name}`",
        stylesheets=[":host { font-size: 11px; color: #666; }"],
    ),
    pn.Row(w_export_frame_label_1, w_export_frame_label_2, w_export_frame_label_3),
    pn.pane.Markdown(
        "*Frame label columns are appended to per-frame filenames. "
        "If the scan lacks a selected column, it is ignored.*",
        stylesheets=[":host { font-size: 11px; color: #666; }"],
    ),
    pn.layout.Divider(),
    pn.pane.Markdown("**Output formats:**"),
    pn.Row(w_export_h5, w_export_png_2d, w_export_png_iq, w_export_png_linecuts),
    pn.Row(w_export_csv_iq, w_export_csv_scalars, w_export_csv_baseline, w_export_metadata),
    pn.layout.Divider(),
    pn.Row(w_btn_export_current, w_btn_export_collection, w_export_auto),
    pn.Row(w_btn_download_current, w_btn_download_collection),
    pn.Row(w_export_status, w_export_spinner),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------------
# Batch export — fetch many scans and run export_scan() without processing
# ---------------------------------------------------------------------------
#
# Mirrors the Process › Batch tab, but each job opens a scan, fetches the
# primary/baseline scalars + raw metadata (and optionally raw detector
# images), then calls export_scan() with the destination/format options
# configured in the Export tab above.  No reduction is performed, so the
# resulting files exclude processing-derived outputs (PNG 2D map, I(q),
# linecuts, CSV I(q), HDF5 reduction groups) — only metadata, scalar CSVs,
# baseline CSVs, and (if HDF5 is enabled) raw detector frames are written.

w_bxp_status = pn.pane.Markdown(
    "*Idle — queue scans from the current search results to export them in "
    "the background (no processing).*",
    margin=(0, 5),
)
w_bxp_progress = pn.indicators.Progress(
    name="Batch export progress", value=0, max=1, width=400, visible=False,
)
w_bxp_table = pn.widgets.Tabulator(
    pd.DataFrame(columns=["uid_short", "label", "state", "duration_s", "error"]),
    height=320, layout="fit_data_stretch", show_index=False, disabled=True,
    sizing_mode="stretch_width",
)
w_bxp_max_workers = pn.widgets.IntInput(
    name="Workers", value=2, start=1, end=16, width=90,
)
w_bxp_skip_existing = pn.widgets.Checkbox(
    name="Skip uids whose export folder already exists", value=True,
)
w_bxp_max_jobs = pn.widgets.IntInput(
    name="Max jobs", value=PAGE_SIZE, start=1, end=BatchProcessor.MAX_QUEUE,
    width=110,
)
w_btn_bxp_queue = pn.widgets.Button(
    name="Queue scans", button_type="primary",
)
w_btn_bxp_cancel = pn.widgets.Button(
    name="Cancel", button_type="warning", disabled=True,
)
w_btn_bxp_clear = pn.widgets.Button(
    name="Clear log", button_type="light", disabled=True,
)

_bxp_state: dict[str, Any] = {"doc": None, "processor": None}


def _bxp_resolve_subdir(uid: str, raw_metadata: dict | None) -> Path | None:
    """Compute the would-be scan_dir for a uid using current Export settings."""
    out_dir = _resolve_export_dir()
    if out_dir is None:
        return None
    uid_short = uid[:8]
    scan_id = ""
    sample_name = ""
    if raw_metadata:
        start = raw_metadata.get("start", {})
        scan_id = str(start.get("scan_id", ""))
        sample_name = start.get(
            "sample_name", start.get("sample", start.get("Sample", ""))
        )

    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

    tmpl = w_export_subdir.value or ""
    if not tmpl:
        return out_dir
    try:
        return out_dir / tmpl.format(
            uid=uid, uid_short=uid_short,
            scan_id=_safe(scan_id), sample_name=_safe(sample_name),
        )
    except Exception:
        return out_dir / uid_short


def _bxp_process_fn(uid: str):
    """BatchProcessor.process_fn for export-only batches."""
    out_dir = _resolve_export_dir()
    if out_dir is None:
        raise RuntimeError(
            "Cannot resolve export directory. Configure it in the Export tab "
            "(or select a proposal)."
        )

    run = _get_cat()[uid]
    try:
        primary_df = _scalar_stream_to_frame(run, "primary", uid=uid)
    except Exception:
        primary_df = None
    try:
        baseline_df = _scalar_stream_to_frame(run, "baseline", uid=uid)
    except Exception:
        baseline_df = None
    try:
        config_df = _config_to_dataframe(run)
    except Exception:
        config_df = None
    try:
        raw_md = dict(run.metadata)
    except Exception:
        raw_md = None

    formats = _export_formats()

    # Optionally fetch raw detector images for HDF5 output.
    raw_images: dict | None = None
    if "h5" in formats:
        raw_images = {}
        try:
            info = tb.stream_info_for(run, "primary")
            image_fields = list(info.get("images", []) or [])
        except Exception:
            image_fields = []
        for field in image_fields:
            try:
                from smi_browser.cache import cache_path as _cache_path
                import h5py as _h5
                cp = _cache_path(uid)
                if cp.exists():
                    with _h5.File(cp, "r") as cf:
                        if f"images/{field}" in cf:
                            raw_images[field] = cf[f"images/{field}"][:]
                            continue
                raw_images[field] = tb.fetch_all_frames(run, "primary", field)
            except Exception:
                log.debug("batch export: failed raw image fetch for %s/%s",
                          uid[:8], field)
        if not raw_images:
            raw_images = None

    # Frame labels from primary scalars.
    frame_labels = None
    label_cols = _get_frame_label_cols()
    if label_cols and primary_df is not None and not primary_df.empty:
        parts = []
        for col in label_cols:
            if col in primary_df.columns:
                parts.append(primary_df[col].astype(str))
        if parts:
            frame_labels = pd.concat(parts, axis=1).apply(
                lambda row: " | ".join(row), axis=1,
            ).tolist()

    scan_dir, files = export_scan(
        out_dir=out_dir,
        uid=uid,
        result=None,
        gi_result=None,
        cuts=None,
        proc_2d_cache=None,
        params=None,
        primary_df=primary_df,
        baseline_df=baseline_df,
        config_df=config_df,
        raw_metadata=raw_md,
        raw_images=raw_images,
        frame_labels=frame_labels,
        formats=formats,
        subdir_template=w_export_subdir.value,
        basename_template=w_export_basename.value,
        frame_label_col=label_cols or None,
    )

    # Return a triple compatible with BatchProcessor; add_fn is a no-op.
    summary = {"uid": uid, "scan_dir": str(scan_dir), "n_files": len(files)}
    return None, summary, {"scan_dir": str(scan_dir), "n_files": len(files)}


def _bxp_skip(uid: str) -> bool:
    if not w_bxp_skip_existing.value:
        return False
    try:
        scan_dir = _bxp_resolve_subdir(uid, raw_metadata=None)
    except Exception:
        return False
    if scan_dir is None or not scan_dir.exists():
        return False
    try:
        return any(scan_dir.iterdir())
    except Exception:
        return False


def _bxp_add_fn(result, summary, params):
    # No collection involvement — export already happened in process_fn.
    return


def _bxp_dispatch(snap: dict) -> None:
    """status_cb: marshal UI updates onto the Bokeh document thread."""

    def _apply():
        try:
            states = snap["states"]
            total = snap["total"]
            done = (
                states["done"] + states["error"]
                + states["skipped"] + states["cancelled"]
            )
            running = snap["running"]

            w_bxp_progress.max = max(total, 1)
            w_bxp_progress.value = done
            w_bxp_progress.visible = total > 0

            label = "running" if running else (
                "cancelling" if snap["cancel_requested"] else "idle"
            )
            w_bxp_status.object = (
                f"**Batch export {label}** — {done}/{total} processed "
                f"(done={states['done']}, error={states['error']}, "
                f"skipped={states['skipped']}, cancelled={states['cancelled']}, "
                f"queued={states['queued']}, running={states['running']})"
            )

            rows = []
            for j in snap["jobs"]:
                dur = j.get("duration_s")
                rows.append({
                    "uid_short": (j["uid"] or "")[:8],
                    "label": j.get("label", ""),
                    "state": j["state"],
                    "duration_s": f"{dur:.1f}" if dur else "",
                    "error": j.get("error", ""),
                })
            new_df = pd.DataFrame(
                rows,
                columns=["uid_short", "label", "state", "duration_s", "error"],
            )
            old_df = w_bxp_table.value
            if (
                old_df is not None
                and len(old_df) == len(new_df)
                and list(old_df.columns) == list(new_df.columns)
            ):
                patches = {}
                for col in new_df.columns:
                    for idx in range(len(new_df)):
                        ov = old_df.iat[idx, old_df.columns.get_loc(col)]
                        nv = new_df.iat[idx, new_df.columns.get_loc(col)]
                        if ov != nv:
                            patches.setdefault(col, []).append((idx, nv))
                if patches:
                    w_bxp_table.patch(patches)
            else:
                w_bxp_table.value = new_df

            def _color_bxp_rows(row):
                css = _BATCH_ROW_COLORS.get(row["state"], "")
                return [css] * len(row)
            w_bxp_table.style.apply(_color_bxp_rows, axis=1)

            w_btn_bxp_queue.disabled = running
            w_btn_bxp_cancel.disabled = not running
            w_btn_bxp_clear.disabled = running or total == 0
        except Exception:
            log.exception("batch export: UI render failed")

    doc = _bxp_state.get("doc")
    if doc is None:
        _apply()
        return
    try:
        doc.add_next_tick_callback(_apply)
    except Exception:
        log.exception("batch export: add_next_tick_callback failed")


def _ensure_bxp_processor() -> BatchProcessor:
    bp = _bxp_state.get("processor")
    workers = max(1, int(w_bxp_max_workers.value or 1))
    if bp is None or bp._max_workers != workers or not bp.is_running:
        if bp is not None and bp.is_running:
            return bp
        bp = BatchProcessor(
            process_fn=_bxp_process_fn,
            add_fn=_bxp_add_fn,
            status_cb=_bxp_dispatch,
            skip_fn=_bxp_skip,
            max_workers=workers,
        )
        _bxp_state["processor"] = bp
    return bp


def _on_bxp_queue(event):
    total = _state.get("total", 0)
    if total == 0:
        pn.state.notifications.warning("No search results to queue.")
        return
    if _resolve_export_dir() is None:
        pn.state.notifications.error(
            "Cannot resolve export directory. Set it in the Export tab "
            "or select a proposal."
        )
        return
    try:
        _bxp_state["doc"] = pn.state.curdoc
    except Exception:
        _bxp_state["doc"] = None

    max_jobs = max(1, int(w_bxp_max_jobs.value or 25))
    skip_existing = w_bxp_skip_existing.value

    items: list[tuple[str, str]] = []
    unified = _with_cycle_filter(_state.get("unified_filters", []))
    page_size = _state.get("page_size", PAGE_SIZE)

    df = w_table.value
    if df is not None and len(df) > 0 and "uid" in df.columns:
        for uid in df["uid"].tolist():
            if not uid:
                continue
            if skip_existing:
                scan_dir = _bxp_resolve_subdir(uid, raw_metadata=None)
                if scan_dir is not None and scan_dir.exists():
                    try:
                        if any(scan_dir.iterdir()):
                            continue
                    except Exception:
                        pass
            items.append((uid, ""))
            if len(items) >= max_jobs:
                break

    offset = page_size
    while len(items) < max_jobs and offset < total:
        page_summaries, _ = tb.fetch_page_fast(
            _get_cat(), unified_filters=unified or None,
            offset=offset, limit=page_size,
        )
        if not page_summaries:
            break
        for s in page_summaries:
            uid = s.get("uid", "")
            if not uid:
                continue
            if skip_existing:
                scan_dir = _bxp_resolve_subdir(uid, raw_metadata=None)
                if scan_dir is not None and scan_dir.exists():
                    try:
                        if any(scan_dir.iterdir()):
                            continue
                    except Exception:
                        pass
            items.append((uid, ""))
            if len(items) >= max_jobs:
                break
        offset += page_size

    if not items:
        pn.state.notifications.info("Nothing to queue (all already exported).")
        return

    bp = _ensure_bxp_processor()
    n = bp.enqueue(items)
    if n == 0:
        pn.state.notifications.info("Nothing to queue (all already tracked).")
        return
    bp.start()
    pn.state.notifications.success(
        f"Queued {n} scan{'s' if n != 1 else ''} for batch export."
    )


def _on_bxp_cancel(event):
    bp = _bxp_state.get("processor")
    if bp is None:
        return
    bp.cancel()
    pn.state.notifications.info(
        "Cancellation requested; the running job will finish."
    )


def _on_bxp_clear(event):
    bp = _bxp_state.get("processor")
    if bp is None:
        return
    bp.clear_terminal()


w_btn_bxp_queue.on_click(_on_bxp_queue)
w_btn_bxp_cancel.on_click(_on_bxp_cancel)
w_btn_bxp_clear.on_click(_on_bxp_clear)


batch_export_panel = pn.Column(
    pn.pane.Markdown(
        "**Batch export scans from the current search results — no "
        "processing.**  Each job opens the scan, fetches primary/baseline "
        "scalars + raw metadata (and raw detector frames, if HDF5 is on), "
        "then writes them with `export_scan()` using the destination and "
        "format options configured in the *Export* sub-tab.  Processing-"
        "derived outputs (2D map PNG, I(q), linecuts, reduction HDF5 "
        "groups) are skipped since no reduction is performed."
    ),
    pn.Row(w_btn_bxp_queue, w_btn_bxp_cancel, w_btn_bxp_clear),
    pn.Row(w_bxp_max_jobs, w_bxp_max_workers, w_bxp_skip_existing),
    w_bxp_status,
    w_bxp_progress,
    w_bxp_table,
    sizing_mode="stretch_width",
)


# Combine the per-scan / collection export panel with the batch panel
# in a sub-tab strip, mirroring the Process tab layout.
export_tabs = pn.Tabs(
    ("Export", export_panel),
    ("Batch", batch_export_panel),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------------
# Live mode — tiled streaming subscriptions
# ---------------------------------------------------------------------------
#
# When live mode is enabled:
#   * The catalog is subscribed for new-run events (auto-switch to newest run).
#   * The active run's primary table is streamed → explore plot refreshes.
#   * Each image array is streamed → slider end advances + frame auto-renders.
#   * Search / pagination / non-Explore tabs / Process / Collection are locked
#     so the user can't accidentally navigate away mid-stream.
# Toggling live mode off restores all widgets to their pre-live state.

EXPLORE_TAB_INDEX = 3

_live: dict[str, Any] = {
    "manager": None,
    "active": False,
    "saved": {},  # widget -> {param_name: prev_value}
    "doc": None,  # captured Bokeh document for cross-thread dispatch
    "frame_seq": 0,  # debounce counter for frame-extended events
}

# ---------------------------------------------------------------------------
# Tiled authentication (login / logout / status)
# ---------------------------------------------------------------------------
#
# The NSLS-II tiled server uses session tokens that expire after a few days.
# When tokens lapse, every fetch starts returning HTTP 401.  These widgets
# let the user re-authenticate from inside the running Panel app (no need
# to drop to a terminal and run ``python -c "from tiled.client import
# from_uri; from_uri(...).login()"``).
#
# After a successful login, the tokens get cached on disk via the standard
# tiled token cache, so the cached tokens are also picked up by any
# subprocess / fresh ``from_uri`` (including the one that PyHyperScattering
# creates internally inside ``reduce_smi_combined``).  We also clear our
# module-level ``_cat`` so the next access re-builds the catalog with the
# refreshed credentials.

def _tiled_whoami() -> str | None:
    """Return the username for the currently cached tiled session, or None."""
    try:
        from tiled.client.context import Context

        context, _ = Context.from_any_uri(DEFAULT_TILED_URI)
        if not context.use_cached_tokens():
            return None
        info = context.whoami()
    except Exception:
        return None
    if not info:
        return None
    identities = info.get("identities") or []
    for ident in identities:
        if ident.get("id"):
            return str(ident["id"])
    return None


def _tiled_login(username: str, password: str) -> str:
    """Authenticate against tiled with username/password.

    Returns the logged-in username on success; raises on failure.
    """
    from tiled.client.context import Context, password_grant

    if not username or not password:
        raise ValueError("Username and password are required.")

    context, _ = Context.from_any_uri(DEFAULT_TILED_URI)
    providers = context.server_info.authentication.providers
    if not providers:
        raise RuntimeError("Tiled server reports no authentication providers.")
    spec = providers[0]
    auth_endpoint = spec.links["auth_endpoint"]
    tokens = password_grant(
        context.http_client, auth_endpoint, spec.provider, username, password,
    )
    context.configure_auth(tokens, remember_me=True)

    # Drop the cached catalog so the next access uses the refreshed tokens.
    global _cat
    _cat = None

    info = context.whoami()
    identities = (info or {}).get("identities") or []
    return identities[0]["id"] if identities else username


def _tiled_logout() -> None:
    """Clear the cached tiled session for this server."""
    try:
        from tiled.client.context import Context

        context, _ = Context.from_any_uri(DEFAULT_TILED_URI)
        if context.use_cached_tokens():
            try:
                context.logout()
            except Exception:
                pass
    except Exception:
        pass
    finally:
        global _cat
        _cat = None


w_login_status = pn.pane.Markdown("*checking…*", width=220)
w_btn_login = pn.widgets.Button(
    name="🔑 Login", button_type="primary", width=90,
)
w_btn_logout = pn.widgets.Button(
    name="Logout", button_type="light", width=80, visible=False,
)

w_login_user = pn.widgets.TextInput(
    name="Username", placeholder="bnl username", width=220,
)
w_login_pass = pn.widgets.PasswordInput(
    name="Password", placeholder="password", width=220,
)
w_login_submit = pn.widgets.Button(
    name="Sign in", button_type="success", width=90,
)
w_login_msg = pn.pane.Markdown("", width=220)
w_login_form = pn.Column(
    pn.pane.Markdown("**Tiled login**"),
    w_login_user,
    w_login_pass,
    pn.pane.Alert(
        "After signing in, check for Duo confirmation.",
        alert_type="warning",
        margin=(0, 0, 8, 0),
    ),
    pn.Row(w_login_submit),
    w_login_msg,
    visible=False,
    width=260,
    styles={
        "background": "#f8f9fa",
        "border": "1px solid #ced4da",
        "border-radius": "6px",
        "padding": "10px",
    },
)


def _refresh_login_status():
    user = _tiled_whoami()
    if user:
        w_login_status.object = f"🟢 **Logged in:** `{user}`"
        w_btn_login.name = "🔄 Re-login"
        w_btn_logout.visible = True
    else:
        w_login_status.object = "🔴 **Not logged in**"
        w_btn_login.name = "🔑 Login"
        w_btn_logout.visible = False


def _toggle_login_form(event=None):
    w_login_form.visible = not w_login_form.visible
    if w_login_form.visible:
        w_login_msg.object = ""
        w_login_pass.value = ""


def _on_login_submit(event=None):
    user_in = (w_login_user.value or "").strip()
    pwd = w_login_pass.value or ""
    w_login_msg.object = "*signing in... check Duo confirmation prompt*"
    w_login_submit.disabled = True
    try:
        user = _tiled_login(user_in, pwd)
        w_login_msg.object = f"✅ Signed in as `{user}`"
        w_login_pass.value = ""
        w_login_form.visible = False
        _refresh_login_status()
        # Load proposals for the newly logged-in user
        _refresh_proposals(w_proposal_cycle.value)
        # Trigger a search now that we have credentials
        try:
            _do_search(page=0)
        except Exception:
            pass
        try:
            pn.state.notifications.success(f"Tiled login OK ({user})")
        except Exception:
            pass
    except Exception as exc:
        w_login_msg.object = f"❌ {type(exc).__name__}: {exc}"
    finally:
        w_login_submit.disabled = False


def _on_logout(event=None):
    _tiled_logout()
    _refresh_login_status()
    # Clear proposals on logout
    w_proposal_select.options = ["(log in first)"]
    w_proposal_select.value = "(log in first)"
    w_proposal_status.object = ""
    try:
        pn.state.notifications.info("Logged out of tiled.")
    except Exception:
        pass


w_btn_login.on_click(_toggle_login_form)
w_login_submit.on_click(_on_login_submit)
# PasswordInput commits `value` on Enter. Mirror that to button clicks so
# pressing Enter in the password field submits just like clicking Sign in.
w_login_pass.jscallback(args={"submit": w_login_submit}, value="submit.clicks += 1")
w_btn_logout.on_click(_on_logout)


w_btn_live = pn.widgets.Toggle(
    name="🔴 Go Live", value=False,
    button_type="danger", width=140,
)

w_btn_update = pn.widgets.Button(
    name="🔄 Update", button_type="light", width=100,
)
w_live_banner = pn.pane.Markdown(
    "", visible=False,
    styles={
        "background": "#fff3cd",
        "border": "2px solid #d62728",
        "padding": "8px 12px",
        "border-radius": "6px",
        "font-size": "13px",
        "color": "#5a1a1a",
    },
    sizing_mode="stretch_width",
)


def _dispatch_to_doc(fn):
    """Schedule ``fn`` (zero-arg) on the Bokeh document thread.

    Streaming callbacks fire on tiled's ThreadPoolExecutor.  Bokeh requires
    that any document mutation happen under the document lock; the only
    safe cross-thread entry point is ``Document.add_next_tick_callback``.

    ``pn.state.curdoc`` is *thread-local*, so reading it from a worker
    thread returns ``None``.  We therefore capture the document once on
    the UI thread when live mode starts (see ``_enter_live_mode``) and
    use that captured reference here.

    If the captured document has been destroyed (e.g. WebSocket reconnect),
    we fall back to scheduling via the Tornado IOLoop directly.
    """
    doc = _live.get("doc")
    if doc is None:
        # No document captured — best effort inline (unit tests, REPL).
        try:
            fn()
        except Exception:
            log.exception("live: inline dispatch failed")
        return
    # Check if doc is still alive (destroyed docs lose _change_callbacks).
    try:
        if not hasattr(doc.callbacks, '_change_callbacks'):
            raise AttributeError("document destroyed")
        doc.add_next_tick_callback(fn)
    except (AttributeError, RuntimeError):
        # Document was destroyed (session ended / WebSocket reconnect).
        # Fall back to the Tornado IOLoop which survives across sessions.
        from tornado.ioloop import IOLoop
        try:
            IOLoop.current().add_callback(fn)
        except Exception:
            log.debug("live: IOLoop fallback dispatch also failed")
    except Exception:
        log.exception("live: add_next_tick_callback failed")


def _live_save(widget, *params):
    """Snapshot widget params into _live['saved'] (idempotent per session)."""
    bucket = _live["saved"].setdefault(widget, {})
    for p in params:
        if p not in bucket:
            try:
                bucket[p] = getattr(widget, p)
            except Exception:
                pass


def _live_restore_all():
    """Restore all snapshotted widget params."""
    for widget, params in _live["saved"].items():
        for p, v in params.items():
            try:
                setattr(widget, p, v)
            except Exception:
                pass
    _live["saved"].clear()


def _live_set_lockout(on: bool) -> None:
    """Disable / re-enable widgets that aren't safe to use mid-stream."""
    if on:
        # Snapshot before mutating.
        for w in (w_btn_search, w_btn_reset, w_btn_add_filter,
                  w_btn_first, w_btn_prev, w_btn_next, w_btn_last,
                  w_btn_process, w_btn_add_collection,
                  w_btn_coll_remove):
            _live_save(w, "disabled")
            w.disabled = True
        for rd in _filter_rows:
            for key in ("type", "key", "val", "suggest", "remove"):
                wgt = rd.get(key)
                if wgt is not None:
                    _live_save(wgt, "disabled")
                    wgt.disabled = True
        # Lock the table (no row selection mid-stream).
        _live_save(w_table, "selectable")
        try:
            w_table.selectable = False
        except TypeError:
            pass  # Panel Tabulator._update_selectable signature mismatch
        # Snapshot current tab + force Explore.
        _live_save(w_detail_tabs, "active")
        w_detail_tabs.active = EXPLORE_TAB_INDEX
        # Hide collection/search cards so they don't tempt clicks.
        _live_save(search_card, "collapsed")
        search_card.collapsed = True
        _live_save(collection_card, "collapsed")
        collection_card.collapsed = True
    else:
        _live_restore_all()


def _live_set_banner(text: str) -> None:
    w_live_banner.object = text
    w_live_banner.visible = bool(text)


def _live_pick_initial_uid() -> str | None:
    """Return the most recent run uid in the catalog, or None."""
    try:
        summaries, _total = tb.fetch_page_fast(
            _get_cat(), unified_filters=_with_cycle_filter([]), offset=0, limit=1,
        )
    except Exception as exc:
        log.warning("live: initial uid fetch failed: %s", exc)
        return None
    if not summaries:
        return None
    return summaries[0].get("uid")


def _live_switch_to(uid: str) -> None:
    """Switch the detail panel to ``uid`` and (re)subscribe to its streams.

    Runs on the Bokeh document thread.
    """
    if not uid or uid == _state.get("selected_uid"):
        # Same uid (or empty) — nothing to do beyond keeping the watch alive.
        if uid and _live["manager"] is not None:
            try:
                run = _ensure_run()
                if run is not None and _live["manager"].watched_uid != uid:
                    _live["manager"].watch_run(uid, run)
            except Exception as exc:
                log.warning("live: re-watch failed: %s", exc)
        return

    _state["selected_uid"] = uid
    _reset_detail(preserve_figure=True)
    _state["selected_uid"] = uid  # _reset_detail clears it
    try:
        _load_metadata(uid)
        _load_primary()
        _load_images()
    except Exception as exc:
        log.exception("live: switch_to load failed")
        _live_set_banner(f"🔴 LIVE — error loading `{uid[:8]}`: `{exc}`")
        return

    summary = _detail_cache.get("summary") or {}
    _live_set_banner(
        f"🔴 **LIVE** — watching `{uid[:8]}` · "
        f"scan {summary.get('scan_id', '?')} · "
        f"{summary.get('sample_name', '?')} · "
        f"{summary.get('detectors', '?')}"
    )

    mgr = _live["manager"]
    if mgr is not None:
        run = _ensure_run()
        if run is not None:
            try:
                mgr.watch_run(uid, run)
            except Exception as exc:
                log.exception("live: watch_run failed")
                _live_set_banner(
                    f"🔴 LIVE — `{uid[:8]}` (subscribe failed: `{exc}`)"
                )


# --- Streaming callbacks (always dispatched onto the document thread) -----

def _live_on_new_run(uid: str) -> None:
    log.info("live: new run %s", uid)
    _live_switch_to(uid)


def _live_on_primary_extended(uid: str) -> None:
    if uid != _state.get("selected_uid"):
        return
    # Force a re-fetch of the primary scalars table by clearing the cache flag.
    _detail_cache["primary_loaded"] = False
    try:
        _load_primary()
    except Exception:
        log.exception("live: primary refresh failed")
        return
    # Rebuild explore plot so the line shows the latest points.
    try:
        _build_explore_plot()
    except Exception:
        log.exception("live: explore plot rebuild failed")


def _live_on_frame_extended(uid: str, field: str, n_total: int) -> None:
    if uid != _state.get("selected_uid"):
        return
    # Drop the cached dataset so the next fetch sees the new frames.
    _detail_cache["primary_dataset"] = None
    _image_cache["dataset"] = None
    if n_total <= 0:
        return
    # Debounce: bump the sequence counter so that if multiple frame events
    # arrive rapidly, only the latest one actually triggers a re-render.
    _live["frame_seq"] = _live.get("frame_seq", 0) + 1
    my_seq = _live["frame_seq"]
    # Extend slider range; auto-advance to the latest frame iff this field
    # is the one currently displayed.
    new_end = max(0, n_total - 1)
    if w_image_slider.end != new_end:
        w_image_slider.end = new_end
    if field == _image_cache.get("field"):
        # Skip render if a newer frame event has already been dispatched.
        if _live.get("frame_seq", 0) != my_seq:
            return
        # Setting .value triggers _on_image_slider → re-renders + cursor sync.
        if w_image_slider.value != new_end:
            w_image_slider.value = new_end
        else:
            # Same index, but the underlying frame changed — force re-render.
            try:
                _render_image_frame(field, new_end)
            except Exception:
                log.exception("live: frame re-render failed")
    # If the Grid tab is currently active, refresh the multiview grid too.
    if w_detail_tabs.active == 4:
        try:
            _fetch_and_build_multiview(field)
        except Exception:
            log.exception("live: multiview grid refresh failed")


def _live_on_error(stage: str, exc: Exception) -> None:
    log.warning("live %s: %s", stage, exc)


# --- Toggle handler --------------------------------------------------------

def _on_live_toggle(event) -> None:
    if event.new and not _live["active"]:
        _enter_live_mode()
    elif (not event.new) and _live["active"]:
        _exit_live_mode()


def _enter_live_mode() -> None:
    log.info("live: entering live mode")
    _live["active"] = True
    # Capture the Bokeh document on the UI thread so worker threads can
    # marshal updates back via add_next_tick_callback.  pn.state.curdoc
    # is thread-local; we cannot read it from inside streaming callbacks.
    _live["doc"] = pn.state.curdoc
    w_btn_live.name = "■ Stop Live"
    _live_set_lockout(True)
    _live_set_banner("🔴 **LIVE** — connecting to tiled stream…")

    mgr = LiveStreamManager(
        _get_cat(),
        on_new_run=_live_on_new_run,
        on_primary_extended=_live_on_primary_extended,
        on_frame_extended=_live_on_frame_extended,
        on_error=_live_on_error,
        dispatcher=_dispatch_to_doc,
    )
    _live["manager"] = mgr
    try:
        mgr.start()
    except Exception as exc:
        log.exception("live: manager.start failed")
        _live_set_banner(f"🔴 LIVE — start failed: `{exc}`")
        return

    # Auto-select most recent run as the initial live target.
    initial_uid = _live_pick_initial_uid()
    if initial_uid:
        _live_switch_to(initial_uid)
    else:
        _live_set_banner("🔴 **LIVE** — waiting for first run…")


def _exit_live_mode() -> None:
    log.info("live: exiting live mode")
    _live["active"] = False
    w_btn_live.name = "🔴 Go Live"
    mgr = _live["manager"]
    _live["manager"] = None
    if mgr is not None:
        try:
            mgr.stop()
        except Exception:
            log.exception("live: manager.stop failed")
    _live["doc"] = None
    _live_set_lockout(False)
    _live_set_banner("")


w_btn_live.param.watch(_on_live_toggle, "value")


def _on_update(_event=None):
    """Refresh search results and reload the selected scan if it has more data."""
    # 1) Re-run the current search to pick up new scans
    try:
        _do_search(_state["page"])
    except Exception as exc:
        log.warning("Update search failed: %s", exc)

    # 2) Check if the currently selected scan has grown (more data points)
    uid = _state.get("selected_uid")
    if not uid:
        return

    try:
        # Re-fetch the run node from tiled (fresh metadata)
        run = _get_cat()[uid]
        stop = run.metadata.get("stop", {})
        num_events = stop.get("num_events", {})
        if isinstance(num_events, dict):
            new_n_steps = num_events.get("primary", 0)
        else:
            new_n_steps = int(num_events) if num_events else 0

        # Compare with what we previously had
        old_summary = _detail_cache.get("summary")
        old_n_steps = 0
        if old_summary:
            try:
                old_n_steps = int(old_summary.get("n_steps", 0))
            except (ValueError, TypeError):
                old_n_steps = 0

        if new_n_steps > old_n_steps:
            log.info(
                "Update: scan %s grew from %d to %d steps — reloading",
                uid[:8], old_n_steps, new_n_steps,
            )
            # Invalidate local caches so fresh data is fetched
            _detail_cache.update(
                run=run, summary=None,
                primary_loaded=False, baseline_loaded=False,
                images_loaded=False,
                primary_info=None, primary_dataset=None,
            )
            # Invalidate the disk cache for scalars so they are re-fetched
            from smi_browser.cache import ScanCache
            cache = ScanCache(uid)
            if cache.exists():
                import h5py
                try:
                    with cache._lock:
                        with h5py.File(cache.path, "a") as f:
                            if "primary" in f:
                                del f["primary"]
                            if "images" in f:
                                del f["images"]
                except Exception:
                    pass

            # Reload metadata and active tab
            _load_metadata(uid)
            active_tab = w_detail_tabs.active
            _load_active_tab(active_tab)

            try:
                pn.state.notifications.info(
                    f"Scan updated: {old_n_steps} → {new_n_steps} points"
                )
            except Exception:
                pass
        else:
            try:
                pn.state.notifications.info("Already up to date.")
            except Exception:
                pass
    except Exception as exc:
        log.warning("Update scan check failed: %s", exc)


w_btn_update.on_click(_on_update)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _filter_summary_text() -> str:
    """Build a compact one-line summary of active filters."""
    filters = _collect_unified_filters()
    cycle = _selected_cycle_value()
    has_explicit_cycle = any(
        ftype == "exact" and key.strip().lower() in {"cycle", "start.cycle"}
        for ftype, key, _val in filters
    )
    if cycle and not has_explicit_cycle:
        filters = list(filters) + [("exact", "cycle", cycle)]
    if not filters:
        return "*No filters active*"
    parts = []
    for ftype, key, val in filters:
        if ftype == "anywhere":
            parts.append(f'"{val}"')
        elif ftype == "exact":
            parts.append(f"{key}=**{val}**")
        else:
            parts.append(f"{key}~**{val}**")
    return " · ".join(parts)


w_filter_summary = pn.pane.Markdown(_filter_summary_text(), margin=(0, 5))

# ---------------------------------------------------------------------------
# Proposal selector (powered by api.nsls2.bnl.gov)
# ---------------------------------------------------------------------------

# Cached proposal list and lookup map
_proposal_cache: list[nsls2api.ProposalInfo] = []
_proposal_map: dict[str, nsls2api.ProposalInfo] = {}  # data_session → info

w_proposal_cycle = pn.widgets.Select(
    name="Cycle", options=["(loading…)"], value="(loading…)", width=110,
)
w_proposal_all = pn.widgets.Checkbox(
    name="All proposals", value=False, width=120,
    margin=(25, 0, 0, 5),
)
w_proposal_select = pn.widgets.Select(
    name="Data session", options=["(select cycle first)"],
    value="(select cycle first)", width=320,
)
w_proposal_project = pn.widgets.Select(
    name="Project", options=["(all)"], value="(all)", width=320,
)
w_proposal_status = pn.pane.Markdown("", width=320, margin=(0, 5))
w_proposal_spinner = pn.indicators.LoadingSpinner(
    value=False, size=18, visible=False,
)

proposal_card = pn.Card(
    pn.Row(w_proposal_cycle, w_proposal_all, w_proposal_spinner),
    w_proposal_select,
    w_proposal_project,
    w_proposal_status,
    title="📋 Proposals",
    collapsed=False,
    sizing_mode="stretch_width",
    margin=(0, 0, 5, 0),
)


def _load_cycles():
    """Populate the cycle dropdown (fast, unauthenticated)."""
    cycles = nsls2api.fetch_cycles()
    current = nsls2api.fetch_current_cycle()
    if not cycles:
        w_proposal_cycle.options = ["(unavailable)"]
        w_proposal_cycle.value = "(unavailable)"
        return
    # Most recent first; prepend "All cycles" and "commissioning"
    cycle_opts = ["All cycles", "commissioning"] + list(reversed(cycles))
    w_proposal_cycle.options = cycle_opts
    # Restore saved cycle if available and still valid
    saved_cycle = _load_saved_cycle()
    if saved_cycle and saved_cycle in cycle_opts:
        w_proposal_cycle.value = saved_cycle
    elif current and current in cycle_opts:
        w_proposal_cycle.value = current
    else:
        w_proposal_cycle.value = cycle_opts[2] if len(cycle_opts) > 2 else cycle_opts[0]


def _refresh_proposals(cycle: str | None = None):
    """Fetch proposals for the logged-in user and populate the dropdown."""
    username = _tiled_whoami()
    if not username:
        w_proposal_select.options = ["(log in first)"]
        w_proposal_select.value = "(log in first)"
        w_proposal_status.object = "*Log in to see your proposals*"
        return

    # Disable all proposal widgets while loading to prevent race conditions
    w_proposal_cycle.disabled = True
    w_proposal_select.disabled = True
    w_proposal_project.disabled = True
    w_proposal_spinner.value = True
    w_proposal_spinner.visible = True
    w_proposal_status.object = "*Loading proposals…*"

    try:
        cycle_filter = cycle if cycle and cycle != "All cycles" else None
        if w_proposal_all.value and cycle_filter:
            # Fetch ALL beamline proposals for this cycle (beamline-scientist mode)
            proposals = nsls2api.build_cycle_proposal_list(cycle_filter)
        else:
            proposals = nsls2api.build_proposal_list(username, cycle=cycle_filter)
    except Exception as exc:
        log.warning("Proposal fetch failed: %s", exc)
        w_proposal_select.options = ["(error loading)"]
        w_proposal_select.value = "(error loading)"
        w_proposal_status.object = f"*Error: {exc}*"
        return
    finally:
        w_proposal_spinner.value = False
        w_proposal_spinner.visible = False
        w_proposal_cycle.disabled = False
        w_proposal_select.disabled = False
        w_proposal_project.disabled = False

    global _proposal_cache, _proposal_map
    _proposal_cache = proposals
    _proposal_map = {p.data_session: p for p in proposals}

    if not proposals:
        # User may have beamline-wide access — show that info
        beamline_access = nsls2api.fetch_user_beamline_access(username)
        has_smi = any(b.lower() == "smi" for b in beamline_access)
        if has_smi:
            w_proposal_select.options = ["(all — beamline access)"]
            w_proposal_select.value = "(all — beamline access)"
            w_proposal_status.object = (
                f"*{username} has full SMI access — "
                "use Search Filters below to find scans*"
            )
        else:
            w_proposal_select.options = ["(none found)"]
            w_proposal_select.value = "(none found)"
            cy = cycle_filter or "all cycles"
            w_proposal_status.object = f"*No SMI proposals for {cy}*"
        return

    # Build dropdown options: data_session as value, display_label as text
    # "(All)" at the top means no data_session filter — show recent scans
    opts = {"(All — no filter)": "(All)"}
    opts.update({p.display_label: p.data_session for p in proposals})
    w_proposal_select.options = opts
    # During init, always default to (All) so initial view is unfiltered.
    # After init, restore saved data session if available.
    if _initializing:
        w_proposal_select.value = "(All)"
    else:
        saved_ds = _load_saved_datasession()
        if saved_ds and saved_ds in opts.values():
            w_proposal_select.value = saved_ds
        else:
            w_proposal_select.value = "(All)"
    w_proposal_status.object = f"*{len(proposals)} proposal{'s' if len(proposals) != 1 else ''}*"


def _on_cycle_change(*_events):
    """Re-fetch proposals when the cycle selection changes."""
    cycle = w_proposal_cycle.value
    if cycle in ("(loading…)", "(unavailable)"):
        return

    # During init, just populate dropdowns without triggering search
    if _initializing:
        _refresh_proposals(cycle)
        return

    # Reset detail pane and clear table to avoid stale state
    _state["selected_uid"] = None
    w_table.selection = []
    _reset_detail()

    # Set proposal select to a safe loading sentinel BEFORE refreshing
    # options.  This prevents a BokehJS race where the old value (e.g.
    # "pass-321164") arrives as a change event after options have been
    # replaced, causing "not in list" ValueError inside Panel.
    w_proposal_select.options = ["(loading…)"]
    w_proposal_select.value = "(loading…)"

    if cycle == "All cycles":
        # Clear any data_session filter and search unfiltered
        _filter_rows.clear()
        _add_filter()
        w_table.value = _EMPTY_DF.copy()
        w_status.object = "*Select a proposal or add filters*"
        _refresh_pagination()
        w_filter_summary.object = _filter_summary_text()

    _refresh_proposals(cycle)
    _save_proposal_state()


def _on_proposal_select(*_events):
    """Apply the selected data-session as a search filter and search."""
    if _initializing:
        return
    ds = w_proposal_select.value
    if not ds or ds.startswith("(") and ds != "(All)":
        return

    # Disable cycle/project while downstream queries run
    w_proposal_cycle.disabled = True
    w_proposal_project.disabled = True

    # Reset project dropdown while we load new options
    w_proposal_project.options = ["(all)"]
    w_proposal_project.value = "(all)"

    # Reset detail pane and clear table selection BEFORE search to avoid
    # Bokeh document-lock errors from the _on_row_select cascade.
    _state["selected_uid"] = None
    w_table.selection = []
    _reset_detail()

    # Clear existing filters
    _filter_rows.clear()
    if ds == "(All)":
        # No data_session filter — show all recent scans
        _add_filter()
        w_proposal_status.object = "*Showing all recent scans*"
    else:
        # Set a single exact data_session filter
        _add_filter(ftype="Exact", key="data_session", val=ds)
        # Show proposal info in the status line
        info = _proposal_map.get(ds)
        if info:
            w_proposal_status.object = (
                f"**{info.pi_name}** — {info.title[:80]}"
            )
    _do_search(0)

    # Re-enable cycle now that search is done
    w_proposal_cycle.disabled = False

    # Fetch distinct project_name values within this data_session
    if ds != "(All)":
        _populate_project_names(ds)
    _save_proposal_state()


def _populate_project_names(data_session: str):
    """Query tiled for distinct project_name values within a data_session."""
    w_proposal_project.disabled = True
    w_proposal_spinner.value = True
    w_proposal_spinner.visible = True
    try:
        ds_filter = _with_cycle_filter([("exact", "data_session", data_session)])
        vals = tb.distinct_values(
            _get_cat(),
            key="project_name",
            unified_filters=ds_filter,
            counts=True,
            size_limit=0,  # skip size check — already filtered to one proposal
        )
    except Exception as exc:
        log.warning("project_name distinct query failed: %s", exc)
        w_proposal_project.options = ["(all)"]
        w_proposal_project.value = "(all)"
        w_proposal_spinner.value = False
        w_proposal_spinner.visible = False
        w_proposal_project.disabled = False
        return

    if vals is None:
        w_proposal_project.options = ["(all)"]
        w_proposal_project.value = "(all)"
        w_proposal_spinner.value = False
        w_proposal_spinner.visible = False
        w_proposal_project.disabled = False
        return

    # Build options: "(all)" plus each project name with count
    project_names = sorted(
        [v["value"] for v in vals if v.get("value")],
        key=str.lower,
    )
    if not project_names:
        w_proposal_project.options = ["(all)"]
        w_proposal_project.value = "(all)"
        w_proposal_spinner.value = False
        w_proposal_spinner.visible = False
        w_proposal_project.disabled = False
        return

    # Add count info to display
    count_map = {v["value"]: v.get("count") for v in vals if v.get("value")}
    opts = {"(all)": "(all)"}
    for name in project_names:
        cnt = count_map.get(name)
        label = f"{name} ({cnt})" if cnt else name
        opts[label] = name
    w_proposal_project.options = opts
    # Restore saved project if available and still a valid option
    saved_project = _load_saved_project()
    if saved_project and saved_project in opts.values():
        w_proposal_project.value = saved_project
    else:
        w_proposal_project.value = "(all)"
    w_proposal_spinner.value = False
    w_proposal_spinner.visible = False
    w_proposal_project.disabled = False


def _on_project_select(*_events):
    """Apply project_name filter on top of the data-session filter."""
    if _initializing:
        return
    project = w_proposal_project.value
    ds = w_proposal_select.value
    if not ds or ds.startswith("("):
        return

    # Reset detail pane and clear table selection BEFORE search to avoid
    # Bokeh document-lock errors from the _on_row_select cascade.
    _state["selected_uid"] = None
    w_table.selection = []
    _reset_detail()

    # Rebuild filters: always include data_session, optionally project_name
    _filter_rows.clear()
    _add_filter(ftype="Exact", key="data_session", val=ds)
    if project and project != "(all)":
        _add_filter(ftype="Exact", key="project_name", val=project)
    _do_search(0)
    _save_proposal_state()


w_proposal_cycle.param.watch(_on_cycle_change, "value")
w_proposal_select.param.watch(_on_proposal_select, "value")
w_proposal_project.param.watch(_on_project_select, "value")
w_proposal_all.param.watch(lambda *_: _refresh_proposals(w_proposal_cycle.value), "value")

search_card = pn.Card(
    w_filter_column,
    pn.Row(w_btn_add_filter,
           pn.Column(pn.Spacer(height=0), w_btn_search),
           pn.Column(pn.Spacer(height=0), w_btn_reset)),
    title="🔍 Search Filters",
    collapsed=False,
    sizing_mode="stretch_width",
    margin=(0, 0, 5, 0),
)

page_row = pn.Row(
    w_btn_first, w_btn_prev, w_page_info, w_btn_next, w_btn_last,
)

left_panel = pn.Column(
    proposal_card,
    w_filter_summary,
    search_card,
    pn.Row(w_status, w_search_spinner),
    page_row,
    w_table,
    width=350,
    sizing_mode="stretch_height",
    scroll=True,
    stylesheets=[
        ":host { font-size: 9px; }",
    ],
)

w_detail_tabs = pn.Tabs(
    (
        "Metadata",
        pn.Column(w_meta_json, sizing_mode="stretch_both"),
    ),
    (
        "Primary",
        pn.Column(
            pn.Row(w_primary_status, w_primary_spinner),
            w_primary_table,
            pn.Row(w_primary_x, w_primary_y),
            pn.Row(w_primary_sort_btn, w_primary_fit, w_primary_fit_btn),
            w_primary_plot,
            w_primary_fit_result,
        ),
    ),
    (
        "Baseline / Config",
        pn.Column(w_baseline_status, w_baseline_table, sizing_mode="stretch_both"),
    ),
    (
        "Explore",
        pn.Column(
            pn.Row(w_image_status, w_image_spinner),
            pn.Row(w_image_field, w_image_slider, sizing_mode="stretch_width"),
            pn.Row(w_explore_x, w_explore_y, sizing_mode="stretch_width"),
            # Line plot stays compact at the top; auto-hidden when empty.
            w_explore_plot_container,
            # Plotting tools card — color scale, mask overlay, alignment.
            pn.Card(
                pn.Card(
                    pn.Row(w_cs_cmap, w_cs_log, w_cs_lock),
                    w_cs_range,
                    pn.Row(w_cs_min, w_cs_max),
                    w_cs_hist,
                    title="🎨 Color scale",
                    collapsed=False, sizing_mode="stretch_width",
                ),
                pn.Card(
                    pn.Row(w_mask_show, w_mask_dynamic, w_mask_edit,
                           w_btn_mask_reload),
                    pn.Row(w_mask_path, w_btn_mask_save, w_btn_mask_use),
                    w_mask_status,
                    title="🛡 Mask overlay",
                    collapsed=True, sizing_mode="stretch_width",
                ),
                pn.Card(
                    pn.Row(w_align_enable, w_align_width, w_btn_align_clear),
                    w_align_stats,
                    w_align_profile,
                    title="📐 Alignment / line profile",
                    collapsed=True, sizing_mode="stretch_width",
                ),
                title="🛠 Plotting tools",
                collapsed=False, sizing_mode="stretch_width",
            ),
            # Image takes the full remaining width / height below
            pn.Column(w_image_thumb, sizing_mode="stretch_both", min_height=500),
            sizing_mode="stretch_both",
        ),
    ),
    (
        "Grid",
        pn.Column(
            pn.Row(w_mv_status, w_mv_spinner),
            pn.Row(w_mv_field, w_mv_cmap, w_mv_log, w_mv_label, sizing_mode="stretch_width"),
            pn.Row(w_mv_range, w_mv_prev, w_mv_next, w_mv_page_status,
                   sizing_mode="stretch_width"),
            pn.Column(w_mv_grid, sizing_mode="stretch_both", min_height=500),
            sizing_mode="stretch_both",
        ),
    ),
    (
        "Process",
        pn.Column(
            # Quick controls — geometry selector + run button
            pn.Row(w_proc_geometry, w_btn_process, w_btn_add_collection,
                   w_proc_spinner),
            w_proc_status,
            # Advanced parameters grouped in a sub-tabs to save vertical space
            pn.Tabs(
                (
                    "Results",
                    pn.Column(
                        pn.Row(w_proc_iq_mode, w_proc_iq_label),
                        w_proc_frame_slider,
                        w_proc_2d_plot,
                        # Cross sections — interactive overlay on the 2D plot above.
                        pn.Card(
                            pn.pane.Markdown(
                                "*Click **+ Horizontal cut** or **+ Vertical cut** to "
                                "drop a dashed slice rectangle on the 2D plot above. "
                                "Drag it to move; drag a corner/edge to resize the "
                                "slice width. Hold shift+drag on empty space to draw "
                                "a new box; click a box and press Backspace to delete. "
                                "Cuts persist across scans \u2014 they are re-applied to "
                                "every newly processed result.*",
                            ),
                            pn.Row(w_btn_add_hcut, w_btn_add_vcut, w_btn_clear_cuts,
                                   w_cuts_log_x, w_cuts_log_y, w_plot_style),
                            w_cuts_table,
                            w_proc_cuts_plot,
                            title="\u2702 Cross sections",
                            collapsed=False, sizing_mode="stretch_width",
                        ),
                        w_proc_iq_plot,
                        sizing_mode="stretch_width",
                    ),
                ),
                (
                    "Parameters",
                    pn.Column(
                        w_card_grid,
                        w_card_masks,
                        w_card_saxs_qrange,
                        w_card_geometry,
                        w_card_dezinger,
                        w_card_intensity,
                        w_card_gi,
                        w_card_backend,
                        w_card_dynamic_mask,
                        w_card_waxs_cal,
                        w_card_waxs_mask_adv,
                        sizing_mode="stretch_width",
                    ),
                ),
                (
                    "Cache",
                    pn.Column(
                        w_cache_enabled,
                        pn.Row(w_btn_cache_refresh, w_btn_cache_clear),
                        w_cache_info,
                        pn.pane.Markdown(
                            "*Geometry cache stores pre-computed integrator "
                            "objects (poni files, solid-angle corrections, etc.) "
                            "to speed up repeated reductions. Clear it if you "
                            "change calibration parameters or to free memory.*",
                        ),
                        sizing_mode="stretch_width",
                    ),
                ),
                ("Batch", batch_panel),
                sizing_mode="stretch_width",
            ),
        ),
    ),
    (
        "Export",
        export_tabs,
    ),
)

w_detail_tabs.param.watch(_on_detail_tab, "active")

# Toggle button to collapse / expand the left search panel
w_btn_toggle_sidebar = pn.widgets.Toggle(
    name="◀", value=False, button_type="light", width=32,
    stylesheets=[":host { font-size: 14px; padding: 0; }"],
)

def _on_toggle_sidebar(event):
    hidden = event.new
    left_panel.visible = not hidden
    w_btn_toggle_sidebar.name = "▶" if hidden else "◀"

w_btn_toggle_sidebar.param.watch(_on_toggle_sidebar, "value")

w_btn_update_scan = pn.widgets.Button(
    name="🔄", button_type="light", width=32,
    stylesheets=[":host { font-size: 14px; padding: 0; }"],
)


def _on_update_scan(_event=None):
    """Re-check the selected scan for new data points and reload if grown."""
    uid = _state.get("selected_uid")
    if not uid:
        return
    try:
        run = _get_cat()[uid]
        stop = run.metadata.get("stop", {})
        num_events = stop.get("num_events", {})
        if isinstance(num_events, dict):
            new_n_steps = num_events.get("primary", 0)
        else:
            new_n_steps = int(num_events) if num_events else 0

        old_summary = _detail_cache.get("summary")
        old_n_steps = 0
        if old_summary:
            try:
                old_n_steps = int(old_summary.get("n_steps", 0))
            except (ValueError, TypeError):
                old_n_steps = 0

        if new_n_steps > old_n_steps:
            log.info("Update scan: %s grew %d → %d", uid[:8], old_n_steps, new_n_steps)
            _detail_cache.update(
                run=run, summary=None,
                primary_loaded=False, baseline_loaded=False,
                images_loaded=False,
                primary_info=None, primary_dataset=None,
            )
            # Invalidate disk cache for this scan
            from smi_browser.cache import ScanCache
            cache = ScanCache(uid)
            if cache.exists():
                import h5py
                try:
                    with cache._lock:
                        with h5py.File(cache.path, "a") as f:
                            if "primary" in f:
                                del f["primary"]
                            if "images" in f:
                                del f["images"]
                except Exception:
                    pass

            _load_metadata(uid)
            active_tab = w_detail_tabs.active
            _load_active_tab(active_tab)
            try:
                pn.state.notifications.info(
                    f"Scan updated: {old_n_steps} → {new_n_steps} points"
                )
            except Exception:
                pass
        else:
            try:
                pn.state.notifications.info("Scan up to date.")
            except Exception:
                pass
    except Exception as exc:
        log.warning("Update scan failed: %s", exc)
        try:
            pn.state.notifications.warning(f"Update failed: {exc}")
        except Exception:
            pass


w_btn_update_scan.on_click(_on_update_scan)

detail_panel = pn.Column(
    pn.Row(w_btn_toggle_sidebar, w_btn_update_scan, w_detail_title, sizing_mode="stretch_width"),
    w_detail_tabs,
    sizing_mode="stretch_both",
    min_width=600,
)

# ---------------------------------------------------------------------------
# Collection Export — widgets + panel
# ---------------------------------------------------------------------------

from smi_browser.export import export_collection, parse_name_parts

w_coll_export_label_src = pn.widgets.Select(
    name="Scan label source",
    options=["(auto — sample name distinct parts)"],
    value="(auto — sample name distinct parts)",
    width=260,
)
w_coll_export_basename = pn.widgets.TextInput(
    name="Output basename",
    value="",
    placeholder="(auto from common sample parts)",
    width=260,
)
w_coll_export_name_info = pn.pane.Markdown(
    "*Add scans to the collection to see name analysis.*",
    stylesheets=[":host { font-size: 11px; color: #555; }"],
    sizing_mode="stretch_width",
)
w_coll_export_dir = pn.widgets.TextInput(
    name="Output directory (relative to proposal)",
    value="projects/{project_name}/analysis",
    width=400,
)
w_coll_export_resolved = pn.pane.Markdown("", sizing_mode="stretch_width")

w_coll_export_h5 = pn.widgets.Checkbox(name="HDF5 combined", value=True)
w_coll_export_csv = pn.widgets.Checkbox(name="CSV multi-I(q)", value=True)
w_coll_export_png = pn.widgets.Checkbox(name="PNG I(q) overlay", value=True)

w_btn_coll_export = pn.widgets.Button(
    name="Export Collection", button_type="success", width=160,
)
w_coll_export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
w_coll_export_spinner = pn.indicators.LoadingSpinner(
    value=False, size=20, visible=False,
)

coll_export_panel = pn.Column(
    pn.pane.Markdown(
        "**Export entire collection** as combined files — "
        "all scans merged into single HDF5, CSV, and plots "
        "with per-scan labels."
    ),
    pn.Row(w_coll_export_label_src, w_coll_export_basename),
    w_coll_export_name_info,
    pn.layout.Divider(),
    pn.Row(w_coll_export_dir, sizing_mode="stretch_width"),
    w_coll_export_resolved,
    pn.layout.Divider(),
    pn.pane.Markdown("**Output formats:**"),
    pn.Row(w_coll_export_h5, w_coll_export_csv, w_coll_export_png),
    pn.layout.Divider(),
    pn.Row(w_btn_coll_export, w_coll_export_status, w_coll_export_spinner),
    sizing_mode="stretch_width",
)


def _refresh_coll_export_labels():
    """Update label source options and name analysis from collection."""
    # Build label source options
    opts = ["(auto — sample name distinct parts)"]
    # Add start metadata fields that vary
    varying = _collection.varying_parameters()
    for key in sorted(varying.keys()):
        opts.append(f"start:{key}")
    # Add available label columns (primary + baseline)
    for col in _collection.available_label_columns():
        opts.append(col)
    w_coll_export_label_src.options = opts
    if w_coll_export_label_src.value not in opts:
        w_coll_export_label_src.value = opts[0]

    # Name analysis — use resolved sample names (templates filled from streams)
    names = [
        _collection.resolved_sample_name(uid)
        for uid in _collection.uids
    ]
    if len(names) >= 2:
        parts = parse_name_parts(names)
        info_lines = []
        if parts["suggested_basename"]:
            info_lines.append(
                f"**Common:** `{parts['suggested_basename']}`"
            )
        if parts["distinct_strings"]:
            distinct_preview = ", ".join(
                f"`{d}`" for d in parts["distinct_strings"][:6]
            )
            if len(parts["distinct_strings"]) > 6:
                distinct_preview += f" … +{len(parts['distinct_strings']) - 6}"
            info_lines.append(f"**Distinct:** {distinct_preview}")
        w_coll_export_name_info.object = "  \n".join(info_lines) if info_lines else ""
        # Auto-fill basename if empty
        if not w_coll_export_basename.value and parts["suggested_basename"]:
            w_coll_export_basename.value = parts["suggested_basename"]
    elif len(names) == 1:
        w_coll_export_name_info.object = f"*Single scan: `{names[0]}`*"
    else:
        w_coll_export_name_info.object = (
            "*Add scans to the collection to see name analysis.*"
        )


def _resolve_coll_export_dir() -> Path | None:
    """Resolve the collection export output directory."""
    rel = w_coll_export_dir.value.strip() or "projects/{project_name}/analysis"
    proj = w_proposal_project.value
    cycle = _selected_cycle_value()
    # Try proposal dropdown
    ds = w_proposal_select.value
    if ds and not ds.startswith("("):
        return resolve_output_dir(ds, proj, cycle=cycle, relative_path=rel)
    # Fallback: use first scan's data_session
    if _collection.uids:
        first_uid = _collection.uids[0]
        md = _collection.get_raw_metadata(first_uid)
        if md:
            scan_ds = md.get("start", {}).get("data_session")
            scan_cycle = md.get("start", {}).get("cycle")
            if scan_ds:
                return resolve_output_dir(
                    scan_ds, proj, cycle=scan_cycle, relative_path=rel,
                )
    return None


def _refresh_coll_export_resolved():
    """Show resolved path in collection export panel."""
    resolved = _resolve_coll_export_dir()
    if resolved:
        w_coll_export_resolved.object = f"*Resolved: `{resolved}`*"
    else:
        w_coll_export_resolved.object = (
            "*Cannot resolve — select a proposal or add scans.*"
        )


def _get_coll_scan_labels() -> list[str]:
    """Build per-scan labels based on the chosen label source."""
    src = w_coll_export_label_src.value
    labels = []
    if src == "(auto — sample name distinct parts)":
        # Use distinct parts from resolved name parsing
        names = [_collection.resolved_sample_name(u) for u in _collection.uids]
        parts = parse_name_parts(names)
        for idx, uid in enumerate(_collection.uids):
            if idx < len(parts["distinct_strings"]) and parts["distinct_strings"][idx]:
                labels.append(parts["distinct_strings"][idx])
            else:
                labels.append(uid[:8])
    else:
        for uid in _collection.uids:
            if src.startswith("start:"):
                field = src[len("start:"):]
                md = _collection.get_raw_metadata(uid)
                if md:
                    val = md.get("start", {}).get(field, "?")
                    labels.append(str(val))
                else:
                    labels.append(uid[:8])
            elif src.startswith("baseline:"):
                labels.append(_collection.get_label_value(uid, src))
            else:
                # Primary column
                labels.append(_collection.get_label_value(uid, src))
    return labels


def _coll_export_formats() -> set[str]:
    """Gather selected collection export formats."""
    fmts: set[str] = set()
    if w_coll_export_h5.value:
        fmts.add("h5")
    if w_coll_export_csv.value:
        fmts.add("csv_iq")
    if w_coll_export_png.value:
        fmts.add("png_iq")
    return fmts


def _on_coll_export(event):
    """Handle the Export Collection button click."""
    if not _collection.uids:
        w_coll_export_status.object = "⚠️ *No scans in collection.*"
        return

    out_dir = _resolve_coll_export_dir()
    if out_dir is None:
        w_coll_export_status.object = "⚠️ *Cannot resolve output directory.*"
        return

    w_coll_export_spinner.value = True
    w_coll_export_spinner.visible = True
    w_coll_export_status.object = ""

    try:
        # Gather results
        results_list = [
            (uid, _collection.get_result(uid))
            for uid in _collection.uids
        ]
        scan_labels = _get_coll_scan_labels()
        basename = (
            w_coll_export_basename.value.strip()
            or parse_name_parts([
                _collection._metadata.get(uid, {}).get("sample_name", "?")
                for uid in _collection.uids
            ])["suggested_basename"]
            or "collection"
        )
        params_list = [
            _collection._processing.get(uid, {})
            for uid in _collection.uids
        ]
        metadata_list = [
            _collection.get_raw_metadata(uid)
            for uid in _collection.uids
        ]

        _, files = export_collection(
            out_dir=out_dir,
            results=results_list,
            scan_labels=scan_labels,
            basename=basename,
            formats=_coll_export_formats(),
            params_list=params_list,
            metadata_list=metadata_list,
            primary_dfs=[
                _collection.get_primary_df(uid) for uid in _collection.uids
            ],
            baseline_dfs=[
                _collection.get_baseline_df(uid) for uid in _collection.uids
            ],
        )
        w_coll_export_status.object = (
            f"✅ Wrote **{len(files)}** files to `{out_dir}`"
        )
    except Exception as exc:
        log.exception("Collection export failed")
        w_coll_export_status.object = f"❌ *Error: {exc}*"
    finally:
        w_coll_export_spinner.value = False
        w_coll_export_spinner.visible = False


w_btn_coll_export.on_click(_on_coll_export)

# Refresh collection export info whenever collection changes
_orig_coll_refresh = _coll_ns.refresh


def _coll_refresh_with_export():
    """Refresh collection UI + export panel info."""
    _orig_coll_refresh()
    _refresh_coll_export_labels()
    _refresh_coll_export_resolved()


_coll_ns.refresh = _coll_refresh_with_export

# ---------------------------------------------------------------------------

collection_card = pn.Card(
    pn.Tabs(
        (
            "Compare",
            pn.Row(
                pn.Column(
                    pn.Row(w_btn_coll_remove, w_coll_label),
                    w_coll_table,
                    sizing_mode="stretch_width",
                    min_width=300,
                    max_width=500,
                ),
                pn.Column(
                    w_coll_compare_plot,
                    sizing_mode="stretch_width",
                    min_width=400,
                    height=500,
                ),
                sizing_mode="stretch_width",
            ),
        ),
        (
            "Export Collection",
            coll_export_panel,
        ),
        sizing_mode="stretch_width",
    ),
    title="📊 Processed Collection",
    collapsed=True,
    sizing_mode="stretch_width",
    margin=(0, 0, 10, 0),
)


def _open_collection_panel(_event=None):
    """Pop the Scan Collection card open (called after add / compare)."""
    collection_card.collapsed = False

browse_row = pn.Row(
    left_panel,
    pn.Spacer(width=10),
    detail_panel,
    sizing_mode="stretch_width",
)

dashboard = pn.Column(
    pn.Row(
        pn.pane.Markdown("# SMI Tiled Browser — NSLS-II"),
        pn.layout.HSpacer(),
        w_login_status,
        w_btn_login,
        w_btn_logout,
        w_btn_update,
        w_btn_live,
        sizing_mode="stretch_width",
    ),
    w_login_form,
    w_live_banner,
    # Collection card sits ABOVE the browse area so it can never overlap
    # tall scrolling content (e.g. a full-resolution detector image).
    pn.Row(w_coll_summary, sizing_mode="stretch_width"),
    collection_card,
    browse_row,
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

dashboard.servable(title="SMI Browser")


_refresh_login_status()
_refresh_pagination()
_reset_detail()
_refresh_collection()

# Populate cycle dropdown and load proposals if already logged in
_load_cycles()
_refresh_proposals(w_proposal_cycle.value)

# Load scans in the background so the UI renders immediately.
# If filters were persisted from a previous session/reconnect, re-apply them.
def _startup_search():
    try:
        # Fast path: use cached count as count hint
        count_hint = tb.load_cached_count()

        cat = _get_cat()

        # Startup search honors selected cycle when one is active.
        unified = _with_cycle_filter([])

        if count_hint is None and not unified:
            # First run or cache missing — use len(cat) which is faster
            # than the REST probe (10-34s vs 30-40s for REST count).
            try:
                t0 = time.time()
                count_hint = len(cat)
                log.info("startup: len(cat) = %d in %.1fs", count_hint, time.time() - t0)
            except Exception:
                count_hint = None  # fall through to slow path

        if count_hint and not unified:
            log.info("startup: using count_hint=%d", count_hint)
        offset = _state["page"] * _state["page_size"]
        limit = _state["page_size"]

        summaries, total = tb.fetch_page_fast(
            cat, unified_filters=unified or None, offset=offset, limit=limit,
            count_hint=count_hint if not unified else None,
        )
        _state["total"] = total

        # Cache the real count for next startup (only when unfiltered)
        if total > 0 and not unified:
            tb.save_cached_count(total)

        if not summaries:
            w_status.object = "**0 scans** — add filters and press Search"
            w_search_spinner.value = False
            w_search_spinner.visible = False
            return

        df = pd.DataFrame(summaries)
        for col in RESULT_COLS:
            if col not in df.columns:
                df[col] = "?"
        df = df[RESULT_COLS].fillna("?")

        w_table.value = df
        n_pg = _n_pages()
        w_status.object = (
            f"**{total} scan{'s' if total != 1 else ''}** — "
            f"page {_state['page'] + 1}/{n_pg}"
        )
        _refresh_pagination()
        w_filter_summary.object = _filter_summary_text()
        search_card.collapsed = True
    except Exception as exc:
        log.warning("startup search failed: %s", exc)
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg or "permissions" in msg.lower():
            w_status.object = "🔒 **Login required** — use the Login button above"
        else:
            w_status.object = "*Ready — add filters and press Search*"
    finally:
        w_search_spinner.value = False
        w_search_spinner.visible = False

_initializing = False  # Allow dropdown callbacks to trigger searches now

_startup_thread = threading.Thread(target=_startup_search, daemon=True)
_startup_thread.start()

if __name__ == "__main__":
    dashboard.show(title="SMI Browser")
