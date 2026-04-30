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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

pn.extension("tabulator", sizing_mode="stretch_width", notifications=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAGE_SIZE = 25

# Canonical defaults & helpers are owned by PyHyperScattering.smi_defaults.
# Importing it triggers PyHyperScattering/__init__.py once (which pulls in
# the heavy integrators), but we need LOADER_DEFAULTS at widget-construct
# time anyway, so eat the cost here rather than mirroring constants.
from PyHyperScattering import smi_defaults as smid

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
DEFAULT_INCIDENT_ANGLE = 0.0
DEFAULT_THETA_OFFSET = -0.5
DEFAULT_N_QXY = 500
DEFAULT_N_QZ = 500

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


# ---------------------------------------------------------------------------
# ScanCollection — holds processed results for comparison
# ---------------------------------------------------------------------------

class ScanCollection:
    """
    Manages a set of processed CombinedReductionResults for comparison
    and parameter-sweep analysis.

    Within-scan variation (e.g. waxs_arc angle per frame) is already
    handled by reduce_smi_combined.  Between-scan variation (sample,
    temperature, etc.) is tracked here.

    Usage
    -----
        coll = ScanCollection()
        coll.add(result, metadata, params)
        coll.varying_parameters()      # which metadata fields differ
        coll.iq_comparison_figure()    # matplotlib overlay
        ds = coll.stack_iq("sample")   # xr.Dataset along a new dim
    """

    def __init__(self):
        self._results: dict[str, Any] = {}           # uid -> CombinedReductionResult
        self._metadata: dict[str, dict] = {}         # uid -> enhanced_summary dict
        self._processing: dict[str, dict] = {}       # uid -> processing kwargs

    @property
    def uids(self) -> list[str]:
        return list(self._results.keys())

    def __len__(self):
        return len(self._results)

    def __contains__(self, uid: str):
        return uid in self._results

    def add(self, result, metadata: dict, params: dict | None = None):
        """Add a processed scan to the collection."""
        self._results[result.uid] = result
        self._metadata[result.uid] = metadata
        if params:
            self._processing[result.uid] = params

    def remove(self, uid: str):
        self._results.pop(uid, None)
        self._metadata.pop(uid, None)
        self._processing.pop(uid, None)

    def get_result(self, uid: str):
        return self._results.get(uid)

    def summary_table(self) -> pd.DataFrame:
        """DataFrame summary of all scans in the collection."""
        rows = []
        for uid in self._results:
            res = self._results[uid]
            meta = self._metadata.get(uid, {})
            timing = res.timing or {}
            rows.append({
                "uid_short": uid[:8],
                "sample":    meta.get("sample_name", "?"),
                "plan":      meta.get("plan_name", "?"),
                "geometry":  res.geometry,
                "total_s":   f"{sum(timing.values()):.1f}" if timing else "?",
                "uid":       uid,
            })
        cols = ["uid_short", "sample", "plan", "geometry", "total_s", "uid"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def varying_parameters(self) -> dict[str, list]:
        """
        Identify metadata fields that differ across scans in the collection.
        Skips fields that are inherently unique per-scan (uid, scan_id, time).
        """
        if len(self._metadata) < 2:
            return {}
        all_metas = list(self._metadata.values())
        all_keys: set[str] = set()
        for m in all_metas:
            all_keys.update(m.keys())

        skip = {
            "uid", "scan_id", "time", "exit_status", "streams",
            "detector_list", "has_saxs", "has_waxs", "n_steps",
        }
        varying = {}
        for key in sorted(all_keys - skip):
            vals = [str(m.get(key, "")) for m in all_metas]
            if len(set(vals)) > 1:
                varying[key] = vals
        return varying

    def iq_comparison_figure(self, figsize=(9, 5)):
        """Matplotlib figure overlaying I(q) from every scan."""
        if not self._results:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        for uid, res in self._results.items():
            meta = self._metadata.get(uid, {})
            label = f"{uid[:8]} — {meta.get('sample_name', '?')}"
            iq = res.merged_iq
            q = iq["q"].values
            I = iq["I"].values
            mask = np.isfinite(I) & (I > 0)
            if mask.any():
                ax.plot(q[mask], I[mask], linewidth=0.8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("q (nm⁻¹)")
        ax.set_ylabel("I(q)")
        ax.set_title("Scan Collection — I(q) Comparison")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        return fig

    def stack_iq(self, dim_name: str = "scan", dim_values=None):
        """
        Stack merged I(q) datasets into a single xr.Dataset along a new dim.
        If dim_values is given, use those as the coordinate; otherwise use
        short UID + sample_name labels.
        """
        import xarray as xr
        if not self._results:
            return xr.Dataset()
        datasets = []
        labels = []
        for i, (uid, res) in enumerate(self._results.items()):
            if dim_values is not None and i < len(dim_values):
                labels.append(dim_values[i])
            else:
                meta = self._metadata.get(uid, {})
                labels.append(f"{uid[:8]}_{meta.get('sample_name', '?')}")
            datasets.append(res.merged_iq)
        return xr.concat(datasets, dim=pd.Index(labels, name=dim_name))


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_cat = None
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
    global _cat
    if _cat is None:
        _cat = tb.connect(DEFAULT_TILED_URI, DEFAULT_CATALOG)
    return _cat


def _n_pages() -> int:
    return max(1, (_state["total"] + _state["page_size"] - 1) // _state["page_size"])


# ---------------------------------------------------------------------------
# Search / data helpers
# ---------------------------------------------------------------------------

SEARCH_TYPES = ["Anywhere", "Text in field", "Exact"]
SEARCH_TYPE_MAP = {"Anywhere": "anywhere", "Text in field": "like", "Exact": "exact"}


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
    unified = _state["unified_filters"]
    offset = _state["page"] * _state["page_size"]
    limit = _state["page_size"]

    summaries, total = tb.fetch_page_fast(
        _get_cat(), unified_filters=unified,
        offset=offset, limit=limit,
    )
    # Update total from REST response (avoids a separate count query)
    _state["total"] = total

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


def _scalar_stream_to_frame(run, stream: str) -> pd.DataFrame:
    """Read scalar fields from a stream into a DataFrame."""
    scalar_data = tb.fetch_scalars(run, stream)
    return _scalars_to_dataframe(scalar_data)


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


def _thumbnail_figure(arr, title):
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import (
        ColorBar, ColumnDataSource, LinearColorMapper, LogColorMapper,
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
    # Vertex source for PolyEditTool (shows draggable handles in edit mode)
    vertex_source = ColumnDataSource(data=dict(x=[], y=[]))
    vertex_renderer = p.scatter(
        x="x", y="y", source=vertex_source,
        size=8, color="white", line_color="black", line_width=1,
    )
    draw_tool = PolyDrawTool(renderers=[mask_renderer], num_objects=200)
    edit_tool = PolyEditTool(renderers=[mask_renderer],
                             vertex_renderer=vertex_renderer)
    p.add_tools(draw_tool, edit_tool)

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

    # Stash on image cache so the Explore controls can read/write it
    _image_cache["mask_source"] = mask_source
    _image_cache["mask_renderer"] = mask_renderer
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

    # Update color mapper range
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

    # Update image data (keeps zoom/pan state)
    source.data = dict(image=[display], x=[0], y=[0], dw=[w], dh=[h])
    fig.title.text = title


# ---------------------------------------------------------------------------
# Widgets — Search  (unified stackable filters)
# ---------------------------------------------------------------------------

# Dynamic filter rows: each is a dict {type, key, val, suggest, remove}
_filter_rows: list[dict] = []
w_filter_column = pn.Column(sizing_mode="stretch_width")

# Cancellation flag — set by Reset to abort in-flight queries
_cancel = threading.Event()


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
                unified_filters=other_filters or None,
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

w_status = pn.pane.Markdown("*Ready*", width=700)
w_search_spinner = pn.indicators.LoadingSpinner(value=False, size=20, visible=False)
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

# Start with one empty filter row
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

w_baseline_table = pn.widgets.Tabulator(
    value=pd.DataFrame(columns=["field", "before", "after"]),
    show_index=False, sizing_mode="stretch_both",
    configuration={"layout": "fitColumns", "rowHeight": 22},
    header_filters={"field": {"type": "input", "func": "like", "placeholder": "filter…"}},
)
w_baseline_status = pn.pane.Markdown("*Click tab to load.*")

# Images tab — frame slider for browsing raw detector images
w_image_thumb = pn.pane.Bokeh(object=None, sizing_mode="stretch_both", min_height=400)
w_image_status = pn.pane.Markdown("")
w_image_spinner = pn.indicators.LoadingSpinner(value=False, size=20, visible=False)
w_image_slider = pn.widgets.IntSlider(
    name="Frame", start=0, end=0, value=0, step=1, sizing_mode="stretch_width",
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

_image_cache = {"field": None, "n_frames": 0, "dataset": None, "fields": [],
                "figure": None, "source": None, "mapper": None,
                "mask_source": None, "mask_renderer": None,
                "draw_tool": None, "edit_tool": None,
                "mask_image_shape": None}


def _render_image_frame(field, idx):
    """Fetch, orient, and render a single image frame (preserves zoom/pan)."""
    run = _ensure_run()
    if run is None:
        return
    ds = _image_cache.get("dataset")
    frame = tb.fetch_frame(run, "primary", field, frame_idx=idx, _dataset=ds)
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
            "*Edit mode on. **Draw a new polygon**: tap to place each "
            "vertex, double-tap to finish (or press Esc to cancel). "
            "**Edit existing polygons**: click the PolyEdit icon "
            "(square-with-handles) in the toolbar, tap a polygon, then "
            "drag its vertices.*"
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

    data = src.data
    field = _image_cache.get("field")
    raw_shape = _image_cache.get("raw_shape")
    if raw_shape is None:
        shape = _image_cache.get("mask_image_shape")
        raw_shape = tuple(shape) if shape else None
    out_dict = _xs_ys_to_normalized_mask(
        data["xs"], data["ys"], data["name"], data["kind"],
        field=field, raw_shape=raw_shape,
    )
    try:
        smid.save_mask_polygons(out_dict, out_path)
    except Exception as exc:
        log.exception("mask save failed")
        w_mask_status.object = f"**Save failed:** `{exc}`"
        return
    w_mask_status.object = f"*Saved {len(data['xs'])} polygon(s) → `{out_path}`.*"


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
w_proc_nq = pn.widgets.IntInput(
    name="n_q", value=DEFAULT_N_Q, start=100, end=10000, step=100, width=90,
)
w_proc_nchi = pn.widgets.IntInput(
    name="n_χ", value=DEFAULT_N_CHI, start=36, end=720, step=36, width=90,
)
w_proc_saxs_mask = pn.widgets.TextInput(
    name="SAXS mask",
    value=DEFAULT_SAXS_MASK,
    placeholder=f"(default: {DEFAULT_SAXS_MASK_NAME})",
    width=320,
)
w_proc_waxs_mask = pn.widgets.TextInput(
    name="WAXS mask",
    value=DEFAULT_WAXS_MASK,
    placeholder=f"(default: {DEFAULT_WAXS_MASK_NAME})",
    width=320,
)
w_proc_saxs_row_delta = pn.widgets.FloatInput(
    name="SAXS Δrow", value=DEFAULT_SAXS_ROW_DELTA, step=0.5, width=80,
)
w_proc_saxs_col_delta = pn.widgets.FloatInput(
    name="SAXS Δcol", value=DEFAULT_SAXS_COL_DELTA, step=0.5, width=80,
)
w_proc_waxs_row_delta = pn.widgets.FloatInput(
    name="WAXS Δrow", value=DEFAULT_WAXS_ROW_DELTA, step=0.5, width=80,
)
w_proc_waxs_col_delta = pn.widgets.FloatInput(
    name="WAXS Δcol", value=DEFAULT_WAXS_COL_DELTA, step=0.5, width=80,
)
w_proc_dist_delta = pn.widgets.FloatInput(
    name="SAXS Δdist (mm)", value=DEFAULT_SAXS_DIST_DELTA, step=1.0, width=110,
)
w_proc_dezinger = pn.widgets.FloatInput(
    name="Dezinger σ", value=DEFAULT_DEZINGER, step=100.0, width=100,
)

# GI-specific parameters (shown only when geometry == "grazing")
w_proc_nqxy = pn.widgets.IntInput(
    name="n_qxy", value=DEFAULT_N_QXY, start=100, end=2000, step=50, width=90,
)
w_proc_nqz = pn.widgets.IntInput(
    name="n_qz", value=DEFAULT_N_QZ, start=100, end=2000, step=50, width=90,
)
w_proc_incident_angle = pn.widgets.FloatInput(
    name="α_i (°)", value=DEFAULT_INCIDENT_ANGLE, step=0.01, width=90,
)
w_proc_incident_angle_auto = pn.widgets.Checkbox(
    name="Auto α_i", value=True, width=80,
)
w_proc_theta_offset = pn.widgets.FloatInput(
    name="θ offset (°)", value=DEFAULT_THETA_OFFSET, step=0.1, width=90,
)
w_gi_row = pn.Row(
    w_proc_nqxy, w_proc_nqz, w_proc_incident_angle,
    w_proc_incident_angle_auto, w_proc_theta_offset,
)
# Transmission-specific row
w_trans_row = pn.Row(w_proc_nq, w_proc_nchi)


def _on_geometry_change(event):
    """Show/hide GI vs transmission params."""
    is_gi = event.new == "grazing"
    w_gi_row.visible = is_gi
    w_trans_row.visible = not is_gi
    # SAXS not used in GI mode
    w_proc_saxs_mask.visible = not is_gi
    w_proc_saxs_row_delta.visible = not is_gi
    w_proc_saxs_col_delta.visible = not is_gi
    w_proc_dist_delta.visible = not is_gi


w_proc_geometry.param.watch(_on_geometry_change, "value")
# Initial visibility
w_gi_row.visible = False

w_btn_process = pn.widgets.Button(
    name="⚙ Process", button_type="success", width=110,
)
w_btn_add_collection = pn.widgets.Button(
    name="+ Add to Collection", button_type="primary", width=150, disabled=True,
)
w_proc_status = pn.pane.Markdown("*Select a scan and click Process.*")
w_proc_spinner = pn.indicators.LoadingSpinner(value=False, size=40, visible=False)
w_proc_iq_plot = pn.pane.Bokeh(object=None, sizing_mode="fixed", width=1000, height=400)
w_proc_2d_plot = pn.pane.Bokeh(object=None, sizing_mode="fixed", width=1000, height=1000)
w_proc_frame_slider = pn.widgets.IntSlider(
    name="Frame", start=0, end=0, value=0, step=1, width=400,
)
_proc_result_cache = {"result": None, "gi_result": None}

# ---------------------------------------------------------------------------
# Widgets — Collection
# ---------------------------------------------------------------------------

_COLL_COLS = ["uid_short", "sample", "plan", "geometry", "total_s", "uid"]

w_coll_table = pn.widgets.Tabulator(
    value=pd.DataFrame(columns=_COLL_COLS),
    show_index=False, sizing_mode="stretch_width", height=200,
    selectable="checkbox",
    configuration={"rowHeight": 22, "layout": "fitColumns"},
    hidden_columns=["uid"],
)
w_btn_coll_remove = pn.widgets.Button(
    name="Remove Selected", button_type="danger", width=130,
)
w_btn_coll_compare = pn.widgets.Button(
    name="Compare I(q)", button_type="primary", width=120,
)
w_coll_varying = pn.pane.Markdown("")
w_coll_compare_plot = pn.pane.Matplotlib(object=None, tight=True, height=400)


# ---------------------------------------------------------------------------
# Helpers — 2D result plotting
# ---------------------------------------------------------------------------

def _plot_2d_transmission(result, frame_idx=None):
    """Plot q-vs-chi as an interactive Bokeh figure."""
    from bokeh.plotting import figure as bk_figure
    from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar

    qchi = result.merged_qchi
    if frame_idx is not None and "frame" in qchi.dims:
        img = qchi["intensity"].isel(frame=frame_idx).values
        title = f"q vs χ — frame {frame_idx}"
    else:
        img = qchi["intensity"].values
        title = "q vs χ (merged)"
    q = qchi["q"].values if "q" in qchi.coords else np.arange(img.shape[1])
    chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(img.shape[0])

    display = np.where(np.isfinite(img), img, 0).astype(np.float32)
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
        title=title, width=1000, height=1000,
        x_range=(q0, q1), y_range=(c0, c1),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
    )
    p.image(image=[display], x=q0, y=c0, dw=q1 - q0, dh=c1 - c0, color_mapper=mapper)
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, width=12), "right")
    p.xaxis.axis_label = "q (nm⁻¹)"
    p.yaxis.axis_label = "χ (°)"
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
        title=title, width=1000, height=1000,
        x_range=(x0, x1), y_range=(y0, y1),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
    )
    p.image(image=[display.T], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
    p.add_layout(ColorBar(color_mapper=mapper, label_standoff=8, width=12), "right")
    p.xaxis.axis_label = "q_xy (nm⁻¹)"
    p.yaxis.axis_label = "q_z (nm⁻¹)"
    return p


def _update_proc_2d(event):
    """Redraw the 2D map when the frame slider changes."""
    gi = _proc_result_cache.get("gi_result")
    trans = _proc_result_cache.get("result")
    idx = event.new
    try:
        if gi is not None:
            w_proc_2d_plot.object = _plot_2d_gi(gi, frame_idx=idx)
        elif trans is not None:
            w_proc_2d_plot.object = _plot_2d_transmission(trans, frame_idx=idx)
    except Exception as exc:
        log.warning("2D plot update failed: %s", exc)


w_proc_frame_slider.param.watch(_update_proc_2d, "value")


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
        df = _fetch_page()  # also sets _state["total"] from REST meta
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


def _reset_detail(preserve_figure=False):
    w_detail_title.object = "### Select a scan"
    w_meta_json.object = {}
    w_primary_table.value = pd.DataFrame()
    w_primary_status.object = "*Click tab to load.*"
    # Don't clear x/y options/values — they persist across scans
    w_primary_plot.object = None
    w_baseline_table.value = pd.DataFrame(columns=["field", "before", "after"])
    w_baseline_status.object = "*Click tab to load.*"
    if not preserve_figure:
        # Full reset — destroy the Bokeh figure and all overlays
        w_image_thumb.object = None
        _image_cache.update(
            figure=None, source=None, mapper=None,
            fig_image_shape=None,
            mask_source=None, mask_renderer=None,
            draw_tool=None, edit_tool=None,
            mask_image_shape=None,
            dyn_source=None, dyn_renderer=None,
        )
    w_image_status.object = ""
    w_image_slider.value = 0
    w_image_slider.end = 0
    # Don't clear image_field options — preserve detector selection
    _image_cache.update(n_frames=0, dataset=None, fields=[],
                        raw_shape=None)
    w_explore_plot.object = None
    w_proc_status.object = "*Select a scan and click Process.*"
    w_proc_spinner.value = False
    w_proc_spinner.visible = False
    w_proc_iq_plot.object = None
    w_proc_2d_plot.object = None
    w_proc_frame_slider.value = 0
    w_proc_frame_slider.end = 0
    w_proc_frame_slider.visible = False
    _proc_result_cache.update(result=None, gi_result=None)
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

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
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
    if not run:
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


def _load_baseline():
    if _detail_cache["baseline_loaded"]:
        return
    run = _ensure_run()
    if not run:
        return
    if "baseline" not in tb.stream_names(run):
        w_baseline_status.object = "*No baseline stream.*"
        _detail_cache["baseline_loaded"] = True
        return
    t0 = time.perf_counter()
    w_baseline_status.object = "*Loading…*"
    # Single read for baseline (avoids per-field .structure() on 300+ fields)
    scalar_data = tb.fetch_scalars(run, "baseline")
    # Transpose into field/before/after rows (baseline typically has 2 readings)
    rows = []
    for key, arr in sorted(scalar_data.items()):
        arr = np.asarray(arr).flatten()
        if arr.size >= 2:
            rows.append({"field": key, "before": str(arr[0]), "after": str(arr[-1])})
        elif arr.size == 1:
            rows.append({"field": key, "before": str(arr[0]), "after": ""})
        else:
            rows.append({"field": key, "before": str(arr.tolist()), "after": ""})
    df = pd.DataFrame(rows, columns=["field", "before", "after"])
    dt_ms = (time.perf_counter() - t0) * 1000
    w_baseline_table.value = df
    w_baseline_status.object = (
        f"**{len(df)} fields** ({dt_ms:.0f} ms)"
    )
    _detail_cache["baseline_loaded"] = True


def _load_images():
    if _detail_cache["images_loaded"]:
        return
    run = _ensure_run()
    if not run:
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
        frame = tb.fetch_frame(run, "primary", field, frame_idx=0, _dataset=ds)
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
        pass  # Process tab — no auto-load


def _on_detail_tab(event):
    if not _state.get("selected_uid"):
        return
    try:
        _load_active_tab(event.new)
    except Exception as exc:
        log.exception("Detail tab load error")


def _update_primary_plot(*_events):
    """Redraw the primary scatter plot as interactive Bokeh."""
    from bokeh.plotting import figure as bk_figure

    df = w_primary_table.value
    x_col = w_primary_x.value
    y_cols = w_primary_y.value
    if df is None or df.empty or not x_col or not y_cols:
        w_primary_plot.object = None
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
    w_proc_2d_plot.object = None
    w_proc_iq_plot.object = None

    try:
        t0 = time.perf_counter()

        if geometry == "grazing":
            from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_gi

            # Build kwargs, omitting (or sending None for) any value that
            # still matches the upstream PyHyper default.
            gi_params: dict[str, Any] = dict(
                uid=uid,
                tiled_uri=DEFAULT_TILED_URI,
                catalog=DEFAULT_CATALOG,
                waxs_mask_path=w_proc_waxs_mask.value or None,
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
            if not w_proc_incident_angle_auto.value:
                gi_params["incident_angle_deg"] = w_proc_incident_angle.value

            gi_result = reduce_smi_gi(**gi_params)
            dt = time.perf_counter() - t0

            _proc_result_cache["gi_result"] = gi_result
            _last_result["result"] = gi_result
            _last_result["params"] = gi_params

            # Configure frame slider for GI result
            n_fr = len(gi_result.frames)
            w_proc_frame_slider.end = max(0, n_fr - 1)
            w_proc_frame_slider.value = 0
            w_proc_frame_slider.visible = n_fr > 1

            # Show summed qxy-vs-qz map
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
            from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined

            # Build kwargs, omitting (or sending None for) any value that
            # still matches the upstream PyHyper / loader default so the
            # loader can fill in its own calibrated values.
            params: dict[str, Any] = dict(
                uid=uid,
                tiled_uri=DEFAULT_TILED_URI,
                catalog=DEFAULT_CATALOG,
                solid_angle_correction=True,
                geometry=geometry,
                saxs_mask_path=w_proc_saxs_mask.value or None,
                waxs_mask_path=w_proc_waxs_mask.value or None,
            )
            if w_proc_nq.value != DEFAULT_N_Q:
                params["n_q"] = w_proc_nq.value
            else:
                params["n_q"] = DEFAULT_N_Q  # smi-browser keeps 2000 default
            if w_proc_nchi.value != DEFAULT_N_CHI:
                params["n_chi"] = w_proc_nchi.value

            # Beam-centre Δ — only send if the user changed at least one
            # axis from the loader's calibrated default.
            saxs_row_changed = w_proc_saxs_row_delta.value != DEFAULT_SAXS_ROW_DELTA
            saxs_col_changed = w_proc_saxs_col_delta.value != DEFAULT_SAXS_COL_DELTA
            if saxs_row_changed or saxs_col_changed:
                params["saxs_beam_delta_px"] = (
                    w_proc_saxs_row_delta.value,
                    w_proc_saxs_col_delta.value,
                )
            waxs_row_changed = w_proc_waxs_row_delta.value != DEFAULT_WAXS_ROW_DELTA
            waxs_col_changed = w_proc_waxs_col_delta.value != DEFAULT_WAXS_COL_DELTA
            if waxs_row_changed or waxs_col_changed:
                params["waxs_beam_delta_px"] = (
                    w_proc_waxs_row_delta.value,
                    w_proc_waxs_col_delta.value,
                )
            if w_proc_dist_delta.value != DEFAULT_SAXS_DIST_DELTA:
                params["saxs_distance_delta_mm"] = w_proc_dist_delta.value
            if w_proc_dezinger.value != DEFAULT_DEZINGER:
                params["dezinger_threshold"] = (
                    w_proc_dezinger.value if w_proc_dezinger.value > 0 else None
                )

            result = reduce_smi_combined(**params)
            dt = time.perf_counter() - t0

            _proc_result_cache["result"] = result
            _last_result["result"] = result
            _last_result["params"] = params

            # 2D q-chi map
            try:
                qchi = result.merged_qchi
                has_frames = "frame" in qchi.dims
                if has_frames:
                    n_fr = qchi.sizes["frame"]
                    w_proc_frame_slider.end = max(0, n_fr - 1)
                    w_proc_frame_slider.value = 0
                    w_proc_frame_slider.visible = n_fr > 1
                else:
                    w_proc_frame_slider.visible = False
                w_proc_2d_plot.object = _plot_2d_transmission(result)
            except Exception as exc:
                log.warning("2D q-chi plot failed: %s", exc)
                w_proc_frame_slider.visible = False

            # I(q) plot (Bokeh)
            from bokeh.plotting import figure as bk_figure

            iq = result.merged_iq
            q = iq["q"].values
            I = iq["I"].values

            p = bk_figure(
                title=f"{uid[:8]} — merged I(q)", width=1000, height=400,
                x_axis_type="log", y_axis_type="log",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll="wheel_zoom",
            )

            mask = np.isfinite(I) & (I > 0)
            if mask.any():
                p.line(q[mask], I[mask], line_color="black", line_width=1.2, legend_label="merged")
            if "saxs_I" in iq:
                sI = iq["saxs_I"].values
                sm = np.isfinite(sI) & (sI > 0)
                if sm.any():
                    p.line(q[sm], sI[sm], line_color="blue", line_width=0.8, alpha=0.6, legend_label="SAXS")
            if "waxs_I" in iq:
                wI = iq["waxs_I"].values
                wm = np.isfinite(wI) & (wI > 0)
                if wm.any():
                    p.line(q[wm], wI[wm], line_color="red", line_width=0.8, alpha=0.6, legend_label="WAXS")
            p.xaxis.axis_label = "q (nm⁻¹)"
            p.yaxis.axis_label = "I(q)"
            p.legend.click_policy = "hide"
            w_proc_iq_plot.object = p

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
    finally:
        w_btn_process.disabled = False
        w_proc_spinner.value = False
        w_proc_spinner.visible = False


def _on_add_to_collection(event):
    result = _last_result.get("result")
    if result is None:
        return
    summary = _detail_cache.get("summary") or {}
    params = _last_result.get("params") or {}
    _collection.add(result, summary, params)
    _refresh_collection()
    _open_collection_panel()  # pop open the floating panel
    pn.state.notifications.success(
        f"Added {result.uid[:8]} to collection ({len(_collection)} scans)"
    )


w_btn_process.on_click(_on_process)
w_btn_add_collection.on_click(_on_add_to_collection)


# ---------------------------------------------------------------------------
# Callbacks — Collection
# ---------------------------------------------------------------------------

def _coll_summary_text() -> str:
    n = len(_collection)
    if n == 0:
        return "*Empty — process scans and add them here*"
    uids = ", ".join(uid[:8] for uid in _collection.uids[:5])
    if n > 5:
        uids += f" … +{n - 5} more"
    return f"**{n} scan{'s' if n != 1 else ''}**: {uids}"


w_coll_summary = pn.pane.Markdown(_coll_summary_text(), margin=(0, 5))


def _refresh_collection():
    w_coll_table.value = _collection.summary_table()
    varying = _collection.varying_parameters()
    if varying:
        lines = ["**Varying parameters across collection:**"]
        for k, vals in varying.items():
            lines.append(f"- `{k}`: {', '.join(str(v) for v in vals)}")
        w_coll_varying.object = "\n".join(lines)
    else:
        w_coll_varying.object = ""
    w_coll_summary.object = _coll_summary_text()


def _on_coll_remove(event):
    sel = w_coll_table.selection
    if not sel:
        return
    df = w_coll_table.value
    for idx in sorted(sel, reverse=True):
        if idx < len(df):
            uid = df.iloc[idx]["uid"]
            _collection.remove(uid)
    _refresh_collection()
    w_coll_compare_plot.object = None


def _on_coll_compare(event):
    if len(_collection) == 0:
        return
    fig = _collection.iq_comparison_figure()
    w_coll_compare_plot.object = fig
    _open_collection_panel()  # ensure the floating panel is visible


w_btn_coll_remove.on_click(_on_coll_remove)
w_btn_coll_compare.on_click(_on_coll_compare)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _filter_summary_text() -> str:
    """Build a compact one-line summary of active filters."""
    filters = _collect_unified_filters()
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

search_card = pn.Card(
    w_filter_column,
    pn.Row(w_btn_add_filter,
           pn.Column(pn.Spacer(height=0), w_btn_search),
           pn.Column(pn.Spacer(height=0), w_btn_reset)),
    title="🔍 Search Filters",
    collapsed=True,
    sizing_mode="stretch_width",
    margin=(0, 0, 5, 0),
)

page_row = pn.Row(
    w_btn_first, w_btn_prev, w_page_info, w_btn_next, w_btn_last,
)

left_panel = pn.Column(
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
            w_primary_plot,
        ),
    ),
    (
        "Baseline",
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
            # Mask overlay controls — open by default so users see the
            # default mask is loaded; collapse if they don't want it.
            pn.Card(
                pn.Row(w_mask_show, w_mask_dynamic, w_mask_edit, w_btn_mask_reload),
                pn.Row(w_mask_path, w_btn_mask_save, w_btn_mask_use),
                w_mask_status,
                title="Mask overlay",
                collapsed=False, sizing_mode="stretch_width",
            ),
            # Image takes the full remaining width / height below
            pn.Column(w_image_thumb, sizing_mode="stretch_both", min_height=500),
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
            # Advanced parameter expanders — collapsed by default; widgets
            # remain at their PyHyper defaults unless the user opens an
            # expander and changes a value.  Unchanged values are sent as
            # None so the upstream loader fills in its calibrated default.
            pn.Card(
                w_trans_row,
                w_gi_row,
                title="Output grid",
                collapsed=True, sizing_mode="stretch_width",
            ),
            pn.Card(
                pn.Row(w_proc_saxs_mask, w_proc_waxs_mask),
                pn.pane.Markdown(
                    "*Leave blank to use the bundled PyHyperScattering "
                    "defaults shown in the placeholder text.  Use the "
                    "Explore tab to view or edit a mask interactively.*",
                ),
                title="Masks",
                collapsed=True, sizing_mode="stretch_width",
            ),
            pn.Card(
                pn.pane.Markdown("**Beam-centre Δ (px)**"),
                pn.Row(
                    w_proc_saxs_row_delta, w_proc_saxs_col_delta,
                    w_proc_waxs_row_delta, w_proc_waxs_col_delta,
                ),
                pn.Row(w_proc_dist_delta),
                pn.pane.Markdown(
                    "*Defaults match the SMI loader's calibrated values "
                    "(SAXS Δrow=2, Δcol=3, Δdist=−20 mm; WAXS Δrow=0, "
                    "Δcol=−2).  Change only if you know the calibration "
                    "has shifted.*",
                ),
                title="Geometry overrides",
                collapsed=True, sizing_mode="stretch_width",
            ),
            pn.Card(
                pn.Row(w_proc_dezinger),
                pn.pane.Markdown("*Set to 0 to disable hot-pixel rejection.*"),
                title="Hot-pixel rejection",
                collapsed=True, sizing_mode="stretch_width",
            ),
            # Results
            w_proc_frame_slider,
            w_proc_2d_plot,
            w_proc_iq_plot,
        ),
    ),
    dynamic=True,
)

w_detail_tabs.param.watch(_on_detail_tab, "active")

detail_panel = pn.Column(
    w_detail_title,
    w_detail_tabs,
    sizing_mode="stretch_both",
    min_width=600,
)

collection_card = pn.Card(
    pn.Row(w_btn_coll_compare, w_btn_coll_remove),
    w_coll_table,
    w_coll_varying,
    w_coll_compare_plot,
    title="📊 Scan Collection",
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
    pn.pane.Markdown(
        "# SMI Tiled Browser — NSLS-II",
    ),
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

_refresh_pagination()
_reset_detail()
_refresh_collection()

# Load the latest page of results on startup
_do_search(0)

if __name__ == "__main__":
    dashboard.show(title="SMI Browser")
