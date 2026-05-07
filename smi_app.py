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
from batch_processor import BatchProcessor
from live_stream import LiveStreamManager

# Re-export relocated modules so internal references still work
from smi_browser.models.summary import enhanced_summary  # noqa: F401
from smi_browser.models.collection import ScanCollection  # noqa: F401
from smi_browser.data.scalars import (  # noqa: F401
    scalars_to_dataframe as _scalars_to_dataframe_impl,
    scalar_stream_to_frame as _scalar_stream_to_frame_impl,
)
from smi_browser.data.frames import (  # noqa: F401
    detector_for_field as _detector_for_field_impl,
    orient_frame as _orient_frame_impl,
)
from smi_browser.data.masks import (  # noqa: F401
    normalized_mask_to_xs_ys as _normalized_mask_to_xs_ys_impl,
    xs_ys_to_normalized_mask as _xs_ys_to_normalized_mask_impl,
    default_mask_path_for as _default_mask_path_for_impl,
    classify_detector_field,
)
from smi_browser.config import (  # noqa: F401
    PAGE_SIZE,
    COMMON_SEARCH_KEYS,
    RESULT_COLS,
    EMPTY_DF as _EMPTY_DF_pkg,
    SEARCH_TYPES as SEARCH_TYPES_pkg,
    SEARCH_TYPE_MAP as SEARCH_TYPE_MAP_pkg,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

pn.extension("tabulator", sizing_mode="stretch_width", notifications=True)

# ---------------------------------------------------------------------------
# Configuration  (canonical source: smi_browser.config)
# ---------------------------------------------------------------------------

# PAGE_SIZE, COMMON_SEARCH_KEYS, RESULT_COLS imported from smi_browser.config above.

# PyHyperScattering imports still needed directly in this file.
from PyHyperScattering import smi_defaults as smid
from PyHyperScattering.SMISWAXSIntegrator import (
    clear_geometry_cache,
    geometry_cache_info,
)

DEFAULT_TILED_URI = smid.DEFAULT_TILED_URI
DEFAULT_CATALOG = smid.DEFAULT_CATALOG
DEFAULT_SAXS_MASK_NAME = smid.DEFAULT_SAXS_MASK_NAME
DEFAULT_WAXS_MASK_NAME = smid.DEFAULT_WAXS_MASK_NAME

SAXS_DETECTOR_NAMES = smid.SAXS_DETECTOR_NAMES
WAXS_DETECTOR_NAMES = smid.WAXS_DETECTOR_NAMES

_LD = smid.LOADER_DEFAULTS
DEFAULT_SAXS_ROW_DELTA = _LD.saxs_row_delta_px
DEFAULT_SAXS_COL_DELTA = _LD.saxs_col_delta_px
DEFAULT_WAXS_ROW_DELTA = _LD.waxs_row_delta_px
DEFAULT_WAXS_COL_DELTA = _LD.waxs_col_delta_px
DEFAULT_SAXS_DIST_DELTA = _LD.saxs_distance_delta_mm

DEFAULT_N_Q = 2000
DEFAULT_N_CHI = 360
DEFAULT_SAXS_MASK = ""
DEFAULT_WAXS_MASK = ""
DEFAULT_DEZINGER = 3000.0
DEFAULT_INCIDENT_ANGLE = 0.0
DEFAULT_THETA_OFFSET = -0.5
DEFAULT_N_QXY = 500
DEFAULT_N_QZ = 500

_EMPTY_DF = _EMPTY_DF_pkg


# enhanced_summary is imported from smi_browser.models.summary above.
# ScanCollection is imported from smi_browser.models.collection above.


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

SEARCH_TYPES = SEARCH_TYPES_pkg
SEARCH_TYPE_MAP = SEARCH_TYPE_MAP_pkg


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
    return _scalars_to_dataframe_impl(scalar_data)


def _scalar_stream_to_frame(run, stream: str) -> pd.DataFrame:
    return _scalar_stream_to_frame_impl(run, stream)


def _detector_for_field(field: str) -> str | None:
    return classify_detector_field(field)


def _orient_frame(arr: np.ndarray, field: str) -> np.ndarray:
    return _orient_frame_impl(arr, field)


# ---------------------------------------------------------------------------
# Polygon mask helpers  (canonical source: smi_browser.data.masks)
# ---------------------------------------------------------------------------


def _normalized_mask_to_xs_ys(mask, field=None, raw_shape=None):
    return _normalized_mask_to_xs_ys_impl(mask, field, raw_shape)


def _xs_ys_to_normalized_mask(xs, ys, names, kinds, field=None, raw_shape=None):
    return _xs_ys_to_normalized_mask_impl(xs, ys, names, kinds, field, raw_shape)


def _default_mask_path_for(detector: str):
    return _default_mask_path_for_impl(detector)


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
            _get_cat(), unified_filters=unified,
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
    tooltips=True,
)
w_mv_status = pn.pane.Markdown("*Click tab to load.*")
w_mv_spinner = pn.indicators.LoadingSpinner(value=False, size=20, visible=False)
w_mv_label = pn.widgets.Select(
    name="Frame label", options=["(frame #)"], value="(frame #)", width=180,
)
w_mv_grid = pn.pane.Bokeh(object=None, sizing_mode="stretch_both", min_height=500)

_multiview_cache: dict = {
    "uid": None, "field": None, "n_frames": 0,
    "frames": None, "renderers": None, "mapper": None,
    "log": None, "data_lo": None, "data_hi": None,
    "suspend_range_cb": False,
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
    label_col = w_mv_label.value
    frame_labels: list[str] = []
    if label_col and label_col != "(frame #)":
        df = w_primary_table.value
        if df is not None and label_col in df.columns:
            vals = df[label_col].values
            for i in range(len(frames)):
                if i < len(vals):
                    v = vals[i]
                    try:
                        frame_labels.append(f"{label_col}={float(v):.4g}")
                    except (ValueError, TypeError):
                        frame_labels.append(f"{label_col}={v}")
                else:
                    frame_labels.append(f"frame {i}")
    if not frame_labels:
        frame_labels = [f"frame {i}" for i in range(len(frames))]

    # Per-frame display arrays + global data range
    displays = [np.where(np.isfinite(a), a, 0).astype(np.float32) for a in frames]
    data_lo, data_hi = _mv_compute_data_range(frames)
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
        # Clamp current value into new bounds
        cur_lo, cur_hi = w_mv_range.value
        cur_lo = max(min(cur_lo, w_mv_range.end - 0.05), w_mv_range.start)
        cur_hi = max(min(cur_hi, w_mv_range.end), cur_lo + 0.05)
        # On a fresh detector load, snap to data percentiles
        if (_multiview_cache.get("field") != field
                or _multiview_cache.get("mapper") is None):
            cur_lo, cur_hi = log10_lo, log10_hi
        w_mv_range.value = (cur_lo, cur_hi)
    finally:
        _multiview_cache["suspend_range_cb"] = False

    lo_val = 10 ** w_mv_range.value[0]
    hi_val = 10 ** w_mv_range.value[1]
    palette = w_mv_cmap.value
    use_log = bool(w_mv_log.value)
    if use_log:
        mapper = LogColorMapper(palette=palette,
                                low=max(lo_val, 1e-9),
                                high=max(hi_val, lo_val * 1.1))
    else:
        mapper = LinearColorMapper(palette=palette, low=lo_val, high=hi_val)

    rows, cols = _mv_grid_dims(len(frames))
    figs = []
    renderers = []
    shared_x = None
    shared_y = None
    for i, disp in enumerate(displays):
        h, w = disp.shape
        kwargs = dict(
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
                ("frame", str(i)),
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
    if not run:
        return
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
    prev_label = w_mv_label.value
    w_mv_label.options = label_options
    if prev_label in label_options:
        w_mv_label.value = prev_label
    else:
        w_mv_label.value = "(frame #)"

    image_fields = list(info["images"])
    # Populate detector dropdown (preserve previous choice if present)
    prev = w_mv_field.value
    if list(w_mv_field.options) != image_fields:
        w_mv_field.options = []
        w_mv_field.options = image_fields
    field = prev if prev in image_fields else image_fields[0]
    if w_mv_field.value != field:
        # Setting value will trigger _on_mv_field which does the fetch.
        w_mv_field.value = field
        return

    _fetch_and_build_multiview(field)


def _fetch_and_build_multiview(field: str):
    """Fetch every frame of `field` and (re)build the grid."""
    run = _ensure_run()
    if not run or not field:
        return
    info = _detail_cache.get("primary_info")
    if not info:
        return
    shape = info["fields"].get(field, ())
    n_frames = shape[0] if len(shape) >= 3 else 1
    if n_frames > MV_MAX_FRAMES:
        capped_n = MV_MAX_FRAMES
        cap_msg = f" (showing first {MV_MAX_FRAMES} of {n_frames})"
    else:
        capped_n = n_frames
        cap_msg = ""

    w_mv_spinner.value = True
    w_mv_spinner.visible = True
    w_mv_status.object = f"*Loading {capped_n} frames…*"
    t0 = time.perf_counter()
    ds = _detail_cache.get("primary_dataset")
    frames = []
    for i in range(capped_n):
        arr = tb.fetch_frame(run, "primary", field, frame_idx=i, _dataset=ds)
        if arr is None:
            continue
        frames.append(_orient_frame(arr, field))
    _multiview_cache["frames"] = frames
    _multiview_cache["n_frames"] = capped_n
    _multiview_cache["uid"] = _state.get("selected_uid")
    _build_multiview_grid(frames, field)
    dt_ms = (time.perf_counter() - t0) * 1000
    w_mv_spinner.value = False
    w_mv_spinner.visible = False
    w_mv_status.object = (
        f"**primary/{field}** — {len(frames)} frames{cap_msg} ({dt_ms:.0f} ms)"
    )


def _on_mv_field(event):
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
# Watch value_throttled — Panel streams this continuously during drag but
# coalesces events so we only process the most recent position, avoiding
# a backlog of Python round-trips.
w_mv_range.param.watch(_on_mv_range, "value_throttled")
# Also watch 'value' (fires on mouse-up) as a final sync to ensure the
# mapper is always exactly at the released position.
w_mv_range.param.watch(_on_mv_range, "value")


def _on_mv_label(event):
    """Rebuild grid with updated frame labels when the label column changes."""
    field = _multiview_cache.get("field")
    frames = _multiview_cache.get("frames")
    if field and frames:
        _build_multiview_grid(frames, field)


w_mv_label.param.watch(_on_mv_label, "value")


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
w_proc_iq_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=400)
w_proc_2d_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_width", height=500)
w_proc_frame_slider = pn.widgets.IntSlider(
    name="Frame", start=0, end=1, value=0, step=1, width=400,
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


def _compute_cross_section(cut: dict):
    """Return ``(axis, intensity, axis_label)`` for one cut, or ``None``."""
    cache = _proc_2d_cache
    x = cache["x"]
    y = cache["y"]
    img = cache["image"]
    if x is None or y is None or img is None:
        return None
    c = float(cut["center"])
    w = float(cut["width"]) or 0.0
    half = max(w / 2.0, 0.0)
    if cut["kind"] == "h":
        mask = (y >= c - half) & (y <= c + half)
        if not np.any(mask):
            # Fall back to nearest single row
            idx = int(np.argmin(np.abs(y - c)))
            section = img[idx, :].astype(float)
        else:
            section = np.nanmean(img[mask, :], axis=0)
        return x, section, cache["x_label"]
    mask = (x >= c - half) & (x <= c + half)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(x - c)))
        section = img[:, idx].astype(float)
    else:
        section = np.nanmean(img[:, mask], axis=1)
    return y, section, cache["y_label"]


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


def _render_cuts_plot():
    """Redraw cross-section plots: separate axes for h and v cuts."""
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

    plots = []

    if h_cuts:
        p_h = bk_figure(
            title="Horizontal cuts \u2014 I(q)", height=300,
            sizing_mode="stretch_width",
            x_axis_type=x_type, y_axis_type=y_type,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        plotted = False
        for i, cut in h_cuts:
            out = _compute_cross_section(cut)
            if out is None:
                continue
            axis, section, axis_label = out
            finite = np.isfinite(section) & ((section > 0) if y_log else np.ones_like(section, dtype=bool))
            if not np.any(finite):
                continue
            alpha = max(0.4, 1.0 - 0.15 * i)
            p_h.line(axis[finite], section[finite],
                     line_color=_CUT_FILL["h"], line_width=1.4, alpha=alpha,
                     legend_label=_format_cut_label(i, cut))
            plotted = True
        if plotted:
            p_h.xaxis.axis_label = _proc_2d_cache["x_label"]
            p_h.yaxis.axis_label = "I"
            p_h.legend.click_policy = "hide"
            p_h.legend.label_text_font_size = "9pt"
            plots.append(p_h)

    if v_cuts:
        p_v = bk_figure(
            title="Vertical cuts \u2014 I(\u03c7)", height=300,
            sizing_mode="stretch_width",
            x_axis_type=x_type, y_axis_type=y_type,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        plotted = False
        for i, cut in v_cuts:
            out = _compute_cross_section(cut)
            if out is None:
                continue
            axis, section, axis_label = out
            finite = np.isfinite(section) & ((section > 0) if y_log else np.ones_like(section, dtype=bool))
            if not np.any(finite):
                continue
            alpha = max(0.4, 1.0 - 0.15 * i)
            p_v.line(axis[finite], section[finite],
                     line_color=_CUT_FILL["v"], line_width=1.4, alpha=alpha,
                     legend_label=_format_cut_label(i, cut))
            plotted = True
        if plotted:
            p_v.xaxis.axis_label = _proc_2d_cache["y_label"]
            p_v.yaxis.axis_label = "I"
            p_v.legend.click_policy = "hide"
            p_v.legend.label_text_font_size = "9pt"
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
# Widgets — Collection
# ---------------------------------------------------------------------------

_COLL_COLS = ["color", "uid_short", "sample", "plan", "detectors", "geometry", "total_s", "uid"]


def _coll_color_formatter(cell_value):
    """Tabulator HTML formatter: render the color column as a swatch."""
    return (
        f'<div style="width:16px;height:16px;border-radius:3px;'
        f'background:{cell_value};margin:auto;"></div>'
    )


w_coll_table = pn.widgets.Tabulator(
    value=pd.DataFrame(columns=_COLL_COLS),
    show_index=False, sizing_mode="stretch_both", height=300,
    selectable="checkbox",
    configuration={"rowHeight": 24, "layout": "fitColumns"},
    hidden_columns=["uid"],
    formatters={"color": {"type": "html"}},
    editors={"color": {"type": "input"}},
    widths={"color": 40},
    titles={"color": "⬤"},
)
w_btn_coll_remove = pn.widgets.Button(
    name="Remove Selected", button_type="danger", width=130,
)
w_coll_label = pn.widgets.Select(
    name="Label column", options=["(none)"], value="(none)", width=180,
)
w_coll_compare_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_both")


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
    q = qchi["q"].values if "q" in qchi.coords else np.arange(img.shape[-1])
    chi = qchi["chi"].values if "chi" in qchi.coords else np.arange(img.shape[0])

    # Ensure img has shape (n_chi, n_q) so Bokeh renders chi on y, q on x.
    if img.shape == (len(q), len(chi)):
        img = img.T

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
    _attach_cuts_to_figure(p, q, chi, display,
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
    _attach_cuts_to_figure(p, qxy, qz, display.T,
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
            new_mask_source=None, new_mask_renderer=None,
            draw_tool=None, edit_tool=None,
            mask_image_shape=None,
            dyn_source=None, dyn_renderer=None,
        )
    w_image_status.object = ""
    w_image_slider.value = 0
    w_image_slider.end = 1
    # Don't clear image_field options — preserve detector selection
    _image_cache.update(n_frames=0, dataset=None, fields=[],
                        raw_shape=None)
    w_explore_plot.object = None
    # Grid (multi-view) tab — drop the figure so a new scan rebuilds fresh
    w_mv_grid.object = None
    w_mv_status.object = "*Click tab to load.*"
    _multiview_cache.update(
        uid=None, field=None, n_frames=0, frames=None,
        renderers=None, mapper=None, log=None,
        data_lo=None, data_hi=None,
    )
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
        _load_multiview()
    elif active == 5:
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
    _processing_guard["active"] = True

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
                cache_geometry=w_cache_enabled.value,
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
                log.exception("2D q-chi plot failed")
                w_proc_2d_plot.object = None
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
    # Bundle primary/baseline/raw metadata with the processed scan
    primary_df = w_primary_table.value if _detail_cache.get("primary_loaded") else None
    # Baseline: fetch as a proper scalar DataFrame (the UI table is transposed
    # into field/before/after rows, which isn't suitable for numeric lookups).
    baseline_df = None
    if _detail_cache.get("baseline_loaded"):
        run = _ensure_run()
        if run and "baseline" in tb.stream_names(run):
            try:
                baseline_df = _scalar_stream_to_frame(run, "baseline")
            except Exception:
                pass
    raw_metadata = w_meta_json.object if w_meta_json.object else None
    _collection.add(
        result, summary, params,
        primary_df=primary_df,
        baseline_df=baseline_df,
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
    old_len = len(w_coll_table.value) if w_coll_table.value is not None else 0
    # Refresh available label columns from stored primary/baseline data
    avail = _collection.available_label_columns()
    label_opts = ["(none)"] + avail
    prev_label = w_coll_label.value
    w_coll_label.options = label_opts
    if prev_label in label_opts:
        w_coll_label.value = prev_label
    else:
        w_coll_label.value = "(none)"
    # Build table with optional label column
    label_col = w_coll_label.value if w_coll_label.value != "(none)" else None
    df = _collection.summary_table(label_column=label_col)
    # Render the color column as HTML swatches.
    if "color" in df.columns:
        df["color"] = df["color"].apply(
            lambda c: (
                f'<div style="width:16px;height:16px;border-radius:3px;'
                f'background:{c};margin:auto;"></div>'
            )
        )
    w_coll_table.value = df
    w_coll_summary.object = _coll_summary_text()
    # Auto-select newly added scans so the I(q) comparison updates live.
    new_len = len(w_coll_table.value)
    if new_len > old_len:
        w_coll_table.selection = list(range(new_len))


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


def _update_coll_compare(*_events):
    """Rebuild I(q) comparison from current table selection."""
    sel = w_coll_table.selection
    df = w_coll_table.value
    if df is None or len(df) == 0 or not sel:
        w_coll_compare_plot.object = None
        return
    uids = [df.iloc[i]["uid"] for i in sel if i < len(df)]
    label_col = w_coll_label.value if w_coll_label.value != "(none)" else None
    fig = _collection.iq_comparison_bokeh(uids, label_column=label_col)
    w_coll_compare_plot.object = fig


def _on_coll_label_change(*_events):
    """Refresh collection table and I(q) comparison when label column changes."""
    _refresh_collection()
    _update_coll_compare()


w_coll_table.param.watch(_update_coll_compare, "selection")
w_coll_label.param.watch(_on_coll_label_change, "value")

w_btn_coll_remove.on_click(_on_coll_remove)


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
    name="Workers", value=1, start=1, end=8, width=90,
)
w_batch_skip_existing = pn.widgets.Checkbox(
    name="Skip uids already in collection", value=True,
)
w_batch_max_jobs = pn.widgets.IntInput(
    name="Max jobs", value=PAGE_SIZE, start=1, end=BatchProcessor.MAX_QUEUE,
    width=110,
)
w_btn_batch_queue = pn.widgets.Button(
    name="Queue current page", button_type="primary",
)
w_btn_batch_cancel = pn.widgets.Button(
    name="Cancel", button_type="warning", disabled=True,
)
w_btn_batch_clear = pn.widgets.Button(
    name="Clear log", button_type="light", disabled=True,
)

_batch_state: dict[str, Any] = {"doc": None, "processor": None}


def _build_proc_params(uid: str) -> tuple:
    """Snapshot current Process-tab widget values into reduction params.

    Mirrors the param-construction in :func:`_on_process` so a batch job
    runs with whatever the user has configured in the Parameters sub-tab.
    Returns ``(callable, params_dict, geometry_label)``.
    """
    geometry = w_proc_geometry.value
    if geometry == "grazing":
        from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_gi

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
        return reduce_smi_gi, gi_params, geometry

    from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined

    params: dict[str, Any] = dict(
        uid=uid,
        tiled_uri=DEFAULT_TILED_URI,
        catalog=DEFAULT_CATALOG,
        solid_angle_correction=True,
        geometry=geometry,
        saxs_mask_path=w_proc_saxs_mask.value or None,
        waxs_mask_path=w_proc_waxs_mask.value or None,
        cache_geometry=w_cache_enabled.value,
    )
    if w_proc_nq.value != DEFAULT_N_Q:
        params["n_q"] = w_proc_nq.value
    else:
        params["n_q"] = DEFAULT_N_Q
    if w_proc_nchi.value != DEFAULT_N_CHI:
        params["n_chi"] = w_proc_nchi.value

    saxs_row_changed = w_proc_saxs_row_delta.value != DEFAULT_SAXS_ROW_DELTA
    saxs_col_changed = w_proc_saxs_col_delta.value != DEFAULT_SAXS_COL_DELTA
    if saxs_row_changed or saxs_col_changed:
        params["saxs_beam_delta_px"] = (
            w_proc_saxs_row_delta.value, w_proc_saxs_col_delta.value,
        )
    waxs_row_changed = w_proc_waxs_row_delta.value != DEFAULT_WAXS_ROW_DELTA
    waxs_col_changed = w_proc_waxs_col_delta.value != DEFAULT_WAXS_COL_DELTA
    if waxs_row_changed or waxs_col_changed:
        params["waxs_beam_delta_px"] = (
            w_proc_waxs_row_delta.value, w_proc_waxs_col_delta.value,
        )
    if w_proc_dist_delta.value != DEFAULT_SAXS_DIST_DELTA:
        params["saxs_distance_delta_mm"] = w_proc_dist_delta.value
    if w_proc_dezinger.value != DEFAULT_DEZINGER:
        params["dezinger_threshold"] = (
            w_proc_dezinger.value if w_proc_dezinger.value > 0 else None
        )
    return reduce_smi_combined, params, geometry


def _batch_process_fn(uid: str):
    """BatchProcessor.process_fn: run one reduction and return the result."""
    run_fn, params, geometry = _build_proc_params(uid)
    if geometry == "grazing":
        # ScanCollection currently only handles transmission results.
        raise NotImplementedError(
            "Batch processing for GI (grazing) geometry is not supported yet."
        )
    # Use pre-snapshotted summary from enqueue time (thread-safe).
    summary = _batch_state.get("summaries", {}).get(uid, {})
    result = run_fn(**params)
    # Fetch primary/baseline scalars and raw metadata so the collection
    # can offer label columns and eventual export.
    try:
        run = _get_cat()[uid]
        primary_df = _scalar_stream_to_frame(run, "primary")
        baseline_df = _scalar_stream_to_frame(run, "baseline")
        raw_md = dict(run.metadata)
    except Exception:
        primary_df = None
        baseline_df = None
        raw_md = None
    # Pack extra data into params (BatchProcessor passes it through).
    params["_primary_df"] = primary_df
    params["_baseline_df"] = baseline_df
    params["_raw_metadata"] = raw_md
    return result, summary, params


def _batch_skip(uid: str) -> bool:
    if not w_batch_skip_existing.value:
        return False
    return uid in _collection


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
            # Only do a full replace when the shape changes (new jobs added
            # or cleared).  Otherwise patch individual cells so Tabulator
            # keeps its scroll position instead of jumping to the top.
            old_df = w_batch_table.value
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
                    w_batch_table.patch(patches)
            else:
                w_batch_table.value = new_df

            # Apply row colours based on job state.
            def _color_batch_rows(row):
                css = _BATCH_ROW_COLORS.get(row["state"], "")
                return [css] * len(row)
            w_batch_table.style.apply(_color_batch_rows, axis=1)

            w_btn_batch_queue.disabled = running
            w_btn_batch_cancel.disabled = not running
            w_btn_batch_clear.disabled = running or total == 0

            # Keep the collection panel in sync as jobs land.
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


def _batch_add_fn(result, summary, params):
    # ScanCollection internals are plain dicts; updates are GIL-protected.
    # We add on the worker thread so the snapshot fired immediately after
    # already reflects the new collection size.
    primary_df = params.pop("_primary_df", None)
    baseline_df = params.pop("_baseline_df", None)
    raw_metadata = params.pop("_raw_metadata", None)
    _collection.add(
        result, summary, params,
        primary_df=primary_df,
        baseline_df=baseline_df,
        raw_metadata=raw_metadata,
    )


def _ensure_batch_processor() -> BatchProcessor:
    bp = _batch_state.get("processor")
    workers = max(1, int(w_batch_max_workers.value or 1))
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
    df = w_table.value
    if df is None or len(df) == 0:
        pn.state.notifications.warning("No search results to queue.")
        return
    # Capture the current Bokeh document on the UI thread for cross-thread
    # dispatch from worker threads.
    try:
        _batch_state["doc"] = pn.state.curdoc
    except Exception:
        _batch_state["doc"] = None

    max_jobs = max(1, int(w_batch_max_jobs.value or 25))
    uids = df["uid"].tolist()[:max_jobs]
    samples = (
        df["sample_name"].tolist()[:max_jobs]
        if "sample_name" in df.columns else [""] * len(uids)
    )
    items = [(u, s) for u, s in zip(uids, samples) if u]
    if not items:
        pn.state.notifications.warning("No valid uids in current page.")
        return

    # Snapshot search-table rows so the worker thread never reads w_table.
    summaries: dict[str, dict] = {}
    for _, row in df.iterrows():
        uid_val = row.get("uid")
        if uid_val:
            summaries[uid_val] = row.to_dict()
    _batch_state["summaries"] = summaries

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


w_btn_batch_queue.on_click(_on_batch_queue)
w_btn_batch_cancel.on_click(_on_batch_cancel)
w_btn_batch_clear.on_click(_on_batch_clear)


batch_panel = pn.Column(
    pn.pane.Markdown(
        "**Batch process scans from the current search results.** "
        "Each job uses the parameters configured in the *Parameters* "
        "sub-tab above.  Reductions run on a background thread so the "
        "interface stays interactive; results land in the Scan Collection "
        "as they complete.  Already-processed uids are skipped if the "
        "checkbox is on.  GI (grazing) geometry is not yet supported in "
        "batch mode.",
    ),
    pn.Row(w_btn_batch_queue, w_btn_batch_cancel, w_btn_batch_clear),
    pn.Row(w_batch_max_jobs, w_batch_max_workers, w_batch_skip_existing),
    w_batch_status,
    w_batch_progress,
    w_batch_table,
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
        from tiled.client import from_uri
        client = from_uri(DEFAULT_TILED_URI)
        info = client.context.whoami()
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
    from tiled.client import from_uri
    from tiled.client.context import password_grant

    if not username or not password:
        raise ValueError("Username and password are required.")

    client = from_uri(DEFAULT_TILED_URI)
    ctx = client.context
    providers = ctx.server_info.authentication.providers
    if not providers:
        raise RuntimeError("Tiled server reports no authentication providers.")
    spec = providers[0]
    auth_endpoint = spec.links["auth_endpoint"]
    tokens = password_grant(
        ctx.http_client, auth_endpoint, spec.provider, username, password,
    )
    ctx.configure_auth(tokens, remember_me=True)

    # Drop the cached catalog so the next access uses the refreshed tokens.
    global _cat
    _cat = None

    info = ctx.whoami()
    identities = (info or {}).get("identities") or []
    return identities[0]["id"] if identities else username


def _tiled_logout() -> None:
    """Clear the cached tiled session for this server."""
    try:
        from tiled.client import from_uri
        client = from_uri(DEFAULT_TILED_URI)
        try:
            client.logout()
        except Exception:
            # logout() can fail if there's no live session; clear cache anyway.
            pass
        try:
            client.context.tokens.clear()
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
    w_login_msg.object = "*signing in…*"
    w_login_submit.disabled = True
    try:
        user = _tiled_login(user_in, pwd)
        w_login_msg.object = f"✅ Signed in as `{user}`"
        w_login_pass.value = ""
        w_login_form.visible = False
        _refresh_login_status()
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
    try:
        pn.state.notifications.info("Logged out of tiled.")
    except Exception:
        pass


w_btn_login.on_click(_toggle_login_form)
w_login_submit.on_click(_on_login_submit)
w_btn_logout.on_click(_on_logout)


w_btn_live = pn.widgets.Toggle(
    name="🔴 Go Live", value=False,
    button_type="danger", width=140,
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
    """
    doc = _live.get("doc")
    if doc is None:
        # No document captured — best effort inline (unit tests, REPL).
        try:
            fn()
        except Exception:
            log.exception("live: inline dispatch failed")
        return
    try:
        doc.add_next_tick_callback(fn)
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
        w_table.selectable = False
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
            _get_cat(), unified_filters=[], offset=0, limit=1,
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
    # Extend slider range; auto-advance to the latest frame iff this field
    # is the one currently displayed.
    new_end = max(0, n_total - 1)
    if w_image_slider.end != new_end:
        w_image_slider.end = new_end
    if field == _image_cache.get("field"):
        # Setting .value triggers _on_image_slider → re-renders + cursor sync.
        if w_image_slider.value != new_end:
            w_image_slider.value = new_end
        else:
            # Same index, but the underlying frame changed — force re-render.
            try:
                _render_image_frame(field, new_end)
            except Exception:
                log.exception("live: frame re-render failed")


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
    collapsed=False,
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
        "Grid",
        pn.Column(
            pn.Row(w_mv_status, w_mv_spinner),
            pn.Row(w_mv_field, w_mv_cmap, w_mv_log, w_mv_label, sizing_mode="stretch_width"),
            w_mv_range,
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
                                   w_cuts_log_x, w_cuts_log_y),
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
                        pn.Card(
                            w_trans_row,
                            w_gi_row,
                            title="Output grid",
                            collapsed=False, sizing_mode="stretch_width",
                        ),
                        pn.Card(
                            pn.Row(w_proc_saxs_mask, w_proc_waxs_mask),
                            pn.pane.Markdown(
                                "*Leave blank to use the bundled PyHyperScattering "
                                "defaults shown in the placeholder text.  Use the "
                                "Explore tab to view or edit a mask interactively.*",
                            ),
                            title="Masks",
                            collapsed=False, sizing_mode="stretch_width",
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
                            collapsed=False, sizing_mode="stretch_width",
                        ),
                        pn.Card(
                            pn.Row(w_proc_dezinger),
                            pn.pane.Markdown("*Set to 0 to disable hot-pixel rejection.*"),
                            title="Hot-pixel rejection",
                            collapsed=False, sizing_mode="stretch_width",
                        ),
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

detail_panel = pn.Column(
    pn.Row(w_btn_toggle_sidebar, w_detail_title, sizing_mode="stretch_width"),
    w_detail_tabs,
    sizing_mode="stretch_both",
    min_width=600,
)

collection_card = pn.Card(
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

# Load the 25 most recent scans in the background so the UI renders
# immediately.  Uses a cached count to skip the expensive count probe.
def _startup_search():
    try:
        # Fast path: use cached count as count hint
        count_hint = tb.load_cached_count()

        cat = _get_cat()

        if count_hint is None:
            # First run or cache missing — use len(cat) which is faster
            # than the REST probe (10-34s vs 30-40s for REST count).
            try:
                t0 = time.time()
                count_hint = len(cat)
                log.info("startup: len(cat) = %d in %.1fs", count_hint, time.time() - t0)
            except Exception:
                count_hint = None  # fall through to slow path

        if count_hint:
            log.info("startup: using count_hint=%d", count_hint)
        offset = _state["page"] * _state["page_size"]
        limit = _state["page_size"]

        summaries, total = tb.fetch_page_fast(
            cat, unified_filters=[], offset=offset, limit=limit,
            count_hint=count_hint,
        )
        _state["total"] = total
        _state["unified_filters"] = []

        # Cache the real count for next startup
        if total > 0:
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
        w_status.object = "*Ready — add filters and press Search*"
    finally:
        w_search_spinner.value = False
        w_search_spinner.visible = False

_startup_thread = threading.Thread(target=_startup_search, daemon=True)
_startup_thread.start()

if __name__ == "__main__":
    dashboard.show(title="SMI Browser")
