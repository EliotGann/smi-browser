"""Live mode UI — tiled streaming subscriptions + lockout."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import panel as pn

from smi_browser._stream import LiveStreamManager

if TYPE_CHECKING:
    from smi_browser.state import AppState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

btn_live = pn.widgets.Toggle(
    name="🔴 Go Live", value=False,
    button_type="danger", width=140,
)
live_banner = pn.pane.Markdown(
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch_to_doc(live: dict, fn: Callable) -> None:
    doc = live.get("doc")
    if doc is None:
        try:
            fn()
        except Exception:
            log.exception("live: inline dispatch failed")
        return
    try:
        doc.add_next_tick_callback(fn)
    except Exception:
        log.exception("live: add_next_tick_callback failed")


def _save(live: dict, widget: Any, *params: str) -> None:
    bucket = live["saved"].setdefault(widget, {})
    for p in params:
        if p not in bucket:
            try:
                bucket[p] = getattr(widget, p)
            except Exception:
                pass


def _restore_all(live: dict) -> None:
    for widget, params in live["saved"].items():
        for p, v in params.items():
            try:
                setattr(widget, p, v)
            except Exception:
                pass
    live["saved"].clear()


# ---------------------------------------------------------------------------
# Wire
# ---------------------------------------------------------------------------


def wire(
    app: AppState,
    *,
    get_cat: Callable,
    fetch_page_fast: Callable,
    load_metadata: Callable,
    load_primary: Callable,
    load_images: Callable,
    ensure_run: Callable,
    reset_detail: Callable,
    build_explore_plot: Callable,
    render_image_frame: Callable,
    lockout_widgets: list,
    search_table: Any,
    filter_rows: list,
    search_card: Any,
    collection_card: Any,
    detail_tabs: Any,
    image_slider: Any,
    image_cache: dict,
    detail_cache: dict,
    explore_tab_index: int = 3,
) -> None:
    """Connect live-mode toggle, streaming callbacks, and lockout."""

    _live = app.live
    _state = app.search

    def _set_banner(text: str) -> None:
        live_banner.object = text
        live_banner.visible = bool(text)

    def _set_lockout(on: bool) -> None:
        if on:
            for w in lockout_widgets:
                _save(_live, w, "disabled")
                w.disabled = True
            for rd in filter_rows:
                for key in ("type", "key", "val", "suggest", "remove"):
                    wgt = rd.get(key)
                    if wgt is not None:
                        _save(_live, wgt, "disabled")
                        wgt.disabled = True
            # Lock the table (no row selection mid-stream).
            _save(_live, search_table, "selectable")
            search_table.selectable = False
            _save(_live, detail_tabs, "active")
            detail_tabs.active = explore_tab_index
            _save(_live, search_card, "collapsed")
            search_card.collapsed = True
            _save(_live, collection_card, "collapsed")
            collection_card.collapsed = True
        else:
            _restore_all(_live)

    def _pick_initial_uid() -> str | None:
        try:
            summaries, _total = fetch_page_fast(
                get_cat(), unified_filters=[], offset=0, limit=1,
            )
        except Exception as exc:
            log.warning("live: initial uid fetch failed: %s", exc)
            return None
        if not summaries:
            return None
        return summaries[0].get("uid")

    def _switch_to(uid: str) -> None:
        if not uid or uid == _state.get("selected_uid"):
            if uid and _live["manager"] is not None:
                try:
                    run = ensure_run()
                    if run is not None and _live["manager"].watched_uid != uid:
                        _live["manager"].watch_run(uid, run)
                except Exception as exc:
                    log.warning("live: re-watch failed: %s", exc)
            return

        _state["selected_uid"] = uid
        reset_detail(preserve_figure=True)
        _state["selected_uid"] = uid
        try:
            load_metadata(uid)
            load_primary()
            load_images()
        except Exception as exc:
            log.exception("live: switch_to load failed")
            _set_banner(f"🔴 LIVE — error loading `{uid[:8]}`: `{exc}`")
            return

        summary = app.detail_cache.get("summary") or {}
        _set_banner(
            f"🔴 **LIVE** — watching `{uid[:8]}` · "
            f"scan {summary.get('scan_id', '?')} · "
            f"{summary.get('sample_name', '?')} · "
            f"{summary.get('detectors', '?')}"
        )

        mgr = _live["manager"]
        if mgr is not None:
            run = ensure_run()
            if run is not None:
                try:
                    mgr.watch_run(uid, run)
                except Exception as exc:
                    log.exception("live: watch_run failed")
                    _set_banner(
                        f"🔴 LIVE — `{uid[:8]}` (subscribe failed: `{exc}`)"
                    )

    # --- Streaming callbacks ---

    def _on_new_run(uid: str) -> None:
        log.info("live: new run %s", uid)
        _switch_to(uid)

    def _on_primary_extended(uid: str) -> None:
        if uid != _state.get("selected_uid"):
            return
        detail_cache["primary_loaded"] = False
        try:
            load_primary()
        except Exception:
            log.exception("live: primary refresh failed")
            return
        try:
            build_explore_plot()
        except Exception:
            log.exception("live: explore plot rebuild failed")

    def _on_frame_extended(uid: str, field: str, n_total: int) -> None:
        if uid != _state.get("selected_uid"):
            return
        detail_cache["primary_dataset"] = None
        image_cache["dataset"] = None
        if n_total <= 0:
            return
        new_end = max(0, n_total - 1)
        if image_slider.end != new_end:
            image_slider.end = new_end
        if field == image_cache.get("field"):
            if image_slider.value != new_end:
                image_slider.value = new_end
            else:
                try:
                    render_image_frame(field, new_end)
                except Exception:
                    log.exception("live: frame re-render failed")

    def _on_error(stage: str, exc: Exception) -> None:
        log.warning("live %s: %s", stage, exc)

    # --- Toggle ---

    def _on_toggle(event) -> None:
        if event.new and not _live["active"]:
            _enter()
        elif (not event.new) and _live["active"]:
            _exit()

    def _enter() -> None:
        log.info("live: entering live mode")
        _live["active"] = True
        _live["doc"] = pn.state.curdoc
        btn_live.name = "■ Stop Live"
        _set_lockout(True)
        _set_banner("🔴 **LIVE** — connecting to tiled stream…")

        dispatcher = lambda fn: _dispatch_to_doc(_live, fn)  # noqa: E731
        mgr = LiveStreamManager(
            get_cat(),
            on_new_run=_on_new_run,
            on_primary_extended=_on_primary_extended,
            on_frame_extended=_on_frame_extended,
            on_error=_on_error,
            dispatcher=dispatcher,
        )
        _live["manager"] = mgr
        try:
            mgr.start()
        except Exception as exc:
            log.exception("live: manager.start failed")
            _set_banner(f"🔴 LIVE — start failed: `{exc}`")
            return

        initial_uid = _pick_initial_uid()
        if initial_uid:
            _switch_to(initial_uid)
        else:
            _set_banner("🔴 **LIVE** — waiting for first run…")

    def _exit() -> None:
        log.info("live: exiting live mode")
        _live["active"] = False
        btn_live.name = "🔴 Go Live"
        mgr = _live["manager"]
        _live["manager"] = None
        if mgr is not None:
            try:
                mgr.stop()
            except Exception:
                log.exception("live: manager.stop failed")
        _live["doc"] = None
        _set_lockout(False)
        _set_banner("")

    btn_live.param.watch(_on_toggle, "value")
