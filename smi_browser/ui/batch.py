"""Batch processing UI — queue scans, track progress, display results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import panel as pn

from smi_browser._batch import BatchProcessor
from smi_browser.config import PAGE_SIZE

if TYPE_CHECKING:
    from smi_browser.models.collection import ScanCollection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH_ROW_COLORS = {
    "running": "background-color: #d4edda",
    "error": "background-color: #f8d7da",
    "done": "background-color: #f0f0f0; color: #888",
    "skipped": "background-color: #f0f0f0; color: #888",
    "cancelled": "background-color: #fff3cd; color: #888",
    "queued": "",
}

# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

batch_status = pn.pane.Markdown(
    "*Idle — queue scans from the current search results to process them in "
    "the background.*",
    margin=(0, 5),
)
batch_progress = pn.indicators.Progress(
    name="Batch progress", value=0, max=1, width=400, visible=False,
)
batch_table = pn.widgets.Tabulator(
    pd.DataFrame(columns=["uid_short", "label", "state", "duration_s", "error"]),
    height=320, layout="fit_data_stretch", show_index=False, disabled=True,
    sizing_mode="stretch_width",
)
batch_max_workers = pn.widgets.IntInput(
    name="Workers", value=1, start=1, end=16, width=90,
)
batch_skip_existing = pn.widgets.Checkbox(
    name="Skip uids already in collection", value=True,
)
batch_max_jobs = pn.widgets.IntInput(
    name="Max jobs", value=PAGE_SIZE, start=1, end=BatchProcessor.MAX_QUEUE,
    width=110,
)
btn_queue = pn.widgets.Button(
    name="Queue scans", button_type="primary",
)
btn_cancel = pn.widgets.Button(
    name="Cancel", button_type="warning", disabled=True,
)
btn_clear = pn.widgets.Button(
    name="Clear log", button_type="light", disabled=True,
)

panel = pn.Column(
    pn.pane.Markdown(
        "**Batch process scans from the current search results.** "
        "Each job uses the parameters configured in the *Parameters* "
        "sub-tab above.  Reductions run on a background thread so the "
        "interface stays interactive; results land in the Scan Collection "
        "as they complete.  Already-processed uids are skipped if the "
        "checkbox is on.",
    ),
    pn.Row(btn_queue, btn_cancel, btn_clear),
    pn.Row(batch_max_jobs, batch_max_workers, batch_skip_existing),
    batch_status,
    batch_progress,
    batch_table,
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------------
# Wire
# ---------------------------------------------------------------------------


def wire(
    batch_state: dict,
    *,
    collection: ScanCollection,
    build_proc_params: Callable,
    get_cat: Callable,
    scalar_stream_to_frame: Callable,
    search_table: Any,
    refresh_collection: Callable,
    get_search_uids: Callable | None = None,
) -> None:
    """Connect batch widgets to processing + collection logic."""

    def _process_fn(uid: str):
        run_fn, params, geometry = build_proc_params(uid)
        summary = batch_state.get("summaries", {}).get(uid, {})
        result = run_fn(**params)
        try:
            run = get_cat()[uid]
            primary_df = scalar_stream_to_frame(run, "primary")
            baseline_df = scalar_stream_to_frame(run, "baseline")
            raw_md = dict(run.metadata)
        except Exception:
            primary_df = None
            baseline_df = None
            raw_md = None
        params["_primary_df"] = primary_df
        params["_baseline_df"] = baseline_df
        params["_raw_metadata"] = raw_md
        return result, summary, params

    def _skip_fn(uid: str) -> bool:
        if not batch_skip_existing.value:
            return False
        return uid in collection

    def _dispatch(snap: dict) -> None:
        def _apply():
            try:
                states = snap["states"]
                total = snap["total"]
                done = (
                    states["done"] + states["error"]
                    + states["skipped"] + states["cancelled"]
                )
                running = snap["running"]

                batch_progress.max = max(total, 1)
                batch_progress.value = done
                batch_progress.visible = total > 0

                label = "running" if running else (
                    "cancelling" if snap["cancel_requested"] else "idle"
                )
                batch_status.object = (
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
                old_df = batch_table.value
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
                        batch_table.patch(patches)
                else:
                    batch_table.value = new_df

                def _color_rows(row):
                    css = _BATCH_ROW_COLORS.get(row["state"], "")
                    return [css] * len(row)
                batch_table.style.apply(_color_rows, axis=1)

                btn_queue.disabled = running
                btn_cancel.disabled = not running
                btn_clear.disabled = running or total == 0

                refresh_collection()
            except Exception:
                log.exception("batch: UI render failed")

        doc = batch_state.get("doc")
        if doc is None:
            _apply()
            return
        try:
            doc.add_next_tick_callback(_apply)
        except Exception:
            log.exception("batch: add_next_tick_callback failed")

    def _add_fn(result, summary, params):
        primary_df = params.pop("_primary_df", None)
        baseline_df = params.pop("_baseline_df", None)
        raw_metadata = params.pop("_raw_metadata", None)
        collection.add(
            result, summary, params,
            primary_df=primary_df,
            baseline_df=baseline_df,
            raw_metadata=raw_metadata,
        )

    def _ensure_processor() -> BatchProcessor:
        bp = batch_state.get("processor")
        workers = max(1, int(batch_max_workers.value or 1))
        if bp is None or bp._max_workers != workers or not bp.is_running:
            if bp is not None and bp.is_running:
                return bp
            bp = BatchProcessor(
                process_fn=_process_fn,
                add_fn=_add_fn,
                status_cb=_dispatch,
                skip_fn=_skip_fn,
                max_workers=workers,
            )
            batch_state["processor"] = bp
        return bp

    def _on_queue(event):
        max_jobs = max(1, int(batch_max_jobs.value or 25))
        skip_existing = batch_skip_existing.value

        try:
            batch_state["doc"] = pn.state.curdoc
        except Exception:
            batch_state["doc"] = None

        # Use cross-page fetcher if available, otherwise fall back to
        # current page only.
        if get_search_uids is not None:
            items, summaries = get_search_uids(
                max_jobs, skip_existing, collection,
            )
        else:
            df = search_table.value
            if df is None or len(df) == 0:
                pn.state.notifications.warning("No search results to queue.")
                return
            items = []
            summaries = {}
            for _, row in df.iterrows():
                uid = row.get("uid", "")
                if not uid:
                    continue
                label = row.get("sample_name", "")
                if skip_existing and uid in collection:
                    continue
                summaries[uid] = row.to_dict()
                items.append((uid, label))
                if len(items) >= max_jobs:
                    break

        if not items:
            pn.state.notifications.info(
                "Nothing to queue (all already processed)."
            )
            return

        batch_state["summaries"] = summaries

        bp = _ensure_processor()
        n = bp.enqueue(items)
        if n == 0:
            pn.state.notifications.info("Nothing to queue (all already tracked).")
            return
        bp.start()
        pn.state.notifications.success(
            f"Queued {n} scan{'s' if n != 1 else ''} for batch processing."
        )

    def _on_cancel(event):
        bp = batch_state.get("processor")
        if bp is None:
            return
        bp.cancel()
        pn.state.notifications.info(
            "Cancellation requested; the running job will finish."
        )

    def _on_clear(event):
        bp = batch_state.get("processor")
        if bp is None:
            return
        bp.clear_terminal()

    btn_queue.on_click(_on_queue)
    btn_cancel.on_click(_on_cancel)
    btn_clear.on_click(_on_clear)
