"""Collection UI — widgets and callbacks for the Scan Collection panel.

Widgets are created per-session inside ``wire()`` so that ``panel serve``
can properly register them with the session's Bokeh Document.  Module-level
widget singletons would persist via Python's import cache and lose their
document binding across sessions.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn

if TYPE_CHECKING:
    from smi_browser.models.collection import ScanCollection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLL_COLS = [
    "pinned", "color", "uid_short", "sample", "plan", "detectors",
    "geometry", "total_s", "uid",
]


# ---------------------------------------------------------------------------
# Factory + Wire
# ---------------------------------------------------------------------------


def wire(collection: ScanCollection, plot_style_widget=None) -> SimpleNamespace:
    """Create collection widgets and connect them to *collection*.

    Returns a ``SimpleNamespace`` with attributes:
        coll_table, summary_md, btn_remove, label_select, compare_plot,
        refresh   (callable — use to push collection state to the UI).
    """

    # --- widgets (created fresh each session) ---
    coll_table = pn.widgets.Tabulator(
        value=pd.DataFrame(columns=_COLL_COLS),
        show_index=False, sizing_mode="stretch_both", min_height=300,
        selectable="checkbox",
        configuration={"rowHeight": 24, "layout": "fitColumns"},
        hidden_columns=["uid"],
        formatters={
            "color": {"type": "html"},
            "pinned": {"type": "tickCross", "allowEmpty": True,
                       "allowTruthy": True},
        },
        editors={"color": {"type": "input"}, "pinned": {"type": "tickCross"}},
        widths={"color": 40, "pinned": 50},
        titles={"color": "⬤", "pinned": "📌"},
    )

    btn_remove = pn.widgets.Button(
        name="Remove Selected", button_type="danger", width=130,
    )
    btn_pin = pn.widgets.Button(
        name="Pin selected", button_type="primary", width=110,
    )
    btn_unpin = pn.widgets.Button(
        name="Unpin selected", button_type="default", width=120,
    )
    btn_clear_unpinned = pn.widgets.Button(
        name="Clear all unpinned", button_type="warning", width=140,
    )

    pinned_only = pn.widgets.Checkbox(
        name="Pinned only", value=False, width=110,
    )

    label_select = pn.widgets.Select(
        name="Label column", options=["(none)"], value="(none)", width=180,
    )

    compare_plot = pn.pane.Bokeh(object=None, sizing_mode="stretch_both")

    summary_md = pn.pane.Markdown(
        "*Empty — process scans and add them here*", margin=(0, 5),
    )

    # --- callbacks ---

    def _summary_text() -> str:
        n_total = len(collection)
        if n_total == 0:
            return "*Empty — process scans and add them here*"
        n_pinned = len(collection.pinned_uids())
        n_unpinned = n_total - n_pinned
        if n_unpinned:
            return (
                f"**{n_pinned} pinned · {n_unpinned} unpinned** "
                f"({n_total} total)"
            )
        return f"**{n_total} pinned scan{'s' if n_total != 1 else ''}**"

    def refresh() -> None:
        old_len = len(coll_table.value) if coll_table.value is not None else 0
        avail = collection.available_label_columns()
        label_opts = ["(none)"] + avail
        prev_label = label_select.value
        label_select.options = label_opts
        if prev_label in label_opts:
            label_select.value = prev_label
        else:
            label_select.value = "(none)"
        label_col = label_select.value if label_select.value != "(none)" else None
        df = collection.summary_table(
            label_column=label_col,
            pinned_only=bool(pinned_only.value),
        )
        if "color" in df.columns:
            df["color"] = df["color"].apply(
                lambda c: (
                    f'<div style="width:16px;height:16px;border-radius:3px;'
                    f'background:{c};margin:auto;"></div>'
                )
            )
        coll_table.value = df
        summary_md.object = _summary_text()
        new_len = len(coll_table.value)
        if new_len > old_len:
            try:
                coll_table.selection = list(range(new_len))
            except Exception:
                log.exception("coll_table.selection assignment failed")

    def _selected_uids_from_table() -> list[str]:
        sel = coll_table.selection
        df = coll_table.value
        if df is None or not sel:
            return []
        out = []
        for idx in sorted(sel):
            if 0 <= idx < len(df):
                uid = df.iloc[idx].get("uid")
                if uid:
                    out.append(uid)
        return out

    def _on_remove(event):
        for uid in _selected_uids_from_table():
            collection.remove(uid)
        refresh()

    def _on_pin(event):
        for uid in _selected_uids_from_table():
            collection.pin(uid)
        refresh()

    def _on_unpin(event):
        for uid in _selected_uids_from_table():
            collection.unpin(uid)
        refresh()

    def _on_clear_unpinned(event):
        n = collection.clear_unpinned()
        if n:
            try:
                pn.state.notifications.success(
                    f"Cleared {n} unpinned scan{'s' if n != 1 else ''}",
                )
            except Exception:
                pass
        refresh()

    def _on_pinned_only_toggle(*_events):
        refresh()

    def _update_compare(*_events):
        sel = coll_table.selection
        df = coll_table.value
        if df is None or len(df) == 0 or not sel:
            compare_plot.object = None
            return
        uids = [df.iloc[i]["uid"] for i in sel if i < len(df)]
        label_col = label_select.value if label_select.value != "(none)" else None
        style = plot_style_widget.value if plot_style_widget else "markers"
        try:
            fig = collection.iq_comparison_bokeh(uids, label_column=label_col,
                                                 plot_style=style)
            compare_plot.object = fig
        except Exception:
            log.exception("_update_compare failed")
            compare_plot.object = None

    def _on_label_change(*_events):
        refresh()
        _update_compare()

    coll_table.param.watch(_update_compare, "selection")
    label_select.param.watch(_on_label_change, "value")
    pinned_only.param.watch(_on_pinned_only_toggle, "value")
    btn_remove.on_click(_on_remove)
    btn_pin.on_click(_on_pin)
    btn_unpin.on_click(_on_unpin)
    btn_clear_unpinned.on_click(_on_clear_unpinned)
    if plot_style_widget is not None:
        plot_style_widget.param.watch(lambda *_: _update_compare(), "value")

    return SimpleNamespace(
        coll_table=coll_table,
        summary_md=summary_md,
        btn_remove=btn_remove,
        btn_pin=btn_pin,
        btn_unpin=btn_unpin,
        btn_clear_unpinned=btn_clear_unpinned,
        pinned_only=pinned_only,
        label_select=label_select,
        compare_plot=compare_plot,
        refresh=refresh,
    )
