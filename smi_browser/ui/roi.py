"""Explore ROI drawing and explicit, cancellable background computation."""
from __future__ import annotations

from concurrent.futures import CancelledError
import logging
import threading
import time

import pandas as pd
import panel as pn
from bokeh.models import BoxEditTool, ColumnDataSource, LabelSet

from smi_browser.cache import ScanCache
from smi_browser.models.roi import compute_roi_scalars, roi_slices

log = logging.getLogger(__name__)
FIELDS = ("name", "x", "y", "width", "height")


class ROIControls:
    """One controller per Panel session; callbacks resolve app state on the doc thread.

    ``context`` supplies uid, stream, field, n_frames, and a snapshot read_frame
    callable. ``refresh`` republishes cached products into the scalar widgets.
    """

    def __init__(self, context, refresh):
        self.context = context
        self.refresh = refresh
        self.key = None
        self.shape = None
        self.source = None
        self.figure = None
        self.tool = None
        self.renderer = None
        self.guard = False
        self.job = None
        self.draw = pn.widgets.Checkbox(name="Draw / edit rectangular ROIs", value=False)
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=FIELDS), show_index=False, layout="fit_columns",
            sizing_mode="stretch_width", height=180,
            titles={"x": "X centre", "y": "Y centre", "width": "Width", "height": "Height"},
        )
        self.clear = pn.widgets.Button(name="Clear ROIs", sizing_mode="stretch_width")
        self.compute = pn.widgets.Button(name="Compute ROI scalars", button_type="primary",
                                         sizing_mode="stretch_width")
        self.cancel = pn.widgets.Button(name="Cancel", disabled=True, sizing_mode="stretch_width")
        self.progress = pn.indicators.Progress(value=0, max=100, sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("*Load a single detector image to define ROIs.*",
                                       sizing_mode="stretch_width")
        self.panel = pn.Column(
            pn.pane.Markdown(
                "Enable drawing, then **drag** a rectangle on the image. Drag an existing "
                "ROI to move it; select and press **Backspace/Delete** to remove it. "
                "Edit names, centres and sizes in the table.\n\n"
                "**Compute** reads all frames of the selected scan/stream/detector once "
                "for all ROIs, reusing cached frames. Results appear under **Scalars** "
                "and **Primary**, named `roi:detector:ROI:stat`.\n\n"
                "Statistics use original pixel values in displayed coordinates: **sum, "
                "mean, std** (population), **min, max, count** (finite pixels). "
                "Nonfinite pixels are excluded; detector mask overlays and colour scaling "
                "do not affect results. Pixel centres determine inclusion."
            ),
            self.draw, self.table, self.clear,
            pn.Row(self.compute, self.cancel, sizing_mode="stretch_width"),
            self.progress, self.status, sizing_mode="stretch_width",
        )
        self.draw.param.watch(self._toggle, "value")
        self.table.on_edit(self._table_edit)
        self.clear.on_click(self._clear)
        self.compute.on_click(self._compute)
        self.cancel.on_click(self._cancel)

    def deactivate(self):
        """Invalidate the image context immediately on navigation."""
        if self.job:
            self.job["cancel"].set()
        self.key = None
        self.draw.value = False
        self.compute.disabled = True
        self.status.object = "*Select one monochrome detector image outside live mode to use ROIs.*"

    def attach(self, figure, shape):
        ctx = self.context()
        if ctx is None or len(shape) != 2:
            self.deactivate()
            return
        key = (ctx["uid"], ctx["stream"], ctx["field"])
        shape = tuple(shape)
        changed = key != self.key or shape != self.shape
        if changed:
            if self.job:
                self.job["cancel"].set()
            self.key, self.shape = key, shape
            product = ScanCache(key[0]).read_rois(key[1]).get(key[2], {})
            rois = product.get("rois", []) if tuple(product.get("shape", ())) == shape else []
        else:
            rois = self._rois()
        rebuilt = self.figure is not figure
        if rebuilt:
            self.figure = figure
            self.source = ColumnDataSource({k: [] for k in FIELDS})
            self.renderer = figure.rect(
                x="x", y="y", width="width", height="height", source=self.source,
                fill_color="#00ADDC", fill_alpha=.12, line_color="#00ADDC", line_width=2,
            )
            figure.add_layout(LabelSet(x="x", y="y", text="name", source=self.source,
                                       text_color="#105C78", background_fill_color="white",
                                       background_fill_alpha=.8, text_font_size="10pt"))
            self.tool = BoxEditTool(renderers=[self.renderer], description="Draw / edit ROIs")
            figure.add_tools(self.tool)
            self.source.on_change("data", self._source_edit)
        if changed or not self.source.data["name"]:
            self._set_rois(rois)
        self.compute.disabled = self.job is not None
        if changed:
            self.status.object = f"**{key[1]}/{key[2]}** — {len(rois)} ROIs. Compute to fill scalar plots."
        if changed or rebuilt:
            self._toggle()

    def _rois(self):
        if self.source is None:
            return []
        return [dict(zip(FIELDS, row)) for row in zip(*(self.source.data[k] for k in FIELDS))]

    def _set_rois(self, rois):
        self.guard = True
        try:
            self.source.data = {k: [r[k] for r in rois] for k in FIELDS}
            self.table.value = pd.DataFrame(rois, columns=FIELDS)
        finally:
            self.guard = False

    def _toggle(self, *_):
        if self.figure is None or self.tool is None:
            return
        if self.draw.value and self.key:
            self.figure.toolbar.active_drag = self.tool
            self.figure.toolbar.active_tap = self.tool
        else:
            if self.figure.toolbar.active_drag is self.tool:
                self.figure.toolbar.active_drag = "auto"
            if self.figure.toolbar.active_tap is self.tool:
                self.figure.toolbar.active_tap = "auto"

    def _source_edit(self, attr, old, new):
        if self.guard or self.key is None:
            return
        rois = self._rois()
        names = {r["name"] for r in rois if r["name"]}
        for roi in rois:
            if not roi["name"]:
                i = 1
                while f"ROI{i}" in names:
                    i += 1
                roi["name"] = f"ROI{i}"
                names.add(roi["name"])
        self._set_rois(rois)
        self._edited(rois)

    def _table_edit(self, _):
        if self.guard or self.key is None:
            return
        rois = self.table.value.to_dict("records")
        try:
            for roi in rois:
                roi["name"] = str(roi["name"]).strip()
            roi_slices(rois, self.shape)
        except (ValueError, TypeError) as exc:
            self.table.value = pd.DataFrame(self._rois(), columns=FIELDS)
            self.status.object = str(exc)
            return
        self._set_rois(rois)
        self._edited(rois)

    def _edited(self, rois):
        if self.job:
            self.job["cancel"].set()
        uid, stream, field = self.key
        try:
            ScanCache(uid).write_rois(stream, field, rois, self.shape, {})
        except Exception:
            log.exception("ROI edit cache failed")
            self.status.object = "Could not save ROI edits; check cache storage."
            return
        self.refresh(uid, stream)
        self.progress.value = 0
        self.status.object = f"**{len(rois)} ROIs** — definitions changed; compute to update scalars."

    def _clear(self, _):
        if self.key is not None:
            self._set_rois([])
            self._edited([])

    def _cancel(self, _):
        if self.job:
            self.job["cancel"].set()
            self.status.object = "Cancelling after the current frame read…"

    def _compute(self, _):
        ctx = self.context()
        if self.job or ctx is None or self.key != (ctx["uid"], ctx["stream"], ctx["field"]):
            self.status.object = "Select a single scalar detector image first."
            return
        rois = self._rois()
        try:
            if not rois:
                raise ValueError("Draw at least one ROI first.")
            roi_slices(rois, self.shape)
        except ValueError as exc:
            self.status.object = str(exc)
            return
        product = ScanCache(self.key[0]).read_rois(self.key[1]).get(self.key[2], {})
        if (product.get("rois") == rois and tuple(product.get("shape", ())) == self.shape
                and product.get("columns") and not product.get("failed")
                and all(len(v) == ctx["n_frames"] for v in product["columns"].values())):
            self.refresh(self.key[0], self.key[1])
            self.progress.value = 100
            self.status.object = "**Cached ROI scalars loaded** — no frame reads needed."
            return
        doc = pn.state.curdoc
        if doc is None:
            self.status.object = "ROI computation requires a served Panel session."
            return
        key, shape = self.key, self.shape
        job = {"cancel": threading.Event()}
        self.job = job
        self.compute.disabled = True
        self.cancel.disabled = False
        self.progress.value = 0
        self.status.object = f"Computing {len(rois)} ROIs across {ctx['n_frames']} frames…"

        def dispatch(callback):
            if not doc.session_context or not doc.session_context.destroyed:
                doc.add_next_tick_callback(callback)

        def report(done, total):
            if self.job is job and self.key == key and not job["cancel"].is_set():
                self.progress.value = int(100 * done / max(1, total))
                self.status.object = f"Computing ROI scalars: **{done} / {total}** frames."

        last = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if now - last[0] >= .25 or done == total:
                last[0] = now
                dispatch(lambda: report(done, total))

        def finish(columns=None, failed=(), error=None):
            if self.job is not job:
                return
            self.job = None
            self.compute.disabled = self.key is None
            self.cancel.disabled = True
            if self.key != key:
                return
            if job["cancel"].is_set():
                self.status.object = "Cancelled. Cached frames are retained; compute again to retry."
                return
            if error:
                self.status.object = f"ROI computation failed: {error}"
                return
            try:
                ScanCache(key[0]).write_rois(key[1], key[2], rois, shape, columns, failed)
                self.refresh(key[0], key[1])
            except Exception as exc:
                log.exception("ROI publish failed")
                self.status.object = f"Could not publish ROI scalars: {exc}"
                return
            self.progress.value = 100
            note = f" **{len(failed)} unreadable frames are NaN**; compute again to retry." if failed else ""
            self.status.object = f"**Done** — {len(columns)} scalar columns available in Scalars / Primary.{note}"

        def worker():
            try:
                columns, failed = compute_roi_scalars(
                    rois, shape, ctx["n_frames"], ctx["read_frame"],
                    cancel=job["cancel"], progress=progress,
                )
            except CancelledError:
                dispatch(lambda: finish())
            except Exception as exc:
                log.exception("ROI computation failed")
                dispatch(lambda error=str(exc): finish(error=error))
            else:
                dispatch(lambda: finish(columns, failed))

        doc.on_session_destroyed(lambda _: job["cancel"].set())
        threading.Thread(target=worker, daemon=True, name="roi-scalars").start()
