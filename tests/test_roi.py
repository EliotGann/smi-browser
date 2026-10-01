"""ROI numerical correctness, bounded reads, persistence and UI lifecycle."""
from concurrent.futures import CancelledError
import threading

import numpy as np
import pandas as pd
import pytest

from smi_browser.cache import ScanCache, get_or_fetch_image_frame
from smi_browser.models.roi import compute_roi_scalars, merge_roi_scalars, roi_slices


def rect(name="ROI1", x=1, y=1, width=2, height=2):
    return dict(name=name, x=x, y=y, width=width, height=height)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMI_BROWSER_CACHE_DIR", str(tmp_path))


def test_all_rois_one_read_per_frame_and_original_values():
    frames = [np.array([[1, 2, 30], [3, 4, 60]]), np.full((2, 3), 2**32, dtype=np.uint64)]
    calls = []

    def read(i):
        calls.append(i)
        return frames[i]

    out, failed = compute_roi_scalars([rect(), rect("right", x=2.5, width=1)], (2, 3), 2, read)
    assert calls == [0, 1]
    assert not failed
    np.testing.assert_allclose(out["ROI1:sum"], [10, 4 * 2**32])
    np.testing.assert_allclose(out["ROI1:mean"], [2.5, 2**32])
    np.testing.assert_allclose(out["ROI1:std"], [np.std([1, 2, 3, 4]), 0])
    np.testing.assert_allclose(out["right:sum"], [90, 2 * 2**32])


def test_pixel_centres_clip_and_empty_validation():
    a = np.arange(12).reshape(3, 4)
    # Bounds [-1, 1) select only pixel centre 0.5, on both axes.
    assert a[roi_slices([rect(x=0, y=0)], a.shape)[0]].tolist() == [[0]]
    # Centre on lower edge is included, centre on upper edge is excluded.
    assert a[roi_slices([rect(x=1, y=1, width=1, height=1)], a.shape)[0]].tolist() == [[0]]
    for roi in [rect(x=20), rect(width=0), rect(x=float("nan"))]:
        with pytest.raises(ValueError):
            roi_slices([roi], a.shape)
    with pytest.raises(ValueError, match="unique"):
        roi_slices([rect(), rect()], a.shape)


def test_nonfinite_pixels_missing_frames_and_shape_mismatch():
    frames = [np.array([[np.nan, np.inf], [-np.inf, -2]]), np.full((2, 2), np.nan), None, np.zeros((3, 3))]
    out, failed = compute_roi_scalars([rect()], (2, 2), 4, frames.__getitem__)
    assert failed == [2, 3]
    np.testing.assert_allclose(out["ROI1:count"], [1, 0, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(out["ROI1:sum"], [-2, np.nan, np.nan, np.nan], equal_nan=True)


def test_cancel_stops_further_reads():
    cancel = threading.Event()
    calls = []

    def read(i):
        calls.append(i)
        cancel.set()
        return np.ones((2, 2))

    with pytest.raises(CancelledError):
        compute_roi_scalars([rect()], (2, 2), 100, read, cancel=cancel)
    assert calls == [0]


def test_orientation_matches_display():
    from smi_browser.data.frames import orient_frame
    raw = np.arange(20).reshape(4, 5)
    displayed = orient_frame(raw, "pil900KW_image")
    out, _ = compute_roi_scalars([rect(x=.5, y=.5, width=1, height=1)], displayed.shape,
                                 1, lambda _: orient_frame(raw, "pil900KW_image"))
    assert out["ROI1:sum"][0] == displayed[0, 0]


def test_frame_cache_reuse_and_stream_isolation():
    calls = []

    def fetch(i):
        calls.append(i)
        return np.full((2, 2), i)

    def run(stream):
        return compute_roi_scalars([rect()], (2, 2), 3, lambda i: get_or_fetch_image_frame(
            "scan", "det", i, fetch_one_fn=fetch, n_frames=3, stream=stream))

    run("arc0")
    run("arc0")
    assert calls == [0, 1, 2]
    run("arc20")
    assert calls == [0, 1, 2, 0, 1, 2]


def test_product_cache_scoping_and_invalidation():
    c = ScanCache("scan")
    c.write_scalars("arc/0", {"motor": np.arange(3)})
    for stream, field, value in [("arc/0", "a/b", 1), ("arc_0", "a/b", 2), ("arc/0", "a_b", 3)]:
        c.write_rois(stream, field, [rect()], (2, 2), {"ROI1:sum": [value, value]}, [1])
    assert c.read_rois("arc/0")["a/b"]["columns"]["ROI1:sum"][0] == 1
    assert c.read_rois("arc_0")["a/b"]["columns"]["ROI1:sum"][0] == 2
    assert c.read_rois("arc/0")["a_b"]["columns"]["ROI1:sum"][0] == 3
    c.write_rois("arc/0", "a/b", [rect(x=2)], (2, 2), {})
    assert c.read_rois("arc/0")["a/b"]["columns"] == {}
    assert c.read_rois("arc/0")["a/b"]["rois"][0]["x"] == 2
    np.testing.assert_array_equal(c.read_scalars("arc/0")["motor"], np.arange(3))


def test_merge_preserves_frame_alignment_and_removes_stale_columns():
    df = pd.DataFrame({"motor": [20, 10, 30], "roi:old:r:sum": [1, 1, 1]}).sort_values("motor")
    out = merge_roi_scalars(df, {"det": {"columns": {"R:sum": [5, 6]}}})
    assert "roi:old:r:sum" not in out
    np.testing.assert_allclose(out["roi:det:R:sum"], [5, 6, np.nan], equal_nan=True)
    assert out["motor"].tolist() == [20, 10, 30]
    empty = merge_roi_scalars(pd.DataFrame(), {"det": {"columns": {"R:sum": [5, 6]}}})
    assert empty["frame"].tolist() == [0, 1]


def test_ui_drawing_edit_invalidation_and_restore():
    import panel as pn
    pn.extension("tabulator")
    from bokeh.plotting import figure
    from smi_browser.ui.roi import ROIControls

    ctx = dict(uid="scan", stream="arc0", field="det", n_frames=2)
    refreshed = []
    ui = ROIControls(lambda: ctx, lambda *key: refreshed.append(key))
    fig = figure()
    ui.attach(fig, (4, 4))
    ui.source.data = dict(name=[""], x=[1.], y=[1.], width=[2.], height=[2.])
    assert ui.table.value["name"].tolist() == ["ROI1"]
    assert refreshed == [("scan", "arc0")]
    ScanCache("scan").write_rois("arc0", "det", [rect()], (4, 4), {"ROI1:sum": [1, 2]})
    ui.source.data = dict(name=["ROI1"], x=[2.], y=[1.], width=[2.], height=[2.])
    assert ScanCache("scan").read_rois("arc0")["det"]["columns"] == {}
    ui.deactivate()
    ui.attach(fig, (4, 4))
    assert ui.source.data["x"] == [2.]
    ui.draw.value = True
    assert fig.toolbar.active_drag is ui.tool
    ctx["stream"] = "arc20"
    ui.attach(fig, (4, 4))
    assert ui.source.data["name"] == []


@pytest.mark.parametrize("interrupt", [None, "edit", "navigate"])
def test_background_publication_is_on_document_thread_and_discards_stale(monkeypatch, interrupt):
    import panel as pn
    from bokeh.document import Document
    from bokeh.plotting import figure
    from smi_browser.ui.roi import ROIControls
    import smi_browser.ui.roi as ui_mod

    doc = Document()
    monkeypatch.setattr(pn.state, "curdoc", doc)
    workers = []

    class DeferredThread:
        def __init__(self, target, **_):
            self.target = target

        def start(self):
            workers.append(self.target)

    monkeypatch.setattr(ui_mod.threading, "Thread", DeferredThread)
    ctx = dict(uid="scan", stream="arc0", field="det", n_frames=3,
               read_frame=lambda i: np.full((2, 2), i + 1))
    refreshed = []
    ui = ROIControls(lambda: ctx, lambda *key: refreshed.append(key))
    ui.attach(figure(), (2, 2))
    ui.source.data = {k: [v] for k, v in rect().items()}
    refreshed.clear()
    ui._compute(None)
    assert ui.compute.disabled
    workers.pop()()
    # Worker completion only queues updates: nothing is published yet.
    assert not ScanCache("scan").read_rois("arc0")["det"]["columns"]
    assert refreshed == []
    if interrupt == "edit":
        ui.source.data = {k: [v] for k, v in rect(width=1).items()}
    elif interrupt == "navigate":
        ui.deactivate()
        ctx["stream"] = "arc20"
        ui.attach(figure(), (2, 2))
    for callback in list(doc.session_callbacks):
        callback.callback()
    product = ScanCache("scan").read_rois("arc0")["det"]
    if interrupt:
        assert not product["columns"]
    else:
        np.testing.assert_allclose(product["columns"]["ROI1:sum"], [4, 8, 12])
        assert refreshed == [("scan", "arc0")]
        # A repeat compute uses the derived cache, not even the frame cache.
        ui._compute(None)
        assert not workers
        assert "no frame reads" in ui.status.object
    assert ui.job is None
    assert not ui.compute.disabled
