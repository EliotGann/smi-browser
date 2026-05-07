"""Smoke test — verify the package imports and key modules are reachable."""
from __future__ import annotations


def test_package_imports():
    import smi_browser
    from smi_browser.config import PAGE_SIZE, RESULT_COLS
    from smi_browser.state import AppState
    from smi_browser.models.collection import ScanCollection
    from smi_browser.models.summary import enhanced_summary
    from smi_browser.data.scalars import scalars_to_dataframe
    from smi_browser.data.frames import detector_for_field, orient_frame
    from smi_browser.data.masks import normalized_mask_to_xs_ys, xs_ys_to_normalized_mask
    from smi_browser._batch import BatchProcessor
    from smi_browser._stream import LiveStreamManager


def test_appstate_defaults():
    from smi_browser.state import AppState
    app = AppState()
    assert app.cat is None
    assert len(app.collection) == 0
    assert app.search["page"] == 0
    assert app.detail_cache["uid"] is None
    assert app.image_cache["field"] is None
    assert app.live["active"] is False
    assert not app.cancel.is_set()


def test_backward_compat_imports():
    """Top-level modules still importable for backward compatibility."""
    import tiled_browser
    import batch_processor
    import live_stream
