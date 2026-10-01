"""Focused smoke tests for smi_app stream-selection helpers."""
from __future__ import annotations

import importlib
import site
import sys
import threading


class _NoopThread:
    """Prevent smi_app's startup search thread from running during import."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def _import_smi_app(monkeypatch):
    # Match the repo's pixi test task (PYTHONNOUSERSITE=1) even when this file
    # is run directly via ``python -m pytest``.
    user_site = site.getusersitepackages()
    sys.path[:] = [p for p in sys.path if p != user_site]
    monkeypatch.setattr(threading, "Thread", _NoopThread)
    return importlib.import_module("smi_app")


def test_stream_names_exclude_baseline_and_keep_primary_first(monkeypatch):
    app = _import_smi_app(monkeypatch)

    monkeypatch.setattr(
        app.tb,
        "stream_names",
        lambda _run: ["arc20", "baseline", "primary", "arc0"],
    )

    assert app._data_stream_names(object()) == ["primary", "arc0", "arc20"]


def test_populate_stream_select_preserves_valid_selection(monkeypatch):
    app = _import_smi_app(monkeypatch)

    monkeypatch.setattr(
        app.tb,
        "stream_names",
        lambda _run: ["primary", "arc0", "arc20", "baseline"],
    )
    app._detail_cache["stream"] = "arc20"

    app._populate_stream_select(object())

    assert app._active_stream() == "arc20"
    assert app.w_stream_select.value == "arc20"
    assert app.w_stream_select.visible is True


def test_populate_stream_select_hides_single_stream(monkeypatch):
    app = _import_smi_app(monkeypatch)

    monkeypatch.setattr(app.tb, "stream_names", lambda _run: ["primary", "baseline"])
    app._detail_cache["stream"] = "arc20"

    app._populate_stream_select(object())

    app._refresh_primary_tab_label()

    assert app._active_stream() == "primary"
    assert app.w_stream_select.visible is False
    assert app._primary_tab_name == "Primary"


def test_roi_products_populate_existing_scalar_selectors(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from smi_browser.cache import ScanCache

    app = _import_smi_app(monkeypatch)
    monkeypatch.setenv("SMI_BROWSER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_selected_uid", lambda: "roi-scan")
    monkeypatch.setattr(app, "_selected_uids", lambda: ["roi-scan"])
    monkeypatch.setattr(app, "_ensure_run", lambda: SimpleNamespace(metadata={"start": {}}))
    monkeypatch.setattr(app.tb, "stream_names", lambda _: ["arc20"])
    monkeypatch.setattr(app.tb, "stream_info_for", lambda *a: {"dataset": None})
    monkeypatch.setattr(app.tb, "fetch_scalars", lambda *a, **kw: {"motor": np.arange(3)})
    monkeypatch.setattr(app, "_refresh_export_labels", lambda: None)
    monkeypatch.setattr(app, "_refresh_export_resolved_path", lambda: None)
    app._detail_cache.update(stream="arc20", primary_loaded=False)
    cache = ScanCache("roi-scan")
    cache.write_rois("arc20", "det", [], (2, 2), {"ROI1:sum": [4, 8, 12]})
    app._load_primary()
    column = "roi:det:ROI1:sum"
    for widget in (app.w_primary_y, app.w_explore_y, app.w_primary_2d_z, app.w_explore_2d_z):
        assert column in widget.options
    np.testing.assert_allclose(app.w_primary_table.value[column], [4, 8, 12])
    cache.write_rois("arc20", "det", [], (2, 2), {})
    app._refresh_roi_scalars("roi-scan", "arc20")
    assert column not in app.w_explore_y.options
    assert column not in app.w_primary_table.value


def test_calibrant_switch_updates_both_panels_and_invalidates_fits(monkeypatch):
    import pytest
    app = _import_smi_app(monkeypatch)
    for slot in (app._CAL_SAXS, app._CAL_WAXS):
        slot["last_fit_px"] = slot["last_multi_fit_px"] = object()
        slot["apply_btn"].disabled = slot["multi_apply_btn"].disabled = False
        slot["calibrant"].value = "gold"
        assert slot["ring"].value == 1
        assert slot["multi_peaks"].value == [1, 2]
        assert (slot["q_min"].value + slot["q_max"].value) / 2 == pytest.approx(26.68)
        assert slot["last_fit_px"] is slot["last_multi_fit_px"] is None
        assert slot["apply_btn"].disabled and slot["multi_apply_btn"].disabled
        slot["ring"].value = 2
        assert (slot["q_min"].value + slot["q_max"].value) / 2 == pytest.approx(30.81)
        slot["calibrant"].value = "silver_behenate"


def test_calibration_picker_snaps_gold_and_preserves_window(monkeypatch):
    from types import SimpleNamespace
    app = _import_smi_app(monkeypatch)
    slot = app._CAL_WAXS
    slot["calibrant"].value = "gold"
    slot["pick_btn"].value = True
    app._on_qchi_tap(SimpleNamespace(x=30.4))
    app._on_qchi_tap(SimpleNamespace(x=31.0))
    assert slot["ring"].value == 2
    assert (slot["q_min"].value, slot["q_max"].value) == (30.4, 31.0)
    assert "200" in slot["status"].object
    assert not slot["pick_btn"].value
    slot["calibrant"].value = "silver_behenate"


def test_single_gold_fit_uses_selected_reference(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import pytest
    app = _import_smi_app(monkeypatch)
    slot = app._CAL_WAXS
    slot["calibrant"].value = "gold"
    slot["ring"].value = 2
    monkeypatch.setattr(app, "_current_qchi_for_detector", lambda _: object())
    monkeypatch.setattr(app, "fit_ring_peaks", lambda *a, **kw: SimpleNamespace(
        chi_deg=np.linspace(-180, 180, 36), q_peak=np.full(36, 30.81), n_accepted=36, n_total=36))
    monkeypatch.setattr(app, "_draw_calibrate_overlay", lambda *a: None)
    app._on_calibrate_fit(slot)(None)
    assert slot["last_fit_px"].ring_q_expected == pytest.approx(30.81)
    assert slot["last_fit_px"].ddist_mm == pytest.approx(0, abs=1e-9)
    assert "Gold" in slot["status"].object and "200" in slot["status"].object
    slot["calibrant"].value = "silver_behenate"
