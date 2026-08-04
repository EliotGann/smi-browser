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
