"""Centralised application state for the SMI Browser.

All mutable module-level state dicts that were previously scattered across
``smi_app.py`` are consolidated into a single ``AppState`` dataclass.  UI
modules receive the instance via function arguments rather than relying on
module globals.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .config import PAGE_SIZE
from .models.collection import ScanCollection


@dataclass
class AppState:
    """Single mutable container for all application-level state."""

    # --- Tiled catalog ---
    cat: Any = None

    # --- Scan collection ---
    collection: ScanCollection = field(default_factory=ScanCollection)

    # --- Search / pagination ---
    search: dict = field(default_factory=lambda: {
        "unified_filters": [],
        "page": 0,
        "page_size": PAGE_SIZE,
        "total": 0,
        "selected_uid": None,
    })

    # --- Detail cache (per-selected-uid) ---
    detail_cache: dict = field(default_factory=lambda: {
        "uid": None,
        "run": None,
        "summary": None,
        "primary_loaded": False,
        "baseline_loaded": False,
        "images_loaded": False,
        "primary_info": None,
        "primary_dataset": None,
    })

    # --- Last processing result ---
    last_result: dict = field(default_factory=lambda: {
        "result": None,
        "params": None,
    })

    # --- Filter rows (search UI) ---
    filter_rows: list = field(default_factory=list)

    # --- Cancellation flag ---
    cancel: threading.Event = field(default_factory=threading.Event)

    # --- Live count debounce timer ---
    live_count_timer: Any = None

    # --- Image viewer cache ---
    image_cache: dict = field(default_factory=lambda: {
        "field": None, "n_frames": 0, "dataset": None, "fields": [],
        "figure": None, "source": None, "mapper": None,
        "mask_source": None, "mask_renderer": None,
        "draw_tool": None, "edit_tool": None,
        "mask_image_shape": None,
    })

    # --- Multi-view grid cache ---
    multiview_cache: dict = field(default_factory=lambda: {
        "uid": None, "field": None, "n_frames": 0,
        "frames": None, "renderers": None, "mapper": None,
        "log": None, "data_lo": None, "data_hi": None,
        "suspend_range_cb": False,
    })

    # --- Process result cache ---
    proc_result_cache: dict = field(default_factory=lambda: {
        "result": None,
        "gi_result": None,
    })

    # --- Processing guard ---
    processing_guard: dict = field(default_factory=lambda: {
        "active": False,
    })

    # --- Persisted cross-section cuts ---
    persisted_cuts: list = field(default_factory=list)

    # --- Process 2D plot cache ---
    proc_2d_cache: dict = field(default_factory=lambda: {
        "x": None, "y": None, "image": None,
        "x_label": "", "y_label": "",
        "title": "",
        "cuts_source": None,
        "cut_renderer": None,
    })

    # --- Cuts recursion guard ---
    cuts_guard: dict = field(default_factory=lambda: {
        "in_progress": False,
    })

    # --- Explore cursor ---
    explore_cursor_source: Any = None

    # --- Batch processing ---
    batch_state: dict = field(default_factory=lambda: {
        "doc": None,
        "processor": None,
    })

    # --- Live mode ---
    live: dict = field(default_factory=lambda: {
        "manager": None,
        "active": False,
        "saved": {},
        "doc": None,
    })
