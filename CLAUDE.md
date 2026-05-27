# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (requires pixi: https://pixi.sh/)
pixi install

# Launch the Panel app with auto-reload
pixi run serve        # opens http://localhost:5006/smi_app

# Quick import sanity check (no server required)
pixi run check

# Run the full test suite
pixi run test

# Run a single test file or test function
pixi run python -m pytest tests/test_cache.py -q
pixi run python -m pytest tests/test_smoke.py::test_appstate_defaults -q
```

## Architecture

### Entry point: `smi_app.py`

The entire Panel application lives in `smi_app.py`. It must be served with `panel serve`; it cannot be run directly. UI sub-modules (`smi_browser/ui/`) are imported **after** `pn.extension()` so that the Tabulator JS extension registers correctly.

Two monkey-patches are applied at startup and must remain near the top of `smi_app.py`:
- **Bokeh property descriptor patch** — BokehJS 3.9 sends visual properties as `{"type": "value", "value": X}`; the patch unwraps these silently to prevent noisy `ValueError`s on every PATCH-DOC round-trip.
- **Tiled `Container` sort patch** (also duplicated in `smi_browser/_tiled.py`) — the SMI tiled server bakes an empty sort string (`sort=''`) into catalog containers; newer tiled servers reject it with 422. The patch clears `_sorting_params` and `_reversed_sorting_params` globally.

### `smi_browser/` package

| Module | Responsibility |
|---|---|
| `state.py` | `AppState` — single dataclass holding all mutable app state; UI modules receive it by argument, no module globals |
| `config.py` | Constants and defaults, mostly re-exported from `PyHyperScattering.smi_defaults` |
| `processing.py` | `build_proc_params()` — pure param builder; returns `(reduce_fn_name, kwargs)` for `reduce_smi_combined` or `reduce_smi_gi` |
| `cache.py` | `ScanCache` — per-UID HDF5 disk cache for scalars, raw image stacks, and reduction outputs; thread-safe with per-file locks and LRU eviction |
| `export.py` | Writes PNG figures and HDF5 files to proposal working directories (uses `nsls2api.py` for path resolution) |
| `nsls2api.py` | Unauthenticated REST calls to `api.nsls2.bnl.gov` for proposal/data-session lookup |
| `_batch.py` | `BatchProcessor` — bounded thread-pool queue runner; decoupled from Panel/Bokeh for testability |
| `_stream.py` | `LiveStreamManager` — wraps tiled's streaming subscription API; callbacks fire on a background thread and must be marshalled to Bokeh's document thread via a `dispatcher` |
| `models/collection.py` | `ScanCollection` — holds processed results for I(q) comparison and xarray parameter-sweep stacking |
| `models/summary.py` | `enhanced_summary()` — merges tiled metadata into a flat dict for display |
| `data/scalars.py`, `data/frames.py`, `data/masks.py` | Pure helpers for scalar DataFrames, image orientation, and mask polygon conversion |
| `figures/` | Bokeh figure builders (image viewer, multiview grid, I(q) cuts, 2D process map) |
| `ui/` | Panel widget wiring for each tab (auth, batch, collection, live) |

### Top-level standalone modules

`tiled_browser.py`, `batch_processor.py`, and `live_stream.py` are kept at the top level for **backward compatibility** only. The canonical implementations now live under `smi_browser/` (`_tiled.py`, `_batch.py`, `_stream.py`). New code should import from the package.

### Tiled access pattern — stay lazy

`tiled_browser.py` / `smi_browser/_tiled.py` enforces a strict rule: **never call `.read()` during search or browsing**. Catalog nodes are kept as lazy references. The `fetch_page_fast()` function bypasses the Python client for pagination and calls the REST endpoint directly (one HTTP round-trip) to retrieve metadata for an entire page. Data arrays are fetched only when a specific scan is selected and a detail tab is opened.

### Data reduction

All reduction goes through `PyHyperScattering` (local editable install, see `pixi.toml`). `processing.build_proc_params()` constructs the kwargs and returns the function name (`reduce_smi_combined` or `reduce_smi_gi`). The caller imports the actual callable. Calibrated defaults (beam-centre deltas, mask filenames, detector names) flow from `PyHyperScattering.smi_defaults` → `smi_browser/config.py`.

### Disk cache

`ScanCache` stores per-scan HDF5 files under `$SMI_BROWSER_CACHE_DIR` (default `$TMPDIR/smi_browser_cache`). Capacity is capped at `$SMI_BROWSER_CACHE_MAX_GB` (default 50 GB) with LRU eviction. Image datasets use per-frame chunking for cheap random-access reads.

### Threading model

Panel/Bokeh is single-threaded on the document. Long-running work (reduction, batch processing, tiled streaming) runs on background threads. Mutations to Bokeh document objects must be marshalled back via `doc.add_next_tick_callback`. `AppState.cancel` is a `threading.Event` used as the cancellation flag for in-flight operations.
