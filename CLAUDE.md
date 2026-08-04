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

### UI layout design standard

The detail tabs follow a consistent **controls-left / visuals-right** convention. Keep new tabs and panels consistent with it:

- **Two-column split.** Wrap a tab's body in a `pn.Row`: a fixed-width **left control column** (`width=460`, `scroll=True`, `sizing_mode="stretch_height"`) and a **right visual column** (`sizing_mode="stretch_both"`) holding the figure(s). The image/plot gets the full viewport height and all remaining width; tall control content scrolls inside the left column instead of pushing the visual down. (Reference: Explore tab and Process → 2D in `smi_app.py`.)
- **Controls stack vertically.** Inside the left column, lay widgets out top-to-bottom. Use `pn.Column` for option groups (one widget per line); only pair widgets `pn.Row`-wise when they're naturally a unit, and never put **more than 2 across**. Checkboxes are always stacked one-per-line.
- **Group controls into sub-tabs, not stacked collapsible cards.** When a tab has several control groups (e.g. Explore's Scalars / Color scale / Mask / Alignment), put each group in its own `pn.Tabs` tab rather than a vertical pile of `pn.Card(collapsed=...)`. Reserve open `pn.Card` tiles for dense parameter sets laid out 3-across (Process → Parameters).
- **Everything is width-responsive.** Prefer `sizing_mode="stretch_width"` over fixed `width=` on text inputs, progress bars, and tables so content compresses to the available space and never forces horizontal scrolling. Tabulators use `layout="fit_columns"`.
- **Dynamic tab labels** (e.g. `Collection (3)`) are set by reassigning the tab tuple (`tabs[i] = (name, obj)`) only when the label actually changes, to avoid re-rendering stateful content.
- **Collapse/show panels with `.visible`,** never by reassigning `*.objects` — the latter tears down and rebuilds heavy widgets (Tabulator) and is visibly slow. `.visible=False` drops a child out of the flex row instantly and the sibling reclaims the space.

### `smi_browser/` package

| Module | Responsibility |
|---|---|
| `state.py` | `AppState` — single dataclass holding all mutable app state; UI modules receive it by argument, no module globals |
| `config.py` | Constants and defaults, mostly re-exported from `smi_tiled.defaults` |
| `processing.py` | `build_proc_params()` — pure param builder; returns `(reduce_fn_name, kwargs)` for `reduce_smi_combined` or `reduce_smi_gi` |
| `cache.py` | `ScanCache` — per-UID HDF5 disk cache for scalars, raw image stacks, reduction outputs, and per-peak fit results (`/peakfit`); also persists the global peak-definition list (`peak_defs.json`). Thread-safe with per-file locks and LRU eviction |
| `export.py` | Writes PNG figures and HDF5 files to proposal working directories. The HDF5 writer is **section-gated** (`h5_sections`): metadata / primary / baseline+config / processed I(q) on by default; raw images, processed q-χ, peak fits opt-in. Also emits per-peak result PNGs (uses `nsls2api.py` for path resolution) |
| `nsls2api.py` | Unauthenticated REST calls to `api.nsls2.bnl.gov` for proposal/data-session lookup |
| `_batch.py` | `BatchProcessor` — bounded thread-pool queue runner; decoupled from Panel/Bokeh for testability |
| `_stream.py` | `LiveStreamManager` — wraps tiled's streaming subscription API; callbacks fire on a background thread and must be marshalled to Bokeh's document thread via a `dispatcher` |
| `models/collection.py` | `ScanCollection` — holds processed results for I(q) comparison and xarray parameter-sweep stacking |
| `models/peakfit.py` | `fit_peak_across_frames()` — per-frame Gaussian/Lorentzian peak fitting across a 1-D I(q) stack; bounded width + SNR/R² gating + per-peak link modes (independent / linked / tracked). Panel/Bokeh-free, unit-tested |
| `models/summary.py` | `enhanced_summary()` — merges tiled metadata into a flat dict for display |
| `data/scalars.py`, `data/frames.py`, `data/masks.py` | Pure helpers for scalar DataFrames, image orientation, and mask polygon conversion. `data/scalars.py` also derives **virtual axes** (`derive_virtual_columns` / `parse_label_number_tokens`) from structured per-frame string fields |
| `figures/` | Bokeh figure builders (image viewer, multiview grid, I(q) cuts, 2D process map) |
| `ui/` | Panel widget wiring for each tab (auth, batch, collection, live) |

### Top-level standalone modules

`tiled_browser.py`, `batch_processor.py`, and `live_stream.py` are kept at the top level for **backward compatibility** only. The canonical implementations now live under `smi_browser/` (`_tiled.py`, `_batch.py`, `_stream.py`). New code should import from the package.

### Tiled access pattern — stay lazy

`tiled_browser.py` / `smi_browser/_tiled.py` enforces a strict rule: **never call `.read()` during search or browsing**. Catalog nodes are kept as lazy references. The `fetch_page_fast()` function bypasses the Python client for pagination and calls the REST endpoint directly (one HTTP round-trip) to retrieve metadata for an entire page. Data arrays are fetched only when a specific scan is selected and a detail tab is opened.

### Multi-stream display (per-arc GIWAXS)

Most runs have a single `primary` event stream, but arc-economy GIWAXS runs (`giwaxs_bar_arc_economy` in `smi-plans`) split data into **one stream per WAXS arc** (`arc0`, `arc20`, `arcm1p5`, …), each with its own detector set (low arc = WAXS-only, high arc = SAXS+WAXS). A **`Stream` dropdown** on the Primary tab (`w_stream_select`, hidden unless a run exposes >1 non-`baseline` stream) selects which stream the **Primary scalars, Explore image viewer, and Grid** target; the Primary tab relabels to e.g. `Primary (arc20)`. The selection lives in `_detail_cache["stream"]` and is read via `_active_stream()`; the low-level `_tiled.py` fetchers and the disk cache are already stream-parameterized. **Reduction, Export, Collection, and live-following remain `primary`-only** for now — see `docs/per_arc_stream_reduction_plan.md` for the `smi-tiled` work needed to reduce non-`primary` streams.

### Data reduction

All reduction goes through `smi-tiled` (local editable install at `../smi-tiled`, see `pixi.toml`). `processing.build_proc_params()` constructs the kwargs and returns the function name (`reduce_smi_combined` or `reduce_smi_gi`). The caller imports the actual callable from `smi_tiled` (both are top-level re-exports). Calibrated defaults (beam-centre deltas, mask filenames, detector names) flow from `smi_tiled.defaults` → `smi_browser/config.py`.

### Disk cache

`ScanCache` stores per-scan HDF5 files under `$SMI_BROWSER_CACHE_DIR` (default `$TMPDIR/smi_browser_cache`). Capacity is capped at `$SMI_BROWSER_CACHE_MAX_GB` (default 50 GB) with LRU eviction. Image datasets use per-frame chunking for cheap random-access reads.

Scalar and image caches are **stream-aware**. Scalars live under `/<stream>` (`/primary`, `/baseline`, or a sanitized custom name). Images use the legacy `images/<field>` path for the `primary` stream (back-compat) and `images/<stream>/<field>` (plus a parallel `images_filled/<stream>/<field>` mask) for any other stream, so arc-economy GIWAXS runs whose per-arc streams (`arc0`/`arc20`/…) share a detector field name don't clobber each other. The path convention is centralized in `cache._image_path` / `_image_fill_path`. Reduction (`/reduction`) and peak-fit (`/peakfit`) outputs remain per-UID with no stream dimension (reduction is still `primary`-only — see `docs/per_arc_stream_reduction_plan.md`).

Per-frame I(q) (`pf_iq_I` / `pf_iq_q`) is written into `/reduction` at reduction time, so the Peak Map tab and exports reuse it across restarts with no recompute. Peak-fit result maps are stored under `/peakfit/<hash>` (keyed by `PeakDef.key()`) and are **invalidated whenever `write_reduction` runs** (re-processing overwrites `pf_iq`). The drawn peak list itself persists globally in `peak_defs.json` under the cache root. **Caveat:** the default cache dir lives under `$TMPDIR`, which survives a browser restart but may be wiped on reboot — set `$SMI_BROWSER_CACHE_DIR` to a persistent path to keep cached results permanently.

> **HDF5 byte strings:** string columns cached via h5py read back as `bytes`. `_scalars_to_dataframe` (in `smi_app.py`) decodes them to `str` so the Tabulator doesn't render `[object ArrayBuffer]` and so virtual-axis parsing works on the cached path as well as fresh-from-tiled.

### Peak fitting & the Peak Map tab

`models/peakfit.py` fits one peak per drawn q-range to every frame's 1-D I(q), producing `amplitude` / `center` / `fwhm` / `area` maps. Robustness rules: the fitted FWHM is bounded by the drawn range, and an SNR/R² gate rejects no-peak frames (reporting amplitude/area `0`, centre/FWHM `NaN`) rather than letting the fit run away. Each peak has a `link` mode — `independent`, `linked` (centre & width shared from a robust aggregate fit; only amplitude varies per frame, via a fast `lstsq`), or `tracked` (warm-start from the previous frame). Fits run on a background thread, are cached per `(uid, PeakDef.key())`, and the result map plots against any per-frame axis.

### Virtual primary axes (filename-derived)

Some quantities (e.g. grazing-incidence angle) are encoded only inside per-frame string fields like `target_file_name` (`..._ai0.50_wa9_degC100.0`). `data/scalars.py::derive_virtual_columns` parses every `label`+`number`(+`unit`) token from such columns into numeric `fn:`-prefixed columns (`fn:ai`, `fn:eV`, …; `NaN` where a token is absent). It is called once inside `smi_app.py::_scalars_to_dataframe`, so the derived axes flow automatically into every axis selector (Primary, Explore, Process 2D map, Peak Map) and into exports — no reduction needed.

### Threading model

Panel/Bokeh is single-threaded on the document. Long-running work (reduction, batch processing, tiled streaming) runs on background threads. Mutations to Bokeh document objects must be marshalled back via `doc.add_next_tick_callback`. `AppState.cancel` is a `threading.Event` used as the cancellation flag for in-flight operations.
