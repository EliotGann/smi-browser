# SMI Tiled Browser

Interactive web application for browsing and processing data from the
[SMI beamline](https://www.bnl.gov/nsls2/beamlines/beamline.php?r=12-ID)
at NSLS-II, served from the [Tiled](https://blueskyproject.io/tiled/) catalog.

Built with [Panel](https://panel.holoviz.org/) and
[Bokeh](https://docs.bokeh.org/) for interactive visualization, and
[smi-tiled](https://github.com/EliotGann/smi-tiled) for data reduction.

## Features

- **Search & filter** — Stackable filters (text match, exact match, fulltext)
  against the `smi/migration` catalog (~2.7M scans). Results are shown
  newest-first with pagination.
- **Metadata** — Full run metadata displayed as collapsible JSON tree.
- **Primary** — Scalar data table with configurable X/Y line plots.
- **Baseline** — Before/after baseline readings with column filtering.
- **Explore** — Side-by-side linked 1D plot and 2D detector image with a
  synced frame cursor. The **ROIs** sub-tab lets you draw/name multiple rectangles
  and explicitly **Compute ROI scalars** across the selected detector's frames.
  Sum, mean, population std, min, max and finite-pixel count appear in the existing
  Scalars 1D/2D selectors and Primary table as `roi:detector:ROI:stat` columns.
  Computation runs in the background with progress/cancellation, processes all
  ROIs in one frame-by-frame pass, and reuses the stream-aware image cache.
  Definitions/results persist in the scan cache; edits invalidate old results.
  ROIs use displayed pixel coordinates and original values (independent of colour
  scaling and mask overlays); nonfinite pixels are excluded and unreadable frames
  are reported as NaN. Select one scan and a monochrome detector image; exit live
  mode before computing. Cache persistence follows `SMI_BROWSER_CACHE_DIR`.
- **Process** — Transmission and grazing-incidence reduction via
  smi-tiled (`reduce_smi_combined`, `reduce_smi_gi`).
  Produces 2D q-chi / qxy-qz maps and merged I(q) curves.
- **Scan Collection** — Add processed scans for side-by-side I(q) comparison.

## Quickstart

### Prerequisites

- [pixi](https://pixi.sh/) package manager
- A local editable clone of
  [smi-tiled](https://github.com/EliotGann/smi-tiled)
  at the path referenced in `pixi.toml` (defaults to `../smi-tiled`)

### Install & run

```bash
# Install all dependencies (first run takes a few minutes)
pixi install

# Launch the browser
pixi run serve
```

This opens the app at `http://localhost:5006/smi_app`.

### Configuration

The Tiled catalog URI defaults to `https://tiled.nsls2.bnl.gov` and the
catalog path to `smi/migration`. To change these, edit the constants at
the top of `smi_app.py`:

```python
TILED_URI = "https://tiled.nsls2.bnl.gov"
CATALOG_PATH = "smi/migration"
```

## Project structure

```
smi-browser/
├── smi_app.py          # Main Panel application
├── tiled_browser.py    # Tiled REST helpers (search, fetch, metadata)
├── pixi.toml           # Pixi environment & dependencies
└── README.md
```

> Default detector masks (`pil2M_mask_polygons.json`, `900KW_mask_polygons.json`)
> are now bundled with smi-tiled and loaded automatically; the Process
> tab "Masks" expander shows the default name in its placeholder. Use the
> Explore tab's *Mask overlay* card to view, edit (PolyDraw / PolyEdit), or
> save a custom mask, then click *Use in Process* to apply it.

## Development

```bash
# Auto-reload on file changes
pixi run serve
# (--autoreload is already included in the task)

# Quick import check
pixi run check
```

## License

TBD
