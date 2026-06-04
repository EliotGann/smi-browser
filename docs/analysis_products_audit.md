# Audit of derived analysis products in `smi-browser`

**Audience:** an agent (or developer) porting these capabilities into the
`smi-tiled` backend so that a single `reduce_smi_combined(...)` call can produce
everything currently exportable from the browser's Export tab, ideally writing
the result straight to a writable tiled catalog with no on-disk files.

**Current state in `smi-tiled`** (verified):
- `smi_tiled.integrator.reduce_smi_combined` / `reduce_smi_gi` return a
  `CombinedReductionResult` / `GIReductionResult` with `merged_iq`,
  `merged_qchi`, and `per_frame_iq` (the last carries per-frame primary scalars
  as data vars on the `frame` dim).
- `smi_tiled.upload.session.UploadSession` is a stub that already maps
  `(uid → merged_iq, merged_qchi, per_frame_iq)` onto a tiled node layout
  (`smi_tiled.upload.schema.REDUCED_DATA_KEYS`). `upload()` raises
  `NotImplementedError` today.

The three derived-analysis products described below all live in `smi-browser`
today and should migrate into `smi-tiled` as composable, optional pipeline
stages that decorate the existing result and are persisted by `UploadSession`.

---

## 1. Virtual primary axes from structured per-frame strings

### What it produces
Numeric per-frame columns parsed out of free-text fields like
`target_file_name` / `sample_name`, e.g.
`Lucas_sample2_pos1_2450.00eV_ai0.50_wa9_bpm1.995_degC100.0` →
`{sample: 2, pos: 1, eV: 2450.0, ai: 0.5, wa: 9, bpm: 1.995, degC: 100.0}`.

These then drive every axis selector in the UI (Primary, Explore, Process 2D
map, Peak Map) and flow into HDF5 exports under `/primary/`.

### Where the code lives
- [smi_browser/data/scalars.py](smi_browser/data/scalars.py) —
  `parse_label_number_tokens`, `derive_virtual_columns`,
  `VIRTUAL_PREFIX = "fn:"`.
- Called once in [smi_app.py](smi_app.py#L514) inside `_scalars_to_dataframe`
  (handles both fresh-from-tiled and HDF5-cached byte-string paths).

### Algorithm details
1. **Token regex** `([A-Za-z][A-Za-z%°µ/]*)?(-?\d+(?:\.\d+)?)([A-Za-z%°µ/]+)?`
   — matches a number with an optional adjacent letter prefix and/or unit.
   Bare numbers with no adjacent letters are ignored.
2. **Label resolution:** the alphabetic *prefix* wins; if absent, the trailing
   *unit* is used as the label. First occurrence of a label within a string wins.
3. **Source auto-discovery:** when `sources=None`, every non-numeric, non-`ts_`
   column is scanned.
4. **Min-fill gate:** a derived column is kept only when its non-NaN fraction
   ≥ `min_fill=0.5` (rejects noise from free-text status fields).
5. **Naming + disambiguation:** `fn:<label>`; on collision across sources,
   `fn:<source>:<label>`.
6. Cache safety: HDF5 string columns come back as `bytes`; the parser
   transparently decodes (`bytes` / `np.bytes_` / `np.str_` / `str`).

### Migration notes for `smi-tiled`
- Move `parse_label_number_tokens` + `derive_virtual_columns` into a new
  `smi_tiled.derived.virtual_axes` module — they are already pure NumPy /
  pandas and have no Bokeh/Panel ties.
- They should run inside `_build_per_frame_iq` (integrator.py:1861) at the
  point where per-frame primary scalars are attached as data vars: parse the
  string-typed scan_info columns into additional `fn:*` 1-D data vars on the
  `frame` dim. That way **`per_frame_iq` carries them natively** — no second
  scan-info pass needed by consumers, and the browser can drop its
  `_scalars_to_dataframe` post-processing step.
- Surface a `virtual_axes_config: VirtualAxesConfig | None = None` kwarg on
  `reduce_smi_combined` so callers can override the source list, `min_fill`,
  or disable parsing.
- Record the source columns + min_fill into result metadata so the upload
  schema can include them in `PROVENANCE_FIELDS`.

```python
@dataclass(frozen=True)
class VirtualAxesConfig:
    sources: tuple[str, ...] | None = None   # None → auto-discover
    min_fill: float = 0.5
    prefix: str = "fn:"
    enabled: bool = True
```

---

## 2. Line cuts of 2D reductions (frame-by-frame)

### What it produces
1-D cross sections of `merged_qchi` (transmission) or `qxy/qz` (GI), one cross
section per "cut" the user drew on the 2-D display. Two kinds:
- `h` — horizontal band over the y-axis → `I(x)` (e.g. averaged over a χ band → `I(q)`)
- `v` — vertical band over the x-axis → `I(y)` (e.g. averaged over a q band → `I(χ)`)

In `smi_app.py::_render_cuts_plot` (around line 4976) the same operation is
applied **per frame** to `saxs_qchi_frames` / `waxs_qchi_frames` (already
produced inside `reduce_smi_combined`) to yield a stack of overlaid cuts.

### Where the code lives
- [smi_browser/figures/cuts.py](smi_browser/figures/cuts.py) —
  `compute_cross_section`, `cuts_to_source_data`, `source_data_to_cuts`.
- Persisted across scans: `_persisted_cuts` list in
  [smi_app.py](smi_app.py#L4782).
- HDF5 export: [smi_browser/export.py](smi_browser/export.py) writes
  `/cuts/cut_NNN/{axis, intensity, kind, center, width, axis_label}` via
  `_save_dataset_h5`.

### Algorithm details
A cut is a dict `{kind: "h"|"v", center: float, width: float}` in the same
units as the 2-D map. For each cut:

```python
half = max(width / 2, 0)
if kind == "h":
    mask = (y >= center-half) & (y <= center+half)
    section = np.nanmean(image[mask, :], axis=0)  # or nearest row if empty
    return x, section, x_label
else:   # "v"
    mask = (x >= center-half) & (x <= center+half)
    section = np.nanmean(image[:, mask], axis=1)
    return y, section, y_label
```

- `width == 0` → fall back to nearest single index (`argmin(|axis - center|)`).
- `nanmean` excludes masked detector pixels properly (the 2-D image is held as
  `np.float64` with NaN sentinels).

### Migration notes for `smi-tiled`
- New `smi_tiled.derived.linecuts` module with:
  ```python
  @dataclass(frozen=True)
  class LineCutSpec:
      kind: Literal["h", "v"]
      center: float
      width: float
      target: Literal["qchi", "qxy_qz"] = "qchi"  # which 2-D product to cut
      name: str | None = None  # optional; defaults to "{kind}_{center:.3g}"

  def apply_line_cuts(
      result: CombinedReductionResult | GIReductionResult,
      cuts: Sequence[LineCutSpec],
      *,
      per_frame: bool = True,
  ) -> xr.Dataset:
      """Returns dims (cut, frame, axis); coords: cut name, frame, axis values.
      data_vars: intensity, plus per-cut attrs kind/center/width/target."""
  ```
- Per-frame uses `saxs_result["q_chi_frames"]` / `waxs_result["q_chi_frames"]`
  that already exist inside `CombinedReductionResult.saxs` / `.waxs` (see
  `smi_browser/export.py::_save_dataset_h5` "per_frame_qchi" branch). For GI,
  per-frame uses `gi_result.frames`.
- Axes coordinates differ across cuts (`q` vs `chi`) so the natural container
  is one xarray `Dataset` per cut (collected into a dict), **or** a `Dataset`
  with a length-N `cut` dim plus a ragged `axis` coord. The former is easier;
  the latter packs better into tiled. Suggested compromise: one `xr.Dataset`
  per `target` (so all `qchi` h-cuts share the q axis, all v-cuts share the
  chi axis), with `cut` as a string-indexed dim.
- Add a `line_cuts: Sequence[LineCutSpec] | None = None` kwarg to
  `reduce_smi_combined` / `reduce_smi_gi`. When supplied, attach the resulting
  datasets to the result as `result.line_cuts` (a dict keyed by target).
- For interactive iteration the same helper is callable on an already-reduced
  result without re-running the full pipeline (it only reads `saxs/waxs[
  "q_chi_frames"]`).

---

## 3. Per-frame peak fitting of 1-D I(q)

### What it produces
Per-frame `(amplitude, center, fwhm, area, success)` arrays for each
user-defined peak. Drives the Peak Map sub-tab (a 1-D or 2-D map of a chosen
fit parameter against any per-frame axis — including the virtual `fn:*` axes
from item 1).

### Where the code lives
- [smi_browser/models/peakfit.py](smi_browser/models/peakfit.py) —
  `PeakDef` dataclass, `fit_peak_across_frames`, `FIT_PARAMS`, gating
  constants `MIN_SNR = 3.0`, `MIN_R2 = 0.2`, `WIDTH_BOUND_TOL = 0.97`.
- Persistence:
  - Per-fit results cached at `/peakfit/<sha1(key)>` inside each scan's
    HDF5 cache file via `ScanCache.write_peakfit` /
    `read_peakfit_full` ([smi_browser/cache.py](smi_browser/cache.py#L347)).
  - Global peak list (cross-scan) persisted as
    `peak_defs.json` under the cache root.
- Invalidation: re-reducing a scan overwrites `/reduction/pf_iq_*`, which
  invalidates `/peakfit/*` (browser drops them).
- Driver: [smi_app.py::_on_peak_fit](smi_app.py#L5720) (background thread,
  cancellable, marshals progress to the Bokeh doc).

### `PeakDef` model
```python
@dataclass(frozen=True)
class PeakDef:
    name: str
    q_min: float
    q_max: float
    model: Literal["gaussian", "lorentzian"] = "gaussian"
    baseline: Literal["none", "linear"] = "linear"
    link: Literal["independent", "linked", "tracked"] = "independent"
    bg_factor: float = 2.0   # widens fit window vs. drawn core (linear baseline only)

    def key(self) -> tuple:    # used as cache key + provenance hash
        ...
```

### Algorithm details (these are the non-trivial bits to preserve verbatim)
- **Fit window vs. peak core.** The user-drawn `[q_min, q_max]` is the *core*
  (peak centre is constrained here). When `baseline=="linear"`, the actual fit
  window is widened to `bg_factor × core` so the slope/intercept are anchored
  by the peak's flanks (`peakfit.py` ~ line 215).
- **Width bound.** `_width_bounds(model, core_range, dq)` clamps the fitted
  σ/γ so the *resulting FWHM* never exceeds the drawn range. This single
  constraint is what prevents the "fit fills the window when there's no peak"
  pathology.
- **Vectorised initial guesses** (`_initial_guesses`): subtracts a straight
  baseline through the window's endpoints, locates the residual argmax, and
  clamps the centre guess into the core. Fast and immune to baseline tilt.
- **Quality gate** (`_accept`): a fit is accepted only when
  `amp > 0 AND snr ≥ MIN_SNR (3) AND r2 ≥ MIN_R2 (0.2) AND |width| < 0.97·w_max`.
  Frames that fail report `amplitude=0, area=0, center=NaN, fwhm=NaN,
  success=False` (distinguishable from "fit didn't run" → all NaN).
- **Link modes:**
  - `independent` — `curve_fit` per frame.
  - `tracked` — `curve_fit` per frame, warm-started from previous frame's popt.
    Substantial speedup + smoother centre track across raster scans.
  - `linked` — **two-step** fit:
    1. Aggregate fit on `np.nanmean(iq, axis=0)` to nail down
       `(center*, width*)`.
    2. Per-frame **linear** solve via `np.linalg.lstsq` on
       `[peak_shape(q; center*, width*), q, 1]` for `(amp, slope, intercept)`.
       Orders of magnitude faster than per-frame `curve_fit`; only varies
       amplitude per frame (intentionally — appropriate when peak position is
       known to be fixed). Falls back to `independent` with a `note`
       in the result dict if the aggregate fit fails the quality gate.
- **Models:** Gaussian `amp·exp(-½((q-μ)/σ)²)` (FWHM=σ·2√(2ln2),
  area=|amp·σ|·√(2π)); Lorentzian `amp·γ²/((q-μ)²+γ²)` (FWHM=2γ,
  area=|amp|·π·γ).
- **Cancellation/progress:** caller passes a `threading.Event`-like `cancel`
  (checked every `cancel_check_every=64` frames) and a `progress(done, total)`
  callback.

### Migration notes for `smi-tiled`
- The module is already Panel/Bokeh-free and unit-tested
  ([tests/test_peakfit.py](tests/test_peakfit.py)) — drop the **entire file**
  into `smi_tiled.derived.peakfit` unchanged.
- Add a `smi_tiled.derived.peakfit.fit_peaks_over_result(result, peaks)` driver
  that:
  1. Pulls `q = result.per_frame_iq["q"].values`, `I = result.per_frame_iq["I"].values`.
  2. Loops `fit_peak_across_frames` over each `PeakDef`.
  3. Returns one `xr.Dataset` with dims `(peak, frame)` and data_vars
     `amplitude, center, fwhm, area, success` plus per-peak attrs
     (q_min, q_max, model, baseline, link, bg_factor, name). Note key/value
     pairs: the dataset's `peak` coord is the peak `name`; a parallel
     `peak_key` data_var (string-encoded `PeakDef.key()`) preserves the
     cache-key identity for `UploadSession` staleness checks.
- Add a `peak_fits: Sequence[PeakDef] | None = None` kwarg to
  `reduce_smi_combined`. When supplied, attach the resulting Dataset to the
  result as `result.peak_fits`.
- Add `apply_peak_fits(result, peaks)` for interactive use (no re-reduction).

---

## 4. Export tab → what `reduce_smi_combined` should ultimately deliver

Currently the Export tab (`smi_browser/export.py::export_scan`) emits sections
gated by `h5_sections`. Mapping each section to its tiled-backend equivalent:

| Export section      | Source today                                | Should come from smi-tiled as                                                                                                                                  |
|---------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `metadata`          | tiled run start/stop docs                   | Already there — `CombinedReductionResult.scan_info` + tiled node metadata. Schema field already in `PROVENANCE_FIELDS`.                                       |
| `primary`           | `_scalars_to_dataframe(run)` + virtual axes | **Pack into `per_frame_iq` data_vars** (incl. `fn:*` columns from item 1).                                                                                    |
| `baseline_config`   | tiled baseline + config streams             | Add to `scan_info` or a sibling `baseline` xr.Dataset on the result; persist as a `baseline` tiled child node.                                                |
| `raw_images`        | tiled or browser cache                      | Out of scope — leave to direct tiled reads. Optionally a `raw_images` child node pointing back to the source run.                                             |
| `processed_iq`      | `result.merged_iq` + `result.per_frame_iq`  | Already produced.                                                                                                                                             |
| `processed_qchi`    | `result.merged_qchi` + `saxs/waxs[q_chi_frames]` | Already produced; bundle the per-frame stacks into the result as a public attribute `result.per_frame_qchi` (currently buried in `result.saxs/waxs` dicts). |
| `cuts`              | `compute_cross_section` from item 2          | New `result.line_cuts` (item 2).                                                                                                                              |
| `peakfit`           | `ScanCache.read_peakfit_full()`              | New `result.peak_fits` (item 3).                                                                                                                              |
| `parameters`        | dict from `processing.build_proc_params`     | Already covered by `PROVENANCE_FIELDS` + `reduction_hash`.                                                                                                    |

---

## 5. Proposed mechanism — interactive iteration **and** batch

Big picture:
- **Interactive (browser)**: the user opens a scan, picks cuts and peaks,
  previews them — without ever re-running the heavy SAXS/WAXS integration.
- **Batch (queue)**: once happy, the same `LineCutSpec`s and `PeakDef`s are
  passed into `reduce_smi_combined(...)` for *all* selected scans; the result
  for each is uploaded to a writable tiled catalog via `UploadSession`.

### 5.1 Pipeline-stage composition inside `smi-tiled`

Augment `reduce_smi_combined` with three opt-in kwargs (all default `None` →
no-op, full back-compat):

```python
def reduce_smi_combined(
    uid: str,
    ...,
    virtual_axes: VirtualAxesConfig | None = VirtualAxesConfig(),  # on by default
    line_cuts: Sequence[LineCutSpec] | None = None,
    peak_fits: Sequence[PeakDef]      | None = None,
) -> CombinedReductionResult: ...
```

Internal flow (additions only):
1. After `_build_per_frame_iq`: if `virtual_axes.enabled`, parse string-typed
   per-frame data_vars on `per_frame_iq` and **attach `fn:*` data_vars on the
   `frame` dim**.
2. After the merge: if `line_cuts`, call `apply_line_cuts(result, line_cuts)`
   and attach to `result.line_cuts`.
3. After (2): if `peak_fits`, call `apply_peak_fits(result, peak_fits)` and
   attach to `result.peak_fits`.

Each stage is also callable standalone on an already-reduced result:

```python
from smi_tiled.derived import apply_line_cuts, apply_peak_fits, derive_virtual_axes

result_v  = derive_virtual_axes(result, config)     # returns a new result
result_lc = apply_line_cuts(result_v, cuts)         # cheap — uses cached per-frame qchi
result_pf = apply_peak_fits(result_lc, peaks)       # cheap — uses cached per_frame_iq
```

The browser's *Peak Map* and *Cuts* tabs become thin clients of these helpers:
they call the helper, render the result, and let the user iterate. No
re-reduction occurs until the user explicitly clicks "Process".

### 5.2 Preview → commit round trip in the browser

1. **Preview:** browser calls `apply_line_cuts` / `apply_peak_fits` on the
   in-memory `CombinedReductionResult`. Results live in the existing
   browser-side caches (`ScanCache`, `_peak_fit_cache`) so behaviour matches
   today.
2. **Commit:** when the user is satisfied, the browser serializes the current
   `line_cuts` + `peak_fits` (plus `VirtualAxesConfig`) into the batch
   processor's job spec. These flow through `processing.build_proc_params`
   into the `reduce_smi_combined(...)` call for every scan in the batch.
3. **Upload:** the batch worker passes the resulting
   `CombinedReductionResult` (now carrying `line_cuts` + `peak_fits` +
   `fn:*`) to `UploadSession.upload(...)`. The browser/export tab can then
   read from the tiled sandbox instead of writing files.

### 5.3 Required `UploadSession` extensions

Extend `smi_tiled.upload.schema.REDUCED_DATA_KEYS`:

```python
REDUCED_DATA_KEYS = (
    "merged_iq",
    "merged_qchi",
    "per_frame_iq",     # now includes fn:* virtual-axis data_vars
    "per_frame_qchi",   # NEW — promote from result.saxs/waxs to a public product
    "line_cuts",        # NEW — dict-of-Datasets, one per target
    "peak_fits",        # NEW — Dataset on (peak, frame)
)
```

Extend `PROVENANCE_FIELDS` with the spec hashes so `reduction_hash` invalidates
correctly:

```python
PROVENANCE_FIELDS += (
    "virtual_axes_spec_hash",
    "line_cuts_spec_hash",
    "peak_fits_spec_hash",
)
```

Each spec hash is computed identically to the cache-key digest already used
in `smi_browser/cache.py::_peak_hash` (sha1 over a JSON dump of the spec
list). Per-peak identity inside the `peak_fits` dataset is preserved by the
`peak_key` data_var (see item 3 above), so partial re-fits remain possible.

### 5.4 Browser-side simplifications enabled by this migration

Once the above lands in `smi-tiled`, the browser can shed:
- `smi_browser/data/scalars.py::derive_virtual_columns` and its call site in
  `_scalars_to_dataframe` (replaced by reading the `fn:*` data_vars off
  `per_frame_iq`).
- The `_peak_fit_cache` + `/peakfit` HDF5 group in `ScanCache` (the result is
  carried on the in-memory `CombinedReductionResult` and stored in the
  upload-tiled catalog; the browser keeps only a session-local UI cache).
- `peak_defs.json` becomes the browser's *user preference* file only — when
  applied, the actual fitted arrays are read from the tiled product, not
  recomputed.
- Most of the HDF5 writing in `smi_browser/export.py` becomes optional — the
  canonical artifact is the tiled node; HDF5 export becomes a "download a
  snapshot" feature that walks the tiled product tree.

---

## 6. Implementation checklist for the `smi-tiled` agent

1. **New package** `smi_tiled/derived/`:
   - `virtual_axes.py` — port `parse_label_number_tokens`,
     `derive_virtual_columns` from `smi_browser/data/scalars.py` verbatim;
     add `VirtualAxesConfig` dataclass + `derive_virtual_axes(result, config)`
     helper.
   - `linecuts.py` — port `compute_cross_section` from
     `smi_browser/figures/cuts.py`; add `LineCutSpec` dataclass +
     `apply_line_cuts(result, cuts, per_frame=True)` helper.
   - `peakfit.py` — drop `smi_browser/models/peakfit.py` in unchanged; add
     `apply_peak_fits(result, peaks)` driver that packs results into one
     `xr.Dataset`.
2. **Integrator hooks** — add `virtual_axes`, `line_cuts`, `peak_fits` kwargs
   to `reduce_smi_combined` / `reduce_smi_gi`; call the three helpers in
   order at the end of the pipeline.
3. **Result extension** — add three optional fields to
   `CombinedReductionResult` / `GIReductionResult`:
   `line_cuts: dict[str, xr.Dataset] | None`,
   `peak_fits: xr.Dataset | None`,
   `per_frame_qchi: xr.Dataset | None` (promoted from `saxs/waxs`).
4. **Schema/upload** — extend `REDUCED_DATA_KEYS` + `PROVENANCE_FIELDS`;
   implement `UploadSession.upload` to walk the result and write every
   non-`None` product as a child node.
5. **Tests** — move `tests/test_peakfit.py`, `tests/test_scalars.py`
   (`derive_virtual_columns` parts) to `smi-tiled`; add round-trip tests
   for `apply_line_cuts` and the upload schema.
6. **Browser migration (follow-up PR)** — once the tiled side ships,
   replace the three browser modules with thin shims that import from
   `smi_tiled.derived`, and switch the Export tab to read from the tiled
   sandbox where available.

---

## 7. Browser cleanup status (June 2026)

The `smi-tiled` migration shipped (see
`smi-tiled/docs/source/reference/derived-products-migration.md`).  The
browser-side cleanup completed so far:

- [smi_browser/models/peakfit.py](../smi_browser/models/peakfit.py) — shim
  re-exporting `PeakDef`, `FIT_PARAMS`, `MIN_SNR`, `MIN_R2`,
  `fit_peak_across_frames`, `apply_peak_fits` from `smi_tiled.derived.peakfit`.
- [smi_browser/figures/cuts.py](../smi_browser/figures/cuts.py) —
  `compute_cross_section` is now a 1-line delegate to
  `smi_tiled.derived.linecuts.compute_cross_section`; Bokeh-glyph helpers
  (`cuts_to_source_data`, `source_data_to_cuts`, `format_cut_label`) stay.
- [smi_browser/data/scalars.py](../smi_browser/data/scalars.py) —
  `derive_virtual_columns`, `parse_label_number_tokens`, `VIRTUAL_PREFIX`
  re-exported from `smi_tiled.derived.virtual_axes`; the tiled-fetch glue
  (`scalars_to_dataframe`, `scalar_stream_to_frame`) stays.
- [smi_browser/processing.py](../smi_browser/processing.py) — new optional
  `virtual_axes` / `line_cuts` / `peak_fits` kwargs on `build_proc_params`,
  forwarded verbatim into the kwargs dict for both `reduce_smi_combined` and
  `reduce_smi_gi`.  Default `None` keeps existing callers untouched.

All 170 tests pass.

### Remaining work

These items were intentionally **deferred** out of this cleanup pass so the
in-browser interactive experience keeps working unchanged.  They should be
picked up in follow-up PRs:

1. **Consume `result.peak_fits` from the Peak Map tab.**
   When a result comes back from `reduce_smi_combined` with `peak_fits`
   attached (i.e. the user committed the current peak list at process time),
   the Peak Map tab should *read* from `result.peak_fits` instead of
   re-fitting client-side via `fit_peak_across_frames`.
   Touch-points:
   - [smi_app.py::_on_peak_fit](../smi_app.py) (~ line 5720) — check
     `result.peak_fits` first; only call `fit_peak_across_frames` for peaks
     not already covered.
   - [smi_browser/cache.py::ScanCache.write_peakfit](../smi_browser/cache.py)
     — once the tiled product is the source of truth, the local
     `/peakfit` HDF5 group becomes a session-only cache (or can be dropped
     entirely when reading from a writable tiled catalog).

2. **Consume `result.line_cuts` from the cross-section tab.**
   Same pattern: when `result.line_cuts[<name>]` exists, render it directly
   instead of calling `compute_cross_section` on the merged image.
   Touch-points:
   - [smi_app.py::_render_cuts_plot](../smi_app.py) (~ line 4976) — prefer
     the pre-computed dataset for any committed cut spec.

3. **Consume `fn:*` data_vars from `per_frame_iq`.**
   When the result was reduced with a `VirtualAxesConfig`, `per_frame_iq`
   already carries the `fn:*` columns as data variables on the `frame` dim.
   The browser's `_scalars_to_dataframe` path can short-circuit its local
   `derive_virtual_columns` call when the same columns are already present.
   Touch-points:
   - [smi_app.py::_scalars_to_dataframe](../smi_app.py) (~ line 510).

4. **UI to author the specs and pass them to `build_proc_params`.**
   The browser already builds `PeakDef` / cut-dict objects locally; expose
   "Process **with these peaks/cuts**" and "Process **with virtual axes
   from these source columns**" toggles in the Processing tab so the user's
   interactive choices propagate to batch reductions via
   `processing.build_proc_params(..., virtual_axes=..., line_cuts=...,
   peak_fits=...)`.
   Touch-points:
   - [smi_browser/ui/batch.py](../smi_browser/ui/batch.py) — collect the
     active peak list + cuts list from session state and forward.
   - [smi_app.py::_on_process](../smi_app.py) — same for the interactive
     "Process" button.

5. **Export tab pivot to read from the tiled sandbox.**
   Once `UploadSession.upload` is implemented in `smi-tiled` (currently
   `NotImplementedError`), the HDF5 export path in
   [smi_browser/export.py](../smi_browser/export.py) becomes a "download a
   snapshot from tiled" feature — most of the per-section gating
   (`h5_sections`) can drop in favour of walking
   `REDUCED_DATA_KEYS`.

6. **Optional: remove the browser-local caches that the tiled catalog
   supersedes.**
   `ScanCache.write_peakfit` / `read_peakfit_full` and
   `peak_defs.json` exist today because the result wasn't persisted
   anywhere durable.  Once writes land in the upload-tiled catalog,
   `peak_defs.json` reduces to a *user preference* (the active peak list
   in the UI), and the per-fit cache can be dropped.
