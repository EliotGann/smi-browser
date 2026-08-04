# Plan: per-arc / non-`primary` stream support for reduction (`smi-tiled`)

**Status:** PROPOSAL — no `smi-tiled` code changed yet. The browser-side
(display/browse) half is implemented in `smi-browser`; see "Done in smi-browser"
below. This document scopes the matching `smi-tiled` work.

**Audience:** whoever extends `smi-tiled`'s `reduce_smi_combined` /
`reduce_smi_gi` to reduce data from a stream other than `primary`.

---

## 1. Why

`smi-plans`' arc-economy GIWAXS path (`giwaxs_bar_arc_economy`) now writes
**one event stream per WAXS arc** — `arc0`, `arc20`, `arcm1p5`, … — instead of a
single `primary` stream (see
`~/git/smi/smi-plans/docs/STREAMS_AND_WORKFLOW.md`). Each arc stream has its own
descriptor: low arc is WAXS-only (`pil900KW`), high arc is SAXS+WAXS
(`pil2M` + `pil900KW`). Each arc-stream event still records `energy_energy`,
`waxs_arc`, `incident_angle`, `xbpm2_sumX`, etc.

`reduce_smi_combined` / `reduce_smi_gi` currently read **only** `run["primary"]`,
so they cannot reduce these per-arc runs. The reduction is the last piece that
hard-codes the stream.

## 2. Done in `smi-browser` (context for the reduction author)

The browser now lets a user pick which stream the **display** tabs target:

- A `Stream` dropdown (hidden unless a run has >1 non-`baseline` stream) drives
  the Primary-scalars tab, Explore image viewer, and Grid. The Primary tab
  relabels to `Primary (arc20)` when a non-`primary` stream is active.
- `smi_browser/_tiled.py` helpers (`fetch_scalars`, `fetch_frame`,
  `stream_info_for`, `stream_fields*`) were already `stream=`-parameterized.
- `smi_browser/cache.py` is now **stream-aware** for scalars *and* images:
  - scalars → group `/<stream>` (already supported; `primary`/`baseline`
    unchanged, others sanitized).
  - images → `images/<field>` for `primary` (legacy, back-compat) and
    `images/<stream>/<field>` otherwise, with a parallel
    `images_filled/<stream>/<field>` mask. Helpers `_image_path(stream, field)`
    / `_image_fill_path(stream, field)` and a `stream="primary"` kwarg on
    `read/write_image_stack`, `has_image_field`, `get_or_fetch_image_stack`,
    `get_or_fetch_image_frame`, `_cache_one_frame`.

**Explicitly still `primary`-only in the browser** (deliberately out of scope of
that pass, and dependent on this reduction work): the Process tab, Export tab
(HDF5 raw-image + reduction sections), Collection comparison, and live-mode
following. `_current_primary_len`, the two export raw-image fetchers, and the
live consumer still pass `"primary"`.

## 3. The `smi-tiled` change

### 3.1 Public API

Add a single optional parameter to both entry points, defaulting to today's
behaviour:

```python
def reduce_smi_combined(uid, ..., stream: str = "primary") -> CombinedReductionResult: ...
def reduce_smi_gi(uid, ..., stream: str = "primary") -> GIReductionResult: ...
```

Thread `stream` down to every helper that currently hard-codes `"primary"`.
Because the value defaults to `"primary"`, **all existing callers and tests are
unaffected**.

### 3.2 Internal sites to parameterize (verified inventory)

All `"primary"` literals are in two modules. Each data-reading helper takes a
`run` (and sometimes a `field`); give each a `stream: str = "primary"` argument
and pass it through from the `reduce_*` call.

`src/smi_tiled/loader.py` (15 occurrences):

| Line | Enclosing function | Note |
|---|---|---|
| 546 | `_get_primary_scalar_fields(run)` | reads per-frame scalars from `run["primary"]` |
| 760-763 | `_primary_conf(run, det_key)` | per-detector configuration block |
| 836-851 | `_has_primary_internal_field(run, field)` | `run["primary"]["internal"]` membership |
| 874-898 | `_read_target_file_name_raw(...)` | `run["primary"].base` — filename-token source |
| 1478-1485 | `_get_primary_field_node(run, field)` | **image node** — the hot path for detector frames |
| 1769-1776 | `_has_primary_field(run, field)` | field presence check |
| 2339-2406 | `infer_detectors_and_steps(run, ...)` | `run["primary"]` → detectors + step count |
| 465-467, 2250-2251 | HDF5 *cache* group `primary` | **see §3.3 — do NOT blindly rename** |
| 2985 | `stop["num_events"]["primary"]` | point count from stop doc — key by `stream` |

`src/smi_tiled/integrator.py` (9 occurrences):

| Line | Enclosing function | Note |
|---|---|---|
| 939-949 | image-array accessor (`run["primary"]["data"][field]` ladder) | **hot path**; mirror the 3-tier fallback on `run[stream]` |
| 1002-1004 | second `run["primary"]` accessor | same treatment |
| 2686 | `primary_ds = run["primary"]` | dataset read |
| 3741 | comment only | update wording |

The function names (`_get_primary_*`, `_has_primary_*`) can stay as-is to keep
the diff small, or be renamed to `_stream_*` for clarity — author's choice. The
behavioural change is only the subscript key.

### 3.3 Disk-cache interplay (important)

`smi-tiled`'s own HDF5 image cache (`loader.py:465`, `2250`) writes a `primary`
group, and `reduce_*` accepts `image_cache_path` pointing at the **same per-UID
file** the browser uses (`smi_browser/cache.py`). Two cooperating caches touch
one file, so they must agree on layout:

- The browser stores non-primary images at `images/<stream>/<field>` (§2).
- `smi-tiled` must use the **same** path when `stream != "primary"`, and keep
  the legacy `primary`/`images/<field>` path when `stream == "primary"`.

**Action:** when wiring `stream` through `smi-tiled`'s cache writer/reader, reuse
the exact path convention from `smi_browser.cache._image_path` /
`_image_fill_path` (copy the two-line helper). A mismatch would silently
double-store frames or read the wrong stream's images. Add a cross-repo test
asserting both writers target identical paths for a sample `(stream, field)`.

### 3.4 Detector set varies per stream

Low-arc streams expose only `pil900KW_image`; high-arc add `pil2M_image`. The
SAXS+WAXS merge in `reduce_smi_combined` must tolerate a **missing SAXS
detector** for WAXS-only arc streams (return WAXS-only `merged_iq`, skip the
SAXS branch) rather than erroring. `infer_detectors_and_steps` already discovers
detectors from the stream, so once it reads `run[stream]` this should mostly
fall out — but verify the merge path degrades gracefully (it is the most likely
place a WAXS-only arc reduction breaks).

## 4. Browser follow-up (after `smi-tiled` lands, separate task)

Once `reduce_*` accepts `stream`, wire the browser's `_build_proc_params` /
`processing.build_proc_params` to pass the active stream, then make
stream-aware: the Process tab, the Export HDF5 raw-image + reduction sections
(`smi_app.py` ~11318, ~12132), and reduction/peakfit cache keys
(`cache.write_reduction` / `/peakfit` are per-UID with no stream dimension —
they would need a `<stream>` component or a per-arc UID-scoping scheme). Track
that as its own change; it is **not** required for per-arc *browsing*, which
already works.

## 5. Test checklist (smi-tiled)

- `reduce_smi_combined(uid, stream="primary")` is byte-identical to the
  no-arg call on a normal run (regression guard).
- A simulated arc-economy run with `arc0` (WAXS-only) + `arc20` (SAXS+WAXS):
  - `reduce_smi_combined(uid, stream="arc20")` produces SAXS+WAXS `merged_iq`;
  - `reduce_smi_combined(uid, stream="arc0")` produces WAXS-only `merged_iq`
    without raising on the absent SAXS detector;
  - `per_frame_iq` carries that stream's `incident_angle` / `waxs_arc` scalars.
- Cache path parity: `smi-tiled`'s writer and `smi_browser.cache._image_path`
  resolve to the same dataset path for `("arc20", "pil900KW_image")`.
