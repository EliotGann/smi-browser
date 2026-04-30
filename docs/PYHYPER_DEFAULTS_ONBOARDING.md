# Onboarding agent notes — adopt PyHyperScattering SMI defaults

Audience: an agent (or developer) updating `smi-browser` so that it stops
shipping its own copies of SMI defaults and instead consumes the canonical
defaults that now live in the **PyHyperScattering** package.

**Date of write-up:** 2026-04-28
**Upstream commits:** new `PyHyperScattering.smi_defaults` module +
bundled masks under `PyHyperScattering/data/smi/masks/`.

---

## TL;DR for the interface

1. **Stop shipping `masks/`.** PyHyperScattering now bundles the canonical
   `pil2M_mask_polygons.json` and `900KW_mask_polygons.json`. Delete the
   `smi-browser/masks/` directory once you're satisfied with parity.
2. **Change widget defaults from filename strings to empty / `None`.**
   The TextInput widgets `w_proc_saxs_mask` / `w_proc_waxs_mask` should
   default to empty strings ("use bundled default") and only be filled in
   when a user wants to override.
3. **Stop sending defaults that match upstream defaults.** Pass `None`
   (or omit the kwarg entirely) for any value that exactly matches the
   PyHyper defaults — the integrators substitute the right thing.
4. **Keep the widgets visible.** Every option below stays exposed for
   power users; we just don't *transmit* values that match the default.
5. **For mask paths, use `PyHyperScattering.smi_defaults` to display the
   default in the placeholder text** so users still see what is being used.

---

## Where the defaults now live

```python
from PyHyperScattering.smi_defaults import (
    DEFAULT_TILED_URI,            # "https://tiled.nsls2.bnl.gov"
    DEFAULT_CATALOG,              # "smi/migration"
    DEFAULT_SAXS_MASK_NAME,       # "pil2M_mask_polygons.json"
    DEFAULT_WAXS_MASK_NAME,       # "900KW_mask_polygons.json"
    default_saxs_mask_path,       # → pathlib.Path to bundled SAXS mask
    default_waxs_mask_path,       # → pathlib.Path to bundled WAXS mask
    resolve_mask_path,            # smart resolver (None | bare name | path)
)
```

`resolve_mask_path(value, detector)` semantics:

| Caller passes                                   | What you get back                  |
|-------------------------------------------------|------------------------------------|
| `None`                                          | bundled default for `detector`     |
| absolute path (existing)                        | the same path                      |
| relative path that exists                       | resolved absolute path             |
| bare filename matching `pil2M_mask_polygons.json` and `detector="saxs"` | bundled default |
| bare filename matching `900KW_mask_polygons.json` and `detector="waxs"` | bundled default |
| anything else                                   | the original `Path` (raises later) |

So `smi-browser` can keep passing `"pil2M_mask_polygons.json"` literally
and it will Just Work — but the cleaner pattern is `None`.

---

## Concrete diffs to apply in `smi_app.py`

### 1. Drop the in-repo mask filenames as the *value* default

Current (`smi_app.py` ~line 67):
```python
DEFAULT_SAXS_MASK = "pil2M_mask_polygons.json"
DEFAULT_WAXS_MASK = "900KW_mask_polygons.json"
```

Change to:
```python
from PyHyperScattering.smi_defaults import (
    DEFAULT_SAXS_MASK_NAME, DEFAULT_WAXS_MASK_NAME,
    default_saxs_mask_path, default_waxs_mask_path,
)
DEFAULT_SAXS_MASK = ""   # empty → use bundled default
DEFAULT_WAXS_MASK = ""
```

### 2. Show the bundled default in the widget placeholder

Current (~line 809):
```python
w_proc_saxs_mask = pn.widgets.TextInput(
    name="SAXS mask", value=DEFAULT_SAXS_MASK, width=220,
)
w_proc_waxs_mask = pn.widgets.TextInput(
    name="WAXS mask", value=DEFAULT_WAXS_MASK, width=220,
)
```

Change to:
```python
w_proc_saxs_mask = pn.widgets.TextInput(
    name="SAXS mask",
    value=DEFAULT_SAXS_MASK,
    placeholder=f"(default: {default_saxs_mask_path().name})",
    width=320,
)
w_proc_waxs_mask = pn.widgets.TextInput(
    name="WAXS mask",
    value=DEFAULT_WAXS_MASK,
    placeholder=f"(default: {default_waxs_mask_path().name})",
    width=320,
)
```

### 3. Send `None` instead of an empty string in `_on_process`

Current (~lines 1564, 1614):
```python
saxs_mask_path=w_proc_saxs_mask.value or None,
waxs_mask_path=w_proc_waxs_mask.value or None,
```

Already correct — the `or None` already swaps "" for `None`. **No change
needed.** But you can drop the `masks/` directory and `pixi` lock entries
referencing those files now.

### 4. Stop transmitting "no override" values

In `_on_process`, audit the kwargs passed to `reduce_smi_combined` and
`reduce_smi_gi`. Pass `None` (or omit the kwarg) when the user has not
changed the widget from its UI default. Specifically the *delta* fields
should default to `None`, **not** `0.0`:

```python
saxs_beam_delta_px=(
    w_proc_saxs_row_delta.value,
    w_proc_saxs_col_delta.value,
) if (w_proc_saxs_row_delta.value or w_proc_saxs_col_delta.value) else None,

waxs_beam_delta_px=(
    w_proc_waxs_row_delta.value,
    w_proc_waxs_col_delta.value,
) if (w_proc_waxs_row_delta.value or w_proc_waxs_col_delta.value) else None,

saxs_distance_delta_mm=(
    w_proc_dist_delta.value if w_proc_dist_delta.value else None
),
```

This restores the upstream calibration-default behavior (e.g.
`_SAXS_DEFAULT_DISTANCE_DELTA_MM = -20.0`, `_SAXS_DEFAULT_BEAM_DELTA_*`)
when the user hasn't entered a custom override.

### 5. Delete `smi-browser/masks/`

After confirming the bundled masks are byte-identical (they are — they
were copied from this repo on 2026-04-28), remove the directory so that
the only source of truth is upstream:

```bash
rm -r masks/
```

---

## Full option matrix exposed by `reduce_smi_combined`

Every kwarg below is currently exposed by the upstream API. Columns:

- **Widget?** — does `smi-browser` already have a UI control for it?
- **Send if equal to default?** — should the browser transmit the value
  even when it matches the default? `No` means: pass `None` / omit.

### Connection / catalog

| kwarg          | type         | default                              | Widget?            | Notes                            |
|----------------|--------------|--------------------------------------|--------------------|----------------------------------|
| `uid`          | `str`        | required                             | derived from row   | run UID                          |
| `tiled_uri`    | `str`        | `"https://tiled.nsls2.bnl.gov"`      | no — module const  | use `DEFAULT_TILED_URI`          |
| `catalog`      | `str`        | `"smi/migration"`                    | no — module const  | use `DEFAULT_CATALOG`            |

### Output grids

| kwarg                       | type     | default          | Widget?              |
|-----------------------------|----------|------------------|----------------------|
| `n_q`                       | `int`    | `1000`           | `w_proc_nq` (=2000)  |
| `n_chi`                     | `int`    | `360`            | `w_proc_nchi`        |
| `solid_angle_correction`    | `bool`   | `False`          | always `True` in app |

### Geometry mode

| kwarg                       | type     | default               | Widget?                         |
|-----------------------------|----------|-----------------------|---------------------------------|
| `geometry`                  | `str`    | `"transmission"`      | `w_proc_geometry`               |
| `incident_angle_deg`        | `float`  | `0.0`                 | `w_proc_incident_angle` (GI)    |

### Masks (now defaulted to bundled)

| kwarg              | type                | default | Widget?            | Notes                          |
|--------------------|---------------------|---------|--------------------|--------------------------------|
| `saxs_mask_path`   | `str | Path | None` | `None`  | `w_proc_saxs_mask` | `None` → bundled SAXS default  |
| `waxs_mask_path`   | `str | Path | None` | `None`  | `w_proc_waxs_mask` | `None` → bundled WAXS default  |

### Geometry overrides (Δ on top of metadata)

| kwarg                       | type             | default | Widget?                   | Notes                                  |
|-----------------------------|------------------|---------|---------------------------|----------------------------------------|
| `saxs_beam_delta_px`        | `(dr, dc)`/`None`| `None`  | `w_proc_saxs_row/col_delta` | `None` → loader's calibrated default |
| `waxs_beam_delta_px`        | `(dr, dc)`/`None`| `None`  | `w_proc_waxs_row/col_delta` | `None` → loader's calibrated default |
| `saxs_distance_delta_mm`    | `float`/`None`   | `None`  | `w_proc_dist_delta`       | `None` → loader's calibrated default   |

Loader-side calibrated defaults (in `SMISWAXSLoader`):

```
_SAXS_DEFAULT_DISTANCE_DELTA_MM = -20.0
_SAXS_DEFAULT_BEAM_DELTA_ROW_PX =   2.0
_SAXS_DEFAULT_BEAM_DELTA_COL_PX =   3.0
_WAXS_DEFAULT_BEAM_DELTA_ROW_PX =   0.0
_WAXS_DEFAULT_BEAM_DELTA_COL_PX =  -2.0
```

### SAXS-specific tuning

| kwarg                       | type         | default | Widget? | Notes                                        |
|-----------------------------|--------------|---------|---------|----------------------------------------------|
| `saxs_q_cutoff`             | `float`/None | `None`  | no      | nm⁻¹; overrides AgBh ring auto-detect        |
| `saxs_agbh_ring_order`      | `int`        | `5`     | no      | which AgBh ring sets the cutoff              |
| `saxs_q_margin_fraction`    | `float`      | `0.01`  | no      | fractional margin above ring                 |

### Hot-pixel rejection

| kwarg                  | type         | default | Widget?         |
|------------------------|--------------|---------|-----------------|
| `dezinger_threshold`   | `float`/None | `3000.0`| `w_proc_dezinger` (0 → None) |
| `dezinger_kernel`      | `int`        | `5`     | no              |

### WAXS-specific tuning

| kwarg                          | type    | default | Widget? | Notes                                      |
|--------------------------------|---------|---------|---------|--------------------------------------------|
| `waxs_beam_col_per_arc_deg`    | `float` | `0.0`   | no      | beam centre drift per deg of `waxs_arc`    |

### Backend / advanced (rarely touched)

| kwarg              | type      | default | Notes                                                        |
|--------------------|-----------|---------|--------------------------------------------------------------|
| `saxs_kwargs`      | `dict`    | `None`  | passes through to `integrate_saxs` (e.g. `dynamic_saxs_mask`) |
| `waxs_kwargs`      | `dict`    | `None`  | passes through to `integrate_waxs`; supports `waxs_bsx_ref`  |
| `backend_options`  | `dict`    | `None`  | `saxs_rotate_cw_90`, `waxs_flip_horizontal`, `waxs_qx_shift_nm`, `waxs_qy_shift_nm` |

---

## Full option matrix exposed by `reduce_smi_gi`

Every kwarg below is currently exposed by the upstream API.

| kwarg                          | type           | default                          | Widget?                       |
|--------------------------------|----------------|----------------------------------|-------------------------------|
| `uid`                          | `str`          | required                         | derived                       |
| `tiled_uri`                    | `str`          | `"https://tiled.nsls2.bnl.gov"`  | no — `DEFAULT_TILED_URI`      |
| `catalog`                      | `str`          | `"smi/migration"`                | no — `DEFAULT_CATALOG`        |
| `waxs_mask_path`               | `str|Path|None`| `None` (was hard-coded filename) | `w_proc_waxs_mask`            |
| `n_qxy`                        | `int`          | `500`                            | `w_proc_nqxy`                 |
| `n_qz`                         | `int`          | `500`                            | `w_proc_nqz`                  |
| `incident_angle_deg`           | `float`/None   | `None`                           | `w_proc_incident_angle` + auto checkbox |
| `theta_offset`                 | `float`        | `-0.5`                           | `w_proc_theta_offset`         |
| `waxs_beam_col_per_arc_deg`    | `float`        | `0.08`                           | no                            |
| `beamstop_max_abs_arc_deg`     | `float`        | `15.0`                           | no                            |
| `dezinger_threshold`           | `float`/None   | `30000.0`                        | `w_proc_dezinger` (0 → None)  |
| `dezinger_kernel`              | `int`          | `5`                              | no                            |
| `waxs_cal_overrides`           | `dict`/None    | `None`                           | no — advanced                 |

### `WAXSCalibration` fields you can override via `waxs_cal_overrides`

These are the dataclass fields of `PyHyperScattering.SMISWAXSIntegrator.WAXSCalibration`:

```
energy_kev                : 16.1
sample_distance_mm        : 270.0   (combined-mode loader default is 274)
pixel_size_mm             : 0.172
beam_center_row           : 217.0
beam_center_col           : 319.0
panel_col_ranges          : ((0, 206), (206, 413), (413, 619))
panel_offsets_deg         : (-7.0, 0.0, 7.0)
panel_row_shifts          : (0.0, 0.0, 0.0)
panel_col_shifts          : (0.0, 0.0, 0.0)
panel_delta_deg           : (0.0, 0.0, 0.0)
theta_zero_deg            : 0.0
sample_offset_x_mm        : 0.0
sample_offset_z_mm        : 0.0   (combined-mode default uses 2.0)
beam_col_per_arc_deg      : 0.0
q_horizontal_sign         : -1.0
q_vertical_sign           : -1.0
rotation_k                : 3
```

These could be exposed in an "Advanced" expander but should default to
*not sent* (i.e. `waxs_cal_overrides=None` → upstream defaults).

---

## Additional defaults relevant to the loader

`PyHyperScattering.SMISWAXSLoader.TiledSMISWAXSLoader`:

| ctor kwarg     | default                              | Notes                                |
|----------------|--------------------------------------|--------------------------------------|
| `tiled_uri`    | `DEFAULT_TILED_URI`                  | match `smi_defaults`                 |
| `catalog`      | `DEFAULT_CATALOG`                    | match `smi_defaults`                 |
| `energy_kev`   | `None` → uses run metadata           | optional override                    |

The loader also auto-falls-back to per-frame chunked reads when a tiled
bulk read returns HTTP 500 — no browser change required (this fixes the
`pil2M_image expected_shape=7,1679,1475` 500 reported on 2026-04-28).

---

## Minimum acceptance criteria for the browser PR

1. `smi-browser/masks/` directory deleted.
2. `DEFAULT_SAXS_MASK` / `DEFAULT_WAXS_MASK` constants either removed or
   replaced with empty strings; widgets show the bundled-default name in
   their placeholder text.
3. `_on_process` passes `None` (or omits) for *every* parameter the user
   has not explicitly set (masks, beam deltas, distance delta, dezinger
   threshold, incident angle when "auto" is checked).
4. `pixi.toml` `[tasks]` and any docs that mention `masks/` are updated.
5. Browser still successfully reduces a known good UID
   (e.g. `ab058833-7f75-4179-832d-3a63e629b555`) end-to-end with all
   widgets at their defaults.

---

## Quick reference snippet — minimal call after the refactor

```python
from PyHyperScattering.SMISWAXSIntegrator import reduce_smi_combined
from PyHyperScattering.smi_defaults import DEFAULT_TILED_URI, DEFAULT_CATALOG

result = reduce_smi_combined(
    uid=uid,
    # Everything below this line is OPTIONAL — defaults are from upstream.
    tiled_uri=DEFAULT_TILED_URI,
    catalog=DEFAULT_CATALOG,
    n_q=2000,
    n_chi=360,
    solid_angle_correction=True,
    geometry="transmission",
    # Mask paths: omit / pass None to use the bundled defaults
    # saxs_mask_path=None,
    # waxs_mask_path=None,
    # Beam-center & distance overrides: omit unless user changed the UI
    # saxs_beam_delta_px=(dr, dc),
    # waxs_beam_delta_px=(dr, dc),
    # saxs_distance_delta_mm=delta,
    dezinger_threshold=3000.0,
)
```
