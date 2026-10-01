# Calibration standards

Both **Calibrate SAXS** and **Calibrate WAXS** use the shared list in
`smi_browser/calibrants.json`. Choose a **Calibrant**, then a **Reflection / order**
for a single-ring fit, or select at least two reflections for a multi-ring fit.
The two-click q-range picker snaps to the nearest reflection of that standard.
Changing the standard clears previous fitted corrections so they cannot be
applied to the new selection.

Included standards:

| Standard | Reference peaks (Å⁻¹) |
| --- | --- |
| Silver behenate (AgBh) | Orders 1–9 of q₁ = 2π / 58.380 Å ≈ 0.107626 Å⁻¹ |
| Gold (Au) | 111: 2.668 Å⁻¹; 200: 3.081 Å⁻¹ |

**File values and reflection labels use Å⁻¹. Fit windows and q-χ plots use nm⁻¹.**
The loader converts automatically: the gold peaks are 26.68 and 30.81 nm⁻¹.
Select data whose processed q range covers the desired peaks. The gold references
are approximate values, not a temperature-specific certified standard.

## Add a calibrant

Add an entry to the bundled JSON list, or put your own list in a separate file
and set `SMI_BROWSER_CALIBRANTS_FILE=/path/to/my_calibrants.json` before launching
the app. Additional entries are merged with bundled standards; an identical `id`
overrides that standard. Restart the app after editing the list.

Lamellar standards need a fundamental q and an optional maximum order (default 9):

```json
[
  {
    "id": "my_lamellar_standard",
    "name": "My lamellar standard",
    "q1_angstrom_inv": 0.12,
    "max_order": 6
  }
]
```

For other systems supply labeled peaks in strictly increasing q order:

```json
[
  {
    "id": "my_peak_standard",
    "name": "My peak standard",
    "peaks": [
      {"label": "111", "q_angstrom_inv": 2.668},
      {"label": "200", "q_angstrom_inv": 3.081}
    ]
  }
]
```

Use one representation per entry. Values must be finite and positive, with unique
IDs, display names, and reflection labels. Fitting uses the supplied references
for both peak windows and distance ratios, without assuming harmonic spacing.
The existing approximate beam-offset model and WAXS per-panel limitations still
apply; reprocess and refit after applying a correction.
