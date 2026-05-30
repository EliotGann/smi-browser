"""Detect whether a primary-stream scan landed on a regular (x, y) grid.

Phase 1 of the Primary-tab 2D plotting feature.  No Bokeh imports — this
module is pure numpy/pandas so it can be unit-tested without a display.

Two responsibilities:

1. :func:`pick_default_axes` — choose initial (X, Y, Z) selectors for the
   2D-plot widgets from the run's ``start`` metadata.  This is purely a
   hint for the UI; the user can override any of them.

2. :func:`gridded_view` — given the *actual* numeric values for the
   chosen X / Y / Z columns, decide empirically whether those values lie
   on a (mostly) regular 2-D grid.  If they do, build a ``(ny, nx)``
   image array (NaN-filled where cells are missing or duplicated); if
   they don't, return ``None`` and let the caller fall back to a scatter
   plot.

The detection is **column-agnostic**: it doesn't care whether X / Y are
the scanned motors or some other columns the user picked.  Snake scans,
partial grids, and reshuffled trajectories all work because the cell
position is read directly from each data point's (x, y) coordinates, not
from the row order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GriddedView",
    "gridded_view",
    "pick_default_axes",
]


# ---------------------------------------------------------------------------
# Empirical grid detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GriddedView:
    """Output of :func:`gridded_view` when the data lies on a regular grid.

    Attributes
    ----------
    image : np.ndarray
        ``(ny, nx)`` array of Z values, NaN-filled where no data point
        landed in that cell.
    x_centres : np.ndarray
        ``(nx,)`` array of unique X-cell centres in ascending order.
    y_centres : np.ndarray
        ``(ny,)`` array of unique Y-cell centres in ascending order.
    counts : np.ndarray
        ``(ny, nx)`` integer count of how many data points landed in each
        cell.  ``counts.max() > 1`` means cells were averaged together.
    n_filled : int
        Number of cells with at least one data point.
    n_total : int
        ``nx * ny``.
    n_points : int
        Number of input points.
    max_collision : int
        ``counts.max()``.  ``1`` means every filled cell got exactly one
        data point; higher means averaging occurred.
    """
    image: np.ndarray
    x_centres: np.ndarray
    y_centres: np.ndarray
    counts: np.ndarray
    n_filled: int
    n_total: int
    n_points: int
    max_collision: int

    @property
    def fill_fraction(self) -> float:
        return self.n_filled / self.n_total if self.n_total else 0.0


def _unique_within_tol(values: np.ndarray, tol: float) -> np.ndarray:
    """Cluster ``values`` to a tolerance, returning sorted cluster centres.

    Two values closer than ``tol`` collapse to their mean.  This handles
    floating-point jitter when a motor revisits "the same" position.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0:
        return v
    if tol <= 0:
        return np.unique(v)
    # Walk the sorted array, starting a new cluster whenever the next
    # value jumps by more than tol.
    breaks = np.concatenate(([True], np.diff(v) > tol))
    starts = np.where(breaks)[0]
    ends = np.concatenate((starts[1:], [v.size]))
    return np.array([v[s:e].mean() for s, e in zip(starts, ends)])


def _snap_to_centres(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Map each value to the index of its nearest cluster centre."""
    if centres.size == 1:
        return np.zeros(values.shape, dtype=int)
    edges = (centres[:-1] + centres[1:]) / 2.0
    return np.searchsorted(edges, values).astype(int)


def gridded_view(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    *,
    tol_frac: float = 1e-4,
    max_sparseness: float = 4.0,
    min_axis_size: int = 2,
    max_axis_size: int = 4096,
) -> GriddedView | None:
    """Try to organise ``(x, y, z)`` triples into a regular 2-D image.

    The algorithm is:

    1. Cluster X-values within ``tol_frac * span_x`` of each other; same
       for Y.  Call the cluster counts ``nx`` and ``ny``.
    2. If ``nx * ny > max_sparseness * len(x)``, the would-be image is
       mostly empty — return ``None`` (caller should scatter-plot).
       Also reject if ``nx < min_axis_size`` or ``ny < min_axis_size``
       (degenerate / 1-D scan) or either axis exceeds ``max_axis_size``.
    3. Assign each point to its nearest ``(ix, iy)`` cell, accumulate
       sums + counts, and divide.  Empty cells become NaN.

    Parameters
    ----------
    x, y, z
        1-D arrays of equal length.  Non-finite values are dropped before
        clustering.
    tol_frac
        Two X-values closer than ``tol_frac * (x.max() - x.min())`` are
        treated as the same cell centre.  Same for Y.  Default ``1e-4``
        is conservative — bumps up the floating-point noise floor far
        below typical motor step sizes.
    max_sparseness
        Reject the gridded view when the implied grid has more than this
        many cells per data point (``nx * ny / n_points``).  ``4.0`` =
        accept up to a 75 %-empty grid.
    min_axis_size, max_axis_size
        Reject when either axis falls outside this range.

    Returns
    -------
    GriddedView | None
        ``None`` when the data doesn't look like a grid.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if not (x_arr.shape == y_arr.shape == z_arr.shape):
        raise ValueError("x, y, z must have the same shape")
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite.any():
        return None
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    z_arr = z_arr[finite]

    span_x = float(x_arr.max() - x_arr.min())
    span_y = float(y_arr.max() - y_arr.min())
    if span_x <= 0 or span_y <= 0:
        return None

    tol_x = span_x * tol_frac
    tol_y = span_y * tol_frac
    x_centres = _unique_within_tol(x_arr, tol_x)
    y_centres = _unique_within_tol(y_arr, tol_y)
    nx, ny = x_centres.size, y_centres.size
    if nx < min_axis_size or ny < min_axis_size:
        return None
    if nx > max_axis_size or ny > max_axis_size:
        return None
    if nx * ny > max_sparseness * x_arr.size:
        return None

    ix = _snap_to_centres(x_arr, x_centres)
    iy = _snap_to_centres(y_arr, y_centres)

    sums = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=int)
    # NaN z-values shouldn't poison the cell — drop them per-point.
    finite_z = np.isfinite(z_arr)
    np.add.at(sums, (iy[finite_z], ix[finite_z]), z_arr[finite_z])
    np.add.at(counts, (iy[finite_z], ix[finite_z]), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(counts > 0, sums / counts, np.nan)

    n_filled = int((counts > 0).sum())
    return GriddedView(
        image=image,
        x_centres=x_centres,
        y_centres=y_centres,
        counts=counts,
        n_filled=n_filled,
        n_total=nx * ny,
        n_points=int(x_arr.size),
        max_collision=int(counts.max()) if counts.size else 0,
    )


# ---------------------------------------------------------------------------
# Default-axis picking
# ---------------------------------------------------------------------------

#: Substrings (case-insensitive) that bump a column up the Z-default list.
_Z_PRIORITY_HINTS = (
    "pin_diode", "monitor", "i_norm", "i0", "saxs_sum", "waxs_sum",
    "intensity", "_sum", "_total", "counts",
)


def _motors_from_start(start_md: dict | None) -> list[str]:
    """Read scanned-motor names from ``start.motors`` / ``start.hints``.

    Outer-first order (``[slow, ..., fast]``) is preserved; this matches
    Bluesky's convention for both ``start.motors`` and
    ``start.hints.dimensions``.
    """
    if not start_md:
        return []
    motors = start_md.get("motors") or []
    if motors:
        return [str(m) for m in motors]
    hints = start_md.get("hints") or {}
    dims = hints.get("dimensions") or []
    out: list[str] = []
    for entry in dims:
        try:
            names = entry[0]
        except (IndexError, TypeError):
            continue
        if not names:
            continue
        out.append(str(names[0]))
    return out


def pick_default_axes(
    df: pd.DataFrame,
    start_md: dict | None,
) -> tuple[str | None, str | None, str | None]:
    """Suggest initial (x, y, z) selections for the 2D-plot widgets.

    The returned columns are guaranteed to exist in ``df`` and to be
    numeric.  Any of them can be ``None`` if no suitable column exists.

    Defaults follow Bluesky's outer-first convention so the *inner* (fast)
    motor becomes X and the *next outer* motor becomes Y, which matches
    the natural reading direction of the resulting image.

    Z is the first numeric column that isn't a motor (preferring names
    matching :data:`_Z_PRIORITY_HINTS`).
    """
    if df is None or df.empty:
        return None, None, None
    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return None, None, None

    motors = _motors_from_start(start_md)
    motor_set = set(motors)

    # X = innermost (fast) motor, Y = next outer.  Fall back to the first
    # two numeric columns if the motors aren't in the dataframe.
    def _first_match(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in numeric_cols:
                return c
        return None

    x_default = _first_match(list(reversed(motors)))
    y_default = _first_match([m for m in reversed(motors) if m != x_default])

    if x_default is None:
        x_default = numeric_cols[0] if numeric_cols else None
    if y_default is None:
        rest = [c for c in numeric_cols if c != x_default]
        y_default = rest[0] if rest else None

    # Z: first non-motor numeric column, prioritising "intensity-like"
    # names.  Falls back to the first remaining numeric column.
    non_motor_cols = [c for c in numeric_cols if c not in motor_set
                      and c not in (x_default, y_default)]
    z_default: str | None = None
    for hint in _Z_PRIORITY_HINTS:
        for c in non_motor_cols:
            if hint in c.lower():
                z_default = c
                break
        if z_default is not None:
            break
    if z_default is None:
        z_default = non_motor_cols[0] if non_motor_cols else None
    # As a last resort, allow Z to be any numeric col not already used as
    # X or Y — even a motor (the user might genuinely want to colour by
    # motor position).
    if z_default is None:
        remaining = [c for c in numeric_cols if c not in (x_default, y_default)]
        z_default = remaining[0] if remaining else None

    return x_default, y_default, z_default
