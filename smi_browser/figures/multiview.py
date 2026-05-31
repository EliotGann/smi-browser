"""Multi-view grid builder — pure geometry and Bokeh figure construction."""
from __future__ import annotations

import numpy as np


def grid_dims(n: int) -> tuple[int, int]:
    """Pick (rows, cols) for a near-square, space-filling grid.

    Columns = ceil(sqrt(n)) so the layout forms a tight rectangle (e.g. 64
    frames → 8×8) rather than a wide ragged block with a half-empty last row.
    """
    if n <= 0:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def compute_data_range(frames) -> tuple[float, float]:
    """Return (lo, hi) finite-positive percentile bounds across all frames."""
    lo, hi = None, None
    for arr in frames:
        finite = arr[np.isfinite(arr) & (arr > 0)]
        if not finite.size:
            continue
        flo = float(np.percentile(finite, 1))
        fhi = float(np.percentile(finite, 99.5))
        if lo is None or flo < lo:
            lo = flo
        if hi is None or fhi > hi:
            hi = fhi
    if lo is None:
        return 1e-3, 1.0
    lo = max(lo, 1e-6)
    if hi <= lo:
        hi = lo * 10
    return lo, hi
