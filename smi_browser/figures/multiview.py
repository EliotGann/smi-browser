"""Multi-view grid builder — pure geometry and Bokeh figure construction."""
from __future__ import annotations

import numpy as np


def grid_dims(n: int) -> tuple[int, int]:
    """Pick (rows, cols) so cols/rows ≈ 2 (approx 2:1 grid aspect)."""
    if n <= 0:
        return 1, 1
    rows = max(1, int(round(np.sqrt(n / 2.0))))
    cols = int(np.ceil(n / rows))
    while rows * cols < n:
        cols += 1
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
