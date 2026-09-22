"""Display-only transformations for I(q) curves."""
from __future__ import annotations

import numpy as np


def scaled_iq_data(q, intensity, *, q_power=0.0, log_x=True, log_y=True):
    """Return finite q and I(q) * q**q_power samples valid for the axes.

    Negative q is outside the radial I(q) domain. Zero q is retained on
    linear axes when the chosen power is defined there. Inputs are not mutated.
    """
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = intensity * np.power(q, q_power)
    mask = np.isfinite(q) & (q >= 0) & np.isfinite(scaled)
    if log_x:
        mask &= q > 0
    if log_y:
        mask &= scaled > 0
    return q[mask], scaled[mask]


def iq_axis_label(q_power=0.0):
    """Label the displayed intensity, with q expressed in nm⁻¹."""
    return "I(q)" if q_power == 0 else f"I(q) × q^{q_power:g}"
