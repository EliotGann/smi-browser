"""Per-frame peak fitting across a stack of 1D I(q) curves.

A processed length/raster scan produces one I(q) curve per frame
(``per_frame_iq`` / cached ``pf_iq_I`` with shape ``(n_frames, n_q)``).  This
module fits a single peak model within a user-chosen q-range to **every** frame
and returns the per-frame fit parameters as flat arrays suitable for driving a
spatial map.

The module is deliberately free of Panel/Bokeh imports so it can be unit
tested and run on a background thread.  Fitting follows the same Gaussian /
``scipy.optimize.curve_fit`` style as ``_fit_gaussian`` in ``smi_app.py`` but
adds a Lorentzian model and an optional linear baseline, and is hardened for
batch use (vectorised initial guesses, bounded fits, NaN on failure, and
cancellation / progress hooks).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "PeakDef",
    "FIT_PARAMS",
    "fit_peak_across_frames",
]

#: Per-frame scalar outputs produced for every fitted peak.
FIT_PARAMS: tuple[str, ...] = ("amplitude", "center", "fwhm", "area")

_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548...


@dataclass(frozen=True)
class PeakDef:
    """A single peak to fit within ``[q_min, q_max]``.

    ``model`` is ``"gaussian"`` or ``"lorentzian"``; ``baseline`` is ``"none"``
    or ``"linear"`` (a sloping ``a*q + b`` background subtracted as part of the
    fit).
    """

    name: str
    q_min: float
    q_max: float
    model: str = "gaussian"
    baseline: str = "linear"

    def key(self) -> tuple:
        """Hashable identity used to cache fit results."""
        return (
            round(float(self.q_min), 6),
            round(float(self.q_max), 6),
            self.model,
            self.baseline,
        )


# --- models ----------------------------------------------------------------

def _gaussian(q, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((q - mu) / sigma) ** 2)


def _lorentzian(q, amp, mu, gamma):
    # gamma is the half-width at half-maximum; peak height is amp.
    return amp * (gamma * gamma) / ((q - mu) ** 2 + gamma * gamma)


def _make_model(model: str, with_baseline: bool):
    """Return a ``curve_fit``-compatible callable ``f(q, amp, mu, w[, m, b])``."""
    peak = _gaussian if model == "gaussian" else _lorentzian
    if with_baseline:
        def f(q, amp, mu, w, m, b):
            return peak(q, amp, mu, w) + m * q + b
    else:
        def f(q, amp, mu, w):
            return peak(q, amp, mu, w)
    return f


def _fwhm(model: str, width: float) -> float:
    if model == "gaussian":
        return abs(width) * _GAUSS_FWHM
    return abs(width) * 2.0  # lorentzian: FWHM = 2*gamma


def _area(model: str, amp: float, width: float) -> float:
    if model == "gaussian":
        return abs(amp) * abs(width) * np.sqrt(2.0 * np.pi)
    return abs(amp) * np.pi * abs(width)  # lorentzian: amp * pi * gamma


# --- driver ----------------------------------------------------------------

def fit_peak_across_frames(
    q: Sequence[float],
    iq: np.ndarray,
    peak: PeakDef,
    *,
    cancel=None,
    progress: Callable[[int, int], None] | None = None,
    maxfev: int = 2000,
    cancel_check_every: int = 64,
) -> dict[str, np.ndarray]:
    """Fit ``peak`` to every frame's I(q) curve.

    Parameters
    ----------
    q : 1-D q axis, shape ``(n_q,)``.
    iq : per-frame intensities, shape ``(n_frames, n_q)``.
    peak : the peak definition (range, model, baseline).
    cancel : optional object with ``is_set()`` (e.g. ``threading.Event``).
        Checked every ``cancel_check_every`` frames; if set, fitting stops and
        the remaining frames are left as NaN.
    progress : optional ``progress(done, total)`` callback.

    Returns
    -------
    dict with keys ``amplitude``, ``center``, ``fwhm``, ``area`` (float arrays,
    NaN where the fit failed/was skipped) and ``success`` (bool array), each of
    shape ``(n_frames,)``.
    """
    q = np.asarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)
    if iq.ndim == 1:
        iq = iq[None, :]
    n_frames = iq.shape[0]

    out = {p: np.full(n_frames, np.nan, dtype=float) for p in FIT_PARAMS}
    out["success"] = np.zeros(n_frames, dtype=bool)

    mask = np.isfinite(q) & (q >= peak.q_min) & (q <= peak.q_max)
    qs = q[mask]
    if qs.size < 4:
        if progress is not None:
            progress(n_frames, n_frames)
        return out

    ys_all = iq[:, mask]  # (n_frames, m)
    with_baseline = peak.baseline == "linear"

    # Vectorised initial guesses across all frames.
    q0, qN = float(qs[0]), float(qs[-1])
    span = max(qN - q0, 1e-12)
    y0 = ys_all[:, 0]
    yN = ys_all[:, -1]
    slope0 = (yN - y0) / span
    intercept0 = y0 - slope0 * q0
    baseline_line = slope0[:, None] * qs[None, :] + intercept0[:, None]
    resid = ys_all - baseline_line          # peak above a straight background
    peak_idx = np.nanargmax(np.where(np.isfinite(resid), resid, -np.inf), axis=1)
    amp0 = resid[np.arange(n_frames), peak_idx]
    amp0 = np.where(np.isfinite(amp0) & (amp0 > 0), amp0, np.nanmax(np.abs(resid)) + 1e-9)
    mu0 = qs[peak_idx]
    width0 = max(span / 6.0, 1e-9)

    from scipy.optimize import curve_fit

    func = _make_model(peak.model, with_baseline)
    lo = [0.0, q0, 1e-9]
    hi = [np.inf, qN, span]
    if with_baseline:
        lo += [-np.inf, -np.inf]
        hi += [np.inf, np.inf]
    bounds = (lo, hi)

    for i in range(n_frames):
        if cancel is not None and (i % cancel_check_every == 0) and cancel.is_set():
            break
        yi = ys_all[i]
        good = np.isfinite(yi)
        if int(good.sum()) < 4:
            continue
        a0 = float(amp0[i]) if np.isfinite(amp0[i]) and amp0[i] > 0 else 1e-9
        m0 = float(mu0[i]) if q0 <= mu0[i] <= qN else 0.5 * (q0 + qN)
        p0 = [a0, m0, width0]
        if with_baseline:
            p0 += [float(slope0[i]) if np.isfinite(slope0[i]) else 0.0,
                   float(intercept0[i]) if np.isfinite(intercept0[i]) else 0.0]
        try:
            popt, _ = curve_fit(
                func, qs[good], yi[good], p0=p0, bounds=bounds, maxfev=maxfev,
            )
        except Exception:
            continue
        amp, mu, width = popt[0], popt[1], popt[2]
        out["amplitude"][i] = amp
        out["center"][i] = mu
        out["fwhm"][i] = _fwhm(peak.model, width)
        out["area"][i] = _area(peak.model, amp, width)
        out["success"][i] = True
        if progress is not None and (i % cancel_check_every == 0):
            progress(i + 1, n_frames)

    if progress is not None:
        progress(n_frames, n_frames)
    return out
