"""Per-frame peak fitting across a stack of 1D I(q) curves.

A processed length/raster scan produces one I(q) curve per frame
(``per_frame_iq`` / cached ``pf_iq_I`` with shape ``(n_frames, n_q)``).  This
module fits a single peak model within a user-chosen q-range to **every** frame
and returns the per-frame fit parameters as flat arrays suitable for driving a
spatial map.

The module is deliberately free of Panel/Bokeh imports so it can be unit
tested and run on a background thread.  Fitting uses
``scipy.optimize.curve_fit`` with a Gaussian / Lorentzian peak and an optional
linear baseline, hardened for batch use:

* **Bounded width** — the fitted FWHM cannot exceed the drawn q-range, so a
  frame with no real peak can no longer "explode" its width to fill the window.
* **Quality gate** — each fit is scored by signal-to-noise and R²; a frame that
  fails the gate is reported as *no peak* (amplitude/area ``0``, centre/FWHM
  ``NaN``) rather than a runaway fit.
* **Wider background** — for a linear baseline the fit window is widened to
  ``bg_factor`` × the drawn range so the slope/intercept are anchored by the
  peak's flanks, while the peak *centre* stays locked to the drawn range.
* **Linking across frames** — ``link`` selects how the peak position/width
  relate across the stack: ``independent`` (per-frame), ``linked`` (centre &
  width shared from a robust aggregate fit; only amplitude varies per frame),
  or ``tracked`` (each frame warm-starts from the previous one).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "PeakDef",
    "FIT_PARAMS",
    "MIN_SNR",
    "MIN_R2",
    "fit_peak_across_frames",
]

#: Per-frame scalar outputs produced for every fitted peak.
FIT_PARAMS: tuple[str, ...] = ("amplitude", "center", "fwhm", "area")

_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548...

#: Quality-gate thresholds.  A fit is accepted only when its peak rises clearly
#: above the residual noise (``MIN_SNR``) and the model explains the windowed
#: data (``MIN_R2``).  ``WIDTH_BOUND_TOL`` rejects a fit whose width pins the
#: upper bound — a sign the "peak" is really just filling the window.
MIN_SNR = 3.0
MIN_R2 = 0.2
WIDTH_BOUND_TOL = 0.97


@dataclass(frozen=True)
class PeakDef:
    """A single peak to fit within ``[q_min, q_max]``.

    ``model`` is ``"gaussian"`` or ``"lorentzian"``; ``baseline`` is ``"none"``
    or ``"linear"`` (a sloping ``a*q + b`` background fitted alongside the peak).
    ``link`` is ``"independent"`` / ``"linked"`` / ``"tracked"`` (see module
    docstring).  ``bg_factor`` widens the fit window (for a linear baseline)
    to ``bg_factor`` × the drawn range so the baseline is anchored by the
    peak's flanks.
    """

    name: str
    q_min: float
    q_max: float
    model: str = "gaussian"
    baseline: str = "linear"
    link: str = "independent"
    bg_factor: float = 2.0

    def key(self) -> tuple:
        """Hashable identity used to cache fit results."""
        return (
            round(float(self.q_min), 6),
            round(float(self.q_max), 6),
            self.model,
            self.baseline,
            self.link,
            round(float(self.bg_factor), 3),
        )


# --- models ----------------------------------------------------------------

def _gaussian(q, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((q - mu) / sigma) ** 2)


def _lorentzian(q, amp, mu, gamma):
    # gamma is the half-width at half-maximum; peak height is amp.
    return amp * (gamma * gamma) / ((q - mu) ** 2 + gamma * gamma)


def _peak_shape(model: str):
    return _gaussian if model == "gaussian" else _lorentzian


def _make_model(model: str, with_baseline: bool):
    """Return a ``curve_fit``-compatible callable ``f(q, amp, mu, w[, m, b])``."""
    peak = _peak_shape(model)
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


def _width_bounds(model: str, core_range: float, dq: float) -> tuple[float, float]:
    """``(w_min, w_max)`` so the fitted FWHM stays within the drawn range."""
    if model == "gaussian":
        w_max = core_range / _GAUSS_FWHM
    else:
        w_max = core_range / 2.0
    w_max = max(w_max, 1e-9)
    w_min = max(dq, 1e-9)
    if w_min >= w_max:
        w_min = w_max * 1e-3
    return w_min, w_max


# --- quality gate -----------------------------------------------------------

def _score(qs, yi, model, with_baseline, popt) -> tuple[float, float, float]:
    """Return ``(amp, snr, r2)`` for a fitted curve over ``qs``/``yi``."""
    func = _make_model(model, with_baseline)
    pred = func(qs, *popt)
    resid = yi - pred
    ss_res = float(np.nansum(resid ** 2))
    mean_y = float(np.nanmean(yi))
    ss_tot = float(np.nansum((yi - mean_y) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = max(int(np.isfinite(resid).sum()), 1)
    noise = float(np.sqrt(ss_res / n))
    amp = float(popt[0])
    snr = amp / noise if noise > 0 else (np.inf if amp > 0 else 0.0)
    return amp, snr, r2


def _accept(amp, snr, r2, width, w_max) -> bool:
    return (
        amp > 0
        and snr >= MIN_SNR
        and r2 >= MIN_R2
        and abs(width) < WIDTH_BOUND_TOL * w_max
    )


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
    peak : the peak definition (range, model, baseline, link, bg_factor).
    cancel : optional object with ``is_set()`` (e.g. ``threading.Event``).
        Checked every ``cancel_check_every`` frames; if set, fitting stops and
        the remaining frames are left as NaN.
    progress : optional ``progress(done, total)`` callback.

    Returns
    -------
    dict with keys ``amplitude``, ``center``, ``fwhm``, ``area`` (float arrays;
    a frame that fails the quality gate reports amplitude/area ``0`` and
    centre/FWHM ``NaN``; a frame that cannot be fit at all is fully ``NaN``)
    and ``success`` (bool array), each of shape ``(n_frames,)``.  An optional
    ``note`` (str) key reports a fallback (e.g. linked → independent).
    """
    q = np.asarray(q, dtype=float)
    iq = np.asarray(iq, dtype=float)
    if iq.ndim == 1:
        iq = iq[None, :]
    n_frames = iq.shape[0]

    out = {p: np.full(n_frames, np.nan, dtype=float) for p in FIT_PARAMS}
    out["success"] = np.zeros(n_frames, dtype=bool)

    # --- fit window -------------------------------------------------------
    # The *core* is the drawn range (peak centre is constrained here).  For a
    # linear baseline we widen the window to bg_factor × the core so the slope
    # and intercept are informed by the peak's flanks.
    with_baseline = peak.baseline == "linear"
    core_lo, core_hi = float(peak.q_min), float(peak.q_max)
    core_range = max(core_hi - core_lo, 1e-12)
    if with_baseline and peak.bg_factor and peak.bg_factor > 1.0:
        mid = 0.5 * (core_lo + core_hi)
        half = 0.5 * core_range * float(peak.bg_factor)
        win_lo, win_hi = mid - half, mid + half
    else:
        win_lo, win_hi = core_lo, core_hi

    mask = np.isfinite(q) & (q >= win_lo) & (q <= win_hi)
    qs = q[mask]
    if qs.size < 4:
        if progress is not None:
            progress(n_frames, n_frames)
        return out

    ys_all = iq[:, mask]  # (n_frames, m)
    dq = float(np.median(np.diff(qs))) if qs.size > 1 else core_range
    w_min, w_max = _width_bounds(peak.model, core_range, dq)
    width0 = float(np.clip(core_range / 4.0, w_min, w_max))

    if peak.link == "linked":
        return _fit_linked(
            qs, ys_all, peak, with_baseline, core_lo, core_hi,
            w_min, w_max, width0, out, n_frames,
            cancel=cancel, progress=progress, maxfev=maxfev,
            cancel_check_every=cancel_check_every,
        )
    return _fit_per_frame(
        qs, ys_all, peak, with_baseline, core_lo, core_hi,
        w_min, w_max, width0, out, n_frames,
        cancel=cancel, progress=progress, maxfev=maxfev,
        cancel_check_every=cancel_check_every,
    )


def _bounds(with_baseline, core_lo, core_hi, w_min, w_max):
    lo = [0.0, core_lo, w_min]
    hi = [np.inf, core_hi, w_max]
    if with_baseline:
        lo += [-np.inf, -np.inf]
        hi += [np.inf, np.inf]
    return (lo, hi)


def _initial_guesses(qs, ys_all, core_lo, core_hi, width0):
    """Vectorised per-frame initial guesses over the fit window."""
    n_frames = ys_all.shape[0]
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
    fallback = np.nanmax(np.abs(resid)) + 1e-9
    amp0 = np.where(np.isfinite(amp0) & (amp0 > 0), amp0, fallback)
    mu0 = qs[peak_idx]
    # Clamp the centre guess into the drawn core.
    mid = 0.5 * (core_lo + core_hi)
    mu0 = np.where((mu0 >= core_lo) & (mu0 <= core_hi), mu0, mid)
    return amp0, mu0, slope0, intercept0


def _fit_per_frame(qs, ys_all, peak, with_baseline, core_lo, core_hi,
                   w_min, w_max, width0, out, n_frames, *,
                   cancel, progress, maxfev, cancel_check_every):
    """``independent`` / ``tracked`` fitting (per-frame ``curve_fit``)."""
    from scipy.optimize import curve_fit

    func = _make_model(peak.model, with_baseline)
    bounds = _bounds(with_baseline, core_lo, core_hi, w_min, w_max)
    amp0, mu0, slope0, intercept0 = _initial_guesses(
        qs, ys_all, core_lo, core_hi, width0)
    tracked = peak.link == "tracked"
    prev_popt = None

    for i in range(n_frames):
        if cancel is not None and (i % cancel_check_every == 0) and cancel.is_set():
            break
        yi = ys_all[i]
        good = np.isfinite(yi)
        if int(good.sum()) < 4:
            continue
        if tracked and prev_popt is not None:
            p0 = list(prev_popt)
        else:
            a0 = float(amp0[i]) if np.isfinite(amp0[i]) and amp0[i] > 0 else 1e-9
            m0 = float(mu0[i]) if core_lo <= mu0[i] <= core_hi else 0.5 * (core_lo + core_hi)
            p0 = [a0, m0, width0]
            if with_baseline:
                p0 += [float(slope0[i]) if np.isfinite(slope0[i]) else 0.0,
                       float(intercept0[i]) if np.isfinite(intercept0[i]) else 0.0]
        try:
            popt, _ = curve_fit(
                func, qs[good], yi[good], p0=p0, bounds=bounds, maxfev=maxfev,
            )
        except Exception:
            continue  # structural failure → leave NaN
        amp, snr, r2 = _score(qs[good], yi[good], peak.model, with_baseline, popt)
        width = popt[2]
        if _accept(amp, snr, r2, width, w_max):
            mu = popt[1]
            out["amplitude"][i] = amp
            out["center"][i] = mu
            out["fwhm"][i] = _fwhm(peak.model, width)
            out["area"][i] = _area(peak.model, amp, width)
            out["success"][i] = True
            prev_popt = popt
        else:
            # Ran but no significant peak → report zero amplitude/area.
            out["amplitude"][i] = 0.0
            out["area"][i] = 0.0
            # centre / fwhm stay NaN; success stays False
        if progress is not None and (i % cancel_check_every == 0):
            progress(i + 1, n_frames)

    if progress is not None:
        progress(n_frames, n_frames)
    return out


def _fit_linked(qs, ys_all, peak, with_baseline, core_lo, core_hi,
               w_min, w_max, width0, out, n_frames, *,
               cancel, progress, maxfev, cancel_check_every):
    """``linked`` fitting: shared centre & width from a robust aggregate fit,
    then a fast linear amplitude (+baseline) solve per frame."""
    from scipy.optimize import curve_fit

    # 1) Aggregate curve — mean over frames (ignoring NaNs).
    agg = np.nanmean(ys_all, axis=0)
    good_agg = np.isfinite(agg)
    center_star = width_star = None
    if int(good_agg.sum()) >= 4:
        func = _make_model(peak.model, with_baseline)
        bounds = _bounds(with_baseline, core_lo, core_hi, w_min, w_max)
        a0, m0, s0, b0 = _initial_guesses(qs, agg[None, :], core_lo, core_hi, width0)
        p0 = [float(a0[0]), float(m0[0]), width0]
        if with_baseline:
            p0 += [float(s0[0]), float(b0[0])]
        try:
            popt, _ = curve_fit(func, qs[good_agg], agg[good_agg],
                                p0=p0, bounds=bounds, maxfev=maxfev)
            amp, snr, r2 = _score(qs[good_agg], agg[good_agg], peak.model,
                                  with_baseline, popt)
            if _accept(amp, snr, r2, popt[2], w_max):
                center_star, width_star = float(popt[1]), float(popt[2])
        except Exception:
            pass

    if center_star is None:
        # No usable aggregate peak → fall back to independent per-frame fits.
        res = _fit_per_frame(
            qs, ys_all, peak, with_baseline, core_lo, core_hi,
            w_min, w_max, width0, out, n_frames,
            cancel=cancel, progress=progress, maxfev=maxfev,
            cancel_check_every=cancel_check_every,
        )
        res["note"] = ("linked: no peak in aggregate curve — fell back to "
                       "independent per-frame fits")
        return res

    # 2) Fixed-shape linear solve per frame.  With centre & width fixed the
    # model is linear in (amp, slope, intercept): A @ coeffs ≈ y.
    shape = _peak_shape(peak.model)(qs, 1.0, center_star, width_star)
    if with_baseline:
        cols = [shape, qs, np.ones_like(qs)]
    else:
        cols = [shape]
    A_full = np.column_stack(cols)
    fwhm_star = _fwhm(peak.model, width_star)

    for i in range(n_frames):
        if cancel is not None and (i % cancel_check_every == 0) and cancel.is_set():
            break
        yi = ys_all[i]
        good = np.isfinite(yi)
        if int(good.sum()) < A_full.shape[1] + 1:
            continue  # structural failure → leave NaN
        A = A_full[good]
        try:
            coeffs, *_ = np.linalg.lstsq(A, yi[good], rcond=None)
        except Exception:
            continue
        amp = float(coeffs[0])
        amp, snr, r2 = _score_linear(A, yi[good], coeffs, amp)
        if amp > 0 and snr >= MIN_SNR and r2 >= MIN_R2:
            out["amplitude"][i] = amp
            out["center"][i] = center_star
            out["fwhm"][i] = fwhm_star
            out["area"][i] = _area(peak.model, amp, width_star)
            out["success"][i] = True
        else:
            out["amplitude"][i] = 0.0
            out["area"][i] = 0.0
        if progress is not None and (i % cancel_check_every == 0):
            progress(i + 1, n_frames)

    if progress is not None:
        progress(n_frames, n_frames)
    return out


def _score_linear(A, y, coeffs, amp) -> tuple[float, float, float]:
    """``(amp, snr, r2)`` for the linear (fixed-shape) linked solve."""
    pred = A @ coeffs
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    noise = float(np.sqrt(ss_res / max(y.size, 1)))
    snr = amp / noise if noise > 0 else (np.inf if amp > 0 else 0.0)
    return amp, snr, r2
