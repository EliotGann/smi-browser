"""AgBh ring-based beam-centre / distance calibration.

The math layer is pure numpy/xarray and knows nothing about Bokeh or Panel.
Phase 1 fits a *single* AgBh ring per detector:

1. ``fit_ring_peaks`` slices the supplied ``q-vs-chi`` map inside a
   ``(q_min, q_max, chi_min, chi_max)`` window and fits a Gaussian-on-poly
   baseline for each chi row.  Slices that fail to converge, are too wide,
   or have SNR below threshold are dropped.

2. ``fit_beam_offset_qspace`` does a linear least-squares fit of

       q_peak(chi) = q0 + A_r * sin(chi) + A_c * cos(chi)

   The coefficients are pure q-shifts; no detector geometry needed.

3. ``q_offset_to_pixel_delta`` converts ``(A_r, A_c, q0)`` plus
   detector geometry ``(wavelength_nm, distance_mm, pixel_mm)`` into
   pixel beam-centre offsets and (optionally) a distance correction
   given an expected ring q.

Conventions
-----------
* ``chi`` is in degrees.  ``q`` is in nm⁻¹.  Pixel size and distance are
  in mm.  Wavelength is in nm.
* AgBh d-spacing taken as 5.8380 nm (q1 = 2π/d = 1.07614 nm⁻¹).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "AGBH_Q1_NM",
    "agbh_q",
    "nearest_agbh_order",
    "PeakFitResult",
    "BeamOffsetResult",
    "MultiRingResult",
    "fit_ring_peaks",
    "fit_beam_offset_qspace",
    "fit_multi_ring",
    "q_offset_to_pixel_delta",
    "q_offset_to_pixel_delta_multi",
]

# Silver behenate d_001 = 58.380 Å  →  q_1 = 2π/d
AGBH_Q1_NM: float = 2.0 * np.pi / 5.8380  # ≈ 1.07614 nm⁻¹


def agbh_q(order: int) -> float:
    """Expected q (nm⁻¹) of the n-th AgBh ring (n ≥ 1)."""
    if order < 1:
        raise ValueError(f"AgBh ring order must be ≥ 1, got {order}")
    return AGBH_Q1_NM * order


def nearest_agbh_order(q_value: float, max_order: int = 9) -> int:
    """Return the AgBh ring order closest to ``q_value`` (nm⁻¹)."""
    n = round(float(q_value) / AGBH_Q1_NM)
    return int(np.clip(n, 1, max_order))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeakFitResult:
    """Per-chi-slice peak fit results.

    Arrays are aligned 1-to-1 with ``chi_deg`` (i.e. one entry per accepted
    chi row); failed slices are dropped before construction.
    """
    chi_deg: np.ndarray            # accepted chi values (deg)
    q_peak: np.ndarray             # fitted Gaussian centre (nm⁻¹)
    sigma: np.ndarray              # Gaussian width (nm⁻¹)
    amplitude: np.ndarray          # Gaussian amplitude (intensity units)
    snr: np.ndarray                # amplitude / baseline RMS
    n_accepted: int
    n_total: int
    q_window: tuple[float, float]
    chi_window: tuple[float, float]


@dataclass(frozen=True)
class BeamOffsetResult:
    """Output of the linear sinusoid fit on ``q_peak(chi)``."""
    q0: float                      # mean fitted ring q (nm⁻¹)
    A_r: float                     # amplitude of sin(chi) term (nm⁻¹)
    A_c: float                     # amplitude of cos(chi) term (nm⁻¹)
    rms: float                     # residual RMS (nm⁻¹)
    n: int                         # number of chi slices used
    # Converted to pixel / mm by q_offset_to_pixel_delta(); None until then.
    drow_px: float | None = None
    dcol_px: float | None = None
    ddist_mm: float | None = None
    ring_q_expected: float | None = None


# ---------------------------------------------------------------------------
# Per-chi Gaussian peak fit
# ---------------------------------------------------------------------------

def _gauss_on_poly(q, amp, mu, sigma, *poly_coefs):
    """Gaussian + polynomial baseline.  ``poly_coefs`` is highest-order first."""
    baseline = np.polyval(poly_coefs, q)
    return amp * np.exp(-0.5 * ((q - mu) / sigma) ** 2) + baseline


def _fit_single_slice(
    q: np.ndarray,
    intensity: np.ndarray,
    bg_order: int,
) -> tuple[float, float, float, float] | None:
    """Fit one chi slice. Returns (mu, sigma, amplitude, snr) or None."""
    from scipy.optimize import curve_fit

    good = np.isfinite(intensity)
    if good.sum() < 6:
        return None
    qf = q[good]
    yf = intensity[good]
    if yf.max() == yf.min():
        return None

    # Initial estimates from data
    i_max = int(np.argmax(yf))
    mu0 = float(qf[i_max])
    span = float(qf[-1] - qf[0])
    sigma0 = max(span / 20.0, 1e-6)
    # Baseline guess: linear poly through endpoints
    bg_init = np.polyfit(qf, yf, bg_order)
    amp0 = float(yf[i_max] - np.polyval(bg_init, mu0))
    if amp0 <= 0:
        return None
    p0 = [amp0, mu0, sigma0, *bg_init.tolist()]

    # Bounds: amplitude ≥ 0, mu inside window, sigma > 0 and < window/2.
    lower = [0.0, qf[0], 1e-9] + [-np.inf] * (bg_order + 1)
    upper = [np.inf, qf[-1], span / 2.0] + [np.inf] * (bg_order + 1)

    try:
        popt, _ = curve_fit(
            _gauss_on_poly, qf, yf, p0=p0, bounds=(lower, upper),
            maxfev=400,
        )
    except (RuntimeError, ValueError):
        return None
    amp, mu, sigma, *poly = popt
    if not np.isfinite([amp, mu, sigma]).all():
        return None
    # SNR from baseline residual.
    resid = yf - _gauss_on_poly(qf, *popt)
    base_rms = float(np.sqrt(np.mean(resid ** 2)))
    snr = float(amp / base_rms) if base_rms > 0 else float("inf")
    return float(mu), float(sigma), float(amp), snr


def fit_ring_peaks(
    qchi,
    *,
    q_min: float,
    q_max: float,
    chi_min: float = -180.0,
    chi_max: float = 180.0,
    bg_order: int = 1,
    snr_threshold: float = 3.0,
    max_sigma_frac: float = 0.5,
    chi_stride: int = 1,
) -> PeakFitResult:
    """Fit one Gaussian peak per chi row inside the q-window.

    Parameters
    ----------
    qchi
        Either an ``xr.Dataset`` carrying ``intensity(q, chi)`` (the shape
        returned by smi-tiled in ``result.merged_qchi``) or an
        ``xr.DataArray`` named ``intensity`` with dims ``(q, chi)`` or
        ``(chi, q)``.
    q_min, q_max
        Radial window for the ring (nm⁻¹).  Must satisfy ``q_min < q_max``.
    chi_min, chi_max
        Azimuthal window (degrees).  Chi rows outside the window are
        ignored; pass full ``(-180, 180)`` to use everything.
    bg_order
        Polynomial order for the baseline under the Gaussian.  ``0`` =
        constant offset, ``1`` = linear (default), ``2`` = quadratic.
    snr_threshold
        Drop slices whose fitted amplitude / baseline-RMS is below this.
    max_sigma_frac
        Drop slices whose fitted Gaussian sigma exceeds this fraction of
        the q-window width — guards against runaway fits.
    chi_stride
        Take every n-th chi row (≥1).  Useful for fast previews.

    Returns
    -------
    PeakFitResult
    """
    import xarray as xr

    if q_min >= q_max:
        raise ValueError(f"q_min ({q_min}) must be < q_max ({q_max})")
    if chi_stride < 1:
        raise ValueError(f"chi_stride must be ≥ 1, got {chi_stride}")

    if isinstance(qchi, xr.Dataset):
        if "intensity" not in qchi.data_vars:
            raise ValueError("qchi Dataset has no 'intensity' variable")
        da = qchi["intensity"]
    else:
        da = qchi

    if "q" not in da.coords or "chi" not in da.coords:
        raise ValueError("qchi must have 'q' and 'chi' coordinates")

    # Reorder to (chi, q) and reduce frame dim if present.
    if "frame" in da.dims:
        da = da.mean(dim="frame", skipna=True)
    if list(da.dims) != ["chi", "q"]:
        da = da.transpose("chi", "q")

    q_all = np.asarray(da["q"].values, dtype=float)
    chi_all = np.asarray(da["chi"].values, dtype=float)

    q_mask = (q_all >= q_min) & (q_all <= q_max)
    if q_mask.sum() < 8:
        raise ValueError(
            f"q window [{q_min}, {q_max}] only covers {q_mask.sum()} samples "
            f"(need ≥ 8 for a fit)"
        )
    chi_mask = (chi_all >= chi_min) & (chi_all <= chi_max)
    if chi_mask.sum() < 4:
        raise ValueError(
            f"chi window [{chi_min}, {chi_max}] only covers {chi_mask.sum()} "
            f"rows (need ≥ 4)"
        )

    qw = q_all[q_mask]
    iw = np.asarray(da.values, dtype=float)[:, q_mask]
    chi_idx = np.where(chi_mask)[0][::chi_stride]

    max_sigma = (q_max - q_min) * max_sigma_frac
    chi_kept, mu_kept, sig_kept, amp_kept, snr_kept = [], [], [], [], []
    for j in chi_idx:
        fit = _fit_single_slice(qw, iw[j], bg_order=bg_order)
        if fit is None:
            continue
        mu, sigma, amp, snr = fit
        if sigma > max_sigma:
            continue
        if snr < snr_threshold:
            continue
        chi_kept.append(chi_all[j])
        mu_kept.append(mu)
        sig_kept.append(sigma)
        amp_kept.append(amp)
        snr_kept.append(snr)

    return PeakFitResult(
        chi_deg=np.asarray(chi_kept, dtype=float),
        q_peak=np.asarray(mu_kept, dtype=float),
        sigma=np.asarray(sig_kept, dtype=float),
        amplitude=np.asarray(amp_kept, dtype=float),
        snr=np.asarray(snr_kept, dtype=float),
        n_accepted=len(chi_kept),
        n_total=len(chi_idx),
        q_window=(float(q_min), float(q_max)),
        chi_window=(float(chi_min), float(chi_max)),
    )


# ---------------------------------------------------------------------------
# Sinusoid fit  →  q-space beam offset
# ---------------------------------------------------------------------------

def fit_beam_offset_qspace(
    chi_deg: Sequence[float],
    q_peak: Sequence[float],
    *,
    ring_q_expected: float | None = None,
) -> BeamOffsetResult:
    """Fit ``q_peak(chi) = q0 + A_r·sin(chi) + A_c·cos(chi)`` by linear lstsq.

    Parameters
    ----------
    chi_deg, q_peak
        Output of :func:`fit_ring_peaks` (already filtered by SNR).
    ring_q_expected
        Optional expected ring q (nm⁻¹) — if given, also stored on the
        result so a downstream converter can solve for distance.

    Returns
    -------
    BeamOffsetResult
        ``drow_px / dcol_px / ddist_mm`` are left as ``None``; call
        :func:`q_offset_to_pixel_delta` to fill them in.
    """
    chi = np.asarray(chi_deg, dtype=float)
    q = np.asarray(q_peak, dtype=float)
    if chi.size != q.size:
        raise ValueError("chi_deg and q_peak length mismatch")
    if chi.size < 4:
        raise ValueError(f"need ≥ 4 points for sinusoid fit, got {chi.size}")

    chi_rad = np.deg2rad(chi)
    A = np.column_stack([np.ones_like(chi_rad), np.sin(chi_rad), np.cos(chi_rad)])
    coefs, *_ = np.linalg.lstsq(A, q, rcond=None)
    q0, A_r, A_c = float(coefs[0]), float(coefs[1]), float(coefs[2])
    resid = q - A @ coefs
    rms = float(np.sqrt(np.mean(resid ** 2)))

    return BeamOffsetResult(
        q0=q0, A_r=A_r, A_c=A_c, rms=rms, n=int(chi.size),
        ring_q_expected=(
            float(ring_q_expected) if ring_q_expected is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# q-space  →  pixel / distance conversion (requires geometry)
# ---------------------------------------------------------------------------

def q_offset_to_pixel_delta(
    fit: BeamOffsetResult,
    *,
    wavelength_nm: float,
    distance_mm: float,
    pixel_mm: float = 0.172,
    chi_convention: str = "smi",
) -> BeamOffsetResult:
    """Convert a ``BeamOffsetResult`` from q-units to pixel + mm deltas.

    Uses the small-angle relation ``q ≈ (2π · r_px · px_mm) / (λ_nm · D_mm)``,
    so ``r_px = q · λ D / (2π px)`` and ``Δr_px = ΔA_q · λ D / (2π px)``.

    Chi convention
    --------------
    smi-tiled's SAXS and WAXS integrators *both* compute
    ``chi = atan2(x_q, y_q)`` — but the two detectors disagree on which
    physical direction is "+y_q":

    * SAXS (pyFAI-style, ``"smi_saxs"`` / default ``"smi"``)::

          x_q ∝  (col − bc_col)        →  +χ=90 toward +col
          y_q ∝ −(row − bc_row)        →  +χ=0  toward −row (top)

      so ``q_app(χ) = q₀ + (δcol/scale)·sin(χ) − (δrow/scale)·cos(χ)``
      and the fit-to-pixel mapping is ``δrow = −A_c·scale, δcol = A_r·scale``.

    * WAXS (``"smi_waxs"``) — the integrator applies
      ``q_horizontal_sign = q_vertical_sign = −1`` to the panel ``qx, qy``,
      which flips the y_q sense relative to SAXS::

          x_q ∝  (col − bc_col)        →  +χ=90 toward +col   (same as SAXS)
          y_q ∝ +(row − bc_row)        →  +χ=0  toward +row   (bottom)

      so ``q_app(χ) = q₀ + (δcol/scale)·sin(χ) + (δrow/scale)·cos(χ)``
      and the mapping is ``δrow = +A_c·scale, δcol = A_r·scale``.

    The returned ``drow_px / dcol_px`` are the corrections to **add** to
    the existing ``beam_delta_row_px / beam_delta_col_px`` widgets to
    move the assumed beam-centre toward the true centre.

    Pass ``chi_convention="atan2_y_x"`` to get the un-flipped mapping
    (``A_r → Δrow``, ``A_c → Δcol``) if a different integrator is used.

    Parameters
    ----------
    fit
        Output of :func:`fit_beam_offset_qspace`.
    wavelength_nm, distance_mm, pixel_mm
        Detector geometry (sample-to-detector distance, pixel pitch).
    chi_convention : {'smi', 'smi_saxs', 'smi_waxs', 'atan2_y_x'}
        How chi was defined in the integrator that produced the q-χ map.
        ``"smi"`` is an alias for ``"smi_saxs"`` (kept for backwards
        compatibility).

    Returns
    -------
    BeamOffsetResult
        A new instance with ``drow_px``, ``dcol_px``, ``ddist_mm``
        populated.
    """
    if wavelength_nm <= 0 or distance_mm <= 0 or pixel_mm <= 0:
        raise ValueError("wavelength, distance, and pixel size must be positive")

    scale_px = (wavelength_nm * distance_mm) / (2.0 * np.pi * pixel_mm)
    if chi_convention in ("smi", "smi_saxs"):
        # SAXS: y_q ∝ −row,  so δrow appears in the fit with a minus sign.
        drow_px = -fit.A_c * scale_px
        dcol_px = fit.A_r * scale_px
    elif chi_convention == "smi_waxs":
        # WAXS: q_vertical_sign = -1 flips the y_q sense vs SAXS.
        drow_px = fit.A_c * scale_px
        dcol_px = fit.A_r * scale_px
    elif chi_convention == "atan2_y_x":
        drow_px = fit.A_r * scale_px
        dcol_px = fit.A_c * scale_px
    else:
        raise ValueError(
            "chi_convention must be 'smi'/'smi_saxs'/'smi_waxs'/'atan2_y_x', "
            f"got {chi_convention!r}"
        )

    if fit.ring_q_expected and fit.q0 > 0:
        # D_correct = D_assumed * (q_observed / q_expected), so
        # ΔD = D_assumed * (q0 / q_expected - 1).
        # Positive ΔD when ring appears at too-high q (detector further
        # than assumed); negative when ring is at too-low q.
        ddist_mm = float(distance_mm * (fit.q0 / fit.ring_q_expected - 1.0))
    else:
        ddist_mm = None

    return BeamOffsetResult(
        q0=fit.q0,
        A_r=fit.A_r,
        A_c=fit.A_c,
        rms=fit.rms,
        n=fit.n,
        drow_px=float(drow_px),
        dcol_px=float(dcol_px),
        ddist_mm=ddist_mm,
        ring_q_expected=fit.ring_q_expected,
    )


# ---------------------------------------------------------------------------
# Multi-ring simultaneous fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiRingResult:
    """Joint beam-offset + distance from fitting multiple AgBh rings.

    The model fits ``q_n(chi) = q0_n + A_r * sin(chi) + A_c * cos(chi)``
    with a *shared* ``(A_r, A_c)`` across all rings.  The per-ring q0
    values give independent distance estimates via ``D_actual/D_assumed =
    q0_n / q_expected_n``, which are combined as a weighted mean.
    """
    orders: tuple[int, ...]
    q0_per_ring: dict[int, float]       # {order: fitted ring centre}
    A_r: float                          # shared sin(chi) amplitude (nm⁻¹)
    A_c: float                          # shared cos(chi) amplitude (nm⁻¹)
    rms: float                          # joint residual RMS (nm⁻¹)
    n_total: int                        # total accepted chi slices
    n_per_ring: dict[int, int]          # {order: n_accepted}
    peak_fits: dict[int, PeakFitResult] # per-ring PeakFitResult
    dist_ratio: float                   # weighted-mean q0/q_expected
    # Filled in by q_offset_to_pixel_delta_multi:
    drow_px: float | None = None
    dcol_px: float | None = None
    ddist_mm: float | None = None


def fit_multi_ring(
    qchi,
    *,
    orders: Sequence[int],
    chi_min: float = -180.0,
    chi_max: float = 180.0,
    q_half_width: float = 0.15,
    bg_order: int = 1,
    snr_threshold: float = 3.0,
    max_sigma_frac: float = 0.5,
    chi_stride: int = 1,
    min_rings: int = 2,
    min_chi_per_ring: int = 8,
) -> MultiRingResult:
    """Fit multiple AgBh rings simultaneously for joint beam-offset + distance.

    For each ring order in ``orders``, the corresponding q-window is
    ``[n*q1 - q_half_width, n*q1 + q_half_width]``.  Per-chi Gaussian
    peaks are fit independently per ring via :func:`fit_ring_peaks`, then
    a joint linear least-squares solves for the shared beam-offset
    ``(A_r, A_c)`` and per-ring intercepts ``q0_n``.

    Parameters
    ----------
    qchi
        Dataset/DataArray with ``intensity(q, chi)`` covering the full q
        range needed for all requested rings.
    orders
        Which AgBh ring orders to fit (e.g. ``[1, 2, 3, 4]`` for SAXS).
    chi_min, chi_max
        Azimuthal window (degrees).
    q_half_width
        Half-width of the q-window around each ring centre.
    bg_order, snr_threshold, max_sigma_frac, chi_stride
        Passed to :func:`fit_ring_peaks` for each ring.
    min_rings
        Minimum number of rings that must succeed for the joint fit.
    min_chi_per_ring
        Minimum accepted chi slices per ring.

    Returns
    -------
    MultiRingResult
        Contains per-ring results and the joint offset/distance solution.
        ``drow_px / dcol_px / ddist_mm`` are ``None`` until converted via
        :func:`q_offset_to_pixel_delta_multi`.
    """
    if len(orders) < 1:
        raise ValueError("Must specify at least one ring order")

    peak_fits: dict[int, PeakFitResult] = {}
    for n in orders:
        q_centre = agbh_q(n)
        q_min = q_centre - q_half_width
        q_max = q_centre + q_half_width
        try:
            pf = fit_ring_peaks(
                qchi,
                q_min=q_min,
                q_max=q_max,
                chi_min=chi_min,
                chi_max=chi_max,
                bg_order=bg_order,
                snr_threshold=snr_threshold,
                max_sigma_frac=max_sigma_frac,
                chi_stride=chi_stride,
            )
        except ValueError:
            continue
        if pf.n_accepted >= min_chi_per_ring:
            peak_fits[n] = pf

    if len(peak_fits) < min_rings:
        raise ValueError(
            f"Only {len(peak_fits)} ring(s) produced enough data "
            f"(need min_rings={min_rings}).  "
            f"Orders attempted: {list(orders)}, "
            f"succeeded: {list(peak_fits.keys())}"
        )

    # --- Joint linear least-squares ---
    # Model per ring n:  q_peak(chi) = q0_n + A_r*sin(chi) + A_c*cos(chi)
    # Parameter vector: [q0_1, q0_2, ..., q0_k, A_r, A_c]
    ring_order_list = sorted(peak_fits.keys())
    k = len(ring_order_list)

    # Build design matrix and target vector
    chi_all = []
    q_all = []
    ring_idx = []  # which ring each row belongs to
    for i, n in enumerate(ring_order_list):
        pf = peak_fits[n]
        chi_all.append(pf.chi_deg)
        q_all.append(pf.q_peak)
        ring_idx.append(np.full(pf.n_accepted, i, dtype=int))

    chi_cat = np.concatenate(chi_all)
    q_cat = np.concatenate(q_all)
    idx_cat = np.concatenate(ring_idx)
    n_total = len(chi_cat)

    # Design matrix: k columns for per-ring intercepts + 2 for sin/cos
    A = np.zeros((n_total, k + 2), dtype=float)
    chi_rad = np.deg2rad(chi_cat)
    for i in range(k):
        A[idx_cat == i, i] = 1.0
    A[:, k] = np.sin(chi_rad)
    A[:, k + 1] = np.cos(chi_rad)

    coefs, *_ = np.linalg.lstsq(A, q_cat, rcond=None)
    q0_vals = coefs[:k]
    A_r = float(coefs[k])
    A_c = float(coefs[k + 1])

    resid = q_cat - A @ coefs
    rms = float(np.sqrt(np.mean(resid ** 2)))

    q0_per_ring = {n: float(q0_vals[i]) for i, n in enumerate(ring_order_list)}
    n_per_ring = {n: peak_fits[n].n_accepted for n in ring_order_list}

    # Distance ratio: weighted mean of q0_n / q_expected_n
    # Weight by number of accepted chi slices per ring.
    ratios = np.array([q0_vals[i] / agbh_q(n) for i, n in enumerate(ring_order_list)])
    weights = np.array([peak_fits[n].n_accepted for n in ring_order_list], dtype=float)
    dist_ratio = float(np.average(ratios, weights=weights))

    return MultiRingResult(
        orders=tuple(ring_order_list),
        q0_per_ring=q0_per_ring,
        A_r=A_r,
        A_c=A_c,
        rms=rms,
        n_total=n_total,
        n_per_ring=n_per_ring,
        peak_fits=peak_fits,
        dist_ratio=dist_ratio,
    )


def q_offset_to_pixel_delta_multi(
    fit: MultiRingResult,
    *,
    wavelength_nm: float,
    distance_mm: float,
    pixel_mm: float = 0.172,
    chi_convention: str = "smi",
) -> MultiRingResult:
    """Convert a :class:`MultiRingResult` to pixel + mm deltas.

    Same geometry conversion as :func:`q_offset_to_pixel_delta`, but
    derives ``ddist_mm`` from the weighted ``dist_ratio`` across all
    fitted rings rather than a single ring.

    Returns a new ``MultiRingResult`` with ``drow_px``, ``dcol_px``,
    ``ddist_mm`` populated.
    """
    if wavelength_nm <= 0 or distance_mm <= 0 or pixel_mm <= 0:
        raise ValueError("wavelength, distance, and pixel size must be positive")

    scale_px = (wavelength_nm * distance_mm) / (2.0 * np.pi * pixel_mm)
    if chi_convention in ("smi", "smi_saxs"):
        drow_px = -fit.A_c * scale_px
        dcol_px = fit.A_r * scale_px
    elif chi_convention == "smi_waxs":
        drow_px = fit.A_c * scale_px
        dcol_px = fit.A_r * scale_px
    elif chi_convention == "atan2_y_x":
        drow_px = fit.A_r * scale_px
        dcol_px = fit.A_c * scale_px
    else:
        raise ValueError(
            "chi_convention must be 'smi'/'smi_saxs'/'smi_waxs'/'atan2_y_x', "
            f"got {chi_convention!r}"
        )

    # dist_ratio = q0_observed / q_expected (weighted mean across rings).
    # D_correct = D_assumed * dist_ratio  →  ΔD = D_assumed * (dist_ratio - 1)
    ddist_mm = float(distance_mm * (fit.dist_ratio - 1.0))

    return MultiRingResult(
        orders=fit.orders,
        q0_per_ring=fit.q0_per_ring,
        A_r=fit.A_r,
        A_c=fit.A_c,
        rms=fit.rms,
        n_total=fit.n_total,
        n_per_ring=fit.n_per_ring,
        peak_fits=fit.peak_fits,
        dist_ratio=fit.dist_ratio,
        drow_px=float(drow_px),
        dcol_px=float(dcol_px),
        ddist_mm=ddist_mm,
    )
