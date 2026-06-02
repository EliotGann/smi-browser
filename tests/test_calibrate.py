"""Tests for smi_browser.calibrate — AgBh ring fit + beam-offset solve."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from smi_browser.calibrate import (
    AGBH_Q1_NM,
    BeamOffsetResult,
    MultiRingResult,
    PeakFitResult,
    agbh_q,
    fit_beam_offset_qspace,
    fit_multi_ring,
    fit_ring_peaks,
    nearest_agbh_order,
    q_offset_to_pixel_delta,
    q_offset_to_pixel_delta_multi,
)


# ---------------------------------------------------------------------------
# AgBh helpers
# ---------------------------------------------------------------------------

def test_agbh_q1_matches_definition():
    assert AGBH_Q1_NM == pytest.approx(2 * np.pi / 5.8380, rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_agbh_q_is_linear_in_order(n):
    assert agbh_q(n) == pytest.approx(n * AGBH_Q1_NM, rel=1e-12)


def test_agbh_q_rejects_zero_order():
    with pytest.raises(ValueError):
        agbh_q(0)


def test_nearest_agbh_order_snaps_correctly():
    # ring 5 q ≈ 5.38 nm⁻¹; nudged values still resolve to 5
    assert nearest_agbh_order(5 * AGBH_Q1_NM - 0.05) == 5
    assert nearest_agbh_order(5 * AGBH_Q1_NM + 0.05) == 5
    # mid-way between ring 3 and 4 should round-to-nearest
    mid = 3.5 * AGBH_Q1_NM
    assert nearest_agbh_order(mid) in (3, 4)


def test_nearest_agbh_order_clamps_to_max():
    assert nearest_agbh_order(50 * AGBH_Q1_NM, max_order=9) == 9
    assert nearest_agbh_order(0.0, max_order=9) == 1


# ---------------------------------------------------------------------------
# Synthetic q-chi map utilities
# ---------------------------------------------------------------------------

def _synth_ring_qchi(
    *,
    q0_true: float,
    A_r: float,
    A_c: float,
    sigma: float = 0.05,
    amp: float = 100.0,
    baseline: float = 5.0,
    noise: float = 0.5,
    n_q: int = 500,
    n_chi: int = 360,
    q_lo: float | None = None,
    q_hi: float | None = None,
    seed: int = 42,
) -> xr.Dataset:
    """Build a Dataset whose ring traces ``q0 + A_r·sin(chi) + A_c·cos(chi)``."""
    rng = np.random.default_rng(seed)
    if q_lo is None:
        q_lo = q0_true - 0.4
    if q_hi is None:
        q_hi = q0_true + 0.4
    q = np.linspace(q_lo, q_hi, n_q)
    chi = np.linspace(-180.0, 180.0, n_chi)
    chi_rad = np.deg2rad(chi)
    mu = q0_true + A_r * np.sin(chi_rad) + A_c * np.cos(chi_rad)
    qq, mm = np.meshgrid(q, mu, indexing="ij")
    intensity = (
        baseline
        + amp * np.exp(-0.5 * ((qq - mm) / sigma) ** 2)
        + rng.normal(scale=noise, size=qq.shape)
    )
    return xr.Dataset(
        {"intensity": (("q", "chi"), intensity)},
        coords={"q": q, "chi": chi},
    )


# ---------------------------------------------------------------------------
# fit_ring_peaks
# ---------------------------------------------------------------------------

def test_fit_ring_peaks_recovers_centred_ring():
    q0 = 5.0
    qchi = _synth_ring_qchi(q0_true=q0, A_r=0.0, A_c=0.0, sigma=0.04)
    pf = fit_ring_peaks(qchi, q_min=q0 - 0.3, q_max=q0 + 0.3)
    assert isinstance(pf, PeakFitResult)
    assert pf.n_accepted > 300, "most chi slices should converge"
    assert pf.q_peak.mean() == pytest.approx(q0, abs=2e-3)


def test_fit_ring_peaks_recovers_off_centre_ring():
    q0, A_r, A_c = 5.0, 0.06, -0.04
    qchi = _synth_ring_qchi(q0_true=q0, A_r=A_r, A_c=A_c, sigma=0.04)
    pf = fit_ring_peaks(qchi, q_min=q0 - 0.3, q_max=q0 + 0.3)
    # Sanity: peak position should track the truth curve to within fit noise
    chi_rad = np.deg2rad(pf.chi_deg)
    true_curve = q0 + A_r * np.sin(chi_rad) + A_c * np.cos(chi_rad)
    assert np.max(np.abs(pf.q_peak - true_curve)) < 0.005


def test_fit_ring_peaks_rejects_invalid_window():
    qchi = _synth_ring_qchi(q0_true=5.0, A_r=0.0, A_c=0.0)
    with pytest.raises(ValueError):
        fit_ring_peaks(qchi, q_min=5.5, q_max=5.0)


def test_fit_ring_peaks_handles_dataset_with_frame_dim():
    qchi = _synth_ring_qchi(q0_true=5.0, A_r=0.0, A_c=0.0)
    # Promote to (frame, q, chi) — fit should reduce across the frame dim
    da = qchi["intensity"].expand_dims(frame=[0, 1])
    ds_with_frame = xr.Dataset({"intensity": da})
    pf = fit_ring_peaks(ds_with_frame, q_min=4.7, q_max=5.3)
    assert pf.n_accepted > 0


def test_fit_ring_peaks_handles_partial_chi_coverage():
    """Slices outside chi_min/chi_max are dropped, sinusoid fit still works."""
    q0, A_r, A_c = 5.0, 0.05, 0.0
    qchi = _synth_ring_qchi(q0_true=q0, A_r=A_r, A_c=A_c, sigma=0.04)
    pf = fit_ring_peaks(
        qchi, q_min=q0 - 0.3, q_max=q0 + 0.3,
        chi_min=-90.0, chi_max=90.0,
    )
    assert pf.n_accepted > 80
    assert (pf.chi_deg.min() >= -90.0) and (pf.chi_deg.max() <= 90.0)


# ---------------------------------------------------------------------------
# fit_beam_offset_qspace
# ---------------------------------------------------------------------------

def test_fit_beam_offset_recovers_amplitudes_with_clean_data():
    q0, A_r, A_c = 5.0, 0.07, -0.03
    chi = np.linspace(-180, 180, 360)
    chi_rad = np.deg2rad(chi)
    q = q0 + A_r * np.sin(chi_rad) + A_c * np.cos(chi_rad)
    fit = fit_beam_offset_qspace(chi, q, ring_q_expected=5.0)
    assert fit.q0 == pytest.approx(q0, abs=1e-10)
    assert fit.A_r == pytest.approx(A_r, abs=1e-10)
    assert fit.A_c == pytest.approx(A_c, abs=1e-10)
    assert fit.rms < 1e-10


def test_fit_beam_offset_end_to_end_through_fit_ring_peaks():
    q0, A_r, A_c = 5.0, 0.05, 0.02
    qchi = _synth_ring_qchi(q0_true=q0, A_r=A_r, A_c=A_c, sigma=0.04, noise=0.3)
    pf = fit_ring_peaks(qchi, q_min=q0 - 0.3, q_max=q0 + 0.3)
    fit = fit_beam_offset_qspace(
        pf.chi_deg, pf.q_peak, ring_q_expected=q0,
    )
    assert fit.q0 == pytest.approx(q0, abs=2e-3)
    assert fit.A_r == pytest.approx(A_r, abs=2e-3)
    assert fit.A_c == pytest.approx(A_c, abs=2e-3)
    assert fit.rms < 0.005


def test_fit_beam_offset_rejects_short_input():
    with pytest.raises(ValueError):
        fit_beam_offset_qspace([0.0, 90.0, 180.0], [5.0, 5.0, 5.0])


# ---------------------------------------------------------------------------
# q_offset_to_pixel_delta
# ---------------------------------------------------------------------------

def test_pixel_delta_round_trip_through_known_geometry():
    """Inject a known pixel offset, propagate to q-space, recover via fit.

    SMI's chi = atan2(x, y) with y = -row gives
        q_app(χ) = q0 + (δcol/scale_px)·sin(χ) − (δrow/scale_px)·cos(χ)
    so A_r encodes δcol and A_c encodes −δrow.
    """
    # SMI-typical geometry
    wavelength_nm = 1.23984198 / 16.1   # ≈ 0.077 nm
    distance_mm = 8500.0
    pixel_mm = 0.172
    scale_px = (wavelength_nm * distance_mm) / (2 * np.pi * pixel_mm)

    drow_true, dcol_true = 1.5, -2.3    # pixels
    A_r_true = dcol_true / scale_px      # sin(χ) coefficient → δcol
    A_c_true = -drow_true / scale_px     # cos(χ) coefficient → −δrow
    q0_true = 5 * AGBH_Q1_NM

    qchi = _synth_ring_qchi(
        q0_true=q0_true, A_r=A_r_true, A_c=A_c_true,
        sigma=0.04, q_lo=q0_true - 0.4, q_hi=q0_true + 0.4,
    )
    pf = fit_ring_peaks(qchi, q_min=q0_true - 0.3, q_max=q0_true + 0.3)
    fit_q = fit_beam_offset_qspace(
        pf.chi_deg, pf.q_peak, ring_q_expected=q0_true,
    )
    fit_px = q_offset_to_pixel_delta(
        fit_q,
        wavelength_nm=wavelength_nm,
        distance_mm=distance_mm,
        pixel_mm=pixel_mm,
    )
    assert fit_px.drow_px == pytest.approx(drow_true, abs=0.15)
    assert fit_px.dcol_px == pytest.approx(dcol_true, abs=0.15)
    # No injected distance error — ddist should be near zero.
    assert abs(fit_px.ddist_mm) < 1.0


def test_pixel_delta_round_trip_waxs_convention():
    """WAXS chi convention flips the row sense.

    For WAXS, q_app(χ) = q0 + (δcol/scale)·sin(χ) + (δrow/scale)·cos(χ),
    so A_c encodes +δrow (no sign flip) — opposite of SAXS.
    """
    wavelength_nm = 1.23984198 / 16.1
    distance_mm = 273.0       # WAXS-typical
    pixel_mm = 0.172
    scale_px = (wavelength_nm * distance_mm) / (2 * np.pi * pixel_mm)

    drow_true, dcol_true = 2.0, -1.8
    A_r_true = dcol_true / scale_px
    A_c_true = drow_true / scale_px       # +δrow (vs −δrow for SAXS)
    q0_true = 4 * AGBH_Q1_NM

    qchi = _synth_ring_qchi(
        q0_true=q0_true, A_r=A_r_true, A_c=A_c_true,
        sigma=0.04, q_lo=q0_true - 0.4, q_hi=q0_true + 0.4,
    )
    pf = fit_ring_peaks(qchi, q_min=q0_true - 0.3, q_max=q0_true + 0.3)
    fit_q = fit_beam_offset_qspace(pf.chi_deg, pf.q_peak, ring_q_expected=q0_true)
    fit_px = q_offset_to_pixel_delta(
        fit_q,
        wavelength_nm=wavelength_nm,
        distance_mm=distance_mm,
        pixel_mm=pixel_mm,
        chi_convention="smi_waxs",
    )
    assert fit_px.drow_px == pytest.approx(drow_true, abs=0.15)
    assert fit_px.dcol_px == pytest.approx(dcol_true, abs=0.15)


def test_pixel_delta_smi_is_alias_for_smi_saxs():
    fit = BeamOffsetResult(q0=5.0, A_r=0.01, A_c=0.02, rms=0.0, n=20)
    a = q_offset_to_pixel_delta(
        fit, wavelength_nm=0.077, distance_mm=8500.0, chi_convention="smi",
    )
    b = q_offset_to_pixel_delta(
        fit, wavelength_nm=0.077, distance_mm=8500.0, chi_convention="smi_saxs",
    )
    assert a.drow_px == b.drow_px
    assert a.dcol_px == b.dcol_px


def test_pixel_delta_saxs_and_waxs_differ_only_in_drow_sign():
    fit = BeamOffsetResult(q0=5.0, A_r=0.01, A_c=0.02, rms=0.0, n=20)
    s = q_offset_to_pixel_delta(
        fit, wavelength_nm=0.077, distance_mm=8500.0, chi_convention="smi_saxs",
    )
    w = q_offset_to_pixel_delta(
        fit, wavelength_nm=0.077, distance_mm=8500.0, chi_convention="smi_waxs",
    )
    assert s.dcol_px == pytest.approx(w.dcol_px)
    assert s.drow_px == pytest.approx(-w.drow_px)


def test_pixel_delta_distance_correction_recovers_known_offset():
    """If the apparent ring is at the wrong q, ddist should compensate."""
    wavelength_nm = 1.23984198 / 16.1
    pixel_mm = 0.172
    distance_true_mm = 8500.0
    distance_assumed_mm = 8700.0  # 200 mm too far
    # Apparent q = q_true * D_actual / D_assumed (ring appears at lower q
    # when assumed distance exceeds actual distance).
    q0_true = 5 * AGBH_Q1_NM
    q0_apparent = q0_true * (distance_true_mm / distance_assumed_mm)

    qchi = _synth_ring_qchi(
        q0_true=q0_apparent, A_r=0.0, A_c=0.0, sigma=0.04,
        q_lo=q0_apparent - 0.4, q_hi=q0_apparent + 0.4,
    )
    pf = fit_ring_peaks(qchi, q_min=q0_apparent - 0.3, q_max=q0_apparent + 0.3)
    fit_q = fit_beam_offset_qspace(
        pf.chi_deg, pf.q_peak, ring_q_expected=q0_true,
    )
    fit_px = q_offset_to_pixel_delta(
        fit_q,
        wavelength_nm=wavelength_nm,
        distance_mm=distance_assumed_mm,
        pixel_mm=pixel_mm,
    )
    # ddist_mm = D_assumed * (q0/q_expected - 1) ≈ -200 mm
    assert fit_px.ddist_mm == pytest.approx(
        distance_true_mm - distance_assumed_mm, abs=1.0
    )


def test_pixel_delta_rejects_bad_geometry():
    fit = BeamOffsetResult(q0=5.0, A_r=0.0, A_c=0.0, rms=0.0, n=10)
    with pytest.raises(ValueError):
        q_offset_to_pixel_delta(fit, wavelength_nm=0.0, distance_mm=8500.0)
    with pytest.raises(ValueError):
        q_offset_to_pixel_delta(fit, wavelength_nm=0.077, distance_mm=-1.0)


# ---------------------------------------------------------------------------
# fit_multi_ring — simultaneous multi-peak calibration
# ---------------------------------------------------------------------------

def _synth_multi_ring_qchi(
    *,
    orders: list[int],
    A_r: float = 0.0,
    A_c: float = 0.0,
    sigma: float = 0.04,
    amp: float = 100.0,
    baseline: float = 5.0,
    noise: float = 0.5,
    n_chi: int = 360,
    q_per_nm: int = 1000,
    seed: int = 42,
    dist_ratio: float = 1.0,
) -> xr.Dataset:
    """Build a wide q-chi Dataset containing multiple AgBh rings.

    ``dist_ratio`` shifts all ring centres by a constant factor (simulating
    a distance error: q_apparent = q_true * dist_ratio).
    """
    rng = np.random.default_rng(seed)
    q_lo = agbh_q(min(orders)) * dist_ratio - 0.5
    q_hi = agbh_q(max(orders)) * dist_ratio + 0.5
    n_q = int((q_hi - q_lo) * q_per_nm)
    q = np.linspace(q_lo, q_hi, n_q)
    chi = np.linspace(-180.0, 180.0, n_chi)
    chi_rad = np.deg2rad(chi)

    # Build intensity as sum of Gaussians for each ring
    intensity = np.full((n_q, n_chi), baseline, dtype=float)
    for n in orders:
        mu_base = agbh_q(n) * dist_ratio
        mu_chi = mu_base + A_r * np.sin(chi_rad) + A_c * np.cos(chi_rad)
        qq, mm = np.meshgrid(q, mu_chi, indexing="ij")
        intensity += amp * np.exp(-0.5 * ((qq - mm) / sigma) ** 2)

    intensity += rng.normal(scale=noise, size=intensity.shape)
    return xr.Dataset(
        {"intensity": (("q", "chi"), intensity)},
        coords={"q": q, "chi": chi},
    )


def test_fit_multi_ring_recovers_centred_rings():
    """Multi-ring fit with no beam offset → A_r ≈ A_c ≈ 0."""
    orders = [3, 4, 5, 6]
    qchi = _synth_multi_ring_qchi(orders=orders)
    result = fit_multi_ring(qchi, orders=orders)
    assert isinstance(result, MultiRingResult)
    assert set(result.orders) == set(orders)
    assert abs(result.A_r) < 0.002
    assert abs(result.A_c) < 0.002
    for n in orders:
        assert result.q0_per_ring[n] == pytest.approx(agbh_q(n), abs=0.005)
    assert result.dist_ratio == pytest.approx(1.0, abs=0.001)


def test_fit_multi_ring_recovers_beam_offset():
    """Shared A_r, A_c recovered across multiple rings."""
    A_r, A_c = 0.06, -0.03
    orders = [2, 3, 4, 5]
    qchi = _synth_multi_ring_qchi(orders=orders, A_r=A_r, A_c=A_c)
    result = fit_multi_ring(qchi, orders=orders)
    assert result.A_r == pytest.approx(A_r, abs=0.003)
    assert result.A_c == pytest.approx(A_c, abs=0.003)
    assert result.rms < 0.01


def test_fit_multi_ring_recovers_distance_error():
    """dist_ratio ≠ 1 when rings appear shifted by a distance error."""
    # Simulate detector 3% closer than assumed (rings at higher q)
    dist_ratio_true = 8500.0 / 8700.0  # ≈ 0.977
    orders = [3, 4, 5, 6]
    qchi = _synth_multi_ring_qchi(
        orders=orders, dist_ratio=dist_ratio_true, noise=0.3,
    )
    result = fit_multi_ring(qchi, orders=orders)
    assert result.dist_ratio == pytest.approx(dist_ratio_true, abs=0.002)


def test_fit_multi_ring_pixel_delta_distance():
    """End-to-end: multi-ring → pixel delta → distance correction."""
    wavelength_nm = 1.23984198 / 16.1
    distance_assumed_mm = 8700.0
    distance_true_mm = 8500.0
    dist_ratio = distance_true_mm / distance_assumed_mm
    orders = [3, 4, 5, 6]
    qchi = _synth_multi_ring_qchi(
        orders=orders, dist_ratio=dist_ratio, noise=0.3,
    )
    result = fit_multi_ring(qchi, orders=orders)
    result_px = q_offset_to_pixel_delta_multi(
        result,
        wavelength_nm=wavelength_nm,
        distance_mm=distance_assumed_mm,
    )
    # Should recover ΔD ≈ -200 mm
    assert result_px.ddist_mm == pytest.approx(
        distance_true_mm - distance_assumed_mm, abs=5.0,
    )
    # No beam offset injected
    assert abs(result_px.drow_px) < 0.5
    assert abs(result_px.dcol_px) < 0.5


def test_fit_multi_ring_with_beam_offset_and_distance():
    """Joint recovery of beam offset + distance from multiple rings."""
    wavelength_nm = 1.23984198 / 16.1
    distance_assumed_mm = 8700.0
    pixel_mm = 0.172
    scale_px = (wavelength_nm * distance_assumed_mm) / (2 * np.pi * pixel_mm)
    drow_true, dcol_true = 1.2, -1.8
    A_r_true = dcol_true / scale_px
    A_c_true = -drow_true / scale_px

    dist_ratio = 8500.0 / 8700.0
    orders = [3, 4, 5, 6]
    qchi = _synth_multi_ring_qchi(
        orders=orders, A_r=A_r_true, A_c=A_c_true,
        dist_ratio=dist_ratio, noise=0.3,
    )
    result = fit_multi_ring(qchi, orders=orders)
    result_px = q_offset_to_pixel_delta_multi(
        result,
        wavelength_nm=wavelength_nm,
        distance_mm=distance_assumed_mm,
        pixel_mm=pixel_mm,
        chi_convention="smi_saxs",
    )
    assert result_px.drow_px == pytest.approx(drow_true, abs=0.3)
    assert result_px.dcol_px == pytest.approx(dcol_true, abs=0.3)
    assert result_px.ddist_mm == pytest.approx(-200.0, abs=5.0)


def test_fit_multi_ring_skips_missing_rings():
    """Rings outside q range are skipped; fit proceeds with available ones."""
    orders_in_data = [3, 4, 5]
    qchi = _synth_multi_ring_qchi(orders=orders_in_data)
    # Request rings 3-7, but 6 and 7 are outside the data
    result = fit_multi_ring(qchi, orders=[3, 4, 5, 6, 7], min_rings=2)
    assert set(result.orders) == set(orders_in_data)


def test_fit_multi_ring_raises_if_too_few_rings():
    """Error if fewer than min_rings succeed."""
    orders = [3, 4, 5]
    qchi = _synth_multi_ring_qchi(orders=[3])  # only ring 3 in data
    with pytest.raises(ValueError, match="Only 1 ring"):
        fit_multi_ring(qchi, orders=orders, min_rings=2)


def test_fit_multi_ring_waxs_convention():
    """WAXS chi convention works for multi-ring."""
    wavelength_nm = 1.23984198 / 16.1
    distance_mm = 273.0
    pixel_mm = 0.172
    scale_px = (wavelength_nm * distance_mm) / (2 * np.pi * pixel_mm)
    drow_true, dcol_true = 2.0, -1.5
    A_r_true = dcol_true / scale_px
    A_c_true = drow_true / scale_px  # +δrow for WAXS

    orders = [5, 6, 7, 8]
    qchi = _synth_multi_ring_qchi(orders=orders, A_r=A_r_true, A_c=A_c_true)
    result = fit_multi_ring(qchi, orders=orders)
    result_px = q_offset_to_pixel_delta_multi(
        result,
        wavelength_nm=wavelength_nm,
        distance_mm=distance_mm,
        pixel_mm=pixel_mm,
        chi_convention="smi_waxs",
    )
    assert result_px.drow_px == pytest.approx(drow_true, abs=0.3)
    assert result_px.dcol_px == pytest.approx(dcol_true, abs=0.3)
