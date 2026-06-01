"""Tests for per-frame peak fitting (`smi_browser.models.peakfit`)."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from smi_browser.models.peakfit import PeakDef, fit_peak_across_frames

_GAUSS_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _gauss(q, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((q - mu) / sigma) ** 2)


@pytest.fixture
def synthetic():
    """A stack of Gaussian frames with varying amplitude and centre."""
    q = np.linspace(0.0, 10.0, 400)
    n = 30
    amps = np.linspace(1.0, 5.0, n)
    mus = np.linspace(4.0, 6.0, n)
    sigma = 0.3
    iq = np.stack([_gauss(q, a, m, sigma) for a, m in zip(amps, mus)])
    return q, iq, amps, mus, sigma


def test_recovers_gaussian_params(synthetic):
    q, iq, amps, mus, sigma = synthetic
    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian", baseline="none")
    res = fit_peak_across_frames(q, iq, peak)

    assert res["success"].all()
    np.testing.assert_allclose(res["amplitude"], amps, rtol=1e-2)
    np.testing.assert_allclose(res["center"], mus, atol=2e-2)
    np.testing.assert_allclose(res["fwhm"], sigma * _GAUSS_FWHM, rtol=2e-2)
    # area = amp * sigma * sqrt(2 pi)
    np.testing.assert_allclose(
        res["area"], amps * sigma * np.sqrt(2 * np.pi), rtol=2e-2,
    )


def test_linear_baseline_subtraction():
    q = np.linspace(0.0, 10.0, 400)
    amp, mu, sigma = 3.0, 5.0, 0.3
    slope, intercept = 0.5, 2.0
    iq = (_gauss(q, amp, mu, sigma) + slope * q + intercept)[None, :]

    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian", baseline="linear")
    res = fit_peak_across_frames(q, iq, peak)
    assert res["success"][0]
    assert res["amplitude"][0] == pytest.approx(amp, rel=2e-2)
    assert res["center"][0] == pytest.approx(mu, abs=2e-2)

    # Without baseline subtraction the sloped background corrupts the fit.
    peak_nb = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian", baseline="none")
    res_nb = fit_peak_across_frames(q, iq, peak_nb)
    assert abs(res_nb["amplitude"][0] - amp) > abs(res["amplitude"][0] - amp)


def test_lorentzian_model():
    q = np.linspace(0.0, 10.0, 600)
    amp, mu, gamma = 4.0, 5.0, 0.4
    iq = (amp * gamma**2 / ((q - mu) ** 2 + gamma**2))[None, :]
    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="lorentzian", baseline="none")
    res = fit_peak_across_frames(q, iq, peak)
    assert res["success"][0]
    assert res["amplitude"][0] == pytest.approx(amp, rel=2e-2)
    assert res["center"][0] == pytest.approx(mu, abs=2e-2)
    assert res["fwhm"][0] == pytest.approx(2 * gamma, rel=3e-2)


def test_nan_on_unfittable(synthetic):
    q, iq, *_ = synthetic
    iq = iq.copy()
    iq[5] = np.nan  # one all-NaN frame
    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian", baseline="none")
    res = fit_peak_across_frames(q, iq, peak)
    assert not res["success"][5]
    assert np.isnan(res["amplitude"][5])
    assert res["success"][0]  # other frames still fit


def test_range_too_narrow_returns_all_nan(synthetic):
    q, iq, *_ = synthetic
    peak = PeakDef("p", q_min=4.99, q_max=5.0, model="gaussian", baseline="none")
    res = fit_peak_across_frames(q, iq, peak)
    assert not res["success"].any()
    assert np.isnan(res["amplitude"]).all()


def test_cancellation_returns_early(synthetic):
    q, iq, *_ = synthetic
    cancel = threading.Event()
    cancel.set()
    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian", baseline="none")
    res = fit_peak_across_frames(q, iq, peak, cancel=cancel, cancel_check_every=1)
    # First frame check happens at i=0 with the flag already set → nothing fit.
    assert not res["success"].any()


def test_progress_reports_completion(synthetic):
    q, iq, *_ = synthetic
    seen = []
    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian", baseline="none")
    fit_peak_across_frames(q, iq, peak, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (iq.shape[0], iq.shape[0])


# --- robustness: no-peak gating & width bounds -----------------------------

def test_no_peak_frame_reports_zero():
    """A flat/noisy frame with no peak gates to amplitude/area 0, centre NaN."""
    rng = np.random.default_rng(0)
    q = np.linspace(0.0, 10.0, 400)
    peak_frame = _gauss(q, 5.0, 5.0, 0.3) + 0.01 * rng.standard_normal(q.size)
    flat_frame = 1.0 + 0.01 * rng.standard_normal(q.size)  # no peak
    iq = np.stack([peak_frame, flat_frame])

    peak = PeakDef("p", q_min=4.0, q_max=6.0, model="gaussian",
                   baseline="linear", link="independent")
    res = fit_peak_across_frames(q, iq, peak)

    assert res["success"][0]                      # real peak fits
    assert not res["success"][1]                  # no peak → gated out
    assert res["amplitude"][1] == 0.0
    assert res["area"][1] == 0.0
    assert np.isnan(res["center"][1])
    assert np.isnan(res["fwhm"][1])


def test_width_cannot_exceed_drawn_range():
    """FWHM is bounded by the drawn q-range even when the data is very broad."""
    q = np.linspace(0.0, 10.0, 400)
    # A broad gaussian (sigma 2.0) viewed through a narrow [4.5, 5.5] window.
    iq = _gauss(q, 5.0, 5.0, 2.0)[None, :]
    peak = PeakDef("p", q_min=4.5, q_max=5.5, model="gaussian",
                   baseline="none", link="independent")
    res = fit_peak_across_frames(q, iq, peak)
    # Whether or not it is accepted, the fitted FWHM may never exceed the range.
    if res["success"][0]:
        assert res["fwhm"][0] <= (5.5 - 4.5) + 1e-9


def test_linked_shares_center_and_width():
    """Linked mode recovers per-frame amplitudes with a single shared centre."""
    q = np.linspace(0.0, 10.0, 400)
    amps = np.array([1.0, 2.0, 3.0, 4.0])
    mu, sigma = 5.0, 0.3
    iq = np.stack([_gauss(q, a, mu, sigma) for a in amps])
    peak = PeakDef("p", q_min=3.0, q_max=7.0, model="gaussian",
                   baseline="none", link="linked")
    res = fit_peak_across_frames(q, iq, peak)
    assert res["success"].all()
    # All frames share one centre/width.
    centers = res["center"][res["success"]]
    assert np.allclose(centers, centers[0])
    assert res["center"][0] == pytest.approx(mu, abs=2e-2)
    # Amplitudes still recovered per frame.
    np.testing.assert_allclose(res["amplitude"], amps, rtol=5e-2)


def test_linked_falls_back_when_no_aggregate_peak():
    """Linked mode with no peak anywhere notes a fallback and gates frames."""
    rng = np.random.default_rng(1)
    q = np.linspace(0.0, 10.0, 200)
    iq = 1.0 + 0.01 * rng.standard_normal((5, q.size))
    peak = PeakDef("p", q_min=4.0, q_max=6.0, model="gaussian",
                   baseline="linear", link="linked")
    res = fit_peak_across_frames(q, iq, peak)
    assert "note" in res
    assert not res["success"].any()


def test_bg_factor_widens_baseline_window():
    """A linear baseline is recovered well when the flank window is widened."""
    q = np.linspace(0.0, 10.0, 600)
    amp, mu, sigma = 3.0, 5.0, 0.3
    slope, intercept = 0.5, 2.0
    iq = (_gauss(q, amp, mu, sigma) + slope * q + intercept)[None, :]
    peak = PeakDef("p", q_min=4.5, q_max=5.5, model="gaussian",
                   baseline="linear", link="independent", bg_factor=3.0)
    res = fit_peak_across_frames(q, iq, peak)
    assert res["success"][0]
    assert res["amplitude"][0] == pytest.approx(amp, rel=5e-2)
    assert res["center"][0] == pytest.approx(mu, abs=2e-2)


def test_peakdef_key_includes_link_and_bg():
    a = PeakDef("p", 3.0, 7.0, link="linked", bg_factor=2.0)
    b = PeakDef("p", 3.0, 7.0, link="independent", bg_factor=2.0)
    c = PeakDef("p", 3.0, 7.0, link="linked", bg_factor=3.0)
    assert a.key() != b.key()
    assert a.key() != c.key()
