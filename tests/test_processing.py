"""Tests for smi_browser.processing — pure param-builder logic."""

import pytest
from smi_browser.processing import build_proc_params


class TestBuildProcParams:
    """Verify kwarg construction for transmission and GI geometry."""

    UID = "abc12345-dead-beef-cafe-000000000001"

    def _defaults(self, **overrides):
        kw = dict(
            tiled_uri="http://tiled:8000",
            catalog="rsoxs",
            saxs_mask_path="", waxs_mask_path="",
            n_q=2000, n_chi=360,
            n_qxy=500, n_qz=500,
            theta_offset=-0.5,
            dezinger=3000.0,
            incident_angle_auto=True,
            incident_angle=0.0,
            saxs_row_delta=0.0, saxs_col_delta=0.0,
            waxs_row_delta=0.0, waxs_col_delta=0.0,
            saxs_dist_delta=0.0,
            cache_geometry=True,
            default_n_q=2000, default_n_chi=360,
            default_n_qxy=500, default_n_qz=500,
            default_theta_offset=-0.5, default_dezinger=3000.0,
            default_saxs_row_delta=0.0, default_saxs_col_delta=0.0,
            default_waxs_row_delta=0.0, default_waxs_col_delta=0.0,
            default_saxs_dist_delta=0.0,
        )
        kw.update(overrides)
        return kw

    # -- Transmission --

    def test_transmission_defaults(self):
        fn, params = build_proc_params(self.UID, "transmission", **self._defaults())
        assert fn == "reduce_smi_combined"
        assert params["uid"] == self.UID
        assert params["geometry"] == "transmission"
        assert params["n_q"] == 2000
        assert "n_chi" not in params  # matches default → omitted
        assert "saxs_beam_delta_px" not in params
        assert "dezinger_threshold" not in params

    def test_transmission_non_default_n_chi(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(n_chi=720))
        assert params["n_chi"] == 720

    def test_transmission_beam_delta_one_axis(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(saxs_row_delta=1.5))
        assert params["saxs_beam_delta_px"] == (1.5, 0.0)

    def test_transmission_dezinger_changed(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(dezinger=500.0))
        assert params["dezinger_threshold"] == 500.0

    def test_transmission_dezinger_zero_becomes_none(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(dezinger=0.0))
        assert params["dezinger_threshold"] is None

    def test_transmission_dist_delta(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(saxs_dist_delta=5.0))
        assert params["saxs_distance_delta_mm"] == 5.0

    def test_transmission_masks(self):
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(saxs_mask_path="/m/s.npy", waxs_mask_path="/m/w.npy"))
        assert params["saxs_mask_path"] == "/m/s.npy"
        assert params["waxs_mask_path"] == "/m/w.npy"

    def test_transmission_empty_mask_becomes_none(self):
        fn, params = build_proc_params(self.UID, "transmission", **self._defaults())
        assert params["saxs_mask_path"] is None
        assert params["waxs_mask_path"] is None

    # -- Grazing incidence --

    def test_gi_defaults(self):
        fn, params = build_proc_params(self.UID, "grazing", **self._defaults())
        assert fn == "reduce_smi_gi"
        assert params["uid"] == self.UID
        assert "n_qxy" not in params
        assert "n_qz" not in params
        assert "theta_offset" not in params
        assert "incident_angle_deg" not in params

    def test_gi_custom_grid(self):
        fn, params = build_proc_params(self.UID, "grazing",
                                       **self._defaults(n_qxy=800, n_qz=600))
        assert params["n_qxy"] == 800
        assert params["n_qz"] == 600

    def test_gi_manual_incident_angle(self):
        fn, params = build_proc_params(
            self.UID, "grazing",
            **self._defaults(incident_angle_auto=False, incident_angle=0.12))
        assert params["incident_angle_deg"] == 0.12

    def test_gi_theta_offset_changed(self):
        fn, params = build_proc_params(self.UID, "grazing",
                                       **self._defaults(theta_offset=0.0))
        assert params["theta_offset"] == 0.0
