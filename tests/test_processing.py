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
            dezinger_kernel=5,
            incident_angle_auto=True,
            incident_angle=0.0,
            saxs_row_delta=0.0, saxs_col_delta=0.0,
            waxs_row_delta=0.0, waxs_col_delta=-4.5,
            saxs_dist_delta=0.0,
            waxs_beam_col_per_arc_deg=0.08,
            cache_geometry=True,
            solid_angle_correction=True,
            saxs_q_cutoff=0.0,
            saxs_agbh_ring_order=5,
            saxs_q_margin_fraction=0.01,
            beamstop_max_abs_arc_deg=15.0,
            saxs_rotate_cw_90=False,
            waxs_flip_horizontal=False,
            waxs_qx_shift_nm=0.0,
            waxs_qy_shift_nm=0.0,
            dynamic_saxs_mask=False,
            dyn_shadow_enabled=True,
            dyn_shadow_beam_visible_deg=14.5,
            dyn_shadow_clear_edge_deg=18.0,
            dyn_aper_enabled=True,
            dyn_aper_agbh_ring_order=5,
            dyn_aper_q_margin_fraction=0.01,
            dyn_aper_q_cutoff=0.0,
            waxs_kwargs=None,
            default_n_q=2000, default_n_chi=360,
            default_n_qxy=500, default_n_qz=500,
            default_theta_offset=-0.5, default_dezinger=3000.0,
            default_dezinger_kernel=5,
            default_saxs_row_delta=0.0, default_saxs_col_delta=0.0,
            default_waxs_row_delta=0.0, default_waxs_col_delta=-4.5,
            default_saxs_dist_delta=0.0,
            default_waxs_beam_col_per_arc_deg=0.08,
            default_beamstop_max_abs_arc_deg=15.0,
            default_saxs_agbh_ring_order=5,
            default_saxs_q_margin_fraction=0.01,
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

    # -- New parameters --

    def test_transmission_dezinger_kernel(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(dezinger_kernel=7))
        assert params["dezinger_kernel"] == 7

    def test_transmission_saxs_q_cutoff(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults(saxs_q_cutoff=2.5))
        assert params["saxs_q_cutoff"] == 2.5

    def test_transmission_saxs_q_cutoff_zero_omitted(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults())
        assert "saxs_q_cutoff" not in params

    def test_transmission_backend_options(self):
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(saxs_rotate_cw_90=True, waxs_qx_shift_nm=1.5))
        assert params["backend_options"]["saxs_rotate_cw_90"] is True
        assert params["backend_options"]["waxs_qx_shift_nm"] == 1.5

    def test_transmission_no_backend_when_defaults(self):
        fn, params = build_proc_params(self.UID, "transmission",
                                       **self._defaults())
        assert "backend_options" not in params

    def test_transmission_dynamic_mask(self):
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(dynamic_saxs_mask=True))
        assert params["saxs_kwargs"]["dynamic_saxs_mask"] is True

    def test_transmission_solid_angle(self):
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(solid_angle_correction=False))
        assert params["solid_angle_correction"] is False

    def test_transmission_waxs_col_per_arc(self):
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(waxs_beam_col_per_arc_deg=0.05))
        assert params["waxs_beam_col_per_arc_deg"] == 0.05

    def test_gi_beamstop_max_arc(self):
        fn, params = build_proc_params(
            self.UID, "grazing",
            **self._defaults(beamstop_max_abs_arc_deg=20.0))
        assert params["beamstop_max_abs_arc_deg"] == 20.0

    def test_gi_waxs_kwargs(self):
        fn, params = build_proc_params(
            self.UID, "grazing",
            **self._defaults(waxs_kwargs={"energy_kev": 18.0}))
        assert params["waxs_cal_overrides"] == {"energy_kev": 18.0}

    def test_transmission_waxs_kwargs(self):
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(waxs_kwargs={"sample_distance_mm": 300.0}))
        assert params["waxs_kwargs"] == {"sample_distance_mm": 300.0}

    # -- Derived-analysis kwargs pass-through --
    #
    # ``reduce_smi_combined`` accepts all three; ``reduce_smi_gi`` only
    # accepts ``line_cuts`` (``virtual_axes`` and ``peak_fits`` operate on
    # ``per_frame_iq``, which GI does not produce).

    def test_transmission_forwards_all_derived_kwargs(self):
        va = object()  # opaque sentinel — build_proc_params is pass-through
        cuts = [{"kind": "h", "center": 0.5, "width": 0.05}]
        peaks = [{"q_min": 0.1, "q_max": 0.2}]
        fn, params = build_proc_params(
            self.UID, "transmission",
            **self._defaults(),
            virtual_axes=va, line_cuts=cuts, peak_fits=peaks,
        )
        assert fn == "reduce_smi_combined"
        assert params["virtual_axes"] is va
        assert params["line_cuts"] == cuts
        assert params["peak_fits"] == peaks

    def test_transmission_omits_unset_derived_kwargs(self):
        fn, params = build_proc_params(self.UID, "transmission", **self._defaults())
        assert "virtual_axes" not in params
        assert "line_cuts" not in params
        assert "peak_fits" not in params

    def test_gi_forwards_only_line_cuts(self):
        # virtual_axes / peak_fits would raise TypeError in reduce_smi_gi;
        # build_proc_params must drop them silently for the GI branch.
        cuts = [{"kind": "v", "center": 0.0, "width": 0.1, "target": "qxy_qz"}]
        fn, params = build_proc_params(
            self.UID, "grazing",
            **self._defaults(),
            virtual_axes=object(),
            line_cuts=cuts,
            peak_fits=[{"q_min": 0.1, "q_max": 0.2}],
        )
        assert fn == "reduce_smi_gi"
        assert params["line_cuts"] == cuts
        assert "virtual_axes" not in params
        assert "peak_fits" not in params

    def test_gi_omits_unset_line_cuts(self):
        fn, params = build_proc_params(self.UID, "grazing", **self._defaults())
        assert "line_cuts" not in params
