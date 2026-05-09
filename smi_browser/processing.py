"""Pure param-builder for SMI data reduction.

Extracts the duplicated kwarg-construction logic that was shared between
``_on_process`` (interactive) and ``_build_proc_params`` (batch).
"""

from __future__ import annotations

from typing import Any


def build_proc_params(
    uid: str,
    geometry: str,
    *,
    tiled_uri: str,
    catalog: str,
    # Mask paths
    saxs_mask_path: str = "",
    waxs_mask_path: str = "",
    # Transmission grid
    n_q: int = 2000,
    n_chi: int = 360,
    # Beam-centre deltas
    saxs_row_delta: float = 0.0,
    saxs_col_delta: float = 0.0,
    waxs_row_delta: float = 0.0,
    waxs_col_delta: float = 0.0,
    saxs_dist_delta: float = 0.0,
    # WAXS beam col per arc
    waxs_beam_col_per_arc_deg: float = 0.0,
    # GI grid
    n_qxy: int = 500,
    n_qz: int = 500,
    theta_offset: float = -0.5,
    # Shared
    dezinger: float = 3000.0,
    dezinger_kernel: int = 5,
    incident_angle_auto: bool = True,
    incident_angle: float = 0.0,
    cache_geometry: bool = True,
    solid_angle_correction: bool = True,
    # SAXS Q-range / aperture
    saxs_q_cutoff: float = 0.0,
    saxs_agbh_ring_order: int = 5,
    saxs_q_margin_fraction: float = 0.01,
    # GI-specific
    beamstop_max_abs_arc_deg: float = 15.0,
    # Backend / display options
    saxs_rotate_cw_90: bool = False,
    waxs_flip_horizontal: bool = False,
    waxs_qx_shift_nm: float = 0.0,
    waxs_qy_shift_nm: float = 0.0,
    # Dynamic SAXS masking
    dynamic_saxs_mask: bool = False,
    dyn_shadow_enabled: bool = True,
    dyn_shadow_beam_visible_deg: float = 14.5,
    dyn_shadow_clear_edge_deg: float = 18.0,
    dyn_aper_enabled: bool = True,
    dyn_aper_agbh_ring_order: int = 5,
    dyn_aper_q_margin_fraction: float = 0.01,
    dyn_aper_q_cutoff: float = 0.0,
    # WAXS calibration overrides (passed as waxs_kwargs / waxs_cal_overrides)
    waxs_kwargs: dict[str, Any] | None = None,
    # Defaults (to know whether to send a value or omit it)
    default_n_q: int = 2000,
    default_n_chi: int = 360,
    default_n_qxy: int = 500,
    default_n_qz: int = 500,
    default_theta_offset: float = -0.5,
    default_dezinger: float = 3000.0,
    default_dezinger_kernel: int = 5,
    default_saxs_row_delta: float = 0.0,
    default_saxs_col_delta: float = 0.0,
    default_waxs_row_delta: float = 0.0,
    default_waxs_col_delta: float = 0.0,
    default_saxs_dist_delta: float = 0.0,
    default_waxs_beam_col_per_arc_deg: float = 0.0,
    default_beamstop_max_abs_arc_deg: float = 15.0,
    default_saxs_agbh_ring_order: int = 5,
    default_saxs_q_margin_fraction: float = 0.01,
) -> tuple[str, dict[str, Any]]:
    """Build reduction kwargs from widget-level values.

    Returns ``(reduce_function_name, params_dict)`` where
    *reduce_function_name* is ``"reduce_smi_gi"`` or ``"reduce_smi_combined"``.

    The caller is responsible for importing the actual callable and invoking it.
    """
    if geometry == "grazing":
        params: dict[str, Any] = dict(
            uid=uid,
            tiled_uri=tiled_uri,
            catalog=catalog,
            waxs_mask_path=waxs_mask_path or None,
        )
        if n_qxy != default_n_qxy:
            params["n_qxy"] = n_qxy
        if n_qz != default_n_qz:
            params["n_qz"] = n_qz
        if theta_offset != default_theta_offset:
            params["theta_offset"] = theta_offset
        if dezinger != default_dezinger:
            params["dezinger_threshold"] = dezinger if dezinger > 0 else None
        if dezinger_kernel != default_dezinger_kernel:
            params["dezinger_kernel"] = dezinger_kernel
        if not incident_angle_auto:
            params["incident_angle_deg"] = incident_angle
        if beamstop_max_abs_arc_deg != default_beamstop_max_abs_arc_deg:
            params["beamstop_max_abs_arc_deg"] = beamstop_max_abs_arc_deg
        if waxs_beam_col_per_arc_deg != default_waxs_beam_col_per_arc_deg:
            params["waxs_beam_col_per_arc_deg"] = waxs_beam_col_per_arc_deg
        if waxs_kwargs:
            params["waxs_cal_overrides"] = waxs_kwargs
        return "reduce_smi_gi", params

    # Transmission / combined
    params = dict(
        uid=uid,
        tiled_uri=tiled_uri,
        catalog=catalog,
        solid_angle_correction=solid_angle_correction,
        geometry=geometry,
        saxs_mask_path=saxs_mask_path or None,
        waxs_mask_path=waxs_mask_path or None,
        cache_geometry=cache_geometry,
    )
    if n_q != default_n_q:
        params["n_q"] = n_q
    else:
        params["n_q"] = default_n_q
    if n_chi != default_n_chi:
        params["n_chi"] = n_chi

    saxs_row_changed = saxs_row_delta != default_saxs_row_delta
    saxs_col_changed = saxs_col_delta != default_saxs_col_delta
    if saxs_row_changed or saxs_col_changed:
        params["saxs_beam_delta_px"] = (saxs_row_delta, saxs_col_delta)

    waxs_row_changed = waxs_row_delta != default_waxs_row_delta
    waxs_col_changed = waxs_col_delta != default_waxs_col_delta
    if waxs_row_changed or waxs_col_changed:
        params["waxs_beam_delta_px"] = (waxs_row_delta, waxs_col_delta)

    if saxs_dist_delta != default_saxs_dist_delta:
        params["saxs_distance_delta_mm"] = saxs_dist_delta

    if waxs_beam_col_per_arc_deg != default_waxs_beam_col_per_arc_deg:
        params["waxs_beam_col_per_arc_deg"] = waxs_beam_col_per_arc_deg

    if dezinger != default_dezinger:
        params["dezinger_threshold"] = dezinger if dezinger > 0 else None
    if dezinger_kernel != default_dezinger_kernel:
        params["dezinger_kernel"] = dezinger_kernel

    # SAXS Q-range / aperture
    if saxs_q_cutoff > 0:
        params["saxs_q_cutoff"] = saxs_q_cutoff
    if saxs_agbh_ring_order != default_saxs_agbh_ring_order:
        params["saxs_agbh_ring_order"] = saxs_agbh_ring_order
    if saxs_q_margin_fraction != default_saxs_q_margin_fraction:
        params["saxs_q_margin_fraction"] = saxs_q_margin_fraction

    # Backend options
    backend_opts: dict[str, Any] = {}
    if saxs_rotate_cw_90:
        backend_opts["saxs_rotate_cw_90"] = True
    if waxs_flip_horizontal:
        backend_opts["waxs_flip_horizontal"] = True
    if waxs_qx_shift_nm != 0.0:
        backend_opts["waxs_qx_shift_nm"] = waxs_qx_shift_nm
    if waxs_qy_shift_nm != 0.0:
        backend_opts["waxs_qy_shift_nm"] = waxs_qy_shift_nm
    if backend_opts:
        params["backend_options"] = backend_opts

    # Dynamic SAXS masking
    if dynamic_saxs_mask:
        saxs_kw: dict[str, Any] = {"dynamic_saxs_mask": True}
        dyn_kwargs: dict[str, Any] = {}
        shadow: dict[str, Any] = {}
        if not dyn_shadow_enabled:
            shadow["enabled"] = False
        if dyn_shadow_beam_visible_deg != 14.5:
            shadow["beam_visible_deg"] = dyn_shadow_beam_visible_deg
        if dyn_shadow_clear_edge_deg != 18.0:
            shadow["clear_edge_deg"] = dyn_shadow_clear_edge_deg
        if shadow:
            dyn_kwargs["waxs_shadow"] = shadow
        aperture: dict[str, Any] = {}
        if not dyn_aper_enabled:
            aperture["enabled"] = False
        if dyn_aper_agbh_ring_order != 5:
            aperture["agbh_ring_order"] = dyn_aper_agbh_ring_order
        if dyn_aper_q_margin_fraction != 0.01:
            aperture["q_margin_fraction"] = dyn_aper_q_margin_fraction
        if dyn_aper_q_cutoff > 0:
            aperture["q_cutoff"] = dyn_aper_q_cutoff
        if aperture:
            dyn_kwargs["aperture"] = aperture
        if dyn_kwargs:
            saxs_kw["dynamic_saxs_kwargs"] = dyn_kwargs
        params["saxs_kwargs"] = saxs_kw

    # WAXS kwargs (calibration + masking overrides)
    if waxs_kwargs:
        params["waxs_kwargs"] = waxs_kwargs

    return "reduce_smi_combined", params
