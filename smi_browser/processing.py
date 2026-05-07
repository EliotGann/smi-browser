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
    # GI grid
    n_qxy: int = 500,
    n_qz: int = 500,
    theta_offset: float = -0.5,
    # Shared
    dezinger: float = 3000.0,
    incident_angle_auto: bool = True,
    incident_angle: float = 0.0,
    cache_geometry: bool = True,
    # Defaults (to know whether to send a value or omit it)
    default_n_q: int = 2000,
    default_n_chi: int = 360,
    default_n_qxy: int = 500,
    default_n_qz: int = 500,
    default_theta_offset: float = -0.5,
    default_dezinger: float = 3000.0,
    default_saxs_row_delta: float = 0.0,
    default_saxs_col_delta: float = 0.0,
    default_waxs_row_delta: float = 0.0,
    default_waxs_col_delta: float = 0.0,
    default_saxs_dist_delta: float = 0.0,
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
        if not incident_angle_auto:
            params["incident_angle_deg"] = incident_angle
        return "reduce_smi_gi", params

    # Transmission / combined
    params = dict(
        uid=uid,
        tiled_uri=tiled_uri,
        catalog=catalog,
        solid_angle_correction=True,
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

    if dezinger != default_dezinger:
        params["dezinger_threshold"] = dezinger if dezinger > 0 else None

    return "reduce_smi_combined", params
