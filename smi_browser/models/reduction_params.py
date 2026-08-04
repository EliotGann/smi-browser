"""Helpers for resolved physical reduction parameters from smi-tiled."""
from __future__ import annotations

import json
from typing import Any

import numpy as np


SCHEMA = "smi_tiled.resolved_reduction_parameters.v1"

FLAT_ATTR_KEYS = (
    "saxs_sdd_mm",
    "saxs_beam_center_row_px",
    "saxs_beam_center_col_px",
    "saxs_poni1_m",
    "saxs_poni2_m",
    "saxs_energy_kev",
    "saxs_wavelength_angstrom",
    "saxs_active_beamstop",
    "waxs_sdd_mm",
    "waxs_beam_center_row_px",
    "waxs_beam_center_col_px",
    "waxs_energy_kev",
    "waxs_wavelength_angstrom",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def loads_params(value: Any) -> dict[str, Any] | None:
    """Decode a resolved-parameter JSON string/dict, if present."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def dumps_params(params: dict[str, Any]) -> str:
    """Encode resolved parameters with stable JSON for attrs/HDF5."""
    return json.dumps(_json_safe(params), sort_keys=True)


def from_result(result: Any) -> dict[str, Any] | None:
    """Return resolved physical parameters from a live or cached result."""
    params = loads_params(getattr(result, "reduction_parameters", None))
    if params:
        return params
    for product_name in ("merged_iq", "merged_qchi", "per_frame_iq"):
        product = getattr(result, product_name, None)
        attrs = getattr(product, "attrs", None)
        if attrs:
            params = loads_params(attrs.get("smi_reduction_parameters"))
            if params:
                return params
    return None


def flat_attrs(params: dict[str, Any] | None) -> dict[str, Any]:
    """Convenient physical attrs matching smi-tiled's xarray output attrs."""
    if not params:
        return {}
    out: dict[str, Any] = {}
    saxs = params.get("saxs") if isinstance(params, dict) else None
    if isinstance(saxs, dict):
        out.update({
            "saxs_sdd_mm": saxs.get("sample_detector_distance_mm"),
            "saxs_beam_center_row_px": saxs.get("beam_center_row_px"),
            "saxs_beam_center_col_px": saxs.get("beam_center_col_px"),
            "saxs_poni1_m": saxs.get("poni1_m"),
            "saxs_poni2_m": saxs.get("poni2_m"),
            "saxs_energy_kev": saxs.get("energy_kev"),
            "saxs_wavelength_angstrom": saxs.get("wavelength_angstrom"),
            "saxs_active_beamstop": saxs.get("active_beamstop"),
        })
    waxs = params.get("waxs") if isinstance(params, dict) else None
    if isinstance(waxs, dict):
        out.update({
            "waxs_sdd_mm": waxs.get("sample_detector_distance_mm"),
            "waxs_beam_center_row_px": waxs.get("beam_center_row_px"),
            "waxs_beam_center_col_px": waxs.get("beam_center_col_px"),
            "waxs_energy_kev": waxs.get("energy_kev"),
            "waxs_wavelength_angstrom": waxs.get("wavelength_angstrom"),
        })
    return {k: v for k, v in out.items() if v is not None}


def attach_attrs(product: Any, params: dict[str, Any] | None) -> None:
    """Attach resolved-parameter JSON and flat attrs to an xarray object."""
    attrs = getattr(product, "attrs", None)
    if attrs is None or not params:
        return
    attrs["smi_reduction_parameters_schema"] = SCHEMA
    attrs["smi_reduction_parameters"] = dumps_params(params)
    attrs.update(flat_attrs(params))


def summary_lines(params: dict[str, Any] | None) -> list[str]:
    """Short, user-facing physical-geometry summary lines."""
    if not params:
        return []
    lines: list[str] = []
    for label, key in (("SAXS", "saxs"), ("WAXS", "waxs")):
        det = params.get(key)
        if not isinstance(det, dict):
            continue
        parts = []
        sdd = det.get("sample_detector_distance_mm")
        if sdd is not None:
            parts.append(f"SDD {float(sdd):.2f} mm")
        row = det.get("beam_center_row_px")
        col = det.get("beam_center_col_px")
        if row is not None and col is not None:
            parts.append(f"beam ({float(row):.2f}, {float(col):.2f}) px")
        energy = det.get("energy_kev")
        if energy is not None:
            parts.append(f"E {float(energy):.4g} keV")
        if parts:
            lines.append(f"{label}: " + ", ".join(parts))
    return lines
