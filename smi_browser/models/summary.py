"""Enhanced metadata summary for tiled runs."""
from __future__ import annotations

import datetime

from ..config import SAXS_DETECTOR_NAMES, WAXS_DETECTOR_NAMES


def enhanced_summary(run) -> dict:
    """
    Build a lightweight summary from a tiled run node.
    Only reads .metadata — zero array I/O.  Extends tb.run_summary with
    detector classification and institution info.
    """
    md = run.metadata
    start = md.get("start", {})
    stop = md.get("stop", {})

    t0 = start.get("time")
    time_str = (
        datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M")
        if t0 else "?"
    )

    # Detectors
    det_list = start.get("detectors", [])
    if isinstance(det_list, str):
        det_list = [det_list]
    det_lower = {d.lower() for d in det_list}
    has_saxs = bool(det_lower & SAXS_DETECTOR_NAMES)
    has_waxs = bool(det_lower & WAXS_DETECTOR_NAMES)
    if has_saxs and has_waxs:
        det_str = "SAXS+WAXS"
    elif has_saxs:
        det_str = "SAXS"
    elif has_waxs:
        det_str = "WAXS"
    else:
        det_str = ", ".join(det_list) if det_list else "?"

    # Steps
    num_events = stop.get("num_events", {})
    if isinstance(num_events, dict):
        n_steps = num_events.get("primary", "?")
    else:
        n_steps = num_events

    # Institution / data session
    institution = start.get(
        "institution",
        start.get("data_session", start.get("proposal_id", "?")),
    )

    exit_status = stop.get("exit_status", "?")

    return {
        "uid":          start.get("uid", "?"),
        "scan_id":      start.get("scan_id", "?"),
        "time":         time_str,
        "plan_name":    start.get("plan_name", "?"),
        "sample_name":  start.get(
            "sample_name",
            start.get("sample", start.get("Sample", "?")),
        ),
        "exit_status":  exit_status,
        "n_steps":      n_steps,
        "institution":  str(institution),
        "detectors":    det_str,
        "has_saxs":     has_saxs,
        "has_waxs":     has_waxs,
        "detector_list": det_list,
    }
