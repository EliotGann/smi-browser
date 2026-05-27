"""Enhanced metadata summary for tiled runs."""
from __future__ import annotations

import datetime
from typing import Any

from ..config import SAXS_DETECTOR_NAMES, WAXS_DETECTOR_NAMES


_MISSING = object()


def _flatten_leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Walk a nested dict and yield ``{dotted_path: leaf_value}``.

    Lists/tuples are kept whole (treated as leaves) — element-wise comparison
    inside a list isn't useful here.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_leaves(v, key))
            else:
                out[key] = v
    else:
        out[prefix] = obj
    return out


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` for ``dotted_key = "a.b.c"``."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
        if not isinstance(cur, dict):
            # Conflict between leaf and subtree — give up rather than overwrite.
            return
    cur[parts[-1]] = value


def varying_keys(metas: list[dict[str, Any]],
                 skip_suffixes: set[str] | None = None) -> dict[str, list]:
    """Return *flattened* metadata paths whose values differ across the dicts.

    Nested dicts are flattened with ``.``-joined paths (e.g.
    ``start.sample_name``) so leaf-level differences in tiled metadata are
    surfaced — comparing only top-level keys is useless when the entire scan
    is buried under ``start``/``stop``.

    The returned mapping is ``{dotted_path: [value_for_meta_0, value_for_meta_1, ...]}``;
    a missing key in a given dict is represented as the sentinel string
    ``"<missing>"``.

    Parameters
    ----------
    metas
        List of summary or full-metadata dicts.
    skip_suffixes
        Paths whose final component is in this set are dropped from the diff.
        Defaults to per-scan-unique identifiers (``uid``, ``scan_id``,
        timestamps, ``run_start``).
    """
    if len(metas) < 2:
        return {}
    if skip_suffixes is None:
        skip_suffixes = {
            "uid", "scan_id", "time", "time_iso", "run_start",
        }

    flat: list[dict[str, Any]] = [_flatten_leaves(m) for m in metas]
    all_keys: set[str] = set()
    for f in flat:
        all_keys.update(f.keys())

    out: dict[str, list] = {}
    for key in sorted(all_keys):
        tail = key.rsplit(".", 1)[-1]
        if tail in skip_suffixes:
            continue
        vals = [f.get(key, _MISSING) for f in flat]
        # Comparable string keys are sufficient — handles unhashable list values.
        repr_vals = ["<missing>" if v is _MISSING else str(v) for v in vals]
        if len(set(repr_vals)) > 1:
            out[key] = [None if v is _MISSING else v for v in vals]
    return out


def reconstruct_nested(varying: dict[str, list], n: int) -> list[dict]:
    """Rebuild ``n`` nested dicts, each containing only the varying paths.

    Useful for the multi-select diff view: pass the result of
    :func:`varying_keys` and get back N dicts (one per original scan) with
    the same nested shape as the source but only the differing leaves.
    """
    out: list[dict] = [{} for _ in range(n)]
    for dotted, vals in varying.items():
        for i in range(min(n, len(vals))):
            if vals[i] is None:
                continue  # was missing in this scan
            _set_nested(out[i], dotted, vals[i])
    return out


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
