"""Opt-in, low-overhead memory tracking for the SMI browser.

Enable by exporting ``SMI_BROWSER_MEM_LOG=1`` before launching the app.  When
enabled, :func:`report` logs the process resident-set size (RSS) plus a
best-effort breakdown of the big in-memory holders, so we can see *where*
memory goes across reductions, image loads, and peak fits.  When disabled,
every entry point is a couple of cheap no-ops.

Deliberately dependency-free (stdlib + numpy/pandas/xarray, all already
present) so it can be imported from any layer, including background threads.
"""
from __future__ import annotations

import logging
import os

__all__ = ["enabled", "rss_mb", "nbytes", "report", "log_rss"]

log = logging.getLogger("smi_browser.memlog")

_ENABLED = os.environ.get("SMI_BROWSER_MEM_LOG", "").strip().lower() in (
    "1", "true", "yes", "on")


def _configure_once() -> None:
    """Give the memlog logger a visible INFO handler (app sets no basicConfig)."""
    if log.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


if _ENABLED:
    _configure_once()


def enabled() -> bool:
    return _ENABLED


def rss_mb() -> float:
    """Resident-set size of this process in MiB (Linux /proc, with fallback)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        # ru_maxrss is KiB on Linux, bytes on macOS — assume Linux here.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def _is_dask(da) -> bool:
    return type(getattr(da, "data", None)).__module__.startswith("dask")


def nbytes(obj) -> int:
    """Best-effort *resident* byte size of common containers.

    Lazy (dask-backed) xarray data counts as 0 — it isn't in RAM.  Unknown
    objects also count as 0 rather than guessing.
    """
    if obj is None:
        return 0
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
    except Exception:
        pass
    try:
        import xarray as xr
        if isinstance(obj, xr.DataArray):
            return 0 if _is_dask(obj) else int(obj.nbytes)
        if isinstance(obj, xr.Dataset):
            return sum(nbytes(obj[v]) for v in obj.variables)
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(deep=True).sum())
        if isinstance(obj, pd.Series):
            return int(obj.memory_usage(deep=True))
    except Exception:
        pass
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def _fmt(b: float) -> str:
    return f"{b / 1048576:.1f}MB"


def report(tag: str, parts: dict | None = None) -> None:
    """Log ``MEM[tag] rss=… | label=size …``.

    ``parts`` maps a label to either a raw byte count (int/float) or an object
    to size via :func:`nbytes`.  No-op unless ``SMI_BROWSER_MEM_LOG`` is set.
    """
    if not _ENABLED:
        return
    try:
        msg = f"MEM[{tag}] rss={rss_mb():.0f}MB"
        if parts:
            sized = []
            for label, val in parts.items():
                b = val if isinstance(val, (int, float)) else nbytes(val)
                if b:
                    sized.append((label, b))
            sized.sort(key=lambda kv: kv[1], reverse=True)
            if sized:
                msg += " | " + " ".join(f"{k}={_fmt(b)}" for k, b in sized)
        log.info(msg)
    except Exception:  # never let instrumentation break the app
        log.debug("memlog.report failed", exc_info=True)


def log_rss(tag: str) -> None:
    """Log just the RSS for ``tag`` (no breakdown)."""
    report(tag)
