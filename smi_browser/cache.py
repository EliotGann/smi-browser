"""Per-scan disk cache to keep batch processing within memory budget.

Each scan UID gets its own HDF5 file under
``${SMI_BROWSER_CACHE_DIR}`` (default ``${TMPDIR:-/tmp}/smi_browser_cache``).
The cache stores the expensive-to-fetch pieces:

* ``/primary``  — primary-stream scalar columns
* ``/baseline`` — baseline-stream scalar columns
* ``/images/<field>`` — raw detector stacks (one dataset per image field)
* ``/reduction`` — latest reduction outputs (overwritten on re-process)

Lightweight metadata (``start``/``stop`` docs) is *not* cached here; the caller
keeps it in memory.

The cache is process-safe enough for the single-worker batch processor: each
scan writes to its own file, and writes happen behind a per-file ``Lock``.
Cross-process safety is not attempted (one panel server per workstation).

A best-effort LRU eviction runs after writes, capping total cache size at
``SMI_BROWSER_CACHE_MAX_GB`` (default 50 GB).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def cache_root() -> Path:
    """Return the directory holding per-scan cache files.

    Resolved from ``$SMI_BROWSER_CACHE_DIR`` (preferred) or
    ``${TMPDIR:-/tmp}/smi_browser_cache``.
    """
    env = os.environ.get("SMI_BROWSER_CACHE_DIR")
    if env:
        root = Path(env).expanduser()
    else:
        root = Path(tempfile.gettempdir()) / "smi_browser_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_max_bytes() -> int:
    """Total disk-usage cap for the cache directory."""
    raw = os.environ.get("SMI_BROWSER_CACHE_MAX_GB", "50")
    try:
        gb = float(raw)
    except ValueError:
        gb = 50.0
    # Clamp to a non-negative value; very small caps are allowed (useful for
    # tests) and simply trigger aggressive eviction.
    return max(0, int(gb * (1024 ** 3)))


def cache_path(uid: str) -> Path:
    """Per-scan HDF5 path."""
    if not uid:
        raise ValueError("cache_path requires a non-empty uid")
    # Keep the full uid in the filename for traceability.
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in uid)
    return cache_root() / f"{safe}.h5"


# ---------------------------------------------------------------------------
# Per-file locking
# ---------------------------------------------------------------------------

_LOCK_TABLE: dict[str, threading.Lock] = {}
_LOCK_TABLE_LOCK = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCK_TABLE_LOCK:
        lock = _LOCK_TABLE.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCK_TABLE[key] = lock
        return lock


# ---------------------------------------------------------------------------
# ScanCache
# ---------------------------------------------------------------------------

class ScanCache:
    """Read/write helpers for a single scan's cache file.

    The class itself is stateless w.r.t. the h5 file: every method opens
    the file, does its work, and closes.  This keeps file descriptors
    bounded when the cache is touched from many call sites and avoids
    long-lived ``h5py.File`` handles that would prevent eviction.
    """

    def __init__(self, uid: str):
        self.uid = uid
        self.path = cache_path(uid)
        self._lock = _file_lock(self.path)

    # -- presence ------------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    def has_group(self, name: str) -> bool:
        import h5py
        if not self.path.exists():
            return False
        with self._lock:
            try:
                with h5py.File(self.path, "r") as f:
                    return name in f
            except OSError:
                return False

    # -- scalars (primary / baseline) ----------------------------------

    def read_scalars(self, stream: str) -> dict[str, np.ndarray] | None:
        """Return cached scalar columns for ``stream`` or ``None``."""
        import h5py
        if not self.path.exists():
            return None
        group = self._scalar_group(stream)
        with self._lock:
            try:
                with h5py.File(self.path, "r") as f:
                    if group not in f:
                        return None
                    g = f[group]
                    out: dict[str, np.ndarray] = {}
                    for key in g.keys():
                        out[key] = g[key][...]
                    return out
            except OSError as exc:
                log.warning("ScanCache.read_scalars(%s): %s", stream, exc)
                return None

    def write_scalars(self, stream: str, data: dict[str, np.ndarray]) -> None:
        import h5py
        if not data:
            return
        group = self._scalar_group(stream)
        with self._lock:
            with h5py.File(self.path, "a") as f:
                if group in f:
                    del f[group]
                g = f.create_group(group)
                for key, arr in data.items():
                    arr = np.asarray(arr)
                    if arr.dtype == object:
                        # h5py cannot store object arrays directly; coerce.
                        try:
                            arr = arr.astype(float)
                        except (TypeError, ValueError):
                            arr = np.array(
                                [str(v) for v in arr.tolist()],
                                dtype=h5py.string_dtype(),
                            )
                    g.create_dataset(key, data=arr, compression="gzip",
                                     compression_opts=4)
        _maybe_evict()

    @staticmethod
    def _scalar_group(stream: str) -> str:
        if stream not in ("primary", "baseline"):
            # Allow custom streams too, just sanitise the path component.
            stream = stream.replace("/", "_")
        return stream

    # -- raw image stacks ---------------------------------------------

    def read_image_stack(self, field: str) -> np.ndarray | None:
        """Return the full image stack for ``field`` or ``None``.

        For very large stacks the caller should prefer ``open_image_dataset``
        which yields an h5py dataset that supports slicing.
        """
        import h5py
        if not self.path.exists():
            return None
        with self._lock:
            try:
                with h5py.File(self.path, "r") as f:
                    ds = f.get(f"images/{field}")
                    if ds is None:
                        return None
                    return ds[...]
            except OSError as exc:
                log.warning("ScanCache.read_image_stack(%s): %s", field, exc)
                return None

    def write_image_stack(self, field: str, arr: np.ndarray) -> None:
        import h5py
        arr = np.asarray(arr)
        with self._lock:
            with h5py.File(self.path, "a") as f:
                grp = f.require_group("images")
                if field in grp:
                    del grp[field]
                # Per-frame chunks keep random-access reads cheap.
                if arr.ndim >= 3:
                    chunks = (1,) + tuple(arr.shape[1:])
                else:
                    chunks = True
                grp.create_dataset(
                    field, data=arr, chunks=chunks,
                    compression="gzip", compression_opts=2,
                )
        _maybe_evict()

    def has_image_field(self, field: str) -> bool:
        return self.has_group(f"images/{field}")

    # -- reduction -----------------------------------------------------

    def read_reduction(self) -> dict[str, Any] | None:
        """Return the cached reduction blob or ``None``.

        The blob mirrors :func:`write_reduction` and is intended to be a
        roundtrip for downstream re-display, not a binary clone of the
        original objects.
        """
        import h5py
        if not self.path.exists():
            return None
        with self._lock:
            try:
                with h5py.File(self.path, "r") as f:
                    if "reduction" not in f:
                        return None
                    g = f["reduction"]
                    return {
                        "params": _attrs_to_dict(g.attrs),
                        "arrays": {k: g[k][...] for k in g.keys()},
                    }
            except OSError as exc:
                log.warning("ScanCache.read_reduction: %s", exc)
                return None

    def write_reduction(self, arrays: dict[str, np.ndarray],
                        params: dict[str, Any] | None = None) -> None:
        """Persist reduction outputs, overwriting any previous run."""
        import h5py
        with self._lock:
            with h5py.File(self.path, "a") as f:
                if "reduction" in f:
                    del f["reduction"]
                g = f.create_group("reduction")
                for k, v in (arrays or {}).items():
                    g.create_dataset(k, data=np.asarray(v),
                                     compression="gzip", compression_opts=4)
                _dict_to_attrs(g.attrs, params or {})
        _maybe_evict()

    # -- maintenance ---------------------------------------------------

    def delete(self) -> None:
        with self._lock:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                log.warning("ScanCache.delete: %s", exc)


# ---------------------------------------------------------------------------
# Attr helpers
# ---------------------------------------------------------------------------

def _dict_to_attrs(attrs, params: dict[str, Any]) -> None:
    for k, v in params.items():
        try:
            if v is None:
                attrs[k] = "None"
            elif isinstance(v, (list, tuple)):
                attrs[k] = list(v)
            elif isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                attrs[k] = v
            else:
                attrs[k] = json.dumps(v, default=str)
        except (TypeError, ValueError):
            attrs[k] = str(v)


def _attrs_to_dict(attrs) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in attrs:
        v = attrs[k]
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

_EVICT_LOCK = threading.Lock()


def _maybe_evict() -> None:
    """Drop oldest cache files when total size exceeds the configured cap."""
    cap = cache_max_bytes()
    # Bail fast if we are nowhere near the cap; rechecking the directory
    # on every write is cheap (one stat per file) but still worth skipping.
    if not _EVICT_LOCK.acquire(blocking=False):
        return
    try:
        root = cache_root()
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in root.iterdir():
            if not p.is_file() or p.suffix != ".h5":
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        if total <= cap:
            return
        # Oldest first.
        entries.sort(key=lambda e: e[0])
        for _, size, p in entries:
            if total <= cap:
                break
            lock = _file_lock(p)
            with lock:
                try:
                    p.unlink()
                    total -= size
                    log.info("cache: evicted %s (%.1f MB)", p.name,
                             size / (1024 ** 2))
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    log.warning("cache: eviction failed for %s: %s", p, exc)
    finally:
        _EVICT_LOCK.release()


# ---------------------------------------------------------------------------
# Cache-aware fetch helpers
# ---------------------------------------------------------------------------

def get_or_fetch_scalars(uid: str, stream: str, fetch_fn) -> dict[str, np.ndarray]:
    """Return cached scalars or fetch + cache.

    ``fetch_fn`` is a zero-arg callable returning a ``{field: ndarray}``
    dict (matching :func:`smi_browser._tiled.fetch_scalars`).  It is only
    invoked on a cache miss.
    """
    cache = ScanCache(uid)
    cached = cache.read_scalars(stream)
    if cached is not None:
        return cached
    try:
        data = fetch_fn() or {}
    except Exception:
        log.exception("cache: fetch_fn failed for %s/%s", uid, stream)
        return {}
    if data:
        try:
            cache.write_scalars(stream, data)
        except Exception:
            log.exception("cache: write_scalars failed for %s/%s", uid, stream)
    return data


def get_or_fetch_image_stack(uid: str, field: str, fetch_fn) -> np.ndarray | None:
    """Return cached image stack or fetch + cache.

    ``fetch_fn`` is a zero-arg callable returning the ndarray.
    """
    cache = ScanCache(uid)
    cached = cache.read_image_stack(field)
    if cached is not None:
        return cached
    try:
        arr = fetch_fn()
    except Exception:
        log.exception("cache: fetch_fn failed for %s/images/%s", uid, field)
        return None
    if arr is None:
        return None
    try:
        cache.write_image_stack(field, arr)
    except Exception:
        log.exception("cache: write_image_stack failed for %s/%s", uid, field)
    return arr


def get_or_fetch_image_frame(
    uid: str,
    field: str,
    frame_idx: int,
    fetch_stack_fn,
) -> np.ndarray | None:
    """Return a single frame from the cached image stack, fetching on miss.

    On a cache miss the *full* stack is fetched via ``fetch_stack_fn()`` and
    persisted so subsequent frame requests are served from disk.

    ``fetch_stack_fn`` is a zero-arg callable returning the full 3-D stack
    (or None).
    """
    import h5py

    cache = ScanCache(uid)

    # Fast path: read just the requested frame slice from the h5 dataset
    if cache.path.exists():
        with cache._lock:
            try:
                with h5py.File(cache.path, "r") as f:
                    ds = f.get(f"images/{field}")
                    if ds is not None:
                        if ds.ndim >= 3:
                            return ds[frame_idx]
                        else:
                            return ds[...]
            except OSError:
                pass

    # Cache miss — fetch the full stack, write it, then return the frame
    try:
        stack = fetch_stack_fn()
    except Exception:
        log.exception("cache: fetch_stack_fn failed for %s/images/%s", uid, field)
        return None
    if stack is None:
        return None
    stack = np.asarray(stack)
    try:
        cache.write_image_stack(field, stack)
    except Exception:
        log.exception("cache: write_image_stack failed for %s/%s", uid, field)
    # Return requested frame
    if stack.ndim >= 3:
        return stack[frame_idx]
    return stack
