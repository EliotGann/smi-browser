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
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_warned_fallback = False


def cache_root() -> Path:
    """Return the directory holding per-scan cache files.

    Resolved from ``$SMI_BROWSER_CACHE_DIR`` (preferred), then
    ``${TMPDIR:-/tmp}/smi_browser_cache`` if that directory is writable,
    otherwise ``~/.local/share/smi_browser_cache`` as a user-owned fallback.
    """
    global _warned_fallback
    env = os.environ.get("SMI_BROWSER_CACHE_DIR")
    if env:
        root = Path(env).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root

    tmp_root = Path(tempfile.gettempdir()) / "smi_browser_cache"
    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
        if os.access(tmp_root, os.W_OK):
            return tmp_root
    except OSError:
        pass

    # Shared /tmp cache exists but is owned by another user — fall back to
    # a user-owned location so writes don't fail silently.
    fallback = Path.home() / ".local" / "share" / "smi_browser_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    # Warn once per process — cache_root() runs on every ScanCache (once per
    # frame), which would otherwise flood the log during live mode.
    if not _warned_fallback:
        _warned_fallback = True
        log.warning(
            "cache: %s is not writable; using %s instead. "
            "Set SMI_BROWSER_CACHE_DIR to override.",
            tmp_root, fallback,
        )
    return fallback


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


def _path_size(path: Path) -> int:
    """Return recursive disk usage for a file or directory, best effort."""
    try:
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
        if not path.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def disk_cache_info() -> dict[str, Any]:
    """Summarize on-disk cache usage under :func:`cache_root`.

    The root is shared by smi-browser and smi-tiled.  Per-scan HDF5 files hold
    raw read caches plus browser-derived reduction/peak-fit outputs; ``*_qchi``
    directories are smi-tiled zarr stores for large per-frame q-chi stacks.
    """
    root = cache_root()
    info: dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "total_bytes": 0,
        "h5_bytes": 0,
        "qchi_bytes": 0,
        "peak_defs_bytes": 0,
        "other_bytes": 0,
        "h5_files": 0,
        "qchi_dirs": 0,
        "other_entries": 0,
    }
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        info["error"] = str(exc)
        return info

    peak_path = peak_defs_path()
    for path in entries:
        size = _path_size(path)
        info["total_bytes"] += size
        if path == peak_path:
            info["peak_defs_bytes"] += size
        elif path.is_file() and path.suffix == ".h5":
            info["h5_bytes"] += size
            info["h5_files"] += 1
        elif path.is_dir() and path.name.endswith("_qchi"):
            info["qchi_bytes"] += size
            info["qchi_dirs"] += 1
        else:
            info["other_bytes"] += size
            info["other_entries"] += 1
    return info


def clear_disk_cache(*, include_peak_defs: bool = False) -> dict[str, Any]:
    """Delete the shared disk cache contents, returning deletion stats.

    Saved peak definitions are preserved by default because they are user-drawn
    ranges, not a derived read/reduction cache.
    """
    root = cache_root()
    stats: dict[str, Any] = {
        "root": str(root),
        "deleted_bytes": 0,
        "deleted_entries": 0,
        "preserved_entries": 0,
        "errors": [],
    }
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        stats["errors"].append(str(exc))
        return stats

    peak_path = peak_defs_path()
    for path in entries:
        if path == peak_path and not include_peak_defs:
            stats["preserved_entries"] += 1
            continue
        size = _path_size(path)
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            stats["errors"].append(f"{path}: {exc}")
            continue
        stats["deleted_bytes"] += size
        stats["deleted_entries"] += 1
    return stats


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


def prune_lock_table(max_size: int = 200) -> int:
    """Remove old entries from _LOCK_TABLE to prevent unbounded growth.

    Only removes locks that are not currently held.  Returns the number
    of entries removed.
    """
    with _LOCK_TABLE_LOCK:
        if len(_LOCK_TABLE) <= max_size:
            return 0
        # Remove unlocked entries from the front (oldest inserted first)
        to_remove = []
        for key, lock in list(_LOCK_TABLE.items()):
            if len(_LOCK_TABLE) - len(to_remove) <= max_size:
                break
            if not lock.locked():
                to_remove.append(key)
        for key in to_remove:
            del _LOCK_TABLE[key]
        return len(to_remove)


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

    def read_reduction_datasets(
        self, keys: Sequence[str],
    ) -> dict[str, np.ndarray] | None:
        """Read only the named ``reduction`` datasets, opening the file once.

        Unlike :meth:`read_reduction`, this never materialises large arrays the
        caller does not ask for (e.g. ``qchi_intensity``).  Missing datasets are
        simply absent from the returned dict; ``None`` means no reduction group.
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
                    return {k: g[k][...] for k in keys if k in g}
            except OSError as exc:
                log.warning("ScanCache.read_reduction_datasets: %s", exc)
                return None

    def write_reduction(self, arrays: dict[str, np.ndarray],
                        params: dict[str, Any] | None = None) -> None:
        """Persist reduction outputs, overwriting any previous run.

        Re-processing invalidates any cached peak fits (they were computed
        against the *previous* ``pf_iq`` stack), so the ``peakfit`` group is
        dropped here too.
        """
        import h5py
        with self._lock:
            with h5py.File(self.path, "a") as f:
                if "reduction" in f:
                    del f["reduction"]
                if "peakfit" in f:
                    del f["peakfit"]
                g = f.create_group("reduction")
                for k, v in (arrays or {}).items():
                    g.create_dataset(k, data=np.asarray(v),
                                     compression="gzip", compression_opts=4)
                _dict_to_attrs(g.attrs, params or {})
        _maybe_evict()

    # -- peak fits -----------------------------------------------------

    def write_peakfit(self, peak_key: Sequence, arrays: dict[str, np.ndarray],
                     attrs: dict[str, Any] | None = None) -> None:
        """Persist one peak's per-frame fit result, keyed by ``peak_key``.

        ``peak_key`` is ``PeakDef.key()`` (a tuple).  Stored under
        ``/peakfit/<hash>``; re-fitting the same peak overwrites it.  The full
        key is recorded as an attr so :meth:`read_peakfit_index` can rebuild it.
        """
        import h5py
        h = _peak_hash(peak_key)
        with self._lock:
            with h5py.File(self.path, "a") as f:
                root = f.require_group("peakfit")
                if h in root:
                    del root[h]
                g = root.create_group(h)
                for k, v in (arrays or {}).items():
                    arr = np.asarray(v)
                    if arr.dtype == bool:
                        arr = arr.astype("u1")
                    g.create_dataset(k, data=arr, compression="gzip",
                                     compression_opts=4)
                g.attrs["key_json"] = json.dumps(list(peak_key), default=str)
                _dict_to_attrs(g.attrs, attrs or {})
        _maybe_evict()

    def read_peakfit_full(self) -> list[dict]:
        """Return ``[{"key", "attrs", "arrays"}, ...]`` for all cached peaks.

        ``arrays`` mirrors the dict produced by ``fit_peak_across_frames``
        (``amplitude``/``center``/``fwhm``/``area`` float arrays + ``success``
        bool array); ``attrs`` carries the stored peak identity (``name``,
        ``q_min``, ``q_max``, ``model``, ...).  Used by the exporter, which
        needs the peak name and q-range for filenames/group names.
        """
        import h5py
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self._lock:
            try:
                with h5py.File(self.path, "r") as f:
                    if "peakfit" not in f:
                        return []
                    for h in f["peakfit"]:
                        g = f["peakfit"][h]
                        raw = g.attrs.get("key_json")
                        if raw is None:
                            continue
                        try:
                            key = tuple(json.loads(raw))
                        except (TypeError, ValueError):
                            continue
                        arrays: dict[str, np.ndarray] = {}
                        for k in g.keys():
                            arr = g[k][...]
                            if k == "success":
                                arr = arr.astype(bool)
                            arrays[k] = arr
                        attrs = _attrs_to_dict(g.attrs)
                        attrs.pop("key_json", None)
                        out.append({"key": key, "attrs": attrs, "arrays": arrays})
            except OSError as exc:
                log.warning("ScanCache.read_peakfit_full: %s", exc)
                return []
        return out

    def read_peakfit_index(self) -> list[tuple[tuple, dict[str, np.ndarray]]]:
        """Return ``[(peak_key, result_arrays), ...]`` for all cached peaks.

        Thin wrapper over :meth:`read_peakfit_full` used to repopulate the
        in-memory fit cache on load so a previously-fit map renders without
        re-fitting.
        """
        return [(e["key"], e["arrays"]) for e in self.read_peakfit_full()]

    def write_peakfit_dataset(self, peak_fits_ds) -> int:
        """Split a packed ``apply_peak_fits`` xr.Dataset into per-peak entries.

        The smi-tiled :func:`apply_peak_fits` returns a single dataset with
        dims ``(peak, frame)``, vars ``amplitude/center/fwhm/area/success``, and
        a ``ds.attrs["peaks"]`` provenance list (one dict per peak).  The cache
        layer keys one HDF5 group per peak (``/peakfit/<hash>``), so this
        helper iterates the peak dim and writes each slice via
        :meth:`write_peakfit`.  Returns the number of peaks written.

        No-ops when ``peak_fits_ds`` is ``None`` or has zero peaks.
        """
        from smi_tiled.derived.peakfit import PeakDef, FIT_PARAMS

        if peak_fits_ds is None:
            return 0
        n_peaks = int(peak_fits_ds.sizes.get("peak", 0))
        if n_peaks == 0:
            return 0
        provenance = peak_fits_ds.attrs.get("peaks") or []
        written = 0
        for pi in range(n_peaks):
            try:
                prov = dict(provenance[pi]) if pi < len(provenance) else {}
                pk = PeakDef(
                    name=str(prov.get("name") or f"p{pi + 1}"),
                    q_min=float(prov.get("q_min", 0.0)),
                    q_max=float(prov.get("q_max", 0.0)),
                    model=str(prov.get("model", "gaussian")),
                    baseline=str(prov.get("baseline", "linear")),
                    link=str(prov.get("link", "independent")),
                    bg_factor=float(prov.get("bg_factor", 2.0)),
                )
            except (TypeError, ValueError):
                continue
            arrays: dict[str, np.ndarray] = {}
            for p in FIT_PARAMS:
                if p in peak_fits_ds:
                    arrays[p] = np.asarray(peak_fits_ds[p].isel(peak=pi).values)
            if "success" in peak_fits_ds:
                arrays["success"] = np.asarray(
                    peak_fits_ds["success"].isel(peak=pi).values)
            if not arrays:
                continue
            self.write_peakfit(
                pk.key(), arrays,
                attrs={
                    "name": pk.name,
                    "q_min": pk.q_min,
                    "q_max": pk.q_max,
                    "model": pk.model,
                    "baseline": pk.baseline,
                    "link": pk.link,
                    "bg_factor": pk.bg_factor,
                },
            )
            written += 1
        return written

    def clear_peakfit(self) -> None:
        import h5py
        if not self.path.exists():
            return
        with self._lock:
            try:
                with h5py.File(self.path, "a") as f:
                    if "peakfit" in f:
                        del f["peakfit"]
            except OSError as exc:
                log.warning("ScanCache.clear_peakfit: %s", exc)

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
# Peak-fit / peak-definition helpers
# ---------------------------------------------------------------------------

def _peak_hash(peak_key: Sequence) -> str:
    """Stable, filesystem-safe digest of a ``PeakDef.key()`` tuple."""
    import hashlib
    raw = json.dumps(list(peak_key), default=str, sort_keys=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def peak_defs_path() -> Path:
    """Path to the global (cross-scan) peak-definition list."""
    return cache_root() / "peak_defs.json"


def read_peak_defs() -> list[dict]:
    """Return the persisted global peak-definition list (``[]`` if none)."""
    path = peak_defs_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("read_peak_defs: %s", exc)
        return []
    return data if isinstance(data, list) else []


def write_peak_defs(defs: list[dict]) -> None:
    """Persist the global peak-definition list (atomic replace)."""
    path = peak_defs_path()
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(defs), indent=2, default=str))
        tmp.replace(path)
    except OSError as exc:
        log.warning("write_peak_defs: %s", exc)


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
    from . import memlog
    memlog.report(f"img_stack:fetch_start {field}")
    try:
        arr = fetch_fn()
    except Exception:
        log.exception("cache: fetch_fn failed for %s/images/%s", uid, field)
        return None
    if arr is None:
        return None
    memlog.report(f"img_stack:fetched {field}", {"stack": arr})
    try:
        cache.write_image_stack(field, arr)
    except Exception:
        log.exception("cache: write_image_stack failed for %s/%s", uid, field)
    return arr


def get_or_fetch_image_frame(
    uid: str,
    field: str,
    frame_idx: int,
    fetch_stack_fn=None,
    *,
    fetch_one_fn=None,
    n_frames: int | None = None,
) -> np.ndarray | None:
    """Return a single image frame, caching frames *individually* on miss.

    Preferred (memory-bounded) usage: pass ``fetch_one_fn`` (``idx -> 2-D
    frame``) and ``n_frames``.  Only the requested frame is read and written
    into a pre-sized, per-frame-chunked HDF5 dataset, with a companion
    ``images_filled/<field>`` mask so repeat views read straight from disk —
    the full stack is never held in RAM.

    Legacy usage: pass ``fetch_stack_fn`` (zero-arg → full 3-D stack).  This
    materialises the entire stack in RAM (a large transient for long scans)
    and is kept only for callers that genuinely need the whole stack.
    """
    import h5py

    cache = ScanCache(uid)
    dset_name = f"images/{field}"
    fill_name = f"images_filled/{field}"

    # Fast path: read just the requested frame from disk.  When a fill mask is
    # present (lazy per-frame cache) the frame is only valid if marked filled;
    # a legacy full-stack write has no mask, so every frame is valid.
    if cache.path.exists():
        with cache._lock:
            try:
                with h5py.File(cache.path, "r") as f:
                    ds = f.get(dset_name)
                    if ds is not None and ds.ndim >= 3:
                        if frame_idx < ds.shape[0]:
                            fl = f.get(fill_name)
                            if fl is None or (frame_idx < fl.shape[0]
                                              and fl[frame_idx]):
                                return ds[frame_idx]
                    elif ds is not None:
                        return ds[...]
            except (OSError, IndexError, ValueError, KeyError):
                pass

    # --- Cache miss ---
    if fetch_one_fn is not None:
        if n_frames:
            return _cache_one_frame(cache, field, int(frame_idx),
                                    int(n_frames), fetch_one_fn)
        # Unknown length — serve a single uncached frame (still bounded RAM).
        try:
            fr = fetch_one_fn(frame_idx)
        except Exception:
            log.exception("cache: fetch_one_fn failed for %s/%s[%d]",
                          uid, field, frame_idx)
            return None
        return None if fr is None else np.asarray(fr)

    if fetch_stack_fn is None:
        return None

    # Legacy whole-stack path (materialises the full stack in RAM).
    from . import memlog
    memlog.report(f"img_stack:fetch_start {field}")
    try:
        stack = fetch_stack_fn()
    except Exception:
        log.exception("cache: fetch_stack_fn failed for %s/images/%s", uid, field)
        return None
    if stack is None:
        return None
    stack = np.asarray(stack)
    memlog.report(f"img_stack:fetched {field}", {"stack": stack})
    try:
        cache.write_image_stack(field, stack)
    except Exception:
        log.exception("cache: write_image_stack failed for %s/%s", uid, field)
    if stack.ndim >= 3:
        return stack[frame_idx]
    return stack


def _cache_one_frame(cache, field, frame_idx, n_frames, fetch_one_fn):
    """Fetch one frame and store it in a per-frame-chunked, partially-filled
    HDF5 dataset — never holding more than a single frame in RAM."""
    import h5py

    try:
        frame = fetch_one_fn(frame_idx)
    except Exception:
        log.exception("cache: fetch_one_fn failed for %s[%d]", field, frame_idx)
        return None
    if frame is None:
        return None
    frame = np.asarray(frame)
    dset_name = f"images/{field}"
    fill_name = f"images_filled/{field}"
    with cache._lock:
        try:
            with h5py.File(cache.path, "a") as f:
                ds = f.get(dset_name)
                if ds is None:
                    ds = f.create_dataset(
                        dset_name, shape=(n_frames,) + frame.shape,
                        dtype=frame.dtype, chunks=(1,) + frame.shape,
                        compression="gzip", compression_opts=2,
                    )
                fl = f.get(fill_name)
                if fl is None:
                    fl = f.create_dataset(fill_name, shape=(ds.shape[0],),
                                          dtype="u1")
                if 0 <= frame_idx < ds.shape[0] and frame.shape == ds.shape[1:]:
                    ds[frame_idx] = frame
                    fl[frame_idx] = 1
        except (OSError, ValueError, TypeError):
            log.exception("cache: per-frame write failed for %s[%d]",
                          field, frame_idx)
    _maybe_evict()
    return frame
