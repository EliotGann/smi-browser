"""Export processing results to the proposal working directory.

Writes PNG figures (2D maps, I(q), linecuts) and an HDF5 file
containing the full xarray dataset plus any cross-section cuts.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import nsls2api

log = logging.getLogger(__name__)

# Matplotlib rendering is not thread-safe; serialize all plot creation/saving.
_MPL_LOCK = threading.Lock()


def _json_default(o):
    """Best-effort JSON encoder for numpy scalars/arrays and unknown types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def resolve_output_dir(
    data_session: str,
    project_name: str | None,
    cycle: str | None = None,
    relative_path: str = "projects/{project_name}/analysis",
) -> Path | None:
    """Build the output directory for a given proposal + project.

    The directory is always rooted within the proposal's directory.
    ``relative_path`` is a template that may contain ``{project_name}``.

    Returns
    -------
    Path or None
        ``{proposal_dir}/{relative_path}/`` with placeholders resolved,
        or *None* if the proposal directory cannot be resolved.
    """
    proposal_id = nsls2api._proposal_id_from_data_session(data_session)
    base = nsls2api.fetch_proposal_directory_for_cycle(proposal_id, cycle)
    if not base:
        return None
    # Resolve template placeholders
    proj = project_name if (project_name and project_name != "(all)") else ""
    rendered = relative_path.format(project_name=proj)
    # Strip leading/trailing slashes and collapse empty segments
    parts = [p for p in rendered.split("/") if p]
    root = Path(base)
    for part in parts:
        root = root / part
    return root


# ---------------------------------------------------------------------------
# Matplotlib figure helpers (non-interactive, Agg backend)
# ---------------------------------------------------------------------------

def _save_2d_map(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    """Save a 2D intensity map as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig = None
    with _MPL_LOCK:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            finite = image[np.isfinite(image) & (image > 0)]
            if finite.size:
                vmin = max(float(np.percentile(finite, 2)), 1e-6)
                vmax = float(np.percentile(finite, 99.5))
                norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 2))
            else:
                norm = None
            extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
            ax.imshow(
                np.where(np.isfinite(image), image, 0),
                origin="lower",
                aspect="auto",
                extent=extent,
                norm=norm,
                cmap="turbo",
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            fig.colorbar(ax.images[0], ax=ax, label="Intensity")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
        finally:
            if fig is not None:
                plt.close(fig)


def _save_iq_plot(
    result,
    title: str,
    path: Path,
) -> None:
    """Save I(q) as a log-log PNG using marker style (no connecting lines)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = None
    with _MPL_LOCK:
        try:
            iq = result.merged_iq
            q = iq["q"].values
            I = iq["I"].values
            det_mode = _iq_detector_mode(iq)

            fig, ax = plt.subplots(figsize=(8, 5))
            series: list[tuple[str, np.ndarray, str]] = []

            mask = np.isfinite(I) & (I > 0)
            if det_mode not in {"saxs_only", "waxs_only"} and mask.any():
                series.append(("merged", I, "black"))
            if "saxs_I" in iq:
                sI = iq["saxs_I"].values
                sm = np.isfinite(sI) & (sI > 0)
                if sm.any():
                    series.append(("SAXS", sI, "blue"))
            if "waxs_I" in iq:
                wI = iq["waxs_I"].values
                wm = np.isfinite(wI) & (wI > 0)
                if wm.any():
                    series.append(("WAXS", wI, "red"))

            # Match UI default style: markers only (no lines).
            single_series = len(series) == 1
            for label, y, color in series:
                m = np.isfinite(y) & (y > 0)
                if not m.any():
                    continue
                ax.scatter(
                    q[m],
                    y[m],
                    s=10,
                    color=("black" if single_series else color),
                    alpha=0.9,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("q (nm⁻¹)")
            ax.set_ylabel("I(q)")
            ax.set_title(title)
            if series:
                ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=150)
        finally:
            if fig is not None:
                plt.close(fig)


def _iq_detector_mode(iq) -> str:
    """Classify I(q) availability as both/saxs_only/waxs_only/none."""
    has_saxs = False
    has_waxs = False
    if iq is None:
        return "none"
    if "saxs_I" in iq:
        s = iq["saxs_I"].values
        has_saxs = bool(np.isfinite(s).any() and (s > 0).any())
    if "waxs_I" in iq:
        w = iq["waxs_I"].values
        has_waxs = bool(np.isfinite(w).any() and (w > 0).any())
    if has_saxs and has_waxs:
        return "both"
    if has_saxs:
        return "saxs_only"
    if has_waxs:
        return "waxs_only"
    return "none"


def _save_linecuts(
    cuts: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    image: np.ndarray,
    x_label: str,
    y_label: str,
    out_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """Save linecut plots (one per cut kind: h → I vs q, v → I vs chi).

    Returns the list of paths written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures.cuts import compute_cross_section

    h_cuts = [c for c in cuts if c["kind"] == "h"]
    v_cuts = [c for c in cuts if c["kind"] == "v"]
    paths: list[Path] = []

    if h_cuts:
        fig = None
        with _MPL_LOCK:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                for i, cut in enumerate(h_cuts):
                    sec = compute_cross_section(cut, x, y, image, x_label, y_label)
                    if sec is None:
                        continue
                    axis, intensity, alabel = sec
                    label = f"h-cut {i}: {y_label}={cut['center']:.3g} ± {cut['width']/2:.3g}"
                    ax.plot(axis, intensity, lw=1.2, alpha=0.8, label=label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("Intensity")
                ax.set_title("Horizontal linecuts")
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = out_dir / f"{prefix}linecuts_h.png"
                fig.savefig(p, dpi=150)
                paths.append(p)
            finally:
                if fig is not None:
                    plt.close(fig)

    if v_cuts:
        fig = None
        with _MPL_LOCK:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                for i, cut in enumerate(v_cuts):
                    sec = compute_cross_section(cut, x, y, image, x_label, y_label)
                    if sec is None:
                        continue
                    axis, intensity, alabel = sec
                    label = f"v-cut {i}: {x_label}={cut['center']:.3g} ± {cut['width']/2:.3g}"
                    ax.plot(axis, intensity, lw=1.2, alpha=0.8, label=label)
                ax.set_xlabel(y_label)
                ax.set_ylabel("Intensity")
                ax.set_title("Vertical linecuts")
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = out_dir / f"{prefix}linecuts_v.png"
                fig.savefig(p, dpi=150)
                paths.append(p)
            finally:
                if fig is not None:
                    plt.close(fig)

    return paths


# ---------------------------------------------------------------------------
# HDF5 dataset export
# ---------------------------------------------------------------------------

def _write_qchi_frames_streamed(grp, frames_ds, progress_cb=None) -> None:
    """Write a per-frame 2-D stack to HDF5 frame-by-frame.

    Handles both transmission ``(frame, q, chi)`` (vars: ``intensity, counts``)
    and grazing-incidence ``(frame, qxy, qz)`` (var: ``intensity`` only)
    layouts.  Every data variable that has a ``frame`` dim is written in a
    single pass over the frame index, so lazy dask/zarr arrays only need to
    be materialised once per frame instead of once per variable.
    """
    n = int(frames_ds.sizes.get("frame", 0))
    grp.attrs["n_frames"] = n
    if n == 0:
        return

    var_names = [
        name for name, da in frames_ds.data_vars.items()
        if "frame" in da.dims
    ]
    if not var_names:
        return

    dsets: dict[str, Any] = {}
    for name in var_names:
        sample = np.asarray(frames_ds[name].isel(frame=0).values)
        dset = grp.create_dataset(
            name, shape=(n,) + sample.shape, dtype=sample.dtype,
            chunks=(1,) + sample.shape, compression="gzip", compression_opts=4,
        )
        dset[0] = sample
        dsets[name] = dset
    if progress_cb:
        progress_cb(1, n)

    for i in range(1, n):
        for name in var_names:
            dsets[name][i] = np.asarray(frames_ds[name].isel(frame=i).values)
        if progress_cb:
            progress_cb(i + 1, n)

    # Coordinate axes: transmission uses (q, chi); GI uses (qxy, qz).
    for coord in ("q", "chi", "qxy", "qz"):
        if coord in frames_ds.coords:
            grp.create_dataset(coord, data=frames_ds[coord].values)


def _write_raw_images_streamed(
    grp, image_source, progress_cb=None, batch_size: int = 16,
) -> None:
    """Stream raw detector images into HDF5 in batches from a lazy source.

    ``image_source`` is a dict mapping field names to callables that return
    ``(n_frames, frame_shape, dtype, batch_iterator)`` where each item yielded
    by ``batch_iterator`` is a numpy array of shape ``(<=batch_size, H, W)``.
    Writing in batches of ~16 frames is substantially faster than one-at-a-time
    because it amortises both network round-trips (tiled) and HDF5 compression
    overhead.
    """
    for field_name, source_fn in image_source.items():
        try:
            n_frames, frame_shape, dtype, batch_iter = source_fn()
        except Exception:
            log.debug("_write_raw_images_streamed: skipping %s", field_name)
            continue
        if n_frames == 0:
            continue
        dset = grp.create_dataset(
            field_name,
            shape=(n_frames,) + frame_shape,
            dtype=dtype,
            chunks=(1,) + frame_shape,
            compression="gzip",
            compression_opts=4,
        )
        written = 0
        for batch in batch_iter:
            batch = np.asarray(batch)
            if batch.ndim == len(frame_shape):
                # Single frame returned as (H, W) — wrap it
                batch = batch[np.newaxis, ...]
            end = min(written + batch.shape[0], n_frames)
            dset[written:end] = batch[:end - written]
            written = end
            if progress_cb:
                progress_cb(written, n_frames)


def build_raw_image_source(
    uid: str,
    image_fields: list[str],
    run=None,
    cache_path_fn=None,
    *,
    force_tiled: bool = False,
    batch_size: int = 16,
) -> dict:
    """Build a batched streaming image source for :func:`_write_raw_images_streamed`.

    For each field, the source first checks the disk cache (if available and
    *complete*), then falls back to streaming from tiled in batches.  Batching
    (~16 frames per read/write) amortises HTTP round-trips and HDF5 compression
    overhead, giving ~10–15× speedup over frame-by-frame on large scans.

    A cached image stack is considered complete only when *all* frames have been
    filled (i.e. no ``images_filled/<field>`` mask exists — a legacy full-write
    — or every entry in the mask is 1).  If the cache is partial (e.g. only one
    grid page was browsed), unfilled frames are fetched from tiled.

    Parameters
    ----------
    uid : str
        Scan UID.
    image_fields : list of str
        Detector field names to include.
    run : tiled run node, optional
        Used for batch fetching when cache misses.
    cache_path_fn : callable, optional
        ``cache_path_fn(uid)`` → Path to the scan's HDF5 cache file.
    force_tiled : bool
        When True, always stream from tiled regardless of cache state.
    batch_size : int
        Number of frames to read/write per batch (default 16).

    Returns
    -------
    dict
        Mapping field_name → callable returning
        ``(n_frames, frame_shape, dtype, batch_iterator)`` where each yielded
        item is a numpy array of shape ``(<=batch_size, H, W)``.
    """
    source = {}
    for field in image_fields:

        def _make_source(f=field, bs=batch_size):
            def _source_fn():
                import h5py

                # When not forcing tiled, check disk cache for a complete stack
                if not force_tiled and cache_path_fn is not None:
                    cp = cache_path_fn(uid)
                    if cp.exists():
                        with h5py.File(cp, "r") as cf:
                            if f"images/{f}" in cf:
                                dset = cf[f"images/{f}"]
                                shape = dset.shape
                                dtype = dset.dtype
                                n_frames = shape[0] if len(shape) >= 3 else 1
                                frame_shape = shape[1:] if len(shape) >= 3 else shape

                                # Check if cache is complete
                                fill_key = f"images_filled/{f}"
                                fill_mask = cf.get(fill_key)
                                if fill_mask is None or fill_mask[...].all():
                                    # Fully cached — read in batches from disk
                                    def _cache_batch_iter(path=cp,
                                                         key=f"images/{f}",
                                                         n=n_frames, b=bs):
                                        with h5py.File(path, "r") as fh:
                                            ds = fh[key]
                                            for start in range(0, n, b):
                                                end = min(start + b, n)
                                                yield ds[start:end]

                                    return (n_frames, frame_shape, dtype,
                                            _cache_batch_iter())

                                # Partial cache — hybrid: read filled from
                                # cache, fetch missing from tiled in batches
                                filled = fill_mask[...]
                                if run is not None:
                                    _filled_arr = filled

                                    def _hybrid_batch_iter(
                                        path=cp, key=f"images/{f}",
                                        n=n_frames, filled_flags=_filled_arr,
                                        fs=frame_shape, b=bs,
                                    ):
                                        with h5py.File(path, "r") as fh:
                                            ds = fh[key]
                                            for start in range(0, n, b):
                                                end = min(start + b, n)
                                                chunk = ds[start:end]
                                                # Fill missing frames from tiled
                                                for j in range(end - start):
                                                    idx = start + j
                                                    if not filled_flags[idx]:
                                                        frame = _fetch_frame_safe(
                                                            run, f, idx, fs)
                                                        chunk[j] = frame
                                                yield chunk

                                    return (n_frames, frame_shape, dtype,
                                            _hybrid_batch_iter())

                                # No tiled run — serve partial as-is
                                def _partial_batch_iter(path=cp,
                                                       key=f"images/{f}",
                                                       n=n_frames, b=bs):
                                    with h5py.File(path, "r") as fh:
                                        ds = fh[key]
                                        for start in range(0, n, b):
                                            end = min(start + b, n)
                                            yield ds[start:end]

                                return (n_frames, frame_shape, dtype,
                                        _partial_batch_iter())

                # Stream from tiled in batches
                if run is None:
                    raise RuntimeError(f"No source available for {f}")

                node = run["primary"][f]
                # Determine shape from the node metadata or first frame
                if hasattr(node, "shape"):
                    shape = tuple(node.shape)
                else:
                    first = np.asarray(
                        node[0].read() if hasattr(node[0], "read") else node[0])
                    while first.ndim > 2 and first.shape[0] == 1:
                        first = first[0]
                    shape = (1,) + first.shape

                n_frames = shape[0] if len(shape) >= 3 else 1
                frame_shape = shape[1:] if len(shape) >= 3 else shape
                # Squeeze leading length-1 dims to match batch normalization
                while len(frame_shape) > 2 and frame_shape[0] == 1:
                    frame_shape = frame_shape[1:]
                # Get dtype from node if possible
                dtype = getattr(node, "dtype", np.float32)

                def _tiled_batch_iter(nd=node, n=n_frames, fs=frame_shape,
                                      b=bs):
                    for start in range(0, n, b):
                        end = min(start + b, n)
                        try:
                            sliced = nd[start:end]
                            if hasattr(sliced, "read"):
                                batch = np.asarray(sliced.read())
                            else:
                                batch = np.asarray(sliced)
                            # Squeeze leading length-1 dims (e.g. (B,1,H,W))
                            while batch.ndim > 3 and batch.shape[1] == 1:
                                batch = batch[:, 0]
                            yield batch
                        except Exception:
                            # Fallback: fetch frames individually
                            frames = []
                            for i in range(start, end):
                                frame = _fetch_frame_safe(run, f, i, fs)
                                frames.append(frame)
                            yield np.stack(frames)

                return n_frames, frame_shape, dtype, _tiled_batch_iter()

            return _source_fn

        source[field] = _make_source()

    return source


def _fetch_frame_safe(run, field: str, idx: int, frame_shape) -> np.ndarray:
    """Fetch a single frame with fallback to zeros on failure."""
    from . import _tiled as tb
    frame = tb.fetch_frame(run, "primary", field, frame_idx=idx)
    if frame is not None:
        return frame
    return np.zeros(frame_shape, dtype=np.float32)


def _save_dataset_h5(
    result,
    gi_result,
    cuts: list[dict],
    x: np.ndarray | None,
    y: np.ndarray | None,
    image: np.ndarray | None,
    x_label: str,
    y_label: str,
    params: dict[str, Any],
    path: Path,
    primary_df=None,
    baseline_df=None,
    config_df=None,
    raw_metadata: dict | None = None,
    raw_images: dict[str, np.ndarray] | None = None,
    raw_image_source: dict | None = None,
    frame_labels: list[str] | None = None,
    sections: set[str] | None = None,
    peak_fits: list[dict] | None = None,
    progress_cb=None,
) -> None:
    """Save the full scan data and processing results to an HDF5 file.

    Handles three cases:
      1. Unprocessed: primary, baseline, metadata, raw images
      2. Processed transmission: all of the above + merged/per-frame I(q),
         merged/per-frame q-chi, line cuts with per-frame labeling, parameters
      3. Processed GI: all of the above + summed/per-frame qxy-qz,
         line cuts, parameters

    ``sections`` selects which groups to write (``None`` = all, for backward
    compatibility).  Recognised keys: ``"metadata"``, ``"primary"``,
    ``"baseline_config"``, ``"raw_images"``, ``"processed_iq"``,
    ``"processed_qchi"``, ``"peakfit"``.  ``peak_fits`` (used by the
    ``"peakfit"`` section) is a list of per-peak dicts with ``name``/``q_min``/
    ``q_max``/``model``/``baseline``/``link``/``bg_factor`` and a ``results``
    dict of per-frame arrays.

    ``raw_image_source`` is an alternative to ``raw_images`` for streaming
    large detector stacks without loading them fully into memory.  It maps
    field names to callables returning
    ``(n_frames, frame_shape, dtype, frame_iterator)``.  When provided,
    ``raw_images`` is ignored for fields present in ``raw_image_source``.
    """
    import h5py
    import pandas as pd
    from .figures.cuts import compute_cross_section

    _ALL_SECTIONS = {
        "metadata", "primary", "baseline_config", "raw_images",
        "processed_iq", "processed_qchi", "peakfit",
    }
    _sec = _ALL_SECTIONS if sections is None else set(sections)

    def on(key: str) -> bool:
        return key in _sec

    with h5py.File(path, "w") as f:
        # --- Raw detector images ---
        if on("raw_images"):
            if raw_image_source:
                img_grp = f.create_group("raw_images")
                _write_raw_images_streamed(img_grp, raw_image_source,
                                           progress_cb=progress_cb)
            elif raw_images:
                img_grp = f.create_group("raw_images")
                for field_name, stack in raw_images.items():
                    if stack is not None:
                        img_grp.create_dataset(
                            field_name, data=stack, compression="gzip",
                            compression_opts=4,
                        )

        # --- Primary stream scalars ---
        if primary_df is not None and not primary_df.empty and on("primary"):
            p_grp = f.create_group("primary")
            for col in primary_df.columns:
                arr = primary_df[col].values
                try:
                    if pd.api.types.is_numeric_dtype(primary_df[col]):
                        p_grp.create_dataset(col, data=arr.astype(np.float64))
                    else:
                        p_grp.create_dataset(
                            col, data=np.array(arr, dtype=h5py.string_dtype()),
                        )
                except (TypeError, ValueError):
                    p_grp.create_dataset(
                        col,
                        data=np.array([str(v) for v in arr], dtype=h5py.string_dtype()),
                    )

        # --- Baseline stream scalars ---
        if baseline_df is not None and not baseline_df.empty and on("baseline_config"):
            b_grp = f.create_group("baseline")
            for col in baseline_df.columns:
                arr = baseline_df[col].values
                try:
                    if pd.api.types.is_numeric_dtype(baseline_df[col]):
                        b_grp.create_dataset(col, data=arr.astype(np.float64))
                    else:
                        b_grp.create_dataset(
                            col, data=np.array(arr, dtype=h5py.string_dtype()),
                        )
                except (TypeError, ValueError):
                    b_grp.create_dataset(
                        col,
                        data=np.array([str(v) for v in arr], dtype=h5py.string_dtype()),
                    )

        # --- Primary stream configuration data ---
        if config_df is not None and not config_df.empty and on("baseline_config"):
            c_grp = f.create_group("config")
            for col in config_df.columns:
                arr = config_df[col].values
                try:
                    if pd.api.types.is_numeric_dtype(config_df[col]):
                        c_grp.create_dataset(col, data=arr.astype(np.float64))
                    else:
                        c_grp.create_dataset(
                            col, data=np.array(arr, dtype=h5py.string_dtype()),
                        )
                except (TypeError, ValueError):
                    c_grp.create_dataset(
                        col,
                        data=np.array([str(v) for v in arr], dtype=h5py.string_dtype()),
                    )

        # --- Raw metadata (start/stop documents) ---
        if raw_metadata and on("metadata"):
            md_grp = f.create_group("metadata")
            for section_key in ("start", "stop"):
                section = raw_metadata.get(section_key)
                if not section or not isinstance(section, dict):
                    continue
                sg = md_grp.create_group(section_key)
                for k, v in section.items():
                    try:
                        if v is None:
                            sg.attrs[k] = "None"
                        elif isinstance(v, (list, tuple)):
                            try:
                                sg.attrs[k] = list(v)
                            except TypeError:
                                sg.attrs[k] = str(v)
                        elif isinstance(v, dict):
                            sg.attrs[k] = str(v)
                        else:
                            sg.attrs[k] = v
                    except (TypeError, ValueError):
                        sg.attrs[k] = str(v)

        # ===== PROCESSING RESULTS (only if processed) =====

        # --- Transmission result ---
        if result is not None and (on("processed_iq") or on("processed_qchi")):
            grp = f.create_group("transmission")
            grp.attrs["geometry"] = result.geometry or ""
            grp.attrs["uid"] = result.uid or ""
            # Provenance: incident angle + scan_info (best-effort serialisation)
            inc_ang = getattr(result, "incident_angle_deg", None)
            if inc_ang is not None:
                try:
                    grp.attrs["incident_angle_deg"] = float(inc_ang)
                except (TypeError, ValueError):
                    pass
            scan_info = getattr(result, "scan_info", None)
            if scan_info:
                try:
                    import json
                    grp.attrs["scan_info"] = json.dumps(
                        scan_info, default=_json_default,
                    )
                except (TypeError, ValueError):
                    grp.attrs["scan_info"] = str(scan_info)

            if on("processed_iq"):
                # Merged I(q)
                iq = result.merged_iq
                if iq is not None:
                    iq_grp = grp.create_group("merged_iq")
                    for var in iq.data_vars:
                        iq_grp.create_dataset(var, data=iq[var].values)
                    if "q" in iq.coords:
                        iq_grp.create_dataset("q", data=iq["q"].values)

                # Per-detector native-grid I(q) (uncombined, native q axis)
                saxs = getattr(result, "saxs", None)
                waxs = getattr(result, "waxs", None)
                saxs_iq = saxs.get("iq") if saxs else None
                waxs_iq = waxs.get("iq") if waxs else None
                if saxs_iq is not None:
                    sg = grp.create_group("saxs_iq")
                    for var in saxs_iq.data_vars:
                        sg.create_dataset(var, data=saxs_iq[var].values)
                    if "q" in saxs_iq.coords:
                        sg.create_dataset("q", data=saxs_iq["q"].values)
                if waxs_iq is not None:
                    wg = grp.create_group("waxs_iq")
                    for var in waxs_iq.data_vars:
                        wg.create_dataset(var, data=waxs_iq[var].values)
                    if "q" in waxs_iq.coords:
                        wg.create_dataset("q", data=waxs_iq["q"].values)

                # Per-frame I(q)
                pf_iq = getattr(result, "per_frame_iq", None)
                if pf_iq is not None and "I" in pf_iq and "frame" in pf_iq.dims:
                    pf_iq_grp = grp.create_group("per_frame_iq")
                    pf_iq_grp.attrs["n_frames"] = pf_iq.sizes["frame"]
                    pf_iq_grp.create_dataset("q", data=pf_iq["q"].values)
                    pf_iq_grp.create_dataset("I", data=pf_iq["I"].values)
                    if "saxs_I" in pf_iq:
                        pf_iq_grp.create_dataset("saxs_I", data=pf_iq["saxs_I"].values)
                    if "waxs_I" in pf_iq:
                        pf_iq_grp.create_dataset("waxs_I", data=pf_iq["waxs_I"].values)
                    # Frame-level scalar labels
                    for var in pf_iq.data_vars:
                        if var in ("I", "saxs_I", "waxs_I"):
                            continue
                        arr = pf_iq[var]
                        if arr.dims == ("frame",):
                            pf_iq_grp.create_dataset(var, data=arr.values)
                    # Store frame label strings if provided
                    if frame_labels:
                        pf_iq_grp.create_dataset(
                            "frame_labels",
                            data=np.array(frame_labels, dtype=h5py.string_dtype()),
                        )

            if on("processed_qchi"):
                # Merged q-chi (all 6 data vars: intensity, counts,
                # saxs_intensity, saxs_counts, waxs_intensity, waxs_counts)
                qchi = result.merged_qchi
                if qchi is not None:
                    qchi_grp = grp.create_group("merged_qchi")
                    for var in qchi.data_vars:
                        qchi_grp.create_dataset(var, data=qchi[var].values)
                    if "q" in qchi.coords:
                        qchi_grp.create_dataset("q", data=qchi["q"].values)
                    if "chi" in qchi.coords:
                        qchi_grp.create_dataset("chi", data=qchi["chi"].values)

                # Per-frame q-chi (from saxs/waxs q_chi_frames)
                saxs = getattr(result, "saxs", None)
                waxs = getattr(result, "waxs", None)
                saxs_qchi_frames = saxs.get("q_chi_frames") if saxs else None
                waxs_qchi_frames = waxs.get("q_chi_frames") if waxs else None
                if saxs_qchi_frames is not None or waxs_qchi_frames is not None:
                    pf_qchi_grp = grp.create_group("per_frame_qchi")
                    # Use the merged grid coords
                    if qchi is not None and "q" in qchi.coords:
                        pf_qchi_grp.create_dataset("q", data=qchi["q"].values)
                    if qchi is not None and "chi" in qchi.coords:
                        pf_qchi_grp.create_dataset("chi", data=qchi["chi"].values)
                    if saxs_qchi_frames is not None:
                        _write_qchi_frames_streamed(
                            pf_qchi_grp.create_group("saxs"),
                            saxs_qchi_frames, progress_cb=progress_cb)
                    if waxs_qchi_frames is not None:
                        _write_qchi_frames_streamed(
                            pf_qchi_grp.create_group("waxs"),
                            waxs_qchi_frames, progress_cb=progress_cb)
                    if frame_labels:
                        pf_qchi_grp.create_dataset(
                            "frame_labels",
                            data=np.array(frame_labels, dtype=h5py.string_dtype()),
                        )

        # --- GI result ---
        if gi_result is not None and (on("processed_iq") or on("processed_qchi")):
            grp = f.create_group("gi")
            # Provenance: scanned motor + per-frame motor values
            scan_motor = getattr(gi_result, "scan_motor", None)
            if scan_motor:
                grp.attrs["scan_motor"] = str(scan_motor)
            scan_motor_values = getattr(gi_result, "scan_motor_values", None)
            if scan_motor_values is not None:
                arr = np.asarray(scan_motor_values)
                if arr.size > 0:
                    grp.create_dataset("scan_motor_values", data=arr)

            if on("processed_iq"):
                grp.create_dataset("summed", data=gi_result.summed)
                grp.create_dataset("qxy_grid", data=gi_result.qxy_grid)
                grp.create_dataset("qz_grid", data=gi_result.qz_grid)
                if gi_result.alpha_i_deg is not None:
                    grp.create_dataset(
                        "alpha_i_deg",
                        data=np.array(gi_result.alpha_i_deg, dtype=np.float64),
                    )
                grp.attrs["alpha_i_source"] = gi_result.alpha_i_source or ""
                if frame_labels:
                    grp.create_dataset(
                        "frame_labels",
                        data=np.array(frame_labels, dtype=h5py.string_dtype()),
                    )

            # Per-frame qxy-vs-qz stack — the GI equivalent of transmission's
            # per_frame_qchi.  Mirrors that layout: a single 3-D dataset with
            # qxy/qz coord arrays, gated under processed_qchi.
            if on("processed_qchi"):
                pf_qxy_qz = getattr(gi_result, "q_chi_frames", None)
                if pf_qxy_qz is not None and pf_qxy_qz.sizes.get("frame", 0) > 0:
                    pf_grp = grp.create_group("per_frame_qxy_qz")
                    _write_qchi_frames_streamed(
                        pf_grp, pf_qxy_qz, progress_cb=progress_cb,
                    )
                    if frame_labels:
                        pf_grp.create_dataset(
                            "frame_labels",
                            data=np.array(frame_labels, dtype=h5py.string_dtype()),
                        )

        # --- Peak-fit results ---
        if on("peakfit") and peak_fits:
            _write_peakfit_h5(f.create_group("peakfit"), peak_fits)

        # --- Cross-section cuts (from the current 2D display) ---
        if (on("processed_iq") and cuts and x is not None and y is not None
                and image is not None):
            cuts_grp = f.create_group("cuts")
            for i, cut in enumerate(cuts):
                sec = compute_cross_section(
                    cut, x, y, image, x_label, y_label,
                )
                if sec is None:
                    continue
                axis, intensity, alabel = sec
                cg = cuts_grp.create_group(f"cut_{i:03d}")
                cg.attrs["kind"] = cut["kind"]
                cg.attrs["center"] = cut["center"]
                cg.attrs["width"] = cut["width"]
                cg.attrs["axis_label"] = alabel
                cg.create_dataset("axis", data=axis)
                cg.create_dataset("intensity", data=intensity)

        # --- Processing parameters (only if processing was done) ---
        if params and (result is not None or gi_result is not None):
            params_grp = f.create_group("parameters")
            for k, v in params.items():
                try:
                    if isinstance(v, (list, tuple)):
                        params_grp.attrs[k] = list(v)
                    elif v is None:
                        params_grp.attrs[k] = "None"
                    else:
                        params_grp.attrs[k] = v
                except TypeError:
                    params_grp.attrs[k] = str(v)


# ---------------------------------------------------------------------------
# Peak-fit helpers (HDF5 group + PNG stack)
# ---------------------------------------------------------------------------

#: Per-frame fit parameters written for each peak.
_PEAK_FIT_PARAMS = ("amplitude", "center", "fwhm", "area")


def _peak_png_name(pk: dict, param: str | None, used: set[str]) -> str:
    """Filename ``peak_<slug>[_<param>]`` with collisions deduped.

    ``<slug>`` is :func:`smi_browser.models.peakfit.peak_slug` of the peak —
    i.e. ``<name>_q<center>`` where ``<center>`` is the drawn band's
    midpoint to 3 decimals.  The ``peak_`` prefix is kept so existing
    glob/regex tooling that walks scan dirs (and the matching HDF5 group
    keys under ``/peakfit/``) continues to work.
    """
    from smi_browser.models.peakfit import peak_slug as _slug

    base = f"peak_{_slug(pk)}"
    if param:
        base = f"{base}_{param}"
    stem = base
    i = 1
    while stem in used:
        stem = f"{base}_{i}"
        i += 1
    used.add(stem)
    return stem


def _write_peakfit_h5(root, peak_fits: list[dict]) -> None:
    """Write one subgroup per peak (datasets + identity attrs)."""
    used: set[str] = set()
    for pk in peak_fits:
        stem = _peak_png_name(pk, None, used)
        g = root.create_group(stem)
        for attr in ("name", "q_min", "q_max", "model", "baseline",
                     "link", "bg_factor"):
            if pk.get(attr) is not None:
                g.attrs[attr] = pk[attr]
        results = pk.get("results") or {}
        for key, arr in results.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.dtype == bool:
                arr = arr.astype("u1")
            g.create_dataset(key, data=arr)


def _save_peak_pngs(
    peak_fits: list[dict],
    axis: dict | None,
    param: str,
    scan_dir: Path,
    prefix: str,
) -> list[Path]:
    """One PNG per peak: ``param`` mapped across the scan.

    ``axis`` (``{x, x_label, y, y_label}``) defaults to a frame-index x-axis.
    A 2-D ``imshow`` is drawn when a usable ``y`` axis is supplied, otherwise a
    1-D line of ``param`` vs ``x``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if param not in _PEAK_FIT_PARAMS:
        param = "area"
    axis = axis or {}
    x = axis.get("x")
    x_label = axis.get("x_label") or "frame"
    y = axis.get("y")
    y_label = axis.get("y_label") or ""

    written: list[Path] = []
    used: set[str] = set()
    with _MPL_LOCK:
        for pk in peak_fits:
            results = pk.get("results") or {}
            z = results.get(param)
            if z is None:
                continue
            z = np.asarray(z, dtype=float)
            n = z.size
            xv = np.asarray(x, dtype=float) if x is not None else np.arange(n, dtype=float)
            if xv.size != n:
                xv = np.arange(n, dtype=float)
            stem = _peak_png_name(pk, param, used)
            fname = f"{prefix}{stem}.png" if prefix else f"{stem}.png"
            path = scan_dir / fname
            fig = None
            try:
                yv = np.asarray(y, dtype=float) if y is not None else None
                if yv is not None and yv.size == n:
                    fig, p = _peak_map_2d(xv, yv, z, x_label, y_label, param, pk)
                else:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    finite = np.isfinite(xv) & np.isfinite(z)
                    order = np.argsort(xv[finite])
                    ax.plot(xv[finite][order], z[finite][order], "-o", ms=3,
                            color="#1f77b4")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(f"{pk.get('name', 'peak')} {param}")
                    ax.set_title(
                        f"{pk.get('name', 'peak')}  "
                        f"q∈[{float(pk.get('q_min', 0)):.3f}, "
                        f"{float(pk.get('q_max', 0)):.3f}]  ·  {param}")
                    fig.tight_layout()
                fig.savefig(path, dpi=150)
                written.append(path)
            finally:
                if fig is not None:
                    plt.close(fig)
    return written


def _peak_map_2d(xv, yv, z, x_label, y_label, param, pk):
    """Build a 2-D scatter/grid map of ``z`` over (x, y) for a peak PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    finite = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(z)
    sc = ax.scatter(xv[finite], yv[finite], c=z[finite], cmap="viridis", s=18)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{pk.get('name', 'peak')}  "
        f"q∈[{float(pk.get('q_min', 0)):.3f}, "
        f"{float(pk.get('q_max', 0)):.3f}]  ·  {param}")
    fig.colorbar(sc, ax=ax, label=param)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Top-level export function
# ---------------------------------------------------------------------------

def export_results(
    *,
    data_session: str,
    project_name: str | None,
    uid: str,
    result=None,
    gi_result=None,
    cuts: list[dict] | None = None,
    proc_2d_cache: dict | None = None,
    params: dict[str, Any] | None = None,
) -> Path:
    """Export processing results to the proposal working directory.

    Parameters
    ----------
    data_session : str
        The data-session string (e.g. ``"pass-318826"``).
    project_name : str or None
        Project name within the proposal, or ``None``/``"(all)"``.
    uid : str
        The scan UID (used for the sub-directory name).
    result : transmission result object, optional
    gi_result : GI result object, optional
    cuts : list of cut dicts, optional
    proc_2d_cache : dict with keys x, y, image, x_label, y_label, optional
    params : processing parameters dict, optional

    Returns
    -------
    Path
        The output directory where files were written.

    Raises
    ------
    RuntimeError
        If the output directory cannot be resolved or created.
    """
    out_root = resolve_output_dir(data_session, project_name)
    if out_root is None:
        raise RuntimeError(
            f"Cannot resolve output directory for {data_session}"
        )

    uid_short = uid[:8]
    out_dir = out_root / uid_short
    out_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    # Extract 2D map data
    x = proc_2d_cache.get("x") if proc_2d_cache else None
    y = proc_2d_cache.get("y") if proc_2d_cache else None
    image = proc_2d_cache.get("image") if proc_2d_cache else None
    x_label = proc_2d_cache.get("x_label", "") if proc_2d_cache else ""
    y_label = proc_2d_cache.get("y_label", "") if proc_2d_cache else ""
    title_2d = proc_2d_cache.get("title", "") if proc_2d_cache else ""

    # --- 2D map PNG ---
    if image is not None and x is not None and y is not None:
        p = out_dir / "map_2d.png"
        _save_2d_map(image, x, y, x_label, y_label, title_2d, p)
        files_written.append("map_2d.png")

    # --- I(q) plot (transmission) ---
    if result is not None and hasattr(result, "merged_iq"):
        p = out_dir / "iq_merged.png"
        _save_iq_plot(result, f"{uid_short} — merged I(q)", p)
        files_written.append("iq_merged.png")

    # --- Linecuts ---
    if cuts and image is not None:
        lc_paths = _save_linecuts(
            cuts, x, y, image, x_label, y_label, out_dir,
        )
        files_written.extend(p.name for p in lc_paths)

    # --- HDF5 dataset ---
    h5_path = out_dir / "result.h5"
    _save_dataset_h5(
        result, gi_result, cuts or [], x, y, image,
        x_label, y_label, params or {}, h5_path,
    )
    files_written.append("result.h5")

    log.info("Exported %d files to %s: %s", len(files_written), out_dir,
             ", ".join(files_written))
    return out_dir


# ---------------------------------------------------------------------------
# Granular export helpers (called from the Export tab UI)
# ---------------------------------------------------------------------------

def export_scan(
    *,
    out_dir: Path,
    uid: str,
    result=None,
    gi_result=None,
    cuts: list[dict] | None = None,
    proc_2d_cache: dict | None = None,
    params: dict[str, Any] | None = None,
    primary_df=None,
    baseline_df=None,
    config_df=None,
    raw_metadata: dict | None = None,
    raw_images: dict[str, np.ndarray] | None = None,
    raw_image_source: dict | None = None,
    frame_labels: list[str] | None = None,
    formats: set[str] | None = None,
    subdir_template: str = "{uid_short}",
    basename_template: str = "",
    frame_label_col: str | list[str] | None = None,
    h5_sections: set[str] | None = None,
    peak_fits: list[dict] | None = None,
    peak_axis: dict | None = None,
    peak_param: str = "area",
    progress_cb=None,
) -> tuple[Path, list[str]]:
    """Export a single scan's data to *out_dir* with configurable outputs.

    Parameters
    ----------
    out_dir : Path
        Root export directory.  A sub-directory is created per scan using
        ``subdir_template`` (if non-empty).
    uid : str
        Scan UID.
    result : transmission result object, optional
    gi_result : GI result object, optional
    cuts : list of cut dicts, optional
    proc_2d_cache : dict with keys x, y, image, x_label, y_label, optional
    params : processing params, optional
    primary_df : DataFrame, optional
    baseline_df : DataFrame, optional
    config_df : DataFrame, optional
        Primary stream configuration data (detector settings, motor offsets).
    raw_metadata : dict, optional
    formats : set of format keys, optional
        Which outputs to produce.  Keys:
        ``"h5"``, ``"png_2d"``, ``"png_iq"``, ``"png_linecuts"``,
        ``"png_peaks"``, ``"csv_iq"``, ``"csv_scalars"``, ``"csv_baseline"``,
        ``"metadata_txt"``, ``"png_grid"``
        If None, all available formats are exported.
    h5_sections : set of section keys, optional
        Which groups the HDF5 writer includes (``None`` = all).  See
        :func:`_save_dataset_h5`.
    peak_fits : list of per-peak dicts, optional
        Peak-fit results for the ``"peakfit"`` HDF5 section and ``"png_peaks"``.
    peak_axis : dict, optional
        ``{x, x_label, y, y_label}`` driving the peak-result PNG maps
        (defaults to a frame-index x-axis).
    peak_param : str
        Which fit parameter the peak PNGs plot (default ``"area"``).
    subdir_template : str
        Template for the scan sub-directory name.  Leave empty to put files
        directly in ``out_dir`` (with ``basename_template`` as filename prefix).
        Available placeholders: ``{uid}``, ``{uid_short}``, ``{scan_id}``,
        ``{sample_name}``.
    basename_template : str
        Template for the filename prefix.  When non-empty, all output files
        are named ``{basename}_{original_name}``.  Same placeholders as
        ``subdir_template``.  When subdir is empty this is the primary way
        to distinguish scans.
    frame_label_col : str or list[str], optional
        Primary scalar column(s) appended to per-frame filenames.

    Returns
    -------
    (scan_dir, files_written) : tuple[Path, list[str]]
    """
    import pandas as pd

    if formats is None:
        formats = {
            "h5", "png_2d", "png_iq", "png_linecuts",
            "csv_iq", "csv_scalars", "csv_baseline", "metadata_txt",
        }

    # Resolve sub-directory name
    uid_short = uid[:8]
    scan_id = ""
    sample_name = ""
    if raw_metadata:
        start = raw_metadata.get("start", {})
        scan_id = str(start.get("scan_id", ""))
        sample_name = start.get(
            "sample_name", start.get("sample", start.get("Sample", ""))
        )
    # Resolve template placeholders in sample_name (e.g. {stage_y}, {target_name})
    if sample_name and "{" in sample_name:
        sample_name = resolve_name_template(
            sample_name, primary_df, baseline_df,
            frame_idx=None,
            extra_context={"uid": uid, "uid_short": uid_short, "scan_id": scan_id},
        )
    # Sanitize for filesystem
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

    if subdir_template:
        subdir_name = subdir_template.format(
            uid=uid,
            uid_short=uid_short,
            scan_id=_safe(scan_id),
            sample_name=_safe(sample_name),
        )
        scan_dir = out_dir / subdir_name
    else:
        scan_dir = out_dir
    scan_dir.mkdir(parents=True, exist_ok=True)

    # Build filename prefix from basename_template
    prefix = ""
    if basename_template:
        prefix = basename_template.format(
            uid=uid,
            uid_short=uid_short,
            scan_id=_safe(scan_id),
            sample_name=_safe(sample_name),
        )
        # Ensure prefix ends with separator
        if prefix and not prefix.endswith("_"):
            prefix += "_"

    def _fname(name: str) -> str:
        """Prepend the basename prefix to a filename."""
        return f"{prefix}{name}" if prefix else name

    files_written: list[str] = []

    # Extract 2D map data
    x = proc_2d_cache.get("x") if proc_2d_cache else None
    y = proc_2d_cache.get("y") if proc_2d_cache else None
    image = proc_2d_cache.get("image") if proc_2d_cache else None
    x_label = proc_2d_cache.get("x_label", "") if proc_2d_cache else ""
    y_label = proc_2d_cache.get("y_label", "") if proc_2d_cache else ""
    title_2d = proc_2d_cache.get("title", "") if proc_2d_cache else ""

    # --- HDF5 dataset ---
    if "h5" in formats:
        h5_path = scan_dir / _fname("result.h5")
        _save_dataset_h5(
            result, gi_result, cuts or [], x, y, image,
            x_label, y_label, params or {}, h5_path,
            primary_df=primary_df, baseline_df=baseline_df,
            config_df=config_df,
            raw_metadata=raw_metadata,
            raw_images=raw_images,
            raw_image_source=raw_image_source,
            frame_labels=frame_labels,
            sections=h5_sections,
            peak_fits=peak_fits,
            progress_cb=progress_cb,
        )
        files_written.append(_fname("result.h5"))

    # --- 2D map PNG ---
    if "png_2d" in formats and image is not None and x is not None and y is not None:
        p = scan_dir / _fname("map_2d.png")
        _save_2d_map(image, x, y, x_label, y_label, title_2d, p)
        files_written.append(_fname("map_2d.png"))

    # --- I(q) plot PNG (transmission) ---
    if "png_iq" in formats and result is not None and hasattr(result, "merged_iq"):
        p = scan_dir / _fname("iq_merged.png")
        _save_iq_plot(result, f"{uid_short} — merged I(q)", p)
        files_written.append(_fname("iq_merged.png"))

    # --- Linecut PNGs ---
    if "png_linecuts" in formats and cuts and image is not None:
        lc_paths = _save_linecuts(
            cuts, x, y, image, x_label, y_label, scan_dir, prefix=prefix,
        )
        files_written.extend(p.name for p in lc_paths)

    # --- Peak-result PNGs (one per peak, chosen parameter) ---
    if "png_peaks" in formats and peak_fits:
        peak_paths = _save_peak_pngs(
            peak_fits, peak_axis, peak_param, scan_dir, prefix,
        )
        files_written.extend(p.name for p in peak_paths)

    # --- CSV: merged I(q) ---
    if "csv_iq" in formats and result is not None and hasattr(result, "merged_iq"):
        iq = result.merged_iq
        det_mode = _iq_detector_mode(iq)
        if det_mode == "saxs_only" and "saxs_I" in iq:
            iq_df = pd.DataFrame({"q": iq["q"].values, "saxs_I": iq["saxs_I"].values})
        elif det_mode == "waxs_only" and "waxs_I" in iq:
            iq_df = pd.DataFrame({"q": iq["q"].values, "waxs_I": iq["waxs_I"].values})
        else:
            iq_df = pd.DataFrame({"q": iq["q"].values, "I": iq["I"].values})
            if "saxs_I" in iq:
                iq_df["saxs_I"] = iq["saxs_I"].values
            if "waxs_I" in iq:
                iq_df["waxs_I"] = iq["waxs_I"].values
        p = scan_dir / _fname("iq_merged.csv")
        iq_df.to_csv(p, index=False)
        files_written.append(_fname("iq_merged.csv"))

        # Per-frame I(q) if available
        pf_iq = getattr(result, "per_frame_iq", None)
        if (
            pf_iq is not None
            and "I" in pf_iq
            and "frame" in pf_iq.dims
            and pf_iq.sizes.get("frame", 0) > 1
        ):
            pf_dir = scan_dir / _fname("per_frame_iq")
            pf_dir.mkdir(exist_ok=True)
            q_vals = pf_iq["q"].values
            n_frames = pf_iq.sizes["frame"]

            # Normalize frame_label_col to a list
            label_cols: list[str] = []
            if isinstance(frame_label_col, str):
                label_cols = [frame_label_col]
            elif isinstance(frame_label_col, (list, tuple)):
                label_cols = list(frame_label_col)

            for fi in range(n_frames):
                if det_mode == "saxs_only" and "saxs_I" in pf_iq:
                    y_key = "saxs_I"
                elif det_mode == "waxs_only" and "waxs_I" in pf_iq:
                    y_key = "waxs_I"
                else:
                    y_key = "I"
                frame_I = pf_iq[y_key].isel(frame=fi).values
                # Build filename from label columns
                label_parts: list[str] = []
                for col in label_cols:
                    if col and col in pf_iq:
                        lbl = pf_iq[col].isel(frame=fi).values
                        label_parts.append(f"{_safe(col)}={_safe(str(lbl))}")
                if label_parts:
                    fname = f"iq_frame_{fi:04d}_{'_'.join(label_parts)}.csv"
                else:
                    fname = f"iq_frame_{fi:04d}.csv"
                if y_key == "saxs_I":
                    frame_df = pd.DataFrame({"q": q_vals, "saxs_I": frame_I})
                elif y_key == "waxs_I":
                    frame_df = pd.DataFrame({"q": q_vals, "waxs_I": frame_I})
                else:
                    frame_df = pd.DataFrame({"q": q_vals, "I": frame_I})
                    if "saxs_I" in pf_iq:
                        frame_df["saxs_I"] = pf_iq["saxs_I"].isel(frame=fi).values
                    if "waxs_I" in pf_iq:
                        frame_df["waxs_I"] = pf_iq["waxs_I"].isel(frame=fi).values
                frame_df.to_csv(pf_dir / fname, index=False)
            files_written.append(f"per_frame_iq/ ({n_frames} files)")

    # --- CSV: linecut data ---
    if "csv_iq" in formats and cuts and x is not None and y is not None and image is not None:
        from .figures.cuts import compute_cross_section
        cuts_dir = scan_dir / _fname("linecuts")
        cuts_dir.mkdir(exist_ok=True)
        for i, cut in enumerate(cuts):
            sec = compute_cross_section(cut, x, y, image, x_label, y_label)
            if sec is None:
                continue
            axis, intensity, alabel = sec
            cut_df = pd.DataFrame({alabel: axis, "intensity": intensity})
            cut_df.to_csv(cuts_dir / f"cut_{i:03d}_{cut['kind']}.csv", index=False)
        files_written.append(_fname("linecuts/"))

    # --- CSV: primary scalars ---
    if "csv_scalars" in formats and primary_df is not None and not primary_df.empty:
        p = scan_dir / _fname("primary_scalars.csv")
        primary_df.to_csv(p, index=False)
        files_written.append(_fname("primary_scalars.csv"))

    # --- CSV: baseline scalars ---
    if "csv_baseline" in formats and baseline_df is not None and not baseline_df.empty:
        p = scan_dir / _fname("baseline_scalars.csv")
        baseline_df.to_csv(p, index=False)
        files_written.append(_fname("baseline_scalars.csv"))

    # --- CSV: primary config ---
    if "csv_baseline" in formats and config_df is not None and not config_df.empty:
        p = scan_dir / _fname("config_scalars.csv")
        config_df.to_csv(p, index=False)
        files_written.append(_fname("config_scalars.csv"))

    # --- Metadata text ---
    if "metadata_txt" in formats and raw_metadata:
        import json
        p = scan_dir / _fname("metadata.json")
        # Make JSON-serializable
        def _default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        p.write_text(json.dumps(raw_metadata, indent=2, default=_default))
        files_written.append(_fname("metadata.json"))

    log.info("export_scan: wrote %d items to %s", len(files_written), scan_dir)
    return scan_dir, files_written


# ---------------------------------------------------------------------------
# Name template resolution — resolve {field} placeholders from stream data
# ---------------------------------------------------------------------------

def resolve_name_template(
    template: str,
    primary_df: "pd.DataFrame | None" = None,
    baseline_df: "pd.DataFrame | None" = None,
    frame_idx: int | None = None,
    extra_context: dict | None = None,
    *,
    max_depth: int = 3,
    fmt: str = ".6g",
) -> str:
    """Resolve ``{field}`` placeholders in a sample name template.

    Placeholders like ``{stage_y}`` or ``{pin_diode_current2_mean_value}``
    are looked up from the primary stream DataFrame (per-frame) or baseline.
    Resolution is recursive: if a resolved value is itself a string containing
    ``{...}`` placeholders, those are resolved too (up to *max_depth* levels).

    Parameters
    ----------
    template : str
        The template string (e.g. ``"y{stage_y}_x{stage_x}_pd{pd_val}"``).
    primary_df : DataFrame, optional
        Primary stream data. Columns = fields, rows = frames.
    baseline_df : DataFrame, optional
        Baseline stream data.  Falls back here if field not in primary.
    frame_idx : int, optional
        Which frame row to use from primary_df.  If None, uses the median
        for numeric columns or the first value for string columns (single
        representative value for the whole scan).
    extra_context : dict, optional
        Additional key-value pairs to substitute (e.g. uid, scan_id).
    max_depth : int
        Maximum recursion depth for nested templates.
    fmt : str
        Format spec for numeric values (default: ".6g").

    Returns
    -------
    str
        The resolved string with placeholders filled in.
    """
    import re
    import pandas as pd

    if not template or "{" not in template:
        return template

    context = dict(extra_context) if extra_context else {}

    def _lookup(field: str) -> str | None:
        """Look up a field value, returning a string or None."""
        # Check explicit extra_context first
        if field in context:
            return str(context[field])

        # Primary stream
        if primary_df is not None and field in primary_df.columns:
            col = primary_df[field]
            if frame_idx is not None and frame_idx < len(col):
                val = col.iloc[frame_idx]
            else:
                # Representative value
                if pd.api.types.is_numeric_dtype(col):
                    val = col.dropna().median() if len(col.dropna()) else None
                else:
                    vals = col.dropna()
                    val = vals.iloc[0] if len(vals) else None
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            if isinstance(val, (float, np.floating)):
                return format(val, fmt)
            return str(val)

        # Baseline stream
        if baseline_df is not None and field in baseline_df.columns:
            col = baseline_df[field]
            vals = col.dropna()
            if len(vals) == 0:
                return None
            val = vals.iloc[0]
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            if isinstance(val, (float, np.floating)):
                return format(val, fmt)
            return str(val)

        return None

    def _resolve(text: str, depth: int) -> str:
        if depth <= 0 or "{" not in text:
            return text
        # Match {field_name} but NOT {{escaped}}
        def _replace(m):
            field = m.group(1)
            val = _lookup(field)
            if val is None:
                return m.group(0)  # Leave unresolved
            # Recurse: the resolved value might itself be a template
            return _resolve(val, depth - 1)

        return re.sub(r"\{([^{}]+)\}", _replace, text)

    return _resolve(template, max_depth)


def resolve_name_template_all_frames(
    template: str,
    primary_df: "pd.DataFrame | None" = None,
    baseline_df: "pd.DataFrame | None" = None,
    extra_context: dict | None = None,
    *,
    max_depth: int = 3,
    fmt: str = ".6g",
) -> list[str]:
    """Resolve a name template for every frame in primary_df.

    Returns a list of resolved strings, one per frame (row in primary_df).
    If primary_df is None or empty, returns a single-element list with the
    scan-level resolution.
    """
    if primary_df is None or primary_df.empty:
        return [resolve_name_template(
            template, primary_df, baseline_df,
            frame_idx=None, extra_context=extra_context,
            max_depth=max_depth, fmt=fmt,
        )]

    n_frames = len(primary_df)
    return [
        resolve_name_template(
            template, primary_df, baseline_df,
            frame_idx=i, extra_context=extra_context,
            max_depth=max_depth, fmt=fmt,
        )
        for i in range(n_frames)
    ]


# ---------------------------------------------------------------------------
# Name parsing utility — find common/distinct parts across sample names
# ---------------------------------------------------------------------------

def parse_name_parts(names: list[str]) -> dict:
    """Analyse a list of sample names to find common and distinct parts.

    Splits each name on common separators (_, -, whitespace) and identifies
    which tokens are shared across all names vs. which vary.

    Returns
    -------
    dict with keys:
        common_prefix : str — longest common prefix string
        common_suffix : str — longest common suffix string
        common_tokens : list[str] — tokens shared by all names
        distinct_tokens : list[list[str]] — per-name list of tokens that differ
        distinct_strings : list[str] — per-name joined distinct parts
        suggested_basename : str — common tokens joined with '_'
        suggested_labels : list[str] — distinct parts as compact labels
    """
    import re

    if not names:
        return {
            "common_prefix": "",
            "common_suffix": "",
            "common_tokens": [],
            "distinct_tokens": [],
            "distinct_strings": [],
            "suggested_basename": "",
            "suggested_labels": [],
        }

    if len(names) == 1:
        return {
            "common_prefix": names[0],
            "common_suffix": "",
            "common_tokens": [names[0]],
            "distinct_tokens": [[]],
            "distinct_strings": [""],
            "suggested_basename": names[0],
            "suggested_labels": [""],
        }

    # Common prefix/suffix (character-level)
    def _common_prefix(strs):
        if not strs:
            return ""
        s0 = strs[0]
        for i, ch in enumerate(s0):
            if any(i >= len(s) or s[i] != ch for s in strs[1:]):
                return s0[:i]
        return s0

    prefix = _common_prefix(names)
    suffix = _common_prefix([n[::-1] for n in names])[::-1]

    # Tokenize by separators (keep separators for reconstruction)
    def _tokenize(name):
        return re.split(r'([_\-\s]+)', name)

    token_lists = [_tokenize(n) for n in names]

    # Find common token positions (tokens that are identical across all names)
    # Use the shortest token list as reference
    min_len = min(len(tl) for tl in token_lists)
    common_positions: set[int] = set()
    for i in range(min_len):
        vals = {tl[i] for tl in token_lists}
        if len(vals) == 1:
            common_positions.add(i)

    common_tokens = []
    if token_lists:
        for i in sorted(common_positions):
            tok = token_lists[0][i]
            if tok.strip():  # skip separator-only tokens
                common_tokens.append(tok)

    distinct_tokens = []
    distinct_strings = []
    for tl in token_lists:
        dparts = []
        for i, tok in enumerate(tl):
            if i not in common_positions and tok.strip():
                dparts.append(tok)
        distinct_tokens.append(dparts)
        distinct_strings.append("_".join(dparts) if dparts else "")

    suggested_basename = "_".join(common_tokens) if common_tokens else ""
    suggested_labels = distinct_strings

    return {
        "common_prefix": prefix,
        "common_suffix": suffix,
        "common_tokens": common_tokens,
        "distinct_tokens": distinct_tokens,
        "distinct_strings": distinct_strings,
        "suggested_basename": suggested_basename,
        "suggested_labels": suggested_labels,
    }


# ---------------------------------------------------------------------------
# Collection export — combine multiple scans into a single output
# ---------------------------------------------------------------------------

def export_collection(
    *,
    out_dir: Path,
    results: list[tuple[str, Any]],  # list of (uid, CombinedReductionResult)
    scan_labels: list[str],           # one label per scan (user-chosen)
    basename: str = "collection",
    formats: set[str] | None = None,
    params_list: list[dict] | None = None,
    metadata_list: list[dict] | None = None,
    primary_dfs: list | None = None,
    baseline_dfs: list | None = None,
) -> tuple[Path, list[str]]:
    """Export a collection of scans as combined single-file outputs.

    Parameters
    ----------
    out_dir : Path
        Output directory (already resolved).
    results : list of (uid, result) tuples
        Each result is a CombinedReductionResult with .merged_iq, .merged_qchi.
    scan_labels : list[str]
        One human-readable label per scan (e.g. "T=25°C" or "wa1").
    basename : str
        Base filename for all outputs.
    formats : set[str], optional
        Which formats to produce. Keys: "h5", "csv_iq", "png_iq", "png_linecuts".
    params_list : list[dict], optional
        Processing parameters per scan (saved in HDF5 metadata).
    metadata_list : list[dict], optional
        Raw metadata per scan (saved in HDF5).

    Returns
    -------
    (out_dir, files_written) : tuple[Path, list[str]]
    """
    import pandas as pd
    import h5py

    if formats is None:
        formats = {"h5", "csv_iq", "png_iq"}

    out_dir.mkdir(parents=True, exist_ok=True)
    files_written: list[str] = []

    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

    safe_basename = _safe(basename) if basename else "collection"
    n_scans = len(results)

    # Collect I(q) data from all scans
    iq_data: list[dict] = []  # {label, q, I, saxs_I?, waxs_I?}
    for i, (uid, result) in enumerate(results):
        if result is None or not hasattr(result, "merged_iq"):
            continue
        iq = result.merged_iq
        entry: dict[str, Any] = {
            "label": scan_labels[i] if i < len(scan_labels) else uid[:8],
            "uid": uid,
            "q": iq["q"].values,
            "I": iq["I"].values,
        }
        if "saxs_I" in iq:
            entry["saxs_I"] = iq["saxs_I"].values
        if "waxs_I" in iq:
            entry["waxs_I"] = iq["waxs_I"].values
        iq_data.append(entry)

    # --- HDF5: combined dataset ---
    if "h5" in formats and iq_data:
        h5_path = out_dir / f"{safe_basename}.h5"
        with h5py.File(h5_path, "w") as f:
            # Store shared q axis (from first scan — assumed common)
            q0 = iq_data[0]["q"]
            f.create_dataset("q", data=q0)
            f.attrs["n_scans"] = n_scans
            f.attrs["basename"] = basename

            # Store scan labels as dataset for easy reading
            label_strings = [
                scan_labels[i] if i < len(scan_labels) else f"frame_{i}"
                for i in range(n_scans)
            ]
            f.create_dataset(
                "scan_labels",
                data=np.array(label_strings, dtype=h5py.string_dtype()),
            )

            # --- Per-scan I(q) (groups + stacked matrix) ---
            iq_grp = f.create_group("iq")
            for entry in iq_data:
                label = _safe(entry["label"])
                sg = iq_grp.create_group(label)
                sg.create_dataset("I", data=entry["I"])
                if "saxs_I" in entry:
                    sg.create_dataset("saxs_I", data=entry["saxs_I"])
                if "waxs_I" in entry:
                    sg.create_dataset("waxs_I", data=entry["waxs_I"])
                sg.attrs["uid"] = entry["uid"]
                sg.attrs["label"] = entry["label"]

            # Stacked I(q) matrix (n_scans x n_q) for easy array access
            I_matrix = np.array([e["I"] for e in iq_data])
            f.create_dataset("I_matrix", data=I_matrix)
            # Also stack saxs/waxs if available
            if all("saxs_I" in e for e in iq_data):
                f.create_dataset(
                    "saxs_I_matrix",
                    data=np.array([e["saxs_I"] for e in iq_data]),
                )
            if all("waxs_I" in e for e in iq_data):
                f.create_dataset(
                    "waxs_I_matrix",
                    data=np.array([e["waxs_I"] for e in iq_data]),
                )

            # --- Per-scan 2D q-chi maps (groups + stacked 3D array) ---
            qchi_grp = f.create_group("qchi")
            qchi_arrays = []
            chi_coords = None
            q_coords = None
            for i, (uid, result) in enumerate(results):
                if result is None or not hasattr(result, "merged_qchi"):
                    continue
                qchi = result.merged_qchi
                label = _safe(scan_labels[i] if i < len(scan_labels) else uid[:8])
                intensity = qchi["intensity"].values
                # Handle per-frame qchi within a single scan: take frame 0 or mean
                if intensity.ndim == 3:
                    # (frame, chi, q) -> average across frames for this scan
                    intensity = np.nanmean(intensity, axis=0)
                sg = qchi_grp.create_group(label)
                sg.create_dataset("intensity", data=intensity)
                sg.attrs["uid"] = uid
                sg.attrs["label"] = scan_labels[i] if i < len(scan_labels) else uid[:8]
                qchi_arrays.append(intensity)
                # Store coordinates from first valid scan
                if chi_coords is None and "chi" in qchi.coords:
                    chi_coords = qchi["chi"].values
                if q_coords is None and "q" in qchi.coords:
                    q_coords = qchi["q"].values

            # Store q-chi coordinate axes
            if chi_coords is not None:
                qchi_grp.create_dataset("chi", data=chi_coords)
            if q_coords is not None:
                qchi_grp.create_dataset("q", data=q_coords)

            # Stacked 3D array (n_scans x n_chi x n_q) if shapes are compatible
            if qchi_arrays and all(
                a.shape == qchi_arrays[0].shape for a in qchi_arrays
            ):
                f.create_dataset(
                    "qchi_matrix", data=np.array(qchi_arrays),
                )

            # --- Per-scan per-frame I(q) (if scans have multi-frame data) ---
            # Store individual frame I(q) curves within each scan, labeled
            pf_grp = f.create_group("per_frame_iq")
            for i, (uid, result) in enumerate(results):
                if result is None:
                    continue
                pf_iq = getattr(result, "per_frame_iq", None)
                if pf_iq is None or "I" not in pf_iq or "frame" not in pf_iq.dims:
                    continue
                scan_label = _safe(
                    scan_labels[i] if i < len(scan_labels) else uid[:8]
                )
                sg = pf_grp.create_group(scan_label)
                sg.attrs["uid"] = uid
                sg.attrs["n_frames"] = pf_iq.sizes["frame"]
                sg.create_dataset("q", data=pf_iq["q"].values)
                sg.create_dataset("I", data=pf_iq["I"].values)  # (n_frames, n_q)
                if "saxs_I" in pf_iq:
                    sg.create_dataset("saxs_I", data=pf_iq["saxs_I"].values)
                if "waxs_I" in pf_iq:
                    sg.create_dataset("waxs_I", data=pf_iq["waxs_I"].values)
                # Store frame-level scalar labels from per_frame_iq
                for var in pf_iq.data_vars:
                    if var in ("I", "saxs_I", "waxs_I"):
                        continue
                    arr = pf_iq[var]
                    if arr.dims == ("frame",):
                        sg.create_dataset(var, data=arr.values)

            # --- Per-scan per-frame 2D q-chi (if multi-frame) ---
            pf_qchi_grp = f.create_group("per_frame_qchi")
            for i, (uid, result) in enumerate(results):
                if result is None or not hasattr(result, "merged_qchi"):
                    continue
                qchi = result.merged_qchi
                if "frame" not in qchi.dims:
                    continue
                scan_label = _safe(
                    scan_labels[i] if i < len(scan_labels) else uid[:8]
                )
                intensity = qchi["intensity"].values  # (n_frames, n_chi, n_q)
                sg = pf_qchi_grp.create_group(scan_label)
                sg.attrs["uid"] = uid
                sg.attrs["n_frames"] = qchi.sizes["frame"]
                sg.create_dataset("intensity", data=intensity)
                if "q" in qchi.coords:
                    sg.create_dataset("q", data=qchi["q"].values)
                if "chi" in qchi.coords:
                    sg.create_dataset("chi", data=qchi["chi"].values)

            # Parameters
            if params_list:
                p_grp = f.create_group("parameters")
                for i, params in enumerate(params_list):
                    if params:
                        label = _safe(
                            scan_labels[i] if i < len(scan_labels) else f"scan_{i}"
                        )
                        sg = p_grp.create_group(label)
                        for k, v in params.items():
                            try:
                                if v is None:
                                    sg.attrs[k] = "None"
                                elif isinstance(v, (list, tuple)):
                                    sg.attrs[k] = list(v)
                                else:
                                    sg.attrs[k] = v
                            except TypeError:
                                sg.attrs[k] = str(v)

            # Metadata
            if metadata_list:
                m_grp = f.create_group("metadata")
                for i, md in enumerate(metadata_list):
                    if md:
                        label = _safe(
                            scan_labels[i] if i < len(scan_labels) else f"scan_{i}"
                        )
                        sg = m_grp.create_group(label)
                        start = md.get("start", {})
                        for k, v in start.items():
                            try:
                                sg.attrs[k] = v if v is not None else "None"
                            except TypeError:
                                sg.attrs[k] = str(v)

            # Primary stream scalars (per-scan)
            if primary_dfs:
                pri_grp = f.create_group("primary")
                for i, pdf in enumerate(primary_dfs):
                    if pdf is None or pdf.empty:
                        continue
                    label = _safe(
                        scan_labels[i] if i < len(scan_labels) else f"scan_{i}"
                    )
                    sg = pri_grp.create_group(label)
                    for col in pdf.columns:
                        arr = pdf[col].values
                        try:
                            if pd.api.types.is_numeric_dtype(pdf[col]):
                                sg.create_dataset(col, data=arr.astype(np.float64))
                            else:
                                sg.create_dataset(
                                    col,
                                    data=np.array(arr, dtype=h5py.string_dtype()),
                                )
                        except (TypeError, ValueError):
                            sg.create_dataset(
                                col,
                                data=np.array(
                                    [str(v) for v in arr], dtype=h5py.string_dtype()
                                ),
                            )

            # Baseline stream scalars (per-scan)
            if baseline_dfs:
                bas_grp = f.create_group("baseline")
                for i, bdf in enumerate(baseline_dfs):
                    if bdf is None or bdf.empty:
                        continue
                    label = _safe(
                        scan_labels[i] if i < len(scan_labels) else f"scan_{i}"
                    )
                    sg = bas_grp.create_group(label)
                    for col in bdf.columns:
                        arr = bdf[col].values
                        try:
                            if pd.api.types.is_numeric_dtype(bdf[col]):
                                sg.create_dataset(col, data=arr.astype(np.float64))
                            else:
                                sg.create_dataset(
                                    col,
                                    data=np.array(arr, dtype=h5py.string_dtype()),
                                )
                        except (TypeError, ValueError):
                            sg.create_dataset(
                                col,
                                data=np.array(
                                    [str(v) for v in arr], dtype=h5py.string_dtype()
                                ),
                            )

        files_written.append(f"{safe_basename}.h5")

    # --- CSV: multi-scan I(q) (q as rows, each scan as a column) ---
    if "csv_iq" in formats and iq_data:
        # Use first scan's q as reference
        q_ref = iq_data[0]["q"]
        df_dict: dict[str, Any] = {"q": q_ref}
        for entry in iq_data:
            label = entry["label"]
            # Interpolate to common q if needed
            if len(entry["q"]) == len(q_ref) and np.allclose(
                entry["q"], q_ref, rtol=1e-6
            ):
                df_dict[f"I_{label}"] = entry["I"]
            else:
                df_dict[f"I_{label}"] = np.interp(
                    q_ref, entry["q"], entry["I"],
                    left=np.nan, right=np.nan,
                )
            if "saxs_I" in entry:
                s_key = f"saxs_I_{label}"
                if len(entry["q"]) == len(q_ref) and np.allclose(
                    entry["q"], q_ref, rtol=1e-6
                ):
                    df_dict[s_key] = entry["saxs_I"]
                else:
                    df_dict[s_key] = np.interp(
                        q_ref, entry["q"], entry["saxs_I"],
                        left=np.nan, right=np.nan,
                    )
            if "waxs_I" in entry:
                w_key = f"waxs_I_{label}"
                if len(entry["q"]) == len(q_ref) and np.allclose(
                    entry["q"], q_ref, rtol=1e-6
                ):
                    df_dict[w_key] = entry["waxs_I"]
                else:
                    df_dict[w_key] = np.interp(
                        q_ref, entry["q"], entry["waxs_I"],
                        left=np.nan, right=np.nan,
                    )
        csv_path = out_dir / f"{safe_basename}_iq.csv"
        pd.DataFrame(df_dict).to_csv(csv_path, index=False)
        files_written.append(f"{safe_basename}_iq.csv")

    # --- PNG: multi-scan I(q) overlay ---
    if "png_iq" in formats and iq_data:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = None
        with _MPL_LOCK:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                for entry in iq_data:
                    q, I = entry["q"], entry["I"]
                    mask = np.isfinite(I) & (I > 0)
                    if mask.any():
                        ax.plot(q[mask], I[mask], lw=1.2, alpha=0.8, label=entry["label"])
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("q (nm⁻¹)")
                ax.set_ylabel("I(q)")
                ax.set_title(f"{basename} — I(q) comparison")
                ax.legend(fontsize=8, loc="best")
                fig.tight_layout()
                png_path = out_dir / f"{safe_basename}_iq.png"
                fig.savefig(png_path, dpi=150)
                files_written.append(f"{safe_basename}_iq.png")
            finally:
                if fig is not None:
                    plt.close(fig)

    log.info("export_collection: wrote %d items to %s", len(files_written), out_dir)
    return out_dir, files_written
