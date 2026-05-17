"""Export processing results to the proposal working directory.

Writes PNG figures (2D maps, I(q), linecuts) and an HDF5 file
containing the full xarray dataset plus any cross-section cuts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import nsls2api

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def resolve_output_dir(
    data_session: str,
    project_name: str | None,
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
    base = nsls2api.fetch_proposal_directory(proposal_id)
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
    plt.close(fig)


def _save_iq_plot(
    result,
    title: str,
    path: Path,
) -> None:
    """Save merged I(q) as a log-log PNG (transmission only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iq = result.merged_iq
    q = iq["q"].values
    I = iq["I"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = np.isfinite(I) & (I > 0)
    if mask.any():
        ax.plot(q[mask], I[mask], "k-", lw=1.2, label="merged")
    if "saxs_I" in iq:
        sI = iq["saxs_I"].values
        sm = np.isfinite(sI) & (sI > 0)
        if sm.any():
            ax.plot(q[sm], sI[sm], "b-", lw=0.8, alpha=0.6, label="SAXS")
    if "waxs_I" in iq:
        wI = iq["waxs_I"].values
        wm = np.isfinite(wI) & (wI > 0)
        if wm.any():
            ax.plot(q[wm], wI[wm], "r-", lw=0.8, alpha=0.6, label="WAXS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q (nm⁻¹)")
    ax.set_ylabel("I(q)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        plt.close(fig)
        paths.append(p)

    if v_cuts:
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
        plt.close(fig)
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# HDF5 dataset export
# ---------------------------------------------------------------------------

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
) -> None:
    """Save the full processing result and cuts to an HDF5 file.

    Uses xarray's h5netcdf engine for the main datasets and h5py for
    additional groups (cuts, metadata).
    """
    import h5py
    import xarray as xr
    from .figures.cuts import compute_cross_section

    with h5py.File(path, "w") as f:
        # --- Transmission result ---
        if result is not None:
            grp = f.create_group("transmission")
            # merged I(q)
            iq = result.merged_iq
            iq_grp = grp.create_group("merged_iq")
            for var in iq.data_vars:
                iq_grp.create_dataset(var, data=iq[var].values)
            if "q" in iq.coords:
                iq_grp.create_dataset("q", data=iq["q"].values)

            # merged q-chi
            qchi = result.merged_qchi
            qchi_grp = grp.create_group("merged_qchi")
            qchi_grp.create_dataset(
                "intensity", data=qchi["intensity"].values,
            )
            if "q" in qchi.coords:
                qchi_grp.create_dataset("q", data=qchi["q"].values)
            if "chi" in qchi.coords:
                qchi_grp.create_dataset("chi", data=qchi["chi"].values)

            grp.attrs["geometry"] = result.geometry or ""
            grp.attrs["uid"] = result.uid or ""

        # --- GI result ---
        if gi_result is not None:
            grp = f.create_group("gi")
            grp.create_dataset("summed", data=gi_result.summed)
            grp.create_dataset("qxy_grid", data=gi_result.qxy_grid)
            grp.create_dataset("qz_grid", data=gi_result.qz_grid)
            if gi_result.frames:
                frames_grp = grp.create_group("frames")
                for i, frame in enumerate(gi_result.frames):
                    frames_grp.create_dataset(f"frame_{i:04d}", data=frame)
            if gi_result.alpha_i_deg:
                grp.create_dataset(
                    "alpha_i_deg",
                    data=np.array(gi_result.alpha_i_deg, dtype=np.float64),
                )
            grp.attrs["alpha_i_source"] = gi_result.alpha_i_source or ""

        # --- Cross-section cuts ---
        if cuts and x is not None and y is not None and image is not None:
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

        # --- Processing parameters ---
        params_grp = f.create_group("parameters")
        for k, v in (params or {}).items():
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
    raw_metadata: dict | None = None,
    formats: set[str] | None = None,
    subdir_template: str = "{uid_short}",
    basename_template: str = "",
    frame_label_col: str | list[str] | None = None,
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
    raw_metadata : dict, optional
    formats : set of format keys, optional
        Which outputs to produce.  Keys:
        ``"h5"``, ``"png_2d"``, ``"png_iq"``, ``"png_linecuts"``,
        ``"csv_iq"``, ``"csv_scalars"``, ``"csv_baseline"``,
        ``"metadata_txt"``, ``"png_grid"``
        If None, all available formats are exported.
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

    # --- CSV: merged I(q) ---
    if "csv_iq" in formats and result is not None and hasattr(result, "merged_iq"):
        iq = result.merged_iq
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
        if pf_iq is not None and "I" in pf_iq and "frame" in pf_iq.dims:
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
                frame_I = pf_iq["I"].isel(frame=fi).values
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
                # Include merged, saxs, and waxs I(q) columns
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
