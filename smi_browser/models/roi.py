"""Memory-bounded rectangular ROI statistics in displayed pixel coordinates."""
from __future__ import annotations

from concurrent.futures import CancelledError

import numpy as np
import pandas as pd

STATISTICS = ("sum", "mean", "std", "min", "max", "count")


def roi_slices(rois: list[dict], shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    """Select pixel centres inside each rectangle, clipped to the image.

    Bokeh images span [0, width] × [0, height]; pixel (row, col) has centre
    (col + 0.5, row + 0.5). Upper rectangle edges are exclusive.
    """
    slices = []
    names = set()
    for roi in rois:
        name = roi["name"]
        if not name or name in names:
            raise ValueError("ROI names must be nonempty and unique.")
        names.add(name)
        x, y, w, h = (float(roi[k]) for k in ("x", "y", "width", "height"))
        if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
            raise ValueError(f"{name}: use finite coordinates and positive width/height.")
        x0, x1 = np.clip(np.ceil(np.array([x - w / 2, x + w / 2]) - .5), 0, shape[1]).astype(int)
        y0, y1 = np.clip(np.ceil(np.array([y - h / 2, y + h / 2]) - .5), 0, shape[0]).astype(int)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"{name}: rectangle contains no pixel centres in the image.")
        slices.append((slice(y0, y1), slice(x0, x1)))
    return slices


def compute_roi_scalars(rois, shape, n_frames, read_frame, *, cancel=None, progress=None):
    """Read each frame once for ALL ROIs; return columns and failed indices.

    ``read_frame`` supplies original-valued, display-oriented 2D arrays.
    Nonfinite pixels are excluded. Empty/all-invalid ROIs have NaN statistics
    and count 0; missing frames have NaN in every column, including count.
    Std is population standard deviation (ddof=0). No display/mask transforms
    are applied here. Memory is one image plus ROI scratch and scalar outputs.
    """
    slices = roi_slices(rois, shape)
    columns = {f"{roi['name']}:{stat}": np.full(n_frames, np.nan)
               for roi in rois for stat in STATISTICS}
    failed = []
    for i in range(n_frames):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        try:
            frame = read_frame(i)
            if frame is None or np.shape(frame) != tuple(shape):
                raise ValueError("Missing frame or inconsistent image shape")
            frame = np.asarray(frame)
        except CancelledError:
            raise
        except Exception:
            failed.append(i)
        else:
            for roi, region in zip(rois, slices):
                pixels = frame[region]
                values = pixels[np.isfinite(pixels)].astype(np.float64)
                prefix = roi["name"] + ":"
                columns[prefix + "count"][i] = values.size
                if values.size:
                    stats = (values.sum(), values.mean(), values.std(), values.min(), values.max())
                    for stat, value in zip(STATISTICS, stats):
                        columns[prefix + stat][i] = value
        if progress is not None:
            progress(i + 1, n_frames)
    if cancel is not None and cancel.is_set():
        raise CancelledError()
    return columns, failed


def merge_roi_scalars(df: pd.DataFrame, products: dict) -> pd.DataFrame:
    """Join by original zero-based frame index, never sorted table position."""
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("roi:")]).copy()
    if not any(p["columns"] for p in products.values()):
        return df
    n = max([len(df)] + [len(v) for p in products.values() for v in p["columns"].values()])
    df = df.reindex(pd.RangeIndex(n))
    if n and "frame" not in df:
        df["frame"] = np.arange(n)
    for field, product in products.items():
        for name, values in product["columns"].items():
            df[f"roi:{field}:{name}"] = pd.Series(values)
    return df
