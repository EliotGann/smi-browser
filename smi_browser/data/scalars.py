"""Scalar stream → DataFrame conversion helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _tiled as tb


def scalars_to_dataframe(scalar_data: dict) -> pd.DataFrame:
    """Convert a {field: ndarray} dict of scalars into a DataFrame."""
    if not scalar_data:
        return pd.DataFrame()

    columns = {}
    max_len = 0
    for key, arr in scalar_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = np.array([arr.item()])
        if arr.ndim != 1:
            continue
        columns[key] = arr
        max_len = max(max_len, len(arr))

    if not columns:
        return pd.DataFrame()

    data = {}
    for key, arr in columns.items():
        if len(arr) < max_len:
            padded = np.full(
                max_len, np.nan,
                dtype=float if np.issubdtype(arr.dtype, np.number) else object,
            )
            padded[:len(arr)] = arr
            data[key] = padded
        else:
            data[key] = arr
    return pd.DataFrame(data)


def scalar_stream_to_frame(run, stream: str) -> pd.DataFrame:
    """Read scalar fields from a stream into a DataFrame."""
    scalar_data = tb.fetch_scalars(run, stream)
    return scalars_to_dataframe(scalar_data)
