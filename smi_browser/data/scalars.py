"""Scalar stream → DataFrame conversion helpers.

The virtual-axis parser (``derive_virtual_columns`` / ``parse_label_number_tokens``)
now lives in :mod:`smi_tiled.derived.virtual_axes`; this module re-exports it so
existing browser imports keep working.  The browser still owns the
tiled-fetch glue (:func:`scalar_stream_to_frame`) and the dict→DataFrame
helper (:func:`scalars_to_dataframe`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _tiled as tb

# Re-export the parser and its config from smi-tiled.  Browser code should
# generally consume the ``fn:*`` data_vars that :func:`smi_tiled.derived.
# apply_virtual_axes` attaches to ``per_frame_iq`` during reduction; this
# direct API is kept for paths that need to parse a DataFrame in isolation
# (e.g. the scalars table built from the raw tiled primary stream before any
# reduction has run).
from smi_tiled.derived.virtual_axes import (  # noqa: F401
    VIRTUAL_PREFIX,
    derive_virtual_columns,
    parse_label_number_tokens,
)


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
