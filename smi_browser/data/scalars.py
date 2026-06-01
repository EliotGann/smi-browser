"""Scalar stream → DataFrame conversion helpers."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import _tiled as tb

#: Matches a ``label`` + ``number`` (+ ``unit``) token, where at least one of
#: the alphabetic groups touches the number, e.g. ``ai0.50`` (prefix ``ai``),
#: ``2450.00eV`` (unit ``eV``), ``degC100.0`` (prefix ``degC``).  Bare numbers
#: with no adjacent letters (``120``) do not yield a label and are ignored.
_TOKEN_RE = re.compile(
    r"([A-Za-z][A-Za-z%°µ/]*)?(-?\d+(?:\.\d+)?)([A-Za-z%°µ/]+)?"
)

#: Default prefix marking columns derived from a string field.
VIRTUAL_PREFIX = "fn:"


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


# ---------------------------------------------------------------------------
# Virtual columns parsed from structured per-frame string fields
# ---------------------------------------------------------------------------

def parse_label_number_tokens(s: str) -> dict[str, float]:
    """Extract ``{label: number}`` pairs from a structured string.

    Tokens are ``label`` + ``number`` (+ ``unit``) runs; the column label is
    the alphabetic *prefix* when present, else the trailing *unit*.  A bare
    number with no adjacent letters yields nothing.

    Examples
    --------
    ``"Lucas_sample2_pos1_2450.00eV_ai0.50_wa9_bpm1.995_degC100.0"`` →
    ``{"sample": 2, "pos": 1, "eV": 2450.0, "ai": 0.5, "wa": 9,
       "bpm": 1.995, "degC": 100.0}``.
    """
    out: dict[str, float] = {}
    # Strings cached in HDF5 come back as ``bytes`` (h5py variable-length
    # strings); tiled returns ``str``.  Accept both, plus numpy string scalars.
    if isinstance(s, (bytes, bytearray, np.bytes_)):
        try:
            s = bytes(s).decode("utf-8", "replace")
        except Exception:
            return out
    elif not isinstance(s, str):
        if isinstance(s, np.str_):
            s = str(s)
        else:
            return out
    if not s:
        return out
    for prefix, number, unit in _TOKEN_RE.findall(s):
        label = prefix or unit
        if not label:
            continue  # bare number — no usable axis label
        try:
            value = float(number)
        except ValueError:
            continue
        # First occurrence of a label within the string wins.
        out.setdefault(label, value)
    return out


def derive_virtual_columns(
    df: pd.DataFrame,
    *,
    prefix: str = VIRTUAL_PREFIX,
    sources: list[str] | None = None,
    min_fill: float = 0.5,
) -> pd.DataFrame:
    """Append numeric columns parsed from structured string columns.

    For each *source* string column, every cell is parsed with
    :func:`parse_label_number_tokens`; the union of labels becomes one float
    column each (``NaN`` where a frame lacks that token).  Columns whose
    non-NaN fraction is below ``min_fill`` are dropped (filters noise from
    free-text status fields).

    Parameters
    ----------
    sources : list of column names, optional
        If ``None`` (default), every object/string per-frame column is scanned
        (numeric and ``ts_`` timestamp columns are skipped).
    prefix : str
        Prepended to every derived column (default ``"fn:"``).  On a label
        collision across sources, the later one is qualified as
        ``prefix + source + ":" + label``.

    Returns the original frame unchanged if nothing is derived.
    """
    if df is None or df.empty:
        return df

    if sources is None:
        sources = [
            c for c in df.columns
            if not c.startswith("ts_")
            and not pd.api.types.is_numeric_dtype(df[c])
        ]

    n = len(df)
    new_cols: dict[str, np.ndarray] = {}
    for src in sources:
        if src not in df.columns:
            continue
        parsed = [parse_label_number_tokens(v) for v in df[src].tolist()]
        labels: list[str] = []
        for d in parsed:
            for lab in d:
                if lab not in labels:
                    labels.append(lab)
        for lab in labels:
            col = np.array([d.get(lab, np.nan) for d in parsed], dtype=float)
            if np.isfinite(col).sum() < min_fill * n:
                continue
            name = f"{prefix}{lab}"
            if name in df.columns or name in new_cols:
                name = f"{prefix}{src}:{lab}"  # disambiguate across sources
            if name in df.columns or name in new_cols:
                continue
            new_cols[name] = col

    if not new_cols:
        return df
    out = df.copy()
    for name, col in new_cols.items():
        out[name] = col
    return out
