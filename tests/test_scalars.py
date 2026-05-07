"""Tests for scalar helper functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from smi_browser.data.scalars import scalars_to_dataframe


class TestScalarsToDataframe:
    def test_empty_dict(self):
        assert scalars_to_dataframe({}).empty

    def test_single_column(self):
        df = scalars_to_dataframe({"x": np.array([1.0, 2.0, 3.0])})
        assert list(df.columns) == ["x"]
        assert len(df) == 3

    def test_padding_unequal_lengths(self):
        data = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([10.0, 20.0]),
        }
        df = scalars_to_dataframe(data)
        assert len(df) == 3
        assert np.isnan(df["b"].iloc[2])

    def test_scalar_value_promoted(self):
        df = scalars_to_dataframe({"x": np.float64(42.0)})
        assert len(df) == 1
        assert df["x"].iloc[0] == 42.0

    def test_skips_multidim(self):
        data = {
            "good": np.array([1.0, 2.0]),
            "bad": np.ones((3, 4)),
        }
        df = scalars_to_dataframe(data)
        assert "good" in df.columns
        assert "bad" not in df.columns
