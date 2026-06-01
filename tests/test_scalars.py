"""Tests for scalar helper functions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from smi_browser.data.scalars import (
    derive_virtual_columns,
    parse_label_number_tokens,
    scalars_to_dataframe,
)


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


class TestParseLabelNumberTokens:
    def test_real_target_file_name(self):
        got = parse_label_number_tokens(
            "Lucas_sample2_pos1_2450.00eV_ai0.50_wa9_bpm1.995_degC100.0")
        assert got == {
            "sample": 2.0, "pos": 1.0, "eV": 2450.0, "ai": 0.5,
            "wa": 9.0, "bpm": 1.995, "degC": 100.0,
        }

    def test_bare_number_and_textonly_excluded(self):
        # "120" has no adjacent letters; "Lucas" has no number → both excluded.
        assert parse_label_number_tokens("Lucas_120_run") == {}

    def test_unit_suffix_used_when_no_prefix(self):
        assert parse_label_number_tokens("2450.00eV") == {"eV": 2450.0}

    def test_negative_and_non_string(self):
        assert parse_label_number_tokens("x-3.5")["x"] == -3.5
        assert parse_label_number_tokens(None) == {}
        assert parse_label_number_tokens("") == {}

    def test_bytes_from_cache_are_decoded(self):
        # HDF5-cached strings read back as bytes / numpy bytes — must still parse.
        assert parse_label_number_tokens(b"x_ai0.50_eV2450.00") == {
            "ai": 0.5, "eV": 2450.0}
        assert parse_label_number_tokens(np.bytes_(b"x_ai0.50")) == {"ai": 0.5}
        assert parse_label_number_tokens(np.str_("x_ai0.50")) == {"ai": 0.5}


class TestDeriveVirtualColumns:
    def _frame(self):
        return pd.DataFrame({
            "target_file_name": [
                "Lucas_pos1_2450.00eV_ai0.50_degC100.0",
                "Lucas_pos2_2460.00eV_ai4.00",  # no degC token
            ],
            "energy_energy": [2450.1, 2460.2],     # numeric → ignored
            "ts_target_file_name": [1.0, 2.0],     # ts_ → ignored
            "att2_9_status": ["+- 120 uA", "+- 120 uA"],  # no glued label
        })

    def test_adds_prefixed_columns(self):
        out = derive_virtual_columns(self._frame())
        assert {"fn:ai", "fn:eV", "fn:pos", "fn:degC"} <= set(out.columns)
        np.testing.assert_allclose(out["fn:ai"], [0.5, 4.0])

    def test_missing_token_is_nan(self):
        out = derive_virtual_columns(self._frame())
        assert out["fn:degC"].iloc[0] == 100.0
        assert np.isnan(out["fn:degC"].iloc[1])

    def test_ignores_numeric_and_ts_and_noise(self):
        out = derive_virtual_columns(self._frame())
        # No fn: column derived from numeric/ts/free-text columns.
        assert not any(c.startswith("fn:") and "energy" in c for c in out.columns)
        assert "fn:120" not in out.columns

    def test_min_fill_drops_sparse_columns(self):
        # Only 1 of 4 rows carries an "ai" token → below min_fill=0.5 → dropped.
        df = pd.DataFrame({"target_file_name": [
            "run_ai0.50", "run", "run", "run",
        ]})
        out = derive_virtual_columns(df, min_fill=0.5)
        assert "fn:ai" not in out.columns

    def test_collision_across_sources_is_qualified(self):
        df = pd.DataFrame({
            "target_file_name": ["a_ai0.5", "a_ai0.6"],
            "other": ["ai9", "ai8"],
        })
        out = derive_virtual_columns(df)
        assert "fn:ai" in out.columns
        assert "fn:other:ai" in out.columns

    def test_empty_frame_returned_unchanged(self):
        empty = pd.DataFrame()
        assert derive_virtual_columns(empty).empty

    def test_bytes_object_column(self):
        # Simulates a primary frame rebuilt from the HDF5 cache (bytes strings).
        df = pd.DataFrame({"target_file_name": np.array(
            [b"r_ai0.50_eV2450", b"r_ai4.00_eV2460"], dtype=object)})
        out = derive_virtual_columns(df)
        assert {"fn:ai", "fn:eV"} <= set(out.columns)
        np.testing.assert_allclose(out["fn:ai"], [0.5, 4.0])
