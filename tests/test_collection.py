"""Tests for ScanCollection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from smi_browser.models.collection import ScanCollection


def _fake_result(uid, q=None, I=None, geometry="transmission"):
    """Build a minimal fake CombinedReductionResult."""
    if q is None:
        q = np.linspace(0.01, 1.0, 50)
    if I is None:
        I = np.random.default_rng(42).exponential(size=len(q))
    merged_iq = xr.Dataset({"q": ("q", q), "I": ("q", I)})
    return SimpleNamespace(uid=uid, merged_iq=merged_iq, geometry=geometry, timing={"total": 1.5})


def _fake_meta(uid, sample="sample_a", plan="count"):
    return {
        "uid": uid,
        "scan_id": 12345,
        "sample_name": sample,
        "plan_name": plan,
        "detectors": "SAXS",
        "detector_list": ["pil1M"],
    }


class TestScanCollection:
    def test_add_remove(self):
        coll = ScanCollection()
        r = _fake_result("aaa")
        coll.add(r, _fake_meta("aaa"))
        assert "aaa" in coll
        assert len(coll) == 1
        coll.remove("aaa")
        assert "aaa" not in coll
        assert len(coll) == 0

    def test_color_assignment(self):
        coll = ScanCollection()
        r1 = _fake_result("aaa")
        r2 = _fake_result("bbb")
        coll.add(r1, _fake_meta("aaa"))
        coll.add(r2, _fake_meta("bbb"))
        c1 = coll.get_color("aaa")
        c2 = coll.get_color("bbb")
        assert c1 != c2
        assert c1.startswith("#")

    def test_varying_parameters_empty_and_single(self):
        coll = ScanCollection()
        assert coll.varying_parameters() == {}
        coll.add(_fake_result("aaa"), _fake_meta("aaa"))
        assert coll.varying_parameters() == {}

    def test_varying_parameters_detects_diff(self):
        coll = ScanCollection()
        coll.add(_fake_result("aaa"), _fake_meta("aaa", sample="A"))
        coll.add(_fake_result("bbb"), _fake_meta("bbb", sample="B"))
        varying = coll.varying_parameters()
        assert "sample_name" in varying
        assert varying["sample_name"] == ["A", "B"]

    def test_summary_table(self):
        coll = ScanCollection()
        coll.add(_fake_result("aaa"), _fake_meta("aaa"))
        df = coll.summary_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "uid_short" in df.columns

    def test_summary_table_with_label_column(self):
        coll = ScanCollection()
        primary_df = pd.DataFrame({"temperature": [25.0, 26.0]})
        coll.add(_fake_result("aaa"), _fake_meta("aaa"), primary_df=primary_df)
        df = coll.summary_table(label_column="temperature")
        assert "label_val" in df.columns

    def test_get_primary_df(self):
        coll = ScanCollection()
        primary_df = pd.DataFrame({"x": [1, 2, 3]})
        coll.add(_fake_result("aaa"), _fake_meta("aaa"), primary_df=primary_df)
        assert coll.get_primary_df("aaa") is primary_df
        assert coll.get_primary_df("nonexistent") is None

    def test_stack_iq(self):
        coll = ScanCollection()
        q = np.linspace(0.01, 1.0, 20)
        coll.add(_fake_result("aaa", q=q), _fake_meta("aaa"))
        coll.add(_fake_result("bbb", q=q), _fake_meta("bbb"))
        ds = coll.stack_iq("sample")
        assert isinstance(ds, xr.Dataset)

    def test_uids_ordering(self):
        coll = ScanCollection()
        for uid in ["ccc", "aaa", "bbb"]:
            coll.add(_fake_result(uid), _fake_meta(uid))
        assert coll.uids == ["ccc", "aaa", "bbb"]

    def test_remove_nonexistent_noop(self):
        coll = ScanCollection()
        coll.remove("nonexistent")  # should not raise
