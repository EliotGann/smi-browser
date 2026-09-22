"""Regression coverage for the Process q-scaling view and its update wiring."""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from bokeh.models import LinearScale, LogScale

from smi_browser.figures.iq import scaled_iq_data
from smi_browser.models.collection import ScanCollection
from test_smi_app_streams import _import_smi_app


@pytest.fixture
def app(monkeypatch):
    app = _import_smi_app(monkeypatch)
    monkeypatch.setattr(app, "_proc_result_cache", {"result": None, "gi_result": None})
    monkeypatch.setattr(app, "_proc_iq_display", {"uids": []})
    monkeypatch.setattr(app, "_selected_uid", lambda: "test-scan")
    monkeypatch.setattr(app, "w_proc_iq_mode", SimpleNamespace(value="merged"))
    monkeypatch.setattr(app, "w_proc_iq_label", SimpleNamespace(value="(frame #)"))
    app.w_proc_iq_plot.object = None
    app.w_proc_q_power.value = 2.0
    app.w_proc_scaled_log_x.value = False
    app.w_proc_scaled_log_y.value = False
    yield app
    app.w_proc_iq_plot.object = None
    app.w_proc_q_power.value = 2.0
    app.w_proc_scaled_log_x.value = False
    app.w_proc_scaled_log_y.value = False


def _result():
    return SimpleNamespace(
        merged_iq=xr.Dataset(
            {"I": ("q", [8., 4., 2.]),
             "saxs_I": ("q", [6., 3., 1.]),
             "waxs_I": ("q", [10., 5., 3.])},
            coords={"q": [0.5, 1., 2.]},
        ),
    )


def _ys(fig):
    return [r.data_source.data["y"] for r in fig.renderers]


def test_merged_curves_update_exponent_and_reset(app):
    result = _result()
    original = result.merged_iq.copy(deep=True)
    app._proc_result_cache["result"] = result
    app._build_proc_iq_plot()

    fig = app.w_proc_scaled_iq_plot.object
    assert isinstance(fig.x_scale, LinearScale)
    assert isinstance(fig.y_scale, LinearScale)
    np.testing.assert_allclose(_ys(fig), [[2, 4, 8], [1.5, 3, 4], [2.5, 5, 12]])
    assert fig.yaxis.axis_label == "I(q) × q^2"

    ordinary = app.w_proc_iq_plot.object
    app.w_proc_q_power.value = 0
    np.testing.assert_allclose(_ys(app.w_proc_scaled_iq_plot.object)[0], [8, 4, 2])
    app.w_proc_q_power.value = 4
    np.testing.assert_allclose(_ys(app.w_proc_scaled_iq_plot.object)[0], [0.5, 4, 32])
    assert app.w_proc_iq_plot.object is ordinary
    xr.testing.assert_identical(result.merged_iq, original)

    app.w_proc_scaled_log_x.value = True
    app.w_proc_scaled_log_y.value = True
    assert isinstance(app.w_proc_scaled_iq_plot.object.x_scale, LogScale)
    assert isinstance(app.w_proc_scaled_iq_plot.object.y_scale, LogScale)
    app.w_proc_iq_plot.object = None
    assert app.w_proc_scaled_iq_plot.object is None


def test_per_frame_scaling_uses_detector_and_frame_labels(app):
    result = _result()
    result.merged_iq["waxs_I"][:] = np.nan  # SAXS-only reduction
    result.per_frame_iq = xr.Dataset(
        {"I": (("frame", "q"), [[99., 99., 99.]] * 2),
         "saxs_I": (("frame", "q"), [[8., 4., 2.], [4., 2., 1.]]),
         "temperature": ("frame", [100., 200.])},
        coords={"q": [0.5, 1., 2.]},
    )
    app._proc_result_cache["result"] = result
    app.w_proc_iq_mode.value = "per-frame"
    app.w_proc_iq_label.value = "temperature"
    app._build_proc_iq_plot()
    fig = app.w_proc_scaled_iq_plot.object
    np.testing.assert_allclose(_ys(fig), [[2, 4, 8], [1, 2, 4]])
    assert [item.label.value for item in fig.legend[0].items] == [
        "temperature=100", "temperature=200",
    ]


def test_per_frame_fallbacks_are_scaled(app):
    result = _result()
    app._proc_result_cache["result"] = result
    app.w_proc_iq_mode.value = "per-frame"
    app._build_proc_iq_plot()
    assert "no per-frame data" in app.w_proc_scaled_iq_plot.object.title.text
    np.testing.assert_allclose(_ys(app.w_proc_scaled_iq_plot.object)[0], [2, 4, 8])

    result.merged_qchi = xr.Dataset(
        {"intensity": (("frame", "chi", "q"), [[[8., 4., 2.], [4., 2., 1.]]])},
        coords={"q": [0.5, 1., 2.], "chi": [0., 90.]},
    )
    app._build_proc_iq_plot()
    np.testing.assert_allclose(_ys(app.w_proc_scaled_iq_plot.object)[0], [1.5, 3, 6])


def test_multi_scan_scaling_survives_pinning_and_switch_to_single(app, monkeypatch):
    coll = ScanCollection()
    for uid in ("scan-a", "scan-b"):
        result = _result()
        result.uid = uid
        coll.add(result, {"uid": uid, "sample_name": uid})
    monkeypatch.setattr(app, "_collection", coll)
    app._proc_iq_display["uids"] = ["scan-a", "scan-b"]
    app.w_proc_iq_plot.object = coll.iq_comparison_bokeh()
    np.testing.assert_allclose(_ys(app.w_proc_scaled_iq_plot.object), [[2, 4, 8]] * 2)
    # Pinning clears the pending-add list, not the displayed overlay.
    app._proc_result_cache["multi_uids"] = []
    app.w_proc_q_power.value = 4
    np.testing.assert_allclose(_ys(app.w_proc_scaled_iq_plot.object), [[0.5, 4, 32]] * 2)
    app._proc_result_cache["result"] = _result()
    app._build_proc_iq_plot()
    assert len(_ys(app.w_proc_scaled_iq_plot.object)) == 3
    assert app._proc_iq_display["uids"] == []


def test_scaling_filters_undefined_values_but_keeps_signed_linear_data():
    q = np.array([0., 0.5, 1., 2., np.nan, np.inf, -1.])
    intensity = np.array([1., -4., 0., 2., 1., 1., 1.])
    x, y = scaled_iq_data(q, intensity, q_power=-1, log_x=False, log_y=False)
    np.testing.assert_allclose(x, [0.5, 1., 2.])
    np.testing.assert_allclose(y, [-8., 0., 1.])
    x, y = scaled_iq_data(q, intensity, q_power=0.5)
    np.testing.assert_allclose(x, [2.])
    np.testing.assert_allclose(y, [2 * np.sqrt(2)])
