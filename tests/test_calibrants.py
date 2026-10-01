"""Shared standards, unit conversion, and user-defined calibrants."""
import json

import numpy as np
import pytest

from smi_browser.calibrants import load_calibrants
from smi_browser.calibrate import agbh_q


def test_bundled_standards_and_units(monkeypatch):
    monkeypatch.delenv("SMI_BROWSER_CALIBRANTS_FILE", raising=False)
    standards = load_calibrants()
    agbh = standards["silver_behenate"]
    for n in range(1, 10):
        assert agbh.q(n) == pytest.approx(agbh_q(n), rel=1e-12)
    gold = standards["gold"]
    assert gold.labels == ("111", "200")
    np.testing.assert_allclose(gold.q_nm, [26.68, 30.81])
    assert gold.nearest(30.7) == 2
    assert gold.nearest(26.6) == 1
    assert list(gold.options().values()) == [1, 2]


def test_custom_standards_and_overrides(tmp_path, monkeypatch):
    path = tmp_path / "standards.json"
    path.write_text(json.dumps([
        dict(id="lamellar", name="My lamellae", q1_angstrom_inv=0.2, max_order=4),
        dict(id="gold", name="Gold reference", peaks=[
            dict(label="111", q_angstrom_inv=2.67), dict(label="200", q_angstrom_inv=3.08)]),
    ]))
    monkeypatch.setenv("SMI_BROWSER_CALIBRANTS_FILE", str(path))
    standards = load_calibrants()
    np.testing.assert_allclose(standards["lamellar"].q_nm, [2, 4, 6, 8])
    assert standards["gold"].q(1) == pytest.approx(26.7)
    assert "silver_behenate" in standards


@pytest.mark.parametrize("definition", [
    {"q1_angstrom_inv": 0}, {"q1_angstrom_inv": float("nan")},
    {"q1_angstrom_inv": .1, "max_order": 1.5},
    {"q1_angstrom_inv": .1, "peaks": []}, {"peaks": []},
    {"peaks": [{"label": "a", "q_angstrom_inv": 2}, {"label": "b", "q_angstrom_inv": 1}]},
    {"peaks": [{"label": "a", "q_angstrom_inv": 1}, {"label": "a", "q_angstrom_inv": 2}]},
])
def test_invalid_definitions_rejected(tmp_path, definition):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([dict(id="custom", name="Custom", **definition)]))
    with pytest.raises(ValueError):
        load_calibrants(path)
