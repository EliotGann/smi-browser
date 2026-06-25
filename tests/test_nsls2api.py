"""Tests for proposal-directory resolution in :mod:`smi_browser.nsls2api`.

Focus: a proposal that spans multiple beamlines (e.g. 320191 lives on both
SST1 and SMI) must resolve the *SMI* working directory for export, not the
first/other beamline's directory.  All tests mock ``_get`` so they run
offline.
"""
from __future__ import annotations

import pytest

from smi_browser import nsls2api


# Mirrors the live response for proposal 320191: two beamlines, two cycles.
_MULTI_BEAMLINE = {
    "directory_count": 4,
    "directories": [
        {"beamline": "SST1", "cycle": "2026-1",
         "path": "/nsls2/data/sst/proposals/2026-1/pass-320191"},
        {"beamline": "SST1", "cycle": "2026-2",
         "path": "/nsls2/data/sst/proposals/2026-2/pass-320191"},
        {"beamline": "SMI", "cycle": "2026-1",
         "path": "/nsls2/data/smi/proposals/2026-1/pass-320191"},
        {"beamline": "SMI", "cycle": "2026-2",
         "path": "/nsls2/data/smi/proposals/2026-2/pass-320191"},
    ],
}

# A plain single-beamline proposal (the common case).
_SINGLE_BEAMLINE = {
    "directory_count": 2,
    "directories": [
        {"beamline": "SMI", "cycle": "2025-2",
         "path": "/nsls2/data/smi/proposals/2025-2/pass-100000"},
        {"beamline": "SMI", "cycle": "2025-3",
         "path": "/nsls2/data/smi/proposals/2025-3/pass-100000"},
    ],
}

# Older entries with no ``beamline`` field at all.
_NO_BEAMLINE = {
    "directory_count": 1,
    "directories": [
        {"cycle": "2022-1", "path": "/nsls2/xf12id/2022-1/pass-99"},
    ],
}


@pytest.fixture
def patch_get(monkeypatch):
    """Patch ``nsls2api._get`` to return a chosen payload for /directories."""
    def _install(payload):
        def fake_get(path, params=None):
            assert path.endswith("/directories")
            return payload
        monkeypatch.setattr(nsls2api, "_get", fake_get)
    return _install


# --- multi-beamline: the bug this fixes -----------------------------------

def test_multi_beamline_picks_smi_for_cycle(patch_get):
    patch_get(_MULTI_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("320191", "2026-1")
        == "/nsls2/data/smi/proposals/2026-1/pass-320191"
    )
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("320191", "2026-2")
        == "/nsls2/data/smi/proposals/2026-2/pass-320191"
    )


def test_multi_beamline_no_cycle_uses_most_recent_smi(patch_get):
    patch_get(_MULTI_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("320191", None)
        == "/nsls2/data/smi/proposals/2026-2/pass-320191"
    )


def test_multi_beamline_unknown_cycle_falls_back_to_smi(patch_get):
    # Requested cycle has no SMI entry -> most recent SMI dir, never SST1.
    patch_get(_MULTI_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("320191", "2025-1")
        == "/nsls2/data/smi/proposals/2026-2/pass-320191"
    )


def test_multi_beamline_explicit_other_beamline(patch_get):
    patch_get(_MULTI_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle(
            "320191", "2026-1", beamline="SST1"
        )
        == "/nsls2/data/sst/proposals/2026-1/pass-320191"
    )


def test_beamline_filter_disabled_is_cycle_only(patch_get):
    # beamline=None restores the original cycle-only behaviour (first match).
    patch_get(_MULTI_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle(
            "320191", "2026-1", beamline=None
        )
        == "/nsls2/data/sst/proposals/2026-1/pass-320191"
    )


def test_fetch_proposal_directory_prefers_smi(patch_get):
    patch_get(_MULTI_BEAMLINE)
    # First overall entry is SST1, but SMI must win.
    assert (
        nsls2api.fetch_proposal_directory("320191")
        == "/nsls2/data/smi/proposals/2026-1/pass-320191"
    )


# --- single beamline / legacy: must be unaffected -------------------------

def test_single_beamline_cycle_match(patch_get):
    patch_get(_SINGLE_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("100000", "2025-2")
        == "/nsls2/data/smi/proposals/2025-2/pass-100000"
    )


def test_single_beamline_unknown_cycle_uses_most_recent(patch_get):
    patch_get(_SINGLE_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("100000", "1999-9")
        == "/nsls2/data/smi/proposals/2025-3/pass-100000"
    )


def test_no_beamline_field_falls_back_to_cycle(patch_get):
    # Entries without a beamline field: SMI filter matches nothing, so the
    # full list is used and the cycle match wins.
    patch_get(_NO_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory_for_cycle("99", "2022-1")
        == "/nsls2/xf12id/2022-1/pass-99"
    )


def test_no_beamline_field_simple_lookup(patch_get):
    patch_get(_NO_BEAMLINE)
    assert (
        nsls2api.fetch_proposal_directory("99")
        == "/nsls2/xf12id/2022-1/pass-99"
    )


# --- empty / error paths --------------------------------------------------

def test_empty_directories_returns_none(patch_get):
    patch_get({"directory_count": 0, "directories": []})
    assert nsls2api.fetch_proposal_directory_for_cycle("0", "2026-1") is None
    assert nsls2api.fetch_proposal_directory("0") is None


def test_filter_dirs_by_beamline_helper():
    dirs = _MULTI_BEAMLINE["directories"]
    smi = nsls2api._filter_dirs_by_beamline(dirs, "smi")  # case-insensitive
    assert len(smi) == 2
    assert all("smi" in e["path"] for e in smi)
    assert nsls2api._filter_dirs_by_beamline(dirs, "XYZ") == []
    assert nsls2api._filter_dirs_by_beamline(dirs, "") == []
