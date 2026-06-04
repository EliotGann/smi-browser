"""Configuration constants and defaults for the SMI Browser.

All beamline-calibrated values are imported from ``smi_tiled.defaults`` so
that a single upstream change propagates everywhere.
"""
from __future__ import annotations

import os

import pandas as pd

from smi_tiled import defaults as smid

# ---------------------------------------------------------------------------
# Tiled connection
# ---------------------------------------------------------------------------

DEFAULT_TILED_URI = smid.DEFAULT_TILED_URI
DEFAULT_CATALOG = smid.DEFAULT_CATALOG
DEFAULT_SAXS_MASK_NAME = smid.DEFAULT_SAXS_MASK_NAME
DEFAULT_WAXS_MASK_NAME = smid.DEFAULT_WAXS_MASK_NAME

# ---------------------------------------------------------------------------
# Cycle defaults  (used by the proposal selector when the NSLS-II API is
# unreachable, e.g. when running on an off-network machine that still has
# HTTPS access to tiled.nsls2.bnl.gov)
# ---------------------------------------------------------------------------

# Cycle to pre-select on first launch when no saved value or API current is
# available.  Override with ``$SMI_BROWSER_DEFAULT_CYCLE``.
DEFAULT_CYCLE = os.environ.get("SMI_BROWSER_DEFAULT_CYCLE", "2026-2")

# Hardcoded fallback list shown in the cycle dropdown when the NSLS-II API is
# unreachable.  Most-recent first; bump as new cycles begin.
RECENT_CYCLES = [
    "2026-3", "2026-2", "2026-1",
    "2025-3", "2025-2", "2025-1",
    "2024-3", "2024-2", "2024-1",
    "2023-3", "2023-2", "2023-1",
]

# ---------------------------------------------------------------------------
# Detector classification
# ---------------------------------------------------------------------------

SAXS_DETECTOR_NAMES = smid.SAXS_DETECTOR_NAMES
WAXS_DETECTOR_NAMES = smid.WAXS_DETECTOR_NAMES

# ---------------------------------------------------------------------------
# Loader-side calibrated defaults (frozen dataclass exposed by smi-tiled)
# ---------------------------------------------------------------------------

_LD = smid.LOADER_DEFAULTS
DEFAULT_SAXS_ROW_DELTA = _LD.saxs_row_delta_px
DEFAULT_SAXS_COL_DELTA = _LD.saxs_col_delta_px
DEFAULT_WAXS_ROW_DELTA = _LD.waxs_row_delta_px
DEFAULT_WAXS_COL_DELTA = _LD.waxs_col_delta_px
DEFAULT_SAXS_DIST_DELTA = _LD.saxs_distance_delta_mm

# ---------------------------------------------------------------------------
# Processing defaults  (UI-side; mirror upstream smi-tiled defaults so widgets
# show meaningful numbers even before any override.  When a widget value
# still equals its default, _on_process passes None so the upstream loader
# supplies its own calibrated default.)
# ---------------------------------------------------------------------------

PAGE_SIZE = 25

DEFAULT_N_Q = 2000          # smi-tiled default is 1000; smi-browser used 2000
DEFAULT_N_CHI = 360
DEFAULT_SAXS_MASK = ""      # empty → use bundled default from smi-tiled
DEFAULT_WAXS_MASK = ""
DEFAULT_DEZINGER = 3000.0
DEFAULT_INCIDENT_ANGLE = 0.0
DEFAULT_THETA_OFFSET = -0.5
DEFAULT_N_QXY = 500
DEFAULT_N_QZ = 500

# ---------------------------------------------------------------------------
# Search / table
# ---------------------------------------------------------------------------

COMMON_SEARCH_KEYS = [
    "sample_name",
    "plan_name",
    "data_session",
    "proposal.first_name",
    "proposal.last_name",
    "project_name",
    "institution",
    "scan_id",
    "uid",
    "detectors",
]

RESULT_COLS = [
    "scan_id", "n_steps", "sample_name", "plan_name",
    "data_session", "detectors", "exit_status", "time", "uid",
]

EMPTY_DF = pd.DataFrame(columns=RESULT_COLS)

# Search filter types
SEARCH_TYPES = ["Anywhere", "Text in field", "Contains", "Exact"]
SEARCH_TYPE_MAP = {
    "Anywhere": "anywhere",
    "Text in field": "like",
    "Contains": "contains",
    "Exact": "exact",
}
