"""Client for the NSLS-II API (api.nsls2.bnl.gov).

Provides helpers to fetch user data-sessions, proposal details, and
facility cycles for populating the proposal selector UI.  All calls
are unauthenticated (no API key needed for the read endpoints used here).
"""
from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_BASE = "https://api.nsls2.bnl.gov/v1"
_TIMEOUT = httpx.Timeout(15.0)
BEAMLINE = "SMI"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProposalInfo:
    """Lightweight summary of a proposal for display in the selector."""
    proposal_id: str
    data_session: str
    title: str
    pi_name: str
    cycles: list[str]

    @property
    def display_label(self) -> str:
        """Human-readable label for a dropdown option."""
        short_title = self.title[:60] + "…" if len(self.title) > 60 else self.title
        return f"{self.data_session}  —  {self.pi_name}  —  {short_title}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict[str, Any] | None = None) -> dict:
    """GET from the NSLS-II API, returning parsed JSON."""
    url = f"{API_BASE}{path}"
    resp = httpx.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_cycles() -> list[str]:
    """Return all NSLS-II cycle names, most recent last."""
    try:
        data = _get("/facility/nsls2/cycles")
        return data.get("cycles", [])
    except Exception:
        log.exception("Failed to fetch NSLS-II cycles")
        return []


def fetch_current_cycle() -> str:
    """Return the current operating cycle string, e.g. '2026-2'."""
    try:
        data = _get("/facility/nsls2/cycles/current")
        return data.get("cycle", "")
    except Exception:
        log.exception("Failed to fetch current cycle")
        return ""


def fetch_user_data_sessions(username: str) -> list[str]:
    """Return the data-session strings the user can access.

    Calls ``GET /v1/data-session/{username}`` which is unauthenticated.
    """
    try:
        data = _get(f"/data-session/{username}")
        return data.get("data_sessions", [])
    except Exception:
        log.exception("Failed to fetch data sessions for %s", username)
        return []


def fetch_user_beamline_access(username: str) -> list[str]:
    """Return the beamlines the user has full access to."""
    try:
        data = _get(f"/data-session/{username}")
        return data.get("beamline_all_access", [])
    except Exception:
        log.exception("Failed to fetch beamline access for %s", username)
        return []


def fetch_proposal(proposal_id: str) -> dict:
    """Return full proposal details for a given proposal ID."""
    try:
        data = _get(f"/proposal/{proposal_id}")
        return data.get("proposal", {})
    except Exception:
        log.exception("Failed to fetch proposal %s", proposal_id)
        return {}


def fetch_pi(proposal_id: str) -> dict:
    """Return the PI info for a proposal."""
    try:
        data = _get(f"/proposal/{proposal_id}/principal-investigator")
        return data.get("user", {})
    except Exception:
        log.debug("Failed to fetch PI for proposal %s", proposal_id)
        return {}


def fetch_proposals_for_cycle(cycle: str) -> list[str]:
    """Return proposal IDs for a given NSLS-II cycle."""
    try:
        data = _get(f"/facility/nsls2/cycle/{cycle}/proposals")
        return data.get("proposals", [])
    except Exception:
        log.exception("Failed to fetch proposals for cycle %s", cycle)
        return []


def fetch_commissioning_proposals(beamline: str = BEAMLINE) -> list[str]:
    """Return commissioning proposal IDs for a beamline."""
    try:
        data = _get("/proposals/commissioning", params={"beamline": beamline})
        return data.get("commissioning_proposals", [])
    except Exception:
        log.exception("Failed to fetch commissioning proposals")
        return []


def fetch_proposal_directory(proposal_id: str) -> str | None:
    """Return the filesystem path for a proposal's working directory.

    Calls ``GET /v1/proposal/{id}/directories`` and returns the first
    directory path, or *None* if unavailable.
    """
    try:
        data = _get(f"/proposal/{proposal_id}/directories")
        dirs = data.get("directories", [])
        if dirs:
            return dirs[0].get("path")
    except Exception:
        log.debug("Failed to fetch directory for proposal %s", proposal_id)
    return None


# ---------------------------------------------------------------------------
# High-level: build enriched proposal list for a user
# ---------------------------------------------------------------------------

def _proposal_id_from_data_session(ds: str) -> str:
    """Extract the numeric proposal ID from 'pass-XXXXXX'."""
    if ds.startswith("pass-"):
        return ds[5:]
    return ds


def build_proposal_list(
    username: str,
    cycle: str | None = None,
) -> list[ProposalInfo]:
    """Build an enriched list of proposals the user can access.

    Parameters
    ----------
    username : str
        The logged-in user's username.
    cycle : str or None
        If provided, only include proposals that overlap this cycle.

    Returns
    -------
    list[ProposalInfo]
        Sorted by proposal_id descending (most recent first).
    """
    data_sessions = fetch_user_data_sessions(username)
    if not data_sessions:
        return []

    # If cycle filter requested, get the set of proposal IDs for that cycle
    cycle_ids: set[str] | None = None
    if cycle:
        if cycle.lower() == "commissioning":
            cycle_proposals = fetch_commissioning_proposals()
        else:
            cycle_proposals = fetch_proposals_for_cycle(cycle)
        cycle_ids = set(cycle_proposals)

    results: list[ProposalInfo] = []
    for ds in data_sessions:
        pid = _proposal_id_from_data_session(ds)

        # Cycle filter: skip if proposal not in the selected cycle
        if cycle_ids is not None and pid not in cycle_ids:
            continue

        # Fetch proposal detail (includes title, cycles, users)
        prop = fetch_proposal(pid)
        if not prop:
            # Still include with minimal info if API call fails
            results.append(ProposalInfo(
                proposal_id=pid,
                data_session=ds,
                title="(details unavailable)",
                pi_name="?",
                cycles=[],
            ))
            continue

        # Filter by beamline: only include if SMI is in the instruments
        instruments = [i.lower() for i in prop.get("instruments", [])]
        if instruments and BEAMLINE.lower() not in instruments:
            continue

        # Find PI from the users list
        pi_name = "?"
        users = prop.get("users", [])
        for u in users:
            if u.get("is_pi"):
                pi_name = f"{u.get('first_name', '')} {u.get('last_name', '')}".strip()
                break

        # If no is_pi flag found, try the dedicated PI endpoint
        if pi_name == "?":
            pi = fetch_pi(pid)
            if pi:
                pi_name = f"{pi.get('first_name', '')} {pi.get('last_name', '')}".strip()

        title = prop.get("title", "(no title)")
        cycles = prop.get("cycles", [])

        results.append(ProposalInfo(
            proposal_id=pid,
            data_session=prop.get("data_session", ds),
            title=title,
            pi_name=pi_name,
            cycles=cycles,
        ))

    # Sort newest first (higher proposal_id = newer)
    results.sort(key=lambda p: p.proposal_id, reverse=True)
    return results


# ---------------------------------------------------------------------------
# High-level: build full beamline proposal list for a cycle
# ---------------------------------------------------------------------------

def build_cycle_proposal_list(
    cycle: str,
    beamline: str = BEAMLINE,
    max_workers: int = 20,
) -> list[ProposalInfo]:
    """Build a list of ALL proposals for a beamline in a given cycle.

    This fetches every proposal in the cycle concurrently and filters
    by instrument.  Useful for beamline scientists who have access to
    all proposals but aren't explicitly listed on each one.

    Parameters
    ----------
    cycle : str
        The cycle name, e.g. '2026-1'.
    beamline : str
        Beamline name to filter by (default: SMI).
    max_workers : int
        Concurrency for API calls.

    Returns
    -------
    list[ProposalInfo]
        Sorted by proposal_id descending (most recent first).
    """
    if cycle.lower() == "commissioning":
        proposal_ids = fetch_commissioning_proposals(beamline)
    else:
        proposal_ids = fetch_proposals_for_cycle(cycle)

    if not proposal_ids:
        return []

    def _fetch_and_filter(pid: str) -> ProposalInfo | None:
        """Fetch a single proposal; return ProposalInfo if it matches beamline."""
        try:
            prop = fetch_proposal(pid)
            if not prop:
                return None
            instruments = [i.upper() for i in prop.get("instruments", [])]
            if beamline.upper() not in instruments:
                return None

            # Extract PI name
            pi_name = "?"
            for u in prop.get("users", []):
                if u.get("is_pi"):
                    pi_name = f"{u.get('first_name', '')} {u.get('last_name', '')}".strip()
                    break
            if pi_name == "?":
                pi = fetch_pi(pid)
                if pi:
                    pi_name = f"{pi.get('first_name', '')} {pi.get('last_name', '')}".strip()

            return ProposalInfo(
                proposal_id=pid,
                data_session=prop.get("data_session", f"pass-{pid}"),
                title=prop.get("title", "(no title)"),
                pi_name=pi_name,
                cycles=prop.get("cycles", []),
            )
        except Exception:
            log.debug("Failed to fetch/filter proposal %s", pid)
            return None

    results: list[ProposalInfo] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_and_filter, pid): pid for pid in proposal_ids}
        for future in concurrent.futures.as_completed(futures):
            info = future.result()
            if info is not None:
                results.append(info)

    results.sort(key=lambda p: p.proposal_id, reverse=True)
    return results
