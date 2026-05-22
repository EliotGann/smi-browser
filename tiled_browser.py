"""
tiled_browser.py  –  Lazy search and metadata helpers for SMI tiled catalogs.

Design principles
-----------------
* Stay lazy: keep tiled nodes as catalog references; never call .read() during
  search or browsing.  Data is only fetched when explicitly requested.
* One frame at a time: image thumbnails fetch a single index slice, not the
  full stack.
* Paginate: never ask for all results – slice the catalog with [offset:limit].
* Metadata first: run_summary() only touches .metadata, zero array I/O.

Quickstart
----------
    from tiled_browser import connect, search, page_summaries, run_summary
    cat = connect()
    results = search(cat, text="AgBH", limit=25)
    summaries = page_summaries(results)          # metadata only, no arrays
    run = results["625e67a7-797b-4af5-9c9b-357e70d2fd9f"]
    print(run_summary(run))
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import pathlib
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global fix: the SMI tiled server returns sorting: [['', 1]] in its catalog
# metadata. The tiled client bakes that into _sorting_params = {'sort': ''}
# at Container construction time, and newer tiled servers reject sort='' with
# 422.  Monkey-patch Container.__init__ to strip the empty sort globally.
# ---------------------------------------------------------------------------

from tiled.client.container import Container as _Container

_original_container_init = _Container.__init__


def _patched_container_init(self, *args, **kwargs):
    _original_container_init(self, *args, **kwargs)
    if getattr(self, "_sorting_params", None) == {"sort": ""}:
        self._sorting_params = {}
    if getattr(self, "_reversed_sorting_params", None) == {"sort": ""}:
        self._reversed_sorting_params = {}


_Container.__init__ = _patched_container_init

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TILED_URI  = "https://tiled.nsls2.bnl.gov"
CATALOG    = "smi/migration"
PAGE_SIZE  = 25    # safe default – avoids hammering tiled with huge requests

_COUNT_CACHE_PATH = pathlib.Path.home() / ".smi_browser_count_cache.json"


def load_cached_count(catalog: str = CATALOG) -> int | None:
    """Load the cached total count for a catalog, or None if unavailable."""
    try:
        data = json.loads(_COUNT_CACHE_PATH.read_text())
        return data.get(catalog)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_cached_count(total: int, catalog: str = CATALOG) -> None:
    """Persist the latest known total count for fast startup next time."""
    try:
        data = {}
        if _COUNT_CACHE_PATH.exists():
            data = json.loads(_COUNT_CACHE_PATH.read_text())
        data[catalog] = total
        _COUNT_CACHE_PATH.write_text(json.dumps(data))
    except OSError:
        pass  # non-critical


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(uri: str = TILED_URI, catalog: str = CATALOG):
    """
    Return a lazy catalog node.  No data is fetched.  Re-use across the
    session to avoid repeated authentication round-trips.

    Parameters
    ----------
    uri : str
        Tiled server base URI.
    catalog : str
        Slash-separated path into the tiled catalog tree,
        e.g. "smi/migration".

    Returns
    -------
    tiled catalog node (lazy).
    """
    import httpx
    from tiled.client.context import Context

    # Build a context without triggering interactive authentication.
    # If cached tokens exist they will be used; otherwise the connection
    # raises so the GUI can prompt for credentials.
    context, node_path_parts = Context.from_any_uri(
        uri, timeout=httpx.Timeout(60.0),
    )
    # Try to load cached tokens (saved from a previous session).
    if context.server_info.authentication.providers:
        context.use_cached_tokens()
    # Prevent from_context from falling into interactive terminal prompts.
    # We mark external auth so it skips context.authenticate(); the GUI
    # login form handles credentials instead.
    context.has_external_auth = True

    from tiled.client.constructors import from_context
    root = from_context(
        context,
        structure_clients="numpy",
        node_path_parts=node_path_parts,
        remember_me=True,
    )
    for part in catalog.split("/"):
        root = root[part]
    return root


# ---------------------------------------------------------------------------
# Fast REST-based bulk page fetch (metadata only, one HTTP round-trip)
# ---------------------------------------------------------------------------

def _build_rest_params(
    offset: int = 0,
    limit: int = PAGE_SIZE,
) -> list[tuple[str, str]]:
    """Build pagination parameters for the tiled REST search endpoint."""
    return [
        ("page[offset]", str(offset)),
        ("page[limit]", str(limit)),
    ]


def _apply_filters(cat, unified_filters: list[tuple[str, str, str]] | None = None):
    """
    Apply search filters to a catalog node using tiled's Python query API.

    Returns the filtered catalog node (or original if no filters).
    """
    if not unified_filters:
        return cat

    from tiled.queries import Contains, Eq, FullText, Like

    node = cat
    for ftype, key, value in unified_filters:
        value = value.strip()
        if not value:
            continue
        ftype = ftype.strip().lower()
        if ftype == "anywhere":
            node = node.search(FullText(value))
        elif ftype == "like":
            # "like" = SQL LIKE substring match (case-insensitive for lowercase)
            key = key.strip()
            if not key:
                continue
            # Auto-add wildcards if not present
            pattern = value if ("%" in value or "_" in value) else f"%{value}%"
            node = node.search(Like(key, pattern))
        elif ftype == "contains":
            # "contains" = check if value is contained in a list/array field
            key = key.strip()
            if not key:
                continue
            node = node.search(Contains(key, value))
        elif ftype == "exact":
            key = key.strip()
            if not key:
                continue
            node = node.search(Eq(key, value))
    return node


def _summary_from_rest_item(item: dict) -> dict:
    """Extract a summary dict from a REST API search result item."""
    md = item.get("attributes", {}).get("metadata", {})
    start = md.get("start", {})
    stop = md.get("stop", {})

    t0 = start.get("time")
    time_str = (
        datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M")
        if t0 else "?"
    )

    det_list = start.get("detectors", [])
    if isinstance(det_list, str):
        det_list = [det_list]

    num_events = stop.get("num_events", {})
    if isinstance(num_events, dict):
        n_steps = num_events.get("primary", "?")
    else:
        n_steps = num_events

    return {
        "uid": start.get("uid", item.get("id", "?")),
        "scan_id": start.get("scan_id", "?"),
        "time": time_str,
        "plan_name": start.get("plan_name", "?"),
        "sample_name": start.get(
            "sample_name",
            start.get("sample", start.get("Sample", "?")),
        ),
        "exit_status": stop.get("exit_status", "?"),
        "n_steps": n_steps,
        "data_session": str(start.get(
            "data_session",
            start.get("institution", start.get("proposal_id", "?")),
        )),
        "detectors": ", ".join(det_list) if det_list else "?",
    }


def fetch_page_fast(
    cat,
    unified_filters: list[tuple[str, str, str]] | None = None,
    offset: int = 0,
    limit: int = PAGE_SIZE,
    count_hint: int | None = None,
) -> tuple[list[dict], int]:
    """
    Fetch one page of scan summaries via the REST API, newest first.

    Uses tiled's Python query API to apply filters (FullText, Contains, Eq),
    then paginates the result node via REST for efficient metadata retrieval.

    Parameters
    ----------
    offset : int
        Logical offset in newest-first order (0 = most recent page).
    limit : int
        Page size.
    count_hint : int or None
        If provided for unfiltered queries, skip the expensive count and
        use this as the approximate total.  The response ``meta.count``
        still returns the real count.

    Returns (summaries, total_count).
    """
    # Apply search filters via tiled Python client
    node = _apply_filters(cat, unified_filters)
    http_client = node.context.http_client
    search_url = node.item["links"]["search"]

    # Extract filter params from the node so REST calls include them.
    # Without this, REST pagination on a filtered node returns unfiltered data.
    filter_params = getattr(node, "_queries_as_params", {})

    # Determine total count
    if count_hint is not None and count_hint > 0 and not unified_filters:
        # Fast path: use cached count, fetch directly from estimated end.
        total_estimate = count_hint
    else:
        # For both filtered and unfiltered: use REST count with filter params.
        params = _build_rest_params(0, 0)
        params.append(("fields", "count"))
        resp = http_client.get(search_url, params={
            **dict(params), **filter_params,
        })
        resp.raise_for_status()
        total_estimate = resp.json().get("meta", {}).get("count", 0)

    if total_estimate == 0:
        return [], 0

    if offset >= total_estimate:
        return [], total_estimate

    # Fetch from end (newest-first)
    real_offset = max(0, total_estimate - offset - limit)
    real_limit = min(limit, total_estimate - offset)
    log.warning(
        "DEBUG fetch_page_fast: total_estimate=%d, offset=%d, limit=%d → real_offset=%d, real_limit=%d, filters=%r, count_hint=%r",
        total_estimate, offset, limit, real_offset, real_limit, unified_filters, count_hint,
    )

    params = {
        "page[offset]": str(real_offset),
        "page[limit]": str(real_limit),
        "fields": "metadata",
        **filter_params,
    }
    resp = http_client.get(search_url, params=params)
    resp.raise_for_status()
    resp_json = resp.json()
    # Server bug: meta.count from data response ignores filters (returns
    # unfiltered total).  Only trust it for unfiltered queries (corrects hints).
    if unified_filters:
        total = total_estimate
    else:
        total = resp_json.get("meta", {}).get("count", total_estimate)
    items = resp_json.get("data", [])
    log.warning(
        "DEBUG fetch_page_fast: server returned %d items, meta.count=%s, using total=%d (estimate was %d)",
        len(items), resp_json.get("meta", {}).get("count", "N/A"), total, total_estimate,
    )

    # If hint was off (got 0 items), correct with real total
    if not items and total > 0 and total != total_estimate:
        real_offset = max(0, total - offset - limit)
        real_limit = min(limit, total - offset)
        params = {
            "page[offset]": str(real_offset),
            "page[limit]": str(real_limit),
            "fields": "metadata",
            **filter_params,
        }
        resp = http_client.get(search_url, params=params)
        resp.raise_for_status()
        items = resp.json().get("data", [])

    # Reverse so newest is first
    items.reverse()

    summaries = [_summary_from_rest_item(item) for item in items]
    return summaries, total


def fetch_uids_fast(
    cat,
    unified_filters: list[tuple[str, str, str]] | None = None,
    max_uids: int = 200,
) -> list[str]:
    """
    Fetch up to *max_uids* UIDs matching the filters, newest-first.

    Uses a single REST request with ``fields=none`` so no metadata is
    transferred — only item IDs (which are the UIDs).  This is
    dramatically faster than fetch_page_fast for queue population.
    """
    node = _apply_filters(cat, unified_filters)
    http_client = node.context.http_client
    search_url = node.item["links"]["search"]
    filter_params = getattr(node, "_queries_as_params", {})

    # Get count first (limit=0 request is fast).
    params = {
        "page[offset]": "0",
        "page[limit]": "0",
        **filter_params,
    }
    resp = http_client.get(search_url, params=params)
    resp.raise_for_status()
    total = resp.json().get("meta", {}).get("count", 0)
    if total == 0:
        return []

    # Fetch from end (newest-first) in one big request.
    # fields="" maps to EntryFields.none — server returns item IDs only,
    # no metadata blobs, making this dramatically faster.
    fetch_count = min(max_uids, total)
    real_offset = max(0, total - fetch_count)

    params = {
        "page[offset]": str(real_offset),
        "page[limit]": str(fetch_count),
        "fields": "",
        **filter_params,
    }
    resp = http_client.get(search_url, params=params)
    resp.raise_for_status()
    items = resp.json().get("data", [])

    # Reverse so newest is first; item "id" is the UID
    uids = [item["id"] for item in reversed(items) if item.get("id")]
    return uids


def count_fast(
    cat,
    unified_filters: list[tuple[str, str, str]] | None = None,
) -> int:
    """
    Return total matching scan count via REST with proper filter params.
    """
    node = _apply_filters(cat, unified_filters)
    http_client = node.context.http_client
    search_url = node.item["links"]["search"]
    filter_params = getattr(node, "_queries_as_params", {})
    params = {
        "page[offset]": "0",
        "page[limit]": "0",
        **filter_params,
    }
    resp = http_client.get(search_url, params=params)
    resp.raise_for_status()
    return resp.json().get("meta", {}).get("count", 0)


# ---------------------------------------------------------------------------
# Search & pagination
# ---------------------------------------------------------------------------

def build_query(cat, text: str = "", filters: dict[str, Any] | None = None,
                like_filters: list[tuple[str, str]] | None = None,
                unified_filters: list[tuple[str, str, str]] | None = None):
    """
    Apply search predicates to a catalog and return a filtered catalog.
    Does NOT slice for pagination – caller slices afterwards.
    Results are sorted reverse-chronologically by default.

    Parameters
    ----------
    cat : tiled catalog node
    text : str
        FullText search string.  Empty → no text filter.
    filters : dict or None
        Exact-match metadata filters, e.g.
        {"start.plan_name": "run_waxs", "start.sample_name": "AgBH"}.
        Each pair is AND-ed together.
    like_filters : list of (key, pattern) or None
        Stacked SQL LIKE filters.  Each (key, pattern) pair is applied
        sequentially.  The pattern should use SQL wildcards (% for any
        characters, _ for single character).  If the user-supplied value
        has no %, it's automatically wrapped as ``%value%`` to match
        anywhere in the string.
    unified_filters : list of (type, key_or_text, value) or None
        Unified stackable filters.  Each tuple is (search_type, key, value):
        - ("anywhere", "", text)    → FullText search
        - ("like", key, pattern)    → SQL LIKE on metadata key (case-sensitive,
          auto-wrapped with %...% if no wildcards present)
        - ("contains", key, value)  → substring match on metadata key
          (case-sensitive; cleaner syntax — no % needed)
        - ("exact", key, value)     → exact-match Eq on metadata key

    Returns
    -------
    Filtered (still lazy) catalog node, sorted newest-first.
    """
    from tiled.queries import Contains, Eq, FullText, Like

    results = cat.sort(("scan_id", -1))
    if text.strip():
        results = results.search(FullText(text.strip()))
    if filters:
        for field, value in filters.items():
            if value not in (None, "", "any"):
                results = results.search(Eq(field, value))
    if like_filters:
        for key, pattern in like_filters:
            key = key.strip()
            pattern = pattern.strip()
            if not key or not pattern:
                continue
            # Auto-wrap in % if user didn't supply wildcards
            if "%" not in pattern and "_" not in pattern:
                pattern = f"%{pattern}%"
            results = results.search(Like(key, pattern))
    if unified_filters:
        for ftype, key, value in unified_filters:
            value = value.strip()
            if not value:
                continue
            ftype = ftype.strip().lower()
            if ftype == "anywhere":
                results = results.search(FullText(value))
            elif ftype == "like":
                key = key.strip()
                if not key:
                    continue
                if "%" not in value and "_" not in value:
                    value = f"%{value}%"
                results = results.search(Like(key, value))
            elif ftype == "contains":
                key = key.strip()
                if not key:
                    continue
                full_key = key if "." in key else f"start.{key}"
                results = results.search(Contains(full_key, value))
            elif ftype == "exact":
                key = key.strip()
                if not key:
                    continue
                results = results.search(Eq(key, value))
    return results


def search(
    cat,
    text: str = "",
    filters: dict[str, Any] | None = None,
    like_filters: list[tuple[str, str]] | None = None,
    unified_filters: list[tuple[str, str, str]] | None = None,
    limit: int = PAGE_SIZE,
    offset: int = 0,
):
    """
    Search a catalog and return a *lazy* paginated slice.

    Parameters
    ----------
    cat : tiled catalog node
    text : str
        FullText search string.
    filters : dict or None
        Exact-match metadata filters (see build_query).
    like_filters : list of (key, pattern) or None
        Stacked Like filters (see build_query).
    unified_filters : list of (type, key, value) or None
        Unified stackable filters (see build_query).
    limit, offset : int
        Pagination window.

    Returns
    -------
    Lazy sequence-like page containing at most `limit` run nodes.
    """
    results = build_query(cat, text, filters, like_filters=like_filters,
                          unified_filters=unified_filters)

    # Some bluesky-tiled catalog adapters only allow negative slicing on the
    # catalog itself. Positive paging must be done through .values().
    return results.values()[offset : offset + limit]


def total_count(
    cat, text: str = "", filters: dict[str, Any] | None = None,
    like_filters: list[tuple[str, str]] | None = None,
    unified_filters: list[tuple[str, str, str]] | None = None,
) -> int:
    """
    Return total number of matching runs.
    Fetches only the count, not any run data.
    """
    return len(build_query(cat, text, filters, like_filters=like_filters,
                           unified_filters=unified_filters))


# --- Default size limit for distinct queries (configurable) ----------------
DISTINCT_SIZE_LIMIT = 10_000
DISTINCT_TIMEOUT = 10  # seconds – give up after this long


def distinct_values(
    cat,
    key: str,
    text: str = "",
    filters: dict[str, Any] | None = None,
    like_filters: list[tuple[str, str]] | None = None,
    unified_filters: list[tuple[str, str, str]] | None = None,
    *,
    counts: bool = False,
    size_limit: int = DISTINCT_SIZE_LIMIT,
    timeout: float = DISTINCT_TIMEOUT,
) -> list[dict] | None:
    """
    Return the unique values for *key* in the (optionally filtered) catalog.

    The ``distinct`` query requires ``start.``-prefixed keys on the SMI
    migration catalog (e.g. ``start.sample_name``).  This helper
    auto-prepends ``start.`` if the key doesn't already have a dot.

    Returns ``None`` (instead of raising) when the filtered catalog exceeds
    *size_limit* or when the query takes longer than *timeout* seconds.

    Parameters
    ----------
    key : str
        Metadata key to inspect (e.g. ``"sample_name"``).
    size_limit : int
        Maximum catalog size for which we attempt the query.  Set to 0 to
        skip the size check entirely (careful – can be very slow on large
        catalogs).
    counts : bool
        If True, each entry includes a ``"count"`` field.
    timeout : float
        Maximum seconds to wait for the distinct query.  Returns ``None``
        if exceeded.

    Returns
    -------
    list[dict]  – ``[{"value": ..., "count": ...}, ...]``  (``count`` is
    ``None`` unless *counts* is True).
    ``None`` if the catalog is too large or the query timed out.
    """
    filtered = build_query(cat, text, filters, like_filters=like_filters,
                           unified_filters=unified_filters)

    def _run():
        if size_limit and len(filtered) > size_limit:
            return None
        distinct_key = key if "." in key else f"start.{key}"
        result = filtered.distinct(distinct_key, counts=counts)
        entries = result.get("metadata", {}).get(distinct_key, [])
        return [e for e in entries if e.get("value") is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning("distinct_values(%r) timed out after %.0fs", key, timeout)
            return None


def iter_uids(results_cat):
    """Iterate over UIDs in a pre-sliced result page."""
    if hasattr(results_cat, "keys"):
        yield from results_cat.keys()
        return

    for run in results_cat:
        try:
            yield run.metadata.get("start", {}).get("uid", "?")
        except Exception:
            yield "?"


def n_pages(total: int, limit: int = PAGE_SIZE) -> int:
    """Total number of pages given a total count and page size."""
    return max(1, (total + limit - 1) // limit)


# ---------------------------------------------------------------------------
# Run-level metadata inspection  (zero data I/O)
# ---------------------------------------------------------------------------

def run_summary(run) -> dict:
    """
    Extract a lightweight summary dict from a run node.
    Reads only .metadata – no arrays are fetched.
    """
    md    = run.metadata
    start = md.get("start", {})
    stop  = md.get("stop",  {})

    t0 = start.get("time")
    if t0:
        time_str = datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M")
    else:
        time_str = "?"

    exit_status = stop.get("exit_status", "?")
    num_events  = stop.get("num_events", {})
    if isinstance(num_events, dict):
        num_events = num_events.get("primary", "?")

    try:
        streams = list(run.keys())
    except Exception:
        streams = []

    return {
        "uid":          start.get("uid", "?"),
        "scan_id":      start.get("scan_id", "?"),
        "time":         time_str,
        "plan_name":    start.get("plan_name", "?"),
        "sample_name":  start.get(
            "sample_name", start.get("sample", start.get("Sample", "?"))
        ),
        "exit_status":  exit_status,
        "num_events":   num_events,
        "data_session": start.get("data_session", start.get("proposal_id", "?")),
        "streams":      streams,
    }


def page_summaries(results_cat) -> list[dict]:
    """
    Collect run_summary() for every run in a (pre-sliced) catalog page.
    Only metadata is accessed – no data arrays.
    """
    out = []

    # Mapping-style (uid -> run) when available.
    if hasattr(results_cat, "items"):
        for uid, run in results_cat.items():
            try:
                s = run_summary(run)
            except Exception as exc:
                log.warning("summary failed for %s: %s", uid, exc)
                s = {"uid": uid, "error": str(exc)}
            out.append(s)
        return out

    # Sequence-style page (e.g. results.values()[start:stop]).
    for run in results_cat:
        uid = "?"
        try:
            s = run_summary(run)
            uid = s.get("uid", "?")
        except Exception as exc:
            log.warning("summary failed for run: %s", exc)
            s = {"uid": uid, "error": str(exc)}
        out.append(s)
    return out


def full_start_doc(run) -> dict:
    """Return the full 'start' metadata document for a run."""
    return dict(run.metadata.get("start", {}))


# ---------------------------------------------------------------------------
# Stream & field inspection  (zero data I/O)
# ---------------------------------------------------------------------------

def stream_names(run) -> list[str]:
    """List streams in a run without loading any data."""
    try:
        return list(run.keys())
    except Exception:
        return []


def stream_fields(run, stream: str = "primary") -> dict[str, tuple]:
    """
    Return {field_name: shape_tuple} for a stream without reading any arrays.
    Shape includes the event/frame dimension as axis 0.

    Uses a single bulk .read() to discover field shapes from the returned
    xarray Dataset, avoiding one HTTP round-trip per field.  Falls back to
    the slow per-field .structure() path only if bulk read fails.
    """
    try:
        node = run[stream]
    except Exception as exc:
        log.warning("stream_fields: cannot access stream %s: %s", stream, exc)
        return {}

    # Fast path: read once, inspect shapes locally.
    try:
        ds = node.read()
        fields = {}
        for key in ds:
            arr = ds[key]
            fields[key] = tuple(arr.shape) if hasattr(arr, "shape") else ()
        return fields
    except Exception:
        pass

    # Slow fallback: per-field structure queries.
    try:
        return {key: child.structure().shape for key, child in node.items()}
    except Exception as exc:
        log.warning("stream_fields failed: %s", exc)
        return {}


def stream_fields_fast(run, stream: str = "primary"):
    """
    Like stream_fields but also returns the already-read scalar DataFrame
    so callers can avoid a second .read().
    Returns (fields_dict, dataset_or_dataframe).
    If the stream cannot be read, returns ({}, None).

    Uses the tiled ``node.base`` container to read the ``internal`` table
    (scalars only) and discover image arrays separately — no bulk image
    download.
    """
    try:
        node = run[stream]
    except Exception:
        return {}, None

    # Prefer .base which exposes the raw REST structure:
    #   base['internal']  → DataFrameClient (scalars)
    #   base['<det>_image'] → ArrayClient (images)
    base = getattr(node, "base", None)
    if base is not None:
        try:
            fields = {}
            df = None
            for key, child in base.items():
                ctype = type(child).__name__
                if ctype == "DataFrameClient":
                    # Scalar table — read it (cheap, no images)
                    df = child.read()
                    for col in df.columns:
                        fields[col] = (len(df),)
                elif ctype == "ArrayClient":
                    fields[key] = child.structure().shape
                else:
                    try:
                        fields[key] = child.structure().shape
                    except Exception:
                        pass
            return fields, df
        except Exception as exc:
            log.warning("stream_fields_fast: base read failed for %s: %s", stream, exc)

    # Fallback: get shapes via structure metadata, read scalars individually
    try:
        fields = {key: child.structure().shape for key, child in node.items()}
    except Exception as exc:
        log.warning("stream_fields_fast: structure failed for %s: %s", stream, exc)
        return {}, None

    import xarray as xr
    scalar_keys = [k for k, sh in fields.items() if len(sh) < IMAGE_NDIM_THRESHOLD]
    if not scalar_keys:
        return fields, None

    arrays = {}
    for key in scalar_keys:
        try:
            arrays[key] = np.asarray(node[key].read())
        except Exception:
            pass
    ds = xr.Dataset({k: (["dim_0"], v) for k, v in arrays.items()}) if arrays else None
    return fields, ds


IMAGE_NDIM_THRESHOLD = 3  # ndim >= 3 → treat as image stack


def classify_fields(
    fields: dict[str, tuple],
) -> tuple[list[str], list[str]]:
    """
    Split a field→shape dict into (scalar_fields, image_fields).
    Images are identified by having ≥ 3 dimensions (events × rows × cols).
    """
    scalars = [k for k, sh in fields.items() if len(sh) < IMAGE_NDIM_THRESHOLD]
    images  = [k for k, sh in fields.items() if len(sh) >= IMAGE_NDIM_THRESHOLD]
    return scalars, images


def all_stream_info(run) -> dict[str, dict]:
    """
    Return {stream → {"fields": {...}, "scalars": [...], "images": [...]}}
    for all streams in a run, without loading any data.
    """
    info = {}
    for name in stream_names(run):
        fields = stream_fields(run, name)
        scalars, images = classify_fields(fields)
        info[name] = {"fields": fields, "scalars": scalars, "images": images}
    return info


def stream_info_for(run, stream: str) -> dict:
    """
    Return {"fields": {...}, "scalars": [...], "images": [...], "dataset": ds}
    for a *single* stream.  Returns the already-fetched dataset so callers
    don't need a second .read().
    """
    fields, ds = stream_fields_fast(run, stream)
    scalars, images = classify_fields(fields)
    return {"fields": fields, "scalars": scalars, "images": images, "dataset": ds}


# ---------------------------------------------------------------------------
# Lazy data access  (fetch only what you need)
# ---------------------------------------------------------------------------

def fetch_frame(
    run,
    stream: str,
    field: str,
    frame_idx: int = 0,
    _dataset=None,
) -> np.ndarray | None:
    """
    Fetch a single image frame by index.  Always returns a 2-D array.

    If _dataset is provided (from a prior .read()), extract from it directly.
    Otherwise, use the tiled node's index + .read() for a single-frame fetch.
    """
    frame = None

    # Fast path: extract from already-loaded dataset
    if _dataset is not None:
        try:
            arr = np.asarray(_dataset[field])
            if arr.ndim >= 3:
                frame = arr[frame_idx]
            else:
                frame = arr
        except Exception:
            pass

    # Tiled node path
    if frame is None:
        try:
            node = run[stream][field]
            sliced = node[frame_idx]
            if hasattr(sliced, "read"):
                frame = np.asarray(sliced.read())
            else:
                frame = np.asarray(sliced)
        except Exception as exc:
            log.warning("fetch_frame failed (%s/%s[%d]): %s", stream, field, frame_idx, exc)
            return None

    # Ensure 2-D (squeeze leading length-1 dims from e.g. (1, H, W))
    if frame is not None:
        while frame.ndim > 2 and frame.shape[0] == 1:
            frame = frame[0]
    return frame


def fetch_all_frames(
    run,
    stream: str,
    field: str,
) -> np.ndarray | None:
    """
    Fetch the full image stack for a field.  Use sparingly – prefer
    fetch_frame() for previews.
    """
    try:
        return np.asarray(run[stream][field].read())
    except Exception as exc:
        log.warning("fetch_all_frames failed (%s/%s): %s", stream, field, exc)
        return None


def fetch_scalars(
    run,
    stream: str = "primary",
    fields: list[str] | None = None,
    _dataset=None,
) -> dict[str, np.ndarray]:
    """
    Read all scalar (non-image) fields for a stream.

    Parameters
    ----------
    run : tiled run node
    stream : str
    fields : list or None
        Subset of fields to return.  None → all scalars.
    _dataset : xarray.Dataset, pandas.DataFrame, or None
        If the caller already has a read result (e.g. from
        stream_fields_fast), pass it here to avoid a second read.
    """
    import pandas as pd

    if _dataset is not None:
        ds = _dataset
    else:
        # Try the fast .base path first
        node = run[stream]
        base = getattr(node, "base", None)
        if base is not None:
            try:
                for child in base.values():
                    if type(child).__name__ == "DataFrameClient":
                        ds = child.read()
                        break
                else:
                    ds = node.read()
            except Exception as exc:
                log.warning("fetch_scalars: read failed for %s: %s", stream, exc)
                return {}
        else:
            try:
                ds = node.read()
            except Exception as exc:
                log.warning("fetch_scalars: bulk read failed for %s: %s", stream, exc)
                return {}

    result = {}
    # Handle pandas DataFrame (from .base['internal'])
    if isinstance(ds, pd.DataFrame):
        for key in ds.columns:
            if fields is not None and key not in fields:
                continue
            result[key] = np.asarray(ds[key])
        return result

    # Handle xarray Dataset or similar
    for key in ds:
        arr = ds[key]
        if not hasattr(arr, "shape"):
            continue
        if len(arr.shape) >= IMAGE_NDIM_THRESHOLD:
            continue  # skip images
        if fields is not None and key not in fields:
            continue
        result[key] = np.asarray(arr)
    return result


# ---------------------------------------------------------------------------
# Convenience: quick metadata-only table for a list of UIDs
# ---------------------------------------------------------------------------

def uid_table(cat, uids: list[str]) -> list[dict]:
    """
    Build a summary table for specific UIDs (e.g. a comparison set).
    Only metadata is read.
    """
    out = []
    for uid in uids:
        try:
            out.append(run_summary(cat[uid]))
        except Exception as exc:
            out.append({"uid": uid, "error": str(exc)})
    return out


# ---------------------------------------------------------------------------
# Common SMI metadata field hints  (for building filter UIs)
# ---------------------------------------------------------------------------

SMI_FILTER_FIELDS = {
    "Plan name":    "start.plan_name",
    "Sample name":  "start.sample_name",
    "Data session": "start.data_session",
    "Operator":     "start.operator",
    "Beamline":     "start.beamline_id",
    "Exit status":  "stop.exit_status",
}

# Sensible defaults for the plan_name dropdown at SMI
SMI_PLAN_NAMES = [
    "any",
    "run_waxs",
    "lup",
    "count",
    "scan",
    "rel_scan",
    "grid_scan",
    "outer_product_scan",
]
