"""
live_stream.py — Tiled streaming subscriptions for live SMI runs.

Wraps tiled's experimental streaming API (Subscription / CallbackRegistry)
into a single ``LiveStreamManager`` that the Panel app can use to react to:

  * new runs appearing in the catalog               → on_new_run(uid)
  * the active run's primary table being extended   → on_primary_extended(uid)
  * an image array gaining new frames               → on_frame_extended(uid, field, n_total)

Design notes
------------
* CallbackRegistry holds *weak* references to callbacks, so we keep hard
  references on the manager instance to prevent silent GC of our handlers.
* Tiled subscription callbacks fire on a background ThreadPoolExecutor.
  The caller passes a ``dispatcher`` (e.g. ``doc.add_next_tick_callback``)
  that is invoked with a zero-arg fn; this lets the UI marshal updates
  onto Bokeh's document thread.
* ``watch_run`` / ``unwatch_run`` / ``stop`` are protected by a single
  lock so callbacks for an old uid can't race a switch to a new uid.
  Each per-run callback also re-checks ``self._watched_uid`` and drops
  stale events.
* Any failure to subscribe to an individual stream is logged and reported
  via ``on_error``; the manager keeps running so partial functionality
  (e.g. images stream OK but primary table doesn't) is preserved.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

# Default no-op dispatcher: run the callback inline on the streaming thread.
# Real apps should pass ``doc.add_next_tick_callback`` so UI mutations land
# on the Bokeh document thread.
def _inline_dispatch(fn: Callable[[], None]) -> None:
    fn()


class LiveStreamManager:
    """Manage tiled streaming subscriptions for the live-mode UI.

    Parameters
    ----------
    catalog
        A tiled container client (the ``smi/migration`` node) that exposes
        ``.subscribe()`` returning a ``ContainerSubscription``.
    on_new_run : callable(uid: str) -> None
        Fired when a new child appears under ``catalog`` (a new bluesky run).
    on_primary_extended : callable(uid: str) -> None
        Fired when the active run's primary table grows.
    on_frame_extended : callable(uid: str, field: str, n_total: int) -> None
        Fired when an image array on the active run gains new frames.
        ``n_total`` is the new leading-dim length.
    on_error : callable(stage: str, exc: Exception) -> None, optional
        Reports subscription / dispatch errors. Default: log only.
    dispatcher : callable(fn) -> None, optional
        Schedules ``fn`` (a zero-arg callable) onto the UI thread.
        Default: invoke inline.
    """

    def __init__(
        self,
        catalog,
        *,
        on_new_run: Callable[[str], None],
        on_primary_extended: Callable[[str], None],
        on_frame_extended: Callable[[str, str, int], None],
        on_error: Callable[[str, Exception], None] | None = None,
        dispatcher: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._on_new_run = on_new_run
        self._on_primary_extended = on_primary_extended
        self._on_frame_extended = on_frame_extended
        self._on_error = on_error or (lambda stage, exc: log.warning("live %s: %s", stage, exc))
        self._dispatch = dispatcher or _inline_dispatch

        self._lock = threading.RLock()
        self._container_sub = None
        # Hard refs to callbacks (CallbackRegistry uses weakrefs!).
        self._container_cb = None
        self._run_subs: list = []          # list of subscription objects
        self._run_callbacks: list = []     # parallel list of hard-ref callbacks
        self._watched_uid: str | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def watched_uid(self) -> str | None:
        return self._watched_uid

    def start(self) -> None:
        """Subscribe to the catalog for new-run notifications."""
        with self._lock:
            if self._started:
                return
            try:
                sub = self._catalog.subscribe()
            except Exception as exc:
                self._on_error("catalog.subscribe", exc)
                return

            def _cb(update):
                # update.key is the new run's uid in bluesky-tiled-plugins.
                uid = getattr(update, "key", None)
                if not uid:
                    return
                self._dispatch(lambda u=uid: self._safe_on_new_run(u))

            self._container_sub = sub
            self._container_cb = _cb
            sub.child_created.add_callback(_cb)
            try:
                sub.start_in_thread()
            except Exception as exc:
                self._on_error("catalog.start", exc)
                return
            self._started = True

    def watch_run(self, uid: str, run) -> None:
        """Switch the active run being watched.

        ``run`` is the resolved tiled node for ``uid`` (the caller already
        navigated to it, so we don't need to look it up again).
        """
        with self._lock:
            self._unwatch_run_locked()
            self._watched_uid = uid

            # --- Primary table subscription ---------------------------------
            primary_node = self._resolve_primary_table(run)
            if primary_node is not None:
                self._subscribe_table(primary_node, uid, kind="primary")

            # --- Image array subscriptions ---------------------------------
            for field, node in self._resolve_image_arrays(run).items():
                self._subscribe_array(node, uid, field)

    def unwatch_run(self) -> None:
        """Disconnect all per-run subscriptions."""
        with self._lock:
            self._unwatch_run_locked()

    def stop(self) -> None:
        """Disconnect everything (container + per-run)."""
        with self._lock:
            self._unwatch_run_locked()
            sub = self._container_sub
            self._container_sub = None
            self._container_cb = None
            self._started = False
        if sub is not None:
            try:
                sub.disconnect()
            except Exception as exc:
                self._on_error("catalog.disconnect", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unwatch_run_locked(self) -> None:
        subs = self._run_subs
        self._run_subs = []
        self._run_callbacks = []
        self._watched_uid = None
        for s in subs:
            try:
                s.disconnect()
            except Exception as exc:
                self._on_error("run.disconnect", exc)

    def _safe_on_new_run(self, uid: str) -> None:
        try:
            self._on_new_run(uid)
        except Exception as exc:
            self._on_error("on_new_run", exc)

    def _safe_on_primary(self, uid: str) -> None:
        # Drop stale events from a previously-watched run.
        if uid != self._watched_uid:
            return
        try:
            self._on_primary_extended(uid)
        except Exception as exc:
            self._on_error("on_primary_extended", exc)

    def _safe_on_frame(self, uid: str, field: str, n_total: int) -> None:
        if uid != self._watched_uid:
            return
        try:
            self._on_frame_extended(uid, field, n_total)
        except Exception as exc:
            self._on_error("on_frame_extended", exc)

    def _subscribe_table(self, node, uid: str, *, kind: str) -> None:
        try:
            sub = node.subscribe()
        except Exception as exc:
            self._on_error(f"{kind}.subscribe", exc)
            return

        def _cb(update, _uid=uid):
            self._dispatch(lambda: self._safe_on_primary(_uid))

        sub.new_data.add_callback(_cb)
        self._run_subs.append(sub)
        self._run_callbacks.append(_cb)
        try:
            # start=0 → catch up on any data already buffered for this stream
            sub.start_in_thread(0)
        except Exception as exc:
            self._on_error(f"{kind}.start", exc)

    def _subscribe_array(self, node, uid: str, field: str) -> None:
        try:
            sub = node.subscribe()
        except Exception as exc:
            self._on_error(f"image[{field}].subscribe", exc)
            return

        def _cb(update, _uid=uid, _field=field):
            shape = getattr(update, "shape", None)
            n = int(shape[0]) if shape else 0
            self._dispatch(lambda: self._safe_on_frame(_uid, _field, n))

        sub.new_data.add_callback(_cb)
        self._run_subs.append(sub)
        self._run_callbacks.append(_cb)
        try:
            sub.start_in_thread(0)
        except Exception as exc:
            self._on_error(f"image[{field}].start", exc)

    # ------------------------------------------------------------------
    # Resolvers — locate the streamable nodes inside a bluesky run
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_primary_table(run) -> Any | None:
        """Find the ``primary`` stream's table node, or None if absent.

        bluesky-tiled-plugins lays runs out as
        ``run["primary"]["internal"]["events"]`` (a table) with image arrays
        at ``run["primary"]["external"][<field>]``.  Older layouts used
        ``run["primary"]["data"]``.  We try both.
        """
        try:
            primary = run["primary"]
        except Exception:
            return None
        for path in (("internal", "events"), ("data",), ("internal",)):
            node = primary
            ok = True
            for part in path:
                try:
                    node = node[part]
                except Exception:
                    ok = False
                    break
            if ok and node is not primary and hasattr(node, "subscribe"):
                return node
        return None

    @staticmethod
    def _resolve_image_arrays(run) -> dict[str, Any]:
        """Return ``{field: array_node}`` for image fields under primary.

        Tries the modern ``primary/external/<field>`` layout first, then
        falls back to ``primary/<field>`` for legacy layouts.  Only nodes
        exposing ``.subscribe`` and a 3-D-ish ``structure().shape`` are kept.
        """
        try:
            primary = run["primary"]
        except Exception:
            return {}

        candidates: dict[str, Any] = {}
        for container_key in ("external", None):
            try:
                node = primary[container_key] if container_key else primary
            except Exception:
                continue
            try:
                keys = list(node)
            except Exception:
                continue
            for key in keys:
                if key in candidates:
                    continue
                try:
                    child = node[key]
                except Exception:
                    continue
                if not hasattr(child, "subscribe"):
                    continue
                # Only keep image-shaped arrays (>=3 dims).
                shape = _safe_shape(child)
                if shape is None or len(shape) < 3:
                    continue
                candidates[key] = child
        return candidates


def _safe_shape(node) -> tuple[int, ...] | None:
    """Best-effort read of an array node's shape without triggering I/O."""
    try:
        struct = node.structure()
        return tuple(struct.shape)
    except Exception:
        pass
    try:
        return tuple(node.shape)
    except Exception:
        return None
