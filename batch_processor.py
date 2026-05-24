"""Background queue/worker for batch-processing many scans.

Decoupled from Panel/Bokeh so it can be unit-tested without a server.
Callers wire it up by providing:

- ``process_fn(uid) -> (result, summary, params)`` — the heavy reduction
- ``add_fn(result, summary, params)`` — append to the destination
  collection (called on the worker thread; the caller is responsible
  for any cross-thread marshalling required by downstream UI updates)
- ``status_cb(snapshot)`` — invoked after every state transition with a
  picklable dict snapshot (see :meth:`BatchProcessor.snapshot`).  Fires
  on the worker thread; the caller marshals to the UI thread if needed.
- ``skip_fn(uid) -> bool`` — return ``True`` to skip a job (for example,
  when the uid is already in the destination collection)

Design notes
------------
- A bounded internal job dict keeps memory in check (``MAX_QUEUE``).
- Cancellation marks queued jobs as ``cancelled`` and drains the queue.
  A job already running is allowed to finish (the heavy reduction has
  no cancellation hooks).
- One failure does not stop subsequent jobs; per-job error messages are
  surfaced via the snapshot so the UI can render a status table.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class JobStatus:
    uid: str
    label: str = ""
    state: str = "queued"   # queued, running, done, error, skipped, cancelled
    started: float | None = None
    finished: float | None = None
    error: str = ""
    duration_s: float | None = None


# Terminal states — once set, a job is not eligible for re-queue tracking.
_TERMINAL = {"done", "error", "skipped", "cancelled"}


class BatchProcessor:
    """Bounded thread-pool queue runner for per-scan processing."""

    MAX_QUEUE = 2000  # safety cap on total job count

    def __init__(
        self,
        process_fn: Callable[[str], tuple],
        add_fn: Callable[[object, dict, dict], None],
        status_cb: Callable[[dict], None] | None = None,
        skip_fn: Callable[[str], bool] | None = None,
        max_workers: int = 1,
    ):
        self._process_fn = process_fn
        self._add_fn = add_fn
        self._status_cb = status_cb or (lambda _snap: None)
        self._skip_fn = skip_fn or (lambda _uid: False)
        self._max_workers = max(1, int(max_workers))

        self._queue: queue.Queue = queue.Queue()
        self._jobs: dict[str, JobStatus] = {}   # insertion-ordered
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._workers: list[threading.Thread] = []
        self._alive_workers = 0
        self._running = False

    # -- inspection ------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a thread-safe summary of all jobs and counters."""
        with self._lock:
            jobs = [asdict(j) for j in self._jobs.values()]
        states = {
            "queued": 0, "running": 0, "done": 0,
            "error": 0, "skipped": 0, "cancelled": 0,
        }
        for j in jobs:
            states[j["state"]] = states.get(j["state"], 0) + 1
        return {
            "running": self._running,
            "cancel_requested": self._cancel.is_set(),
            "total": len(jobs),
            "states": states,
            "jobs": jobs,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    # -- control ---------------------------------------------------------

    def enqueue(self, items: list[tuple[str, str]]) -> int:
        """Add ``(uid, label)`` jobs.  Returns the number actually added.

        Skips uids already present in a non-terminal state so a user can
        safely click "queue page" twice in a row.  Caps the total job
        count at :attr:`MAX_QUEUE`.
        """
        added = 0
        with self._lock:
            for uid, label in items:
                existing = self._jobs.get(uid)
                if existing is not None and existing.state not in _TERMINAL:
                    continue
                if len(self._jobs) >= self.MAX_QUEUE and uid not in self._jobs:
                    break
                self._jobs[uid] = JobStatus(uid=uid, label=label)
                self._queue.put(uid)
                added += 1
        if added:
            self._fire(force=True)
        return added

    def start(self) -> None:
        """Spawn worker threads if not already running."""
        if self._running:
            return
        self._running = True
        self._cancel.clear()
        self._workers = []
        self._alive_workers = self._max_workers
        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"batch-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        self._fire(force=True)

    def cancel(self) -> None:
        """Drain pending queue and request workers to stop after current job."""
        self._cancel.set()
        with self._lock:
            for j in self._jobs.values():
                if j.state == "queued":
                    j.state = "cancelled"
        # Drain queue so workers exit promptly.
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass
        self._fire(force=True)

    def clear_terminal(self) -> None:
        """Forget jobs in terminal states (done/error/skipped/cancelled)."""
        with self._lock:
            self._jobs = {
                uid: j for uid, j in self._jobs.items()
                if j.state not in _TERMINAL
            }
        self._fire()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for all workers to exit.  Returns True if all stopped."""
        end = None if timeout is None else time.monotonic() + timeout
        for w in list(self._workers):
            t = None if end is None else max(0.0, end - time.monotonic())
            w.join(timeout=t)
            if w.is_alive():
                return False
        return True

    # -- internals -------------------------------------------------------

    _FIRE_INTERVAL = 1.0  # min seconds between status callbacks

    def _fire(self, force: bool = False) -> None:
        """Invoke status_cb with a snapshot, throttled to avoid flooding UI.

        When ``force=True`` (used for terminal state transitions like
        batch-complete or cancel), the callback fires unconditionally.
        """
        now = time.monotonic()
        if not force:
            last = getattr(self, "_last_fire", 0.0)
            if now - last < self._FIRE_INTERVAL:
                return
        self._last_fire = now
        try:
            self._status_cb(self.snapshot())
        except Exception:
            log.exception("batch: status_cb raised")

    def _worker_loop(self) -> None:
        try:
            while not self._cancel.is_set():
                try:
                    uid = self._queue.get(timeout=0.2)
                except queue.Empty:
                    # Idle: if no peer worker is running and queue is empty,
                    # this worker can exit (the pool dissolves naturally).
                    with self._lock:
                        any_busy = any(
                            j.state == "running"
                            for j in self._jobs.values()
                        )
                    if not any_busy and self._queue.empty():
                        break
                    continue
                self._run_one(uid)
                self._queue.task_done()
        finally:
            # Use an atomic counter to detect the last worker exiting.
            with self._lock:
                self._alive_workers -= 1
                is_last = self._alive_workers == 0
            if is_last:
                self._running = False
                self._fire(force=True)

    def _run_one(self, uid: str) -> None:
        if self._cancel.is_set():
            with self._lock:
                j = self._jobs.get(uid)
                if j is not None and j.state == "queued":
                    j.state = "cancelled"
            self._fire()
            return

        if self._skip_fn(uid):
            with self._lock:
                j = self._jobs.get(uid)
                if j is not None:
                    j.state = "skipped"
                    j.finished = time.time()
            self._fire()
            return

        with self._lock:
            j = self._jobs[uid]
            j.state = "running"
            j.started = time.time()
        self._fire()

        t0 = time.perf_counter()

        try:
            result, summary, params = self._process_fn(uid)
        except Exception as exc:
            log.exception("batch: process_fn failed for %s", uid)
            self._mark_error(uid, f"{type(exc).__name__}: {exc}", t0)
            return

        try:
            self._add_fn(result, summary, params)
        except Exception as exc:
            log.exception("batch: add_fn failed for %s", uid)
            self._mark_error(uid, f"add failed: {exc}", t0)
            return
        finally:
            # Immediately release references to large objects so they can
            # be collected even if this frame is kept alive briefly.
            del result, summary, params

        with self._lock:
            j = self._jobs[uid]
            j.state = "done"
            j.finished = time.time()
            j.duration_s = time.perf_counter() - t0
        self._fire()

        # Prune completed jobs to prevent unbounded dict growth.
        self._prune_terminal_jobs()

    # Maximum number of terminal (done/error/skipped/cancelled) jobs to
    # keep in memory.  Matches MAX_QUEUE so all jobs in a single batch
    # stay visible in the UI.
    _MAX_TERMINAL_KEPT = 2000

    def _prune_terminal_jobs(self) -> None:
        """Drop old terminal jobs so _jobs doesn't grow unboundedly."""
        with self._lock:
            terminal = [
                uid for uid, j in self._jobs.items()
                if j.state in _TERMINAL
            ]
            excess = len(terminal) - self._MAX_TERMINAL_KEPT
            if excess > 0:
                for uid in terminal[:excess]:
                    del self._jobs[uid]

    def _mark_error(self, uid: str, msg: str, t0: float) -> None:
        with self._lock:
            j = self._jobs.get(uid)
            if j is None:
                return
            j.state = "error"
            j.error = msg
            j.finished = time.time()
            j.duration_s = time.perf_counter() - t0
        self._fire()
