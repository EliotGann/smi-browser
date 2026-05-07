"""Tests for batch_processor.BatchProcessor.

Exercises the queue/worker state machine without touching tiled or
PyHyperScattering.  ``process_fn`` is replaced with a controllable fake
that lets us release jobs one at a time, drive errors, and inspect the
snapshot the UI would receive.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from batch_processor import BatchProcessor, JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_until(pred, timeout=2.0, interval=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


class _FakeProcess:
    """process_fn factory with controllable per-uid behaviour."""

    def __init__(self):
        self.calls: list[str] = []
        self.gate = threading.Event()
        self.gate.set()             # default: do not block
        self.errors: dict[str, Exception] = {}
        self.lock = threading.Lock()

    def __call__(self, uid: str):
        with self.lock:
            self.calls.append(uid)
        # Allow tests to pause inside the worker.
        self.gate.wait(timeout=5.0)
        if uid in self.errors:
            raise self.errors[uid]
        result = SimpleNamespace(uid=uid)
        summary = {"sample_name": f"s-{uid}"}
        params = {"uid": uid}
        return result, summary, params


class _Recorder:
    """Captures add_fn calls and status snapshots."""

    def __init__(self):
        self.added: list[str] = []
        self.snapshots: list[dict] = []
        self.add_lock = threading.Lock()

    def add(self, result, summary, params):
        with self.add_lock:
            self.added.append(result.uid)

    def status(self, snap):
        # Lists are append-thread-safe in CPython; copy snapshot first.
        self.snapshots.append(snap)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_processes_all_queued_jobs_in_order():
    proc = _FakeProcess()
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)

    bp.enqueue([("a", "A"), ("b", "B"), ("c", "C")])
    bp.start()
    assert bp.join(timeout=3.0)

    assert proc.calls == ["a", "b", "c"]
    assert rec.added == ["a", "b", "c"]
    snap = bp.snapshot()
    assert snap["states"]["done"] == 3
    assert snap["running"] is False


def test_skip_fn_marks_skipped_and_does_not_call_processor():
    proc = _FakeProcess()
    rec = _Recorder()
    bp = BatchProcessor(
        proc, rec.add, status_cb=rec.status,
        skip_fn=lambda uid: uid == "skip-me",
        max_workers=1,
    )
    bp.enqueue([("a", ""), ("skip-me", ""), ("b", "")])
    bp.start()
    assert bp.join(timeout=3.0)

    assert proc.calls == ["a", "b"]
    assert rec.added == ["a", "b"]
    snap = bp.snapshot()
    assert snap["states"]["skipped"] == 1
    assert snap["states"]["done"] == 2


def test_one_failing_job_does_not_stop_subsequent_jobs():
    proc = _FakeProcess()
    proc.errors["bad"] = RuntimeError("kaboom")
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)
    bp.enqueue([("a", ""), ("bad", ""), ("c", "")])
    bp.start()
    assert bp.join(timeout=3.0)

    assert rec.added == ["a", "c"]
    snap = bp.snapshot()
    assert snap["states"]["done"] == 2
    assert snap["states"]["error"] == 1
    bad = next(j for j in snap["jobs"] if j["uid"] == "bad")
    assert "kaboom" in bad["error"]


def test_cancel_marks_queued_jobs_cancelled():
    proc = _FakeProcess()
    proc.gate.clear()  # block in process_fn
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)
    bp.enqueue([("a", ""), ("b", ""), ("c", "")])
    bp.start()

    # Wait until first job has entered the worker.
    assert _wait_until(lambda: proc.calls == ["a"])

    bp.cancel()
    proc.gate.set()  # let the running job finish (or fail-through)

    assert bp.join(timeout=3.0)
    snap = bp.snapshot()
    states = snap["states"]
    # 'a' was already running — it can finish (done) under cancel because we
    # do not interrupt mid-reduction.  'b' and 'c' must be marked cancelled.
    assert states["cancelled"] == 2
    assert states["done"] + states["error"] == 1


def test_enqueue_dedupes_active_uids():
    proc = _FakeProcess()
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)

    n1 = bp.enqueue([("a", ""), ("b", "")])
    n2 = bp.enqueue([("a", ""), ("c", "")])  # 'a' already queued
    assert n1 == 2
    assert n2 == 1

    bp.start()
    assert bp.join(timeout=3.0)
    assert sorted(proc.calls) == ["a", "b", "c"]


def test_status_cb_fires_on_state_changes():
    proc = _FakeProcess()
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)
    bp.enqueue([("a", "")])
    bp.start()
    assert bp.join(timeout=3.0)

    states_seen = {
        s["uid"]: s["state"]
        for snap in rec.snapshots
        for s in snap["jobs"]
    }
    # last seen state must be terminal
    assert states_seen["a"] == "done"
    # at least: enqueue snapshot, running snapshot, done snapshot, exit snapshot
    assert len(rec.snapshots) >= 3


def test_max_queue_cap_respected(monkeypatch):
    proc = _FakeProcess()
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)
    monkeypatch.setattr(bp, "MAX_QUEUE", 3, raising=False)

    items = [(f"u{i}", "") for i in range(10)]
    added = bp.enqueue(items)
    assert added == 3
    assert len(bp) == 3


def test_clear_terminal_removes_done_jobs():
    proc = _FakeProcess()
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=1)
    bp.enqueue([("a", ""), ("b", "")])
    bp.start()
    assert bp.join(timeout=3.0)
    assert len(bp) == 2
    bp.clear_terminal()
    assert len(bp) == 0


def test_two_workers_run_concurrently():
    proc = _FakeProcess()
    proc.gate.clear()
    rec = _Recorder()
    bp = BatchProcessor(proc, rec.add, status_cb=rec.status, max_workers=2)
    bp.enqueue([("a", ""), ("b", ""), ("c", ""), ("d", "")])
    bp.start()

    # Wait until two jobs are in-flight simultaneously.
    assert _wait_until(lambda: len(proc.calls) >= 2, timeout=2.0)

    proc.gate.set()
    assert bp.join(timeout=3.0)
    snap = bp.snapshot()
    assert snap["states"]["done"] == 4
