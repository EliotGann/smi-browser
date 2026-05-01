"""Tests for live_stream.LiveStreamManager.

Focus on the state machine and dispatch wiring; the actual tiled
streaming protocol is not exercised (no Redis-backed server in CI).
We replace tiled Subscription objects with simple fakes that record
calls and let the test fire updates manually.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from live_stream import LiveStreamManager


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRegistry:
    """Stand-in for tiled.client.stream.CallbackRegistry.

    Holds *strong* references (the real one uses weakrefs); the manager
    keeps its own strong refs so this is fine for testing.
    """

    def __init__(self):
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)
        return self

    def fire(self, update):
        for cb in list(self.callbacks):
            cb(update)


class FakeSubscription:
    def __init__(self, kind="container"):
        self.kind = kind
        if kind == "container":
            self.child_created = FakeRegistry()
        else:
            self.new_data = FakeRegistry()
        self.started = False
        self.start_arg = None
        self.disconnected = False

    def start_in_thread(self, start=None):
        self.started = True
        self.start_arg = start

    def disconnect(self):
        self.disconnected = True


class FakeArrayNode:
    def __init__(self, shape=(0, 100, 100)):
        self._shape = shape
        self.sub = FakeSubscription("array")

    def structure(self):
        return SimpleNamespace(shape=self._shape)

    def subscribe(self):
        return self.sub


class FakeTableNode:
    def __init__(self):
        self.sub = FakeSubscription("table")

    def subscribe(self):
        return self.sub


class FakeContainer:
    """A minimal dict-like tiled container that returns child nodes."""

    def __init__(self, children: dict):
        self._children = children
        self.sub = FakeSubscription("container")

    def __getitem__(self, key):
        if key not in self._children:
            raise KeyError(key)
        return self._children[key]

    def __iter__(self):
        return iter(self._children)

    def subscribe(self):
        return self.sub


def make_run(*, with_table=True, image_fields=("pil1m_image", "rayonix_image")):
    """Build a fake bluesky run laid out as primary/internal/events + primary/external/<field>."""
    external_children = {f: FakeArrayNode() for f in image_fields}
    external = FakeContainer(external_children)
    internal_children = {}
    if with_table:
        internal_children["events"] = FakeTableNode()
    internal = FakeContainer(internal_children)
    primary = FakeContainer({"external": external, "internal": internal})
    run = FakeContainer({"primary": primary})
    return run, external_children, internal_children


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _collect():
    """Return (sink_dict, callbacks) where callbacks append to sink_dict."""
    sink = {"new_run": [], "primary": [], "frame": [], "errors": []}
    callbacks = dict(
        on_new_run=lambda uid: sink["new_run"].append(uid),
        on_primary_extended=lambda uid: sink["primary"].append(uid),
        on_frame_extended=lambda uid, field, n: sink["frame"].append((uid, field, n)),
        on_error=lambda stage, exc: sink["errors"].append((stage, exc)),
    )
    return sink, callbacks


def test_start_subscribes_and_dispatches_new_run():
    sink, cb = _collect()
    cat = FakeContainer({})
    mgr = LiveStreamManager(cat, **cb)
    mgr.start()

    assert cat.sub.started
    assert len(cat.sub.child_created.callbacks) == 1

    cat.sub.child_created.fire(SimpleNamespace(key="abc-123"))
    assert sink["new_run"] == ["abc-123"]


def test_start_is_idempotent():
    _, cb = _collect()
    cat = FakeContainer({})
    mgr = LiveStreamManager(cat, **cb)
    mgr.start()
    first_sub = cat.sub
    mgr.start()  # no-op
    assert mgr._container_sub is first_sub


def test_watch_run_subscribes_table_and_arrays():
    sink, cb = _collect()
    cat = FakeContainer({})
    run, externals, internals = make_run()
    mgr = LiveStreamManager(cat, **cb)
    mgr.watch_run("uid-1", run)

    assert mgr.watched_uid == "uid-1"
    assert internals["events"].sub.started
    assert internals["events"].sub.start_arg == 0
    for node in externals.values():
        assert node.sub.started

    # Fire a primary update → callback gets uid.
    internals["events"].sub.new_data.fire(SimpleNamespace())
    assert sink["primary"] == ["uid-1"]

    # Fire a frame update → callback gets (uid, field, n_total).
    target_field = next(iter(externals))
    externals[target_field].sub.new_data.fire(SimpleNamespace(shape=(7, 100, 100)))
    assert sink["frame"] == [("uid-1", target_field, 7)]


def test_unwatch_run_disconnects_and_drops_stale_events():
    sink, cb = _collect()
    cat = FakeContainer({})
    run, externals, internals = make_run()
    mgr = LiveStreamManager(cat, **cb)
    mgr.watch_run("uid-1", run)

    table_sub = internals["events"].sub
    image_subs = [n.sub for n in externals.values()]

    mgr.unwatch_run()
    assert mgr.watched_uid is None
    assert table_sub.disconnected
    assert all(s.disconnected for s in image_subs)

    # Late-arriving update from the old run must NOT reach the user callback.
    table_sub.new_data.fire(SimpleNamespace())
    assert sink["primary"] == []


def test_watch_run_replaces_previous():
    _, cb = _collect()
    cat = FakeContainer({})
    run1, _, internals1 = make_run()
    run2, _, internals2 = make_run(image_fields=("pil1m_image",))

    mgr = LiveStreamManager(cat, **cb)
    mgr.watch_run("uid-1", run1)
    mgr.watch_run("uid-2", run2)

    assert mgr.watched_uid == "uid-2"
    assert internals1["events"].sub.disconnected
    assert not internals2["events"].sub.disconnected


def test_stale_uid_filter_after_switch():
    sink, cb = _collect()
    cat = FakeContainer({})
    run1, externals1, internals1 = make_run()
    run2, _, _ = make_run()
    mgr = LiveStreamManager(cat, **cb)
    mgr.watch_run("uid-1", run1)

    # Capture the old sub *before* switching, then switch.
    old_table_sub = internals1["events"].sub
    mgr.watch_run("uid-2", run2)

    # Even if the old sub somehow fires after disconnect (race), the
    # uid filter inside the manager drops it.
    old_table_sub.new_data.fire(SimpleNamespace())
    assert sink["primary"] == []


def test_stop_disconnects_everything():
    _, cb = _collect()
    cat = FakeContainer({})
    run, _, internals = make_run()
    mgr = LiveStreamManager(cat, **cb)
    mgr.start()
    mgr.watch_run("uid-1", run)

    mgr.stop()
    assert cat.sub.disconnected
    assert internals["events"].sub.disconnected
    assert mgr.watched_uid is None


def test_dispatcher_is_used_when_provided():
    sink, cb = _collect()
    cat = FakeContainer({})
    queue = []
    mgr = LiveStreamManager(cat, dispatcher=queue.append, **cb)
    mgr.start()

    cat.sub.child_created.fire(SimpleNamespace(key="abc"))
    # Update was queued, not dispatched inline.
    assert sink["new_run"] == []
    assert len(queue) == 1
    queue[0]()  # simulate UI thread running it
    assert sink["new_run"] == ["abc"]


def test_subscribe_failure_reports_via_on_error():
    sink, cb = _collect()

    class BadCat(FakeContainer):
        def subscribe(self):
            raise RuntimeError("redis down")

    mgr = LiveStreamManager(BadCat({}), **cb)
    mgr.start()
    assert sink["errors"]
    stage, exc = sink["errors"][0]
    assert stage == "catalog.subscribe"
    assert isinstance(exc, RuntimeError)


def test_resolve_primary_table_falls_back_to_legacy_layout():
    table = FakeTableNode()
    primary = FakeContainer({"data": table})
    run = FakeContainer({"primary": primary})
    assert LiveStreamManager._resolve_primary_table(run) is table


def test_resolve_primary_table_returns_none_when_missing():
    primary = FakeContainer({})
    run = FakeContainer({"primary": primary})
    assert LiveStreamManager._resolve_primary_table(run) is None


def test_resolve_image_arrays_skips_non_image_nodes():
    img = FakeArrayNode(shape=(5, 100, 100))
    scalar = FakeArrayNode(shape=(5,))
    external = FakeContainer({"pil1m_image": img, "scalar_field": scalar})
    primary = FakeContainer({"external": external})
    run = FakeContainer({"primary": primary})
    out = LiveStreamManager._resolve_image_arrays(run)
    assert "pil1m_image" in out
    assert "scalar_field" not in out


def test_callback_exception_does_not_break_manager():
    sink, _ = _collect()

    def boom(uid):
        raise ValueError("user code crashed")

    cb = dict(
        on_new_run=boom,
        on_primary_extended=lambda uid: sink["primary"].append(uid),
        on_frame_extended=lambda *_a: None,
        on_error=lambda stage, exc: sink["errors"].append((stage, exc)),
    )
    cat = FakeContainer({})
    mgr = LiveStreamManager(cat, **cb)
    mgr.start()
    cat.sub.child_created.fire(SimpleNamespace(key="abc"))

    assert sink["errors"]
    assert sink["errors"][0][0] == "on_new_run"
    # Manager still functional after a callback exception.
    assert mgr._started
