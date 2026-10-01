"""Device-code protocol, token lifecycle, and document-thread UI publication."""
from concurrent.futures import CancelledError
from types import SimpleNamespace
from urllib.parse import parse_qs
from unittest.mock import Mock
import queue
import threading

import httpx
import pytest

from smi_browser import auth


def provider(**overrides):
    data = dict(provider="pam", mode="external", extra_scopes=["profile", "api://tiled/access_as_user"],
                links={"auth_endpoint": "https://idp.test/devicecode", "client_id": "test-client",
                       "token_endpoint": "https://idp.test/token"})
    data.update(overrides)
    return SimpleNamespace(**data)


@pytest.fixture
def setup_login(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock[0])
    real_client = httpx.Client
    context = SimpleNamespace(
        server_info=SimpleNamespace(authentication=SimpleNamespace(
            providers=[provider()], links=SimpleNamespace(whoami="https://tiled.test/whoami"))),
        close=Mock(), api_key="old-api-key", configure_auth=Mock(),
    )
    context.http_client = real_client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"identities": [{"id": "scientist"}]})))
    monkeypatch.setattr(auth, "_context", lambda _: context)
    requests = []
    responses = [httpx.Response(200, json={
        "verification_uri": "https://idp.test/verify", "user_code": "ABCD-EFGH",
        "device_code": "private-device-code", "expires_in": 900, "interval": 5,
    })]

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    client = real_client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(auth.httpx, "Client", lambda **_: client)

    class Cancel:
        cancelled = False
        waits = []

        def is_set(self):
            return self.cancelled

        def wait(self, seconds):
            self.waits.append(seconds)
            clock[0] += seconds
            return self.cancelled

    yield SimpleNamespace(context=context, client=client, responses=responses,
                          requests=requests, cancel=Cancel(), clock=clock)
    client.close()
    context.http_client.close()


def test_discovery_uses_external_mode_not_legacy_pam_name(setup_login):
    env = setup_login
    login = auth.begin_login("https://tiled.test")
    request = env.requests[0]
    form = parse_qs(request.content.decode())
    assert form["client_id"] == ["test-client"]
    assert set(form["scope"][0].split()) == {"openid", "offline_access", "profile", "api://tiled/access_as_user"}
    assert "authorization" not in request.headers
    assert "cookie" not in request.headers
    assert login.user_code == "ABCD-EFGH"
    assert "private-device-code" not in repr(login)
    env.context.configure_auth.assert_not_called()


@pytest.mark.parametrize("key", ["verification_uri_complete", "verification_url"])
def test_microsoft_verification_url_variants(setup_login, key):
    env = setup_login
    data = env.responses[0].json()
    data[key] = data.pop("verification_uri")
    env.responses[0] = httpx.Response(200, json=data)
    assert auth.begin_login("uri").verification_uri == "https://idp.test/verify"


def test_pending_slowdown_success_validates_before_saving(setup_login):
    env = setup_login
    login = auth.begin_login("uri")
    env.responses.extend([
        httpx.Response(400, json={"error": "authorization_pending"}),
        httpx.Response(400, json={"error": "slow_down"}),
        httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh", "id_token": "id"}),
    ])
    tokens, user = auth.poll_login(login, env.cancel)
    assert env.cancel.waits == [5, 5, 10]
    assert user == "scientist"
    env.context.configure_auth.assert_not_called()
    assert parse_qs(env.requests[-1].content.decode()) == {
        "client_id": ["test-client"], "device_code": ["private-device-code"],
        "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
    }
    auth.remember_login(login, tokens)
    env.context.configure_auth.assert_called_once_with(tokens, remember_me=True)
    assert env.context.api_key is None
    assert env.context.client_id == "test-client"
    assert "offline_access" in env.context.scopes


@pytest.mark.parametrize("error,match", [
    ("authorization_declined", "declined"), ("access_denied", "declined"),
    ("expired_token", "expired"), ("invalid_client", "failed"),
])
def test_terminal_errors_do_not_leak_response_or_save_tokens(setup_login, error, match):
    env = setup_login
    login = auth.begin_login("uri")
    env.responses.append(httpx.Response(400, json={"error": error, "error_description": "sensitive-response"}))
    with pytest.raises(auth.LoginError, match=match) as exc:
        auth.poll_login(login, env.cancel)
    assert "sensitive-response" not in str(exc.value)
    env.context.configure_auth.assert_not_called()


def test_cancel_and_deadline_stop_polling(setup_login):
    env = setup_login
    login = auth.begin_login("uri")
    env.cancel.cancelled = True
    with pytest.raises(CancelledError):
        auth.poll_login(login, env.cancel)
    assert len(env.requests) == 1
    env.cancel.cancelled = False
    env.clock[0] = login.deadline
    with pytest.raises(auth.LoginError, match="expired"):
        auth.poll_login(login, env.cancel)
    assert len(env.requests) == 1


def test_tiled_rejection_does_not_save_tokens(setup_login, monkeypatch):
    env = setup_login
    login = auth.begin_login("uri")
    env.responses.append(httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh"}))
    monkeypatch.setattr(env.context.http_client, "get", lambda *a, **kw: httpx.Response(401))
    with pytest.raises(auth.LoginError, match="Tiled did not accept"):
        auth.poll_login(login, env.cancel)
    env.context.configure_auth.assert_not_called()


def test_invalid_provider_or_link_closes_clients(setup_login):
    env = setup_login
    env.context.server_info.authentication.providers = [provider(mode="internal")]
    with pytest.raises(auth.LoginError, match="external"):
        auth.begin_login("uri")
    env.context.close.assert_called_once()
    assert env.client.is_closed


def test_real_tiled_token_cache_refresh_and_logout(tmp_path, monkeypatch):
    """Exercise the installed TiledAuth, including Entra refresh form/scopes."""
    from tiled.client.context import Context
    context = Context.__new__(Context)
    context.api_uri = httpx.URL("https://tiled.test/api/v1/")
    context._token_cache = tmp_path
    context.client_id = "test-client"
    context.scopes = "openid offline_access profile"
    context.server_info = SimpleNamespace(authentication=SimpleNamespace(links=SimpleNamespace(
        refresh_session="https://idp.test/token", whoami="https://tiled.test/whoami",
        logout="https://idp.test/logout")))
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "idp.test" and request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh"})
        if request.url.path == "/logout":
            return httpx.Response(500)  # local logout must work even if IdP logout fails
        if request.headers.get("authorization") == "Bearer new-access":
            return httpx.Response(200, json={"identities": [{"id": "scientist"}]})
        return httpx.Response(401)

    context.http_client = httpx.Client(transport=httpx.MockTransport(handle), cookies={"tiled_csrf": "csrf"})
    context.configure_auth({"access_token": "expired", "refresh_token": "refresh", "id_token": "id"})
    assert context.use_cached_tokens()
    form = parse_qs(next(r for r in requests if r.url.path == "/token").content.decode())
    assert form["grant_type"] == ["refresh_token"]
    assert form["client_id"] == ["test-client"]
    assert "offline_access" in form["scope"][0]
    monkeypatch.setattr(auth, "_context", lambda _: context)
    # Avoid retrying the failing remote logout; verify the fallback clearing path.
    monkeypatch.setattr(context, "logout", Mock(side_effect=RuntimeError("unavailable")))
    auth.tiled_logout("uri")
    assert not list(context._token_directory().iterdir())
    assert context.http_client.is_closed


@pytest.mark.parametrize("cancel", [False, True])
def test_ui_worker_handoff_and_cancel_before_token_publication(monkeypatch, cancel):
    import panel as pn
    from bokeh.document import Document
    from smi_browser.ui import auth as ui
    doc = Document()
    monkeypatch.setattr(pn.state, "curdoc", doc)
    monkeypatch.setattr(pn.state, "add_periodic_callback", lambda *a, **k: SimpleNamespace(stop=Mock()))
    workers = []

    class Thread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            workers.append(self.target)

    monkeypatch.setattr(ui.threading, "Thread", Thread)
    monkeypatch.setattr(ui, "tiled_whoami", lambda _: None)
    login = SimpleNamespace(verification_uri="https://idp.test/verify", user_code="ABCD", expires_in=900, close=Mock())
    monkeypatch.setattr(ui, "begin_login", lambda _: login)
    monkeypatch.setattr(ui, "poll_login", lambda *a: ({"access_token": "private"}, "scientist"))
    save = Mock()
    monkeypatch.setattr(ui, "remember_login", save)
    logged_in = Mock()
    controls = ui.AuthControls("uri", logged_in, Mock())
    controls.start()
    job = controls.attempt
    workers.pop()()
    save.assert_not_called()
    logged_in.assert_not_called()
    if cancel:
        controls.cancel_login()
    controls._drain(job)
    if cancel:
        save.assert_not_called()
        logged_in.assert_not_called()
    else:
        save.assert_called_once()
        logged_in.assert_called_once_with("scientist")
        assert "scientist" in controls.status.object
        assert not controls.form.visible
    login.close.assert_called_once()
    assert controls.attempt is None
    assert not controls.login.disabled


def test_ui_link_opens_client_browser_and_escapes_code():
    from smi_browser.ui.auth import AuthControls
    controls = AuthControls("uri", Mock(), Mock())
    job = {"events": queue.Queue(), "timer": SimpleNamespace(stop=Mock()),
           "cancel": threading.Event(), "lock": threading.Lock()}
    controls.attempt = job
    job["events"].put(("code", ("https://idp.test/verify?a=1&b=2", "<ABCD>", 900)))
    controls._drain(job)
    assert 'target="_blank"' in controls.instructions.object
    assert "&lt;ABCD&gt;" in controls.instructions.object
    assert "a=1&amp;b=2" in controls.instructions.object


def test_catalog_connection_never_prompts(monkeypatch):
    from tiled.client.context import Context
    import tiled.client.constructors as constructors
    from smi_browser._tiled import connect
    context = SimpleNamespace(api_key=None, server_info=SimpleNamespace(
        authentication=SimpleNamespace(providers=[provider()])),
        use_cached_tokens=Mock(return_value=False), close=Mock())
    monkeypatch.setattr(Context, "from_any_uri", lambda *a, **k: (context, []))

    def from_context(ctx, **kwargs):
        assert ctx.has_external_auth is True
        return {"smi": {"migration": "catalog"}}

    monkeypatch.setattr(constructors, "from_context", from_context)
    assert connect("uri", "smi/migration") == "catalog"
    context.use_cached_tokens.assert_called_once()


@pytest.mark.parametrize("force_device", [False, True])
def test_login_reuses_cached_session_unless_device_fallback_requested(monkeypatch, force_device):
    import panel as pn
    from bokeh.document import Document
    from smi_browser.ui import auth as ui

    monkeypatch.setattr(pn.state, "curdoc", Document())
    monkeypatch.setattr(pn.state, "add_periodic_callback", lambda *a, **k: SimpleNamespace(stop=Mock()))
    workers = []

    class Thread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            workers.append(self.target)

    monkeypatch.setattr(ui.threading, "Thread", Thread)
    cached = Mock(return_value="scientist")
    monkeypatch.setattr(ui, "tiled_whoami", cached)
    begin = Mock(side_effect=auth.LoginError("device flow requested"))
    monkeypatch.setattr(ui, "begin_login", begin)
    logged_in = Mock()
    controls = ui.AuthControls("uri", logged_in, Mock())
    controls.start(force_device=force_device)
    job = controls.attempt
    workers.pop()()
    logged_in.assert_not_called()
    controls._drain(job)
    if force_device:
        cached.assert_not_called()
        begin.assert_called_once_with("uri")
        logged_in.assert_not_called()
    else:
        cached.assert_called_once_with("uri")
        begin.assert_not_called()
        logged_in.assert_called_once_with("scientist")
        assert "no code needed" in controls.message.object
    assert not controls.device_login.disabled
