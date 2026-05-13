"""Authentication UI — tiled login/logout widgets and callbacks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from smi_browser.state import AppState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (no widget access)
# ---------------------------------------------------------------------------


def tiled_whoami(tiled_uri: str) -> str | None:
    """Return the username for the currently cached tiled session, or None."""
    try:
        from tiled.client.context import Context

        context, _ = Context.from_any_uri(tiled_uri)
        if not context.use_cached_tokens():
            return None
        info = context.whoami()
    except Exception:
        return None
    if not info:
        return None
    identities = info.get("identities") or []
    for ident in identities:
        if ident.get("id"):
            return str(ident["id"])
    return None


def tiled_login(tiled_uri: str, username: str, password: str) -> str:
    """Authenticate against tiled with username/password.

    Returns the logged-in username on success; raises on failure.
    """
    from tiled.client.context import Context, password_grant

    if not username or not password:
        raise ValueError("Username and password are required.")

    context, _ = Context.from_any_uri(tiled_uri)
    providers = context.server_info.authentication.providers
    if not providers:
        raise RuntimeError("Tiled server reports no authentication providers.")
    spec = providers[0]
    auth_endpoint = spec.links["auth_endpoint"]
    tokens = password_grant(
        context.http_client, auth_endpoint, spec.provider, username, password,
    )
    context.configure_auth(tokens, remember_me=True)

    info = context.whoami()
    identities = (info or {}).get("identities") or []
    return identities[0]["id"] if identities else username


def tiled_logout(tiled_uri: str) -> None:
    """Clear the cached tiled session for this server."""
    try:
        from tiled.client.context import Context

        context, _ = Context.from_any_uri(tiled_uri)
        if context.use_cached_tokens():
            try:
                context.logout()
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

login_status = pn.pane.Markdown("*checking…*", width=220)
btn_login = pn.widgets.Button(
    name="🔑 Login", button_type="primary", width=90,
)
btn_logout = pn.widgets.Button(
    name="Logout", button_type="light", width=80, visible=False,
)
login_user = pn.widgets.TextInput(
    name="Username", placeholder="bnl username", width=220,
)
login_pass = pn.widgets.PasswordInput(
    name="Password", placeholder="password", width=220,
)
login_submit = pn.widgets.Button(
    name="Sign in", button_type="success", width=90,
)
login_msg = pn.pane.Markdown("", width=220)
login_form = pn.Column(
    pn.pane.Markdown("**Tiled login**"),
    login_user,
    login_pass,
    pn.Row(login_submit),
    login_msg,
    visible=False,
    width=260,
    styles={
        "background": "#f8f9fa",
        "border": "1px solid #ced4da",
        "border-radius": "6px",
        "padding": "10px",
    },
)


# ---------------------------------------------------------------------------
# Wire callbacks
# ---------------------------------------------------------------------------


def wire(app: AppState, tiled_uri: str) -> None:
    """Connect auth widgets to the app state."""

    def _refresh_login_status():
        user = tiled_whoami(tiled_uri)
        if user:
            login_status.object = f"🟢 **Logged in:** `{user}`"
            btn_login.name = "🔄 Re-login"
            btn_logout.visible = True
        else:
            login_status.object = "🔴 **Not logged in**"
            btn_login.name = "🔑 Login"
            btn_logout.visible = False

    def _toggle_login_form(event=None):
        login_form.visible = not login_form.visible
        if login_form.visible:
            login_msg.object = ""
            login_pass.value = ""

    def _on_login_submit(event=None):
        user_in = (login_user.value or "").strip()
        pwd = login_pass.value or ""
        login_msg.object = "*signing in…*"
        login_submit.disabled = True
        try:
            user = tiled_login(tiled_uri, user_in, pwd)
            login_msg.object = f"✅ Signed in as `{user}`"
            login_pass.value = ""
            login_form.visible = False
            app.cat = None  # force reconnect with new credentials
            _refresh_login_status()
            try:
                pn.state.notifications.success(f"Tiled login OK ({user})")
            except Exception:
                pass
        except Exception as exc:
            login_msg.object = f"❌ {type(exc).__name__}: {exc}"
        finally:
            login_submit.disabled = False

    def _on_logout(event=None):
        tiled_logout(tiled_uri)
        app.cat = None
        _refresh_login_status()
        try:
            pn.state.notifications.info("Logged out of tiled.")
        except Exception:
            pass

    btn_login.on_click(_toggle_login_form)
    login_submit.on_click(_on_login_submit)
    btn_logout.on_click(_on_logout)

    # Initial check
    _refresh_login_status()
