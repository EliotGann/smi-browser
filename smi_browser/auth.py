"""Tiled-discovered OAuth device login, without terminal prompts or browser launches."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field
import time
from urllib.parse import urlparse

import httpx


class LoginError(RuntimeError):
    """A user-facing login failure (never includes token response bodies)."""


def _https_url(value):
    value = str(value or "")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise LoginError("The authentication provider returned an invalid HTTPS login URL.")
    return value


def _context(uri):
    from tiled.client.context import Context
    return Context.from_any_uri(uri, timeout=httpx.Timeout(20.0))[0]


def identity(info):
    return next((str(i["id"]) for i in (info or {}).get("identities", []) if i.get("id")), None)


def tiled_whoami(uri):
    context = None
    try:
        context = _context(uri)
        if not context.api_key and not context.use_cached_tokens():
            return None
        return identity(context.whoami())
    except Exception:
        return None
    finally:
        if context is not None:
            context.close()


@dataclass(repr=False)
class DeviceLogin:
    context: object = field(repr=False)
    client: httpx.Client = field(repr=False)
    token_endpoint: str
    client_id: str
    scopes: str
    verification_uri: str
    user_code: str
    device_code: str = field(repr=False)
    expires_in: float
    interval: float
    deadline: float

    def close(self):
        self.client.close()
        self.context.close()


def begin_login(uri):
    """Discover the external provider and request a code. No cached auth is sent to Entra."""
    context = _context(uri)
    # Separate client keeps Tiled API keys, cookies, and tokens off IdP requests.
    client = httpx.Client(timeout=20.0)
    try:
        specs = [s for s in context.server_info.authentication.providers
                 if s.mode == "external" and s.links.get("client_id") and s.links.get("token_endpoint")]
        if len(specs) != 1:
            raise LoginError("Expected one external OAuth device-login provider from Tiled.")
        spec = specs[0]
        scopes = " ".join(sorted({"openid", "offline_access"} | set(spec.extra_scopes or [])))
        response = client.post(_https_url(spec.links["auth_endpoint"]),
                               data={"client_id": spec.links["client_id"], "scope": scopes})
        if response.status_code != 200:
            raise LoginError(f"Could not start Microsoft sign-in (HTTP {response.status_code}). Try again.")
        data = response.json()
        verification_uri = _https_url(next((data[k] for k in (
            "verification_uri_complete", "verification_uri", "verification_url") if data.get(k)), None))
        expires = float(data["expires_in"])
        interval = max(1.0, float(data.get("interval", 5)))
        if not 0 < expires <= 86400 or not 0 < interval <= expires:
            raise LoginError("The provider returned an invalid login expiry or polling interval.")
        return DeviceLogin(context, client, _https_url(spec.links["token_endpoint"]),
                           str(spec.links["client_id"]), scopes, verification_uri,
                           str(data["user_code"]), str(data["device_code"]),
                           expires, interval, time.monotonic() + expires)
    except Exception:
        client.close()
        context.close()
        raise


def poll_login(login, cancel):
    """Wait off the document thread. Validate with Tiled before persisting anything."""
    interval = login.interval
    while True:
        remaining = login.deadline - time.monotonic()
        if remaining <= 0:
            raise LoginError("The sign-in code expired. Click Login to get a new code.")
        if cancel.wait(min(interval, remaining)):
            raise CancelledError()
        if time.monotonic() >= login.deadline:
            raise LoginError("The sign-in code expired. Click Login to get a new code.")
        try:
            response = login.client.post(login.token_endpoint, data={
                "device_code": login.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": login.client_id,
            })
        except httpx.TransportError:
            interval = min(interval * 2, 60)
            continue
        if cancel.is_set():
            raise CancelledError()
        if response.status_code == 429 or response.status_code >= 500:
            interval = min(interval + 5, 60)
            continue
        data = response.json()
        error = data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error in ("authorization_declined", "access_denied"):
            raise LoginError("Microsoft sign-in was declined. Click Login to try again.")
        if error in ("expired_token", "code_expired"):
            raise LoginError("The sign-in code expired. Click Login to get a new code.")
        if response.status_code != 200 or error:
            raise LoginError(f"Microsoft sign-in failed (HTTP {response.status_code}). Click Login to retry.")
        if not data.get("access_token") or not data.get("refresh_token"):
            raise LoginError("The provider did not return the tokens needed for a Tiled session.")
        # Do not save tokens until the UI confirms this is still the current attempt.
        result = login.context.http_client.get(
            login.context.server_info.authentication.links.whoami,
            headers={"Authorization": f"Bearer {data['access_token']}"}, auth=None,
        )
        if result.status_code != 200:
            raise LoginError("Microsoft sign-in succeeded, but Tiled did not accept the session.")
        user = identity(result.json())
        if not user:
            raise LoginError("Tiled did not return an identity for this account.")
        if cancel.is_set():
            raise CancelledError()
        return data, user


def remember_login(login, tokens):
    """Use Tiled's standard secure token cache and automatic refresh handling."""
    login.context.api_key = None
    login.context.client_id = login.client_id
    login.context.scopes = login.scopes
    login.context.configure_auth(tokens, remember_me=True)


def tiled_logout(uri):
    """Clear local tokens even when the old session cannot refresh or revoke."""
    context = _context(uri)
    try:
        try:
            if not context.api_key:
                context.use_cached_tokens()
            context.logout()
        except Exception:
            # A stale PAM refresh token cannot be revoked at the new IdP.
            pass
        finally:
            # Use Tiled's token-store API without requiring a valid session.
            # Include id_token left by Microsoft (or an earlier account).
            from tiled.client.auth import TiledAuth
            auth = TiledAuth(
                context.server_info.authentication.links.refresh_session,
                context.http_client.cookies.get("tiled_csrf", ""),
                context._token_directory(), context.client_id,
            )
            for key in ("access_token", "refresh_token", "id_token"):
                auth.sync_clear_token(key)
            context.http_client.auth = None
    finally:
        context.close()
