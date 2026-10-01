"""Per-session Panel controls for Microsoft/Tiled device-code sign-in."""
from __future__ import annotations

from concurrent.futures import CancelledError
from html import escape
import queue
import threading

import panel as pn

from smi_browser.auth import begin_login, poll_login, remember_login, tiled_logout, tiled_whoami


class AuthControls:
    def __init__(self, uri, on_login, on_logout):
        self.uri, self.on_login, self.on_logout = uri, on_login, on_logout
        self.attempt = None
        self.status = pn.pane.Markdown("*checking…*", width=220)
        self.login = pn.widgets.Button(name="Login", button_type="primary", width=100)
        self.logout = pn.widgets.Button(name="Logout", width=80, visible=False)
        self.cancel = pn.widgets.Button(name="Cancel sign-in", width=140)
        self.instructions = pn.pane.HTML("", sizing_mode="stretch_width")
        self.message = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.device_login = pn.widgets.Button(name="Use device code", width=150)
        self.form = pn.Column(
            pn.pane.Markdown("**Microsoft sign-in for Tiled**"),
            self.instructions, self.message, pn.Row(self.device_login, self.cancel),
            visible=False, width=380,
            styles={"background": "#F2F5F7", "border": "1px solid #C0C0C0", "padding": "10px"},
        )
        self.login.on_click(self.start)
        self.device_login.on_click(lambda _: self.start(force_device=True))
        self.cancel.on_click(self.cancel_login)
        self.logout.on_click(self.sign_out)

    def refresh(self, user=None):
        user = user or tiled_whoami(self.uri)
        self.status.object = f"**Logged in:** {escape(user)}" if user else "**Not logged in**"
        self.login.name = "Re-login" if user else "Login"
        self.logout.visible = bool(user)

    def cancel_login(self, _=None):
        job, self.attempt = self.attempt, None
        if job:
            job["cancel"].set()
            job["timer"].stop()
            # A completed worker may have transferred its context to the queue.
            # Serialize this drain with publication so no context is stranded.
            with job["lock"]:
                while not job["events"].empty():
                    kind, value = job["events"].get_nowait()
                    if kind == "success":
                        value[0].close()
        self.login.disabled = False
        self.device_login.disabled = False
        self.cancel.visible = False
        self.instructions.object = ""
        self.message.object = "Sign-in cancelled. Click Login to start again."

    def start(self, _=None, *, force_device=False):
        if self.attempt:
            return
        if pn.state.curdoc is None:
            self.form.visible = True
            self.message.object = "Open the served Panel app to sign in."
            return
        job = {"cancel": threading.Event(), "events": queue.Queue(), "lock": threading.Lock()}
        self.attempt = job
        self.form.visible = True
        self.instructions.object = ""
        self.message.object = ("*Requesting a Microsoft sign-in code…*" if force_device
                               else "*Checking the existing Tiled session…*")
        self.login.disabled = True
        self.device_login.disabled = True
        self.cancel.visible = True

        def publish(kind, value):
            with job["lock"]:
                if job["cancel"].is_set():
                    return False
                job["events"].put((kind, value))
                return True

        def worker():
            login = None
            try:
                if not force_device:
                    user = tiled_whoami(self.uri)
                    if job["cancel"].is_set():
                        return
                    if user:
                        publish("cached", user)
                        return
                login = begin_login(self.uri)
                if not publish("code", (login.verification_uri, login.user_code, login.expires_in)):
                    return
                tokens, user = poll_login(login, job["cancel"])
                if publish("success", (login, tokens, user)):
                    login = None  # the document callback now owns cleanup
            except CancelledError:
                pass
            except Exception as exc:
                # HTTP errors may contain credential-bearing URLs/bodies.
                from smi_browser.auth import LoginError
                text = str(exc) if isinstance(exc, LoginError) else "Could not complete sign-in. Check the connection and try Login again."
                publish("error", text)
            finally:
                if login is not None:
                    login.close()

        job["timer"] = pn.state.add_periodic_callback(lambda: self._drain(job), period=200)
        pn.state.curdoc.on_session_destroyed(lambda _: self.cancel_login() if self.attempt is job else None)
        threading.Thread(target=worker, daemon=True, name="tiled-login").start()

    def _drain(self, job):
        """Panel periodic callbacks hold the document lock; workers only queue data."""
        if self.attempt is not job:
            return
        while not job["events"].empty():
            kind, value = job["events"].get_nowait()
            if kind == "code":
                uri, code, expires = value
                self.instructions.object = (
                    f'<p>Enter this code: <strong style="font-size:1.5em">{escape(code)}</strong></p>'
                    f'<p><a href="{escape(uri, quote=True)}" target="_blank" rel="noopener noreferrer" '
                    'style="color:#105C78">Open Microsoft sign-in</a></p>'
                    '<p>Complete sign-in in the new tab, then return here. '
                    'No password is entered in SMI Browser.</p>'
                )
                self.message.object = f"*Waiting for approval (code valid for {expires / 60:.0f} minutes)…*"
                continue
            self.attempt = None
            job["timer"].stop()
            self.login.disabled = False
            self.device_login.disabled = False
            self.cancel.visible = False
            self.instructions.object = ""
            if kind == "error":
                self.message.object = value
                return
            if kind == "cached":
                self.refresh(value)
                self.form.visible = True
                self.message.object = ("**Existing Tiled session restored** — no code needed. "
                                       "Use device code below if you need to sign in again or switch accounts.")
                self.on_login(value)
                return
            login, tokens, user = value
            try:
                remember_login(login, tokens)
            except Exception:
                self.message.object = "Signed in, but could not save the Tiled session. Check token-cache permissions and retry."
                return
            finally:
                login.close()
            self.refresh(user)
            self.form.visible = False
            self.message.object = ""
            self.on_login(user)

    def sign_out(self, _=None):
        self.cancel_login()
        try:
            tiled_logout(self.uri)
        except Exception:
            self.form.visible = True
            self.message.object = "Could not clear the Tiled session. Check the connection and retry Logout."
            return
        self.form.visible = False
        self.status.object = "**Not logged in**"
        self.login.name = "Login"
        self.logout.visible = False
        self.on_logout()


def wire(app, tiled_uri):
    """Construct controls for package consumers without sharing widget globals."""
    def reset(*_):
        app.cat = None
    controls = AuthControls(tiled_uri, reset, reset)
    controls.refresh()
    return controls
