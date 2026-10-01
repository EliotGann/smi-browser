# Microsoft browser SSO for SMI Browser

## Current behavior

SMI Browser restores Tiled's cached access/refresh tokens on startup. **Login**
also checks that cache in a worker before requesting a device code. If valid,
it reconnects without a code or Microsoft interaction. **Use device code**
bypasses that check for reauthentication/account switching.

A Microsoft session in Office/Teams or another browser tab is different from
the Tiled token cache on the machine running Panel. Microsoft can reuse its
browser cookie during an authorization redirect, but SMI Browser cannot read
that cookie or another origin's localStorage. The current device flow may
reuse the Microsoft account after entering the code; it still requires the code.

## Live-server findings (2026-10-01)

Discovery at `https://tiled.nsls2.bnl.gov/api/v1/` advertises an external provider
named `pam`, with Microsoft device-code and token endpoints.

`GET /api/v1/auth/provider/pam/authorize` responds with a redirect to Microsoft's
authorization endpoint containing:

- `response_type=code`
- `prompt=login` (forces authentication rather than browser-session SSO)
- `redirect_uri=https://tiled.nsls2.bnl.gov/api/v1/auth/provider/pam/code`

Adding `prompt=none` and a different `redirect_uri` to the Tiled request does not
change either value. The deployed Tiled UI stores its tokens on the Tiled origin;
no supported cross-origin token handoff to this Panel app was found. Simply
opening that endpoint cannot complete SMI Browser's login.

## Requirements for a code-free primary login

Choose and configure a supported integration with the Tiled/Entra operators:

1. **Direct authorization-code + PKCE login for SMI Browser.** Register a callback
   for the actual browser-facing Panel deployment (and a localhost callback for
   local development if needed). Grant the client the Tiled delegated API scope.
   Confirm the appropriate client type and refresh-token configuration. Use a
   supported OAuth/MSAL library, bind state and PKCE to the requesting session,
   validate Tiled access, and retain device-code login as an explicit fallback.
   A different client registration also needs its client ID retained on refresh;
   the existing Tiled cache assumes the client ID from server discovery.
2. **Tiled-mediated browser login.** Tiled must allow session reuse instead of
   unconditionally sending `prompt=login`, and provide a documented return/handoff
   mechanism bound to the originating Panel session and an allowlisted callback.
   Account for the distinction between Tiled-issued session tokens and directly
   issued Entra tokens and their respective refresh endpoints.

The desired experience is: restore app tokens first; then a browser popup or
redirect that lets Microsoft reuse its session; show account selection/MFA only
when required; offer device codes if the browser flow cannot complete. Truly
silent iframe SSO is best-effort because third-party-cookie blocking, multiple
accounts, consent, and conditional-access policies may require interaction.

References:

- [Microsoft browser SSO](https://learn.microsoft.com/en-us/entra/msal/javascript/browser/single-sign-on)
- [Authorization-code flow and registered callbacks](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow)
