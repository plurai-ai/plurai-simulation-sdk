"""The browser login of the SDK: ``plurai login`` and ``plurai logout``.

- :mod:`plurai.login.cli`: the two commands, which compose the parts below with :mod:`plurai.credentials`.
- :mod:`plurai.login.oauth_client`: a generic OAuth 2.1 public client (discovery, PKCE, the loopback listener, the code
  exchange, revocation) that knows no Plurai host. Authlib loads only when a sign-in runs, so this package imports it
  lazily and ``plurai --version`` never pays for it.
- :mod:`plurai.login._environments`: the table that maps a backend host to its sign-in issuer, client and key routes.
- :mod:`plurai.login._pages`: the constant pages the listener shows the browser tab, read from ``pages/*.html``.

The credentials file stays in :mod:`plurai.credentials`: every command reads it to find its backend, not only these two.
"""

from __future__ import annotations

from plurai.login.cli import add_arguments, main, main_logout, run, run_logout

__all__ = ['add_arguments', 'main', 'main_logout', 'run', 'run_logout']
