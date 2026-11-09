"""Single source of the Plurai backend's single-URL path scheme.

A backend box exposes ONE base URL — locally the single server itself
(one process, one port, REST and MCP alike) and, when
deployed, the fronted URL its TLS front forwards to. Every surface is a path
off that one base:

  ``<base>``               → orchestration REST (catch-all: /threads, /user-turn, /finalize)
  ``<base>/mcp``           → diagnostic MCP transport
  ``<base>/mock/<env_id>/``→ deployed mock proxy   (server-ADVERTISED as a relative path — resolved client-side here)
  ``<base>/documents/*``   → binary document store (self-describing ``<scheme>:///documents/...`` references — resolved client-side here)
  ``<base>/storage/*``     → object storage (ingest)
  ``<base>/telemetry``     → telemetry forwarder (plugin-side consumer)

The scheme is identical for loopback and fronted URLs; what loopback signals
is the BEARER posture (tokenless dev), not a different URL layout.

``<base>`` may itself carry a PATH — a proxy that forwards a versioned prefix to
the box serves every surface below that prefix. Anything computing one of these
URLs therefore keeps the base's path; rebuilding from scheme and host alone
yields a URL the proxy does not route.

This module is the ONLY place the SDK computes the relationship between those
URLs. In particular :func:`diagnostic_base` is the documented inverse of the
``<base>/mcp`` prefix rule — no other caller may do its own strip-``/mcp``
string surgery, :mod:`plurai.ingest.storage_client` included.

This module is also the SDK's ONE resolver of which backend a process talks to and with what
token (:func:`resolve`). The answer is one atomic :class:`Endpoint`, ``(url, token, source)``:

* the environment pair, ``PLURAI_URL`` with ``PLURAI_TOKEN``; else
* ``PLURAI_URL`` alone with the credentials file's token, when the file's url is that same backend; else
* the credentials file pair (:mod:`plurai.credentials`, written by ``plurai login``).

There is no default backend and no third source: a ``.env`` file never names the backend. A
configuration that takes its url from one source and its token from the other is an error rather
than a guess, because a token sent to a url it was not issued for is a leaked token. ``PLURAI_URL``
naming the saved login's own backend is no such mix: the token goes to the url it was issued for. Loopback is
the one tokenless case: ``PLURAI_URL`` naming a loopback address needs no ``PLURAI_TOKEN``.

The bearer travels only to the resolved origin, exactly: scheme, host and port (:meth:`Endpoint.bearer_for`), so a
mock url, a pinned redirect or a url a model wrote into a file cannot collect it.

Stdlib-only (``urllib.parse``): imported from config-time paths that must not pull
optional dependencies.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

from plurai import credentials

URL_ENV: Final = 'PLURAI_URL'
TOKEN_ENV: Final = 'PLURAI_TOKEN'  # noqa: S105 - an env-var NAME, not a credential

#: The path prefix the diagnostic MCP endpoint hangs off the single base URL.
MCP_PATH_PREFIX = '/mcp'


def diagnostic_base(url: str) -> str:
    """Return the box base URL for a diagnostic endpoint URL (``<base>/mcp`` → ``<base>``).

    Accepts either form — the full MCP endpoint (``https://box/mcp``, trailing slash
    tolerated) or an already-bare base (returned unchanged, sans trailing slash) — so
    existing ``$PLURAI_URL`` values keep working verbatim. Path-aware: only
    a trailing ``/mcp`` PATH segment is removed, never a substring elsewhere in the URL.
    """
    stripped = url.rstrip('/')
    if urlsplit(stripped).path.endswith(MCP_PATH_PREFIX):
        return stripped[: -len(MCP_PATH_PREFIX)]
    return stripped


def resolve_server_relative(base_url: str, value: str) -> str:
    """Resolve a server-relative path (leading ``/``) against the box base URL.

    The server advertises its ``/mock`` and ``/documents`` surfaces as
    server-RELATIVE paths — only the client knows which base URL actually reaches
    the box (behind a fronting router, the server cannot know a client-reachable
    address). STRICT by design: anything but a leading-``/`` path raises — the
    callers decide what is a server path (a relative ``mock_service_url``, a
    server reference's path) BEFORE calling, so a non-path here is a bug or
    a version skew, and either must fail loudly rather than pass through.

    The join keeps the base's OWN path, because a box can be fronted under a
    sub-path: a proxy that forwards a versioned prefix to the box serves every
    surface below that prefix, so dropping it yields a URL the proxy does not
    route. The base's trailing slash is stripped before the join, so a base with
    one and a base without one resolve identically.
    """
    if not value.startswith('/'):
        raise ValueError(f'expected a server-relative path (leading "/"), got {value!r} — is the backend out of date?')
    parts = urlsplit(base_url)
    prefix = parts.path.rstrip('/')
    return f'{parts.scheme}://{parts.netloc}{prefix}{value}'


#: Scheme of self-describing server references the box folds into customer-schema
#: data (``plurai:///documents/...``). Scheme-tagged like the ``data:`` URIs that
#: share the agent-context channel, so a server-authored value is unmistakable
#: inside customer data — resolution dispatches on the scheme, never on a value's
#: shape. This is the only scheme read; a value with any other scheme is customer data.
DOCUMENT_REF_SCHEME = 'plurai'


def resolve_document_reference(base_url: str, value: str) -> str:
    """Resolve a ``<scheme>:///<path>`` server reference against the box base URL.

    Any value NOT carrying the reference scheme is returned unchanged — it is
    customer data (including customer-authored absolute URLs and ``data:``
    URIs), which this function must never rewrite; callers may therefore map it
    over a whole customer-schema dict. A value that DOES carry the scheme is
    server-authored by definition and must resolve — a malformed one (no path)
    raises rather than passing through.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if parts.scheme != DOCUMENT_REF_SCHEME:
        return value
    return resolve_server_relative(base_url, parts.path)


def is_loopback_url(url: str) -> bool:
    """True iff ``url`` targets a loopback host (``localhost``, ``127.0.0.0/8``, ``::1``).

    The single mode signal for the fail-closed bearer posture:
    loopback keeps the tokenless dev contract; anything else is the deployed posture and
    must carry the bearer. An unparseable URL or missing hostname returns ``False``
    (treated as remote), so a malformed value fails CLOSED rather than open.
    """
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return False
    return bool(hostname) and credentials.is_loopback_host(hostname)


#: Where the Terms of Service and the Privacy Policy are linked: the "Terms and privacy" section of the public README, the one
#: place that holds both addresses.
TERMS_URL: Final = 'https://github.com/plurai-ai/plurai-simulation#terms-and-privacy'

#: What a user without any configuration is told.
MISSING_CONFIGURATION_MESSAGE: Final = (
    'Plurai is not configured: PLURAI_URL and PLURAI_TOKEN are not set and there is no saved login.\n'
    'Sign up in the Plurai dashboard, or on the sign-in page the login opens.\n'
    f'Signing up accepts the Terms of Service and the Privacy Policy, linked at {TERMS_URL}\n'
    "Log in once with the plugin's setup skill (/harness-setup), or in a terminal with: plurai login (it opens the browser)\n"
    'Otherwise export PLURAI_URL and PLURAI_TOKEN, with a key from the Plurai dashboard, in the environment that launches this command.'
)

#: How a failed bearer is fixed, for the lines that report a refused token.
CREDENTIAL_FIX: Final = 'run `plurai logout` and then `plurai login`, or export PLURAI_URL and PLURAI_TOKEN'

Source = Literal['environment', 'credentials file']


class EndpointError(RuntimeError):
    """The backend cannot be resolved. The message says what to do and never repeats a token."""


class NotConfiguredError(EndpointError):
    """Neither the environment pair nor a saved login names a backend."""


class PartialConfigurationError(EndpointError):
    """Only half of a pair is set, or the halves come from different sources."""


class SavedLoginError(EndpointError):
    """The credentials file exists but cannot be used."""


def _origin(url: str) -> tuple[str, str, int] | None:
    """``(scheme, host, port)`` of ``url``, the default port filled in; ``None`` when it has no host."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if not host or scheme not in ('http', 'https'):
        return None
    return scheme, host.lower(), port or (443 if scheme == 'https' else 80)


@dataclass(frozen=True)
class Endpoint:
    """The backend a process talks to: its base url, the bearer for it (``None`` only for tokenless loopback) and where both came from."""

    url: str
    token: str | None
    source: Source

    def __repr__(self) -> str:
        return f'Endpoint(url={self.url!r}, token={"<set>" if self.token else None}, source={self.source!r})'

    def bearer_for(self, target: str) -> str | None:
        """The token, if ``target`` has exactly this endpoint's origin; otherwise ``None``."""
        return bearer_for_origin(self.url, self.token, target)

    def auth_headers(self, target: str) -> dict[str, str]:
        token = self.bearer_for(target)
        return {'Authorization': f'Bearer {token}'} if token else {}


def bearer_for_origin(base: str, token: str | None, target: str) -> str | None:
    """``token`` if ``target`` has exactly the origin of ``base`` (scheme, host and port); otherwise ``None``.

    A loopback origin is no exception, for a remote backend or a local one: any process on the machine can listen on
    a loopback port, so a token goes to a loopback url only when that is the backend's own origin. Hosted mocks live
    under the backend origin (``<base>/mock/<env_id>/``), so this rule covers them. A mock the SDK starts on a local
    port (``mock_mode='local'``) needs no token and gets none. An unparseable target gets nothing.
    """
    if not token:
        return None
    wanted, held = _origin(target), _origin(base)
    return token if wanted is not None and wanted == held else None


def _environment_url(text: str) -> str:
    """The base ``PLURAI_URL`` names, or an :class:`EndpointError`: any origin, but an absolute url without userinfo or whitespace,
    over https (plain http only for loopback)."""
    try:
        parts = urlsplit(text)
        usable = parts.scheme.lower() in ('http', 'https') and bool(parts.hostname) and not parts.username and not parts.password
        _ = parts.port
    except ValueError:
        usable = False
    if not usable or any(ord(char) <= 0x20 or ord(char) == 0x7F for char in text):
        raise EndpointError(
            f'PLURAI_URL is not usable: expected http(s)://host[:port][/prefix] with no userinfo or whitespace (got a value of {len(text)} characters)'
        )
    if parts.scheme.lower() != 'https' and not is_loopback_url(text):
        raise EndpointError(
            f'PLURAI_URL must use https: the token would cross the network in clear text over plain http, which is allowed only for loopback '
            f'(got scheme {parts.scheme.lower()!r} for a host that is not on this machine)'
        )
    return diagnostic_base(text)


def resolve(env: Mapping[str, str] | None = None) -> Endpoint:
    """The backend ``env`` (default ``os.environ``) names: the environment pair, else the saved login.

    Raises:
        NotConfiguredError: neither source names a backend (the message is :data:`MISSING_CONFIGURATION_MESSAGE`).
        PartialConfigurationError: one half of the pair is set without the other.
        SavedLoginError: the credentials file is unusable.
        EndpointError: ``PLURAI_URL`` is not a usable url.
    """
    environment = os.environ if env is None else env
    url = environment.get(URL_ENV) or ''
    token = (environment.get(TOKEN_ENV) or '').strip()
    if token and len(token) < credentials.MIN_TOKEN_LENGTH:
        raise EndpointError(f'{TOKEN_ENV} is not usable: {credentials.TOKEN_SHAPE}.')
    if url.strip():
        base = _environment_url(url)
        if token:
            return Endpoint(base, token, 'environment')
        if is_loopback_url(base):
            return Endpoint(base, None, 'environment')
        if (saved := _saved_login_for(base, environment)) is not None:
            return saved
        raise PartialConfigurationError(
            f'Plurai is partly configured: {URL_ENV} is set but {TOKEN_ENV} is not, and no login is saved for that backend. '
            f'A backend that is not on this machine needs its token. Log in to it with `plurai login`, set both {URL_ENV} and '
            f'{TOKEN_ENV}, or unset {URL_ENV} to use the saved login (`plurai login --status`).'
        )
    if token:
        raise PartialConfigurationError(
            f'Plurai is partly configured: {TOKEN_ENV} is set but {URL_ENV} is not. A token is only sent to the backend it was issued for, '
            f'so the url of a saved login is not combined with it. Set both {URL_ENV} and {TOKEN_ENV}, '
            f'or unset {TOKEN_ENV} to use the saved login (`plurai login --status`).'
        )
    if moved := credentials.home_problem(environment):
        raise NotConfiguredError(f'{moved}\n{MISSING_CONFIGURATION_MESSAGE}')
    saved = credentials.read()
    if saved.problem:
        raise SavedLoginError(f'Plurai cannot use the saved login: {saved.problem}')
    if not saved.token:
        raise NotConfiguredError(MISSING_CONFIGURATION_MESSAGE)
    return Endpoint(diagnostic_base(saved.url), saved.token, 'credentials file')


def _saved_login_for(base: str, env: Mapping[str, str]) -> Endpoint | None:
    """The saved login when it is usable and for ``base``, else ``None``; a file that cannot be used is no login here."""
    if credentials.home_problem(env):
        return None
    saved = credentials.read()
    if saved.problem or not saved.token or diagnostic_base(saved.url) != base:
        return None
    return Endpoint(base, saved.token, 'credentials file')


def bearer_for(target: str, env: Mapping[str, str] | None = None) -> str | None:
    """The token a request to ``target`` carries, or ``None``.

    For a caller that reaches a url which is not necessarily the backend (a mock). With nothing configured there is no
    token to send, and a url whose origin is not the resolved endpoint's gets none (:meth:`Endpoint.bearer_for`). A
    configuration that is present but unusable (a half-set pair, an unusable url, a broken saved login) is an error
    and raises, because it would otherwise read as "no token".

    Raises:
        PartialConfigurationError, SavedLoginError, EndpointError: as :func:`resolve`.
    """
    try:
        endpoint = resolve(env)
    except NotConfiguredError:
        return None
    return endpoint.bearer_for(target)
