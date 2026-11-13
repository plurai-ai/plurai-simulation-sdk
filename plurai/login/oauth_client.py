"""An OAuth 2.1 public client for a command line: authorization code with PKCE S256 and a loopback redirect.

    with Authorization(issuer=..., client_id=..., scopes=...) as flow:
        show(flow.authorization_url)
        tokens = flow.wait()          # opens the browser, waits for the callback, exchanges the code
        ...                           # the caller's own work with tokens.access_token
        flow.respond(succeeded=True)  # only now does the browser tab get its page
        flow.revoke(tokens.refresh_token, 'refresh_token')

The module knows no product: it takes the issuer, the client id, the scopes and the timeouts, and returns the tokens.
It prints nothing, reads and writes no file, and raises the typed errors below, so a caller maps outcomes to exit codes
without matching strings. No error message carries a token, the authorization code or the callback's query.

What it checks, following RFC 8252 (native apps), RFC 8414 (server metadata), RFC 9207 (``iss`` on the response) and
RFC 9700 (security best current practice):

* The issuer and every endpoint it advertises are ``https``; plain ``http`` is accepted only on a loopback address.
* The metadata's ``issuer`` equals the issuer it was fetched for, and it supports PKCE ``S256``.
* The listener is bound to ``127.0.0.1`` on a port the system picks before the authorize url exists, so the url never
  names a port another process could take first.
* The listener completes on one ``GET /callback`` whose ``Host`` is exactly ``127.0.0.1:<port>`` and whose ``state``
  matches (compared in constant time). Anything else gets a 404, and the wait goes on: a stray or crafted request can
  neither finish nor abort the login. Nothing about a request is logged.
* Every page it serves is a constant document from :mod:`plurai.login._pages` with a content security policy that allows
  nothing but its inline style and image; no part of a request reaches a page.
* The callback must carry ``iss`` equal to the issuer, which stops a code from another authorization server being
  replayed here (mix-up). A denial is reported as such even without ``iss``: it can only abort the login.

Authlib's httpx client builds the authorize url and makes the token and revocation requests; discovery, the listener,
the ``iss`` check and the browser launch are this module's. The access token's ``sub`` and ``org_id`` claims are read
from its payload without verifying the signature: they only label the caller's own request to a server that verifies
the token itself.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hmac
import json
import queue
import re
import secrets
import socket
import threading
import time
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Final, Literal, Self
from urllib.parse import parse_qs, urlsplit

import httpx2
from authlib.integrations.base_client import OAuthError as _AuthlibOAuthError
from authlib.integrations.httpx_client import OAuth2Client as _OAuth2Client

from plurai import _platform, credentials
from plurai.login import _pages

#: How long the user has to sign in and consent.
DEFAULT_CONSENT_TIMEOUT: Final = 3600.0
#: The limit of each request to the authorization server.
DEFAULT_HTTP_TIMEOUT: Final = 10.0
#: How long the browser's callback request waits for :meth:`Authorization.respond` before it gets the failure page.
RESPONSE_TIMEOUT: Final = 120.0
#: The limit of one request to the listener, so a client that connects and sends nothing cannot hold a thread.
_LISTENER_REQUEST_TIMEOUT: Final = 10.0
#: How long closing waits for the browser's page to be written, so the process does not exit under it.
_PAGE_WRITE_TIMEOUT: Final = 5.0
#: The longest single wait for the callback. The wait is a loop of short waits because a long lock wait is not interrupted by Ctrl-C on Windows.
_POLL_INTERVAL: Final = 0.5

CALLBACK_PATH: Final = '/callback'
_LOOPBACK: Final = '127.0.0.1'
_ERROR_CODE = re.compile(r'[a-z_]{1,64}')
_MAX_METADATA_BYTES = 256 * 1024


class OAuthClientError(Exception):
    """The sign-in did not produce tokens. The message is safe to print."""


class DiscoveryError(OAuthClientError):
    """The issuer's metadata could not be fetched, or it advertises something this client does not accept."""


class ListenerError(OAuthClientError):
    """The loopback listener could not be started."""


class IssuerMismatchError(OAuthClientError):
    """The metadata or the callback names another issuer than the one this client was given."""


class ConsentDeniedError(OAuthClientError):
    """The user declined, or the authorization server refused, at the consent screen (``access_denied``)."""


class AuthorizationError(OAuthClientError):
    """The callback carried an error other than a denial, or no code."""

    def __init__(self, message: str, error: str | None = None) -> None:
        super().__init__(message)
        self.error = error


class CallbackTimeoutError(OAuthClientError):
    """No callback arrived within the consent timeout."""


class TokenEndpointError(OAuthClientError):
    """The token request failed. ``status`` is the HTTP status (``None`` when no answer arrived); ``error`` the OAuth error code."""

    def __init__(self, message: str, *, status: int | None, error: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.error = error


class RevocationError(OAuthClientError):
    """A token could not be revoked."""


@dataclass(frozen=True)
class Tokens:
    """What the token endpoint issued, with the two claims a caller labels its own request with."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(repr=False)
    #: The ``sub`` claim of the access token, unverified.
    subject: str | None
    #: The ``org_id`` claim of the access token, unverified.
    organization: str | None


def _require_secure(url: object, what: str) -> str:
    """``url`` when it is an absolute ``https`` url, or ``http`` on a loopback host; else a :class:`DiscoveryError`."""
    if not isinstance(url, str):
        raise DiscoveryError(f'the {what} is missing from the authorization server metadata')
    try:
        parts = urlsplit(url)
        host = parts.hostname or ''
    except ValueError:
        raise DiscoveryError(f'the {what} is not a url') from None
    secure = parts.scheme == 'https' or (parts.scheme == 'http' and credentials.is_loopback_host(host))
    if not host or not secure or parts.fragment or parts.username or parts.password:
        raise DiscoveryError(f'the {what} must be an https url (plain http only on loopback)')
    # urlsplit drops tab, CR and LF before it parses, so a url carrying them passes every check above; a url that is
    # printed or sent must not hold any character that moves a terminal or splits a header.
    if any(ord(char) <= 0x20 or ord(char) == 0x7F for char in url):
        raise DiscoveryError(f'the {what} must not contain whitespace or control characters')
    return url


def _metadata_url(issuer: str) -> str:
    """RFC 8414: the well-known segment goes between the host and the issuer's path."""
    parts = urlsplit(issuer)
    return f'{parts.scheme}://{parts.netloc}/.well-known/oauth-authorization-server{parts.path.rstrip("/")}'


def _error_code(value: object) -> str:
    """An OAuth error code fit to print: ``value`` when it is one, else ``'invalid'``."""
    return value if isinstance(value, str) and _ERROR_CODE.fullmatch(value) else 'invalid'


def _claims(token: str) -> dict[str, Any]:
    """The payload of a JWT-shaped ``token``, unverified; ``{}`` for any other shape."""
    parts = token.split('.')
    if len(parts) != 3:
        return {}
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4)))
    except (binascii.Error, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_claim(claims: dict[str, Any], name: str) -> str | None:
    value = claims.get(name)
    return value if isinstance(value, str) and value else None


def _hold_port_exclusively(sock: socket.socket, *, is_win: bool | None = None) -> None:
    """On Windows, forbid any other socket to bind the port ``sock`` is about to take, whatever options that socket sets.

    ``SO_REUSEADDR`` there lets a second socket bind a port in use, and switching it off on the listener does not stop
    the other socket from asking for it; ``SO_EXCLUSIVEADDRUSE`` does, and it must be set before the bind. Elsewhere
    ``SO_REUSEADDR`` only allows a port in ``TIME_WAIT`` to be reused and the system already refuses a second listener.
    """
    if _platform.is_windows() if is_win is None else is_win:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]  # a Windows-only constant


class _Listener(ThreadingHTTPServer):
    """The loopback listener. One callback is accepted; its request thread holds the browser until the caller responds."""

    daemon_threads = True
    #: The stdlib server sets this to 1. The bind goes through :func:`_hold_port_exclusively`, which is what keeps a second socket out on Windows.
    allow_reuse_address = False

    def __init__(self, state: str, success_url: str) -> None:
        super().__init__((_LOOPBACK, 0), _CallbackHandler)
        self.success_url = success_url
        self.expected_host = f'{_LOOPBACK}:{self.server_address[1]}'
        self.state = state.encode('ascii')
        self.callbacks: queue.Queue[dict[str, list[str]]] = queue.Queue(maxsize=1)
        self.claimed = False
        self.claim_lock = threading.Lock()
        #: The page the browser gets once :meth:`Authorization.respond` is called.
        self.outcome: Literal['signed_in', 'failed'] = 'failed'
        self.responded = threading.Event()
        self.page_written = threading.Event()

    def server_bind(self) -> None:
        _hold_port_exclusively(self.socket)
        super().server_bind()

    def handle_error(self, request: object, client_address: object) -> None:
        """Silenced: the default prints a traceback, and a request's failure is no failure of the sign-in."""

    @property
    def redirect_uri(self) -> str:
        return f'http://{self.expected_host}{CALLBACK_PATH}'

    def claim(self, handler: BaseHTTPRequestHandler) -> dict[str, list[str]] | None:
        """The callback's query when ``handler`` is the one expected callback, claiming it; ``None`` for anything else."""
        try:
            parts = urlsplit(handler.path)
        except ValueError:
            return None
        if handler.command != 'GET' or parts.path != CALLBACK_PATH or handler.headers.get_all('Host') != [self.expected_host]:
            return None
        query = parse_qs(parts.query, keep_blank_values=True)
        states = query.get('state', [])
        if len(states) != 1 or not hmac.compare_digest(states[0].encode('utf-8', 'replace'), self.state):
            return None
        with self.claim_lock:
            if self.claimed:
                return None
            self.claimed = True
        return query


class _CallbackHandler(BaseHTTPRequestHandler):
    server: _Listener
    timeout = _LISTENER_REQUEST_TIMEOUT

    def _route(self) -> None:
        query = self.server.claim(self)
        if query is None:
            self._page('not_found', 404)
            return
        self.server.callbacks.put(query)
        answered = self.server.responded.wait(RESPONSE_TIMEOUT)
        try:
            if answered and self.server.outcome == 'signed_in':
                self._redirect(self.server.success_url)
            else:
                self._page('failed', 200)
        finally:
            self.server.page_written.set()

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _route  # noqa: N815 - the names http.server dispatches to

    def _page(self, state: _pages.State, status: int) -> None:
        body = _pages.page(state)
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._security_headers()
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        """A 303 to ``location``, a constant the caller chose: it carries nothing from the request."""
        self.send_response(303)
        self.send_header('Location', location)
        self.send_header('Content-Length', '0')
        self._security_headers()
        self.end_headers()

    def _security_headers(self) -> None:
        for name, value in _pages.SECURITY_HEADERS.items():
            self.send_header(name, value)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the stdlib's parameter name
        """Silenced: the request line holds the authorization code."""


def _open_quietly(opener: Callable[[str], object], url: str) -> None:
    # The url is the caller's to show as well, so a browser that cannot be opened is not an error here.
    try:
        opener(url)
    except Exception:  # noqa: BLE001, S110 - any failure of a third-party opener leaves the printed url as the way in
        pass


class Authorization:
    """One sign-in: discovery and the listener on entry, the callback and the code exchange in :meth:`wait`.

    ``open_browser`` is called once, on its own thread, with the authorize url (default: :func:`webbrowser.open`); it
    may block. Leaving the ``with`` block answers a browser still waiting with the failure page and closes the listener.

    ``success_url`` is where the browser is sent (a 303) after ``respond(succeeded=True)``: the caller's own page, so this
    module has no success page. It is a constant of the caller, never read from a request.
    """

    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        scopes: Sequence[str],
        open_browser: Callable[[str], object] | None = None,
        consent_timeout: float = DEFAULT_CONSENT_TIMEOUT,
        http_timeout: float = DEFAULT_HTTP_TIMEOUT,
        success_url: str,
    ) -> None:
        self.success_url = success_url
        self.issuer = issuer
        self.client_id = client_id
        self.scopes = tuple(scopes)
        self.open_browser = webbrowser.open if open_browser is None else open_browser
        self.consent_timeout = consent_timeout
        self.http_timeout = http_timeout
        self.authorization_url = ''
        self._metadata: dict[str, Any] = {}
        self._listener: _Listener | None = None
        self._client: _OAuth2Client | None = None
        self._verifier = ''

    def __enter__(self) -> Self:
        _require_secure(self.issuer, 'issuer')
        if urlsplit(self.issuer).query:
            raise DiscoveryError('the issuer must not carry a query')
        _require_secure(self.success_url, 'success url')
        state = secrets.token_urlsafe(32)
        self._verifier = secrets.token_urlsafe(64)
        try:
            self._listener = _Listener(state, self.success_url)
        except OSError as failure:
            raise ListenerError(f'the loopback listener on {_LOOPBACK} could not be started ({failure.strerror or type(failure).__name__})') from None
        threading.Thread(target=self._listener.serve_forever, kwargs={'poll_interval': 0.05}, name='oauth-loopback', daemon=True).start()
        try:
            self._client = _OAuth2Client(
                client_id=self.client_id,
                token_endpoint_auth_method='none',  # noqa: S106 - the auth method of a public client, not a password
                revocation_endpoint_auth_method='none',
                code_challenge_method='S256',
                redirect_uri=self._listener.redirect_uri,
                scope=' '.join(self.scopes),
                timeout=self.http_timeout,
                follow_redirects=False,
            )
            self._metadata = self._discover()
            self.authorization_url, _ = self._client.create_authorization_url(
                self._metadata['authorization_endpoint'], state=state, code_verifier=self._verifier
            )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        if self._listener is not None:
            self.respond(succeeded=False)
            if self._listener.claimed:
                self._listener.page_written.wait(_PAGE_WRITE_TIMEOUT)
            self._listener.shutdown()
            self._listener.server_close()
            self._listener = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def _discover(self) -> dict[str, Any]:
        """Fetch and check the issuer's metadata. The body is read in pieces and refused past the size limit, so a hostile server cannot fill memory."""
        try:
            with (
                httpx2.Client(timeout=self.http_timeout, follow_redirects=False) as plain,
                plain.stream('GET', _metadata_url(self.issuer)) as response,
            ):
                if response.status_code != 200:
                    raise DiscoveryError(f'the authorization server metadata could not be fetched (HTTP {response.status_code})')
                content = bytearray()
                for chunk in response.iter_bytes():
                    content += chunk
                    if len(content) > _MAX_METADATA_BYTES:
                        raise DiscoveryError('the authorization server metadata is too large')
        except httpx2.HTTPError as failure:
            raise DiscoveryError(f'the authorization server metadata could not be fetched ({type(failure).__name__})') from None
        try:
            metadata = json.loads(content)
        except ValueError:
            metadata = None
        if not isinstance(metadata, dict):
            raise DiscoveryError('the authorization server metadata is not a JSON object')
        if metadata.get('issuer') != self.issuer:
            raise IssuerMismatchError('the authorization server metadata names another issuer than the one it was fetched for')
        for name in ('authorization_endpoint', 'token_endpoint'):
            _require_secure(metadata.get(name), name)
        # Optional: a JSON null is the same as no field, and :meth:`revoke` then says there is no endpoint.
        if metadata.get('revocation_endpoint') is not None:
            _require_secure(metadata['revocation_endpoint'], 'revocation_endpoint')
        # A server that lists no methods is taken to support S256, as Clerk does; one that lists methods must list S256.
        methods = metadata.get('code_challenge_methods_supported')
        if methods is not None and (not isinstance(methods, list) or 'S256' not in methods):
            raise DiscoveryError('the authorization server does not support PKCE S256')
        return metadata

    def respond(self, *, succeeded: bool) -> None:
        """Answer the browser's callback with the redirect to the success url, or with the failure page. Only the first answer counts."""
        listener = self._listener
        if listener is not None and not listener.responded.is_set():
            listener.outcome = 'signed_in' if succeeded else 'failed'
            listener.responded.set()

    def _next_callback(self, listener: _Listener) -> dict[str, list[str]]:
        deadline = time.monotonic() + self.consent_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CallbackTimeoutError(f'no sign-in arrived within {self.consent_timeout:g} seconds')
            try:
                return listener.callbacks.get(timeout=min(remaining, _POLL_INTERVAL))
            except queue.Empty:
                continue

    def wait(self) -> Tokens:
        """Open the browser, wait for the callback, check it and exchange the code."""
        listener, client = self._listener, self._client
        if listener is None or client is None:
            raise RuntimeError('Authorization.wait() runs inside the with block')
        threading.Thread(target=_open_quietly, args=(self.open_browser, self.authorization_url), name='oauth-browser', daemon=True).start()
        query = self._next_callback(listener)
        values = {name: entries[0] for name, entries in query.items() if len(entries) == 1}
        issuer = values.get('iss')
        if issuer is not None and issuer != self.issuer:
            raise IssuerMismatchError('the sign-in answer names another issuer')
        if 'error' in values:
            if values['error'] == 'access_denied':
                raise ConsentDeniedError('the sign-in was declined at the consent screen')
            error = _error_code(values['error'])
            raise AuthorizationError(f'the authorization server refused the sign-in ({error})', error)
        if issuer is None:
            raise IssuerMismatchError('the sign-in answer does not name its issuer (iss)')
        if not values.get('code'):
            raise AuthorizationError('the sign-in answer carries no authorization code')
        return self._exchange(client, values['code'])

    def _exchange(self, client: _OAuth2Client, code: str) -> Tokens:
        statuses: list[int] = []

        def record(response: Any) -> Any:  # noqa: ANN401 - a response of the package in _http
            statuses.append(response.status_code)
            return response

        def last_status() -> int | None:
            return statuses[-1] if statuses else None

        client.register_compliance_hook('access_token_response', record)
        try:
            token = client.fetch_token(self._metadata['token_endpoint'], grant_type='authorization_code', code=code, code_verifier=self._verifier)
        except _AuthlibOAuthError as refused:
            status = last_status()
            error = _error_code(refused.error)
            raise TokenEndpointError(f'the token request was refused (HTTP {status}, {error})', status=status, error=error) from None
        except httpx2.HTTPStatusError as failure:
            status = failure.response.status_code
            raise TokenEndpointError(f'the token request failed (HTTP {status})', status=status) from None
        except (httpx2.HTTPError, ValueError, TypeError) as failure:  # TypeError: Authlib reads expires_in as a number
            raise TokenEndpointError(f'the token request failed ({type(failure).__name__})', status=last_status()) from None
        access = token.get('access_token')
        status = last_status()
        refresh = token.get('refresh_token')
        kind = token.get('token_type')
        unusable = (
            f'the token answer carries no access token (HTTP {status})'
            if not isinstance(access, str) or not access
            else f'the token answer is not a bearer token (HTTP {status})'
            if kind is not None and str(kind).lower() != 'bearer'
            else None
        )
        if unusable is not None:
            # A refresh token that arrived with an unusable answer is still a live credential: end it before failing.
            if isinstance(refresh, str) and refresh:
                with contextlib.suppress(RevocationError):
                    self.revoke(refresh, 'refresh_token')
            raise TokenEndpointError(unusable, status=status)
        claims = _claims(access)
        return Tokens(
            access_token=access,
            refresh_token=refresh if isinstance(refresh, str) and refresh else None,
            subject=_string_claim(claims, 'sub'),
            organization=_string_claim(claims, 'org_id'),
        )

    def revoke(self, token: str, hint: str) -> None:
        """Revoke ``token`` (RFC 7009); ``hint`` is ``'refresh_token'`` or ``'access_token'``."""
        client = self._client
        endpoint = self._metadata.get('revocation_endpoint')
        if client is None or not endpoint:
            raise RevocationError('the authorization server has no revocation endpoint')
        try:
            response = client.revoke_token(endpoint, token=token, token_type_hint=hint)
        except httpx2.HTTPError as failure:
            raise RevocationError(f'the token could not be revoked ({type(failure).__name__})') from None
        if response.status_code != 200:
            raise RevocationError(f'the token could not be revoked (HTTP {response.status_code})')
