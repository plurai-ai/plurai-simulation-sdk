"""HTTP client for the plurai mock service.

Imports only the stdlib and the stdlib-only ``turn_context``, ``_http_retry``, ``_endpoints`` and ``client_handshake``
modules: it runs inside the customer's agent process.
"""

from __future__ import annotations

import errno
import http.client
import io
import json
import logging
import math
import os
import re
import select
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, Literal

from plurai import client_handshake
from plurai._endpoints import bearer_for, is_loopback_url
from plurai._http_retry import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    RATE_LIMIT_RETRIES,
    RATE_LIMITED_STATUS,
    TRANSIENT_RETRIES,
    TRANSIENT_STATUS_CODES,
    is_name_not_found,
    rate_limit_wait,
)
from plurai._mocking.turn_context import note_harness_fault, world_write_source
from plurai._platform import is_windows
from plurai.capacity import BACKEND_REFUSED_PHRASE, CAPACITY_CODE, UnavailableBlock, unavailable_block_from_body
from plurai.credit import CREDIT_KINDS, resets_at_of

logger = logging.getLogger(__name__)

# Compatible plurai-mockenv MAJOR version. Single source of truth for the SDK.
# The gate is a SOFT MAJOR CHECK: the server's major must equal this; any
# patch/minor within the same major (1.x) is accepted without a per-release SDK
# edit. Keep in sync with the dependency pin in pyproject.toml:
#   dependencies → "plurai-mockenv==<MAJOR>.y.z"
_COMPATIBLE_MOCKENV_MAJOR = 1

#: The minor within that major at which ``POST /session/turn`` appears. The route is
#: additive, so an older server is fully usable — it simply cannot be told which turn
#: is open, and every event an external caller produces stamps turn 0. The push is
#: therefore gated on this minor rather than raised on, unlike the major above.
_SESSION_TURN_MIN_MINOR = 2

#: The minor at which ``POST /session/turn`` also takes the conversation so far (``dialog``) and
#: the results of the tools the harness really executed (``live_tool_calls``), which a
#: runtime-generated tool answer reads as interpretation context. Gated like the route itself: an
#: older server gets the turn alone, and one warning.
_SESSION_DIALOG_MIN_MINOR = 4

#: The url this package falls back to when NOBODY configured one. The backend hands out
#: the identical literal for the same reason, so it reaches a local/mockless child as a "requested" url that
#: no human ever requested. Named here so the stub launcher can recognise it as the absence
#: it is instead of announcing it (see ``_mocking.local``).
DEFAULT_MOCK_SERVICE_URL = 'http://localhost:8199'

#: Backend ``code`` discriminators on a ``/mock/<env_id>`` 503 (owned by the backend;
#: the vocabulary is the "/mock route-state").
#: ``restarting`` self-heals in seconds — the liveness watcher is already
#: bringing the subprocess back. ``awaiting_edit_finalize`` does NOT: the env is
#: held because an interrupted edit session still owes a finalize, which is
#: blocked on a HUMAN up to the session TTL. Retrying the second one burns the
#: whole budget on a wait that cannot end, so it is the one 503 we fail fast on.
MOCK_503_RESTARTING = 'restarting'
MOCK_503_AWAITING_EDIT_FINALIZE = 'awaiting_edit_finalize'

#: Bounded retry for the answers that clear by themselves, applied to EVERY mock call (this client
#: is urllib — the orchestrator's tenacity policy covers only the httpx REST client, so without
#: this retry a ``restarting`` 503 becomes an instant stub fallback and fails the scenario on a
#: condition that clears in seconds). The answers are a gateway 502, 503 or 504 and a connection
#: failure (refused, reset, unresolvable): the backend is not there for a moment, and a backend upgrade
#: is exactly that for about 35 seconds.
#:
#: A POST repeats only what was never forwarded to the mock: a 503 (the proxy answers it before it forwards anything), a 502
#: in which the proxy says it could not connect to the mock, and a connection that failed while it was being made. Any other
#: 502, a 504 and a connection that broke after the request went out may follow a tool call the mock already ran, so a repeat
#: could run it twice: those are never re-sent, and they fail the scenario at once, which is no spent budget and no outage.
#: A host name that does not resolve is nobody's outage and is never repeated.
#:
#: A GET also gets this budget when its read times out: every GET route on the mock is a pure read,
#: so a re-read returns the same log or state and cannot apply anything twice. A POST whose read
#: timed out may already have been applied, so it is never retried on that; a timeout while
#: connecting is retried for every method, because nothing reached the backend.
#:
#: One attempt and ``TRANSIENT_RETRIES`` retries (six waits at the defaults: 1 + 2 + 4 + 8 + 16 + 30 = 61s of back-off) outlast the 35s a
#: backend upgrade rejects every request for, and stay well under the per-agent-turn budget
#: (1560s qualification / 2160s full): a redeploy that outlives it
#: is a restart the SCENARIO cannot absorb, and it escalates to the re-drive path,
#: whose readiness gate has the generous (~300s) bound.
_MOCK_OUTAGE_RETRY_ATTEMPTS = TRANSIENT_RETRIES + 1
_MOCK_OUTAGE_BACKOFF_BASE = BACKOFF_BASE_SECONDS
_MOCK_OUTAGE_BACKOFF_CAP = BACKOFF_CAP_SECONDS
#: The sum of those waits. With the per-attempt ceiling it bounds the whole retry loop in wall
#: clock (see ``_with_retry``).
_MOCK_OUTAGE_BACKOFF_WINDOW = sum(min(_MOCK_OUTAGE_BACKOFF_CAP, _MOCK_OUTAGE_BACKOFF_BASE * 2**n) for n in range(_MOCK_OUTAGE_RETRY_ATTEMPTS - 1))

#: Per-attempt read ceilings for the control routes: ``/session/*``, ``/state/*``,
#: ``/mock/observe`` and every GET. They do bookkeeping on the mock's own state and answer in
#: milliseconds, so a short ceiling turns a dead hop into a fast, attributable failure.
_GET_TIMEOUT_SECONDS = 10.0
_POST_TIMEOUT_SECONDS = 30.0

#: ``/mock/execute`` runs the tool itself. A model-backed mock (a handler that calls an LLM)
#: can take well over a minute, and a SYNCHRONOUS workflow tool
#: (one whose answer a backend workflow computes inside the call) may take up to 240s. The backend
#: chain is ordered so the innermost bound always fires first: the sync workflow call (240s) <
#: the hosted ``/mock`` proxy's own upstream bound (270s) < this
#: ceiling (300s). So on a hosted env the proxy, which tracks whether the deployment is alive,
#: ends an overrun call with its own typed 502 instead of this client giving up first and turning
#: a slow-but-live mock into a transport failure. The 30s on top covers the proxy's work before it
#: forwards and the leg back. A loopback mock has no proxy in front, so there this ceiling is the
#: binding one. It also sits above the backend's runtime deadline for a runtime-generated tool
#: answer (a model call with one grounding repair, kept below the hosted proxy's bound): a timeout here would turn that answer into a
#: schema stub. The Node bridge (``node/code_hook.cjs``) carries the same default.
MOCK_EXECUTE_TIMEOUT_ENV = 'PLURAI_MOCK_EXECUTE_TIMEOUT'
DEFAULT_MOCK_EXECUTE_TIMEOUT = 300.0
#: The largest override accepted. One tool call runs inside one agent turn, whose budget is 26-36
#: minutes by default, so a larger value is a unit slip (milliseconds written as seconds) rather
#: than a ceiling anyone needs. Past about 1e10 seconds the socket layer cannot represent the
#: value at all and raises on every call, which would stub every tool call of the run.
MAX_MOCK_EXECUTE_TIMEOUT = 3600.0

# Unusable values already reported, so a client built per call (``code_intercept``) names each
# one once instead of on every dispatch.
_warned_execute_timeouts: set[str] = set()


def mock_execute_timeout() -> float:
    """The ``/mock/execute`` ceiling in seconds: ``$PLURAI_MOCK_EXECUTE_TIMEOUT``, else the default.

    Read from the process environment rather than :mod:`plurai.settings`, because this module is
    stdlib-only and runs in the agent's process and in processes it spawns, all of which inherit
    the variable. The value must be a finite number of seconds above zero, the rule the
    ``NETWORK__*`` timeouts enforce, and at most :data:`MAX_MOCK_EXECUTE_TIMEOUT`. Anything else
    falls back to the default and is named on stderr: a mock dispatch must not die on a malformed
    knob, and a silent fallback would leave an operator believing a raised ceiling was in force.
    """
    raw = (os.environ.get(MOCK_EXECUTE_TIMEOUT_ENV) or '').strip()
    if not raw:
        return DEFAULT_MOCK_EXECUTE_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isfinite(value) and 0 < value <= MAX_MOCK_EXECUTE_TIMEOUT:
        return value
    if raw not in _warned_execute_timeouts:
        _warned_execute_timeouts.add(raw)
        print(
            f'  [plurai] WARNING {MOCK_EXECUTE_TIMEOUT_ENV}={raw!r} is not a finite positive number of seconds '
            f'up to {MAX_MOCK_EXECUTE_TIMEOUT:.0f}; '
            f'/mock/execute uses the default {DEFAULT_MOCK_EXECUTE_TIMEOUT:.0f}s.',
            file=sys.stderr,
        )
    return DEFAULT_MOCK_EXECUTE_TIMEOUT


#: How a mock call failed when no HTTP status came back. ``refused``: nothing answered — the
#: connection was refused or reset, the host or network was unreachable, or the host did not resolve (the signature of a mock that is
#: not there right now). ``timeout``: the call was sent and no answer came within the ceiling.
#: ``non_json``: something answered 2xx with a body that is not JSON, i.e. not a plurai mock.
#: ``other``: any other transport failure.
TransportFailure = Literal['refused', 'timeout', 'non_json', 'other']


#: A host or network the connect could not route to: nothing answered, like a refused connection. Read while connecting only.
_NO_ROUTE_ERRNOS = frozenset({errno.EHOSTUNREACH, errno.ENETUNREACH})


def transport_failure(exc: BaseException) -> TransportFailure:
    """The :data:`TransportFailure` an exception raised while calling the mock stands for.

    Typed by exception class, never by message text. ``urllib`` wraps a connect-time failure in
    ``URLError`` with the cause on ``reason``; a failure while reading the answer arrives bare.
    """
    cause: object = exc.reason if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError) else exc
    if isinstance(cause, TimeoutError | socket.timeout):
        return 'timeout'
    if isinstance(cause, ConnectionRefusedError | ConnectionResetError | socket.gaierror):
        return 'refused'
    if cause is not exc and isinstance(cause, OSError) and cause.errno in _NO_ROUTE_ERRNOS:
        return 'refused'  # while connecting only: the same errno on a read says nothing about whether the request was heard
    if isinstance(cause, json.JSONDecodeError):
        return 'non_json'
    return 'other'


def _failed_connecting(exc: BaseException) -> bool:
    """Whether *exc* is a failure of making the connection, so the request was never sent.

    ``urllib`` wraps both a connect-phase failure and a send-phase one (a reset or a broken pipe while the request goes out) in
    ``URLError``, so the wrapper alone does not say which: the reason does. A refused connection, a name that does not
    resolve, a timeout and the no-route errnos are connect-phase; a reset or a broken pipe may follow a request the mock
    already heard.
    """
    if not isinstance(exc, urllib.error.URLError) or isinstance(exc, urllib.error.HTTPError):
        return False
    reason = exc.reason
    return isinstance(reason, ConnectionRefusedError | socket.gaierror | TimeoutError) or (isinstance(reason, OSError) and reason.errno in _NO_ROUTE_ERRNOS)


class MockServiceError(Exception):
    """Raised by MockServiceClient._get/_post on HTTP or transport failure.

    Carries the HTTP ``status_code`` (an int) when the server answered with a
    non-2xx response, or ``None`` when the failure was at the transport level.
    High-level methods propagate this exception rather than returning an in-band
    ``{'error': ..., 'status_code': ...}`` dict.

    ``transport`` is the typed transport failure (:data:`TransportFailure`) when no status came
    back, and ``None`` when the service answered — with a status, or with its own in-band error
    envelope. REQUIRED at every raise site: "how did this call fail" is what decides whether a
    re-drive can help, and a defaulted value would let a new raise site read as one that answered.

    ``code`` is the backend's own machine-readable discriminator parsed off the
    error body (``restarting`` / ``awaiting_edit_finalize`` / ``needs_redeploy``),
    or ``None`` when the body carried none. It exists so callers branch on a
    TYPED field instead of substring-matching the human ``error`` prose — two
    503s with opposite remedies are otherwise indistinguishable.

    ``retry_after`` is the raw ``Retry-After`` header of the answer, when it carried one; the retry
    loop reads it on a 429. ``resets_at`` is the credit block's reset time (the free tier's), or the capacity block's, when
    the body carried one; ``reset_source`` is the capacity block's (``named_reset`` / ``ttl``), when it carried one.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        code: str | None,
        transport: TransportFailure | None,
        retry_after: str | None = None,
        resets_at: str | None = None,
        reset_source: str | None = None,
        never_connected: bool = False,
        never_forwarded: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.transport = transport
        self.retry_after = retry_after
        self.resets_at = resets_at
        self.reset_source = reset_source
        #: The request never reached the backend: the connection failed while it was being made (refused, unresolvable or timed
        #: out), so a repeat cannot apply anything twice, whatever the method.
        self.never_connected = never_connected
        #: The backend's proxy answered that it could not connect to the mock, so the tool call never ran: a repeat cannot
        #: apply anything twice, whatever the method.
        self.never_forwarded = never_forwarded
        #: Set by the retry loop when its budget was spent on a transient failure: the mock is unavailable, not a mock that
        #: went away. :func:`plurai.capacity.unavailable_block_of` reads it.
        self.unavailable: UnavailableBlock | None = None


def _json_or_none(body_text: str) -> object:
    """*body_text* as JSON, or ``None`` when it is not."""
    try:
        return json.loads(body_text)
    except (ValueError, TypeError):
        return None


def parse_error_code(body_text: str) -> str | None:
    """Best-effort: the ``code`` discriminator inside a JSON error body, else ``None``.

    Never raises — a non-JSON body (an HTML error page from a front, an empty
    body from an old backend) simply carries no code.
    """
    try:
        payload = json.loads(body_text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get('code')
    return code if isinstance(code, str) else None


def parse_resets_at(body_text: str) -> str | None:
    """Best-effort: the ``resets_at`` of a JSON error body (the credit block's reset time), else ``None``. Never raises."""
    try:
        payload = json.loads(body_text)
    except (ValueError, TypeError):
        return None
    return resets_at_of(payload)


def parse_reset_source(body_text: str) -> str | None:
    """Best-effort: the ``reset_source`` of a JSON error body (the capacity block's), else ``None``. Never raises."""
    try:
        payload = json.loads(body_text)
    except (ValueError, TypeError):
        return None
    source = payload.get('reset_source') if isinstance(payload, dict) else None
    return source if isinstance(source, str) and source.strip() else None


def _parse_major_minor(version: str) -> tuple[int, int]:
    """Best-effort ``(major, minor)`` from a version string; a string that does not
    match reads as ``(0, 0)``.

    Degrading beats raising here: this second, finer read only decides whether one
    optional call goes out, so an odd version string costs a feature, not the run. A
    suffix on the patch or on the minor itself (``1.2rc1``) is ignored, not fatal.
    """
    m = re.match(r'(\d+)\.(\d+)', version)
    return (int(m[1]), int(m[2])) if m else (0, 0)


class _Reply:
    """An answered request: the body is already read, so the connection can go back to the pool."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses every redirect, so a 3xx surfaces as :class:`urllib.error.HTTPError` and the bearer never follows it off the mock's origin."""

    def redirect_request(self, *_: object, **__: object) -> None:
        return None


class _KeepAliveOpener:
    """The ``urlopen`` the client calls, over connections that outlive one request.

    A tool call is one small request, and a run makes hundreds of them to the same host, so a new
    TCP (and TLS) handshake per call is most of their cost. Idle connections wait in a pool keyed
    by origin; a request takes one out, so two threads never share a connection mid-exchange.

    Same contract as :func:`urllib.request.urlopen` for what the client reads: the body of a 2xx,
    :class:`urllib.error.HTTPError` for any other status (a redirect is not followed, which also keeps
    the bearer from following one off the mock's origin), the socket error itself for a transport
    failure. A request that ``urllib`` would send through a proxy named in the environment keeps going
    through a ``urllib`` opener (no pooling), because the proxy handling lives there; it refuses redirects too.
    An opener built with ``no_proxy`` never takes that route: on Windows ``urllib`` reads the proxy from the registry and
    bypasses it for a loopback address only when the override list names it, so a call to the in-process mock would go to
    a proxy that cannot reach this machine's loopback.

    A pooled connection the server closed while it was idle is detected before it is used and replaced.
    One that dies in the middle of an exchange surfaces as a transport failure: the
    request may have been applied, so it is not silently repeated.
    """

    Request = urllib.request.Request

    def __init__(self, *, no_proxy: bool = False) -> None:
        self._no_proxy = no_proxy
        self._idle: dict[tuple[str, str, int], list[http.client.HTTPConnection]] = {}
        self._proxy_routes: dict[tuple[str, str], bool] = {}
        self._opener: urllib.request.OpenerDirector | None = None
        self._lock = threading.Lock()

    def urlopen(self, req: urllib.request.Request, timeout: float) -> _Reply:
        # Stamped here, before the route splits, so the pooled path and the proxied ``urllib`` path both carry it.
        req.add_unredirected_header(client_handshake.HEADER, client_handshake.header_value())
        # Some HTTP front ends refuse urllib's default ``Python-urllib/*`` agent, so every request names the SDK as its agent,
        # as :mod:`plurai._http` does. Do not remove it.
        if not req.has_header(client_handshake.USER_AGENT_HEADER.capitalize()):  # ``Request`` keeps header names capitalized
            req.add_unredirected_header(client_handshake.USER_AGENT_HEADER, client_handshake.user_agent())
        parts = urllib.parse.urlsplit(req.full_url)
        scheme, host = parts.scheme, parts.hostname or ''
        if scheme not in ('http', 'https') or self._proxied(scheme, host):
            with self._proxy_opener().open(req, timeout=timeout) as resp:  # the scheme is http(s) or urllib refuses it
                return _Reply(resp.read())
        port = parts.port or (443 if scheme == 'https' else 80)
        key = (scheme, host, port)
        target = parts.path or '/'
        if parts.query:
            target = f'{target}?{parts.query}'
        headers = dict(req.header_items())
        conn = self._checkout(key, timeout)
        try:
            if conn.sock is None:
                # Connected apart from the request, so a failure here is the connection never made, whatever the method: the
                # proxied path (urllib) wraps it in ``URLError`` the same way. A failure after this point may follow a request that was sent.
                try:
                    conn.connect()
                except OSError as connect_failure:
                    raise urllib.error.URLError(connect_failure) from connect_failure
            conn.request(req.get_method(), target, body=req.data, headers=headers)
            response = conn.getresponse()
            body = response.read()
        except BaseException:
            conn.close()
            raise
        if response.will_close:
            conn.close()
        else:
            self._checkin(key, conn)
        if not 200 <= response.status < 300:
            raise urllib.error.HTTPError(req.full_url, response.status, response.reason, response.headers, io.BytesIO(body))
        return _Reply(body)

    def _proxy_opener(self) -> urllib.request.OpenerDirector:
        """The ``urllib`` opener for the proxied route: the environment's proxies, no redirects. Built on first use (it loads CA roots)."""
        if self._opener is None:
            self._opener = urllib.request.build_opener(_NoRedirect)
        return self._opener

    def _proxied(self, scheme: str, host: str) -> bool:
        """Whether the environment routes this origin through a proxy; read once per origin, since the lookup can be slow."""
        if self._no_proxy:
            return False
        key = (scheme, host)
        if key not in self._proxy_routes:
            self._proxy_routes[key] = scheme in urllib.request.getproxies() and not urllib.request.proxy_bypass(host)
        return self._proxy_routes[key]

    def _checkout(self, key: tuple[str, str, int], timeout: float) -> http.client.HTTPConnection:
        scheme, host, port = key
        while True:
            with self._lock:
                idle = self._idle.get(key)
                conn = idle.pop() if idle else None
            if conn is None:
                factory = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
                return factory(host, port, timeout=timeout)
            if self._is_usable(conn):
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                return conn
            conn.close()

    def _checkin(self, key: tuple[str, str, int], conn: http.client.HTTPConnection) -> None:
        with self._lock:
            self._idle.setdefault(key, []).append(conn)

    @staticmethod
    def _is_usable(conn: http.client.HTTPConnection) -> bool:
        """Whether an idle connection is still open: a socket that is readable while idle has been closed (or sent garbage)."""
        sock = conn.sock
        if sock is None:
            return False
        try:
            return not select.select([sock], [], [], 0)[0]
        except (OSError, ValueError):
            return False

    def close(self) -> None:
        with self._lock:
            pooled = [conn for conns in self._idle.values() for conn in conns]
            self._idle.clear()
        for conn in pooled:
            conn.close()


#: The backend's mock proxy answers 502 with this body when its own connection to the mock subprocess failed (the mock is
#: restarting), before the request was sent. Any other reason it names (``no answer within ...``, ``Server disconnected``)
#: followed a request the mock may have run.
_PROXY_CONNECT_FAILED = re.compile(r'^mock service for \S+ unreachable: Cannot connect to host\b')


def _proxy_never_reached_mock(payload: object) -> bool:
    """Whether a 502 body is the proxy saying it could not connect to the mock, so the call was never forwarded."""
    error = payload.get('error') if isinstance(payload, dict) else None
    return isinstance(error, str) and _PROXY_CONNECT_FAILED.match(error) is not None


def _stalled_read(exc: MockServiceError, *, idempotent: bool) -> bool:
    """A read of a pure read route that timed out: slow, and heard by the backend, so never an outage."""
    return idempotent and exc.transport == 'timeout' and not exc.never_connected


def _outage_rule(exc: MockServiceError, *, idempotent: bool) -> Literal['wait', 'raise']:
    """Whether a failed mock call is waited out under the transient budget (``'wait'``) or raised on the spot (``'raise'``).

    Only what the backend cannot have taken is repeated on a POST: a 503 (the proxy answers it before it forwards anything), a
    502 the proxy marked as never connected to the mock, and a connection that failed while it was being made. A 502, a 504 and a
    connection that broke after the request went out may follow a tool call the mock already ran, so a POST never repeats them
    and they fail the scenario unretried. A GET is a pure read: it also waits out a 502, a 504, a broken connection and a read
    that timed out. Never waited: a 503 held for a human, a credit refusal, and a host name that does not resolve.
    """
    connect_failed = exc.never_connected and exc.transport in ('refused', 'timeout') and not is_name_not_found(exc)
    broke_after_sending = exc.transport == 'refused' and not exc.never_connected
    held_for_a_human = exc.code == MOCK_503_AWAITING_EDIT_FINALIZE  # a standing refusal on any status: waiting does not clear it
    repeatable_status = exc.status_code == 503 or exc.never_forwarded or (idempotent and exc.status_code in TRANSIENT_STATUS_CODES)
    clears_by_itself = repeatable_status and not held_for_a_human and exc.code not in CREDIT_KINDS
    waits = connect_failed or _stalled_read(exc, idempotent=idempotent) or clears_by_itself or (idempotent and broke_after_sending)
    return 'wait' if waits else 'raise'


#: What a read that spent its retry budget, or kept stalling, says on stderr before it raises.
_READ_GAVE_UP = '  [plurai] the mock did not answer this read after repeated tries; giving up on it'


class MockServiceClient:
    """HTTP client. Same API whether server is local or remote."""

    def __init__(self, base_url: str = DEFAULT_MOCK_SERVICE_URL, *, is_win: bool | None = None):
        self.base_url = base_url.rstrip('/')
        windows = is_windows() if is_win is None else is_win
        self._urllib = _KeepAliveOpener(no_proxy=windows and is_loopback_url(self.base_url))
        # When the mock is served through the backend's /mock proxy (a public_base_url deployment)
        # the request must carry the backend's bearer, since the proxy is behind its bearer
        # middleware. The resolved token goes only to a mock under the backend's origin or on
        # loopback; a mock anywhere else, or no configuration at all, gets no header.
        token = bearer_for(self.base_url)
        self._auth_headers = {'Authorization': f'Bearer {token}'} if token else {}
        self._version_checked: bool = False
        # The version string /health reported, filled in by _check_version_compat.
        # None until the first check: an unverified server gets no optional traffic.
        self._server_version: str | None = None
        self._warned_no_session_turn: bool = False
        self._warned_no_session_dialog: bool = False
        self._execute_timeout = mock_execute_timeout()

    def close(self) -> None:
        """Close the pooled connections. The client stays usable: the next call opens a new one."""
        self._urllib.close()

    def _send(self, path: str, timeout: float, *, data: bytes | None = None, headers: dict[str, str]) -> dict:
        """One request, its 2xx body parsed. A non-2xx answer and a transport failure each raise :class:`MockServiceError`.

        Says nothing itself: :meth:`_with_retry` reports each failure, because only it knows whether the failure is one it retries.
        """
        try:
            req = self._urllib.Request(f'{self.base_url}{path}', data=data, headers=headers)
            with self._urllib.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            # The server answered with a non-2xx (e.g. the hosted /mock proxy's
            # 401 auth-reject, 410 needs-redeploy, 502 mock-dead, 503 restarting).
            body_text = self._read_http_error_body(e)
            code = parse_error_code(body_text)
            payload = _json_or_none(body_text)
            # The backend's typed unavailable block, read by where it sits in the body (never by its code). A credit
            # or capacity code on the same answer wins, as it does in every other reader.
            unavailable = None if code in CREDIT_KINDS or code == CAPACITY_CODE else unavailable_block_from_body(payload)
            # A credit refusal names its kind and nothing else: the status and the raw body (which carries an error id)
            # are not for the terminal.
            if code in CREDIT_KINDS:
                msg = f'{path}: refused for credit ({code})'
            elif code == CAPACITY_CODE or unavailable is not None:
                msg = f'refused, {BACKEND_REFUSED_PHRASE}'  # neither the body's message nor the route is for the terminal
            else:
                msg = f'HTTP {e.code} {path}: {body_text or e}'
            refused = MockServiceError(
                msg,
                status_code=e.code,
                code=code,
                transport=None,
                retry_after=e.headers.get('Retry-After') if e.headers else None,
                resets_at=parse_resets_at(body_text),
                reset_source=parse_reset_source(body_text),
                never_forwarded=e.code == 502 and _proxy_never_reached_mock(payload),
            )
            refused.unavailable = unavailable
            raise refused from e
        except Exception as e:
            msg = f'ERROR {path}: {e}'
            raise MockServiceError(
                msg,
                status_code=None,
                code=None,
                transport=transport_failure(e),
                never_connected=_failed_connecting(e),
            ) from e
        return self._parse_json(raw, path=path)

    def _post(self, path: str, body: dict, *, timeout: float = _POST_TIMEOUT_SECONDS) -> dict:
        data = json.dumps(body, default=str).encode()
        headers = {'Content-Type': 'application/json', **self._auth_headers}
        return self._with_retry(lambda: self._send(path, timeout, data=data, headers=headers), path=path, timeout=timeout, idempotent=False)

    def _get(self, path: str) -> Any:  # noqa: ANN401 - whatever JSON the server answered; callers check its shape
        return self._with_retry(
            lambda: self._send(path, _GET_TIMEOUT_SECONDS, headers=self._auth_headers), path=path, timeout=_GET_TIMEOUT_SECONDS, idempotent=True
        )

    def _with_retry(self, attempt: Callable[[], dict], *, path: str, timeout: float, idempotent: bool) -> dict:
        """Run *attempt*, retrying the answers that clear by themselves.

        A gateway 502, 503 or 504 and a connection failure are retried under the bounded budget above (what a POST may repeat
        is :func:`_outage_rule`'s), a 503 unless the body's ``code`` is
        ``awaiting_edit_finalize``: that marks a hold blocked on a human, so it fails fast on the
        first answer (see MOCK_503_* above). A 429 is a rate limiter refusing before it forwards, so
        it was never applied and is retried for every route: the wait follows ``Retry-After``, with
        jitter and a cap, under the SDK-wide budget in :mod:`plurai._http_retry`. The two budgets are
        independent, and a 429 is never reclassified as "not a mock". When *idempotent* (a GET, a pure
        read), a read that timed out is retried under the same budget, and a timeout while connecting is retried for every
        method: nothing reached the backend. Every other status and
        every other transport failure propagates on the first attempt, unchanged and untagged; so does the last failure once the
        budget is spent, and it then carries the unavailable block (:class:`~plurai.capacity.UnavailableBlock`): only a spent
        budget is an outage.

        *timeout* is the per-attempt ceiling, and it bounds the 503 loop in wall clock too: a
        retry starts only while the time already spent plus its wait fits in one ceiling
        plus the back-off window, so the whole call takes at most ``2 * timeout +
        _MOCK_OUTAGE_BACKOFF_WINDOW``. The 503s this loop exists for are the proxy's
        pre-forward answers and arrive in milliseconds, so the bound never binds on them.
        It binds when every attempt runs near its ceiling before a 503 comes back, which
        would otherwise cost that ceiling once per attempt: 7 x 300s on ``/mock/execute``,
        past the whole per-turn budget of a qualification pass. Bounded, it is at most
        2 x 300s + the back-off window, inside the turn's workflow allowance
        (``plurai.settings.WORKFLOW_TURN_ALLOWANCE_SECONDS``, 960s). A 429 arrives in milliseconds
        and its waits total at most ``RATE_LIMIT_RETRIES * 30s``, so it needs no such bound.

        Sleeps with ``time.sleep``: this client is synchronous by construction (it
        runs inside the customer's agent process, which may not have an event loop).
        """
        started = time.monotonic()
        outages = 0
        retries_429 = 0
        while True:
            try:
                return attempt()
            except MockServiceError as exc:
                typed = exc.code == CAPACITY_CODE or exc.unavailable is not None
                verdict = 'raise' if typed else _outage_rule(exc, idempotent=idempotent)
                self._report(exc, quiet=verdict == 'wait')
                if typed:
                    # The backend's typed capacity refusal answers every call the same way for now, and its typed
                    # unavailable refusal already outlasted the backend's own retries: neither is waited out, whatever status
                    # carries it (a 429 here would otherwise spend the throttle budget).
                    raise
                if exc.status_code == RATE_LIMITED_STATUS:
                    if retries_429 >= RATE_LIMIT_RETRIES:
                        raise
                    retries_429 += 1
                    delay = rate_limit_wait(retries_429, exc.retry_after)
                    print(
                        f'  [plurai] mock throttled (HTTP 429) on {path}; retrying in {delay:.1f}s (retry {retries_429}/{RATE_LIMIT_RETRIES})',
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
                if verdict == 'raise':
                    raise
                outages += 1
                stalled = _stalled_read(exc, idempotent=idempotent)
                delay = min(_MOCK_OUTAGE_BACKOFF_CAP, _MOCK_OUTAGE_BACKOFF_BASE * 2 ** (outages - 1))
                if outages >= _MOCK_OUTAGE_RETRY_ATTEMPTS or time.monotonic() - started + delay > timeout + _MOCK_OUTAGE_BACKOFF_WINDOW:
                    # The budget is spent: the last failure stands, with the status and code callers classify on. A gateway answer
                    # or a connection failure says the mock is unavailable; a read that kept stalling does not.
                    if not stalled:
                        exc.unavailable = UnavailableBlock('status' if exc.status_code is not None else 'transport')
                    if idempotent:
                        # The tries were quiet, so the failure that ends them says so: a read has no later line of its own, and the
                        # line carries no status, URL or id.
                        print(_READ_GAVE_UP, file=sys.stderr)
                    raise
                cause = 'mock read timed out' if stalled else f'mock {exc.status_code or exc.transport} ({exc.code or "no code"})'
                logger.debug('%s on %s; retrying in %.1fs (attempt %d/%d)', cause, path, delay, outages, _MOCK_OUTAGE_RETRY_ATTEMPTS)
                time.sleep(delay)

    @staticmethod
    def _report(exc: MockServiceError, *, quiet: bool) -> None:
        """Say what failed. A failure the loop retries, or one it gives up on as an outage, is for the logger: the run says so
        once, at its end. Any other failure goes to stderr, because stdout is reserved for the JSONL IPC handshake under the run
        orchestrator and any stray stdout write breaks it; the same holds for every other print in this module."""
        if quiet:
            logger.debug('%s', exc)
        else:
            print(f'  [plurai] {exc}', file=sys.stderr)

    @staticmethod
    def _parse_json(raw: bytes, *, path: str) -> dict:
        """A 2xx body as JSON, or :class:`MockServiceError` typed ``non_json``: whatever answered is
        not a plurai mock (an HTML page from a front, a wrong URL), which no wait and no re-drive fixes."""
        try:
            return json.loads(raw)
        except ValueError as e:
            msg = f'ERROR {path}: the answer is not JSON ({e})'
            print(f'  [plurai] {msg}', file=sys.stderr)
            raise MockServiceError(msg, status_code=None, code=None, transport='non_json') from e

    @staticmethod
    def _read_http_error_body(err: urllib.error.HTTPError) -> str:
        """Best-effort read of an HTTPError's response body (the proxy puts a
        helpful JSON message there). Never raises — returns '' on any failure."""
        try:
            return err.read().decode('utf-8', 'replace')
        except Exception:
            return ''

    def health(self) -> bool:
        try:
            return self._get('/health').get('status') == 'ok'
        except Exception:
            return False

    def _check_version_compat(self) -> None:
        """Verify the running mockenv MAJOR version matches the SDK's expected major.

        Soft major check: the server's major version must equal
        ``_COMPATIBLE_MOCKENV_MAJOR``; any patch/minor within that major line is
        accepted. Stays fail-closed — raises ``RuntimeError`` with a clear,
        actionable message when:
        - The server is unreachable.
        - The ``/health`` body is not JSON (the server is reachable but is not mockenv).
        - The ``/health`` body is not a JSON object (a list, string or number is
          not a mockenv answer, and holds no version to read).
        - The server does not advertise a version (older mockenv build).
        - The advertised version is unparseable or its major does not match.
        """
        try:
            health: object = self._get('/health')
        except MockServiceError as e:
            if e.transport == 'non_json':
                raise RuntimeError(
                    f'{self.base_url} answered /health with a body that is not JSON, so it is not a plurai-mockenv '
                    f'service (SDK expects major v{_COMPATIBLE_MOCKENV_MAJOR}.x). Check that the URL points at a plurai-mockenv service.'
                ) from e
            raise RuntimeError(f'plurai-mockenv not reachable at {self.base_url}: {e}. Is the mock service running?') from e
        if not isinstance(health, dict):
            raise RuntimeError(
                f'plurai-mockenv at {self.base_url} answered /health with a JSON {type(health).__name__} '
                f'instead of an object, so it reports no version (SDK expects major v{_COMPATIBLE_MOCKENV_MAJOR}.x). '
                'Check that the URL points at a plurai-mockenv service.'
            )
        server_version: str | None = health.get('version')
        if server_version is None:
            raise RuntimeError(
                f'plurai-mockenv at {self.base_url} did not report a version '
                f'(SDK expects major v{_COMPATIBLE_MOCKENV_MAJOR}.x). '
                'Upgrade plurai-mockenv: '
                f'pip install "plurai-mockenv>={_COMPATIBLE_MOCKENV_MAJOR}.{_SESSION_TURN_MIN_MINOR}.0,<{_COMPATIBLE_MOCKENV_MAJOR + 1}"'
            )
        try:
            server_major = int(server_version.split('.')[0])
        except (ValueError, AttributeError, IndexError):
            # Fail-closed on an unparseable version string — never silently
            # accept a server whose version we can't reason about.
            raise RuntimeError(
                f'plurai-mockenv at {self.base_url} reported an unparseable version {server_version!r} (SDK expects major v{_COMPATIBLE_MOCKENV_MAJOR}.x).'
            ) from None
        if server_major != _COMPATIBLE_MOCKENV_MAJOR:
            raise RuntimeError(
                f'plurai-mockenv major version mismatch: server reports v{server_version} '
                f'(major {server_major}), SDK expects major v{_COMPATIBLE_MOCKENV_MAJOR}.x. '
                'Upgrade plurai-mockenv: '
                f'pip install "plurai-mockenv>={_COMPATIBLE_MOCKENV_MAJOR}.{_SESSION_TURN_MIN_MINOR}.0,<{_COMPATIBLE_MOCKENV_MAJOR + 1}"'
            )
        self._server_version = server_version

    @property
    def supports_session_turn(self) -> bool:
        """Whether this server serves ``POST /session/turn``.

        False before any version check: without evidence the route exists, the
        optional push stays home rather than spending a round-trip on a 404.
        """
        if self._server_version is None:
            return False
        major, minor = _parse_major_minor(self._server_version)
        # The major is re-checked because a hand-set ``_server_version`` (tests, stand-ins)
        # reaches this property without having passed the gate in ``_check_version_compat``.
        return major == _COMPATIBLE_MOCKENV_MAJOR and minor >= _SESSION_TURN_MIN_MINOR

    @property
    def supports_session_dialog(self) -> bool:
        """Whether ``POST /session/turn`` on this server also stores ``dialog`` and ``live_tool_calls``.

        False before any version check, for the reason :attr:`supports_session_turn` is.
        """
        if self._server_version is None:
            return False
        major, minor = _parse_major_minor(self._server_version)
        return major == _COMPATIBLE_MOCKENV_MAJOR and minor >= _SESSION_DIALOG_MIN_MINOR

    def warn_once_no_session_dialog(self) -> None:
        """Name what an older server costs, and the way out of it, once per client."""
        if self._warned_no_session_dialog:
            return
        self._warned_no_session_dialog = True
        print(
            f'  [plurai] WARNING plurai-mockenv at {self.base_url} reports v{self._server_version or "unknown"}, below '
            f'v{_COMPATIBLE_MOCKENV_MAJOR}.{_SESSION_DIALOG_MIN_MINOR}: it cannot be given the conversation so far, so a '
            'mocked tool that generates its answer at run time reads the request without it. Upgrade the mock service to '
            f'plurai-mockenv v{_COMPATIBLE_MOCKENV_MAJOR}.{_SESSION_DIALOG_MIN_MINOR}.0 or later where it runs.',
            file=sys.stderr,
        )

    def warn_once_no_session_turn(self) -> None:
        """Name what an older server costs, and the way out of it, once per client."""
        if self._warned_no_session_turn:
            return
        self._warned_no_session_turn = True
        print(
            f'  [plurai] WARNING plurai-mockenv at {self.base_url} reports v{self._server_version or "unknown"}, below '
            f'v{_COMPATIBLE_MOCKENV_MAJOR}.{_SESSION_TURN_MIN_MINOR}: it cannot be told which turn is open, so events '
            'from callers outside this process are stamped turn 0. Tools dispatched in-process carry the right turn '
            'either way; the turn-bound checks on a hosted agent do not. Upgrade the mock service to plurai-mockenv '
            f'v{_COMPATIBLE_MOCKENV_MAJOR}.{_SESSION_TURN_MIN_MINOR}.0 or later where it runs (a hosted env is upgraded '
            'by whoever deploys it; a local stub follows the SDK pin).',
            file=sys.stderr,
        )

    # --- Session ---
    def create_session(self, sid: str, initial_state: dict, agent_context: dict) -> dict:
        if not self._version_checked:
            self._check_version_compat()
            self._version_checked = True
        return self._post('/session/create', {'session_id': sid, 'initial_state': initial_state, 'agent_context': agent_context})

    def set_turn(self, sid: str, turn: int, *, dialog: dict | None = None, live_tool_calls: list[dict] | None = None) -> dict | None:
        """Tell the service which turn is open.

        A caller that cannot know the harness's turn — a hosted agent's callouts, which
        reach the deployed mock directly — produces events the service stamps from this
        value. In-process dispatchers send their own turn on every execute and do not
        depend on it.

        ``dialog`` (a ``plurai_mockenv.SessionDialog``, JSON-shaped) and ``live_tool_calls``
        (``plurai_mockenv.PriorToolCall`` dicts) ride the same push when given. Each REPLACES
        what the session held, so the caller sends the whole history every turn. This client
        sends what it is given. A server below
        :data:`_SESSION_DIALOG_MIN_MINOR` gets the turn alone, and one warning.

        Three outcomes: the service's reply when it takes the turn; the service's error
        envelope (``{'error': ..., 'status': 'error'}``, which the service returns at
        HTTP 200, so nothing raises for it) when it refuses; ``None``, after one warning,
        when the server predates the route.
        """
        if not self.supports_session_turn:
            self.warn_once_no_session_turn()
            return None
        body: dict[str, Any] = {'session_id': sid, 'turn': turn}
        if dialog is not None or live_tool_calls is not None:
            if not self.supports_session_dialog:
                self.warn_once_no_session_dialog()
            else:
                if dialog is not None:
                    body['dialog'] = dialog
                if live_tool_calls is not None:
                    body['live_tool_calls'] = live_tool_calls
        return self._post('/session/turn', body)

    # --- Mock execution ---
    def execute_mock(self, sid: str, tool: str, args: dict, turn: int = 0) -> dict:
        return self._post('/mock/execute', {'session_id': sid, 'tool': tool, 'args': args, 'turn': turn}, timeout=self._execute_timeout)

    def observe_result(self, sid: str, tool: str, args: dict, result: Any) -> dict:
        return self._post('/mock/observe', {'session_id': sid, 'tool': tool, 'args': args, 'result': result})

    # --- State (used by replacement nodes, observers, any code) ---
    def get_full_state(self, sid: str) -> dict:
        return self._get(f'/state/{sid}')

    def get_entity(self, sid: str, entity_id: str) -> dict | None:
        """The entity at ``entity_id``, ``None`` when there is none.

        Raises :class:`MockServiceError` on the service's error envelope (HTTP 200,
        ``status: 'error'``) — e.g. an untyped read of an id two entity types share — which
        would otherwise read as "not found".
        """
        r = self._get(f'/state/{sid}/{entity_id}')
        if r.get('status') == 'error':
            raise MockServiceError(f'GET /state/{sid}/{entity_id}: {r.get("error")}', status_code=None, code=None, transport=None)
        return r.get('entity')

    def update_state(self, sid: str, updates: dict, source: str | None = None) -> dict:
        """Apply ``updates`` outside a mock handler; the mock records it as a ``state_update``.

        ``source`` says who wrote. Omitted, it is read off the scope the child opened for this
        context (:func:`~plurai._mocking.turn_context.world_write_source`): ``'agent'`` inside the
        agent's own invocation, ``'harness'`` inside the harness's ``reset`` / ``seed_state`` hook,
        ``'external'`` everywhere else. An explicit value always wins.
        """
        if source is None:
            source = world_write_source()
        result = self._post(f'/state/{sid}/update', {'updates': updates, 'source': source})
        if isinstance(result, dict) and result.get('status') == 'write_refused':
            # The mock refused a write whose key names no entity: the harness's fault, recorded here
            # because the frames between this call and the child may be the agent's own ``except``.
            note_harness_fault('plurai_mockenv.MisaddressedWrite', str(result.get('error', 'world write refused')), kind='harness_fault')
        return result

    def query_state(self, sid: str, entity_type: str, **filters) -> list[dict]:
        params = urllib.parse.urlencode({'entity_type': entity_type, **filters})
        r = self._get(f'/state/{sid}/query?{params}')
        return r.get('results', [])

    def get_context(self, sid: str) -> dict:
        return self._get(f'/state/{sid}/context')

    def get_events(self, sid: str) -> dict:
        return self._get(f'/events/{sid}')
