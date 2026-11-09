"""Mock session lifecycle — configure, start_session, and state-read helpers.

``_local_service`` is the in-process mock that ``configure(mode='local')`` starts; the atexit hook and the
child's teardown both read it from here.
"""

from __future__ import annotations

import atexit
import sys

from plurai_mockenv import MockService, _registry
from plurai_mockenv._handler_state import _hs, _inside_handler
from plurai_mockenv._registry import _active_session_id, _set_active_session_id

from plurai._protocol import _STUB_MODES, MockMode
from plurai._mocking import client as _mock_client
from plurai._mocking.client import MOCK_503_AWAITING_EDIT_FINALIZE, MOCK_503_RESTARTING, MockServiceError
from plurai._mocking.local import _serve_local_mock

#: The in-process local mock that ``configure(mode='local')`` and ``mode='mockless'`` start.
_local_service: MockService | None = None


def active_session_id() -> str | None:
    """The id of the session active in this context, or ``None``.

    ``simulate.py`` in the ``code`` action channel reads it to hand the nested skill session
    ``PLURAI_SESSION_ID``, so an intercepted vendor SDK call routes to the same session over the mock wire.
    """
    return _active_session_id() or None


def mock_service_url() -> str | None:
    """The mock service base url this process is configured against, or ``None`` before :func:`configure`.

    The ``code`` action channel passes it to the processes the agent spawns as ``PLURAI_MOCK_URL``: they dial
    the same mock this process does (on a hosted mock, the backend's ``/mock/<id>`` url), authenticated with the
    bearer they read from ``PLURAI_TOKEN``. A caller fails closed on ``None`` instead of reaching the real
    vendor API.
    """
    client = _registry._client
    if client is None:
        return None
    return client.base_url


def configure(service_url: str | None = None, *, mode: MockMode = 'connect', mock_service_url: str | None = None) -> None:
    """Connect this process to a mock service — or, in local mode, host one.

    ``mock_service_url=`` is a true ALIAS of ``service_url``: it is the name the
    rest of the SDK uses for the same value (``plurai.mock_service_url()``, the
    ``plurai_sim/.mock_service_url`` marker, ``Init.mock_service_url``), so a caller
    reaching for it is right about the thing and wrong only about this signature.
    Pass exactly one of the two.

    ``mode='connect'`` (default, the production posture): point a client at the
    deployed ``service_url`` and verify ``/health``. If the mock is unreachable
    or answers unhealthy this **raises** — a transient outage must fail loud, not
    silently fall back to an in-process mock that would green the run on stub
    data. A transport-level failure (connection refused / DNS / timeout) is a hard
    error here too: local mode is never INFERRED from a failed connect, only
    requested explicitly.

    ``mode='local'`` (explicit opt-in for local dev / tests): spin up an
    in-process :class:`MockService` via ``build_service(host='127.0.0.1', port=<port>)`` —
    which collects all staged ``@plurai.mock`` handlers / observers / descriptors /
    typed-RPC schemas / decorator opt-outs from the staging registry and threads them
    into the engine explicitly.

    **In ``mode='local'`` the url is OPTIONAL, and a given one is a loopback TOKEN — not an
    address.** The in-process mock binds its OWN free loopback port so parallel scenarios
    cannot collide, so the host and port of any url you pass are NOT honored; read the bound
    url with :func:`plurai.mock_service_url`. What a given url still buys is the loopback
    CHECK: a non-loopback one is refused (see
    :func:`plurai._mocking.local._serve_local_mock`) so a stray ``mode='local'`` in a
    production deploy cannot shadow a real deployed mock.

    ``mode='mockless'``: nothing is mocked — a tool-less agent, or one whose every tool runs
    ``mode: real``. It takes the SAME path as ``'local'`` (an idle loopback stub, so
    :func:`plurai.mock_service_url` still answers for harness ``seed_state`` templates) and
    differs only in what is CLAIMED: no surface is held constant, and every tool call reached
    the real service. That distinction is not cosmetic — it is what the run report's posture
    line, and every reader deciding whether a result is reproducible, is reading. Declaring a
    mockless run as ``'local'`` would assert a held-constant world that does not exist.
    """
    if mock_service_url is not None:
        if service_url is not None:
            raise TypeError(
                f'plurai.configure() got both service_url={service_url!r} and mock_service_url={mock_service_url!r} — they are the same argument; pass one.'
            )
        service_url = mock_service_url
    # CLOSED, not `== 'connect'`. `MockMode` is a Literal — unenforced at runtime — and
    # `configure()` is public API a customer hand-writes in simulate.py, so a typo ('Local',
    # 'mockless ') reaches here. Keyed on connect alone, an unknown mode would fall past
    # both branches into the connect path with base_url=None and die as
    # `AttributeError: 'NoneType' object has no attribute 'rstrip'` deep inside the client,
    # instead of the TypeError below that names every valid mode.
    if service_url is None and mode not in _STUB_MODES:
        raise TypeError(
            "plurai.configure() needs the mock service url in mode='connect': plurai.configure('http://…') "
            "(or service_url= / its mock_service_url= alias). In mode='local' / mode='mockless' the url is "
            'optional: the in-process stub binds its OWN free loopback port — host/port in a given url are '
            'NOT honored; read the bound url with plurai.mock_service_url().'
        )

    if mode in _STUB_MODES:
        global _local_service
        svc = _serve_local_mock(service_url, posture=mode)
        _local_service = svc
        atexit.register(lambda: _local_service.stop() if _local_service else None)
        return

    _registry._client = _mock_client.MockServiceClient(base_url=service_url)
    try:
        health = _registry._client._get('/health')
    except MockServiceError as exc:
        # _get raised: the server either returned a non-2xx or was unreachable.
        # Pull the status_code from the exception to build the most specific
        # remediation message, then raise with the original error preserved.
        status_code = exc.status_code
        hint = _configure_hint(service_url, status_code, str(exc), body='', code=exc.code)
        raise RuntimeError(f'plurai.configure({service_url!r}) could not reach a healthy mock: {hint}. Error: {exc}') from exc

    if health.get('status') == 'ok':
        print(f'[plurai] Connected to {service_url}', file=sys.stderr)
        return

    # Server answered 2xx but health says not-ok (unexpected body shape).
    # Treat the same as an HTTP failure — build the most specific hint we can.
    status_code = health.get('status_code')
    code = health.get('code') if isinstance(health.get('code'), str) else None
    hint = _configure_hint(service_url, status_code, health.get('error') or '', body=str(health), code=code)
    raise RuntimeError(f'plurai.configure({service_url!r}) could not reach a healthy mock: {hint}. Server said: {health.get("error")!r}.')


def _configure_hint(service_url: str, status_code: int | None, error_text: str, *, body: str, code: str | None = None) -> str:
    """Return the most specific remediation hint for a configure failure.

    ``code`` is the backend's machine-readable discriminator (``MockServiceError.code``)
    and takes precedence over the prose wherever it is present.
    """
    if status_code in (401, 403):
        return 'the remote backend rejected the request — log in to the backend that hosts this mock (`plurai login`, or PLURAI_URL and PLURAI_TOKEN)'
    if status_code == 410:
        # The backend distinguishes "env definition exists but isn't deployed"
        # (code=needs_redeploy) from "env unknown". The former just needs a
        # deploy_env / re-activation; only the latter needs a full regenerate.
        if code == 'needs_redeploy' or 'needs_redeploy' in error_text or 'needs_redeploy' in body:
            return "the mock env exists on the backend but isn't deployed — re-activate it with /use-env (which calls deploy_env)"
        return 'the mock env is unknown on the backend — regenerate it with /generate-mock-env and use the new mock_service_url'
    if status_code == 503:
        # The two states that produce a 503 here: a restart in flight, and an edit
        # session holding the env.
        if code == MOCK_503_AWAITING_EDIT_FINALIZE:
            return (
                'the mock env is held while an interrupted edit session still owes a finalize — '
                'finish (or abandon) that edit session; the hold also clears when the session expires. '
                'Waiting will not help: it is blocked on a human, not on a restart'
            )
        if code == MOCK_503_RESTARTING:
            return 'the mock env crashed and the backend is restarting it — this clears on its own in a few seconds; retry'
        return f'the backend is temporarily unavailable (HTTP 503{f", code={code}" if code else ""})'
    if status_code is not None:
        return f'the backend returned HTTP {status_code}'
    # No HTTP status -> transport-level failure. In connect mode this is a
    # hard error: for an in-process mock, pass mode='local' EXPLICITLY (it is
    # never inferred from a failed connect — inferring it would conflate a
    # momentary remote outage with "spin up a local mock and run green on stubs").
    return f"nothing is serving {service_url} (connection refused / DNS / timeout). If you meant an in-process local mock, call configure(..., mode='local')"


def start_session(session_id: str, initial_world_state: dict, agent_context: dict | None = None) -> dict:
    """Begin a simulation session against the configured mock service.

    Sets the active session id (so MockToolNode invocations route to it via
    ``_active_session_id()``) and creates the session with the given world state
    + agent context. Returns the server's response.

    Concurrent scenarios are isolated via a ``ContextVar`` — each ``asyncio.Task``
    started by ``asyncio.gather`` gets its own context copy, so two parallel
    ``Orchestrator.run()`` calls don't clobber each other's session id. The
    module-level fallback (``_module_session_id``) keeps non-async / sequential
    callers reading the latest value.
    """
    if _registry._client is None:
        raise RuntimeError('plurai.configure() must be called before start_session().')
    # Create the session server-side BEFORE writing the active id locally.
    # If create_session raises (network / 5xx / validation), we don't want
    # the ContextVar + module-global to point at a session_id that has no
    # server-side state — subsequent MockToolNode lookups would route to a
    # dead session and produce confusing 404s deep in tool execution.
    response = _registry._client.create_session(session_id, initial_world_state, agent_context or {})
    _set_active_session_id(session_id)
    return response


def get_full_state(session_id: str | None = None) -> dict:
    """Return the current world state for the given session (defaults to the active one)."""
    if _registry._client is None:
        raise RuntimeError('plurai.configure() must be called first.')
    sid = session_id or _active_session_id()
    if not sid:
        raise RuntimeError('No active session — call plurai.start_session(...) first or pass session_id explicitly.')
    return _registry._client.get_full_state(sid)


def get_events(session_id: str | None = None) -> dict:
    """Return the recorded tool events for the given session (defaults to the active one)."""
    if _registry._client is None:
        raise RuntimeError('plurai.configure() must be called first.')
    sid = session_id or _active_session_id()
    if not sid:
        raise RuntimeError('No active session — call plurai.start_session(...) first or pass session_id explicitly.')
    return _registry._client.get_events(sid)


def world_state_snapshot(session_id: str | None = None) -> dict[str, dict]:
    """Return the full ``{entity_id → entity_dict}`` mapping for a session.

    The explicit bulk-read API for mock handlers + observer hooks that need
    a plain ``dict`` of entities. Prefer it over iterating
    ``ctx.world_state.entities`` directly when you want a single explicit
    HTTP round-trip (the proxy is a Mapping but refetches per call — fine
    for occasional reads, expensive in tight loops).

    Inside a ``@plurai.mock`` handler this is zero-cost (returns a copy of
    the thread-local ``_hs().state`` snapshot the mock service already
    materialised before invoking the handler). Outside a handler it makes
    one HTTP call to ``/state/<sid>`` and returns the ``state`` payload.

    Three canonical read patterns the simulation-setup skill documents:
      1. ``ctx.world_state.get_entity(eid)`` — single entity by id
      2. ``ctx.world_state.query(entity_type, **filters)`` — list by type
      3. ``plurai.world_state_snapshot()`` — full ``dict`` (this function)

    Raises ``RuntimeError`` when no session is active and none is provided.
    With ``_client`` unconfigured returns ``{}`` — same lenient pattern the
    proxy itself follows so test/scratch code can construct an empty
    snapshot without crashing.
    """
    if _inside_handler():
        # Copy so the caller can mutate without affecting the live thread-local.
        return dict(_hs().state)
    if _registry._client is None:
        return {}
    sid = session_id or _active_session_id()
    if not sid:
        raise RuntimeError('No active session — call plurai.start_session(...) first or pass session_id explicitly.')
    return _registry._client.get_full_state(sid).get('state', {}) or {}
