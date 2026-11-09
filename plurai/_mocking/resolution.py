"""Mock-response resolver — routes tool calls through the configured mock env.

Owns the connection-error tuple, the schema-stub fallback helper, and the main
``_resolve_mock_response`` dispatcher used by both ``MockToolNode`` and
``mock_wrap``.

``_warned_no_client`` latches the one warning about a tool call made before ``plurai.configure()``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple
from urllib.parse import urlsplit

from plurai_mockenv import _registry
from plurai_mockenv._registry import _active_session_id

from plurai._http_retry import RATE_LIMITED_STATUS
from plurai._mocking.codecs import PayloadEncodeError, _plain, apply_payload_codecs
from plurai._mocking.client import (
    MOCK_503_AWAITING_EDIT_FINALIZE,
    MockServiceError,
    TransportFailure,
    transport_failure,
)
from plurai._mocking.result import ENGINE_REFUSAL_KEY, declared_error_channel
from plurai._mocking.turn_context import MockRouteCause, current_turn_or_zero, note_harness_fault, refusal_cause
from plurai._stubs import is_no_handler_sentinel, schema_stub_from_annotation
from plurai.capacity import (
    BACKEND_REFUSED_PHRASE,
    CAPACITY_CODE,
    CapacityBlock,
    UnavailableBlock,
    capacity_block_from_body,
    unavailable_block_from_body,
    unavailable_block_from_record,
    unavailable_block_of,
)
from plurai.credit import CREDIT_KINDS, CreditBlock, credit_block_from_body, credit_block_or_unknown

logger = logging.getLogger(__name__)

#: One-shot latch for the missing-``configure()`` warning.
_warned_no_client: bool = False

#: The sibling a runtime-generated mock answer carries when its generation faulted (the backend
#: spells the same tag; it cannot import this SDK, so the literal is duplicated).
RUNTIME_FAULT_TAG = '_plurai_runtime_fault'
#: The grader-visible sibling of a runtime answer; on a fault it carries the runtime's own cause
#: (``grounding``) and ``fault_class`` (``transient`` | ``deterministic``). Spelled twice, like the tag.
RUNTIME_SIBLING = '_plurai_runtime'
#: The ``cause`` a runtime fault's sibling carries when the backend refused the mock's model call for credit; the
#: credit block sits beside it as ``credit``. Spelled twice, like the tag.
RUNTIME_CREDIT_CAUSE = 'credit'
#: The ``cause`` a runtime fault's sibling carries when the backend refused the mock's model call because its provider
#: account was refused; the capacity block sits beside it as ``capacity``. Spelled twice, like the tag.
RUNTIME_CAPACITY_CAUSE = 'capacity'
#: The ``cause`` a runtime fault's sibling carries when the backend could not take the mock's model call because it is
#: unavailable (a provider outage past its own retries); the unavailable block sits beside it as ``unavailable``.
RUNTIME_UNAVAILABLE_CAUSE = 'unavailable'

#: What the agent was handed in place of a tool answer: the explicit error stub (a ``dict``-typed
#: tool, :func:`_stub_or_legacy_sentinel`), a schema-shaped stub of the tool's declared return type —
#: which reads to the agent as a SUCCESSFUL empty answer — or the untyped generic sentinel. ``unknown``: a
#: row the parent SYNTHESIZED from an older child's bare stub count (``_protocol._RETIRED_FIELDS``), which
#: named neither the call nor what the agent was served.
ServedStub = Literal['error_stub', 'schema_stub', 'sentinel', 'unknown']


class UnansweredCall(NamedTuple):
    """One tool call the mock did not answer, so the agent was handed a stub.

    Recorded only where ``/mock/execute`` itself failed (a non-2xx status, or a transport failure):
    the mock's own in-band envelopes (an unknown session, a handler that raised) are the mock
    ANSWERING, and they are already visible in the event trail. ``restart_shaped`` is the one field
    retry policy reads (:func:`_is_restart_shaped`); everything else is evidence.
    """

    tool: str
    turn: int
    status_code: int | None
    code: str | None
    transport: TransportFailure | None
    served: ServedStub
    restart_shaped: bool
    #: The credit block's reset time, on a credit refusal of the free tier; the capacity block's, on a capacity refusal;
    #: ``None`` otherwise.
    resets_at: str | None = None
    #: On a capacity refusal only: whether the provider named the reset (``named_reset``) or the backend's window did (``ttl``).
    reset_source: str | None = None
    #: On an unavailable answer or a spent retry budget: the unavailable block ``{cause, code?}``; ``None`` otherwise.
    unavailable: dict[str, str] | None = None


# Per-session list of the tool calls served a stub because /mock/execute failed, keyed by
# session id. Filled by ``_resolve_mock_response``; taken (and cleared) by the child at finalize.
_unanswered_calls: dict[str, list[UnansweredCall]] = {}


def take_unanswered_calls(session_id: str) -> list[UnansweredCall]:
    """Every call recorded for ``session_id``, in order, and forget them."""
    return _unanswered_calls.pop(session_id, [])


def _record_unanswered(session_id: str | None, call: UnansweredCall) -> None:
    if session_id:
        _unanswered_calls.setdefault(session_id, []).append(call)


def _served_stub(stub: Any) -> ServedStub:
    """What :func:`_stub_or_legacy_sentinel` actually handed back, read off its return value."""
    if isinstance(stub, dict) and '_plurai_stub_error' in stub:
        return 'error_stub'
    if isinstance(stub, dict) and stub.get('mock') is True and set(stub) == {'status', 'mock', 'tool'}:
        return 'sentinel'
    return 'schema_stub'


def _is_restart_shaped(*, status_code: int | None, code: str | None, transport: TransportFailure | None) -> bool:
    """Whether a failed mock call is the signature of a mock that went away and is coming back.

    True for a 503 the client's bounded retry could not outlast, and for a connection nothing
    answered (``refused``). Those are what a mock restart looks like, and a restart loses the seeded
    in-memory session, so a re-drive is the only way back to a full measurement.

    FALSE for everything else, because the re-drive budget must not be spent on a failure a re-drive
    repeats: 401/403 is a token problem, 410 needs a ``deploy_env``, any other status is a hop in
    front of the mock answering for it, a ``timeout`` is a slow mock that is still there, a
    ``non_json`` 2xx is something that is not a plurai mock, and a 503 carrying
    ``awaiting_edit_finalize`` is blocked on a human, not on a restart.
    """
    if code in (MOCK_503_AWAITING_EDIT_FINALIZE, CAPACITY_CODE) or code in CREDIT_KINDS:
        return False
    return status_code == 503 or transport == 'refused'


#: The statuses a credit refusal arrives with: the 402 every kind carries, and the 503 the credit store's outage can
#: still arrive as; a credit kind in the body is what makes either one a refusal. Any other status with such a code is
#: not the backend's credit answer.
_CREDIT_STATUSES: frozenset[int] = frozenset({402, 503})
#: The statuses a hop IN FRONT of the mock answers with when it could not get an answer from it —
#: the hosted proxy's 502 (mock dead or over its bound) and 503 (restarting), a gateway's 504 — and
#: when it would not take the call right now: a 429 throttle. The mock itself answers every call with
#: HTTP 200 and errors in-band, so none of these is the mock's. A 429 reaches here only after the
#: client waited out its retry budget, so one more try later answers it; typed ``transport``, never
#: ``not_a_mock``, which would read as a permanent fault and send the operator to the mock URL.
_GATEWAY_STATUSES: frozenset[int] = frozenset({502, 503, 504, RATE_LIMITED_STATUS})
#: A failure with no status in which nothing answered: the connection was refused or reset, the call
#: timed out, or the transport failed otherwise. ``non_json`` is absent: something DID answer.
_TRANSPORT_DROPS: frozenset[TransportFailure] = frozenset({'refused', 'timeout', 'other'})
#: The statuses that name their own cause; any other is :data:`MockRouteCause` ``not_a_mock``.
_STATUS_CAUSES: dict[int, MockRouteCause] = {401: 'auth', 403: 'auth', 410: 'not_deployed'}

#: Whether one more try can answer, per cause: only a dropped transport. Everything else is the same
#: answer on every try — a token, an undeployed env, a human's hold, a URL that is not a mock.
MOCK_ROUTE_CAUSE_TRANSIENT: dict[MockRouteCause, bool] = {
    'transport': True,
    'held_for_edit': False,
    'auth': False,
    'not_deployed': False,
    'not_a_mock': False,
    'credit': False,
    'capacity': False,
    'unavailable': False,
}

_MOCK_ROUTE_REMEDY: dict[MockRouteCause, str] = {
    'transport': 'the route to the mock dropped this call (a gateway error, or a refused, reset or timed-out connection) — '
    'a transient transport fault, not the mock env: re-run the scenario',
    'held_for_edit': 'mock env held until an interrupted edit session is finalized — finish or abandon that session (waiting will not clear it)',
    'auth': 'the mock rejected the token: check PLURAI_TOKEN, or log in again with `plurai login` (the token must belong to the backend that hosts this mock)',
    'not_deployed': 'the mock env is not deployed (or was redeployed): re-activate it with deploy_env',
    'not_a_mock': 'something that is not a plurai mock answered at the mock URL: check the mock URL and anything in front of it',
    'credit': 'the backend refused the mock call for credit: the run stops, and the stop message says what to do',
    'capacity': f'{BACKEND_REFUSED_PHRASE}: the run stops, and the stop message asks for a later try',
    'unavailable': f'{BACKEND_REFUSED_PHRASE}: the run stops, and the stop message says to try again in a few minutes',
}


def mock_route_cause(*, status_code: int | None, code: str | None, transport: TransportFailure | None) -> MockRouteCause:
    """The :data:`MockRouteCause` of one unanswered ``/mock/execute`` call, from its typed status, code and transport."""
    if code == MOCK_503_AWAITING_EDIT_FINALIZE:
        return 'held_for_edit'
    if status_code in _CREDIT_STATUSES and code in CREDIT_KINDS:
        return 'credit'
    if code == CAPACITY_CODE:
        # The backend's typed capacity refusal, whatever status carries it (a 429 or a 503 in front of the mock is
        # otherwise a dropped transport): every further call is refused the same way until it resets.
        return 'capacity'
    if status_code is None:
        return 'transport' if transport in _TRANSPORT_DROPS else 'not_a_mock'
    if status_code in _GATEWAY_STATUSES:
        return 'transport'
    return _STATUS_CAUSES.get(status_code, 'not_a_mock')


#: The causes that stop a whole run. Each is the same answer on every try and for every scenario, so continuing would
#: score the agent against tools that never worked. ``transport`` is excluded because one more try answers it, and
#: ``held_for_edit`` because a human clears the hold and the run waits for it.
ABORTING_ROUTE_CAUSES: frozenset[MockRouteCause] = frozenset({'auth', 'not_deployed', 'not_a_mock'})

_MOCK_PATH_MARKER = '/mock/'


def aborting_route_cause(call: Mapping[str, Any]) -> MockRouteCause | None:
    """The :data:`ABORTING_ROUTE_CAUSES` member that one ``unanswered_mock_calls`` row names, else ``None``.

    A row an older child synthesized from a bare count (``served == 'unknown'``) carries no status, code or
    transport, so its cause is unknowable and it never stops a run. A row that carries the backend's typed ``unavailable``
    block is the backend's answer, whatever status carried it (:func:`unavailable_block_of_call`), never a mock that cannot
    answer.
    """
    if call.get('served') == 'unknown' or unavailable_block_of_call(call) is not None:
        return None
    cause = mock_route_cause(status_code=call.get('status_code'), code=call.get('code'), transport=call.get('transport'))
    return cause if cause in ABORTING_ROUTE_CAUSES else None


def credit_block_of_call(call: Mapping[str, Any]) -> CreditBlock | None:
    """The credit block one ``unanswered_mock_calls`` row carries, else ``None``.

    The credit abort reads it apart from :func:`aborting_route_cause`: a refusal for credit stops a run, but it keeps
    what the run measured, which a mock that cannot answer does not.
    """
    if mock_route_cause(status_code=call.get('status_code'), code=call.get('code'), transport=call.get('transport')) != 'credit':
        return None
    return credit_block_from_body({'code': call.get('code'), 'resets_at': call.get('resets_at')})


def capacity_block_of_call(call: Mapping[str, Any]) -> CapacityBlock | None:
    """The capacity block one ``unanswered_mock_calls`` row carries, else ``None``: like the credit block, it stops a run
    that keeps what it measured (:func:`credit_block_of_call`)."""
    if call.get('served') == 'unknown' or call.get('code') != CAPACITY_CODE:
        return None
    return capacity_block_from_body({'code': call.get('code'), 'resets_at': call.get('resets_at'), 'reset_source': call.get('reset_source')})


def unavailable_block_of_call(call: Mapping[str, Any]) -> UnavailableBlock | None:
    """The unavailable block one ``unanswered_mock_calls`` row carries, else ``None``, read like the capacity block
    (:func:`capacity_block_of_call`). Unlike a capacity refusal it does not stop the run: the scenario is re-driven once, as for
    a transient mock fault, and a row the re-drive does not cure is one that was not completed."""
    if call.get('served') == 'unknown':
        return None
    return unavailable_block_from_record(call.get('unavailable'))


def mock_location(mock_url: str | None) -> tuple[str | None, str | None]:
    """``(url, env_id)`` of a mock URL, safe to print: scheme, host and path only, never userinfo, query or fragment.

    ``env_id`` is the path segment after ``/mock/`` when the URL has one.
    """
    if not mock_url:
        return None, None
    parts = urlsplit(mock_url)
    if not parts.scheme or not parts.hostname:
        return None, None
    host = parts.hostname if parts.port is None else f'{parts.hostname}:{parts.port}'
    url = f'{parts.scheme}://{host}{parts.path}'
    env_id = None
    if _MOCK_PATH_MARKER in parts.path:
        env_id = parts.path.split(_MOCK_PATH_MARKER, 1)[1].strip('/').split('/', 1)[0] or None
    return url, env_id


def abort_remedy(cause: MockRouteCause, mock_url: str | None) -> str:
    """What the operator changes so the mock answers, for a cause in :data:`ABORTING_ROUTE_CAUSES`, naming the env or URL involved."""
    url, env_id = mock_location(mock_url)
    remedy = _MOCK_ROUTE_REMEDY[cause]
    if cause == 'not_deployed' and env_id:
        return f'{remedy} (mock env {env_id})'
    if cause in ('not_deployed', 'not_a_mock') and url:
        return f'{remedy} (mock URL {url})'
    return remedy


_THROTTLED_REMEDY = (
    'the Plurai backend is rate limiting this run (HTTP 429) and the retry budget ran out — a transient fault of the route, '
    'not the mock env: wait a minute and re-run the scenario, or run fewer scenarios at once'
)


def unanswered_call_remedy(call: UnansweredCall) -> str:
    """What would get this call answered, keyed on its :data:`MockRouteCause`."""
    if call.status_code == RATE_LIMITED_STATUS:
        return _THROTTLED_REMEDY
    if call.served == 'unknown':
        return (
            'an older plurai in the agent interpreter (`PLURAI_CHILD_PYTHON`?) reported only a COUNT of stubbed tool calls: '
            'upgrade plurai there so each unanswered call is named with its status and remedy'
        )
    return _MOCK_ROUTE_REMEDY[mock_route_cause(status_code=call.status_code, code=call.code, transport=call.transport)]


def _build_connection_error_types() -> tuple[type[BaseException], ...]:
    """Connection-class errors worth swallowing into the schema-stub fallback.

    Scope is **deliberately narrow** — only true network/transport errors.
    ``OSError`` is intentionally NOT included even though many connection
    failures are subclasses of it: ``OSError`` is also the parent of
    ``FileNotFoundError`` / ``PermissionError`` / ``IsADirectoryError``,
    which the httpx stack can raise during cert / .netrc / proxy-config
    loading. Catching those as "mock-env unreachable" hides real
    misconfigurations behind a phantom-connectivity warning.

    Computed once at module load (``_CONNECTION_ERROR_TYPES``); the
    per-call resolver reads the cached tuple.
    """
    import socket  # stdlib, always available

    types: list[type[BaseException]] = [
        ConnectionError,  # ConnectionRefused/Reset/Aborted
        TimeoutError,  # builtin (3.10+); a sibling of socket.timeout
        socket.gaierror,  # DNS resolution
        socket.herror,  # host lookup
        MockServiceError,  # raised by _get/_post on HTTP or transport failure
    ]
    try:
        import httpx

        types.append(httpx.ConnectError)
        types.append(httpx.ReadError)
    except ImportError:  # pragma: no cover — httpx is a hard plurai dep
        pass
    try:
        from urllib.error import URLError

        types.append(URLError)
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


# Cache at module load — the tuple's contents don't change per-call.
_CONNECTION_ERROR_TYPES: tuple[type[BaseException], ...] = _build_connection_error_types()


def _stub_or_legacy_sentinel(
    name: str,
    return_type: Any,
    *,
    error: str | None = None,
) -> Any:
    """Schema-shaped stub when ``return_type`` is usable, else the
    ``{"status": "ok", "mock": True, "tool": <name>}`` generic sentinel.

    Untyped tools (no return annotation, ``Any``, or ``None``) get the
    generic sentinel.

    When ``error`` is provided, returns an EXPLICIT error stub instead of
    a schema-shaped one. Caller-visible shape::

        {
          "status": "error",
          "_plurai_stub_error": "<the error message>",
          "_plurai_tool": "<tool name>",
          "message": "Mock for '<tool>' failed: <error>",
        }

    The two synthetic keys (``_plurai_stub_error``, ``_plurai_tool``) let
    the customer's tool-wrapper template distinguish a real-handler-error
    fallback from a legitimate empty-success response. Without them the
    schema stub for a ``dict``-typed tool is just ``{}``, which the agent
    sees as ``ToolMessage(content='{}', status='success')`` and tries to
    rephrase-and-retry — driving the LangGraph React loop into the
    recursion limit. Surfacing the error shape stops the loop because
    the wrapper can emit ``status='error'`` on the ToolMessage, which
    the agent treats as a stop signal.

    The in-process ``no_handler_sentinel`` path keeps the plain stub — that's the
    legitimate "I haven't written this mock yet" state during incremental
    handler buildup. A DEPLOYED mock's no-handler refusal (``ENGINE_REFUSAL_KEY``)
    is not silent: :func:`resolve` records it as a non-transient ``mock_runtime_fault``,
    so the scenario is not valid.
    """
    # Local import: the no-useful-type sentinel is internal to _stubs.py +
    # this resolver. Importing it package-level would surface it as
    # ``plurai._NO_USEFUL_TYPE`` for customers, which is private API.
    from plurai._stubs import _NO_USEFUL_TYPE  # noqa: PLC0415

    # Error stubs are scoped to explicit ``return_type=dict`` callers —
    # that's the ``_install_mock_coroutines`` wrapper path, which
    # always passes ``return_type=dict`` because it serializes the result
    # into a ToolMessage's string content. The ``mock_wrap`` path
    # uses the @tool's actual return annotation (often
    # ``list[dict[str, str]]`` etc.); silently substituting a dict-shaped
    # error stub there would break callers iterating ``for item in
    # result:`` over a typed response. Untyped tools (``return_type is
    # None``) keep the generic sentinel for the same reason.
    # For shaped return types, the schema stub keeps the annotated shape
    # and the error context is only logged.
    if error is not None and return_type is dict:
        return {
            'status': 'error',
            '_plurai_stub_error': error,
            '_plurai_tool': name,
            'message': f'Mock for {name!r} failed: {error}',
        }
    if return_type is None:
        return {'status': 'ok', 'mock': True, 'tool': name}
    stub = schema_stub_from_annotation(return_type)
    if stub is _NO_USEFUL_TYPE:
        return {'status': 'ok', 'mock': True, 'tool': name}
    return stub


def _warn_once_no_client(name: str) -> None:
    global _warned_no_client
    if _warned_no_client:
        return
    _warned_no_client = True
    logger.warning(
        'plurai.configure() has not been called; tool %r is returning a '
        'schema stub without a mock-env roundtrip. Subsequent uncalled-'
        'configure tool calls will be silent.',
        name,
    )


def _resolve_mock_response(
    name: str,
    args: dict,
    return_type: Any,
    *,
    turn: int | None = None,
) -> Any:
    """Route a tool call through the configured mock-env; substitute a
    schema-shaped stub when the backend has no handler or is unreachable.

    Failure modes that flow into the stub fallback (each emits a log line
    or — for the in-process no-handler sentinel — passes through without one
    because that's the expected steady state during incremental handler buildup):
      1. ``plurai.configure()`` was never called (``_client`` is None) —
         logged once per process; subsequent calls are silent so the
         customer's logs don't drown in repeats.
      2. The mock-env raised a network/transport error mid-run — logged
         per occurrence, observability-must-not-abort-simulation contract.
      3. The mock-env returned a non-dict body (None, list, scalar) — the
         shape is malformed; we can't trust ``.get("response", ...)`` and
         we don't want a passthrough that explodes downstream.
      4. The mock-env returned ``{"error": ...}`` without ``response`` —
         the server signaled trouble; substitute a stub instead of
         handing the agent an error dict that masquerades as data.
      5. The in-process mock returned the no-handler sentinel — substitute
         the schema stub. No log; this is the expected state for any tool
         the customer hasn't written a handler for yet.

    A deployed mock's refusal of a tool it has no handler for
    (``ENGINE_REFUSAL_KEY``) is NOT silent: it is recorded as a non-transient
    ``mock_runtime_fault`` (``plurai.UnregisteredTool``), so the scenario is
    not valid rather than graded as the agent's. A refusal a process the agent
    spawned received is read off the event log at finalize the same way
    (``plurai.child``).

    Real handler responses pass through verbatim, for tools that have
    ``@plurai.mock`` handlers and for env-generator-generated handlers in
    the backend's mock environments alike.

    ``turn`` is the recorded event's OUTER CONVERSATION turn. Leave it unset
    (the normal path): the shared turn context supplies the index the child is
    driving, and falls back to 0 where no harness ever reported one. Pass it
    explicitly only to override that resolution.
    """
    # NB: each branch that goes to the stub fallback passes its specific
    # ``error=...`` so the customer's wrapper can render a meaningful
    # ToolMessage(status='error') instead of an empty-dict ``{}`` — which
    # reaches the agent's LLM as a success, and it rephrases the request
    # and retries indefinitely.
    # The single exception is the in-process no-handler-sentinel path: that's the
    # legitimate "I haven't written this mock yet" state, not a failure. (A deployed
    # mock's no-handler refusal is recorded as a mock_runtime_fault below.)
    # Declared encoded fields reach the wire, the handler and the trail decoded; the reply is encoded below.
    args = apply_payload_codecs(name, args, 'args')
    if _registry._client is None:
        _warn_once_no_client(name)
        return _stub_or_legacy_sentinel(
            name,
            return_type,
            error='plurai.configure() not called; no mock service available',
        )
    sid = _active_session_id()
    try:
        resp = _registry._client.execute_mock(sid, name, args, current_turn_or_zero() if turn is None else turn)
    except _CONNECTION_ERROR_TYPES as exc:
        # The mock did not answer this call: a non-2xx from a hop in front of it (the mock itself
        # answers every call with HTTP 200, errors in-band), or a transport failure. The agent gets
        # a stub whose text names no harness remedy, the typed fact is recorded for the finding the
        # runner writes, and the call is noted as a mock runtime fault — the same verdict as a mock
        # that answered with a runtime fault (below): the scenario did not measure the agent, so it
        # is not_valid. Its typed cause decides whether a re-run answers it (``transient``): a dropped
        # transport does, a token / deploy / hold / wrong URL does not. ``restart_shaped`` is the
        # narrower fact the mid-scenario re-drive reads (a restart that lost the seeded session).
        if isinstance(exc, MockServiceError):
            status_code, code, transport = exc.status_code, exc.code, exc.transport
        else:
            status_code, code, transport = None, None, transport_failure(exc)
        resets_at = exc.resets_at if isinstance(exc, MockServiceError) else None
        restart_shaped = _is_restart_shaped(status_code=status_code, code=code, transport=transport)
        route_cause = mock_route_cause(status_code=status_code, code=code, transport=transport)
        # A retry budget the client spent on a transient failure: the mock is unavailable, whatever status it ended on.
        # The refusals rank credit, capacity, unavailable (see ``refusal_cause``): a block beside a credit or capacity code is not read.
        unavailable = None if route_cause in ('credit', 'capacity') else unavailable_block_of(exc)
        if unavailable is not None:
            route_cause = 'unavailable'
        credit = credit_block_from_body({'code': code, 'resets_at': resets_at}) if route_cause == 'credit' else None
        capacity = (
            capacity_block_from_body({'code': code, 'resets_at': resets_at, 'reset_source': exc.reset_source if isinstance(exc, MockServiceError) else None})
            if route_cause == 'capacity'
            else None
        )
        stub = _stub_or_legacy_sentinel(name, return_type, error=f'tool {_plain(repr(name))} returned no result (mock transport error)')
        call = UnansweredCall(
            tool=name,
            turn=current_turn_or_zero() if turn is None else turn,
            status_code=status_code,
            code=code,
            transport=transport,
            served=_served_stub(stub),
            restart_shaped=restart_shaped,
            resets_at=resets_at if credit is not None or capacity is not None else None,
            reset_source=capacity.reset_source if capacity is not None else None,
            unavailable=unavailable.to_record() if unavailable is not None else None,
        )
        _record_unanswered(sid, call)
        if credit is not None:
            # A refusal for credit names its kind and nothing else: the status and the backend's body are not for the terminal.
            message = f'mocked tool {name!r} was refused for credit ({credit.code})'
            logger.warning('mock-env refused tool %r for credit (%s); serving %s.', name, credit.code, call.served)
        elif capacity is not None or unavailable is not None:
            message = f'mocked tool {name!r} was refused: {BACKEND_REFUSED_PHRASE}'
            logger.debug('mock-env refused tool %r: %s; serving %s.', name, BACKEND_REFUSED_PHRASE, call.served)
        else:
            message = (
                f'mocked tool {name!r} got no answer from the mock (HTTP {status_code}, code {code}, transport {transport}): {unanswered_call_remedy(call)}'
            )
            logger.warning(
                'mock-env did not answer tool %r (HTTP %s, code %s, transport %s): %r; serving %s. %s.',
                name,
                status_code,
                code,
                transport,
                exc,
                call.served,
                unanswered_call_remedy(call),
            )
        note_harness_fault(
            'plurai.MockRouteFault',
            message,
            kind='mock_runtime_fault',
            transient=MOCK_ROUTE_CAUSE_TRANSIENT[route_cause],
            route_cause=route_cause,
            credit=credit.to_record() if credit is not None else None,
            capacity=capacity.to_record() if capacity is not None else None,
            unavailable=unavailable.to_record() if unavailable is not None else None,
        )
        return stub

    # Defensive: mock-env contract says ``{"response": ...}`` envelope, but
    # transport quirks (200 OK with null body, malformed JSON deserialized
    # as scalar, custom proxy responses) can produce other shapes. Without
    # this guard, ``.get("response", ...)`` AttributeErrors and aborts the
    # simulation — exactly the contract violation the resolver exists to
    # prevent.
    if not isinstance(resp, dict):
        logger.warning(
            'mock-env returned non-dict for tool %r (got %s); using schema stub.',
            name,
            type(resp).__name__,
        )
        return _stub_or_legacy_sentinel(
            name,
            return_type,
            error=f'mock-env returned non-dict body ({type(resp).__name__})',
        )
    if 'error' in resp and 'response' not in resp:
        # Server told us something went wrong. Don't pretend the error dict
        # is a tool result; surface it via the log + serve a stub instead.
        logger.warning(
            'mock-env error for tool %r: %r; using schema stub.',
            name,
            resp.get('error'),
        )
        return _stub_or_legacy_sentinel(
            name,
            return_type,
            error=f'mock-env: {resp.get("error")}',
        )

    response = resp.get('response', resp)
    if isinstance(response, dict) and response.get('_plurai_write_refused') is True:
        # A handler write the mock refused (its key names no entity): the harness's fault. Recorded
        # before the stub swap below, which would otherwise hand the agent a clean-looking result.
        note_harness_fault(
            'plurai_mockenv.MisaddressedWrite', str(response.get('error', 'world write refused')).removeprefix('MisaddressedWrite: '), kind='harness_fault'
        )
    if isinstance(response, dict) and response.get(ENGINE_REFUSAL_KEY) is True:
        # The deployed mock has no handler for this tool: the mock's gap, not the agent's. The agent is still
        # handed the failed call (``mock_result``); the turn is a mock runtime fault, as for a runtime fault below.
        note_harness_fault(
            'plurai.UnregisteredTool',
            f'mocked tool {name!r} has no handler in the deployed mock; the scenario did not measure the agent on it',
            kind='mock_runtime_fault',
            transient=False,
        )
    if isinstance(response, dict) and response.get(RUNTIME_FAULT_TAG) is True:
        # A mocked tool that generates its answer at run time could not (its model timed out, its
        # transport failed): the backend's mock runtime failed, not the world and not the harness's code.
        # The runtime's own cause and class ride its sibling; a mock that predates them reads as not transient.
        runtime = response.get(RUNTIME_SIBLING)
        runtime = runtime if isinstance(runtime, dict) else {}
        # A call the backend refused for credit is the organization's standing, not a fault a retry clears, whatever
        # class the runtime gave it; a block that names no known kind reads as the unknown-kind refusal.
        refused = runtime.get('cause') == RUNTIME_CREDIT_CAUSE
        credit = credit_block_or_unknown(runtime.get('credit')) if refused else None
        # A call refused because the provider refused the backend's account is the same answer on every call for now:
        # never transient, and it stops the run like a credit refusal. The block is read only when it is the typed one;
        # a cause with no readable block still names the stop, with no reset time.
        at_capacity = not refused and runtime.get('cause') == RUNTIME_CAPACITY_CAUSE
        capacity = (capacity_block_from_body(runtime.get('capacity')) or CapacityBlock()) if at_capacity else None
        # An unavailable backend answers every call of this turn the same way: never transient within the turn. The run
        # re-drives the scenario once on a fresh mock world, as for a transient fault, and goes on whatever that gives. The
        # block is read only when it is the typed one; a cause with no readable block still names the cause.
        unavailable_now = not refused and not at_capacity and runtime.get('cause') == RUNTIME_UNAVAILABLE_CAUSE
        unavailable = (unavailable_block_from_body(runtime) or UnavailableBlock('backend')) if unavailable_now else None
        grounding = runtime.get('grounding') or response.get('error', 'runtime fault')
        if at_capacity or unavailable_now:
            # The grounding names the provider and its status: operator diagnostics, kept off every customer line.
            logger.debug('mock-env tool %r refused, %s: %s', name, BACKEND_REFUSED_PHRASE, grounding)
            message = f'mocked tool {name!r} was refused: {BACKEND_REFUSED_PHRASE}'
        else:
            message = f'mocked tool {name!r} could not generate its answer: {grounding}'
        note_harness_fault(
            'plurai.RuntimeToolFault',
            message,
            kind='mock_runtime_fault',
            transient=False if refused or at_capacity or unavailable_now else runtime.get('fault_class') == 'transient',
            route_cause=refusal_cause(credit, capacity, unavailable),
            credit=credit.to_record() if credit is not None else None,
            capacity=capacity.to_record() if capacity is not None else None,
            unavailable=unavailable.to_record() if unavailable is not None else None,
        )
    # Handler-raised errors are tagged with ``_plurai_handler_error`` by
    # ``LocalMockService._handle`` so the resolver can tell them apart from
    # a real handler returning a multi-key dict that happens to include
    # ``error``. When the tag is present, swap in the schema stub — the
    # customer's NotImplementedError-bodied scaffold flows through here
    # and lands the agent on a structurally-correct response, not on a
    # raw error dict. A tool whose ``error_channel`` is DECLARED gets the
    # explicit error stub whatever its return type: a handler that raised is
    # a failed call (the envelope's ``handler_error`` row), which
    # ``plurai.mock_result`` reads as ``MockToolError``.
    if isinstance(response, dict) and response.get('_plurai_handler_error') is True:
        err = response.get('error', '<no message>')
        logger.warning(
            'mock-env handler raised for tool %r: %s; using schema stub.',
            name,
            err,
        )
        return _stub_or_legacy_sentinel(
            name,
            dict if declared_error_channel(name) is not None else return_type,
            error=f'handler raised: {err}',
        )
    if is_no_handler_sentinel(response, name):
        # Legitimate "no handler written yet" — keep the silent stub.
        return _stub_or_legacy_sentinel(name, return_type)
    try:
        return apply_payload_codecs(name, response, 'returns')
    except PayloadEncodeError:
        # Recorded as a mock runtime fault where it was found; the agent gets a failed call, never the decoded shape.
        return _stub_or_legacy_sentinel(name, return_type, error=f'tool {_plain(repr(name))} returned no result (mock answer could not be encoded)')
