"""Code action-channel runtime — route an intercepted vendor SDK call to the mock.

The ``code`` action channel runs a skill agent the way production does: the skill
writes and runs real SDK code (e.g. ``resend.Emails.send(...)``) via Bash. A generic
import hook (``code_hook.py``) shadows the vendor module and, for every public-method
call, resolves the ``(module, method) → tool`` mapping from a ``code_surface.yaml`` and
dispatches here.

``code_intercept_call`` is just *another caller of the frozen ``/mock/execute`` wire* —
it changes nothing about the contract. It reads the active session id and mock
url from the environment (injected by ``simulate.py`` into the nested skill session via
``ClaudeAgentOptions(env=…)``), POSTs ``{session_id, tool, args, turn}``, and returns the
mock tool's native return value. Per-method *return-shape* fidelity (Python ``dict`` vs
Node ``{data, error}``) is the hook's job — see ``shape_native_return`` — because only
the hook knows which vendor method produced the call.

No intra-plurai imports beyond the client and the payload codecs (both stdlib-only): the runtime
must import cleanly inside the skill's grandchild process, which only has the SDK.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from plurai._mocking.client import MockServiceClient, MockServiceError
from plurai._mocking.codecs import PayloadEncodeError, _plain, apply_payload_codecs
from plurai._mocking.turn_context import current_turn_or_zero

# Env var names the hook + simulate.py agree on. PLURAI_MOCK_URL is the mock
# service base url (the same one `plurai.configure` dialled); PLURAI_SESSION_ID
# is the active session id read from `plurai.active_session_id()` at seed time.
# The conversation turn rides its own env var, read via the shared turn context
# (``plurai._mocking.turn_context.TURN_ENV_VAR``) — the nested session inherits
# it from the child process that is driving the turn.
_ENV_SESSION_ID = 'PLURAI_SESSION_ID'
_ENV_MOCK_URL = 'PLURAI_MOCK_URL'


# One client per mock url for the life of the process, so the connection (and its handshake) is
# reused across the tool calls of a run instead of reopened for each.
_clients: dict[str, MockServiceClient] = {}
_clients_lock = threading.Lock()


def _client_for(mock_url: str) -> MockServiceClient:
    with _clients_lock:
        client = _clients.get(mock_url)
        if client is None:
            client = _clients[mock_url] = MockServiceClient(base_url=mock_url)
        return client


class CodeInterceptConfigError(RuntimeError):
    """Raised when the interception env (session id / mock url) is not present.

    Fail-closed: an intercepted vendor call with no session/mock wiring must NOT
    silently reach the real vendor API — it raises here so the skill snippet
    errors loudly instead of degrading to a live send.
    """


class CodeInterceptTransportError(RuntimeError):
    """Raised when the ``/mock/execute`` wire itself fails (HTTP error / transport / bad body).

    Distinct from :class:`CodeInterceptConfigError` (env/config wiring): this is a wire-level
    failure once the env IS present — an HTTP non-2xx, a connection error, or a non-JSON
    body on a 2xx. Fail-closed: surfacing it loudly means a dead/misbehaving mock can never
    be mistaken for a successful native return. The Node hook raises an identically-named
    ``CodeInterceptTransportError`` on the same cases; this is the Python parity.
    """


def code_intercept_call(tool: str, args: dict[str, Any]) -> Any:
    """Route an intercepted vendor SDK call through the frozen ``/mock/execute`` wire.

    Reads ``PLURAI_SESSION_ID`` + ``PLURAI_MOCK_URL`` from the environment, builds a
    :class:`MockServiceClient`, and calls ``execute_mock(sid, tool, args)``. Returns the
    mock tool's **native return value** (the inner ``response`` payload) — the raw dict a
    Python vendor SDK method would return. Node ``{data, error}`` enveloping is applied
    by the import hook, which alone knows the per-method return shape from the
    ``code_surface.yaml``; this runtime stays language-agnostic.

    Raises :class:`CodeInterceptConfigError` when the env wiring is missing (fail-closed:
    never fall through to the real vendor API), and :class:`CodeInterceptTransportError`
    when the ``/mock/execute`` wire itself fails (HTTP error / connection error / non-JSON
    body) — fail-closed so a dead mock is never mistaken for a native return. An in-band
    ``/mock/execute`` *error envelope* (unknown session, etc.) is surfaced unchanged so the
    snippet sees the same failure the semantic-tools channel would.
    """
    sid = os.environ.get(_ENV_SESSION_ID)
    mock_url = os.environ.get(_ENV_MOCK_URL)
    if not sid or not mock_url:
        missing = [name for name, val in ((_ENV_SESSION_ID, sid), (_ENV_MOCK_URL, mock_url)) if not val]
        raise CodeInterceptConfigError(
            f'code action-channel interception is not wired: missing {", ".join(missing)} in the environment. '
            'The nested skill session must inject PLURAI_SESSION_ID (from plurai.active_session_id()) and '
            'PLURAI_MOCK_URL via ClaudeAgentOptions(env=…). Refusing to reach the real vendor API.'
        )

    client = _client_for(mock_url)
    # The same codec seam as ``_resolve_mock_response``. Inert in the skill's grandchild: nothing there
    # reads the harness config (``use_harness_config``), so no codec is declared — and the plugin refuses
    # ``payload_codecs`` on code-surface tools (``payload_codec_unsupported_surface``).
    args = apply_payload_codecs(tool, args, 'args')
    # The client raises MockServiceError on any wire-level failure (HTTP non-2xx, transport
    # error, non-JSON body). Re-raise as a CodeInterceptTransportError so the code channel
    # fails closed with a channel-native error (Node-parity) rather than leaking the
    # client's internal exception type up into the snippet.
    try:
        # The recorded event's turn is the OUTER conversation turn, inherited from the
        # driving child process; 0 when nothing reported one (a hand-run snippet).
        result = client.execute_mock(sid, tool, args, current_turn_or_zero())
    except MockServiceError as e:
        # The text the SNIPPET (and so the agent) sees is the semantic channel's neutral stub text: a
        # non-2xx is a hop in front of the mock, and its body names harness remedies (a token,
        # deploy_env) that would read as the tool's answer. The typed cause stays chained on __cause__.
        raise CodeInterceptTransportError(f'tool {_plain(repr(tool))} returned no result (mock transport error)') from e
    # `/mock/execute` returns {'response': <native tool return>, 'state_version': N}
    # on success, or {'error': ..., 'status': ...} on a bad session. Hand back the
    # inner native return on success; surface the error envelope as-is otherwise so
    # the snippet's own error handling fires (the hook maps it into the vendor's
    # native error shape when needed).
    #
    # Success requires BOTH a `response` key AND no `error`: a `{'response': null,
    # 'error': ...}` envelope (a tool that failed but still carries a response slot)
    # must be surfaced as the error envelope, not unwrapped to a misleading `None`
    # "native return".
    #
    # The mock envelope contract guarantees `error` is set on EVERY failure, so a
    # `{'response': None, 'error': None}` envelope is an intentional success that returns
    # `None` (a tool whose native return genuinely is null) — not a masked failure.
    if isinstance(result, dict) and 'response' in result and result.get('error') is None:
        try:
            return apply_payload_codecs(tool, result['response'], 'returns')
        except PayloadEncodeError as e:
            raise CodeInterceptTransportError(f'tool {_plain(repr(tool))} returned no result (mock answer could not be encoded)') from e
    return result
