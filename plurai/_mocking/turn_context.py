"""The conversation turn index that every ``/mock/execute`` dispatch stamps on its event.

The mock engine records one ``tool_call`` event per call and files it under the ``turn``
the caller sent; every consumer (``tool_events.json``, the per-turn observability graft,
any turn-scoped judge) reads that field as the OUTER CONVERSATION turn. Only the child
knows that index — it arrives on the ``UserTurn`` envelope — while the three dispatchers
that POST the wire (:class:`~plurai._mocking.interception.MockToolNode`,
:func:`~plurai._mocking.interception.mock_wrap` and the code action channel's
:func:`~plurai._mocking.code_intercept.code_intercept_call`) sit well below it, one of
them in a different process. This module is the single place that carries it across both
seams:

* in-process (``MockToolNode`` / ``mock_wrap``): the module global, set by
  ``plurai.child`` at the top of each user turn;
* out-of-process (the code action channel runs inside a nested skill session's own
  process): the ``PLURAI_TURN_INDEX`` environment variable, which
  :func:`set_current_turn` mirrors so a session spawned during that turn inherits it
  (``ClaudeAgentOptions(env=…)`` merges over ``os.environ``). A session spawned ONCE and
  reused across turns keeps the value it inherited — stale, not wrong-by-construction,
  and the merge in ``plurai.observability.tool_call_recovery`` never lets a stale stamp
  reorder the mock lane against itself.

"No turn reported" is a first-class state, not an error: a harness that drives the mocks
without ``plurai.child`` (a unit test, an embedding runner) never reports one, and those
dispatches stamp turn 0.

It also carries the other fact only the child knows — whether the AGENT is running — which
``MockServiceClient.update_state`` stamps on every world write as its ``source`` (see
:func:`agent_invocation`).

Stdlib-only by design — ``code_intercept`` imports it inside the skill's grandchild
process, which has nothing but the SDK.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Literal, NamedTuple

# Read by the out-of-process code action channel; written by set_current_turn.
TURN_ENV_VAR = 'PLURAI_TURN_INDEX'

_current_turn: int | None = None

#: Raw env values already reported as unparseable. ``current_turn`` runs once per dispatch, so a
#: persistently malformed breadcrumb would otherwise print on every mocked tool call.
_warned_env_values: set[str] = set()


def set_current_turn(turn_idx: int | None) -> None:
    """Record the conversation turn now being driven; ``None`` clears it.

    Mirrors the value into ``os.environ`` so processes spawned by the harness during
    this turn (the code action channel's nested session) inherit it.
    """
    global _current_turn
    _current_turn = turn_idx
    if turn_idx is None:
        os.environ.pop(TURN_ENV_VAR, None)
    else:
        os.environ[TURN_ENV_VAR] = str(turn_idx)


def current_turn() -> int | None:
    """The conversation turn index, or ``None`` when nothing has reported one.

    In-process state wins over the environment: a child that set it directly is
    authoritative, and the env var only matters where the global was never set (a
    spawned process). An unparseable env value is treated as "not reported" rather than
    raising — a mock dispatch must never die on a malformed breadcrumb.
    """
    if _current_turn is not None:
        return _current_turn
    raw = os.environ.get(TURN_ENV_VAR)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        # Named on stderr (the ``[plurai] …`` convention the parent's pump promotes to WARNING):
        # the caller then stamps turn 0, so an unreported bad value produces a mis-ordered event
        # indistinguishable from a harness that legitimately reports no turn.
        if raw not in _warned_env_values:
            _warned_env_values.add(raw)
            print(f'[plurai] {TURN_ENV_VAR}={raw!r} is not an integer — treating it as "no turn reported"; events stamp turn 0.', file=sys.stderr)
        return None


def current_turn_or_zero() -> int:
    """The turn index to put on the wire — 0 when no turn was reported."""
    turn = current_turn()
    return 0 if turn is None else turn


# --- Who made a world write ---------------------------------------------------------------------
#
# Every world write made outside a mock handler — the agent's storage adapter, a ``mode: real``
# tool, the harness's seed — crosses ONE funnel, ``MockServiceClient.update_state``, and the
# mock records it verbatim as ``{'type': 'state_update', 'source': …}``. The record carries no
# tool call id and no turn, and none can be derived for it, so the one fact worth stamping is
# WHOSE CONTEXT the write came from — and only the child knows that, because only the child
# knows when the agent is running.
#
# A ``ContextVar``, not a module global and not ``current_turn()``: both of those are
# process-wide, so a thread the harness started BEFORE the turn and that writes DURING it reads
# them as "the agent is running", and ``current_turn()`` also reads an inherited
# ``PLURAI_TURN_INDEX`` during the seed. A context variable is inherited only by work the agent
# call itself starts — an awaited coroutine, an ``asyncio`` task, and the executor threads
# LangGraph and ``asyncio.to_thread`` dispatch sync tools on (both copy the context) — and by
# nothing else. A raw ``threading.Thread`` the agent starts itself does not copy it, so its write
# stays ``'external'``: the conservative direction.

#: ``source`` on a world write made from the agent invocation's own context. Shared wire
#: vocabulary with the Plurai backend; a parity test keeps the two values equal.
AGENT_WRITE_SOURCE = 'agent'
#: ``source`` on a world write the HARNESS made while preparing the world — inside the child's call
#: to the harness's ``reset`` / ``seed_state`` hook, before any turn. The seed is setup, not a state
#: change anyone made during the conversation. Shared wire vocabulary with the Plurai backend, held
#: equal by the same parity test.
HARNESS_WRITE_SOURCE = 'harness'
#: ``source`` on a world write the simulated USER made: an act she took on what the product shows
#: her (approve, pay), applied by the child at the open of the agent's next turn. Never the agent's.
#: Shared wire vocabulary with the backend, parity-checked by the
#: same test.
USER_WRITE_SOURCE = 'user'
#: ``source`` on every other outside-handler write — the mock engine's own default, which means
#: only "not inside a mock handler" and says nothing about who wrote.
EXTERNAL_WRITE_SOURCE = 'external'

_world_write_source: ContextVar[str | None] = ContextVar('plurai_world_write_source', default=None)


@contextmanager
def world_write_scope(source: str) -> Iterator[None]:
    """Stamp every world write made from this context with ``source`` until the block exits."""
    token = _world_write_source.set(source)
    try:
        yield
    finally:
        _world_write_source.reset(token)


def agent_invocation() -> AbstractContextManager[None]:
    """The scope the child opens around ONE agent call — the call and its ``await`` both."""
    return world_write_scope(AGENT_WRITE_SOURCE)


def harness_setup() -> AbstractContextManager[None]:
    """The scope the child opens around the harness's ``reset`` / ``seed_state`` hook — the call and its ``await`` both."""
    return world_write_scope(HARNESS_WRITE_SOURCE)


def user_action() -> AbstractContextManager[None]:
    """The scope the child opens around applying ONE act of the simulated user — the call and its ``await`` both."""
    return world_write_scope(USER_WRITE_SOURCE)


def world_write_source() -> str:
    """The ``source`` a world write made from the current context carries."""
    return _world_write_source.get() or EXTERNAL_WRITE_SOURCE


# --- Harness faults, recorded where they are raised ------------------------------------------------
#
# A harness fault raised inside the agent's frames — a world write the mock refused, a harness
# seed-incoherence raise — crosses the AGENT's code on its way out, and an agent tool that catches
# ``Exception`` turns it into an ordinary tool error. Typed where it is caught, it is lost there. So
# it is recorded where it is RAISED, and the child reads the record after each hook.
#
# Process-level, not a ``ContextVar``: a tool run on a thread that did not copy the context must
# still reach the child. The child runs one session and serialises its hooks, and drains the record
# at the start of each one.

#: ``harness_fault``: the harness's own defect (a refused write, a ``HarnessFault``). ``mock_runtime_fault``:
#: the backend's mock runtime could not generate a mocked tool's answer — not the harness's code to fix.
#: ``provider_infra``: the agent's Claude Code CLI ended a turn's stop on a provider or sign-in error
#: (``plurai.host_effects``' ``StopFailure`` hook) — recorded the moment that stop happens, so it is
#: typically neither the agent's behaviour nor the harness's code. On a turn that RETURNED the child
#: settles these CLI notes by how the turn's last query ended (a retry that ended cleanly supersedes an
#: earlier provider stop); on a turn that RAISED the record is not cleared by a later stop, so a turn that
#: failed on a provider error and was then retried into an agent-owned stop still classifies ``provider_infra``.
HarnessFaultKind = Literal['harness_fault', 'mock_runtime_fault', 'provider_infra']

#: Why a ``/mock/execute`` call went unanswered (``plurai.MockRouteFault``), typed by who answered it.
#: ``transport``: nothing that is the mock answered — a gateway 502/503/504 or a 429 throttle in front of it, or a refused,
#: reset or timed-out connection; the route dropped this one call and a re-run answers it.
#: ``held_for_edit``: a 503 holding the env until an interrupted edit session is finalized (a human's to clear).
#: ``auth``: a 401/403, the token. ``not_deployed``: a 410. ``not_a_mock``: any other status, or a 2xx
#: that is not JSON — something that is not a plurai mock answered. ``credit``: the backend refused the call for credit
#: (a 402 or a 503 whose body names a credit kind); the credit block rides beside the cause. ``capacity``: the backend
#: refused it because the provider refused its account (a body whose ``code`` is ``capacity_exhausted``, or a
#: runtime fault whose cause is ``capacity``); the capacity block rides beside the cause. ``unavailable``: the backend said
#: it is unavailable (an answer carrying the typed ``unavailable`` block, read by placement, or a runtime fault whose cause is
#: ``unavailable``); the unavailable block rides beside the cause.
MockRouteCause = Literal['transport', 'held_for_edit', 'auth', 'not_deployed', 'not_a_mock', 'credit', 'capacity', 'unavailable']


def refusal_cause(credit: object | None, capacity: object | None, unavailable: object | None) -> MockRouteCause | None:
    """The route cause of the refusal block that is set, in the order the run reads them: credit, capacity, unavailable;
    ``None`` when none is. The one place that precedence lives."""
    if credit is not None:
        return 'credit'
    if capacity is not None:
        return 'capacity'
    return 'unavailable' if unavailable is not None else None


class HarnessFaultNote(NamedTuple):
    exc_class: str
    message: str
    kind: HarnessFaultKind
    #: A ``mock_runtime_fault`` the runtime typed as transient (one more try could answer); ``None`` otherwise.
    transient: bool | None = None
    #: On a ``plurai.MockRouteFault`` only: its typed cause; ``None`` otherwise.
    route_cause: MockRouteCause | None = None
    #: On a credit refusal only (``route_cause == 'credit'``): the credit block ``{code, resets_at?}``; ``None`` otherwise.
    credit: dict[str, str] | None = None
    #: On a capacity refusal only (``route_cause == 'capacity'``): the capacity block ``{code, resets_at?, reset_source?}``; ``None`` otherwise.
    capacity: dict[str, str] | None = None
    #: On an unavailable answer only (``route_cause == 'unavailable'``): the unavailable block ``{cause, code?}``; ``None`` otherwise.
    unavailable: dict[str, str] | None = None


_harness_faults: list[HarnessFaultNote] = []


def note_harness_fault(
    exc_class: str,
    message: str,
    *,
    kind: HarnessFaultKind,
    transient: bool | None = None,
    route_cause: MockRouteCause | None = None,
    credit: dict[str, str] | None = None,
    capacity: dict[str, str] | None = None,
    unavailable: dict[str, str] | None = None,
) -> None:
    """Record a harness fault at its raise site — whatever frame catches it afterwards."""
    _harness_faults.append(HarnessFaultNote(exc_class, message, kind, transient, route_cause, credit, capacity, unavailable))


def take_harness_faults() -> list[HarnessFaultNote]:
    """Every note recorded since the last take, oldest first; the record is left empty."""
    taken = list(_harness_faults)
    del _harness_faults[: len(taken)]
    return taken


def take_harness_faults_raised_by(exc_class: str) -> list[HarnessFaultNote]:
    """The notes ``exc_class`` recorded since the last take, oldest first; every other note stays."""
    taken = [note for note in _harness_faults if note.exc_class == exc_class]
    _harness_faults[:] = [note for note in _harness_faults if note.exc_class != exc_class]
    return taken


# --- The agent's CLI stops -------------------------------------------------------------------------
#
# How each query the agent's Claude Code CLI ran this turn ENDED (``plurai.host_effects``' ``Stop`` and
# ``StopFailure`` hooks), in order — a clean end, a stop the AGENT owns (its model's reply hit its output
# budget), or an error ending the CLI's result cannot tell apart from the agent's words. Not a harness
# fault: the LAST query's end is how the turn ended. The child reads it for a raise the turn made on it,
# and for a turn that returned (``plurai.child``: an infrastructure ending is typed on the side).
#
# Each query's START is recorded too (the ``UserPromptSubmit`` hook), with the CLI's own ``prompt_id`` —
# the id every hook payload of that query carries — so an end is paired with ITS query, never by count:
# the CLI fires no end hook at all for a query its step cap (``max_turns``) ended.


class CliStop(NamedTuple):
    #: The ``StopFailure`` error type, the CLI's own enum; ``''`` for a clean end.
    error: str
    #: Whether the agent owns it.
    agent_owned: bool
    #: A clean end (the ``Stop`` hook), not a failure.
    ok: bool = False
    #: The CLI's own text for a failure (``StopFailure``'s ``last_assistant_message``: its "API Error: …" banner).
    last_assistant_message: str = ''
    #: The CLI's id for the query this ended (``''`` when the payload carried none).
    prompt_id: str = ''


class CliQueryStart(NamedTuple):
    #: The CLI's id for the query (``UserPromptSubmit``'s ``prompt_id``).
    prompt_id: str
    #: The CLI's own session transcript (``transcript_path``), where a step-capped query leaves its only trace.
    transcript_path: str = ''


_cli_stops: list[CliStop] = []
_cli_queries_started: list[CliQueryStart] = []


def note_cli_query_started(prompt_id: str, *, transcript_path: str = '') -> None:
    """Record that the CLI started one query (a prompt was submitted to it)."""
    _cli_queries_started.append(CliQueryStart(prompt_id, transcript_path))


def note_cli_stop(error: str, *, agent_owned: bool, ok: bool = False, last_assistant_message: str = '', prompt_id: str = '') -> None:
    """Record how one CLI query ended — a clean ``Stop`` or a ``StopFailure``, whoever owns it."""
    _cli_stops.append(CliStop(error, agent_owned, ok, last_assistant_message, prompt_id))


def last_cli_query() -> CliQueryStart | None:
    """The last query the CLI started since the last take, or ``None``. Read, not taken."""
    return _cli_queries_started[-1] if _cli_queries_started else None


def cli_query_end(prompt_id: str) -> CliStop | None:
    """How the query ``prompt_id`` ended, if its end was recorded since the last take. Read, not taken."""
    return next((stop for stop in reversed(_cli_stops) if stop.prompt_id == prompt_id), None)


def peek_cli_stops() -> list[CliStop]:
    """Every end recorded since the last take, oldest first. Read, not taken."""
    return list(_cli_stops)


def take_cli_stops() -> list[CliStop]:
    """Every end recorded since the last take, oldest first; the record, and the recorded starts, are left empty."""
    taken = list(_cli_stops)
    del _cli_stops[: len(taken)]
    _cli_queries_started.clear()
    return taken
