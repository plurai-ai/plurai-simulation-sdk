"""Child-side entry point: ``python -m plurai.child <simulate_path>``.

Imports the customer's ``simulate.py`` by path, then drives a JSONL IPC loop
on stdin / stdout. The parent (``Orchestrator._run_subprocess``) sends
``Init`` → ``UserTurn`` ... → ``Finalize`` → ``Shutdown`` envelopes; the child
replies with ``InitOk`` / ``InitError``, ``AgentResponse`` / ``AgentError``,
``FinalizeResponse``.

The per-message handler logic lives on :class:`_ChildSession` (pure, no I/O)
so it can be unit-tested by feeding envelope objects directly; the I/O loop
just reads JSONL, dispatches, and writes responses.
"""

from __future__ import annotations

import asyncio
import datetime
import decimal
import enum
import importlib.util
import inspect
import io
import json
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath, PurePosixPath
from types import ModuleType
from typing import Any, Final, TypeGuard, cast

from plurai_mockenv import _registry
from plurai_mockenv._registry import _set_active_session_id
from plurai_mockenv.dialog import DialogEntry, PriorToolCall, cap_dialog, cap_prior_calls

import plurai
from plurai import egress_guard, file_rows, world_delta
from plurai.world import seed_rows
from plurai.world.seed_landing import BACKEND_TRANSPORT_FAILURE_KEY
from plurai._entry_witness import EntryWitness
from plurai._platform import is_windows, utf8_stdio
from plurai._input_reads import InputReads, is_bound_document, unread_inputs
from plurai._mocking import session as mock_session
from plurai._mocking.client import MockServiceError
from plurai._mocking.interception_watch import interception_changed
from plurai._mocking.result import ENGINE_REFUSAL_KEY, use_harness_config
from plurai._mocking.turn_context import (
    CliQueryStart,
    CliStop,
    HarnessFaultNote,
    agent_invocation,
    cli_query_end,
    harness_setup,
    last_cli_query,
    note_harness_fault,
    peek_cli_stops,
    refusal_cause,
    set_current_turn,
    take_cli_stops,
    take_harness_faults,
    take_harness_faults_raised_by,
    user_action,
)
from plurai._protocol import (
    _json_shaped,
    INTERRUPT_ACT,
    IPC_LINE_LIMIT_BYTES,
    IPC_TRIMMED_KEY,
    PIPELINE_NO_HOOK_MESSAGE,
    PROVIDER_INFRA_KIND,
    PIPELINE_NOT_INITIALIZED_MESSAGE,
    AGENT_RAISED_KIND,
    AdapterErrorSubkind,
    AgentError,
    AgentFault,
    AgentResponse,
    ErrorKind,
    Finalize,
    FinalizeResponse,
    HarnessFault,
    Init,
    InitError,
    InitOk,
    Message,
    MockMode,
    OversizedIpcLine,
    PipelineError,
    PipelineInvoke,
    PipelineResult,
    ProtocolDecodeError,
    Shutdown,
    UserTurn,
    decode,
    encode,
    is_async_run_footgun,
    is_recursion_limit_error,
    observed_actions_or_none,
    read_ipc_line,
)
from plurai.agent_io import (
    AGENT_IO_KEY,
    DEFAULT_AGENT_IO,
    INPUT_FIELD,
    INPUT_PART_KINDS,
    MAY_OPEN_FIELD,
    AgentIO,
    declared_agent_io,
)
from plurai.channels import (
    CHANNEL_INPUTS_KEY,
    EMPTY_PLAN,
    AppliedChannelInput,
    ChannelPlan,
    ChannelStatus,
    PlannedChannelInput,
    item_write,
    mutation_write,
    parse_channel_plan,
    scenario_plants_channels,
)
from plurai.host_effects import INFRA_STOP_CLASSES, STOP_FAILURE_SOURCE, RedactingStream, end_cli_processes_in_loop, redact_cli_credentials
from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, bench_config_roots, discover_config_documents
from plurai.observability.tool_call_recovery import (
    AGENT_STATE_TOOL_TRAIL_KEY,
    TRAIL_ARGS_OMITTED_KEY,
    ThreadCalls,
    _declared_type,
    _names_a_call,
    _parse_content_blocks,
    _tool_calls_from_content_blocks,
    derive_tool_calls_from_messages,
    normalize_message,
    recovered_tool_calls,
)
from plurai.scan_contract.input_channels import ChannelKind

# The world-owner vocabulary, from the module that OWNS it. Imported rather than re-spelled here:
# the parent's run-start gates and this child answer the same two questions off the same two local
# files, and a second copy of either name is how one of them comes to act on a word the other no
# longer uses. Free at import time — ``import plurai`` above already loads this module.
from plurai.orchestrator.orchestrator import WORLD_OWNER_MOCK, WORLD_SNAPSHOT_HOOK, _declared_native_tools, _declared_world_owner
from plurai.run_artifacts import _NON_AGENT_TYPES, LC_MESSAGE_TYPES, SEED_REPORT_UNREADABLE_KEY, _text_of, declared_message_type
from plurai.sdk_turn_faults import PROVIDER_TURN_FAULT_KEY, TURN_END_UNOBSERVED_KEY, provider_turn_fault_marker
from plurai.user_surfaces import UserSurfacesConfigError, discover_surface_types


# ---------------------------------------------------------------------------
# simulate.py loading + attribute resolution
# ---------------------------------------------------------------------------


def _load_simulate_module(simulate_path: Path) -> ModuleType:
    """Import the simulate.py at ``simulate_path`` and return the module.

    ``spec_from_file_location`` plus prepending the file's parent dir to
    ``sys.path`` so the simulate.py's relative imports resolve without the
    user adding it to PYTHONPATH.
    """
    simulate_path = Path(simulate_path).resolve()
    parent = str(simulate_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(simulate_path.stem, simulate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {simulate_path}')
    module = importlib.util.module_from_spec(spec)
    # The tools' declared error channels are read from the config beside THIS harness.
    use_harness_config(simulate_path)
    spec.loader.exec_module(module)
    return module


def _optional_hook(module: ModuleType, name: str) -> Callable[..., Any] | None:
    """``module.<name>`` when the harness exports it, ``None`` when it does not; an export that is
    not callable raises ``TypeError``. The one discipline every hook below is resolved by."""
    fn = getattr(module, name, None)
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(f'{getattr(module, "__file__", "<simulate>")} exports `{name}` but it is not callable.')
    return fn


def _resolve_invoke_agent(module: ModuleType) -> Callable[..., Any] | None:
    """Auto-discover the optional ``invoke_agent`` callable for dialog scenarios.

    Returns None when absent — the caller (`_ChildSession.__init__`) enforces
    that AT LEAST ONE of ``invoke_agent`` / ``invoke_pipeline`` exists. Both
    may co-exist when a project mixes dialog + one-shot scenarios.
    """
    return _optional_hook(module, 'invoke_agent')


def _resolve_invoke_pipeline(module: ModuleType) -> Callable[..., Any] | None:
    """Auto-discover the optional ``invoke_pipeline`` callable for one-shot scenarios.

    Signature: ``invoke_pipeline(inputs: dict) -> dict`` (sync or async).
    Returns None when absent. Either ``invoke_agent`` or ``invoke_pipeline``
    must exist; otherwise ``_ChildSession.__init__`` raises.
    """
    return _optional_hook(module, 'invoke_pipeline')


def _resolve_seed_state(module: ModuleType) -> Callable[..., Any] | None:
    """Auto-discover an optional ``seed_state`` callable on the module.

    Under the subprocess model there is no ``Orchestrator(seed_state=...)``
    kwarg; the child looks the attribute up directly. The verifier
    (verify_simulate.py) is responsible for catching shape mistakes at
    static-check time.
    """
    return _optional_hook(module, 'seed_state')


def _resolve_reset(module: ModuleType) -> Callable[..., Any] | None:
    """Auto-discover an optional ``reset`` callable on the module — the hook that CLEARS.

    Signature: ``reset() -> None`` (sync or async). Sibling of :func:`_resolve_seed_state` and
    resolved the same way, by ``getattr`` on the imported module, so the two hooks a harness
    declares are discovered by one discipline.

    The two hooks are opposites and both are needed. ``seed_state`` puts this scenario's
    preconditions IN; nothing puts the previous scenario's writes OUT. Everything the CHILD
    PROCESS owns is already isolated — a module-level store dies with the per-scenario child, and
    the mock's world is keyed on the session the backend opens per scenario — so what survives is
    exactly the state the harness keeps somewhere else: a file, a real database, a CRM, a
    server-side conversation store. This hook is the only thing that can reach those, because
    only the harness knows where they are.
    """
    return _optional_hook(module, 'reset')


def _resolve_world_snapshot(module: ModuleType) -> Callable[..., Any] | None:
    """Auto-discover an optional ``world_snapshot`` callable — the hook that OBSERVES.

    Signature: ``world_snapshot() -> dict[str, Any]``, returning the ``{entity_id: {...}}`` map
    this scenario's ``final_world_state.state`` should carry. Resolved the same way as the four
    hooks above — ``getattr`` on the imported module, with the same not-callable ``TypeError`` —
    so every hook a harness exports is discovered by one discipline and a name bound by an
    assignment or a ``from … import`` counts exactly as a ``def`` does.

    The SIBLING of :func:`_resolve_reset`, and the other half of what a non-mock world owner
    implies. ``reset`` puts the previous scenario's writes OUT of a store this process does not
    own; this one gets the current scenario's writes OUT OF IT and onto the wire. Neither is
    reachable any other way, for the same reason: only the harness knows where that store is.

    Optional, and silent when absent: a bench whose world really is the mock's needs nothing here,
    because the child snapshots the session the backend opened. What the absence MEANS is the one
    thing the owner declaration decides — see :meth:`_ChildSession.handle_finalize`.
    """
    return _optional_hook(module, WORLD_SNAPSHOT_HOOK)


#: The optional harness export that applies ONE act of the simulated user: ``apply_user_action(act: dict)
#: -> None`` (sync or async), ``act`` being the whole ``UserActionWrite`` (``turn_idx``, ``surface``,
#: ``action``, ``entity_id``, ``updates``, ``inline_row``). Required for an act on a row inside a list;
#: without it see :data:`_USER_ACTION_ROUTES`.
USER_ACTION_HOOK = 'apply_user_action'


#: How an act is written when the harness exports no :data:`USER_ACTION_HOOK`, by ``(row shape, the mock
#: owns the world)``: the mock's world is written directly for an entity row; anything else is the
#: reason it is ``unexecutable``. A row inside a list field is NEVER written generically: the product
#: versions such lists (a new revision per edit), which only the harness knows how to do.
_MOCK_WRITE = 'mock'
_INLINE_NEEDS_HOOK = f'the act edits a row inside a list, which only the harness can write; export `{USER_ACTION_HOOK}(act)` from simulate.py'
_USER_ACTION_ROUTES: dict[tuple[str, bool], str] = {
    ('entity', True): _MOCK_WRITE,
    ('entity', False): f'this harness keeps its world outside the mock and exports no `{USER_ACTION_HOOK}(act)`; export it from simulate.py',
    ('inline', True): _INLINE_NEEDS_HOOK,
    ('inline', False): _INLINE_NEEDS_HOOK,
}


def _resolve_user_action_hook(module: ModuleType) -> Callable[..., Any] | None:
    """Auto-discover the optional :data:`USER_ACTION_HOOK`, by the discipline every hook above uses."""
    return _optional_hook(module, USER_ACTION_HOOK)


def _accepts_reset_contract_arity(fn: Callable[..., Any]) -> bool:
    """Whether ``fn`` can be CALLED with no arguments, the way the reset contract calls it.

    The same narrow answer :func:`_accepts_seed_contract_arity` gives, and narrow for the same
    reason: ``False`` only when the signature is readable AND provably demands an argument. A
    callable whose signature cannot be introspected reads as acceptable — refusing a hook that
    would have worked costs a whole run, while calling one that turns out to need an argument
    costs a TypeError that is caught and named.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind()
    except TypeError:
        return False
    return True


def _accepts_seed_contract_arity(fn: Callable[..., Any]) -> bool:
    """Whether ``fn`` can be CALLED with the three positional arguments the seed contract passes.

    A narrow answer on purpose: ``False`` only when the signature is readable AND provably too
    narrow. A callable whose signature cannot be introspected — a C function, an exotic wrapper —
    reads as acceptable, because the cost of the two mistakes is not symmetric. Calling a hook
    that turns out to be two-parameter costs a TypeError that is caught and named; refusing a hook
    that would have worked costs a whole run for a shape this cannot read.

    Bound with three ``None`` placeholders rather than matched on a parameter count, so
    ``*args``, keyword-only extras, defaults and a callable object's ``__call__`` all answer for
    themselves instead of through a rule that has to enumerate them.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(None, None, None)
    except TypeError:
        return False
    return True


def _content_of(msg: Any) -> Any:
    """One message's ``content``, whichever form the message takes.

    Written once because this file reads it in four places, and hand-spellings drift apart
    (``getattr(msg, 'content', msg)`` vs ``getattr(msg, 'content', None)``). A bare value
    that is not a message at all IS its own content — that is what makes ``_extract_reply``'s
    string result work.
    """
    return msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', msg)


def _message_content_text(msg: Any) -> tuple[str, bool]:
    """``(a message's content as text, was the shape READABLE)``.

    Reads structure through :func:`~plurai.run_artifacts._text_of` — the reader one module
    away that already understands Anthropic content blocks — rather than ``str(content)``,
    which stringifies a block list into a Python repr and hands ``"[{'type': 'text', …}]"``
    to the simulated user and to every judge as the agent's own words.

    The second value is the readability half: ``_text_of`` returns ``None`` for a shape it does not
    understand, and that is NOT the same as an empty message. The repr fallback is kept so no
    turn that would produce text ends up with none, but the caller is told the text is a
    guess and labels the reply ``unknown`` rather than ``assistant``.

    An OBJECT message goes through the same reader, via its ``content``. ``_text_of`` only walks
    dicts, and the LIVE shape here is the object — this function is called on the agent's raw
    result.
    """
    content = _content_of(msg)
    if isinstance(content, str):
        return content, True
    text = _text_of({'content': content})
    if text:
        return text, True
    # ``_text_of`` answers '' for BOTH "genuinely no prose" and "a shape I model as empty", and
    # this caller cannot treat those as one: an empty text is SKIPPED, and a suffix where every
    # message is skipped RAISES — so collapsing them turns a shape that produces a (poor)
    # reply into a turn-killing AgentError.
    #
    # A LIST content is the one shape where '' is a real answer: a block list carrying no text
    # block is a tool-calling turn with no prose, and ``run_artifacts.load_transcript`` already
    # drops exactly those from ``agent_texts`` rather than putting a repr of a tool call in the
    # list a judge reads. Everything else that yields no prose — a mapping, a bare number — falls
    # back to the repr, UNREADABLE, and the label says so. ``None`` is excluded: 'None' is not text anybody wrote.
    if text is None or (content is not None and not isinstance(content, list)):
        return str(content), False
    return '', True


def _carries_tool_result(msg: Any) -> bool:
    """Does this message carry an Anthropic ``tool_result`` block — whatever its envelope role?

    An Anthropic-native chain puts a tool's output inside the NEXT ``user``-role message, so a
    role read alone calls that message a user turn: the turn's suffix comes out empty and the
    turn RAISES for a chain that plainly ended on a tool result. Read STRUCTURALLY, so the same
    situation gets the same answer whichever provider spelled it.
    """
    content = _content_of(msg)
    return isinstance(content, list) and any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in content)


def _is_transport_envelope(msg: Any) -> bool:
    """A ``user``-role message that is PURELY a tool-result carrier — no words of the user's own.

    The predicate :func:`_turn_suffix` uses for the turn boundary, and it is deliberately
    narrower than :func:`_carries_tool_result`: Anthropic permits a user message to carry a
    ``tool_result`` block AND the user's next question together, and skipping THAT as a boundary
    would reach back past a genuine user turn and hand the PREVIOUS turn's answer over as this
    one's — the stale reply ``_turn_suffix`` exists to prevent.
    """
    # Through ``_message_content_text`` rather than a parallel ``_text_of`` call, so "this envelope
    # has no words of its own" is decided by the same reader that later extracts them — the two
    # cannot disagree about one message.
    return _carries_tool_result(msg) and not _message_content_text(msg)[0].strip()


def _tool_result_text(msg: Any) -> str:
    """The text a message's Anthropic ``tool_result`` blocks carry.

    ``_text_of`` deliberately keeps only ``text`` blocks, so a message holding nothing but a
    ``tool_result`` reads as empty — correct for "what did the agent SAY", and wrong here, where
    the whole question is what the mid-tool-chain fallback would hand over. A block's own
    ``content`` is a string or a nested block list, and both spellings are read.
    """
    content = _content_of(msg)
    if not isinstance(content, list):
        return ''
    parts: list[str] = []
    for blk in content:
        if not isinstance(blk, dict) or blk.get('type') != 'tool_result':
            continue
        inner = blk.get('content')
        # Nested blocks go through ``_text_of``, the reader this module already imports. A
        # hand-rolled walk here would take ANY block carrying a ``text`` key, where ``_text_of``
        # requires ``type == 'text'`` — a silently wider match, in the one place the two are
        # meant to render the same bytes the same way.
        parts.append(inner if isinstance(inner, str) else (_text_of({'content': inner}) or ''))
    return ''.join(parts)


def _terminal_tool_text(suffix: Sequence[Any]) -> str:
    """The text of the tool results the turn's suffix ENDS on, in order — ``''`` when it ends elsewhere.

    Walks back from the last message, skipping ``system`` notes, and stops at the first message
    that is not a tool result — the calls-only agent message that issued them. So only the
    TRAILING run is read: a result the model consumed earlier in the turn, and a turn whose last
    message is an agent CALL (``[ai(call A), tool(A), ai(call B)]``), relay nothing.
    """
    tail: list[str] = []
    for m in reversed(suffix):
        if declared_message_type(m) == 'system':
            continue
        if not (declared_message_type(m) in _NON_AGENT_TYPES or _carries_tool_result(m)):
            break
        tail.append(_message_content_text(m)[0].strip() or _tool_result_text(m).strip())
    return '\n\n'.join(t for t in reversed(tail) if t)


#: The stop reasons by which a model says it ENDED its step rather than paused for a tool: OpenAI's and Gemini's
#: ``finish_reason`` (``stop`` / ``STOP``) and Anthropic's ``stop_reason`` (``end_turn``). Compared lower-cased.
_END_OF_STEP_REASONS: Final = frozenset({'stop', 'end_turn'})


def _ended_its_step(msg: Any) -> bool:
    """Does this message carry an explicit end-of-step stop reason — on its ``response_metadata`` or on itself?

    Absent is ``False``: a message that does not say its model call ended is not read as one that did.
    Read once per message, into :attr:`_SuffixMessage.ended_step`.
    """

    def _field(holder: Any, key: str) -> Any:
        return holder.get(key) if isinstance(holder, Mapping) else getattr(holder, key, None)

    for holder in (_field(msg, 'response_metadata'), msg):
        for key in ('finish_reason', 'stop_reason'):
            value = _field(holder, key)
            if isinstance(value, str) and value.strip().lower() in _END_OF_STEP_REASONS:
                return True
    return False


@dataclass(frozen=True, slots=True)
class _SuffixMessage:
    """One message of a turn's suffix, read once: the facts :func:`_extract_reply`'s final-speech walk turns on.

    * ``index`` — its position in the suffix.
    * ``speech`` — ``(text, readable)`` when it is agent words, else ``None``.
    * ``is_tool`` — a tool result, a ``tool_result`` envelope or a system note: never speech.
    * ``names_a_call`` — it issues a tool call, in any spelling :func:`_names_a_call` reads.
    * ``ended_step`` — its model call carried an explicit end-of-step stop reason (:func:`_ended_its_step`).
    * ``answer_at`` — on an ``is_tool`` message: the index of the DELIVERED answer its round-trip
      follows, else ``None``. Set by :func:`_read_suffix` when it reads the wordless calls-only
      message that opened the round-trip.
    """

    index: int
    speech: tuple[str, bool] | None
    is_tool: bool
    names_a_call: bool
    ended_step: bool
    answer_at: int | None = None

    @property
    def delivers_an_answer(self) -> bool:
        """Readable words that call nothing, from a model call that ended its step."""
        return self.speech is not None and self.speech[1] and not self.names_a_call and self.ended_step


def _read_suffix(suffix: Sequence[Any]) -> list[_SuffixMessage]:
    """Each message of ``suffix`` as a :class:`_SuffixMessage`, in order.

    A wordless message that names a call opens a round-trip. When the message before it (system
    notes skipped) delivers an answer, that answer's index is carried as ``answer_at`` on every
    ``is_tool`` message up to the next agent message.
    """
    out: list[_SuffixMessage] = []
    answer_at: int | None = None
    previous: _SuffixMessage | None = None
    for index, msg in enumerate(suffix):
        if declared_message_type(msg) in _NON_AGENT_TYPES or _carries_tool_result(msg):
            read = _SuffixMessage(index, None, True, False, False, answer_at)
        else:
            text, readable = _message_content_text(msg)
            speech = (text, readable) if text.strip() else None
            names_a_call = _names_a_call(normalize_message(msg))
            read = _SuffixMessage(index, speech, False, names_a_call, _ended_its_step(msg))
            opens_a_round_trip = speech is None and names_a_call
            answer_at = previous.index if opens_a_round_trip and previous is not None and previous.delivers_an_answer else None
        out.append(read)
        if declared_message_type(msg) != 'system':
            previous = read
    return out


class NoAssistantReplyError(RuntimeError):
    """The agent produced no reply THIS turn — messages after the last user turn carry no text."""


class UndeclaredTurnInputError(RuntimeError):
    """This turn's input is not something the harness's ``agent.io`` declares it can be given.

    Raised while BUILDING the payload, before the agent is called, and carried out as a typed
    ``AgentError`` subkind rather than a raise: the turn is undeliverable, but the rest of the
    dialog is still measurable, and the customer's remedy is a line in ``plurai.config.yaml``.

    The message names the DECLARATION and the kind observed — never the part's payload. A
    ``file`` part carries a base64 document; putting it in an error message would ship the
    customer's document to the run log, the transcript and every reader of both.
    """

    def __init__(self, subkind: AdapterErrorSubkind, message: str) -> None:
        super().__init__(message)
        self.subkind: AdapterErrorSubkind = subkind


def _turn_suffix(messages: list[Any]) -> list[Any]:
    """The messages produced AFTER the last genuine user turn — THIS turn's output.

    "Genuine" is the structural part: a ``user``-role message whose content is only a
    ``tool_result`` block is a transport envelope, not the user speaking, and treating it as the
    boundary hides the whole suffix on every Anthropic-native chain.
    """
    last_user = max(
        (i for i, m in enumerate(messages) if declared_message_type(m) in ('human', 'user') and not _is_transport_envelope(m)),
        default=-1,
    )
    return messages[last_user + 1 :]


@dataclass(frozen=True, slots=True)
class TurnReply:
    """THIS turn's agent output, typed: what it SAID, what it DID, and where the words came from.

    A turn is not a single string projected several different ways, where no two consumers
    would agree. The string is here (``speech``) because a conversation is made of words, but
    it is not the whole turn: an agent whose output is a typed action list legitimately acts
    WITHOUT speaking, and a projection with nowhere to put that would drop the act, its trail
    and its label together and report the turn as the agent going silent.

    THE RULE this type exists to state: **a turn is legitimate when EITHER ``speech`` or
    ``actions`` is non-empty.** Only when both are empty has the agent produced nothing — and
    that is :class:`NoAssistantReplyError`, enforced in ``__post_init__`` so the type cannot be
    used to state the opposite of its own rule.
    """

    speech: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    reply_source: str = 'unknown'

    def __post_init__(self) -> None:
        """Refuse a turn carrying neither speech nor actions — the rule above, enforced.

        ``speech`` and ``actions`` are settled by two DIFFERENT passes over one suffix (the
        reversed prose walk and :func:`turn_actions`), so they can disagree: a suffix that ends
        on a tool result whose call the recovery could not name yields ``TurnReply('', [],
        'tool_result')`` — a turn that ships as complete carrying nothing, with no raise to say
        the agent delivered nothing. ``interrupt`` is the one label for which both-empty is a
        real turn: the parent's interrupt path owns it and the prompt rides ``interrupt_context``.
        """
        if not self.speech and not self.actions and self.reply_source != 'interrupt':
            raise NoAssistantReplyError(
                f'a {self.reply_source!r} turn was built with neither speech nor actions — the two readers of this '
                'turn disagreed. Shipping it would report a complete turn that carries nothing, which every judge '
                'downstream reads as the agent going silent.'
            )


def turn_actions(messages: Sequence[Any], *, open_calls: ThreadCalls) -> list[dict[str, Any]]:
    """What the agent DID over ``messages`` — ``[{name, args, status}]``, in execution order.

    Deliberately NOT a second reader of ``tool_calls``. It projects
    :func:`~plurai.observability.tool_call_recovery.derive_tool_calls_from_messages`, which
    already reads every spelling this SDK has met — the LangChain attribute, OpenAI's nested
    ``{'type': 'function', 'function': {'name', 'arguments'}}``, Anthropic ``tool_use`` blocks —
    and already pairs each call with its result and settles the status. The WIRE projection
    keeps only calls carrying a top-level ``name``, which would send a call-bearing turn to the
    judges with ``tool_calls: []``: two readers of one field, one of them right. This is the
    right one, projected.

    ``status`` is the recovery's own settlement — ``'unknown'`` for a call whose result was
    never observed (a render verb the harness dispatched itself, a graph interrupted mid-chain),
    ``'ok'`` / ``'error'`` where one was.

    ``open_calls`` is the THREAD's ledger (:meth:`_ChildSession.handle_user_turn` keeps it): a
    result answering an earlier turn's call settles that call and is no act of THIS turn. An
    ``_unpaired`` record (args never seen) is skipped, as the orchestrator's per-turn reader skips
    it: shipped, it is an act named from a result with ``args: {}`` that no one made.
    """
    return [
        _action_record(call.get('tool'), call.get('args'), call.get('status'))[0]
        for call in derive_tool_calls_from_messages(list(messages or []), open_calls=open_calls)
        if call.get('tool') and not call.get('_unpaired')
    ]


def _extract_reply(result: Any, *, interrupted: bool = False, declared_actions: Sequence[dict[str, Any]] = (), open_calls: ThreadCalls) -> TurnReply:
    """THIS TURN's agent output as a :class:`TurnReply` — speech, actions, and the source label.

    Only the suffix AFTER the last human/user message is considered — that suffix IS the
    current turn's output. A blind ``messages[-1]`` would ship the USER's own words as the
    reply when the state ends on a HumanMessage, and a whole-list backward walk would
    substitute the PREVIOUS turn's answer on checkpointed graphs ([Human q1, AI a1, Human q2]
    → 'a1') — a plausible-looking stale reply, silently scored as this turn's.

    Within the suffix: the last assistant/roleless message with non-empty text is the SPEECH.
    Everything the suffix shows the agent doing is the ACTIONS (:func:`turn_actions`). It
    raises :class:`NoAssistantReplyError` only when BOTH are empty, so a ``content: '' +
    tool_calls`` message (a render-only turn) is read as the act it is. With
    ``interrupted=True`` (a LangGraph ``interrupt()`` — the parent's interrupt path owns the
    turn, the prompt text rides interrupt_context) an empty turn is a legitimate ``''``.

    The LABEL says where ``speech`` came from, and it is decided STRUCTURALLY, never by
    framework: a unit is a tool result when its normalized type is non-agent
    (:data:`~plurai.run_artifacts._NON_AGENT_TYPES`, which covers OpenAI's legacy
    ``role: function`` as well as ``tool``) or when it carries an Anthropic ``tool_result``
    block whatever its envelope role. A tool result is an ACTION, and it never outranks speech:
    a turn with agent words anywhere in its final block reports those words, whatever tool noise
    trails them. A turn with NO agent words that ENDS on a run of tool results after a calls-only
    agent message, and was not paused by a typed interrupt, is a graph that returned on that
    result — a tool that ends the run (``Command(goto=END)``, ``return_direct``, a ``tools → END``
    edge) — and the text of that trailing run is what the agent's caller is handed, so it is the
    turn's ``speech``, still labelled ``tool_result`` so the server and the run summary can tell
    it from the agent's own words. Relaying ``''`` there would show the simulated user a blank
    turn where the product shows its user a question. A PAUSED graph (``interrupted=True``) keeps ``''`` — its prompt rides
    ``interrupt_context``. ``interrupt`` needs the TYPED signal — an untyped pause
    (``{'status': 'awaiting_approval'}``) is ``unknown``, never ``assistant``.
    """
    if isinstance(result, str):
        return TurnReply(result, [], 'assistant')
    if isinstance(result, dict):
        messages = result.get('messages') or []
        suffix = _turn_suffix(messages)
        # ``declared_actions`` is the envelope's ``agent_actions``, already read by the caller. It
        # counts BEFORE the emptiness ladder below: an interrupt or render declared on the envelope
        # with no speech is a real turn, not a silence.
        actions = [*turn_actions(suffix, open_calls=open_calls), *declared_actions]

        # THE FINAL SPEECH BLOCK: walk back from the LAST texted assistant message and keep
        # taking texted assistant messages until a tool result interrupts. Two boundaries:
        #
        # * a TOOL ROUND-TRIP resets it — an agent that says "let me look that up", calls a tool
        #   and then answers reports the answer alone, and folding its pre-call
        #   preamble in would change what every judge grades for every tool-calling agent;
        # * …unless the speech before the round-trip CALLED NOTHING, its model call ENDED THE STEP
        #   (an explicit end-of-step stop reason, ``ended_step``), and the round-trip was issued by
        #   a separate, wordless calls-only message (the round-trip then carries that answer's index,
        #   ``answer_at``, from :func:`_read_suffix`). That speech is a delivered answer,
        #   not a preamble, and a framework that re-enters the loop after it (a middleware step
        #   that updates a task list, then signs off) must not erase it. Without the stop reason the
        #   reset stands: a harness that splits ONE model response into a text message and a call
        #   message produces the same sequence for a plain preamble;
        # * TRAILING tool noise after the reply does not end the block — walking from the last
        #   SPEECH rather than the last message is what keeps a late ToolMessage from erasing a
        #   real answer above it.
        #
        # What is left is the assistant messages of that block, joined in order (tool results and
        # wordless calls inside it carry no speech), so prose the agent produced across several
        # messages reaches ``agent_response`` whole. Identity on one message:
        # ``'\n'.join([x]) == x``, so a turn with a single texted reply is byte-identical.
        read = _read_suffix(suffix)
        last_speech = max((m.index for m in read if m.speech is not None), default=-1)
        start = 0
        i = last_speech
        while i >= 0:
            if read[i].is_tool:
                answer_at = read[i].answer_at
                if answer_at is None:
                    start = i + 1
                    break
                i = answer_at
            i -= 1
        spoken: list[str] = []
        readable_all = True
        for m in read[start : last_speech + 1]:
            if m.speech is None:
                continue
            spoken.append(m.speech[0])
            readable_all = readable_all and m.speech[1]
        if spoken:
            return TurnReply('\n'.join(spoken), actions, 'assistant' if readable_all else 'unknown')
        # No speech anywhere in the suffix. Did the turn nonetheless END on a tool result? Only
        # what follows the last speech can decide that, which with no speech is the whole suffix —
        # the same set a reversed walk reaches before it finds anything to say.
        ended_on_tool_result = any(
            (declared_message_type(m) in _NON_AGENT_TYPES or _carries_tool_result(m)) and (_message_content_text(m)[0].strip() or _tool_result_text(m).strip())
            for m in suffix[last_speech + 1 :]
        )
        if ended_on_tool_result:
            # The trailing tool-result run is what this graph handed its caller (see the
            # docstring); a typed interrupt is a pause, not a return, and relays nothing.
            return TurnReply('' if interrupted else _terminal_tool_text(suffix), actions, 'tool_result')
        if actions:
            # The agent ACTED and said nothing. That
            # is a real, complete turn — ``assistant`` is whose turn it was — and the emptiness
            # of ``speech`` is a fact about this turn, not a failure to read it.
            return TurnReply('', actions, 'assistant')
        # A DECLARED chain — even an empty one — is this branch's, not the chainless one's below.
        # ``messages: []`` says the harness read the turn and found nothing in it; sending it on
        # to the repr branch would ship ``"{'messages': [], …}"`` as the agent's reply whenever the
        # envelope also carried a top-level string (a ``title``), which defeats that branch's
        # no-text guard.
        if 'messages' in result:
            if interrupted:
                return TurnReply('', actions, 'interrupt')
            raise NoAssistantReplyError(
                'the agent returned no reply this turn — no message after the last user turn carries text, '
                'and no message names a tool call or render action either. Returning earlier or user-authored '
                'text as the reply would silently mis-attribute it to the agent.'
            )
        # A dict result with NO ``messages`` key at all — an empty declared chain is handled above.
        # An untyped pause lands here, and so does any state shape this SDK does not model —
        # ``str(result)`` is the text, and the label refuses to call a Python repr
        # the assistant's words.
        #
        # Keeping the repr is only right while there IS text to keep. An envelope emptily filled
        # — every channel present and every one of them empty — has none, and ``str(result)``
        # then ships braces and quotes as the agent's reply: the simulated user READS it, answers
        # it ("that output didn't answer my question"), and spends a beat of the plan on a string
        # nobody wrote. A turn with no words in it is the agent
        # delivering nothing, which this function already has one answer for — so it raises like
        # any other empty turn, and the parent's ``_scoreable_empty_turn`` degrades that to the
        # ``''`` / ``empty`` reply the run records as unmeasured. Read at the dict's OWN level and
        # never deeper: a nested value is a shape this SDK does not model either, and stringifying
        # it is the same repr defect one indent down. ``interrupted`` is excluded because the
        # parent's interrupt path owns that turn and its prompt rides ``interrupt_context``.
        if not interrupted and not any(isinstance(value, str) and value.strip() for value in result.values()):
            raise NoAssistantReplyError(
                'the agent returned a state dict with no message chain and no text of its own — nothing in it is '
                f'a reply that could be attributed to the agent (keys: {sorted(map(str, result))}). Returning its '
                'Python repr would hand the simulated user and every judge a string the agent never said.'
            )
        return TurnReply(str(result), [], 'unknown')
    if hasattr(result, 'content'):
        # THROUGH THE ONE READER, not ``str(content)``. An ``invoke_agent`` that returns an
        # ``AIMessage`` directly rather than a state dict is a shape this child models, and
        # an ordinary shape, and ``str(content)`` on a PARTS-LIST content would hand the simulated
        # user and every judge ``"[{'type': 'text', 'text': 'Your balance is $42.'}]"`` as the
        # agent's reply. ``_message_content_text`` is the SAME reader every other message goes
        # through here, and it already carries the readable/unreadable half: a ``str`` content is
        # the identity (``assistant``), a parts list is projected, and a shape it cannot model
        # still falls back to the repr labelled ``unknown``.
        text, readable = _message_content_text(result)
        return TurnReply(text, [], 'assistant' if readable else 'unknown')
    return TurnReply(str(result), [], 'unknown')


def _extract_ai_text(result: Any, *, interrupted: bool = False) -> str:
    """:func:`_extract_reply`'s speech half."""
    return _extract_reply(result, interrupted=interrupted, open_calls=ThreadCalls()).speech


def _extract_messages_list(result: Any) -> list[Any]:
    """Pull the ``messages`` list (LangChain ``BaseMessage`` objects) out of the result."""
    if isinstance(result, dict):
        return list(result.get('messages') or [])
    msgs = getattr(result, 'messages', None)
    if isinstance(msgs, list):
        return msgs
    return []


_warned_serialize_fallback: set[str] = set()

# One warning per distinct non-dict result type — see _extract_agent_state.
_warned_non_dict_state: set[str] = set()


#: Role spellings an agent-supplied dict may use, mapped onto the LangChain discriminator
#: (:data:`~plurai.run_artifacts.LC_MESSAGE_TYPES`' values). An OpenAI-shaped chain says
#: ``role='assistant'``; a google-genai one says ``role='model'``; a LangChain
#: ``model_dump()`` says ``type='ai'``; a class name says ``AIMessage`` (the ``message``
#: suffix is stripped before the lookup).
_LC_TYPE_ALIASES = {'assistant': 'ai', 'model': 'ai', 'user': 'human', 'function': 'tool'}


def _lc_message_type(msg: dict[str, Any]) -> str | None:
    """The LangChain discriminator for an agent-supplied message dict, or None if it names no role.

    A role this does not recognise resolves to None rather than to a guess: inventing a
    discriminator for an unknown spelling would mis-attribute the message, which is strictly
    worse than leaving it as the agent wrote it.
    """
    declared = declared_message_type(msg)
    resolved = _LC_TYPE_ALIASES.get(declared, declared)
    return resolved if resolved in set(LC_MESSAGE_TYPES.values()) else None


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize a LangChain ``BaseMessage`` (or compatible) to a JSON-safe dict.

    Prefers Pydantic ``model_dump()`` / ``dict()``; falls back to a minimal
    ``{type, content}`` so non-LangChain adapters still transport something.
    Fallback hits log once per type — silent degradation here strips
    ``tool_calls`` / ``additional_kwargs.__interrupt__`` from the wire, which
    in turn makes the parent's interrupt detection blind.

    An agent that returns message DICTS (``{'role': 'assistant', ...}`` — a shape
    ``declared_message_type`` / ``_extract_ai_text`` already support) passes through with its own
    keys intact, gaining only a ``type`` normalized from whichever of ``type`` / ``role`` it
    declared. Without that, a transcript stamped ``transcript_schema: 3`` — which promises
    one message vocabulary — would carry agent messages with no ``type`` key at all, and a
    consumer written against that promise would read an empty conversation. The dict is COPIED:
    it may be the agent's own state, and this is not the place to mutate it.
    """
    if isinstance(msg, dict):
        lc_type = _lc_message_type(msg)
        return {**msg, 'type': lc_type} if lc_type else msg
    for serialiser in ('model_dump', 'dict'):
        method = getattr(msg, serialiser, None)
        if callable(method):
            try:
                out = method()
            except Exception as exc:  # noqa: BLE001 — best-effort serialisation
                cls_name = type(msg).__name__
                if cls_name not in _warned_serialize_fallback:
                    _warned_serialize_fallback.add(cls_name)
                    import logging  # noqa: PLC0415

                    logging.getLogger(__name__).warning(
                        '%s.%s() raised; trying the next serializer / minimal-dict '
                        'fallback (which preserves dict tool_calls + additional_kwargs '
                        'best-effort, dropping other fields): %s',
                        cls_name,
                        serialiser,
                        exc,
                    )
                continue
            if isinstance(out, dict):
                return out
    # model_dump()/dict() both failed. Build the minimal fallback BUT best-effort
    # preserve the fields the parent's tool-call recovery + interrupt detection
    # depend on — otherwise every mode:real tool call goes invisible to the eval
    # (faithfulness/tool-claim judges then misfire).
    #
    # Only keep DICT tool-call entries: the parent's recovery
    # (derive_tool_calls_from_messages) skips any non-dict entry, and the IPC
    # encoder's ``default=str`` would silently stringify a non-dict entry into a
    # value that is then dropped on the far side — i.e. preserving non-dict
    # entries would re-create the very invisibility this fallback exists to
    # prevent, with no warning.
    # ``type`` is normalized to the LangChain discriminator the model_dump() path above emits
    # ('ai' / 'human' / 'tool'), not the class name: the two land in the SAME transcript array,
    # and a reader that meets both concludes one conversation was recorded two ways. Same rule
    # the backend's own readers apply on the way in (``.lower().removesuffix('message')``).
    fallback: dict[str, Any] = {
        'type': type(msg).__name__.lower().removesuffix('message'),
        'content': str(getattr(msg, 'content', msg)),
    }
    tool_calls = getattr(msg, 'tool_calls', None)
    if isinstance(tool_calls, list):
        dict_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
        if dict_calls:
            fallback['tool_calls'] = dict_calls
        if len(dict_calls) != len(tool_calls):
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                '%s: %d non-dict tool_call entrie(s) could not be preserved across IPC and were dropped.',
                type(msg).__name__,
                len(tool_calls) - len(dict_calls),
            )
    additional_kwargs = getattr(msg, 'additional_kwargs', None)
    if isinstance(additional_kwargs, dict) and additional_kwargs:
        fallback['additional_kwargs'] = additional_kwargs
    return fallback


def _normalize_interrupt_value(value: Any) -> dict[str, Any]:
    """Coerce a LangGraph interrupt payload into ``{prompt, choices?, actions?, raw_payload}``.

    LangGraph ``interrupt()`` accepts arbitrary JSON-serialisable values. Common
    shapes seen in the wild: a string prompt, ``{description, options}``,
    ``{prompt, choices}``, or a typed HITL payload from langgraph-prebuilt
    helpers. Normalise so the user-simulator can rely on ``.prompt`` /
    ``.choices`` regardless of the agent's chosen shape; ``raw_payload``
    preserves the original for callers that need it.

    ``approve`` is the choice that RELEASES the paused calls, declared by the harness (it is the
    one that knows which button runs them) and kept only when it is one of ``choices``. Nothing
    reads it off the choice's wording. Without it, the answer to the pause cannot be typed and a
    result arriving after it is undecidable (``tool_call_recovery.settle_paused_calls``).
    """
    if isinstance(value, dict):
        if isinstance(value.get('action_requests'), list):
            return _normalize_hitl_request(value)
        prompt = value.get('prompt') or value.get('description') or value.get('question')
        choices = value.get('choices') or value.get('options')
        out: dict[str, Any] = {'raw_payload': value}
        if prompt is not None:
            out['prompt'] = str(prompt)
        else:
            out['prompt'] = str(value)
        if choices is not None:
            out['choices'] = list(choices) if isinstance(choices, (list, tuple)) else [str(choices)]
        # ``actions: [{name, args}]`` — what the user is asked to approve; absent when not declared.
        if isinstance(value.get('actions'), list):
            out['actions'] = value['actions']
        if (approve := value.get('approve')) is not None:
            if approve in out.get('choices', ()):
                out['approve'] = approve
            else:
                print(
                    f'[plurai.child] the interrupt declares approve={approve!r}, which is not one of its choices '
                    f'{out.get("choices")!r}; ignored, so the answer to this pause cannot be typed as approved or denied.',
                    file=sys.stderr,
                )
        return out
    return {'prompt': str(value), 'raw_payload': value}


#: LangChain's own decision token for "run the paused call as proposed" (``HumanInTheLoopMiddleware``).
_HITL_APPROVE_DECISION: Final = 'approve'


def _normalize_hitl_request(value: dict[str, Any]) -> dict[str, Any]:
    """LangChain v1 ``HumanInTheLoopMiddleware``'s typed interrupt (``{action_requests, review_configs}``).

    The framework types the pause itself, so nothing needs declaring: the choices are the union of
    each review's ``allowed_decisions``, the actions are the requests, and ``approve`` is the
    middleware's native ``approve`` decision whenever it is allowed. ``edit`` / ``respond`` carry an
    edited call or a message a choice string cannot, so they are offered but never read as approval.
    """
    requests = [r for r in value['action_requests'] if isinstance(r, dict)]
    choices: list[str] = []
    for config in value.get('review_configs') or []:
        for decision in (config.get('allowed_decisions') or []) if isinstance(config, dict) else []:
            if isinstance(decision, str) and decision not in choices:
                choices.append(decision)
    descriptions = [str(r['description']) for r in requests if r.get('description')]
    out: dict[str, Any] = {
        'raw_payload': value,
        'prompt': '\n\n'.join(descriptions) or str(value),
        'actions': [{'name': r.get('name'), 'args': r.get('args') or {}} for r in requests],
    }
    if choices:
        out['choices'] = choices
    if _HITL_APPROVE_DECISION in choices:
        out['approve'] = _HITL_APPROVE_DECISION
    return out


def _detect_interrupt(result: Any, messages: list[Any]) -> tuple[bool, dict[str, Any]]:
    """Detect a LangGraph ``interrupt()`` in the agent's result.

    Two signals, in priority order:

    1. ``result['__interrupt__']`` — LangGraph's native marker on
       stream-updates / state-snapshot results. Carries an ``Interrupt`` object
       (with a ``.value`` attribute) or a tuple of them.
    2. ``message.additional_kwargs['__interrupt__']`` on the trailing AIMessage
       — customer adapters that don't surface interrupts at the top level
       sometimes attach the metadata to the last message instead.

    Returns ``(False, {})`` when no signal is found. Detection is best-effort:
    scenario-level pattern matching (``SimpleUserProfile.interrupt_responses``)
    is the recommended fallback for agents that emit freeform interrupt prompts
    without LangGraph metadata.
    """
    if isinstance(result, dict):
        raw = result.get('__interrupt__')
        if raw:
            interrupt_obj = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
            value = getattr(interrupt_obj, 'value', interrupt_obj)
            return True, _normalize_interrupt_value(value)

    for msg in reversed(messages):
        kwargs = getattr(msg, 'additional_kwargs', None)
        if isinstance(kwargs, dict) and kwargs.get('__interrupt__'):
            return True, _normalize_interrupt_value(kwargs['__interrupt__'])

    return False, {}


def _json_default(obj: Any) -> Any:
    """``json`` ``default`` for non-native values: STRUCTURE-serialise via ``model_dump()`` / ``dict()``
    (pydantic models, LangChain objects, dataclass-likes) before falling back to ``str()``. So a
    structured state value — e.g. a list of pydantic ``last_suggestions`` — crosses the wire as a
    dict the eval can index, not a flattened repr string. Mirrors ``_serialize_message``'s
    structured-first strategy rather than the blunt ``default=str`` that silently mangles such values."""
    for attr in ('model_dump', 'dict'):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001 — try the next strategy, then str()
                pass
    return str(obj)


#: The ``source`` labels the action channel carries (``product``: the product's own code made the act). The wire
#: field stays an open string; this side sends only a label it knows.
AGENT_ACTION_SOURCES: Final = frozenset({'product'})


def _action_record(name: Any, raw_args: Any, status: Any, source: Any = None) -> tuple[dict[str, Any], bool]:
    """ONE act in the shape this channel carries — ``{name, args, status}``, plus the harness's ``source``
    label when it declared a known one (:data:`AGENT_ACTION_SOURCES`, case and surrounding whitespace ignored) —
    and whether part of it (its ARGUMENTS, or a ``source`` label this channel does not know) had to be left behind.

    Both producers of the channel project through here: :func:`turn_actions` (what the message
    CHAIN shows) and :func:`_envelope_actions` (what the harness DECLARED), so the rule below
    holds for both.

    THE RULE — ``args`` rides only if THIS BOUNDARY CAN CARRY IT, and the boundary is ASKED
    rather than described, in the two steps the boundary itself would take. FIRST the
    ``json.loads(json.dumps(v, default=_json_default))`` round trip :func:`_extract_agent_state`
    puts every state value through, one screen down: STRUCTURE-first, so a pydantic model rides
    as a dict an eval can index rather than as a flattened repr — the RICHER of the two
    renderings. Only if that RAISES, the plain ``default=str`` the IPC itself ships with
    (:func:`plurai._protocol.encode`), so this check can never declare uncarryable a value the
    boundary would have delivered. A value that survives either step and lands as a mapping is
    carryable by construction — ``json.loads`` yields STRING keys, which is exactly what the
    wire's ``AgentAction.args`` (``dict[str, Any] | None``, ``@generated``) requires — so the
    constraint is satisfied STRUCTURALLY and there is no second, hand-written copy of it here to
    drift from the model it must agree with.

    Why not a type test. ``args`` passing through with only a falsy-to-``{}`` fallback would admit
    a verb-with-one-argument written the natural way (``{'name': 'open_sluice', 'args': 'gate-3'}``),
    count it as read, and then take the WHOLE ``/user-turn`` POST down one process later on a turn
    the agent really performed. ``isinstance(args, dict)`` closes only part of
    that: a TUPLE-KEYED mapping is a mapping, and JSON has no spelling for that key at all — it
    raises inside this child's own ``encode`` (:func:`plurai._protocol.encode`, the write that
    delivers the turn), one hop EARLIER than the wire and with no request to blame. Asking the
    boundary is also what keeps the rule from over-reaching in the other direction: a non-string
    KEY or a ``datetime`` value looks fatal against ``AgentAction`` alone, yet crosses
    because that same encode settles it — so the round trip RECOVERS the value
    (``{3: 'gate'}`` → ``{'3': 'gate'}``) instead of discarding it.

    WHAT THIS CLOSES AND WHAT IT LEAVES OPEN, because the paragraph above describes the hazard
    and this channel is only one of the two that carry it:

    * the STRING-``args`` case is closed end to end — the act ships with empty args and the
      warning names it, on both producers;
    * the ENVELOPE copy is safe: the raw declaration kept on ``agent_state`` goes through
      :func:`_extract_agent_state`, which drops an unserializable key with a named warning;
    * a NON-JSON-ABLE KEY still kills the child on the MESSAGE-CHAIN path.
      :func:`_serialize_message` returns the LangChain ``model_dump()`` verbatim, so the same
      ``tool_calls[i]['args']`` rides ``AgentResponse.full_messages`` into ``encode``, where
      ``default=`` is never consulted for KEYS.

    THE DISPOSITION is the one
    :func:`~plurai.observability.tool_call_recovery._normalize_openai_nested` already takes for
    the single spelling of this problem it covers: keep the ACT, empty its ARGS, say so. The act
    is the half that says the turn happened at all, and the half the server dispatches the
    reserved ``end_conversation`` / ``handoff`` verbs on by NAME alone — so dropping the entry
    would silently stop ending dialogs, trading a loud fatal for a quiet wrong answer.
    ``{'value': 'gate-3'}`` is not an option either: it invents an argument name the harness
    never wrote.

    The second return value says that happened, so the caller can keep the raw declaration where
    it was: a read that took the NAME but not the ARGUMENTS did not take the value WHOLE.
    """
    args: dict[str, Any] = {}
    unreadable = False
    # A FALSY ``args`` is not an unreadable one: ``None`` / ``{}`` / ``''`` become ``{}`` with no
    # warning.
    if raw_args:
        # The boundary's OWN two steps, in its order — structure-first, then the ``default=str``
        # ``encode`` ships with. Structure-first ALONE would lose a value the IPC delivers
        # (a model whose ``model_dump()`` has tuple keys: it raises here, where ``str`` renders it).
        try:
            carried = json.loads(json.dumps(raw_args, default=_json_default))
        except Exception:  # noqa: BLE001 — the richer rendering is not always a possible one; ask the one the IPC itself would
            try:
                carried = json.loads(json.dumps(raw_args, default=str))
            except Exception:  # noqa: BLE001 — "can this boundary carry it" is the question; only a no from BOTH is a no
                carried = None
        if isinstance(carried, dict):
            args = carried
        else:
            unreadable = True
            import logging  # noqa: PLC0415

            # ``[plurai.child]`` is LOAD-BEARING: the child installs no handler, so the parent
            # promotes this line only via ``plurai._subprocess._STDERR_WARN_RE``. See
            # :func:`_note_dropped_declaration`.
            logging.getLogger(__name__).warning(
                '[plurai.child] action %r declared arguments this channel cannot carry (%s); the act is reported with its NAME '
                'and EMPTY args, so an argument assertion over it grades against arguments nobody saw. Emit `args` as a '
                'JSON object.',
                name,
                type(raw_args).__name__,
            )
    record: dict[str, Any] = {'name': str(name), 'args': args, 'status': str(status or 'unknown')}
    label = source.strip().lower() if isinstance(source, str) else source
    if isinstance(label, str) and label in AGENT_ACTION_SOURCES:
        record['source'] = label
    elif label is not None and label != '':
        # Not read whole: the act ships without the label, and the raw declaration stays on ``agent_state``.
        unreadable = True
        _note_dropped_declaration(1, 1, read_any=True, unknown_source=source)
    return record, unreadable


def _envelope_actions(result: Any) -> tuple[list[dict[str, Any]], bool, int]:
    """What the harness DECLARED the agent did, read off the result envelope's ``agent_actions``
    — whether that read CONSUMED the key, and how much of the declaration it could NOT take.

    The second spelling of one channel. :func:`turn_actions` reads what the MESSAGE CHAIN shows —
    which is every agent that expresses an act as a function call, the OpenAI-standard shape and
    the one that needs no plumbing — but an agent that tracks its own acts in state, or a harness
    that assembles the turn itself, says so on the envelope instead. Without this read that key
    would fall through to ``agent_state``: carried over the wire, but on the channel nothing
    routes a render, a legitimacy check or a termination on, so a declared verb would be present
    and unreachable.

    Same admission rule as the parent's :func:`~plurai.orchestrator.orchestrator._agent_actions_of`
    — a dict with a truthy ``name`` — and the same normalised ``{name, args, status}`` shape
    :func:`turn_actions` emits, so one channel carries one shape whichever spelling produced it.
    ``status`` defaults to ``unknown``: a declared act is one the harness saw and whose result it
    did not report.

    THE SECOND RETURN VALUE exists because the caller drops this key from ``agent_state``, and a
    drop wider than the read loses the value entirely: a customer's own ``agent_actions`` that is
    a dict, a list of bare strings, or entries with no name projects to nothing here, and an
    unconditional drop would leave it on NEITHER channel. ``True`` means the read took the value
    WHOLE (a list, every entry admitted — vacuously so for ``[]``); anything else is ``False`` and
    the caller leaves the raw value on ``agent_state``. A
    PARTIALLY readable list is deliberately in the second bucket: what could be read is still
    projected, and the state copy is kept because it is then the only record of the part that
    could not — the same partial-drop posture ``_agent_actions_of`` takes one layer up.
    An ENTRY is read in two halves, its NAME and its ARGUMENTS (:func:`_action_record`), so an
    entry whose arguments this boundary cannot carry is a partial read too: the act ships, the
    arguments report ``{}``, and the raw declaration therefore stays on ``agent_state`` as the
    only remaining copy of what the harness wrote — it is not a DROPPED act, and the count
    below does not move for it.

    THE THIRD is how many declared acts this read could NOT take, and it is what stops a turn that
    acted being reported as a turn that did not. A declaration nothing can be read out of makes the
    turn report ``None`` — "these actions could not be observed" — the statement
    ``_agent_actions_of``'s docstring already makes for the whole channel. It is a COUNT and not a second flag because the
    ``consumed`` bit above already means three different things, and because the warning names it.
    """
    declared = result.get('agent_actions') if isinstance(result, dict) else None
    if not isinstance(declared, list):
        # A declaration this reader cannot take is NOT an absent one. A key never written — or
        # written as a null — declares nothing; anything else on it (a mapping of verb -> args, a
        # bare string, a number, a shape nobody has met yet) is a declaration that was made and
        # could not be read. Keying the difference on ``isinstance(..., list)`` instead would make
        # the answer depend on a TYPE rather than on whether anything was lost.
        #
        # COUNT the declaration where it can be counted: a sized shape declares as many acts as it
        # holds, so a two-verb mapping that lost both says "2 of 2" and — the reason this is not
        # cosmetic — an EMPTY one lost nothing and must stay "observed, and none", exactly like
        # ``[]`` and an absent key. A shape with no length is one act's worth of declaration.
        #
        # A ``str`` IS SIZED AND IS NOT A COLLECTION OF ACTS: ``agent_actions: 'end_conversation'``
        # is ONE declaration written the natural way, and counting its characters would report
        # "17 of 17 act(s) … could not be read" for a harness that declared one.
        if declared is None:
            unreadable = 0
        elif isinstance(declared, str) or not hasattr(declared, '__len__'):
            unreadable = 1
        else:
            unreadable = len(declared)
        _note_dropped_declaration(unreadable, unreadable, read_any=False)
        return [], False, unreadable
    records = [
        _action_record(action['name'], action.get('args'), action.get('status'), action.get('source'))
        for action in declared
        if isinstance(action, dict) and action.get('name')
    ]
    actions = [record for record, _ in records]
    emptied = any(was_emptied for _, was_emptied in records)
    dropped = len(declared) - len(actions)
    _note_dropped_declaration(dropped, len(declared), read_any=bool(actions))
    # ``emptied`` feeds CONSUMED, never ``dropped``: an act whose arguments could not be carried
    # was still read, so the counted drop and the tri-state it decides stay exactly where they
    # were — and the raw declaration stays on ``agent_state``, as the only remaining record of
    # the arguments this boundary could not take.
    return actions, dropped == 0 and not emptied, dropped


def _note_dropped_declaration(dropped: int, declared: int, *, read_any: bool, unknown_source: Any = None) -> None:
    """Say how much of the envelope's declaration this read could not take, and what it costs.

    The child's twin of the parent's
    :func:`~plurai.orchestrator.orchestrator._note_dropped_calls`, and it has to exist HERE: by the
    time the parent re-reads the turn, this gate has already filtered the declaration, so the
    parent's own counter sees 0 of 0 on a total drop and 1 of 1 on a partial one and never fires.
    A drop nobody is told about would make a turn that acted look, for a whole run, like one that
    did not.

    The ``[plurai.child]`` prefix is LOAD-BEARING, not decoration. The child installs no logging
    handler, so this record reaches the parent as a bare stderr line and is re-classified against
    :data:`plurai._subprocess._STDERR_WARN_RE`, which promotes only lines opening with a
    ``[plurai...]`` tag (or a traceback / exception / warnings header). Without the prefix the
    warning is demoted to DEBUG and the drop stays exactly as silent as it was — the same reason
    :func:`_extract_agent_state`'s two warnings carry it.

    Both consequence sentences are about THIS DECLARATION and not about the turn's verdict, which
    this gate cannot see: a turn whose message CHAIN also showed an act still ships that act, so
    "the turn is unobservable" would be a false statement here for exactly the turn that is hardest
    to reason about.

    ``unknown_source`` is a declared act's ``source`` label this channel does not know: the act is
    read, and only its label is left behind.
    """
    if not dropped:
        return
    import logging  # noqa: PLC0415

    if unknown_source is not None:
        logging.getLogger(__name__).warning(
            '[plurai.child] agent_actions: an act declared `source` %r, which the action channel does not know (known: %s); the act ships without it.',
            unknown_source,
            ', '.join(sorted(AGENT_ACTION_SOURCES)),
        )
        return
    logging.getLogger(__name__).warning(
        '[plurai.child] agent_actions: %d of %d act(s) declared on the result envelope could not be read (each entry needs a truthy `name`); %s',
        dropped,
        declared,
        'the turn ships the rest, so its action list is narrower than what the harness declared.'
        if read_any
        else 'nothing from this declaration reaches the action channel, so a turn with no other act to show reports that its '
        'actions could NOT be observed rather than that there were none.',
    )


def _interrupt_action(interrupt_context: dict[str, Any]) -> dict[str, Any]:
    """The pause, as an ACTION: ``interrupt`` carrying the prompt the agent asked, its choices and its actions.

    Stopping to ask a human is something the agent DID, and this is the only channel that says
    so. The prompt is also typed on ``interrupt_context``, but nothing upstream reads that
    field — so without this a HITL turn would cross the wire with no speech and no actions, the shape
    every judge, the turn-legitimacy check and the give-up corroboration all read as the agent
    going silent, with the question it actually asked nowhere. On the action channel the
    simulator's view and the judge transcript render it through the marker they already use for
    every other verb.

    ``status='unknown'`` by construction: an interrupt that had an answer would not be one.
    """
    # ``prompt`` is OMITTED when the key is absent, never defaulted to ``''`` — mirroring the
    # ``choices`` arm below. Absent and empty are different facts on
    # the one channel that says whether a HITL turn happened, and a manufactured empty prompt
    # reads as an interrupt that asked nothing.
    args: dict[str, Any] = {} if (prompt := interrupt_context.get('prompt')) is None else {'prompt': prompt}
    if interrupt_context.get('choices') is not None:
        args['choices'] = interrupt_context['choices']
    if interrupt_context.get('actions'):
        args['actions'] = interrupt_context['actions']
    return {'name': INTERRUPT_ACT, 'args': args, 'status': 'unknown'}


_JSON_SCALARS: Final = (str, int, float, bool, type(None))
#: The classes whose instances LangChain and LangGraph inject into a tool's input (the model never sends them), as
#: ``(module, class)``: LangChain's ``ToolRuntime`` and every other directly injected argument, LangGraph's ``Runtime`` and
#: its store. The child is where the harness's trail still holds these as objects; past :func:`_extract_agent_state` they
#: are text (a ``ToolRuntime`` renders as the whole message history). Read from ``sys.modules`` (:func:`_injected_classes`),
#: so a harness that never imported a framework imports none here.
_INJECTED_CLASSES: Final = (
    ('langchain_core.tools.base', '_DirectlyInjectedToolArg'),
    ('langgraph.runtime', 'Runtime'),
    ('langgraph.store.base', 'BaseStore'),
)
#: Stdlib values with a plain JSON rendering (``_json_default`` writes them as text): data, never an injected object.
_PLAIN_SCALARS: Final = (datetime.date, datetime.time, datetime.timedelta, decimal.Decimal, uuid.UUID, PurePath, enum.Enum, bytes)


def _injected_classes() -> tuple[type, ...]:
    """The :data:`_INJECTED_CLASSES` this process has loaded. A class that is not loaded has no instance to find."""
    return tuple(klass for module, name in _INJECTED_CLASSES if isinstance(klass := getattr(sys.modules.get(module), name, None), type))


def _model_call_args(messages: Sequence[Any]) -> dict[str, set[str]]:
    """``{call id: the argument names the MODEL sent}`` off the turn's chain, read through :func:`normalize_message` (a
    LangChain ``AIMessage.tool_calls``, its dict and OpenAI forms) or an assistant's ``tool_use`` blocks. A call whose
    arguments cannot be read is left out."""
    out: dict[str, set[str]] = {}
    for raw in messages:
        msg = normalize_message(raw)
        calls = msg.get('tool_calls')
        if not isinstance(calls, list) or not calls:
            blocks = _parse_content_blocks(msg.get('content')) if _declared_type(msg) == 'ai' else None
            calls = _tool_calls_from_content_blocks(blocks) if blocks else []
        for call in calls:
            if isinstance(call, Mapping) and isinstance(call_id := call.get('id'), str) and call_id and isinstance(args := call.get('args'), Mapping):
                out[call_id] = {str(name) for name in args}
    return out


def _plain_data(value: Any) -> bool:
    """Whether ``value`` is plain data all the way down: JSON scalars and :data:`_PLAIN_SCALARS` in lists, tuples, sets
    and dicts. A message object, a model instance or a framework object is not."""
    if isinstance(value, (*_JSON_SCALARS, *_PLAIN_SCALARS)):
        return True
    if isinstance(value, Mapping):
        return all(_plain_data(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_plain_data(item) for item in value)
    return False


def _model_visible_trail(trail: list[Any], messages: Sequence[Any]) -> list[Any]:
    """The harness's ``tool_trail`` with the arguments the FRAMEWORK injected into a call left out of its ``args``.

    A harness hooked below the framework records the call's input after the framework injected its own runtime into
    it (LangChain's ``ToolRuntime`` carries the whole message history), and that copy rides every call, every turn, in
    the transcript and in ``tool_events.json`` alike. Left out: a value of an injected framework class
    (:data:`_INJECTED_CLASSES`), always; and, when the turn's chain shows the model's own call
    (:func:`_model_call_args`) AND the harness recorded every name the model sent, a name the model did not send whose
    value is not plain data (:func:`_plain_data` — injected state, a message history). Everything else stays: a wrapper
    (``{"input": raw}``), an alias or a filled-in default keeps its arguments. An entry that lost arguments names them
    under :data:`TRAIL_ARGS_OMITTED_KEY`; every other entry is passed as is.
    """
    sent = _model_call_args(messages)
    injected = _injected_classes()
    out: list[Any] = []
    for entry in trail:
        args = entry.get('args') if isinstance(entry, dict) else None
        if not isinstance(args, Mapping) or not args:
            out.append(entry)
            continue
        call_id = entry.get('call_id')
        declared = sent.get(call_id) if isinstance(call_id, str) else None
        names = {str(name) for name in args}
        unsent_cut = declared is not None and declared <= names
        omitted = sorted(
            str(name)
            for name, value in args.items()
            if isinstance(value, injected) or (unsent_cut and str(name) not in declared and not _plain_data(value))  # type: ignore[operator]
        )
        if not omitted:
            out.append(entry)
            continue
        kept = {name: value for name, value in args.items() if str(name) not in omitted}
        out.append({**entry, 'args': kept, TRAIL_ARGS_OMITTED_KEY: omitted})
    return out


def _extract_agent_state(result: Any) -> dict[str, Any]:
    """Capture the agent's NON-message return state so it isn't discarded at the IPC boundary.

    For a dict result, returns every key EXCEPT ``messages`` (carried separately as
    ``full_messages``). Values are coerced JSON-safe via ``_json_default`` (structured
    ``model_dump``/``dict`` first, then ``str``). A key that STILL won't serialise — e.g. a value
    whose ``model_dump``/``__str__`` raises — is dropped with a WARNING (never silently, never
    failing the turn) naming the key: a SILENT drop would leave an eval seeing nothing, which is the
    failure this capture exists to prevent.

    A NON-dict result (a harness returning a bare string / message object) has no addressable
    state → ``{}``, warned once per returned type — the shape is a property of the harness, not
    of the turn.

    Both warnings carry the ``[plurai.child]`` prefix. The child installs no logging handler, so
    ``logging.lastResort`` writes the BARE message to stderr, and the parent promotes a child
    stderr line to WARNING only when it matches :data:`plurai._subprocess._STDERR_WARN_RE` —
    unprefixed, an operator-facing message lands at DEBUG and is never seen.
    """
    if not isinstance(result, dict):
        type_name = type(result).__name__
        if type_name not in _warned_non_dict_state:
            _warned_non_dict_state.add(type_name)
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                '[plurai.child] invoke_agent returned %s, not a dict — agent_state is empty for every turn that '
                'returns this shape, and this line is emitted once per type, so later ones are silent. '
                'Return the agent state dict (e.g. the LangGraph result) if an eval scores a non-message field.',
                type_name,
            )
        return {}
    state: dict[str, Any] = {}
    for key, value in result.items():
        if key == 'messages':
            continue
        try:
            carried = _model_visible_trail(value, _extract_messages_list(result)) if key == AGENT_STATE_TOOL_TRAIL_KEY and isinstance(value, list) else value
            state[key] = json.loads(json.dumps(carried, default=_json_default))
        except Exception as exc:  # noqa: BLE001 — one bad key must not sink the turn, but say so
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                '[plurai.child] agent_state key %r dropped: not JSON-serialisable even via model_dump/str (%s: %s). '
                'An eval scoring this field will see nothing.',
                key,
                type(exc).__name__,
                exc,
            )
    return state


def _build_agent_response(result: Any, *, open_calls: ThreadCalls) -> AgentResponse:
    """Build an :class:`AgentResponse` envelope from the agent's raw result.

    Preserves the back-compat ``ai_text`` field (trailing AIMessage content)
    while also packing the full serialised message chain (intermediate
    ToolMessages + tool-call AIMessages), interrupt metadata, and the agent's
    non-message return state (``agent_state``). The single point where the IPC
    envelope is constructed — keeps ``handle_user_turn`` clean and is the
    natural seam for unit tests. ``open_calls`` is the thread's call ledger (:func:`turn_actions`).
    """
    messages = _extract_messages_list(result)
    is_interrupted, interrupt_context = _detect_interrupt(result, messages)
    # Execution order, as far as it is knowable: what the chain SHOWS the agent doing, then what
    # the envelope DECLARES (a summary the harness writes for the turn as a whole), then the
    # pause — which is last by definition, because it is where the turn stopped. The envelope is
    # read FIRST so the reply reader counts it: a declared act with no speech is a turn.
    envelope_actions, envelope_consumed, envelope_dropped = _envelope_actions(result)
    reply = _extract_reply(result, interrupted=is_interrupted, declared_actions=envelope_actions, open_calls=open_calls)
    agent_actions = list(reply.actions)
    if is_interrupted:
        agent_actions.append(_interrupt_action(interrupt_context))
    # A declaration the read took WHOLE does not also ride the state channel: one fact on two
    # channels is a fact a reader can grade the stale copy of, and the action channel is the one
    # that decides whether a turn happened at all. Anything the read could not take whole stays
    # put — the drop is exactly as wide as the read, never wider, or a shape this SDK cannot use
    # reaches neither channel and the turn's artifacts lose it.
    #
    # Dropped HERE and not inside ``_extract_agent_state``, which :func:`_error_trail` shares: on
    # THAT path the projection failed, the actions reached no channel, and the degraded turn ships
    # ``agent_actions=None`` by design — so the state copy is the only record of them left, and
    # dropping it there would trade one unreachable copy for none at all.
    agent_state = _extract_agent_state(result)
    if envelope_consumed:
        agent_state.pop('agent_actions', None)
    return AgentResponse(
        ai_text=reply.speech,
        full_messages=[_serialize_message(m) for m in messages],
        is_interrupted=is_interrupted,
        interrupt_context=interrupt_context,
        agent_state=agent_state,
        reply_source=reply.reply_source,
        # THE TRI-STATE, applied over the turn's COMBINED list and read from the one place that
        # states it. Combined, because a turn whose CHAIN showed a real call and whose ENVELOPE
        # declared acts this gate could not read is partially observed: reporting it as
        # unobservable would discard a call the SDK genuinely saw. Only a turn left with nothing
        # to show after a declaration was made says "could not observe" — never "observed, and
        # none", which is the reading ``AgentResponse.agent_actions`` keeps a third state for and
        # the one the parent's guard can no longer restore once this list is empty.
        agent_actions=observed_actions_or_none(agent_actions, saw_declaration=bool(envelope_dropped)),
    )


def _error_trail(result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(full_messages, agent_state)`` to carry on an :class:`AgentError`, or empty on any raise.

    The trail is what the parent rebuilds a degraded turn out of, and it is read out of the same
    unvalidated customer shape that just made the projection fail — so reading it must not be the
    second raise that kills the child after the first one was caught. Empty is the honest
    fallback: the turn is still delivered as an empty scoreable one, with the underlying
    exception on the envelope.
    """
    try:
        return [_serialize_message(m) for m in _extract_messages_list(result)], _extract_agent_state(result)
    except Exception:  # noqa: BLE001 - the trail is a nicety; losing the whole turn to it is not
        traceback.print_exc(file=sys.stderr)
        return [], {}


def _warn_turn_not_pushed(turn_idx: int, reason: str) -> None:
    """Name the turn the service was not told about, and what that costs."""
    # stderr only — stdout carries the JSONL IPC handshake.
    print(
        f'  [plurai] could not set the mock session turn to {turn_idx} ({reason}); '
        'events from callers outside this process keep the last turn the service saw.',
        file=sys.stderr,
    )


def _push_turn_to_mock(turn_idx: int, *, dialog: dict[str, Any], live_tool_calls: list[dict[str, Any]]) -> None:
    """Tell the mock service which turn is open, for callers outside this process.

    A hosted agent reasons and acts on a vendor's servers; its actions reach the
    deployed mock as plain callouts that carry no turn, so the service stamps its
    session turn on them. Nothing in this process can supply that for them.

    ``dialog`` and ``live_tool_calls`` ride the same push (see
    :meth:`_ChildSession._turn_context`): already capped, and the WHOLE history,
    because each push replaces what the session held.

    A failed push costs only that stamping — the three in-process dispatchers read
    ``current_turn()`` and are untouched — so it degrades to a warning. Ending the
    turn here would trade one wrong event field for a dead scenario.

    The push travels the client's ``_post`` path, so it shares that path's 503 retry
    (up to ~15 s) and its 30 s timeout, synchronously and ahead of the agent call,
    inside the same per-turn budget the agent call itself spends.
    """
    client = _registry._client
    sid = plurai.active_session_id()
    if client is None or not sid:
        return
    # the turn leaves the process here too: no credential of the agent's CLI rides it
    dialog, live_tool_calls = json.loads(redact_cli_credentials(json.dumps([dialog, live_tool_calls], default=str)))
    try:
        resp = client.set_turn(sid, turn_idx, dialog=dialog, live_tool_calls=live_tool_calls)
    except MockServiceError as exc:
        _warn_turn_not_pushed(turn_idx, str(exc))
        return
    # The service reports a domain refusal (unknown session, non-integer turn) as an
    # in-body envelope at HTTP 200, so the transport layer never raises for one.
    if resp is not None and resp.get('status') == 'error':
        _warn_turn_not_pushed(turn_idx, f'the service refused it: {resp.get("error")}')
    elif resp is not None and client.supports_session_dialog and ('dialog_entries' not in resp or 'live_tool_calls_stored' not in resp):
        # stderr only — stdout carries the JSONL IPC handshake.
        print(
            f'  [plurai] the mock service took turn {turn_idx} but did not acknowledge the conversation so far; '
            'a mocked tool that generates its answer at run time reads the request without it.',
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Per-scenario handler state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PlantedTurn:
    """What the scenario's channel plan put on one turn (:mod:`plurai.channels`): the payload fields, the
    host-injected messages that precede the user's, and the record of every entry applied."""

    fields: dict[str, Any] = field(default_factory=dict)
    messages: list[Any] = field(default_factory=list)
    records: list[AppliedChannelInput] = field(default_factory=list)


#: A host-injected message's role → the LangChain message class it is handed to the agent as. Names only,
#: resolved at use: ``langchain_core`` stays off the import path of a harness that never plants one.
_HOST_MESSAGE_CLASSES: Mapping[str, str] = {'system': 'SystemMessage', 'user': 'HumanMessage', 'assistant': 'AIMessage'}


class _ChildSession:
    """Stateful per-scenario handler. One instance per ``python -m plurai.child`` process."""

    def __init__(self, module: ModuleType):
        self.module = module
        # The posture this scenario was initialised with. Only 'connect' reaches a
        # service an external caller can also reach, so only it pushes the turn.
        self.mock_mode: MockMode = 'connect'
        # Two hooks, at least one required. Dialog scenarios use invoke_agent;
        # one-shot pipelines use invoke_pipeline. Both may co-exist, and the child
        # answers whichever envelope arrives — but the SHAPE is a property of the
        # HARNESS, declared once (``shape:`` in plurai.config.yaml) and resolved once
        # per run, so every scenario of a run takes the same route.
        self.invoke_agent = _resolve_invoke_agent(module)
        self.invoke_pipeline = _resolve_invoke_pipeline(module)
        if self.invoke_agent is None and self.invoke_pipeline is None:
            raise AttributeError(
                f'{getattr(module, "__file__", "<simulate>")} must export at least '
                'one of `invoke_agent(payload, config)` (dialog) or '
                '`invoke_pipeline(inputs)` (one-shot).'
            )
        self.seed_state = _resolve_seed_state(module)
        # The hook that CLEARS, resolved beside the one that seeds. Optional: a harness whose
        # whole world is the mock's needs nothing here, because the backend already opens a fresh
        # session per scenario. See :func:`_resolve_reset`.
        self.reset = _resolve_reset(module)
        # The hook that OBSERVES, resolved beside the one that clears. Optional, and its absence
        # is the only thing the world-owner declaration decides — see :meth:`handle_finalize`.
        self.world_snapshot = _resolve_world_snapshot(module)
        # The hook that applies the simulated user's act to a world the harness owns. Optional.
        self.apply_user_action = _resolve_user_action_hook(module)
        # What the simulated user can see, captured after every agent turn (see ``_turn_world``):
        # the declared surface row types (read once), the last capture reported as observed, the
        # post-seed delta the first observed report carries, and why every turn is unobservable
        # when the post-seed baseline itself could not be read.
        self._surface_types: frozenset[str] | None = None
        self._surface_types_read = False
        self._surface_types_unreadable = False
        self._world_before: dict[str, dict[str, Any]] = {}
        self._seed_delta: dict[str, Any] | None = None
        self._baseline_missing: world_delta.UnobservableReason | None = None
        # The result of each user act this child applied, by the ``turn_idx`` that asked for it:
        # an act is applied once, whatever re-delivers it.
        self._user_action_results: dict[int, dict[str, Any]] = {}
        # The scenario's channel plan (read at Init off ``Init.scenario_metadata``) and what it put on each turn,
        # by ``turn_idx``: like a user act, a turn's channel inputs are applied once, whatever re-delivers it.
        self._channel_plan: ChannelPlan = EMPTY_PLAN
        self._planted_turns: dict[int, _PlantedTurn] = {}
        self.thread_id: str = ''
        self.initial_agent_context: dict[str, Any] = {}
        # WHICH injected keys the harness read: the Init keys, and every mapping they were handed
        # to the harness in (``seed_state`` param 1, the turn-1 payload, a hop's inputs). ``None``
        # until Init, so a child that never initialised claims nothing (``FinalizeResponse``).
        self._declared_inputs: frozenset[str] | None = None
        self._input_reads: list[InputReads] = []
        # EVERY DOCUMENT HANDED OVER AS A ``file`` PART, as ``(filename, its spec as handed)``: the spec is an
        # :class:`InputReads`, so reading its ``file_data`` is recorded exactly as reading a payload key is. A
        # handed-over document arrives only on a turn's parts (the opener's included), so a harness that never reads
        # the part runs without the document — and is named in ``unread_inputs`` by the field (the filename's stem).
        self._handed_parts: list[tuple[str, InputReads]] = []
        # Whether the mock can serve a key ON DEMAND (a handler reads ``ctx.agent_context`` only when
        # the agent calls its tool; a deployed mock is out of sight). A read on any lane credits a key;
        # a key nobody read is named only when no such lane exists — a non-read proves nothing there.
        self._mock_on_demand = True
        # The in-process mock whose copy is recorded: a handler it gets AFTER Init (mocks imported in
        # ``seed_state``) is served on demand too, so finalize re-reads its registry.
        self._mock_service: Any = None
        self._initialized = False
        # agent_context is the scenario's startup state (preconditions). It's
        # seeded into the agent's graph state on the FIRST user turn only; after
        # that the agent's checkpointer (keyed by the stable thread_id we pass
        # every turn) owns it — re-seeding the STATIC values each turn would
        # clobber any field the agent mutated mid-conversation. See handle_user_turn.
        self._agent_context_seeded = False
        # The harness's own ``agent.io`` declaration, read at most ONCE and only when a turn
        # actually needs it (see :meth:`agent_io`). Lazily, because a ``str`` turn asks nothing
        # of the declaration (the rule, once, on :class:`plurai._protocol.UserTurn`): a bench
        # driven with strings alone never opens a config file at all. A parts list and the empty
        # opener both DO — and both ride scored runs, so this is a deferral and not dead code.
        self._agent_io: AgentIO | None = None
        # Whether the bench's own ``world_state.owner`` says the mock owns the world, read at most
        # ONCE and only when finalize has no snapshot hook to call (see :meth:`mock_owns_the_world`).
        # Deferred rather than read in this constructor for the reason ``_agent_io`` is: a bench
        # that exports the hook never needs the word at all.
        self._mock_owns_the_world: bool | None = None
        # Set at Init only when the parent names an entry to witness; see ``plurai._entry_witness``.
        self._entry_witness: EntryWitness | None = None
        # A finalize-time snapshot read failed in a way one more read can clear (see :meth:`_capture`).
        self._capture_retryable = False
        # A reply already typed the scenario not_valid (see :meth:`answered`): a later finalize reads crash evidence only.
        self._not_valid_reported = False
        # The finalize world snapshot's values carried in their JSON form, ``{path: source type}``.
        self._world_values_coerced: dict[str, str] = {}
        # The conversation so far and the results of the tools the harness ran for real, pushed on
        # every connect-mode turn as a runtime mock's interpretation context. Kept raw and in the
        # trail shape; capped at each push (see :meth:`_turn_context`).
        self._dialog: list[dict[str, Any]] = []
        self._live_tool_calls: list[dict[str, Any]] = []
        self._live_tools: frozenset[str] | None = None
        self._owned_types: frozenset[str] | None = None
        self._owned_types_read = False
        self._undeclared_hook_types_named = False
        self._seen_live_call_ids: set[tuple[str, str]] = set()
        # The thread's unresolved tool calls by id (see :func:`turn_actions`). Emptied at Init: a
        # resume seed's calls are history, and a live result must not settle one of them.
        self._open_calls = ThreadCalls()

    def mock_owns_the_world(self) -> bool:
        """Does the bench's ``world_state.owner`` say the world this suite scores is the MOCK's?

        ``True`` for a declared ``mock`` AND for an absent declaration, which are one answer: a
        bench that declares nothing has the mock's world. It is the ONLY thing that decides what the ABSENCE of a
        :data:`~plurai.orchestrator.orchestrator.WORLD_SNAPSHOT_HOOK` means; it never decides the
        world channel on its own (see :meth:`handle_finalize`).

        Resolved from the config BESIDE the harness, exactly as :attr:`agent_io` resolves
        ``agent.io`` and as ``orchestrator._declared_world_owner`` resolves this same key in the
        parent — so there is no ``Init`` field to add, no protocol change, and the child and the
        parent cannot come to hold two answers to one question. That matters more here than
        convenience: ``plurai._protocol.decode`` hard-refuses an unknown field NAME and the
        envelope carries no version negotiation, so a child venv pinned to an older ``plurai``
        (``PLURAI_CHILD_PYTHON`` exists precisely so one can be) would die at init on every
        scenario the moment the parent started sending a field it does not know.

        A module with no ``__file__`` — a synthesized test module — declares nothing, which is the
        same answer an absent config gives.

        A CONFIG THAT CANNOT TESTIFY ANSWERS ``False``, loudly. An illegal value or two rungs that
        disagree is refused at run start on the CLI path, so this arm is reachable only by an
        embedder driving the Orchestrator directly — and there the alternative to abstaining is
        worse in both directions: reading it as ``mock`` certifies a world nobody established the
        ownership of, and letting the error escape kills the child at finalize and discards a
        completed transcript, which is exactly what this method's caller must never do.
        """
        if self._mock_owns_the_world is None:
            try:
                source = getattr(self.module, '__file__', None)
                declared = _declared_world_owner(Path(source)) if source else None
            except WorkspaceConfigError as exc:
                print(f'[plurai.child] world owner unreadable at finalize, so no world was claimed: {exc}', file=sys.stderr, flush=True)
                self._mock_owns_the_world = False
            else:
                self._mock_owns_the_world = declared is None or declared[0] == WORLD_OWNER_MOCK
        return self._mock_owns_the_world

    @property
    def agent_io(self) -> AgentIO:
        """The ``agent.io`` declared beside this ``simulate.py``, or the undeclared default.

        Resolved from the config BESIDE the harness, exactly as ``orchestrator._declared_shape``
        resolves ``shape:`` — so there is no ``Init`` field to add, no protocol change, and the
        child reads the same two rungs the parent does. A module with no ``__file__`` (a
        synthesized test module) declares nothing and gets the default.
        """
        if self._agent_io is None:
            source = getattr(self.module, '__file__', None)
            self._agent_io = (declared_agent_io(source) or DEFAULT_AGENT_IO) if source else DEFAULT_AGENT_IO
        return self._agent_io

    @property
    def live_tools(self) -> frozenset[str]:
        """The tools this harness runs for real, read once from the config beside it.

        A tool is live when it is declared ``mode: real`` / ``mode: observe`` under ``tools:``, or
        listed in ``agent.native_tools``. Every other call reached the mock, whose own log already
        holds it. A module with no ``__file__`` declares none; a config that cannot be read declares
        none, loudly.
        """
        if self._live_tools is None:
            source = getattr(self.module, '__file__', None)
            try:
                self._live_tools = _declared_live_tools(Path(source)) if source else frozenset()
            except WorkspaceConfigError as exc:
                print(f'[plurai.child] tool modes unreadable, so no live tool result is given to the mock: {exc}', file=sys.stderr, flush=True)
                self._live_tools = frozenset()
        return self._live_tools

    @property
    def owned_types(self) -> frozenset[str] | None:
        """``world_state.owned_types`` beside this harness: the entity types its own store holds, read once.

        ``None`` when nothing declares it (the hook's world is reported verbatim). A
        declaration that cannot be read declares nothing, loudly.
        """
        if not self._owned_types_read:
            self._owned_types_read = True
            source = getattr(self.module, '__file__', None)
            try:
                self._owned_types = _declared_owned_types(Path(source)) if source else None
            except WorkspaceConfigError as exc:
                print(f'[plurai.child] world_state.owned_types unreadable, so the world is reported as the hook returns it: {exc}', file=sys.stderr, flush=True)
        return self._owned_types

    def _compose_world(self, mock_state: Mapping[str, Any], hook_state: Mapping[str, Any], owned: frozenset[str]) -> dict[str, Any]:
        """:func:`compose_split_world`, naming once on stderr a hook type ``owned`` does not declare."""
        composed, undeclared = compose_split_world(mock_state, hook_state, owned)
        if undeclared and not self._undeclared_hook_types_named:
            self._undeclared_hook_types_named = True
            print(
                f'[plurai.child] {WORLD_SNAPSHOT_HOOK}() returned types world_state.owned_types does not declare: {", ".join(undeclared)}; '
                "kept, and the mock's rows of those types left out. Declare them in world_state.owned_types.",
                file=sys.stderr,
                flush=True,
            )
        return composed

    def _declared_surface_types(self) -> frozenset[str] | None:
        """The entity types the ``world_state.user_surfaces`` beside this harness show, read once.

        ``None`` when nothing declares the key. Read per entry: an entry whose paths name no entity
        type is dropped (named on stderr) and the other surfaces are still captured. A declaration
        that cannot be read at all — or whose every entry was dropped — sets
        ``_surface_types_unreadable``, loudly: every turn is then ``read_failed``, never
        ``not_declared``, because a declaration exists.
        """
        if not self._surface_types_read:
            self._surface_types_read = True
            source = getattr(self.module, '__file__', None)
            try:
                self._surface_types, dropped = discover_surface_types(Path(source)) if source else (None, ())
            except UserSurfacesConfigError as exc:
                print(
                    f'[plurai.child] the user surfaces cannot be read, so the simulated user is shown nothing of the world: {exc}', file=sys.stderr, flush=True
                )
                self._surface_types_unreadable = True
                return self._surface_types
            for defect in dropped:
                print(
                    f'[plurai.child] user surface {defect.name!r} is not captured: its `{defect.code}` names no entity type ({defect.value}).',
                    file=sys.stderr,
                    flush=True,
                )
            if dropped and not self._surface_types:
                self._surface_types_unreadable = True
        return self._surface_types

    async def _capture(self, read: Callable[[], Any], success_key: str, *, at_finalize: bool) -> dict[str, Any]:
        """Run one snapshot read, converting any failure into a crash marker.

        A read at finalize that fails because the MOCK went away, or timed out, also sets the
        session's ``capture_retryable`` bit: the scenario reached finalize and one more read after the
        mock proves ready can still capture it. Recording it here is what keeps the parent's reading typed — the
        alternative is reading the marker's message text back out. It is NOT an unanswered tool
        call: a failed read fabricated nothing for the agent. A per-turn read
        (``at_finalize=False``) records nothing: that classification is finalize's.

        AWAIT-AWARE, so the ONE error path covers a CUSTOMER hook too. The mock reads are
        plain calls into this SDK; the world-snapshot
        hook is a harness export, and the harness may reach an async store to answer it — the
        same latitude ``reset`` and ``seed_state`` already have. Resolving the awaitable
        inside this try is what keeps a hook that raises AFTER its first suspension point on
        the ``_finalize_error`` marker rather than on a dead child.
        """
        from plurai._mocking.resolution import _is_restart_shaped  # noqa: PLC0415

        try:
            result = read()
            if inspect.isawaitable(result):
                result = await result
            return _as_crash_marker_if_error(result, success_key)
        except Exception as exc:
            if (
                at_finalize
                and isinstance(exc, MockServiceError)
                and (exc.transport == 'timeout' or _is_restart_shaped(status_code=exc.status_code, code=exc.code, transport=exc.transport))
            ):
                self._capture_retryable = True
            marker: dict[str, Any] = {'_finalize_error': f'{_exc_class(exc)}: {exc}'}
            if isinstance(exc, MockServiceError) and (exc.transport is not None or (exc.status_code is not None and exc.status_code >= 500)):
                # The backend did not answer (a timeout, a dropped connection, a 5xx), typed off the error's own fields:
                # not the harness's read failing, so the seed read-back does not void the scenario on it.
                marker[BACKEND_TRANSPORT_FAILURE_KEY] = True
            return marker

    async def _post_seed_read(self) -> dict[str, Any] | None:
        """One read of the world right after ``seed_state`` (:meth:`_read_world`), a read with no ``state`` map typed as
        the crash marker: it observed no world, so it is never a bare ``{}`` read as an empty one."""
        post_seed = await self._read_world(at_finalize=False)
        if post_seed is not None and '_finalize_error' not in post_seed and not isinstance(post_seed.get('state'), Mapping):
            post_seed = {'_finalize_error': f'the world read returned no state map ({type(post_seed.get("state")).__name__})'}
        return post_seed

    async def _read_world(self, *, at_finalize: bool) -> dict[str, Any] | None:
        """:meth:`_read_world_route`, with what a row says of a produced image set by who reads it.

        The finalize read keeps the rows' ``_image_path`` (the file's location, never its bytes) for the
        parent to upload. Every other read (the post-seed baseline and the per-turn capture the simulated user
        is shown) drops it and ``image``: she reads text, and the path means nothing to her.
        """
        world = await self._read_world_route(at_finalize=at_finalize)
        state = world.get('state') if world is not None else None
        if at_finalize or world is None or '_finalize_error' in world or not isinstance(state, Mapping):
            return world
        return {**world, 'state': file_rows.without_images(state)}

    async def _read_world_route(self, *, at_finalize: bool) -> dict[str, Any] | None:
        """The world this scenario is scored on, read by the one table finalize and every turn share.

        THE WORLD CHANNEL DISPATCHES ON THE HOOK, and the owner decides ONLY what the
        hook's absence means. Three routes, one table:

          hook exported          -> what the harness handed over, whoever owns the world.
                                    An explicit override: a mock-owned bench that exports
                                    it has asked for its own answer and gets it. With
                                    ``world_state.owned_types`` declared (a split world),
                                    composed with the mock's rows of every other type
                                    (:meth:`_compose_world`).
          no hook, mock's world  -> the mock's session, for every bench that declares
                                    nothing, which is most of them.
          no hook, someone else's-> ``None``. The store the declaration names is out of
                                    this process's reach and the only thing here to read is
                                    the MOCK's session, which is not it. Reporting that one
                                    scores a conclusive verdict about a world nobody
                                    snapshotted; saying nothing demotes those policies to
                                    not-measured. Run start already warned, by name, with
                                    the hook to add (``unobservable_world_warning``).

        NOT on the owner alone, which is the tempting one-liner and is wrong: a bench may
        declare ``agent`` and still keep part of its world in the mock — the declaration is
        about RESETTABILITY — so suppressing the read on the word would throw away a world
        that is really there. And never inferred from an empty-looking result: the third
        state is DECLARED, and a mock holding one stale row would defeat the inference
        anyway.

        The hook's read goes through the SAME :meth:`_capture` as the mock's, so a hook that
        raises lands on ``_finalize_error`` exactly as a failed mock read does — one error
        path for the channel, not two. What it returns is the ``state`` map itself; the
        envelope around it is this child's, so both routes hand the parent one shape.
        """
        thread_id = self.thread_id
        if (snapshot_hook := self.world_snapshot) is not None:

            async def _declared_world() -> dict[str, Any]:
                """The harness's own world, in the envelope the mock read hands back.

                The ENVELOPE is this child's, not the harness's: the hook returns the
                ``{entity_id: {...}}`` map and nothing else, so both routes out of this
                method hand the parent one shape and a harness author has one thing to
                return. ``_json_shaped`` names an unserializable value instead of letting
                the envelope encoder replace it with its repr — the same belt the pipeline
                outputs wear, and for the same reason: a value that evaporates in transit
                reaches the judges as a world, not as a failure.
                """
                result = snapshot_hook()
                if inspect.isawaitable(result):
                    result = await result
                coerced: dict[str, str] = {}
                shaped = _json_shaped(result, coerced=coerced, _path=f'{WORLD_SNAPSHOT_HOOK}()')
                if at_finalize:
                    self._world_values_coerced = coerced
                return {'session_id': thread_id, 'state': shaped}

            declared = await self._capture(_declared_world, 'state', at_finalize=at_finalize)
            owned = self.owned_types
            if owned is None or '_finalize_error' in declared or not isinstance(declared.get('state'), Mapping):
                return declared
            # A split world (``world_state.owned_types``): the mock holds the rest, so both halves
            # are read and composed here — never by hand in the hook. A failed read on either side
            # is the channel's one error path.
            mock = await self._capture(lambda: plurai.get_full_state(thread_id), 'state', at_finalize=at_finalize)
            if '_finalize_error' in mock:
                return mock
            mock_state = mock.get('state')
            return {**declared, 'state': self._compose_world(mock_state if isinstance(mock_state, Mapping) else {}, declared['state'], owned)}
        if self.mock_owns_the_world():
            return await self._capture(lambda: plurai.get_full_state(thread_id), 'state', at_finalize=at_finalize)
        return None

    @staticmethod
    def _rows_of(captured: dict[str, Any] | None, types: frozenset[str]) -> dict[str, dict[str, Any]] | world_delta.UnobservableReason:
        """The rows of ``types`` in one world read, or why that read holds none."""
        if captured is None:
            return 'no_world_route'
        state = captured.get('state')
        if '_finalize_error' in captured or not isinstance(state, Mapping):
            return 'read_failed'
        return world_delta.surface_rows(state, types)

    def _read_baseline(self, initial_world_state: Mapping[str, Any], post_seed: dict[str, Any] | None) -> None:
        """The post-seed world the first turn's delta is taken from, and the seed's own delta.

        The harness's ``seed_state`` may reconcile the world it was handed, so the server's fold
        starts from ``initial_world_state + seed_delta``, never from ``initial_world_state`` alone.
        ``post_seed`` is the one post-seed read (``InitOk.post_seed_world_state``); only its surface rows are kept.
        """
        types = self._declared_surface_types()
        if types is None or self._surface_types_unreadable:
            return
        rows = self._rows_of(post_seed, types) if types else {}
        if isinstance(rows, str):
            self._baseline_missing = rows
            return
        self._world_before = rows
        self._seed_delta = world_delta.diff_world(world_delta.surface_rows(world_delta.initial_entities(initial_world_state), types), rows)

    async def _turn_world(self) -> dict[str, Any]:
        """``TurnWorld`` for the agent turn that just ended: its delta, or why there is none."""
        types = self._declared_surface_types()
        if self._surface_types_unreadable:
            return world_delta.unobservable('read_failed', types=())
        if types is None:
            return world_delta.unobservable('not_declared', types=())
        if self._baseline_missing is not None:
            return world_delta.unobservable(self._baseline_missing, types=types)
        rows = self._rows_of(await self._read_world(at_finalize=False), types) if types else {}
        if isinstance(rows, str):
            return world_delta.unobservable(rows, types=types)
        delta = world_delta.diff_world(self._world_before, rows)
        if any(d is not None and world_delta.delta_chars(d) > world_delta.TURN_WORLD_MAX_CHARS for d in (delta, self._seed_delta)):
            return world_delta.unobservable('overflow', types=types)
        seed_delta, self._seed_delta, self._world_before = self._seed_delta, None, rows
        return world_delta.observed(types=types, delta=delta, seed_delta=seed_delta)

    async def _apply_user_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply the simulated user's act to the world as HER write, before the agent's turn runs.

        The route is :data:`_USER_ACTION_ROUTES`, inside :func:`user_action` (every world write stamped
        ``source='user'``). ``refused`` means the world refused the write: the hook raised, or the mock
        refused the key (a harness fault noted at its raise site, taken off the record here so the
        agent's turn is never charged with the user's act). A product outcome of her act (a declined
        card) is the hook writing that outcome and returning: ``applied``. A refusal names WHO refused
        (``refused_by``): ``'hook'`` when the harness's hook wrote it, ``'mock'`` when the child wrote the
        mock's world itself. Applied once per asking ``turn_idx``.
        """
        turn_idx = action.get('turn_idx')
        if isinstance(turn_idx, int) and turn_idx in self._user_action_results:
            return self._user_action_results[turn_idx]
        entity_id, updates = action.get('entity_id'), action.get('updates')
        refused_by: str | None = None
        unexecutable_by: str | None = None
        detail: str | None
        if not isinstance(entity_id, str) or not entity_id or not isinstance(updates, dict):
            status, detail = 'unexecutable', 'the act names no row or no field values to write'
        else:
            status, detail, refused_by, unexecutable_by = await self._write_as_user(action)
        result = {
            'turn_idx': turn_idx,
            'status': status,
            **({'detail': detail} if detail else {}),
            **({'refused_by': refused_by} if refused_by else {}),
            **({'unexecutable_by': unexecutable_by} if unexecutable_by else {}),
        }
        if isinstance(turn_idx, int):
            self._user_action_results[turn_idx] = result
        return result

    async def _write_as_user(self, act: dict[str, Any]) -> tuple[str, str | None, str | None, str | None]:
        """``(status, detail, refused_by, unexecutable_by)`` of writing one act to the world as the USER's write.

        The one world-write path of everything the user does outside her words: her own act on a surface
        (:meth:`_apply_user_action`) and the scenario's planted channel writes (:meth:`_plant_channel_inputs`).
        """
        take_harness_faults()
        with user_action():
            status, detail, unexecutable_by = await self._write_user_action(act)
        if faults := take_harness_faults():
            status, detail, unexecutable_by = 'refused', '; '.join(dict.fromkeys(f.message for f in faults)), None
        refused_by = ('hook' if self.apply_user_action is not None else 'mock') if status == 'refused' else None
        return status, detail, refused_by, unexecutable_by

    async def _write_channel_row(self, entry: PlannedChannelInput, entity_id: str, updates: dict[str, Any], **extra: Any) -> tuple[ChannelStatus, str | None]:
        """Write one planted row through :meth:`_write_as_user`. The act names the channel as its ``surface`` and the
        channel's ``field`` as its ``action``, so an ``apply_user_action`` hook dispatches it like any act."""
        act = {
            'turn_idx': entry.turn,
            'surface': entry.channel.name,
            'action': entry.channel.field,
            'channel': entry.channel.name,
            'kind': entry.channel.kind,
            'entity_id': entity_id,
            'updates': updates,
            **extra,
        }
        status, detail, refused_by, unexecutable_by = await self._write_as_user(act)
        why = refused_by or unexecutable_by
        return cast('ChannelStatus', status), (f'{detail} ({why})' if detail and why else detail)

    async def _plant_file_by_reference(self, entry: PlannedChannelInput, planted: _PlantedTurn) -> AppliedChannelInput:
        """Write each item into the store the agent reads, and pass its id (or the id list) in the declared field."""
        items = entry.value if isinstance(entry.value, list) else [entry.value]
        ids: list[str] = []
        status: ChannelStatus = 'applied'
        detail: str | None = None
        for item in items:
            entity_id, fields = item_write(item)
            ids.append(entity_id)
            written, why = await self._write_channel_row(entry, entity_id, fields)
            if written != 'applied' and status == 'applied':
                status, detail = written, f'{entity_id}: {why}'
        handed: Any = ids if isinstance(entry.value, list) else ids[0]
        planted.fields[entry.channel.field] = handed
        return AppliedChannelInput(entry.channel.name, entry.channel.kind, entry.channel.field, status, value=handed, detail=detail)

    async def _plant_structured_context(self, entry: PlannedChannelInput, planted: _PlantedTurn) -> AppliedChannelInput:
        """Set the per-turn host payload field."""
        planted.fields[entry.channel.field] = entry.value
        return AppliedChannelInput(entry.channel.name, entry.channel.kind, entry.channel.field, 'applied', value=entry.value)

    async def _plant_host_injected_message(self, entry: PlannedChannelInput, planted: _PlantedTurn) -> AppliedChannelInput:
        """Add the host's message to the history the agent receives this turn, ahead of the user's."""
        import langchain_core.messages as lc_messages  # noqa: PLC0415 - see _HOST_MESSAGE_CLASSES

        message_class = getattr(lc_messages, _HOST_MESSAGE_CLASSES[entry.value['role']])
        planted.messages.append(message_class(content=entry.value['content']))
        return AppliedChannelInput(entry.channel.name, entry.channel.kind, entry.channel.field, 'applied', value=dict(entry.value))

    async def _plant_between_turn_action(self, entry: PlannedChannelInput, planted: _PlantedTurn) -> AppliedChannelInput:  # noqa: ARG002 - one planter signature
        """Apply the user/UI change to the world as the user's write, before the next agent call."""
        entity_id, fields, created = mutation_write(entry)
        status, detail = await self._write_channel_row(entry, entity_id, fields, **({'create': True} if created else {}))
        value = {'entity_id': entity_id, **({'create': fields} if created else fields)}
        return AppliedChannelInput(entry.channel.name, entry.channel.kind, entry.channel.field, status, value=value, after_turn=entry.after_turn, detail=detail)

    async def _plant_channel_inputs(self, turn_idx: int) -> _PlantedTurn:
        """Apply the plan's entries due at the open of ``turn_idx``, once per turn, in :meth:`ChannelPlan.for_turn` order."""
        if (planted := self._planted_turns.get(turn_idx)) is not None:
            return planted
        planted = self._planted_turns[turn_idx] = _PlantedTurn()
        for entry in self._channel_plan.for_turn(turn_idx):
            planted.records.append(await _CHANNEL_PLANTERS[entry.channel.kind](self, entry, planted))
        return planted

    async def _write_user_action(self, act: dict[str, Any]) -> tuple[str, str | None, str | None]:
        """``(status, detail, unexecutable_by)`` of writing one act by its route in :data:`_USER_ACTION_ROUTES`.

        ``unexecutable_by`` types why nothing was written: ``'no_hook'`` (only the harness's hook could
        write this act, and it exports none) or ``'no_mock'`` (no mock client or session could be
        reached for the write). ``None`` otherwise.
        """
        if self.apply_user_action is not None:
            try:
                applied = self.apply_user_action(dict(act))
                if inspect.isawaitable(applied):
                    await applied
            except Exception as exc:  # noqa: BLE001 - the harness refused her act; the turn goes on
                return 'refused', f'{_exc_class(exc)}: {exc}', None
            return 'applied', None, None
        route = _USER_ACTION_ROUTES['inline' if act.get('inline_row') is not None else 'entity', self.mock_owns_the_world()]
        if route != _MOCK_WRITE:
            return 'unexecutable', route, 'no_hook'
        client, sid = _registry._client, plurai.active_session_id()
        if client is None or not sid:
            return 'unexecutable', 'there is no mock session to write the act to', 'no_mock'
        try:
            written = client.update_state(sid, {act['entity_id']: dict(act['updates'])})
        except MockServiceError as exc:
            return 'unexecutable', f'the write did not reach the mock: {exc}', 'no_mock'
        # The mock answered and did not apply the write: the mock refused it, like a refused key (which
        # was also noted as a harness fault; the caller turns that into the same ``refused``).
        if isinstance(written, dict) and 'error' in written:
            return 'refused', f'the mock did not apply the write: {written["error"]}', None
        return 'applied', None, None

    def _record_user(self, msg: UserTurn) -> None:
        """Append this turn's user entry to the dialog: its text, a non-text part as ``[<kind> part]``."""
        parts: list[str] = []
        if isinstance(msg.user_message, str):
            parts.append(msg.user_message)
        elif isinstance(msg.user_message, list):
            for part in msg.user_message:
                kind = part.get('type') if isinstance(part, dict) else None
                parts.append(str(part.get('text', '')) if kind == 'text' and isinstance(part, dict) else f'[{kind or "unknown"} part]')
        if msg.interrupt_choice is not None:
            parts.append(f'[choice: {msg.interrupt_choice}]')
        if parts:
            self._dialog.append({'turn': msg.turn_idx, 'role': 'user', 'text': '\n'.join(parts)})

    def _record_reply(self, turn_idx: int, response: AgentResponse) -> None:
        """Append the agent's reply to the dialog, and the results of the live tools it ran this turn."""
        if response.ai_text:
            self._dialog.append({'turn': turn_idx, 'role': 'agent', 'text': response.ai_text})
        live = self.live_tools
        if not live:
            return
        try:
            # A live-tool relay: a render is never a live tool, so this reader has no action channel.
            calls = recovered_tool_calls(response.full_messages, response.agent_state, agent_actions_by_turn=None)
        except Exception as exc:  # noqa: BLE001 - the mock's context is a nicety; the turn is not
            # Never silent: the parent relays this stderr line into its log at WARNING, and the class
            # names a reader that broke.
            print(
                f'[plurai.child] the live tool results of turn {turn_idx} could not be read, so the mock is not given them: {_exc_class(exc)}: {exc}',
                file=sys.stderr,
            )
            return
        for call in calls:
            if call['tool'] not in live or call.get('_unpaired'):
                continue
            # The parent's rule (``orchestrator._fresh_tool_calls``): an agent that returns its whole
            # history every turn re-lists every earlier call, so a call seen before belongs to the
            # turn that issued it and is not pushed again. A call with no id is kept. Identity is
            # PER TOOL (``tool_call_recovery.union_by_call_identity``): a harness that numbers its
            # dispatches per tool gives two tools the same id, and they are two calls.
            key = (call['tool'], call['id'])
            if call['id'] and key in self._seen_live_call_ids:
                continue
            if call['id']:
                self._seen_live_call_ids.add(key)
            self._live_tool_calls.append({'turn': turn_idx, 'tool': call['tool'], 'args': call['args'], 'result': call['response']})

    def _turn_context(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """``(dialog, live_tool_calls)`` for the turn push: the whole history, capped."""
        dialog = cap_dialog([DialogEntry(**entry) for entry in self._dialog])
        live = cap_prior_calls([PriorToolCall(turn=c['turn'], tool=c['tool'], args=c['args'], content=c['result']) for c in self._live_tool_calls])
        return dialog.model_dump(mode='json'), [call.model_dump(mode='json') for call in live]

    def _turn_messages(self, user_message: str | list[dict[str, Any]] | None) -> list[Any]:
        """THIS turn's message list, projected against the harness's declaration.

        Three shapes, and the declaration decides only the last two:

        * ``str`` — one ``HumanMessage``. NEVER refused, whatever ``agent.io.input`` says: a ``str`` turn asks nothing of
          the declaration (:class:`plurai._protocol.UserTurn` states the rule), so refusing it on
          one would break the run path a declaration is supposed to widen — including the
          agent-first bench, whose user still SPEAKS after turn 0.
        * ``None`` — the EMPTY OPENER: ``[]``, so ``invoke_agent`` is called with
          ``{'messages': []}``. Refused unless ``may_open`` declares the harness can serve one.
        * ``list`` — OpenAI content parts, each passed VERBATIM once the declaration admits its
          kind. Projected, not filtered: an undeclared part is refused, never dropped, because a
          dropped attachment is an agent answering a question it was never shown.
        """
        # Local import — keep langchain_core off the import path of customers who use the SDK
        # without a LangChain agent (e.g., for unit-testing the protocol).
        from langchain_core.messages import HumanMessage  # noqa: PLC0415

        if isinstance(user_message, str):
            return [HumanMessage(content=user_message)]
        if user_message is None:
            if not self.agent_io.may_open:
                raise UndeclaredTurnInputError(
                    'undeclared_empty_opener',
                    f'this turn has no user message (the agent opens the conversation), but {self._declared_by()} does not declare '
                    f'`{AGENT_IO_KEY}: {{{MAY_OPEN_FIELD}: true}}`. Calling `invoke_agent` with an empty message list at a harness '
                    f'that never said it can serve one is how a bench dies on its first turn.',
                )
            return []
        if isinstance(user_message, list):
            return [HumanMessage(content=[self._tracked_part(self._declared_part(part, index)) for index, part in enumerate(user_message)])]
        raise UndeclaredTurnInputError(
            'undeclared_input_part',
            f'this turn carries a {type(user_message).__name__} user message; a turn is a string, a list of OpenAI content parts, '
            f'or nothing at all (the empty opener).',
        )

    def _tracked_part(self, part: Any) -> Any:
        """``part``, with a ``file`` part's spec handed over as an :class:`InputReads` whose ``file_data`` read is recorded.

        Every other part, and a ``file`` part with no ``file_data``, is returned as it is. The copy is the spec the
        harness receives — the turn's own record (``msg.user_message``) is never wrapped, so nothing the child reads
        for its own bookkeeping counts as the harness reading the document.
        """
        spec = part.get('file') if isinstance(part, Mapping) and part.get('type') == 'file' else None
        if not isinstance(spec, Mapping) or 'file_data' not in spec:
            return part
        tracked = InputReads(dict(spec))
        self._handed_parts.append((str(spec.get('filename') or ''), tracked))
        return {**part, 'file': tracked}

    def _declared_part(self, part: Any, index: int) -> Any:
        """One content part, verbatim — or a typed refusal naming the kind and the declaration.

        The kind is resolved through :data:`~plurai.agent_io.INPUT_PART_KINDS`, a LOOKUP: a kind
        this build has never heard of does not resolve, and a kind that merely LOOKS like one it
        knows (``text_v2``) is refused by name rather than flattered into ``text``.
        """
        if not isinstance(part, Mapping):
            raise UndeclaredTurnInputError(
                'undeclared_input_part',
                f"message content part {index} is a {type(part).__name__}, not a content-part mapping (`{{'type': <kind>, <kind>: …}}`, the OpenAI shape).",
            )
        kind = part.get('type')
        if not isinstance(kind, str) or not kind.strip():
            raise UndeclaredTurnInputError(
                'undeclared_input_part',
                f'message content part {index} names no `type` (its keys are {sorted(map(str, part))}), so there is no kind to '
                f'check against {self._declared_by()}.',
            )
        payload_key = INPUT_PART_KINDS.get(kind)
        if payload_key is None:
            raise UndeclaredTurnInputError(
                'undeclared_input_part',
                f'message content part {index} is of kind {kind!r}, which this build of plurai cannot carry; it carries '
                f'{sorted(INPUT_PART_KINDS)}. Upgrade plurai on both sides if a newer build declares it.',
            )
        if not self.agent_io.accepts_part(kind):
            raise UndeclaredTurnInputError(
                'undeclared_input_part',
                f'message content part {index} is of kind {kind!r}, and {self._declared_by()} declares '
                f'`{AGENT_IO_KEY}.{INPUT_FIELD}: {list(self.agent_io.input_parts)}`. Declare the kind there (and read it in '
                f'`invoke_agent`), or stop sending it — dropping the part would let the agent answer a question it was never shown.',
            )
        if payload_key not in part:
            raise UndeclaredTurnInputError(
                'undeclared_input_part',
                f'message content part {index} declares `type: {kind}` but carries no {payload_key!r} key (its keys are '
                f'{sorted(map(str, part))}), so there is nothing of that kind on it to hand the agent.',
            )
        return part

    def _declared_by(self) -> str:
        """Where the declaration in force came from, for an error message."""
        source = self.agent_io.source
        return str(source) if source is not None else f'this harness ({CONFIG_FILENAME} declares no `{AGENT_IO_KEY}`)'

    def _build_turn_payload(self, msg: UserTurn, planted: _PlantedTurn | None = None) -> dict[str, Any]:
        """The dict handed to ``invoke_agent`` this turn.

        Its own function, called inside the dispatch ``try``: a failure building the payload (an empty
        opener failing ``HumanMessage`` validation, say) then surfaces as a typed envelope with a
        subkind and a remedy, not as a bare process exit that loses the server-side failure bookkeeping.

        Send only the NEW user message; the agent's checkpointer (keyed by the stable thread_id)
        accumulates history across turns. Seed the scenario's ``agent_context`` (startup
        preconditions) the SAME way — ONCE, on the first turn. Re-spreading the static
        agent_context on later turns would overwrite any field the agent mutated mid-conversation
        (e.g. a status field a tool changed mid-conversation), reverting it to the seed and producing
        narrative↔state incoherence. (A checkpointer-less harness loses cross-turn state
        regardless — see the simulation-setup skill's MemorySaver guidance; this keeps
        agent_context consistent with messages.)
        """
        messages = self._turn_messages(msg.user_message)
        payload: dict[str, Any] = {'messages': [*planted.messages, *messages] if planted is not None else messages}
        # Only on the turn that answers an interrupt, so every other payload is unchanged.
        if msg.interrupt_choice is not None:
            payload['interrupt_choice'] = msg.interrupt_choice
        if not self._agent_context_seeded:
            payload = InputReads(payload)
            dict.update(payload, self.initial_agent_context)
            self._input_reads.append(payload)
            self._agent_context_seeded = True
        # The scenario's channel inputs for this turn, last: a planted field is this turn's value, over a seed's.
        # Only on a turn that carries any, so every other payload is unchanged.
        if planted is not None and planted.records:
            dict.update(payload, planted.fields)
            dict.__setitem__(payload, CHANNEL_INPUTS_KEY, tuple(planted.records))
        return payload

    def _record_mock_lane(self, thread_id: str) -> bool:
        """Put a read-recorder in the in-process mock's copy of agent_context, where
        ``get_simulation_context()`` and every handler read it, and say whether that mock serves keys on
        demand: ``True`` when a registered handler can read ``ctx.agent_context`` only if the agent calls
        its tool, or when the copy is not in this process (``connect``; no session)."""
        service = mock_session._local_service if self.mock_mode != 'connect' else None
        session = getattr(service, 'sessions', {}).get(thread_id) if service is not None else None
        if not isinstance(session, dict) or not isinstance(session.get('agent_context'), dict):
            return True
        session['agent_context'] = reads = InputReads(session['agent_context'])
        self._input_reads.append(reads)
        self._mock_service = service
        return bool(getattr(service, 'mock_handlers', None))

    async def handle_init(self, msg: Init) -> InitOk | InitError:
        """Configure the SDK, start the session, run seed_state if present."""
        self.mock_mode = msg.mock_mode
        if msg.entry_witness:
            self._entry_witness = EntryWitness(msg.entry_witness, self.module)
        # The scenario's channel plan, read against this harness's own declaration. The parent refused a plan that
        # does not fit before starting this child; this is the belt, and a plan-free scenario reads no config here.
        if scenario_plants_channels(msg.scenario_metadata):
            try:
                self._channel_plan = parse_channel_plan(msg.scenario_metadata, self.agent_io, scenario_id=msg.thread_id, declared_by=self._declared_by())
            except WorkspaceConfigError as refusal:
                print(f'[plurai.child] {_exc_class(refusal)}: {refusal}', file=sys.stderr)
                return InitError(kind='other', exc_class=_exc_class(refusal), message=str(refusal))
        # Split configure from start_session so a connect-mode unreachable mock is
        # classified separately. In ``mock_mode='connect'`` a failing
        # ``plurai.configure(...)`` is a normal, actionable operational condition
        # (wrong/stale mock URL, mock not deployed, transient outage) — collapsing
        # it into kind='other' makes it indistinguishable from a real
        # start_session bug. Tag it ``mock_unreachable`` so the parent's
        # ``_raise_for_error`` can render a targeted "is your mock URL reachable?"
        # message. ``mode='local'`` / ``'mockless'`` configure failures are setup defects, not
        # reachability problems, so we only apply the dedicated kind in connect mode.
        try:
            plurai.configure(service_url=msg.mock_service_url, mode=msg.mock_mode)
        except Exception as exc:
            if msg.mock_mode == 'connect':
                traceback.print_exc(file=sys.stderr)
                return InitError(
                    kind='mock_unreachable',
                    exc_class=_exc_class(exc),
                    message=str(exc),
                )
            traceback.print_exc(file=sys.stderr)
            return InitError(
                kind='other',
                exc_class=_exc_class(exc),
                message=str(exc),
            )

        try:
            plurai.start_session(
                msg.thread_id,
                msg.initial_world_state,
                msg.initial_agent_context,
            )
            self.thread_id = msg.thread_id
            self._open_calls = ThreadCalls()
            self.initial_agent_context = dict(msg.initial_agent_context)
            self._declared_inputs = frozenset(msg.initial_agent_context)
            self._mock_on_demand = self._record_mock_lane(msg.thread_id)
        except MockServiceError as exc:
            # start_session dialed a mock that then failed a state-shaped call (e.g. a /state
            # read timing out) — the mock was reachable enough to be dialed, so this is NOT the
            # same condition ``configure``'s 'mock_unreachable' branch above covers. Typed apart
            # so the parent renders the right remedy (mock up but slow/overloaded) instead of
            # the connect-posture one.
            traceback.print_exc(file=sys.stderr)
            return InitError(
                kind='mock_init_state_fault',
                exc_class=_exc_class(exc),
                message=str(exc),
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            # The session's first call re-reads the mock's /health for its version and wraps a failed
            # read (a timeout under load) in a RuntimeError; the mock was dialed by ``configure`` above,
            # so a failure CAUSED by a MockServiceError is the same state fault as the branch above.
            return InitError(
                kind='mock_init_state_fault' if isinstance(exc.__cause__, MockServiceError) else 'other',
                exc_class=_exc_class(exc),
                message=str(exc),
            )

        if msg.mock_mode == 'connect':
            # Session start is the one moment that knows both the posture and the server's
            # version. A stub (local / mockless) is never told a turn, so it is not warned
            # about one either.
            client = _registry._client
            if client is not None and not client.supports_session_turn:
                client.warn_once_no_session_turn()

        # CLEAR, then SEED. Called on EVERY scenario, before the preconditions go in and before
        # the first turn is driven — the only ordering that makes the two hooks composable: a
        # reset after seeding would erase the scenario it was meant to prepare, and a reset that
        # ran only on "dirty" scenarios would make what a scenario starts from depend on which
        # ones ran before it, which is the property this exists to remove.
        #
        # AFTER ``start_session``: the mock session and the active session id exist by here, so a
        # harness whose store is reachable only through a mock call (or that logs the clear) can
        # do it. The mock's OWN per-session world is deliberately not touched — the backend opens a
        # new session per scenario and seeds it, so clearing it here would fight the seeding.
        #
        # Unconditional on the declaration: the hook is called whenever the harness exports it.
        # A run whose world the mock does not own and whose harness exports NO hook is refused at
        # run start instead (``orchestrator.unreset_external_world_finding``), where it costs
        # nothing — by this point a thread and an opener have already been paid for — unless the
        # operator opted into a shared world (``world_state.shared_world_ok``), which runs with a named risk.
        if self.reset is not None:
            take_harness_faults()
            try:
                # The harness preparing the world, not a state change anyone made during the
                # conversation: every world write inside is stamped 'harness' (see
                # ``turn_context.harness_setup``). The call AND its await, like the agent's scope.
                with harness_setup():
                    reset_result = self.reset()
                    if inspect.isawaitable(reset_result):
                        await reset_result
            except Exception as exc:
                if (fault := _ledgered_harness_fault(InitError, raised=exc)) is not None:
                    return fault
                if isinstance(exc, TypeError) and not _accepts_reset_contract_arity(self.reset):
                    # The hook could not be CALLED — it wants an argument the contract does not
                    # pass — so CPython's message is the whole of what the customer would see,
                    # routed through a generic InitError into a per-scenario error.json naming
                    # neither the file nor the shape to change it to. EVERY scenario then fails
                    # identically. Named here the way the seed hook's arity refusal is: the file,
                    # the required signature, and CPython's own words kept as the cause.
                    exc = TypeError(
                        f'{getattr(self.module, "__file__", "<simulate>")} defines `reset` with a signature that takes arguments. '
                        'The hook is called as `reset()` with none, once per scenario, before `seed_state` — it is asked to CLEAR the '
                        'stores this harness owns, and it is told nothing about the scenario because it is not preparing one. Change '
                        f'the signature to `def reset() -> None:`. Underlying: {exc}'
                    )
                    # The prefix the parent's ``_STDERR_WARN_RE`` promotes to WARNING; a bare
                    # ``builtins.TypeError:`` line would drop to DEBUG and be invisible.
                    print(f'[plurai.child] {_exc_class(exc)}: {exc}', file=sys.stderr)
                    return InitError(
                        kind='reset_contract',
                        exc_class=_exc_class(exc),
                        message=str(exc),
                    )
                if isinstance(exc, MockServiceError):
                    # reset() dialed a mock that then failed a state-shaped call — the mock was
                    # reachable enough to be dialed, so this is not a reset-contract shape
                    # problem or an opaque 'other'. Typed the same way the start_session site
                    # above is, for the same reason.
                    traceback.print_exc(file=sys.stderr)
                    return InitError(
                        kind='mock_init_state_fault',
                        exc_class=_exc_class(exc),
                        message=str(exc),
                    )
                traceback.print_exc(file=sys.stderr)
                return InitError(
                    kind='other',
                    exc_class=_exc_class(exc),
                    message=str(exc),
                )
            if (fault := _ledgered_harness_fault(InitError, raised=None)) is not None:
                return fault

        if msg.resume_seed is not None and self.seed_state is None:
            # A resumed scenario needs somewhere to PUT the seed. With no hook the seeding
            # block below is skipped entirely and the agent starts from an empty history,
            # while the orchestrator — which keys excision off Init.resume_seed, not off the
            # template — pre-populates its record with the seed and strips turn 1 as
            # already-said. That is a green run whose transcript reads as a conversation the
            # agent never had. Refuse at init instead, the same way a two-parameter
            # seed_state does: TypeError, kind='seed_state_contract', message carrying the fix.
            exc = TypeError(
                f'{getattr(self.module, "__file__", "<simulate>")} exports no `seed_state`, but this '
                'scenario resumes from a trace (Init.resume_seed is set). Running it would execute '
                'the agent with an EMPTY history while the run is recorded and scored as resumed. '
                'Declare `seed_state(initial_agent_context, initial_world_state, resume_seed=None)` and apply '
                'the seed to your agent, or run this scenario without a resume seed.'
            )
            # The neighbouring branches reach stderr via ``traceback.print_exc``, whose
            # header the parent's ``_STDERR_WARN_RE`` promotes to WARNING. There is no live
            # exception to print here, and a bare ``builtins.TypeError:`` line does NOT match
            # that pattern (the lowercase module prefix) — it would drop to DEBUG and be
            # invisible at default verbosity. The ``[plurai.child]`` prefix does match, and
            # is the same one the module-load failure below uses.
            print(f'[plurai.child] {_exc_class(exc)}: {exc}', file=sys.stderr)
            return InitError(
                kind='seed_state_contract',
                exc_class=_exc_class(exc),
                message=str(exc),
            )

        resume_fidelity: dict[str, Any] | None = None
        # Param 1 as a read-recorder, so the harness's mutations (schema defaults) land in it.
        seeded_context = InputReads(msg.initial_agent_context)
        self._input_reads.append(seeded_context)
        if self.seed_state is not None:
            take_harness_faults()
            try:
                # ``seed_state`` takes THREE positional arguments; ``resume_seed`` is the
                # resume seed, passed VERBATIM and ``None`` for a fresh scenario. It is
                # always passed — a template that only accepts two is refused below, by
                # name, rather than running on and quietly discarding the conversation it
                # was meant to resume from. What the seed does to the agent's history and
                # store is the customer's decision; the SDK never reads inside it. Its world
                # writes are the harness's seed, stamped 'harness' by the same scope as reset's.
                with harness_setup():
                    seed_result = self.seed_state(
                        seeded_context,
                        msg.initial_world_state,
                        msg.resume_seed,
                    )
                    if inspect.isawaitable(seed_result):
                        seed_result = await seed_result
            except Exception as exc:
                if (fault := _ledgered_harness_fault(InitError, raised=exc)) is not None:
                    return fault
                if isinstance(exc, TypeError) and not _accepts_seed_contract_arity(self.seed_state):
                    # The hook could not be CALLED — its signature is too narrow for the three
                    # arguments above — so CPython's message ("takes 2 positional arguments but 3
                    # were given") is the whole of what the customer would otherwise see, routed
                    # through a generic InitError into a per-scenario error.json that names
                    # neither the file nor the shape to change it to. Every scenario in the run
                    # then fails identically, fresh ones included, because the third argument is
                    # passed unconditionally. Named here the way the missing-`seed_state` refusal
                    # above is: the file, the required signature, and CPython's own words kept as
                    # the underlying cause. The arity is checked rather than assumed so that a
                    # TypeError raised INSIDE a correctly-shaped hook keeps its own report.
                    exc = TypeError(
                        f'{getattr(self.module, "__file__", "<simulate>")} defines `seed_state` with a signature '
                        'that cannot take the three arguments this SDK calls it with. The hook is called as '
                        '`seed_state(initial_agent_context, initial_world_state, resume_seed)` on EVERY scenario — '
                        '`resume_seed` is None for a fresh one — so a two-parameter hook fails every run, not only '
                        'a resumed one. Change the signature to '
                        '`def seed_state(initial_agent_context, initial_world_state, resume_seed=None)` and apply '
                        f'the seed to your agent when it is not None. Underlying: {exc}'
                    )
                    # Printed with the prefix the parent promotes to WARNING, for the same reason
                    # the refusals above use it: what reaches stderr should be the sentence that
                    # names the fix, not the raw traceback of the call that could not be made.
                    print(f'[plurai.child] {_exc_class(exc)}: {exc}', file=sys.stderr)
                    return InitError(
                        kind='seed_state_contract',
                        exc_class=_exc_class(exc),
                        message=str(exc),
                    )
                if isinstance(exc, RuntimeError) and is_async_run_footgun(exc):
                    traceback.print_exc(file=sys.stderr)
                    return InitError(
                        kind='adapter',
                        subkind='seed_state_async_footgun',
                        exc_class=_exc_class(exc),
                        message=str(exc),
                    )
                if isinstance(exc, MockServiceError):
                    # seed_state dialed a mock that then failed a state-shaped call — the mock
                    # was reachable enough to be dialed, so this is not a seed-contract shape
                    # problem, an async footgun, or an opaque 'other'. Typed the same way the
                    # start_session and reset sites above are, for the same reason.
                    traceback.print_exc(file=sys.stderr)
                    return InitError(
                        kind='mock_init_state_fault',
                        exc_class=_exc_class(exc),
                        message=str(exc),
                    )
                traceback.print_exc(file=sys.stderr)
                return InitError(
                    kind='other',
                    exc_class=_exc_class(exc),
                    message=str(exc),
                )
            if (fault := _ledgered_harness_fault(InitError, raised=None)) is not None:
                return fault

            # The return is the resume-fidelity report, and only a RESUMED scenario reads it: on a fresh
            # one nothing does, so a return of another shape there is noted and ignored. On a resumed one
            # a report that cannot be carried verbatim (a coerced key, an unencodable value) is replaced
            # by the SDK-namespaced marker, which every reader treats as a reduced restore.
            if seed_result is None:
                resume_fidelity = None
            elif isinstance(seed_result, dict) and all(isinstance(key, str) for key in seed_result):
                resume_fidelity = seed_result
            else:
                returned = 'a dict with non-string keys' if isinstance(seed_result, dict) else type(seed_result).__name__
                where = getattr(self.module, '__file__', '<simulate>')
                if msg.resume_seed is None:
                    print(f'[plurai.child] WARNING: {where} returned {returned} from `seed_state` on a fresh scenario; nothing reads it.', file=sys.stderr)
                    resume_fidelity = None
                else:
                    print(
                        f'[plurai.child] WARNING: {where} returned {returned} from `seed_state`; the resume-fidelity report is `None` or a '
                        'dict with string keys, so this restore is recorded as reduced.',
                        file=sys.stderr,
                    )
                    resume_fidelity = {SEED_REPORT_UNREADABLE_KEY: {'returned': returned}}

        # Re-snapshot after seed_state so in-place mutations (notably
        # _apply_schema_defaults backfilling required agent_context fields)
        # are reflected in the payload built for every subsequent user turn.
        # Without this, schema defaults applied during seed_state are
        # silently dropped — the agent sees the pre-default-fill dict and
        # tools that gate on those fields fail every turn.
        self.initial_agent_context = seeded_context.snapshot()
        # The world the agent meets, after the seed, read ONCE and whole: the backend takes every row's
        # origin against it, and its surface rows are the start of every per-turn delta.
        post_seed = await self._post_seed_read()
        if post_seed is not None and '_finalize_error' in post_seed:
            # ONE re-read. A failed post-seed read of a seeded scenario voids it (the runner does not drive it, and
            # the backend records ``undelivered_seed``), so a transient blip (a slow mock ``/state``, a dropped connection)
            # gets a second look first; a read that is broken fails the same way again.
            post_seed = await self._post_seed_read()
        if post_seed is not None and '_finalize_error' in post_seed:
            print(f'[plurai.child] the world could not be read after seed_state: {post_seed["_finalize_error"]}', file=sys.stderr)
        self._read_baseline(msg.initial_world_state, post_seed)

        self._initialized = True
        return InitOk(
            resume_fidelity=resume_fidelity,
            egress_guard_armed=egress_guard.armed_in_process(),
            user_actions=True,
            post_seed_world_state=post_seed,
            channel_inputs=True if self._channel_plan else None,
        )

    async def handle_user_turn(self, msg: UserTurn) -> AgentResponse | AgentError:
        """Apply the simulated user's act, then build the payload and call the agent."""
        if not self._initialized:
            return AgentError(
                kind='other',
                exc_class='RuntimeError',
                message='received user_turn before init',
                turn_idx=msg.turn_idx,
            )
        # The conversation turn every /mock/execute dispatch stamps on its event. The
        # child is the only place that knows it: the dispatchers (MockToolNode /
        # mock_wrap / the code action channel, the last one in a nested process) sit
        # below the agent call and would otherwise all report turn 0.
        set_current_turn(msg.turn_idx)
        # The scenario's channel inputs due at this turn's open, first: a between-turn action happened after the
        # previous turn, before anything she does now; a planted file must be in the store before the agent reads it.
        planted = await self._plant_channel_inputs(msg.turn_idx)
        # The act the simulated user took on what she saw, applied BEFORE the agent runs — so her
        # "I approved it" and the approved row reach the agent together — and before the turn's
        # fault record is cleared, which would otherwise sweep a refusal of her write away.
        action_result = await self._apply_user_action(msg.user_action) if msg.user_action is not None else None
        reply = await self._drive_user_turn(msg, planted)
        # Compared after every turn, not only at finalize: a file the agent rewrote and later put back
        # still left the turns between without the interception.
        interception_changed()
        # EVERY envelope the turn ends in carries how her act went, and what the plan planted, whichever way the
        # agent's turn ended: a result lost on an error path would read as an act nobody reported.
        if planted.records:
            reply = replace(reply, channel_results=[record.as_record() for record in planted.records])
        return reply if action_result is None else replace(reply, user_action_result=action_result)

    async def _drive_user_turn(self, msg: UserTurn, planted: _PlantedTurn | None = None) -> AgentResponse | AgentError:
        """Build the payload and call the agent."""
        try:
            payload = self._build_turn_payload(msg, planted)
        except UndeclaredTurnInputError as exc:
            # A TYPED envelope, never a child crash. The turn is undeliverable — the parent
            # offered an input this harness's ``agent.io`` does not declare — but the envelope
            # keeps the thread_id, the turn index and the remedy, and the parent degrades it to
            # a recorded empty turn instead of losing the conversation to an exit code.
            #
            # No ``traceback.print_exc``: there is a live exception, but its traceback is this
            # module's own call stack and says nothing a customer can act on, while the
            # ``[plurai.child]`` prefix is what the parent's ``_STDERR_WARN_RE`` promotes to
            # WARNING (the same shape the seed-contract refusals above use).
            print(f'[plurai.child] {_exc_class(exc)}: {exc}', file=sys.stderr)
            return AgentError(
                kind='adapter',
                subkind=exc.subkind,
                exc_class=_exc_class(exc),
                message=str(exc),
                turn_idx=msg.turn_idx,
            )
        except Exception as exc:
            # The belt for everything the projection above does not name — a malformed
            # ``agent.io`` that cannot testify, a message shape ``HumanMessage`` rejects. It
            # exists so the INVARIANT is structural rather than a list: no exception from
            # building this payload may escape the dispatch loop and kill the child.
            traceback.print_exc(file=sys.stderr)
            return AgentError(
                kind='other',
                exc_class=_exc_class(exc),
                message=str(exc),
                turn_idx=msg.turn_idx,
            )
        config = {'configurable': {'thread_id': self.thread_id}}

        # The turn index, pushed to the service for the callers this process cannot
        # reach. A stub mock (local / mockless) is dialled by nobody else, so it is
        # told nothing. The push carries the dialog THROUGH this user message.
        if self.mock_mode == 'connect':
            self._record_user(msg)
            dialog, live_tool_calls = self._turn_context()
            _push_turn_to_mock(msg.turn_idx, dialog=dialog, live_tool_calls=live_tool_calls)

        take_harness_faults()
        take_cli_stops()
        try:
            if self.invoke_agent is None:
                raise RuntimeError('Received user_turn but no invoke_agent is configured on the simulate.py module.')
            # Sync AND async invoke_agent are both valid: the setup skill and the
            # shipped example document the sync form (promising awaitable detection),
            # and seed_state above already gets exactly this treatment.
            #
            # The agent-invocation scope covers the call AND its await: an async agent's body
            # runs inside the await, and every world write it (or a tool it dispatches) makes
            # is stamped as the agent's — see ``turn_context.agent_invocation``.
            with agent_invocation(), self._witnessed_turn():
                result = self.invoke_agent(payload, config=config)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as exc:
            if (fault := _ledgered_harness_fault(AgentError, raised=exc, turn_idx=msg.turn_idx)) is not None:
                return fault
            # Order matters: ``langgraph.errors.GraphRecursionError`` IS a
            # RuntimeError subclass, so checking the recursion fingerprint
            # first prevents the async-footgun branch from swallowing it.
            traceback.print_exc(file=sys.stderr)
            if is_recursion_limit_error(exc):
                return AgentError(
                    kind='adapter',
                    subkind='graph_recursion',
                    exc_class=_exc_class(exc),
                    message=str(exc),
                    turn_idx=msg.turn_idx,
                )
            if isinstance(exc, RuntimeError) and is_async_run_footgun(exc):
                return AgentError(
                    kind='adapter',
                    subkind='invoke_async_footgun',
                    exc_class=_exc_class(exc),
                    message=str(exc),
                    turn_idx=msg.turn_idx,
                )
            stops = take_cli_stops()  # the turn's LAST stop is the one its raise ended on
            return AgentError(
                kind=_invoke_raise_kind(exc, harness_file=getattr(self.module, '__file__', None), agent_stop=bool(stops) and stops[-1].agent_owned),
                exc_class=_exc_class(exc),
                message=str(exc),
                turn_idx=msg.turn_idx,
            )
        # How the turn's LAST CLI query ended, read once its end has arrived (the CLI fires StopFailure
        # without awaiting it). On a turn that returned, that end settles the CLI's own provider notes: a
        # retry that ended any other way supersedes an earlier provider stop.
        turn_end = await _await_cli_turn_end()
        # every infrastructure banner the CLI handed over this turn — the CLI's words, never the agent's
        infra_banners = [stop.last_assistant_message for stop in take_cli_stops() if not stop.ok and stop.error in INFRA_STOP_CLASSES]
        cli_notes = take_harness_faults_raised_by(STOP_FAILURE_SOURCE) if turn_end.settled else []
        infra = INFRA_STOP_CLASSES.get(turn_end.end.error) if turn_end.end is not None and not turn_end.end.ok else None

        def _typed_as_infrastructure() -> AgentError | None:
            for note in cli_notes:  # the whole turn is the provider's: its notes type it
                note_harness_fault(note.exc_class, note.message, kind=note.kind, transient=note.transient, route_cause=note.route_cause)
            return _ledgered_harness_fault(AgentError, raised=None, turn_idx=msg.turn_idx)

        if infra is None:
            cli_notes = []  # superseded by how the turn ended
        # A harness fault the agent's own code caught (a refused world write, a swallowed
        # ``plurai.HarnessFault``) was recorded where it was raised; the turn is the harness's.
        if (fault := _ledgered_harness_fault(AgentError, raised=None, turn_idx=msg.turn_idx)) is not None:
            return fault
        try:
            response = self._with_witness(_build_agent_response(result, open_calls=self._open_calls))
        except Exception as exc:
            if infra is not None and (fault := _typed_as_infrastructure()) is not None:
                return fault
            # A typed envelope, never a child crash: unhandled, this raise would escape the
            # dispatch loop, exit the process with code 1, and reach the parent as a bare
            # 'child exited before responding' RuntimeError — losing the explanation AND the
            # server-side failure bookkeeping (no thread_id on the generic error).
            #
            # EVERY exception, not just the one this module declares: the projection READS the
            # chain (``turn_actions`` → ``derive_tool_calls_from_messages``) over whatever shape
            # the customer's adapter returns, and any of those readers can raise; a subkind
            # outside ``FATAL_ADAPTER_SUBKINDS`` degrades to a scoreable turn either way.
            #
            # The RESULT rides along. It is what the parent's ``_scoreable_empty_turn`` builds
            # the degraded turn from; without it that turn is an ``AgentResponse`` made out of
            # nothing, and a turn the agent really did run reaches transcript.json with
            # ``turn_messages: []`` and ``agent_state: {}`` — the trail dropped at the one
            # boundary where it can still be kept, and with it the only surface that can say
            # what the agent did.
            if not isinstance(exc, NoAssistantReplyError):
                traceback.print_exc(file=sys.stderr)
            full_messages, agent_state = _error_trail(result)
            return AgentError(
                kind='adapter',
                subkind='no_assistant_reply' if isinstance(exc, NoAssistantReplyError) else 'reply_projection_failed',
                exc_class=_exc_class(exc),
                message=str(exc),
                turn_idx=msg.turn_idx,
                full_messages=full_messages,
                agent_state=agent_state,
                # The agent ran, so the world may have changed: the parent scores this turn.
                turn_world=await self._turn_world(),
            )
        # An infrastructure banner is the CLI's text, and the only text ever removed: what the agent said or did
        # stays its reply. Infrastructure that ENDED the turn is typed on the side; a turn it left with nothing of
        # the agent's is the provider's whole turn.
        response = _without_cli_banners(response, infra_banners)
        if infra is not None:
            if not response.ai_text.strip() and not response.agent_actions and (fault := _typed_as_infrastructure()) is not None:
                return fault
            fault_class, terminal = infra
            marker = provider_turn_fault_marker(fault_class, source=STOP_FAILURE_SOURCE, terminal=terminal)
            response = replace(response, agent_state={**response.agent_state, PROVIDER_TURN_FAULT_KEY: marker})
        elif turn_end.unobserved:
            unobserved = {'waited_s': _CLI_TURN_END_WAIT_S}
            response = replace(response, agent_state={**response.agent_state, TURN_END_UNOBSERVED_KEY: unobserved})
        if self.mock_mode == 'connect':
            self._record_reply(msg.turn_idx, response)
        # What the simulated user can see now: the world at the END of this turn, as a delta. A read,
        # outside the agent's scope, so it writes nothing and is charged to nobody.
        return replace(response, turn_world=await self._turn_world())

    async def handle_pipeline_invoke(
        self,
        msg: PipelineInvoke,
    ) -> PipelineResult | PipelineError:
        """Drive a one-shot pipeline call. No turn loop, no recursion budget.

        Returns ``PipelineError`` (not ``AgentError``) so the parent's
        recursion-error classifier never fires for one-shot — the dispatch
        semantics are simpler: handler-raise / non-dict-return / timeout
        (timeout is parent-enforced via wall-clock; this method just routes
        the call).
        """
        if self.invoke_pipeline is None:
            return PipelineError(
                exc_class='AttributeError',
                message=PIPELINE_NO_HOOK_MESSAGE,
                kind='harness_fault',
            )
        if not self._initialized:
            return PipelineError(
                exc_class='RuntimeError',
                message=PIPELINE_NOT_INITIALIZED_MESSAGE,
                kind='harness_fault',
            )
        t0 = time.monotonic()
        take_harness_faults()
        try:
            # Same sync-or-async contract as invoke_agent — the resolver's own
            # docstring promises "(sync or async)". Same agent-invocation scope, too: a pipeline
            # that writes the world is the agent writing it.
            inputs = InputReads(msg.inputs)
            self._input_reads.append(inputs)
            with agent_invocation(), self._witnessed_turn():
                result = self.invoke_pipeline(inputs)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as exc:
            if (fault := _ledgered_harness_fault(PipelineError, raised=exc)) is not None:
                return fault
            traceback.print_exc(file=sys.stderr)
            return PipelineError(
                exc_class=_exc_class(exc),
                message=str(exc),
            )
        if (fault := _ledgered_harness_fault(PipelineError, raised=None)) is not None:
            return fault
        if not isinstance(result, dict):
            return PipelineError(
                exc_class='TypeError',
                message=(f'invoke_pipeline must return dict, got {type(result).__name__}'),
            )
        coerced: dict[str, str] = {}
        try:
            outputs = _json_shaped(result, coerced=coerced)
        except Exception as exc:
            # STRUCTURAL, not a list of classes — the same belt as the user-turn path, and for
            # the same reason: no exception raised while BUILDING this payload
            # may escape the dispatch loop and kill the child. ``_json_shaped`` names what it can
            # as a ``TypeError``, but it walks CUSTOMER objects, and a customer object can raise
            # anything on the way through (a model whose ``model_dump`` raises ``ValueError``; a
            # self-referential dict raising ``RecursionError``). A hop whose outputs cannot be
            # serialized is a SCORED outcome; a dead child is an untyped ``HarnessChildExited``
            # with no subkind and no remedy in it.
            return PipelineError(
                exc_class='SerializationError',
                message=f'invoke_pipeline returned a value that is not JSON-serializable: {exc}',
            )
        return self._with_witness(
            PipelineResult(
                outputs=outputs,
                duration_s=time.monotonic() - t0,
                values_coerced=coerced or None,
            )
        )

    def _witnessed_turn(self) -> AbstractContextManager[None]:
        return nullcontext() if self._entry_witness is None else self._entry_witness.turn()

    def _with_witness[R: (AgentResponse, PipelineResult)](self, reply: R) -> R:
        return reply if self._entry_witness is None else replace(reply, entry_witness=self._entry_witness.report())

    async def handle_finalize(self, _msg: Finalize) -> FinalizeResponse | AgentError:
        """Capture the final world state + recorded tool events for the parent.

        Best-effort: a successful N-turn dialog has already happened by the
        time we get here, so a failure pulling state out of the SDK
        (corrupted entity, unreachable mock-env, ...) must NOT discard the
        transcript. We try each call independently, fall back to a dict carrying
        an ``_finalize_error`` marker the caller can surface, and always
        return a well-formed envelope.

        A read that FAILED and a read that never happened are different answers and get
        different values: the marker above says "the snapshot was attempted and did not
        materialize" (the parent renders it as a finalize error), whereas a child with no
        session returns ``None`` on both channels and claims nothing at all. Neither is a
        ``{}``, which would say the channel was read and was empty.

        WHICH world the world channel carries is decided by the table in the body: the harness's
        own :data:`~plurai.orchestrator.orchestrator.WORLD_SNAPSHOT_HOOK` when it exports one, the
        mock's session when it does not and the mock owns the world, and nothing at all when it
        does not and something else does.

        Also hands over the session's unanswered tool calls (``unanswered_mock_calls``) so the
        parent can record which calls the agent was served a stub for, and why.
        """
        from plurai._mocking.resolution import take_unanswered_calls  # noqa: PLC0415

        if self._entry_witness is not None:
            self._entry_witness.close()

        # ONE rule for BOTH channels: ``None`` until the channel is actually read, and it STAYS
        # ``None`` when there is no session to read. This child took no snapshot, so it says
        # nothing rather than reporting an empty world or an empty trail it never looked at.
        # A dict — ``{}``, ``{'events': []}``, ``{'state': {}}`` — is the answer of a run that
        # DID look; the two are different claims and the server's judges route on the difference
        # (an absent channel demotes its policy to not-measured instead of failing the agent for
        # "invoking zero tools", or passing it for "creating no booking" in a store nobody read).
        # The gate is the DECLARED one below — was there a session — never an inference from
        # what came back.
        world: dict[str, Any] | None = None
        events: dict[str, Any] | None = None
        if self.thread_id:
            thread_id = self.thread_id
            # WHICH world: the table in :meth:`_read_world`, the one every per-turn capture reads too.
            world = await self._read_world(at_finalize=True)
            events = await self._capture(lambda: plurai.get_events(thread_id), 'events', at_finalize=True)
            if not self._not_valid_reported and (fault := _unregistered_tool_fault(events)) is not None:
                return fault
        # Every call the mock did not answer, typed (status, code, transport, what the agent was
        # served), so the runner can record a finding and re-drive only a restart-shaped one. A
        # bare log line on the child's stderr would be invisible to the run's own record. ``None``
        # with no session: this child claims nothing.
        unanswered = [call._asdict() for call in take_unanswered_calls(self.thread_id)] if self.thread_id else None
        return FinalizeResponse(
            final_world_state=world,
            tool_events=events,
            unanswered_mock_calls=unanswered,
            capture_retryable=True if self._capture_retryable else None,
            world_values_coerced=self._world_values_coerced or None,
            unread_inputs=self._unread_record(),
            unread_parts=self._unread_parts_record(),
            interception_changed=interception_changed() or None,
        )

    def answered[R](self, reply: R) -> R:
        """``reply``, noting whether it typed the scenario not_valid (the parent raises on it and finalizes for evidence only)."""
        self._not_valid_reported |= getattr(reply, 'kind', None) in ('mock_runtime_fault', 'harness_fault', PROVIDER_INFRA_KIND)
        return reply

    def _unread_record(self) -> list[str] | None:
        """The declared keys no lane read; ``[]`` when every key was read somewhere. ``None`` (claims
        nothing) before Init, and when an unread key could still reach the agent through an on-demand mock —
        except an unread bound document, which is named whatever the mock is."""
        if self._declared_inputs is None:
            return None
        unread = unread_inputs(self._declared_inputs, self._input_reads)
        on_demand = self._mock_on_demand or bool(getattr(self._mock_service, 'mock_handlers', None))
        if not (unread and on_demand):
            return unread
        # A document the backend bound for the user to hand over reaches the agent only through the harness:
        # no mock serves it on demand, so its non-read is a fact whatever the mock is.
        documents = [k for k in unread if is_bound_document(self.initial_agent_context.get(k))]
        return documents or None

    def _unread_parts_record(self) -> list[str] | None:
        """The handed-over ``file`` parts whose ``file_data`` the harness never read, by field (``<field>.<ext>`` →
        ``field``); ``[]`` when every part was read, ``None`` before Init. No mock serves a part on demand, so a
        non-read is a fact whatever the mock is. Kept apart from :meth:`_unread_record`: a part is not a key."""
        if self._declared_inputs is None:
            return None
        return sorted({PurePosixPath(name).stem or name for name, spec in self._handed_parts if 'file_data' not in spec.read})

    def shutdown(self) -> None:
        """Reset the active session id, turn context and unanswered-call record so a re-used interpreter doesn't leak."""
        from plurai._mocking.resolution import take_unanswered_calls  # noqa: PLC0415

        if self.thread_id:
            take_unanswered_calls(self.thread_id)
        # A stale turn would otherwise outlive the conversation — both in-process and
        # in the env any later child-spawned process inherits.
        set_current_turn(None)
        _set_active_session_id('')


#: The model-runtime SDK's failures (the CLI missing, its connection or process dying, its output undecodable, an
#: error result): the runtime's, never the agent's, whatever status or stderr text they carry. Matched on the
#: raised class's MRO by qualified name, so a runtime the child never imported still reads the same.
#: ``MessageParseError`` and ``CLIJSONDecodeError`` are the agent's own client refusing what the CLI sent (a message
#: type it cannot parse, a message over its own ``max_buffer_size``) and are left to the raise site — unless the
#: refusal's typed ``original_error`` is a ``json.JSONDecodeError``: the CLI wrote a line no parser could decode.
_RUNTIME_SDK_FAILURE: str = 'claude_agent_sdk._errors.ClaudeSDKError'
_RUNTIME_SDK_UNOWNED: frozenset[str] = frozenset({'claude_agent_sdk._errors.MessageParseError', 'claude_agent_sdk._errors.CLIJSONDecodeError'})
#: The runtime's error RESULT (its turn ended ``is_error``), which is not by itself a runtime failure: it is the
#: provider's only on a provider signal in its typed data (:func:`_provider_result`), and otherwise read at its raise site.
_RUNTIME_SDK_RESULT: str = 'claude_agent_sdk._errors.ResultError'

#: The HTTP client TRANSPORT failure classes, by module (a timeout, a dropped, refused or unresolved connection,
#: a peer that closed mid-response). Some carry no stdlib cause. A client's misuse errors are not here.
_TRANSPORT_FAILURES: Mapping[str, tuple[str, ...]] = {
    'httpx': ('TimeoutException', 'NetworkError', 'RemoteProtocolError'),
    'httpx2': ('TimeoutException', 'NetworkError', 'RemoteProtocolError'),
    'requests.exceptions': ('ConnectionError', 'Timeout'),
    'aiohttp': ('ClientConnectionError',),
}

#: The HTTP clients' own STATUS errors, by module: the status the CALLER's request got back (its own
#: ``raise_for_status``; urllib raising on any non-2xx). Read like a bare ``response.status_code``: only a 429/5xx is
#: the far end failing, and a 401/403/404 is the caller's own call, never a provider refusing it.
_CLIENT_STATUS_ERRORS: Mapping[str, tuple[str, ...]] = {'aiohttp': ('ClientResponseError',), 'urllib.error': ('HTTPError',)}

#: Code that is never the AGENT's own (besides the harness directory): plurai's runner, mocks and helpers.
_NOT_AGENT_CODE: frozenset[str] = frozenset({'plurai', 'plurai_mockenv'})

#: The provider refusing the agent's credentials or model (an expired key, no permission, a model not found).
#: Read off the provider exception's own typed status only, never a bare ``response.status_code``: that is also
#: where the agent's own HTTP call to a tool (``raise_for_status``) carries a 404, which stays the agent's.
_PROVIDER_REFUSED_STATUSES: frozenset[int] = frozenset({401, 403, 404})

#: The provider's own spend-cap sentences, verbatim: it answers a spend cap as a 400, so its status cannot tell it
#: from a malformed request. Read off the provider exception's error message, or the model runtime's 400 result text
#: (the CLI's own spellings), never an agent reply (a billing agent says these).
_SPEND_CAP_ERROR = re.compile(
    r'^\W*(?:api error: 400\s+)?(?:(?:your )?credit balance is too low\b|you have reached your specified (?:workspace )?api usage limits\b)',
    re.IGNORECASE,
)


#: A channel kind → the session method that plants an entry of it (declare and dispatch). Keyed on the scan
#: contract's :data:`~plurai.scan_contract.input_channels.ChannelKind`, the one definition of the vocabulary.
_CHANNEL_PLANTERS: Final[Mapping[ChannelKind, Callable[[_ChildSession, PlannedChannelInput, _PlantedTurn], Awaitable[AppliedChannelInput]]]] = {
    'file_by_reference': _ChildSession._plant_file_by_reference,
    'structured_context': _ChildSession._plant_structured_context,
    'host_injected_message': _ChildSession._plant_host_injected_message,
    'between_turn_action': _ChildSession._plant_between_turn_action,
}


def _is_status(status: object) -> TypeGuard[int]:
    return isinstance(status, int) and not isinstance(status, bool)


def _declared_classes(declared: Mapping[str, tuple[str, ...]]) -> tuple[type[BaseException], ...]:
    """The ``declared`` classes of the clients the agent has loaded (a client never imported cannot be in the
    chain): read by class from ``sys.modules``, never imported on the crash path."""
    return tuple(
        cls
        for module_name, names in declared.items()
        if (module := sys.modules.get(module_name)) is not None
        for name in names
        if isinstance(cls := getattr(module, name, None), type) and issubclass(cls, BaseException)
    )


def _lineage(err: BaseException) -> set[str]:
    return {f'{cls.__module__}.{cls.__qualname__}' for cls in type(err).__mro__}


def _runtime_sdk_failure(err: BaseException) -> bool:
    lineage = _lineage(err)
    undecodable = isinstance(getattr(err, 'original_error', None), json.JSONDecodeError)
    return _RUNTIME_SDK_FAILURE in lineage and (lineage.isdisjoint(_RUNTIME_SDK_UNOWNED) or undecodable)


def _provider_result(err: BaseException) -> bool:
    """Whether a runtime error RESULT carries a provider signal: a 401/403/404/429/5xx ``api_error_status``, or a 400
    whose result is the provider's spend-cap sentence (:data:`_SPEND_CAP_ERROR`). A result with no status carries
    none: the CLI types that stop only on its ``StopFailure`` hook (:func:`plurai.cli_hooks`), read off the ledger."""
    status = getattr(err, 'api_error_status', None)
    if _is_status(status):
        if status == 400:  # a spend cap is a 400: the CLI's result text is the provider's own sentence
            text = getattr(err, 'result', None)
            return isinstance(text, str) and bool(_SPEND_CAP_ERROR.match(text))
        return status in (401, 403, 404, 429) or 500 <= status <= 599
    return False


def _provider_unavailable(exc: BaseException) -> bool:
    """Whether the agent's PROVIDER (or its model runtime) refused or failed its call: never the agent's outcome.

    Read off the raise chain (``__cause__`` / ``__context__`` / an exception group's members), from
    the TYPED status the provider SDKs put on their exceptions: a 429 or any 5xx in ``status_code``,
    ``api_error_status``, ``code``, ``response.status_code`` or a ``ResponseMetadata.HTTPStatusCode``; a
    401/403/404 in any of those but ``response.status_code``, on anything but a client's own status error
    (:data:`_PROVIDER_REFUSED_STATUSES`, :data:`_CLIENT_STATUS_ERRORS`); a stdlib ``TimeoutError`` /
    ``ConnectionError`` / ``socket.gaierror`` or an HTTP client's transport failure; a model-runtime SDK failure
    (:data:`_RUNTIME_SDK_FAILURE`); an error RESULT with no provider signal (a run budget included), and the exit it
    caused, are skipped and read at the raise site.
    The provider's typed error message is matched by the ONE capacity recognizer
    (:func:`plurai.run_simulation._is_provider_capacity_reply`) and :data:`_SPEND_CAP_ERROR`.
    """
    from plurai.run_simulation import _is_provider_capacity_reply  # noqa: PLC0415  # the crash path only

    transport = _declared_classes(_TRANSPORT_FAILURES)
    client_status = _declared_classes(_CLIENT_STATUS_ERRORS)
    seen: set[int] = set()
    todo: list[BaseException] = [exc]
    while todo:
        err = todo.pop()
        if id(err) in seen:
            continue
        seen.add(id(err))
        if _RUNTIME_SDK_RESULT in _lineage(err) and not _provider_result(err):
            continue  # an error result with no provider signal (a run budget included): neither it nor the exit it caused
        response = getattr(err, 'response', None)
        typed = (
            getattr(err, 'status_code', None),
            getattr(err, 'api_error_status', None),
            getattr(err, 'code', None),
            (response.get('ResponseMetadata') or {}).get('HTTPStatusCode') if isinstance(response, dict) else None,
        )
        statuses = (*typed, getattr(response, 'status_code', None))
        if any(_is_status(status) and (status == 429 or 500 <= status <= 599) for status in statuses):
            return True
        if not isinstance(err, client_status) and any(_is_status(status) and status in _PROVIDER_REFUSED_STATUSES for status in typed):
            return True
        if isinstance(err, (TimeoutError, ConnectionError, socket.gaierror, *transport)) or _runtime_sdk_failure(err):
            return True
        body = getattr(err, 'body', None)
        detail = body.get('error', body) if isinstance(body, dict) else None
        message = detail.get('message') if isinstance(detail, dict) else None
        if isinstance(message, str) and (_is_provider_capacity_reply(message) or _SPEND_CAP_ERROR.match(message)):
            return True
        todo.extend(link for link in (err.__cause__, err.__context__, *(getattr(err, 'exceptions', None) or ())) if isinstance(link, BaseException))
    return False


def _raised_in_agent_code(exc: BaseException, *, harness_file: str | None) -> bool:
    """Whether the raise is the AGENT's own code: read at its innermost frame outside the stdlib and the installed
    libraries (a library raises on its caller's behalf, the agent's own ``raise_for_status`` included), which must
    not be :data:`_NOT_AGENT_CODE`. A frame with no module name to read is not positively the agent's.

    The harness (``simulate.py`` and the helpers beside it) owns the raise when its frame is the innermost frame
    outside the stdlib: it raised itself, or through a stdlib call it made (``json.loads``, ``Path.read_text``). A
    harness frame the raise only passed through on its way out of an installed library (a wrapper the harness put
    around the agent's own object, delegating to it) is read through: the frame out of it decides. With no agent
    frame further out (a harness that calls the agent's prebuilt graph itself) the raise stays the harness's.
    Residual: harness code the agent calls directly, outside ``plurai.mock*``, that fails inside an installed library
    reads as the agent's — the frames cannot tell such a stub from a pass-through."""
    harness_dir = str(Path(harness_file).resolve().parent) if harness_file is not None else None
    installed = _installed_library_roots()
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    raise_site = True  # no frame outside the stdlib read yet
    for frame in reversed(frames):
        spec = frame.f_globals.get('__spec__')
        module = getattr(spec, 'name', None) or frame.f_globals.get('__name__')
        if not isinstance(module, str) or not module or module == '__main__':
            return False
        if module.partition('.')[0] in sys.stdlib_module_names:
            continue
        if any(module == owner or module.startswith(f'{owner}.') for owner in _NOT_AGENT_CODE):
            return False
        path = os.path.realpath(frame.f_code.co_filename)
        if harness_dir is not None and path.startswith(harness_dir + os.sep):
            if raise_site:
                return False
            continue
        raise_site = False
        if any(path.startswith(root + os.sep) for root in installed):
            continue
        return True
    return False


def _installed_library_roots() -> tuple[str, ...]:
    """Where this interpreter's installed libraries live (its site-packages)."""
    import site  # noqa: PLC0415  # the crash path only
    import sysconfig  # noqa: PLC0415

    return tuple({os.path.realpath(root) for root in (sysconfig.get_paths()['purelib'], sysconfig.get_paths()['platlib'], *site.getsitepackages())})


def _invoke_raise_kind(exc: BaseException, *, harness_file: str | None, agent_stop: bool) -> ErrorKind:
    """The owner of a raise inside ``invoke_agent``: ``'provider_infra'`` when its provider, transport or model
    runtime failed the call; ``'agent_raised'`` (a failed trial the agent owns) when it was raised in the agent's
    own code, or is the runtime's error result on a turn whose last CLI stop the agent owns (``agent_stop``: its
    model's reply hit its output budget), or is a :class:`plurai.AgentFault` a harness raised on the agent's behalf
    (read on the raise itself, never its chain); anything else (a harness bug, plurai) stays ``'other'``. Both but
    ``agent_raised`` are errored."""
    if _provider_unavailable(exc):
        return PROVIDER_INFRA_KIND
    if isinstance(exc, AgentFault):
        return AGENT_RAISED_KIND
    if agent_stop and _RUNTIME_SDK_RESULT in _lineage(exc):
        return AGENT_RAISED_KIND
    return AGENT_RAISED_KIND if _raised_in_agent_code(exc, harness_file=harness_file) else 'other'


def _exc_class(exc: BaseException) -> str:
    """Fully-qualified exception class name (``module.ClassName``)."""
    cls = type(exc)
    return f'{cls.__module__}.{cls.__name__}'


#: How long the child waits, after a turn returned, for the turn's last CLI query to report its end. The CLI
#: fires ``StopFailure`` without awaiting it, and the hook normally lands BEFORE the result,
#: so the wait is normally nil. A query its step cap ended has no end hook, and is read off the CLI's transcript.
_CLI_TURN_END_WAIT_S = 2.0

#: How much of the CLI's session transcript is read, from its end, to find a step-capped query's trace.
_TRANSCRIPT_TAIL_BYTES = 512 * 1024


@dataclass(frozen=True, slots=True)
class _CliTurnEnd:
    """How the turn's last CLI query ended. ``end``: its recorded end (``None``: none recorded). ``settled``:
    that ending is known — recorded, or the step cap's — so it decides the turn over any earlier query's
    provider stop. ``unobserved``: the query started and neither its end nor a step cap was seen in the wait."""

    end: CliStop | None
    settled: bool
    unobserved: bool = False


def _step_capped(start: CliQueryStart) -> bool:
    """Whether the CLI's own transcript shows ``start``'s query ended on its step cap: a ``max_turns_reached``
    attachment after that query's last row. The CLI fires no end hook for that ending."""
    if not start.transcript_path or not start.prompt_id:
        return False
    try:
        with open(start.transcript_path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - _TRANSCRIPT_TAIL_BYTES))
            lines = fh.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return False
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # the partial first line of the tail, or a row still being written
        if isinstance(row, dict):
            rows.append(row)
    last = max((i for i, row in enumerate(rows) if row.get('promptId') == start.prompt_id), default=None)
    if last is None:
        return False
    return any(row.get('type') == 'attachment' and (row.get('attachment') or {}).get('type') == 'max_turns_reached' for row in rows[last + 1 :])


async def _await_cli_turn_end() -> _CliTurnEnd:
    """How the turn's LAST CLI query ended, waiting (bounded) for it — paired by the query's ``prompt_id``.

    No query started (a harness that drives no Claude Code CLI, or installs none of :func:`plurai.cli_hooks`):
    the last end recorded, if any, without waiting — and with none, nothing is settled. Yields first so a hook
    already dispatched on this loop runs."""
    await asyncio.sleep(0)
    start = last_cli_query()
    if start is None:
        stops = peek_cli_stops()
        return _CliTurnEnd(stops[-1], settled=True) if stops else _CliTurnEnd(None, settled=False)
    deadline = time.monotonic() + _CLI_TURN_END_WAIT_S
    # a sync harness runs its CLI session's hooks on its own thread and loop: no asyncio.Event spans both
    while True:  # noqa: ASYNC110
        if (end := cli_query_end(start.prompt_id)) is not None:
            return _CliTurnEnd(end, settled=True)
        if _step_capped(start):
            return _CliTurnEnd(None, settled=True)  # the agent's own step cap ended it
        if time.monotonic() >= deadline:
            return _CliTurnEnd(None, settled=False, unobserved=True)
        await asyncio.sleep(0.02)


def _without_cli_banners(response: AgentResponse, banners: Sequence[str]) -> AgentResponse:
    """``response`` with each of the CLI's own ``banners`` (exactly the text it handed ``StopFailure``) cut from this
    turn's reply and message chain. Nothing else is touched: every other word stays the agent's."""
    banners = [b for b in dict.fromkeys(banners) if b]
    if not banners:
        return response

    def _has(text: str) -> bool:
        return any(banner in text for banner in banners)

    def _cut(text: str) -> str:
        if not _has(text):
            return text
        for banner in banners:
            text = text.replace(banner, '')
        return text.strip()

    messages = list(response.full_messages)
    for i in range(len(messages) - len(_turn_suffix(messages)), len(messages)):
        content = messages[i].get('content')
        if isinstance(content, str) and _has(content):
            messages[i] = {**messages[i], 'content': _cut(content)}
        elif isinstance(content, list) and any(isinstance(b, dict) and b.get('type') == 'text' and _has(str(b.get('text'))) for b in content):
            blocks = [{**b, 'text': _cut(str(b.get('text')))} if isinstance(b, dict) and b.get('type') == 'text' else b for b in content]
            messages[i] = {**messages[i], 'content': [b for b in blocks if not (isinstance(b, dict) and b.get('type') == 'text' and not b['text'])]}
    return replace(response, ai_text=_cut(response.ai_text), full_messages=messages)


def _unregistered_tool_fault(events: dict[str, Any] | None) -> AgentError | None:
    """The mock runtime fault for a tool the deployed mock refused as unregistered, read off the session's event log.

    A process the agent spawns (a Node hook, a nested Claude Code) calls ``/mock/execute`` itself, so its
    refusal never reaches this process's fault record (``resolution`` notes the in-process ones). The
    mock logs every call, so the refusal is in the log: recorded here exactly as ``resolution`` records
    it, and the scenario is not_valid. ``None`` when no logged call was refused.
    """
    rows = events.get('events') if isinstance(events, dict) else None
    refused = [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get('response'), dict) and row['response'].get(ENGINE_REFUSAL_KEY) is True]
    if not refused:
        return None
    for row in refused:
        note_harness_fault(
            'plurai.UnregisteredTool',
            f'mocked tool {row.get("tool")!r} has no handler in the deployed mock; the scenario did not measure the agent on it',
            kind='mock_runtime_fault',
            transient=False,
        )
    turn = refused[0].get('turn')
    return _ledgered_harness_fault(AgentError, raised=None, turn_idx=turn if isinstance(turn, int) else 0)


def _ledgered_harness_fault[E: (InitError, AgentError, PipelineError)](envelope: type[E], *, raised: BaseException | None, **fields: Any) -> E | None:
    """The hook's harness-fault envelope when a harness fault was recorded during it, else ``None``.

    The one reader of a harness fault, whether the hook raised it, returned, or an agent tool
    caught it on the way: every fault is recorded at its raise site (``turn_context.note_harness_fault``).
    ``raised`` is what the hook raised (``None`` when it returned): a ``HarnessFault`` whose
    subclass ``__init__`` never reached the base one recorded nothing, and is still the harness's.

    ``kind='mock_runtime_fault'`` when EVERY recorded fault is the mock runtime's (``transient`` when all
    of them are) or any of them is a credit, capacity or unavailable refusal; otherwise any real harness fault in the hook keeps
    ``kind='harness_fault'``; a provider failure otherwise makes the turn ``kind='provider_infra'``:
    infrastructure, typically not the agent's. This is checked (and, if it matches, returned) BEFORE any
    "last stop" fallback, so a provider failure recorded earlier in the turn is not cleared by a later
    agent-owned stop (e.g. a retry that then hit the model's own output budget).
    """
    faults = take_harness_faults()
    if not faults and isinstance(raised, HarnessFault):
        faults = [HarnessFaultNote(_exc_class(raised), str(raised), 'harness_fault')]
    if not faults:
        return None
    message = '; '.join(dict.fromkeys(f.message for f in faults))
    print(f'[plurai.child] {faults[0].exc_class}: {message}', file=sys.stderr)
    # A refusal for credit or for the backend's capacity stops the run, and an unavailable backend fails the scenario for a
    # re-drive, whatever else went wrong in the turn, so each is read first and wins over a harness fault that shares the
    # turn; in the order the run reads them.
    refused = (
        next((f for f in faults if f.credit is not None), None)
        or next((f for f in faults if f.capacity is not None), None)
        or next((f for f in faults if f.unavailable is not None), None)
    )
    if refused is not None or all(f.kind == 'mock_runtime_fault' for f in faults):
        if refused is None:
            # The first fault a re-run would NOT answer, when there is one.
            route_cause = next((f.route_cause for f in faults if not f.transient), faults[0].route_cause)
            credit = capacity = unavailable = None
        else:
            route_cause = refusal_cause(refused.credit, refused.capacity, refused.unavailable)
            credit, capacity, unavailable = (
                refused.credit if route_cause == 'credit' else None,
                refused.capacity if route_cause == 'capacity' else None,
                refused.unavailable if route_cause == 'unavailable' else None,
            )
        return envelope(
            kind='mock_runtime_fault',
            exc_class=(refused or faults[0]).exc_class,
            message=message,
            transient=all(f.transient for f in faults) and refused is None,
            route_cause=route_cause,
            credit=credit,
            capacity=capacity,
            unavailable=unavailable,
            **fields,
        )
    if any(f.kind == 'harness_fault' for f in faults):
        return envelope(kind='harness_fault', exc_class=faults[0].exc_class, message=message, **fields)
    provider = next(f for f in faults if f.kind == PROVIDER_INFRA_KIND)
    return envelope(kind=PROVIDER_INFRA_KIND, exc_class=provider.exc_class, message=message, **fields)


def _as_crash_marker_if_error(result: Any, success_key: str) -> dict[str, Any]:
    """Convert a ``MockServiceClient`` error-envelope into a ``_finalize_error``
    crash marker.

    ``MockServiceClient._get`` does NOT raise on a dead/unreachable mock-env at
    finalize time — it RETURNS ``{"error": ..., "status_code": code}``. Without
    this conversion that error-dict flows through ``handle_finalize`` as if it were
    a real snapshot, and the backend scores the run ``completed`` on empty/vacuous data
    (every entities_of_type(...) resolves to []). A genuine success always carries
    ``success_key`` (``state`` for get_full_state, ``events`` for get_events), so an
    ``error``-keyed dict lacking it is unambiguously a failed read. Map it to the
    same ``_finalize_error`` marker an exception would have produced, so the backend
    scores the run with the world unobserved instead of over an empty world.
    """
    if isinstance(result, dict) and 'error' in result and success_key not in result:
        status = result.get('status_code')
        detail = f'HTTP {status}: {result["error"]}' if status is not None else str(result['error'])
        return {'_finalize_error': f'mock read failed at finalize ({detail})'}
    return result


# ---------------------------------------------------------------------------
# I/O loop
# ---------------------------------------------------------------------------


async def _run_io_loop(
    session: _ChildSession,
    reader: asyncio.StreamReader,
    write: Callable[[Message], Awaitable[None]],
) -> int:
    """Read JSONL from ``reader``, dispatch to ``session``, write responses via ``write``.

    Returns 0 on clean shutdown / EOF; 1 on a protocol error (decode failure,
    a line over the IPC limit, or unexpected envelope).
    """
    while True:
        line = await read_ipc_line(reader)
        if isinstance(line, OversizedIpcLine):
            print(f'[plurai.child] protocol error: an envelope over the {line.limit}-byte IPC line limit', file=sys.stderr, flush=True)
            return 1
        if not line:
            # EOF — parent closed stdin or died. Treat as graceful shutdown.
            return 0
        try:
            msg = decode(line)
        except ProtocolDecodeError as exc:
            print(
                f'[plurai.child] protocol decode error: {exc}; line={line!r}',
                file=sys.stderr,
                flush=True,
            )
            return 1

        if isinstance(msg, Init):
            await write(session.answered(await session.handle_init(msg)))
        elif isinstance(msg, UserTurn):
            await write(session.answered(await session.handle_user_turn(msg)))
        elif isinstance(msg, PipelineInvoke):
            await write(session.answered(await session.handle_pipeline_invoke(msg)))
        elif isinstance(msg, Finalize):
            await write(await session.handle_finalize(msg))
        elif isinstance(msg, Shutdown):
            return 0
        else:
            print(
                f'[plurai.child] unexpected message type: {type(msg).__name__}',
                file=sys.stderr,
                flush=True,
            )
            return 1


def _claim_ipc_stdio() -> tuple[io.FileIO, io.BufferedWriter]:
    """Take the parent's two pipes for the IPC channel; give the agent stdio that cannot reach them.

    The agent runs in this process, and its output reaches fd 1 by every route there is: ``print``,
    a console library, a C extension, a subprocess that inherits the fd. So the channel moves to
    private descriptors (``os.dup`` makes them non-inheritable), fd 1 becomes a copy of stderr,
    which the parent tees to its log, and fd 0 reads ``/dev/null``. Must run before simulate.py
    is imported.
    """
    ipc_in = os.fdopen(os.dup(0), 'rb', buffering=0)
    ipc_out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    sys.stdout.flush()  # after the dup2: text buffered before the claim drains to the log, not the channel
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    utf8_stdio()  # Windows: the parent decodes this log as UTF-8, and an agent's print of a non-cp1252 character must not raise
    if isinstance(sys.stdout, io.TextIOWrapper):
        # Line-buffered, so the agent's prints reach the log while the turn runs, not at exit.
        sys.stdout.reconfigure(line_buffering=True)
    # Both reach the parent's log and a death message: a credential the harness handed its CLI (which
    # only this process knows) goes as its name. A write below Python's streams (a C extension, ``.buffer``,
    # a subprocess on the inherited fd 2 — the agent's CLI too, unless the harness sets ``options.stderr``,
    # the one case the claude-agent-sdk relays its CLI's stderr through ``sys.stderr``) is the parent's to
    # redact, against its own environment only.
    sys.stdout, sys.stderr = RedactingStream(sys.stdout), RedactingStream(sys.stderr)
    return ipc_in, ipc_out


#: Bytes one blocking read of the IPC pipe asks for on Windows (:func:`_feed_from_thread`).
_IPC_READ_CHUNK = 64 * 1024


def _feed_from_thread(reader: asyncio.StreamReader, ipc_in: io.FileIO, loop: asyncio.AbstractEventLoop) -> threading.Thread:
    """Feed ``reader`` from blocking reads of ``ipc_in`` on a daemon thread, then its EOF.

    Windows only. Its event loop (Proactor) reads a pipe with overlapped I/O, and the end of the
    pipe a parent's ``create_subprocess_exec`` hands this process is not opened for it: an
    overlapped read there completes synchronously, so ``connect_read_pipe`` would block the loop
    in each re-armed read until the parent's next envelope, which comes only after this process
    answers. A plain blocking read on a thread needs no overlapped handle.
    """

    def _pump() -> None:
        try:
            while chunk := ipc_in.read(_IPC_READ_CHUNK):
                loop.call_soon_threadsafe(reader.feed_data, chunk)
        except (OSError, ValueError):
            pass  # the pipe broke or was closed under the read: the same as its end
        except RuntimeError:
            return  # the loop is closed: nobody is reading any more
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(reader.feed_eof)

    thread = threading.Thread(target=_pump, name='plurai-ipc-in', daemon=True)
    thread.start()
    return thread


async def _setup_stdio(
    ipc_in: io.FileIO, ipc_out: io.BufferedWriter, *, is_win: bool | None = None
) -> tuple[asyncio.StreamReader, Callable[[Message], Awaitable[None]]]:
    """Wire ``ipc_in`` as an async StreamReader; return a sync-flush writer for ``ipc_out``.

    Writing is done synchronously because the parent is always reading the
    channel (it has a dedicated reader task), so blocking writes don't
    deadlock. This avoids depending on
    ``asyncio.streams.FlowControlMixin`` which is technically private API.
    On Windows the reader is fed from a thread (:func:`_feed_from_thread`).
    """
    loop = asyncio.get_running_loop()
    # Explicit ``limit=`` — the one shared ``IPC_LINE_LIMIT_BYTES`` both ends read.
    reader = asyncio.StreamReader(limit=IPC_LINE_LIMIT_BYTES)
    if is_windows() if is_win is None else is_win:
        _feed_from_thread(reader, ipc_in, loop)
    else:
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, ipc_in)

    def _line(msg: Message) -> bytes:
        # every turn reaches the parent here, and from it the service, the judges and the artifacts
        return redact_cli_credentials(encode(msg).decode('utf-8')).encode('utf-8')

    async def _write(msg: Message) -> None:
        line = _line(msg)
        if len(line) > IPC_LINE_LIMIT_BYTES:
            line = _line(_fit_to_ipc_line(msg, size=len(line)))
        ipc_out.write(line)
        ipc_out.flush()

    return reader, _write


#: The agent-state markers a trimmed reply keeps: the parent types the turn by them (a provider fault the
#: turn ended on, a turn end nobody observed), and dropping them would re-type the turn as the agent's.
_IPC_KEPT_STATE_KEYS: frozenset[str] = frozenset({PROVIDER_TURN_FAULT_KEY, TURN_END_UNOBSERVED_KEY})


def _fit_to_ipc_line(msg: Message, *, size: int) -> Message:
    """``msg`` reduced to what can cross one IPC line, for the two envelopes that carry customer bulk.

    A finalize whose world (then trail) will not fit ships that channel as a ``_finalize_error``
    marker — the parent lands the dialog with the channel unread, exactly as for a failed read. A
    reply drops its message chain and the bulk of its agent state and keeps ``ai_text``,
    ``agent_actions`` and the state markers the parent types the turn by (:data:`_IPC_KEPT_STATE_KEYS`,
    marked ``agent_state[IPC_TRIMMED_KEY]``, which the parent records per turn); a reply still too long becomes a scoreable
    ``reply_oversized`` adapter error that keeps how the user's act went. Any other envelope is sent as it is, and the parent's reader
    names the overrun.
    """

    def _fits(candidate: Message) -> bool:
        return len(encode(candidate)) <= IPC_LINE_LIMIT_BYTES

    if isinstance(msg, FinalizeResponse):
        marker = {'_finalize_error': f'IpcLineOverrun: the snapshot does not fit one IPC line ({size} bytes > {IPC_LINE_LIMIT_BYTES})'}
        trimmed = replace(msg, final_world_state=marker)
        return trimmed if _fits(trimmed) else replace(trimmed, tool_events=marker)
    if isinstance(msg, AgentResponse):
        kept = {key: value for key, value in (msg.agent_state or {}).items() if key in _IPC_KEPT_STATE_KEYS}
        trimmed = replace(msg, full_messages=[], agent_state={**kept, IPC_TRIMMED_KEY: True})
        if _fits(trimmed):
            return trimmed
        return AgentError(
            kind='adapter',
            subkind='reply_oversized',
            exc_class='plurai.IpcLineOverrun',
            message=f'the reply does not fit one IPC line even without its message chain ({size} bytes > {IPC_LINE_LIMIT_BYTES})',
            agent_state=kept,
            turn_world=msg.turn_world,
            user_action_result=msg.user_action_result,
            channel_results=msg.channel_results,
        )
    return msg


def compose_split_world(mock_state: Mapping[str, Any], hook_state: Mapping[str, Any], owned: frozenset[str]) -> tuple[dict[str, Any], list[str]]:
    """A split world, composed in ONE place: every row the hook returned, plus the mock's rows of
    every type the harness does not hold — ``owned`` and any type the hook returned. Also returns
    the hook's types ``owned`` does not declare.

    Nothing the hook returns is dropped: a type the declaration forgot is still the harness's, and
    its stale mock copy is what would otherwise be scored. A declared type the agent emptied keeps
    the mock's copy out. A mock row with no ``_entity_type`` is kept: nothing says whose it is.
    """
    hook_types = {str(row['_entity_type']) for row in hook_state.values() if isinstance(row, Mapping) and row.get('_entity_type') is not None}
    held = owned | hook_types
    kept = {key: row for key, row in mock_state.items() if not (isinstance(row, Mapping) and row.get('_entity_type') in held)}
    return {**kept, **hook_state}, sorted(hook_types - owned)


def _declared_owned_types(simulate_path: Path) -> frozenset[str] | None:
    """``world_state.owned_types`` from the config beside this harness: the entity types the
    harness's own store holds in a split world. ``None`` when nothing declares it; a value that is
    not a list of type names cannot testify and raises."""
    declared: set[str] | None = None
    for root in bench_config_roots(simulate_path):
        for path, document in discover_config_documents(root):
            world = document.get('world_state')
            value = world.get('owned_types') if isinstance(world, dict) else None
            if value is None:
                continue
            if not isinstance(value, list) or not all(isinstance(name, str) and name.strip() for name in value):
                raise WorkspaceConfigError(f'{path}: `world_state.owned_types: {value!r}` is not a list of entity type names.')
            declared = (declared or set()) | {name.strip() for name in value}
    return frozenset(declared) if declared is not None else None


def _declared_live_tools(simulate_path: Path) -> frozenset[str]:
    """``tools.<name>.mode`` ``real`` / ``observe``, plus ``agent.native_tools``, from the config beside this harness."""
    names = set(_declared_native_tools(simulate_path))
    for root in bench_config_roots(simulate_path):
        for _path, document in discover_config_documents(root):
            tools = document.get('tools')
            if isinstance(tools, dict):
                names |= {str(name) for name, entry in tools.items() if isinstance(entry, dict) and entry.get('mode') in ('real', 'observe')}
    return frozenset(names)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _head_path_with_running_interpreter() -> None:
    """The agent's shell finds THIS interpreter's ``python3`` first.

    ``sys.executable`` is the interpreter actually running the agent, whatever launched it (a
    venv path, a relative path, a wrapper script, a version-manager shim); the launcher's own
    directory need not hold a ``python3``.
    """
    if sys.executable:
        os.environ['PATH'] = os.pathsep.join(p for p in (str(Path(sys.executable).parent), os.environ.get('PATH', '')) if p)


async def _amain(argv: list[str]) -> int:
    ipc_in, ipc_out = _claim_ipc_stdio()
    if len(argv) != 2:
        print('Usage: python -m plurai.child <simulate_path>', file=sys.stderr, flush=True)
        return 2
    simulate_path = Path(argv[1])
    _head_path_with_running_interpreter()
    # Armed BEFORE the harness loads, so an import-time client is guarded too; `env:` production hosts
    # resolve AFTER it, from the environment the harness built (see plurai.egress_guard).
    guard = egress_guard.arm_from_env()
    # ``plurai.store_record`` reads the declared world schema beside THIS harness, wherever it is called from.
    seed_rows.bind_harness(simulate_path)
    try:
        module = _load_simulate_module(simulate_path)
        if guard is not None:
            guard.after_harness_loaded()
        session = _ChildSession(module)
    except Exception as exc:
        print(f'[plurai.child] failed to load {simulate_path}: {exc}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1

    reader, write = await _setup_stdio(ipc_in, ipc_out)

    # SIGTERM → close stdin so the loop sees EOF and exits cleanly. Skip on
    # platforms where signal handling is unavailable in asyncio (Windows).
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, reader.feed_eof)
    except (NotImplementedError, ValueError):
        pass

    try:
        return await _run_io_loop(session, reader, write)
    finally:
        session.shutdown()
        await end_cli_processes_in_loop()
        _end_own_process_group()


#: Set to ``1`` in the environment of a child the parent started in its own session (``plurai._subprocess``).
GROUP_LEADER_ENV = 'PLURAI_CHILD_GROUP_LEADER'


def _end_own_process_group() -> None:
    """When this child leads its own session — the parent started it with ``start_new_session`` — end what the harness
    left running in its process group as it exits. The parent does the same at its close; this covers a parent that died
    hard (stdin EOF, no close), whose group signal can no longer reach a server the harness spawned. Only a child the
    parent marked (:data:`GROUP_LEADER_ENV`) that does lead its session acts: one run from a shell, or in-process, is left
    alone. SIGTERM to the group reaches this process too, so it is ignored here first."""
    if os.name != 'posix' or os.environ.get(GROUP_LEADER_ENV) != '1':
        return
    try:
        if os.getsid(0) != os.getpid():
            return
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(os.getpid(), signal.SIGTERM)
    except (OSError, ValueError):
        pass


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv if argv is not None else sys.argv))


if __name__ == '__main__':
    sys.exit(main())
