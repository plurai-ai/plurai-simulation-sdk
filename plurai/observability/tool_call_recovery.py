"""Recover tool_call records from chatbot_messages and from the agent's own turn state.

``tool_events.json`` only records calls dispatched through the SDK's
``/mock/execute`` HTTP endpoint — i.e. ``mode: mock`` tools. ``mode: real``
tools that go through a storage adapter leave
no ``tool_call`` event behind. Every consumer that asks "did tool X
fire?" needs to fall back to ``chatbot_messages`` to avoid blind spots
on all-mode:real architectures.

``chatbot_messages`` closes that gap only for an agent whose tool structure reaches the
transcript (a LangChain ``tool_calls`` attribute, an Anthropic ``tool_use`` block, a
ToolMessage). An agent that runs its OWN tool loop and returns prose does not: message recovery
finds nothing, and a call that failed before it reached ``/mock/execute`` leaves no event
either, so its failure would be invisible. For that topology the harness reports the calls itself, on ``agent_state`` — see
:data:`AGENT_STATE_TOOL_TRAIL_KEY` and :func:`trail_tool_calls`.

Public surface: :func:`derive_tool_calls_from_messages` (+ :func:`derive_turn_tool_calls`, one
turn of a thread over its call ledger, settling an interrupt's answer — :func:`settle_paused_calls`,
:func:`tool_decision_events`), :func:`trail_tool_calls` +
:func:`trail_call_errored`, :func:`recovered_tool_calls` (both out-of-band lanes, settled),
:func:`tool_event_errored` + :func:`normalize_tool_event_status` (the ONE tool-failure
vocabulary across the mocked and live recording lanes), :func:`tool_status_unrecognized` +
:func:`results_unobservable` (the two receipts behind a ``status: 'unknown'`` — which token
neither vocabulary could read, and which transcript shape carries no results at all),
:func:`union_by_call_identity` (the ONE dedup rule the two
lanes are unioned under — declared ``call_id`` first, the tool name only where no id can
answer, and it returns what it dropped unverified), :func:`merged_tool_calls` (flattened list
— for the rollup math + observability) and :func:`merged_tool_calls_with_floor` (the same union
with the floor KEPT — what any caller that renders, counts or asserts on a length wants, since
that count can be a lower bound), :func:`merge_recovered_tool_calls` (returns the
finalize ``tool_events`` dict, turn-ordered — for the orchestrator's /finalize payload),
:func:`merge_pipeline_trail` (the same payload for a one-shot hop's returned trail, call-ordered), and
:func:`attribute_tool_events_to_turns` (grafts mock ``tool_events`` onto the
per-turn ``agent_tool_calls`` so out-of-band ``code`` / ``cli_http`` tool calls
render as observability ``tool`` spans). The recovery records match the
``tool_events.events`` ``tool_call`` schema so callers can union them into
existing event lists without re-shaping.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    # Type-only back-reference: ``sink_protocol`` imports ``RecoveredToolCall`` from
    # THIS module at runtime, so importing ``TranscriptTurn`` here at runtime would be
    # circular. Under ``TYPE_CHECKING`` (never executed) it stays a pure annotation —
    # ``attribute_tool_events_to_turns`` only reads/writes the dict structurally.
    from plurai.observability.sink_protocol import TranscriptTurn

logger = logging.getLogger(__name__)


# Messages reaching this module are heterogeneous: the SDK passes
# ``dict[str, Any]`` (the orchestrator's ``chatbot_full_messages``); other callers
# (e.g. the backend) pass LangChain ``BaseMessage`` objects, and Anthropic-SDK message
# objects also appear. langchain is an OPTIONAL dependency, so we cannot import
# ``BaseMessage`` to narrow the OBJECT case. These per-attribute, runtime-checkable
# Protocols let each branch read the one field it needs off the object shape with a
# concrete type (instead of ``getattr`` returning ``Any``), while the
# ``isinstance(msg, dict)`` fallback keeps reading the SDK's dict messages. A bare
# ``dict`` exposes none of these attributes, so a dict message never matches a
# Protocol and always falls through to the dict branch. They are intentionally
# single-attribute because each field is consulted in a different branch — an object
# need only expose the one field that branch reads.
@runtime_checkable
class _HasStatus(Protocol):
    status: object


@runtime_checkable
class _HasToolCalls(Protocol):
    tool_calls: object


@runtime_checkable
class _HasType(Protocol):
    type: object


@runtime_checkable
class _HasContent(Protocol):
    content: object


@runtime_checkable
class _HasToolCallId(Protocol):
    tool_call_id: str | None


@runtime_checkable
class _HasName(Protocol):
    name: str | None


@runtime_checkable
class _HasRole(Protocol):
    role: object


@runtime_checkable
class _HasId(Protocol):
    id: object


class RecoveredToolCall(TypedDict):
    """A ``tool_events.events`` ``tool_call``-shaped record recovered from messages.

    Matches the ``tool_call`` schema downstream helpers (operation types, revision
    counts, tool arguments …) consume, so a recovered record unions
    into an existing event list without re-shaping. The ``_recovered`` /
    ``state_delta`` distinction is correctness-relevant: ``_recovered=True`` marks a
    SYNTHETIC delta (we know only that the tool fired, not whether it mutated state),
    so mutation-counters must NOT read the synthetic delta or the default ``'ok'``
    status as authoritative evidence (that would over-count read-only mode:real tools).

    ``status`` is THREE-VALUED (``'ok' | 'error' | 'unknown'``) and starts ``'unknown'``: a
    request that has been SEEN is not a call that has COMPLETED, and stamping ``'ok'`` the
    moment the request appears would make "the agent executed X successfully" true for a LangGraph
    ``interrupt()`` that paused for human approval and for a middleware that strips results.
    It settles only on an OBSERVED result (:func:`_result_status`), and ``'ok'`` needs a
    POSITIVE structured witness — a stamped status token, or a top-level ``status`` on a
    structured envelope — never the mere absence of English error words, which would read
    ``"שגיאה: …"`` / ``"错误：…"`` as a success. ``response`` is filled at the same moment.

    ``error_check: 'unchecked'`` marks a record whose result WAS observed and could not be
    decided (no structured witness, or a token outside the vocabulary) — the honest receipt
    behind an ``'unknown'``, and what tells it apart from a request that never came back at
    all (no ``error_check``, ``response`` still ``None``). ``status_token`` carries the RAW
    token that fell outside the vocabulary, so a miss is countable rather than merely absent
    (:func:`tool_status_unrecognized`).

    ``_call_shape`` names WHICH transcript shape produced the record, so a run whose call lane
    recovered records the result lane structurally could not pair is attributable to the shape
    rather than to the agent (:func:`results_unobservable`).

    ``turn`` is the OUTER CONVERSATION turn the call belongs to — the same index the
    mock lane's ``/mock/execute`` events carry, so the two lanes are comparable and
    :func:`merge_recovered_tool_calls` can order one against the other. It is present
    only when the caller knew the turn (recovery over a per-turn message chain); a
    whole-transcript recovery omits it rather than inventing 0.

    ``_lane`` names WHICH out-of-band lane produced the record; this module stamps it only
    on the trail lane (:data:`TRAIL_LANE`), since that is the one distinction
    :func:`attribute_tool_events_to_turns` needs — everything else is already rendered.

    ``_recovered=True`` means only "not a ``/mock/execute`` mutation record", and what it
    licenses DIFFERS BY LANE. On the message lane the delta is synthetic and the status is only
    as good as the result the transcript carried, so neither is evidence. On the TRAIL lane the delta is EMPTY (not
    synthetic) and the status is the harness's own reported verdict — the load-bearing signal
    the lane exists to carry, and a consumer that discards it because ``_recovered`` is set
    throws away exactly the tool failures this lane exists to surface.
    """

    type: str  # always 'tool_call'
    tool: str
    args: dict[str, Any]
    id: str
    state_delta: dict[str, Any]  # message lane: SYNTHETIC; trail lane: EMPTY. Never a mutation record.
    new_entities: dict[str, Any]
    _recovered: bool  # True ⇒ not a /mock/execute mutation record (status IS authoritative on the trail lane)
    #: A result nothing could attach (``derive_tool_calls_from_messages`` rule b): args UNKNOWN, never
    #: pending by id or name, only ever stamped when the transcript carries call blocks. The union
    #: floors it; per-turn readers drop it.
    _unpaired: NotRequired[Literal[True]]
    status: str  # 'unknown' until a result is OBSERVED; then 'ok' | 'error' | 'unknown'
    response: Any  # the ToolMessage content once observed (None until then)
    error_check: NotRequired[Literal['unchecked']]  # a result was observed and could not be decided
    status_token: NotRequired[str]  # the RAW status token that fell outside the vocabulary
    _call_shape: NotRequired[str]  # which transcript shape produced this record (message lane only)
    turn: NotRequired[int]  # conversation turn; absent when the caller did not know it
    _lane: NotRequired[Literal['trail', 'action']]  # stamped on trail and action records; absent on message-lane ones
    #: The harness typed this trail entry or action :data:`PRODUCT_SOURCE` (the agent's own call, not a
    #: fixed one it was handed), or this trail entry :data:`MOCK_SOURCE` (produced by the simulation's mock, not the
    #: system). Exact match only; absent on every other record.
    source: NotRequired[Literal['product', 'mock']]
    #: The ``call_id`` of the trail entry this call ran inside (the entry's :data:`TRAIL_PARENT_KEY`), e.g. a sub-agent's
    #: call made while a delegation call ran. Trail lane only; absent means top-level or unknown.
    parent_id: NotRequired[str]
    #: The argument names the child left out of ``args`` (:data:`TRAIL_ARGS_OMITTED_KEY`). Trail lane only.
    args_omitted: NotRequired[list[str]]
    _not_dispatchable: NotRequired[Literal[True]]  # trail only: the call never left (status stays 'unknown')
    #: A gate said no and the call never ran: the user's interrupt answer was not the harness's
    #: declared ``approve`` choice, or the trail stamped :data:`_TRAIL_DENIED_STATUS`. ``status``
    #: stays ``'unknown'`` and ``response`` keeps what the harness said.
    _denied: NotRequired[Literal[True]]
    #: WHO said no and why — ``{by, reason}``, ``by`` one of :data:`_REFUSED_BY` — on every record a
    #: gate refused for a permission, approval or policy reason (:func:`_as_refused`). Always beside
    #: ``_denied``; the one key a reader asks "was this attempt refused, and by whom".
    refusal: NotRequired[dict[str, str]]
    #: The pause was answered, but no ``approve`` choice was declared, so it cannot be told whether the
    #: call ran: ``unknown`` + ``error_check: 'unchecked'``, never ``ok`` (:func:`settle_paused_calls`).
    _undecided: NotRequired[Literal[True]]
    result_turn: NotRequired[int]  # the turn whose messages carried the result, when it was not ``turn``
    decision_turn: NotRequired[int]  # the turn that answered the interrupt this call was paused on


# Unambiguous error signals inside a serialized ToolMessage *content*. Matched
# with spaces stripped, so ``{"status": "error"}`` and ``{"status":"error"}``
# both hit. Kept narrow on purpose — see ``_tool_response_errored``.
_ERROR_CONTENT_MARKERS: tuple[str, ...] = ('"status":"error"', '_plurai_handler_error')
# Words that, at the START of the (stripped) content, mean the call failed.
# Anchored to the start so a success message that merely mentions "error"
# (e.g. "0 errors found") is not misread.
_ERROR_CONTENT_PREFIXES: tuple[str, ...] = ('error', 'exception', 'traceback')
# A raised Python exception rendered into a ToolMessage leads with its class
# name + colon: ``Error: ...``, ``ValueError: ...``, ``RuntimeException: ...``.
# Anchored to the start; the class token must END in error/exception so prose
# like "No Error: ..." (space breaks the token) does not match.
_EXCEPTION_PREFIX_RE = re.compile(r'^\s*\w*(?:error|exception)\s*:', re.IGNORECASE)
#: A SUCCESS report that merely LEADS with an error tally — "Errors: 0. Deleted 3 rows."
#: The prefix rule above reads any leading error word as failure. A zero count right after
#: the word is the one unambiguous exclusion.
#:
#: The Plurai backend applies the same rule, and the two must agree. This side runs FIRST and
#: stamps ``status: 'error'`` onto the recovered record, which the backend then honours, so a
#: carve-out present only on the backend never runs and a correct action is counted as a failed call.
_ZERO_ERROR_REPORT_RE = re.compile(r'^\s*errors?\b[^0-9]{0,12}0\b', re.IGNORECASE)

#: The status token this module reads as an OBSERVED SUCCESS — the positive counterpart of
#: :data:`_TRAIL_FAILED_STATUSES`, and the only thing that promotes a record to ``'ok'``.
#:
#: It exists because "not an error" and "a success" are different claims and only the second one
#: licenses ``tool_call_succeeded``. The failure side is lexical of necessity (a tool renders its
#: failure however it likes) and therefore ENGLISH: a mock answering ``"שגיאה: הרשומה לא נמצאה"``
#: or ``"错误：未找到记录"`` trips none of the prefixes, the glyph or the markers. Reading that
#: silence as a success would apply an English-only rule to a multilingual pipeline at the grading
#: core, so a result carrying no witness this set can read is ``'unknown'`` with
#: ``error_check: 'unchecked'`` — it abstains, and never grades either way.
#:
#: THREE tokens, and no more, because the trail contract never specified a success spelling: these
#: are the ones in use — LangChain's ``ToolMessage.status`` says ``'success'``, harness trails say
#: ``'returned'``, and resumed runs carry ``'ok'``.
#: Guessing at synonyms would be a hand-rolled table; instead anything
#: OUTSIDE is counted by its raw spelling (:func:`tool_status_unrecognized`), so a harness saying
#: ``'completed'`` shows up as a named miss to widen rather than as a silent green.
#:
#: The Plurai backend reads the same vocabulary, the same obligation :data:`_TRAIL_FAILED_STATUSES`
#: carries.
_OK_STATUSES: frozenset[str] = frozenset({'ok', 'success', 'returned'})

#: Role spellings that name the same message TYPE. The CALL lane reads Anthropic ``tool_use``
#: blocks off a message it recognizes as the assistant's and the RESULT lane pairs one it
#: recognizes as a tool's. ``type`` alone is carried by no OpenAI-shaped role dict and no
#: Anthropic-native message, so reading it alone would pair nothing on those transcripts, and an
#: unpaired call defaults to ``'unknown'``, which would red every scenario for a TRANSCRIPT SHAPE. Read by SHAPE, never by provider — the same resolution
#: ``run_artifacts.declared_message_type`` and ``child._LC_TYPE_ALIASES`` already apply.
_TYPE_ALIASES: dict[str, str] = {'assistant': 'ai', 'model': 'ai', 'function': 'tool', 'user': 'human'}


def _declared_type(msg: dict[str, Any]) -> str:
    """One message's normalized type: ``type`` first, then ``role``, aliases resolved.

    ``''`` when the message names neither — which is not a shape to guess at, so every caller
    below compares against a concrete type and a nameless message simply matches none of them.
    """
    declared = msg.get('type') or msg.get('role')
    normalized = str(declared or '').strip().lower().removesuffix('message')
    return _TYPE_ALIASES.get(normalized, normalized)


def _envelope_status_token(content: Any) -> str:
    """The TOP-LEVEL ``status`` token of a structured tool-response envelope, or ``''``.

    The second witness after the message's own stamp: a ``mode: real`` tool's result reaches the
    transcript as the envelope TEXT, so the structured signal is inside the payload rather than
    beside it. TOP LEVEL ONLY and PARSED, never substring-scanned, for the reason
    ``envelope_refused`` states downstream: a polling agent's success page
    (``{"status": "ok", "jobs": [{"status": "failed"}]}``) must not be read by a nested record's
    field. Anything that is not an object blob carries no witness — ``''``, i.e. abstain.
    """
    payload = content
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped.startswith('{'):
            return ''
        try:
            payload = json.loads(stripped)
        except ValueError:
            # A SINGLE-QUOTED repr is the other serialisation this lane actually receives, and
            # the failure side already falls back for exactly this shape. Giving up on only the
            # success side would make every repr-serialised SUCCESS a permanent abstention
            # while its failures still decided — a one-directional blindness, which is the
            # asymmetry this module refuses everywhere else.
            import ast  # noqa: PLC0415 — only the malformed lane needs it

            try:
                payload = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return ''
    if not isinstance(payload, dict):
        return ''
    return str(payload.get('status') or '').strip().lower()


def _result_status(msg: Any, content_str: str) -> tuple[str, str]:
    """``(status, error_check, unrecognized token)`` for ONE observed tool result.

    The three-valued settlement, in one place so both lanes and both repos answer alike:

    * the failure rules fire (``_tool_response_errored``) → ``'error'``, decided;
    * a status token is present — stamped on the message, or top-level on a structured envelope —
      and it is one this module reads as success → ``'ok'``, decided;
    * a token is present and is NOT one of those → ``'unknown'``, and the RAW token comes back so
      a caller can count it (a closed set says what fell outside it);
    * no token at all → ``'unknown'`` with no token to report. This is the multilingual case: the
      failure lexicon is English, so its silence is not evidence of success.

    TWO values, not three. "Was the status decided" is exactly ``status != 'unknown'``, so a third
    slot carrying it would be a derived field two repos and a parity contract had to keep in step.
    The RECORD still carries ``error_check`` — there it is not derivable, because it separates a
    result observed and undecidable from a request whose result never came back at all.
    """
    if _tool_response_errored(msg, content_str):
        return 'error', ''
    stamped = msg.get('status') if isinstance(msg, dict) else getattr(msg, 'status', None)
    token = str(stamped or '').strip().lower() or _envelope_status_token(msg.get('content') if isinstance(msg, dict) else None)
    if not token:
        return 'unknown', ''
    if token in _OK_STATUSES:
        return 'ok', ''
    if token in _TRAIL_FAILED_STATUSES:
        # RECOGNIZED and CONTRADICTED: the stamp says failure, the failure rules declined it (the
        # zero-error-tally carve-out is the one shape that does this). Nothing decides, so abstain
        # — but do NOT report it as an unrecognized token, which would send someone widening a
        # vocabulary that already holds it.
        return 'unknown', ''
    return 'unknown', token


def _result_observed(record: Mapping[str, Any]) -> bool:
    """Did a RESULT for this call reach the reader at all — whatever it then said?

    Not the same question as ``status != 'unknown'``: a result observed and undecidable is also
    ``'unknown'``, and the two have opposite remedies (widen the witness vocabulary vs. fix the
    harness so results reach the transcript). The ``error_check`` receipt is what separates them.
    """
    return record.get('status') != 'unknown' or record.get('error_check') == 'unchecked'


def tool_status_unrecognized(records: list[dict[str, Any]] | list[RecoveredToolCall]) -> dict[str, int]:
    """``{raw status token: how many records carried it}`` for tokens outside the vocabularies.

    The receipt for :data:`_OK_STATUSES` / :data:`_TRAIL_FAILED_STATUSES`: both are closed sets,
    and a closed set that silently absorbed what it does not recognize would read ``pending`` /
    ``timeout`` / ``skipped`` / ``queued`` — and any non-English spelling — as success. Empty on
    the healthy path, and it must stay silent there.

    ``_unpaired`` rows are skipped, for the same reason :func:`results_unobservable` skips them:
    the union keeps them out of the merged list, so counting one asks an operator to widen a
    vocabulary on the strength of a row nothing grades.
    """
    return dict(Counter(token for record in records if not record.get('_unpaired') and (token := record.get('status_token'))))


def results_unobservable(records: list[dict[str, Any]] | list[RecoveredToolCall]) -> dict[str, dict[str, int]]:
    """``{call shape: {undecided, decided}}`` for every shape carrying an UNDECIDED record.

    THE guard against turning a silent green into a manufactured red. The call lane recovers a
    shape whose outcome the result lane cannot settle (e.g. an Anthropic-native or
    OpenAI-role-dict transcript), and every record of that shape then stays ``'unknown'`` — which
    must be attributable to the TRANSCRIPT SHAPE rather than arriving as a whole run's
    scenarios grading red on their agent.

    UNDECIDED, not unpaired, and reported at ANY count above zero. Counting "did a result
    message ARRIVE" would miss a plain ToolMessage with no status, which pairs perfectly and
    decides nothing; reporting only at 100% would publish NOTHING for nine unresolved calls
    beside one resolved.

    ``decided`` rides along because it is what separates the two findings a reader must act on
    differently: ``decided: 0`` is a BLIND LANE (fix the harness — nothing the agent did could
    change it), while a non-zero ``decided`` beside it is this run's own unresolved calls (an
    interrupt awaiting approval, a middleware drop). Empty when every recovered call settled —
    the healthy path, and it must stay silent there.
    """
    undecided: Counter[str] = Counter()
    decided: Counter[str] = Counter()
    for record in records:
        shape = record.get('_call_shape')
        if not shape or record.get('_unpaired'):
            continue
        # A denied call's outcome IS known — it never ran — so it is not this receipt's blind lane.
        if record.get('status') == 'unknown' and not record.get('_denied'):
            undecided[shape] += 1
        else:
            decided[shape] += 1
    return {shape: {'undecided': count, 'decided': decided[shape]} for shape, count in undecided.items()}


def _normalized_tool_call_entry(entry: Any) -> Any:
    """One ``tool_calls`` entry in the ONE shape every reader below expects: ``name`` / ``args``.

    The entry that carries its callable under a NESTED mapping is the other wire shape a
    ``tool_calls`` list is written in — ``{'id', 'type': 'function', 'function': {'name',
    'arguments'}}``. ``name`` is looked up on the entry itself, so without this every such call
    would be dropped by :func:`derive_tool_calls_from_messages` and invisible to
    :func:`_names_a_call` — and worse than an undercount: with no call on record, the tool RESULT
    that followed would have nothing to pair with and no call blocks anywhere in the slice to mark
    it an echo, so it would be synthesized as a call in its own right (``tool: 'unknown_tool'``,
    ``args: {}``) and graded.

    Unpacked HERE, at the one seam every reader is downstream of, rather than taught to each
    reader — the same reason :func:`normalize_message` exists at all. It is the same rule
    ``langchain_core.messages.convert_to_messages`` applies on the way in (``if 'function' in
    tool_call``), which is what makes it a wire shape rather than one provider's spelling.

    Read by SHAPE, never by provider: an entry that already names its tool is returned
    untouched, whoever wrote it. ``arguments`` given as a JSON string is decoded; one that does
    not decode to a mapping leaves ``args`` absent rather than carrying a string where every
    consumer (``tool_args_for``) reads a dict — the name and the id are
    still recovered, which is what stops the fabrication above.
    """
    if not isinstance(entry, dict) or entry.get('name'):
        return entry
    nested = entry.get('function')
    if not isinstance(nested, dict):
        return entry
    name = nested.get('name')
    if not name:
        return entry
    out = {**entry, 'name': name}
    raw_args = entry.get('args', nested.get('arguments'))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except ValueError:
            raw_args = None
    if isinstance(raw_args, dict):
        out['args'] = raw_args
    else:
        out.pop('args', None)
        # SAID OUT LOUD. Recovering the name and dropping unreadable arguments is the right
        # trade (it stops the fabrication above), but the loss is not free: ``tool_args_for``
        # then returns ``[{}]`` and any argument assertion grades a flat False — a red
        # manufactured by a serialisation, which is the error this module ranks worst.
        logger.warning(
            'tool_call %r declared arguments this reader could not decode into a mapping; the call is recovered with its '
            'NAME but with EMPTY args, so an argument assertion over it grades against arguments nobody saw. Emit '
            '`arguments` as a JSON object.',
            name,
        )
    return out


def _normalized_tool_calls(value: Any) -> Any:
    """``value`` with every entry through :func:`_normalized_tool_call_entry`, or ``value``
    ITSELF when nothing needed unpacking — a message already in the canonical shape stays the
    very object it was, which is what keeps :func:`normalize_message`'s dict pass-through free
    and identity-preserving."""
    if not isinstance(value, list):
        return value
    normalized = [_normalized_tool_call_entry(entry) for entry in value]
    return value if all(new is old for new, old in zip(normalized, value, strict=True)) else normalized


def normalize_message(msg: object) -> dict[str, Any]:
    """Normalize ONE heterogeneous chatbot message to the saved-transcript dict shape.

    THE message normalizer (the single live-vs-saved seam): live runs hand this module
    framework OBJECTS (LangChain ``BaseMessage`` / Anthropic-SDK messages exposing
    ``.type`` / ``.content`` / ``.tool_calls`` / ``.tool_call_id`` / ``.name`` /
    ``.status`` as ATTRIBUTES), while saved transcripts and the SDK's own
    ``chatbot_full_messages`` hand it plain dicts. Everything downstream
    (:func:`derive_tool_calls_from_messages` → :func:`merged_tool_calls` → the rollup
    tool-call metrics) reads ONLY the normalized dict, so a live
    object and its saved dict form are GUARANTEED to render identically.

    A dict passes through untouched — it already IS the canonical shape — with ONE
    exception, and it is the exception that makes the claim true: a ``tool_calls`` entry
    that names its tool under a nested ``function`` mapping is unpacked to the canonical
    ``name``/``args`` (:func:`_normalized_tool_call_entry`). A dict needing nothing is
    returned as the same object. An object is read via the single-attribute runtime
    Protocols above, lifting EXACTLY the fields the recovery consumes — through the same
    entry rule, so an SDK message object and its saved dict form still render identically.
    Total — never raises: an object exposing none of the fields normalizes to ``{}`` and
    recovers nothing, the same as an empty dict.
    """
    if isinstance(msg, dict):
        calls = msg.get('tool_calls')
        normalized_calls = _normalized_tool_calls(calls)
        return msg if normalized_calls is calls else {**msg, 'tool_calls': normalized_calls}
    out: dict[str, Any] = {}
    if isinstance(msg, _HasType):
        out['type'] = msg.type
    if isinstance(msg, _HasRole):
        # The OTHER spelling of the same field, lifted for the same reason ``_declared_type``
        # reads it: an Anthropic-SDK message object names its role and no ``type`` at all, so
        # without this the object form of a shape the dict form reads would still
        # fall through untyped and pair nothing.
        out['role'] = msg.role
    if isinstance(msg, _HasContent):
        out['content'] = msg.content
    if isinstance(msg, _HasToolCalls):
        out['tool_calls'] = _normalized_tool_calls(msg.tool_calls)
    if isinstance(msg, _HasToolCallId):
        out['tool_call_id'] = msg.tool_call_id
    if isinstance(msg, _HasName):
        out['name'] = msg.name
    if isinstance(msg, _HasStatus):
        out['status'] = msg.status
    if isinstance(msg, _HasId) and isinstance(msg.id, str):
        # The MESSAGE's own identity (LangChain stamps one on every message a graph holds): what
        # tells a re-listed message from a new one that reuses a tool-call id (``ThreadCalls``).
        out['id'] = msg.id
    return out


def _tool_response_errored(msg: dict[str, Any], content_str: str) -> bool:
    """True when a ToolMessage represents a FAILED tool call.

    Takes the NORMALIZED message dict (:func:`normalize_message` lifts the
    authoritative live ``ToolMessage.status`` attribute into the dict, so the dict
    read below covers both the live-object and saved-transcript forms). Priority is
    that authoritative LangChain ``status`` field — ``ToolNode`` stamps it ``'error'``
    whenever the wrapped tool raises (so an uncaught ``int('rev_8f2a1b')`` surfaces
    here even when the rendered content is just the raw exception text). Falls back to
    content markers for paths that drop the field: dict-serialised messages, custom
    mocks, and transcripts where only the error string survives. Conservative on the
    text side — a status marker, the ✋ refusal glyph, or an error-word *prefix* —
    so a successful response is never mislabelled an error (which would inflate
    the error count)."""
    status = msg.get('status')
    if isinstance(status, str) and status.lower() == 'error':
        return True
    if not content_str:
        return False
    lowered_compact = content_str.lower().replace(' ', '')
    if any(marker in lowered_compact for marker in _ERROR_CONTENT_MARKERS):
        return True
    if '✋' in content_str:
        return True
    if content_str.lstrip().lower().startswith(_ERROR_CONTENT_PREFIXES) and not _ZERO_ERROR_REPORT_RE.match(content_str):
        return True
    return _EXCEPTION_PREFIX_RE.match(content_str) is not None


def _parse_content_blocks(content: Any) -> list[Any] | None:
    """Recover a ``[{type, ...}]`` content-block list, including from
    a ``str(repr)`` serialisation that some IPC paths produce."""
    if isinstance(content, list):
        return content
    if isinstance(content, str) and content.startswith('[') and content.endswith(']'):
        import ast  # noqa: PLC0415

        try:
            parsed = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _tool_calls_from_content_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    """Extract Anthropic-style ``tool_use`` content blocks from a list."""
    out: list[dict[str, Any]] = []
    for blk in blocks:
        if not isinstance(blk, dict) or blk.get('type') != 'tool_use':
            continue
        name = blk.get('name') or ''
        if not name:
            continue
        out.append(
            {
                'name': name,
                'args': blk.get('input') or blk.get('args') or {},
                'id': blk.get('id') or '',
            }
        )
    return out


def _tool_results_from_content_blocks(blocks: list[Any], envelope: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Anthropic ``tool_result`` content blocks, each shaped as a canonical ToolMessage dict.

    The RESULT-lane counterpart of :func:`_tool_calls_from_content_blocks`: an Anthropic
    transcript carries its results as blocks inside a ``user``-role message, so without this
    every recovered call on that shape would stay ``'unknown'`` — a red on every scenario of that
    agent class. Shaped into the ToolMessage dict rather than handled separately so ONE result
    path settles every shape.

    ``is_error`` is this shape's structured witness, and it is read only where it is actually
    PRESENT. Mapping an ABSENT ``is_error`` to ``success`` would manufacture the witness out of
    a silence — and then a tool answering ``"שגיאה: הרשומה לא נמצאה"`` inside a block, with no
    flag either way, would read ``ok`` and `tool_call_succeeded` would return True. That would
    re-create the English-lexicon defect one provider over. Absent ⇒ no stamp ⇒ the result settles on its own envelope, or abstains.

    The block's ``content`` is FLATTENED to text when it is a nested block list, because that is
    the string the failure rules then read: left as a list, ``str(content)`` hands them
    ``"[{'type': 'text', …}]"`` and every start-anchored rule is structurally dead — the same
    failure reads ``error`` as a string and ``unknown`` as blocks, which is precisely the
    read-by-provider divergence this module refuses everywhere else.
    """
    carrier = envelope or {}
    out: list[dict[str, Any]] = []
    for blk in blocks:
        if not isinstance(blk, dict) or blk.get('type') != 'tool_result':
            continue
        result_msg: dict[str, Any] = {
            'type': 'tool',
            # The BLOCK first, its ENVELOPE second. A ToolMessage whose content happens to be a
            # tool_result block still carries the authoritative pairing key and status on the
            # message itself, and reading the block alone would discard them: the real call would
            # stay unresolved and the result be re-synthesized as an `unknown_tool` orphan.
            'tool_call_id': blk.get('tool_use_id') or blk.get('id') or carrier.get('tool_call_id') or '',
            'name': blk.get('name') or carrier.get('name'),
            'content': _flattened_block_text(blk.get('content')),
        }
        if 'is_error' in blk:
            result_msg['status'] = 'error' if blk.get('is_error') else 'success'
        elif carrier.get('status') is not None:
            result_msg['status'] = carrier['status']
        out.append(result_msg)
    return out


def _flattened_block_text(content: Any) -> Any:
    """A nested content-block list rendered to its text; anything else unchanged.

    Only the ``text`` blocks, joined — the same projection the reply readers apply. A list that
    carries no text block at all is left as it is rather than flattened to ``''``: dropping a
    payload nobody could render would delete the response a judge reads.
    """
    if not isinstance(content, list):
        return content
    texts = [t for b in content if isinstance(b, dict) and b.get('type') == 'text' and isinstance(t := b.get('text'), str)]
    return ''.join(texts) if texts else content


class ThreadCalls(dict[str, RecoveredToolCall]):
    """One thread's call ledger, kept by a PER-TURN caller: the calls still OPEN, by id (the mapping
    itself), and ``seen_messages`` — the ids of the MESSAGES already read on earlier turns.

    A resumed graph that returns its whole state hands the answering turn every message since the
    last user message, the ones already read included — a call the user DENIED last turn beside
    its denial. Read afresh, that pair is an executed call. A re-listed message is the same message,
    so it is history and skipped; a tool-call id alone proves nothing, because harnesses reuse them
    (a per-turn counter restarts at ``call_0``) for calls that are genuinely new. A message with no
    id of its own cannot be told apart and is read as new. A plain ``{}`` (a whole-chain caller)
    carries no history.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen_messages: set[str] = set()


def derive_tool_calls_from_messages(messages: list[Any], *, turn: int | None = None, open_calls: dict[str, RecoveredToolCall]) -> list[RecoveredToolCall]:
    """Build tool_call-shaped records from chatbot_messages.

    Three recovery paths in priority order:
      1. ``msg.tool_calls`` attribute (LangChain BaseMessage / dict form).
      2. Anthropic-style content blocks with ``type=tool_use``.
      3. Orphan ToolMessages — synthesise a record from the
         ToolMessage alone (common when middleware filters tool-call
         AIMessages from the transcript).

    A ToolMessage whose ``tool_call_id`` names no pending call is NOT always an orphan, and
    reading it as one manufactures verdicts: a harness that mints FRESH ids for its result
    messages leaves every AI-side call unresolved and every result synthesized as a second call
    with ``args: {}``, which deterministic policies then grade red on the empty args. So an id miss is resolved in order:

      a. PAIR BY NAME with the oldest unresolved call of the same tool — the harness numbered
         its results differently from its calls, and the call is right there.
      b. Otherwise, when the transcript carries call blocks at all, the result is an ECHO or a
         second copy of one already resolved: it is kept as a record for the count floor but
         stamped ``_unpaired: True`` and its ``args`` are UNKNOWN, not empty. The union
         (:func:`union_by_call_identity`) keeps such a record out of the merged list and counts
         it in the floor, so no expression grades an argument nobody saw.
      c. Only a MESSAGE SLICE with no call blocks anywhere in it (a loop whose middleware
         strips the tool-call AIMessages) synthesizes an orphan as a call in its own right — the
         only channel that topology has. "Anywhere" is the list this call received: a
         per-turn caller decides per turn, a whole-chain caller over the chain.

    Each record is a :class:`RecoveredToolCall` matching the
    ``tool_events.events`` ``tool_call`` schema so downstream helpers
    (operation types, revision counts, tool arguments, …) need no changes.

    ``turn`` stamps every produced record with the conversation turn ``messages``
    belongs to. Pass it when recovering a SINGLE turn's message chain (which is what
    makes the recovered records orderable against the mock lane's events); leave it
    unset for a whole-transcript recovery, where the records carry no turn at all
    rather than a fabricated one.

    ``open_calls`` is the THREAD's unresolved calls by id, and REQUIRED: a per-turn caller keeps
    one ledger for the whole thread; a whole-chain caller passes ``{}`` and says so. A call and
    its result straddle a turn exactly when the agent paused for approval — the call ends the
    interrupted turn, the result opens the answering one — so a per-turn slice holds an orphan
    that rule c would mint as a second call (``unknown_tool``, ``args: {}``, and LangChain's
    default ``status='success'`` read as ``ok``). A result whose id names an open call settles
    THAT record in place (stamping ``result_turn``) and adds nothing to the returned list; so
    does a call whose id is already open — a resumed graph that returns its whole state re-lists
    the paused call beside its answer, and that is the same MESSAGE, read already. The ledger
    is updated with this slice's calls still unresolved. By id, and only when the result does not
    name another tool — pairing an earlier turn's call by NAME alone would take a never-answered
    call over the one this turn issued. A per-turn caller
    passes a :class:`ThreadCalls`, whose ``seen_messages`` makes a re-listed message history.
    """
    out: list[RecoveredToolCall] = []
    pending_by_id: dict[str, RecoveredToolCall] = {}
    # Unresolved CALLS by tool name, in order — the id-miss fallback (rule a above).
    pending_by_name: dict[str, list[RecoveredToolCall]] = {}
    normalized = [normalize_message(raw) for raw in messages or []]
    has_call_blocks = any(_names_a_call(msg) for msg in normalized)
    seen_messages: set[str] | None = getattr(open_calls, 'seen_messages', None)
    if seen_messages is not None:
        # A message an earlier turn already read is history (see ``ThreadCalls``): the call it carries
        # was recorded on that turn, and the result it carries already settled it.
        fresh = [msg for msg in normalized if not (isinstance(msg.get('id'), str) and msg['id'] in seen_messages)]
        seen_messages.update(msg['id'] for msg in normalized if isinstance(msg.get('id'), str))
        normalized = fresh
        has_call_blocks = any(_names_a_call(msg) for msg in normalized)

    def _add_call(name: str, args: dict[str, Any], call_id: str, *, shape: str, unpaired: bool = False) -> RecoveredToolCall:
        record: RecoveredToolCall = {
            'type': 'tool_call',
            'tool': name,
            'args': args,
            'id': call_id,
            # UNKNOWN until a result is observed. See ``RecoveredToolCall``: a request that has
            # been seen is not a call that completed, and an ``'ok'`` default would make an
            # interrupted or middleware-stripped call read as a success to every reader.
            'status': 'unknown',
            'response': None,
            '_call_shape': shape,
            # SYNTHETIC delta — we only know the tool fired, not whether it mutated
            # state. ``_recovered`` marks this so the mutation and error counters
            # don't read the
            # synthetic delta or a settled status as authoritative evidence —
            # which would over-count read-only mode:real tools. operation_types() still
            # surfaces the tool name (that the tool fired IS authoritative).
            'state_delta': {'type': name},
            'new_entities': {},
            '_recovered': True,
        }
        if turn is not None:
            record['turn'] = turn
        if unpaired:
            # Rule b: a result nothing here can attach. Its args were never seen.
            record['_unpaired'] = True
        out.append(record)
        if not unpaired:
            if call_id:
                pending_by_id[call_id] = record
            pending_by_name.setdefault(name, []).append(record)
        return record

    def _resolve(record: RecoveredToolCall) -> None:
        pending_by_id.pop(record.get('id') or '', None)
        queue = pending_by_name.get(record.get('tool') or '')
        if queue and record in queue:
            queue.remove(record)

    def _apply_result(result_msg: dict[str, Any]) -> None:
        """Attach ONE observed result — from any shape — to the call it belongs to.

        Every result shape funnels here (a LangChain / OpenAI-role tool message, an Anthropic
        ``tool_result`` block already shaped like one) so the pairing rules a/b/c and the status
        settlement are written once and cannot answer differently per provider.
        """
        tool_call_id = result_msg.get('tool_call_id')
        content = result_msg.get('content')
        record = pending_by_id.get(tool_call_id) if tool_call_id else None
        tool_name = result_msg.get('name')
        if record is None and tool_call_id and (earlier := open_calls.get(tool_call_id)) is not None and tool_name in (None, '', earlier.get('tool')):
            # An earlier turn's call, answered here: settled in place, never re-minted.
            record = earlier
        if record is None and tool_name and pending_by_name.get(tool_name):
            # Rule a: the harness numbered its results differently from its calls.
            record = pending_by_name[tool_name][0]
        if record is None:
            record = _add_call(tool_name or 'unknown_tool', {}, tool_call_id or '', shape='orphan_result', unpaired=has_call_blocks)
        record['response'] = content
        status, token = _result_status(result_msg, str(content) if content is not None else '')
        record['status'] = status
        if status == 'unknown':
            # The receipt, stamped on the record itself: a result WAS observed here and could not
            # be decided. Without it an undecidable result and a call that never came back are
            # both merely ``'unknown'``, and they have opposite remedies.
            record['error_check'] = 'unchecked'
        if token:
            record['status_token'] = token
        if record is open_calls.get(record.get('id') or ''):
            open_calls.pop(record['id'])
            if turn is not None:
                record['result_turn'] = turn
        # This record is now resolved. Clear it from both pending maps so a transcript
        # that REUSES the same tool_call_id (retry / malformed) doesn't clobber this
        # resolved record's response/status — the next result with the same id falls
        # through to the rules above (a fresh record) instead.
        _resolve(record)

    for msg in normalized:
        # Normalized ABOVE: every branch below reads the ONE canonical dict shape, so a
        # live framework object and its saved-transcript dict form take identical paths
        # (the live-vs-saved render-divergence gotcha is structurally impossible here).
        tc_list = msg.get('tool_calls')
        if isinstance(tc_list, list) and tc_list:
            for tc in tc_list:
                if not isinstance(tc, dict):
                    continue
                name = tc.get('name') or ''
                if not name:
                    continue
                _add_call(name, tc.get('args') or {}, tc.get('id') or '', shape='tool_calls_attribute')
            continue

        msg_type = _declared_type(msg)
        content = msg.get('content')
        blocks = _parse_content_blocks(content)
        # ``tool_result`` blocks are read off ANY envelope role, because that is where they live:
        # an Anthropic-native transcript carries the result of an ``assistant`` tool_use inside
        # the NEXT ``user`` message. ``tool_use`` blocks stay scoped to the assistant's own turn —
        # a call is a thing the agent does, and reading one off a user message would recover a
        # request nobody made.
        if blocks:
            calls = _tool_calls_from_content_blocks(blocks) if msg_type == 'ai' else []
            results = _tool_results_from_content_blocks(blocks, envelope=msg if msg_type == 'tool' else None)
            for tc in calls:
                _add_call(tc['name'], tc['args'], tc['id'], shape='anthropic_tool_use')
            for result_msg in results:
                _apply_result(result_msg)
            if calls or results:
                continue
        elif msg_type == 'ai' and isinstance(content, str) and content.strip().startswith('['):
            # Content LOOKED like a content-block list but could not be parsed
            # (e.g. malformed repr that ast.literal_eval rejects). This module
            # exists to eliminate tool-call blind spots, so don't drop it
            # silently — warn (with a truncated snippet, never the full content).
            logger.warning(
                "AI message's tool-use content could not be parsed into content blocks; skipping it (possible lost tool calls). Truncated content: %r",
                content[:200],
            )

        if msg_type == 'tool':
            _apply_result(msg)
    open_calls.update(pending_by_id)
    return out


#: ``tool_decision.decision`` values — the user's answer to an interrupt, per paused call.
DECISION_APPROVED: Final = 'approved'
DECISION_DENIED: Final = 'denied'
#: The pause was answered, but the harness declared no ``approve`` choice, so the answer cannot be typed.
DECISION_UNDECIDED: Final = 'undecided'
#: ``tool_events.events`` type of the record that carries one decision. Not a ``tool_call``, so no
#: reader of calls counts it and a consumer that re-derives the calls from the chain keeps it.
TOOL_DECISION_EVENT: Final = 'tool_decision'


def settle_paused_calls(paused: list[RecoveredToolCall], *, interrupt: Mapping[str, Any] | None, choice: str | None, turn: int) -> None:
    """Apply the user's answer to the calls an interrupt paused, once their result arrived on ``turn``.

    ``paused`` is the ledger's content at the moment the interrupt was raised (``open_calls`` before
    the answering turn is derived); ``interrupt`` its normalized contract, whose ``approve`` names
    the choice that releases the calls. Only a call the answering turn SETTLED is decided — one still
    open stays paused for a later answer:

    * ``choice == approve``: executed — the status is the result's own; ``decision_turn`` stamped;
    * any other choice, or none: denied — ``status: 'unknown'``, ``_denied``, ``decision_turn``, and
      ``response`` keeps the harness's text. The result is a gate's answer, not the tool's, so it
      settles neither ``ok`` nor ``error``;
    * no ``approve`` declared: the answer cannot be typed, so the result is undecidable —
      ``unknown`` + ``error_check: 'unchecked'`` + ``_undecided``, never ``ok``. It is a decision
      record all the same (``decision_turn``), because a consumer that re-derives the call from the
      chain reads the harness's answer as the tool's ``ok`` unless it is told otherwise.

    Mutates the records in place. No text is read: the choice is compared to the declared token.
    """
    if not interrupt:
        return
    approve = interrupt.get('approve')
    for record in paused:
        if record.get('result_turn') != turn:
            continue
        record['decision_turn'] = turn
        if not isinstance(approve, str):
            record['status'] = 'unknown'
            record['error_check'] = 'unchecked'
            record['_undecided'] = True
            record.pop('status_token', None)
            continue
        if choice == approve:
            continue
        record['status'] = 'unknown'
        record['_denied'] = True
        record['refusal'] = _refusal('user', _NOT_APPROVED)
        record.pop('error_check', None)
        record.pop('status_token', None)


def derive_turn_tool_calls(
    messages: list[Any], *, turn: int, open_calls: ThreadCalls, paused_by: Mapping[str, Any] | None, choice: str | None
) -> list[RecoveredToolCall]:
    """ONE turn's recovery on a thread: :func:`derive_tool_calls_from_messages` over the ledger, then
    :func:`settle_paused_calls` when the previous turn ended on an interrupt (``paused_by``, its
    normalized contract) that ``choice`` answers."""
    paused = list(open_calls.values()) if paused_by else []
    out = derive_tool_calls_from_messages(messages, turn=turn, open_calls=open_calls)
    settle_paused_calls(paused, interrupt=paused_by, choice=choice, turn=turn)
    return out


def tool_decision_events(records: list[RecoveredToolCall]) -> list[dict[str, Any]]:
    """One :data:`TOOL_DECISION_EVENT` per record the user's answer decided (:func:`settle_paused_calls`).

    Each names the call OCCURRENCE it answers — verb, id, args, and ``occurrence``, the ordinal of
    that verb-id-args among ``records`` in thread order — because an id alone names no call: a
    harness numbering per turn reuses ``call_0``, and a denial of the first must not mark a later
    one a consumer re-derives from the chain.
    """
    ordinal: Counter[tuple[str, str, str]] = Counter()
    out: list[dict[str, Any]] = []
    for record in records:
        if record.get('_unpaired') or (key := _decision_key(record)) is None:
            continue
        occurrence = ordinal[key]
        ordinal[key] += 1
        if 'decision_turn' not in record:
            continue
        decision = DECISION_DENIED if record.get('_denied') else DECISION_UNDECIDED if record.get('_undecided') else DECISION_APPROVED
        out.append(
            {
                'type': TOOL_DECISION_EVENT,
                'tool': record.get('tool'),
                'call_id': record.get('id'),
                'args': record.get('args') or {},
                'occurrence': occurrence,
                'decision': decision,
                'turn': record['decision_turn'],
            }
        )
    return out


def _names_a_call(msg: dict[str, Any]) -> bool:
    """Does this (normalized) message carry a tool CALL — a ``tool_calls`` list or a
    ``tool_use`` content block? The transcript-level fact rule b above turns on."""
    tc_list = msg.get('tool_calls')
    if isinstance(tc_list, list) and any(isinstance(tc, dict) and tc.get('name') for tc in tc_list):
        return True
    if _declared_type(msg) == 'ai':
        blocks = _parse_content_blocks(msg.get('content'))
        return bool(blocks and _tool_calls_from_content_blocks(blocks))
    return False


#: Where a harness whose agent runs its OWN tool loop reports the calls it made — a list of
#: ``{tool, args, call_id, result, status, refusal?}`` entries in execution order, on the same
#: ``agent_state`` payload the backend already reads. It is the ONLY channel that
#: topology has: a ``mode: real`` dispatch writes no ``/mock/execute`` event, and a self-contained
#: loop puts no tool_use block in the transcript, so without it a call that failed and a call that
#: never happened are byte-identical to every reader.
AGENT_STATE_TOOL_TRAIL_KEY: Final = 'tool_trail'

#: ``RecoveredToolCall._lane`` on a trail record — the lane whose records nothing else renders
#: (see :func:`attribute_tool_events_to_turns`). The backend tags its MESSAGE records
#: ``'message'``, so ``_lane`` is a shared wire vocabulary, not this
#: module's alone; everything that is not ``TRAIL_LANE`` is read here as already-rendered.
TRAIL_LANE: Final = 'trail'
#: ``RecoveredToolCall._lane`` on a record read off the typed ``agent_actions`` channel — shared with the backend.
ACTION_LANE: Final = 'action'
#: ``RecoveredToolCall.source`` on a trail or action record the harness typed as the product's own
#: call. Shared wire vocabulary with the backend.
PRODUCT_SOURCE: Final = 'product'
#: ``RecoveredToolCall.source`` on a trail record the harness typed as produced by the simulation's own mock, not by the
#: system under test. Shared wire vocabulary with the backend.
MOCK_SOURCE: Final = 'mock'
#: The ``tool_trail`` entry key naming the ``call_id`` of the trail entry it ran inside. Shared with the
#: backend; both readers carry it as the record's ``parent_id``.
TRAIL_PARENT_KEY: Final = 'parent_call_id'
#: The ``tool_trail`` entry key naming the arguments the child left out of the entry's ``args`` (an injected framework
#: object, injected state the model never sent — ``plurai.child._model_visible_trail``). Carried onto the trail record
#: as is, so a reader of ``tool_events.json`` (an auditor, the upload) can tell an argument was removed rather than
#: never passed.
TRAIL_ARGS_OMITTED_KEY: Final = 'args_omitted'

#: A trail entry's own ``status``, when it names a FAILED dispatch. This reads the refusal
#: vocabulary about the CALL RECORD: the harness stamped it on the dispatch it made, not on a
#: payload a tool returned (where ``failed`` is a job poll's legitimate answer, which is why
#: :func:`tool_event_errored` does not read these synonyms off a response envelope).
#: ``unavailable`` is the harness gate refusing a call before it left.
#:
#: The backend reads the same vocabulary; the two are identical BY VALUE, not by construction.
_TRAIL_FAILED_STATUSES: frozenset[str] = frozenset({'error', 'unavailable', 'failed', 'failure', 'refused'})

#: A trail entry's own ``status`` when the call NEVER LEFT because the name is not one this agent
#: has — the trail contract's third dispatch outcome. MIRRORS
#: :data:`plurai._protocol.TRAIL_NOT_DISPATCHABLE_STATUSES` by VALUE rather than importing it, on
#: the same rule the vocabularies above obey: this module stays loadable BY FILE with nothing but
#: the stdlib. A test pins the two equal.
#:
#: Deliberately NOT folded into :data:`_TRAIL_FAILED_STATUSES`, which stays identical to the
#: backend's: a name that was never dispatchable did not FAIL, and counting it as a failed
#: dispatch would blame the environment for an invented verb.
#: ``refused`` is a member of BOTH sets and never reaches the branch below — it stays a
#: failure so the two sides cannot disagree about a token they both list, and only the
#: unambiguous spelling changes reading here.
_TRAIL_NOT_DISPATCHABLE_STATUSES: frozenset[str] = frozenset({'refused', 'not_dispatchable'})

#: A trail entry's own ``status`` when a gate said no and the call never ran — MIRRORS
#: :data:`plurai._protocol.TRAIL_STATUS_DENIED` by value, on the same stdlib-only rule as above, and
#: pinned equal by a test. Deliberately outside
#: :data:`_TRAIL_FAILED_STATUSES`: a decision, not a failed dispatch.
_TRAIL_DENIED_STATUS: Final = 'denied'

#: The typed refusal record — MIRRORS :data:`plurai._protocol.TRAIL_REFUSAL_KEY` and
#: :data:`plurai._protocol.REFUSED_BY` by value, on the stdlib-only rule above; pinned equal by a test.
_REFUSAL_KEY: Final = 'refusal'
_REFUSED_BY: frozenset[str] = frozenset({'mock', 'harness', 'confinement', 'user'})
#: The reason on a call the simulated user's interrupt answer did not release.
_NOT_APPROVED: Final = 'the user did not approve the call'


def _refusal(by: str, reason: Any) -> dict[str, str]:
    """One ``{by, reason}`` record; a structured reason is kept as its JSON text."""
    return {'by': by, 'reason': reason if isinstance(reason, str) else '' if reason is None else _canonical(reason)}


def _as_refused(record: Mapping[str, Any], refusal: dict[str, str]) -> dict[str, Any]:
    """``record`` as a refused attempt: the ``_denied`` shape (``status: 'unknown'``, no receipt of an
    undecidable result — the outcome is known, the call never took effect) with ``refusal`` stamped."""
    kept = {k: v for k, v in record.items() if k not in ('error_check', 'status_token')}
    return {**kept, 'status': 'unknown', '_denied': True, _REFUSAL_KEY: refusal}


def mock_refusal(response: Any) -> dict[str, str] | None:
    """The refusal a MOCK declared on its envelope — a top-level ``refusal: {reason}`` sibling — or ``None``.

    Typed, never read off the text: a permission error a mock renders the way the real tool prints
    it (``Error from server (Forbidden)``, ``EACCES``, an HTTP 403 body) is also what a log line or a
    fetched page legitimately contains, so only the envelope's own declaration says the SYSTEM said
    no. The sibling never reaches the agent (:func:`plurai.mock_result` hands it ``content``).
    """
    declared = response.get(_REFUSAL_KEY) if isinstance(response, Mapping) else None
    if not isinstance(declared, Mapping):
        return None
    reason = declared.get('reason')
    return _refusal('mock', reason if reason not in (None, '') else response.get('message') or response.get('error'))


def _trail_refusal(entry: Mapping[str, Any]) -> dict[str, str] | None:
    """The refusal ONE trail entry records, or ``None``: the ``refusal`` it declares, else the
    harness's own ``denied`` stamp, else a mock envelope's declaration in its ``result``."""
    declared = entry.get(_REFUSAL_KEY)
    if isinstance(declared, Mapping):
        by = str(declared.get('by') or '').strip().lower()
        if by not in _REFUSED_BY:
            # The harness DID declare a refusal; only its ``by`` is outside the closed set, so it is
            # kept as the harness's own and the token is said out loud rather than absorbed.
            logger.warning('a tool_trail refusal names by=%r, outside %s; recorded as by=harness.', declared.get('by'), sorted(_REFUSED_BY))
            by = 'harness'
        reason = declared.get('reason')
        return _refusal(by, reason if reason not in (None, '') else entry.get('result'))
    if str(entry.get('status') or '').strip().lower() == _TRAIL_DENIED_STATUS:
        return _refusal('harness', entry.get('result'))
    return mock_refusal(entry.get('result'))


#: TRANSPORT failures rendered as PROSE where the data belongs — the shape a self-reported trail
#: carries ("The response from the database is <endpoint>: fetch failed").
#: None of the response rules above sees it: it is not a structured
#: envelope, it does not LEAD with an error word, and its ``status`` says the call returned.
#:
#: Every phrase here names a failure to REACH the tool, which is what makes widening to it safe
#: where the envelope contract refuses ``rejected`` / ``denied`` / a poll's ``status: failed``:
#: those have a successful business reading and a connection that was never made does not.
#:
#: ANCHORED TO THE END, which is the rule that keeps it honest. The engine substitutes its error
#: text FOR the data, so the phrase terminates the content; a tool that legitimately RETURNS text
#: about a network problem ("the customer said the connection failed at their end") keeps talking
#: afterwards. The cost is that a trailing fragment hides a real one ("request timed out after
#: 30s") — an UNDERcount, which is the direction this module chooses elsewhere too, and the
#: harness's own ``status`` stamp (:data:`_TRAIL_FAILED_STATUSES`) is the reliable channel for
#: anything the prose misses. The backend applies the same rule (see above).
_TRANSPORT_FAILURE_RE = re.compile(
    r'(?:'
    r'\b(?:fetch|request|connection|connect|socket|lookup|dns)\s+(?:failed|error|refused|reset|timed\s+out)'
    r'|\b(?:econnrefused|econnreset|enotfound|etimedout|ehostunreach|epipe)'
    r'|\b(?:service\s+unavailable|bad\s+gateway|gateway\s+time-?out|internal\s+server\s+error)'
    r')\s*[.!]?\s*$',
    re.IGNORECASE,
)


def trail_call_errored(entry: dict[str, Any]) -> bool:
    """True when a self-reported trail entry records a failed DISPATCH — the call never went
    through. Two readings: the harness's own status stamp, and the transport-failure prose that
    stamp structurally cannot carry (:data:`_TRANSPORT_FAILURE_RE`).

    Deliberately NOT the response envelope. A tool that dispatched fine and answered with an
    error-shaped payload REFUSED, which is a different finding with a different owner, and the
    consumers of these records already read that payload for themselves
    (:func:`tool_event_errored` here, and the backend's own reader). Folding
    it in would report a tool that correctly refused as a broken dispatch.
    """
    if str(entry.get('status') or '').strip().lower() in _TRAIL_FAILED_STATUSES:
        return True
    result = entry.get('result')
    return result is not None and _TRANSPORT_FAILURE_RE.search(result if isinstance(result, str) else str(result)) is not None


def _trail_status(entry: dict[str, Any]) -> tuple[str, str]:
    """``(status, unrecognized token)`` for ONE self-reported trail entry.

    :data:`_TRAIL_FAILED_STATUSES` is a closed set, so a token it does not list — ``pending``,
    ``timeout``, ``skipped``, ``queued``, any non-English spelling — must not read as a SUCCESS:
    a harness reporting a call it never got an answer for would be graded as one that completed.
    The set stays closed; a token outside it abstains and says so, instead of picking the
    flattering side.

    An ABSENT status stays a success. That is not an oversight: the trail contract enumerates
    only the failure vocabulary and has never specified a success spelling, the shipped
    ``simulate.py`` template names the five keys without saying what to write in ``status``, and
    the plugin's own dispatch-unreadable marker is DEFINED on that absence. Flipping it would
    silently neuter every harness that omits the field — the manufactured-red direction — so absence is left where the contract left it and only a token that
    positively says something unreadable abstains.
    """
    if trail_call_errored(entry):
        return 'error', ''
    if 'status' not in entry:
        return 'ok', ''
    # PRESENT but falsy (``False``, ``0``, ``''``) is NOT the absence carved out above: the
    # harness reported something and this reader cannot read it, which is counted rather
    # than folded into the success branch. ``or ''`` would collapse the two before the
    # vocabulary check and make the token permanently invisible.
    token = str(entry.get('status')).strip().lower()
    if token in _OK_STATUSES:
        return 'ok', ''
    if token in _TRAIL_NOT_DISPATCHABLE_STATUSES or token == _TRAIL_DENIED_STATUS:
        # DECLARED and deliberately ungradeable: the call never left, so it is neither an observed
        # success nor a failed dispatch, and it abstains like any token this reader cannot decide.
        # But it comes back with NO token to report: ``tool_status_unrecognized`` is an ask to
        # WIDEN a vocabulary, and asking an operator to widen to a word the contract already names
        # would fire on every gated verb of every run.
        return 'unknown', ''
    return 'unknown', token


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _trail_identity(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    """One trail entry's identity: the ``call_id`` it declares, else its shape (tool, args, result).

    The same SHAPE as :func:`union_by_call_identity`'s rule between lanes — an id first, a weaker
    fallback second — though the fallback here is an identical shape, not name + count.
    """
    call_id = entry.get('call_id')
    if isinstance(call_id, str) and call_id:
        return ('id', call_id)
    return ('shape', entry.get('tool'), _canonical(entry.get('args')), _canonical(entry.get('result')))


def seeded_trail_identities(seed_agent_state: Any) -> Counter[tuple[Any, ...]]:
    """How many trail entries of each identity a resume seed's ``agent_state`` already carried."""
    trail = seed_agent_state.get(AGENT_STATE_TOOL_TRAIL_KEY) if isinstance(seed_agent_state, Mapping) else None
    if not isinstance(trail, list):
        return Counter()
    return Counter(_trail_identity(entry) for entry in trail if isinstance(entry, Mapping))


def strip_seeded_trail(agent_states: Any, seed_agent_state: Any) -> tuple[Any, int]:
    """``agent_states`` with the resume seed's own trail entries removed from EVERY turn, and how many went.

    On a trace-resumed run the seed's ``agent_state`` is restored into the agent verbatim
    (``seed_state``), and a harness that keeps its ``tool_trail`` in graph state then re-reports
    the PRIOR session's dispatches on turn 0 — and on every later turn when the trail is
    cumulative. The message lane is sliced at the seeded boundary; this lane is indexed by this
    run's turns, so position cannot express the boundary and the seed's own entries are removed by
    identity instead. Without it every count helper over ``tool_calls`` over-reports, a
    ``len(tool_calls) >= 1`` guard admits a run that touched no tool, and the name-settle between
    the two out-of-band lanes can drop a genuine live call that shares a name with a restored one.

    COUNT-BOUNDED, per turn: each identity is removed at most as many times as the seed carried
    it, so a harness that re-reports the seed's entries once per turn loses exactly those, and a
    call the agent genuinely repeated — same tool, args and result, no ``call_id`` to tell it
    apart — keeps every occurrence beyond the seed's count. The residual runs the other way:
    with no ids, ONE live repeat of a seeded shape is indistinguishable from the re-report and is
    the occurrence dropped. Ids close that gap, which is why the trail contract asks for them.

    Non-list states and states without a trail pass through untouched; a turn whose trail lost
    nothing keeps its original mapping. ``0`` dropped means the lane was not bounded — either there
    was no seed trail to bound it by, or the harness did not echo it.
    """
    seeded = seeded_trail_identities(seed_agent_state)
    if not seeded or not isinstance(agent_states, list):
        return agent_states, 0
    out: list[Any] = []
    dropped = 0
    for state in agent_states:
        trail = state.get(AGENT_STATE_TOOL_TRAIL_KEY) if isinstance(state, Mapping) else None
        if not isinstance(trail, list):
            out.append(state)
            continue
        budget = Counter(seeded)
        kept: list[Any] = []
        for entry in trail:
            identity = _trail_identity(entry) if isinstance(entry, Mapping) else None
            if identity is not None and budget[identity] > 0:
                budget[identity] -= 1
                dropped += 1
                continue
            kept.append(entry)
        out.append({**state, AGENT_STATE_TOOL_TRAIL_KEY: kept} if len(kept) != len(trail) else state)
    return out, dropped


def _args_dict(args: Any) -> dict[str, Any]:
    """A self-reported call's ``args`` as a dict: a dict as-is; a JSON string that decodes to an
    object (crewAI's ``ToolUsageStartedEvent.tool_args`` is one) decoded; anything else ``{}``."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, RecursionError):
            return {}
    return args if isinstance(args, dict) else {}


def trail_tool_calls(agent_states: Any) -> list[RecoveredToolCall]:
    """Build ``tool_call``-shaped records from the agent's self-reported per-turn tool trail.

    Takes the ``agent_states`` LIST built BY TURN INDEX — the orchestrator's
    ``transcript_turns[].agent_state``, appended once per turn — so the position IS the turn
    number, and these records carry the ``turn`` a whole-transcript message recovery cannot.
    One turn's ``agent_state`` mapping is accepted too, for a per-turn caller; there the
    position carries no such meaning and every record reads ``turn 0``.

    ``_recovered`` is stamped for the same reason the message lane stamps it: the record did not
    come through ``/mock/execute``, so nothing here is evidence of a write. ``state_delta`` is
    therefore EMPTY rather than the message lane's synthetic ``{'type': name}`` — the backend's
    revision count falls back to counting non-empty deltas when no authoritative call
    exists, and on a trail-only run every call would count as a revision.
    """
    if isinstance(agent_states, dict):
        agent_states = [agent_states]
    if not isinstance(agent_states, list):
        # ``None`` is the documented default — no trail reported, nothing to say. Any OTHER
        # shape is a caller handing this reader something it cannot walk, which zeroes the only
        # lane a self-looping topology has, so it must not pass in silence.
        if agent_states is not None:
            logger.warning(
                'agent_states is %s, not a list of per-turn state dicts — the whole tool trail is DROPPED (possible lost tool calls).',
                type(agent_states).__name__,
            )
        return []
    out: list[RecoveredToolCall] = []
    # Entries the harness DECLARED but this reader could not use. Counted by reason and reported
    # once per call rather than per entry: the producer is the customer's own simulate.py, so a
    # renamed key ('name' for 'tool') would silently eat the entire trail — the blind spot this
    # lane exists to close, one JSON key away. Aggregated so a large drifted trail is one line,
    # not one per entry.
    unusable: Counter[str] = Counter()
    for turn_idx, state in enumerate(agent_states):
        if not isinstance(state, dict):
            continue
        trail = state.get(AGENT_STATE_TOOL_TRAIL_KEY)
        if trail is None:
            continue
        if not isinstance(trail, list):
            # The key is PRESENT: the harness positively declared a trail. No ambiguity with
            # "this harness reports none", so this one is always worth saying out loud.
            logger.warning(
                "agent_state[%r] on turn %d is %s, not a list — that turn's tool trail is DROPPED (possible lost tool calls).",
                AGENT_STATE_TOOL_TRAIL_KEY,
                turn_idx,
                type(trail).__name__,
            )
            continue
        for entry in trail:
            if not isinstance(entry, dict):
                unusable[f'not-a-dict:{type(entry).__name__}'] += 1
                continue
            name = entry.get('tool')
            if not isinstance(name, str) or not name:
                unusable['no-usable-tool-name'] += 1
                continue
            args = entry.get('args')
            refusal = _trail_refusal(entry)
            # A refusal is decided before the status vocabulary: the attempt's outcome is known
            # whatever token the harness stamped beside it (``error`` is the common one).
            status, token = ('unknown', '') if refusal is not None else _trail_status(entry)
            record = RecoveredToolCall(
                type='tool_call',
                tool=name,
                args=_args_dict(args),
                id=str(entry.get('call_id') or ''),
                status=status,
                response=entry.get('result'),
                state_delta={},
                new_entities={},
                turn=turn_idx,
                _recovered=True,
                _lane=TRAIL_LANE,
            )
            if token:
                # The harness DID report on this dispatch — this reader could not read what it
                # said. Both halves of that are stamped, so the abstention is attributable and
                # the token is countable (:func:`tool_status_unrecognized`).
                record['status_token'] = token
                record['error_check'] = 'unchecked'
            if refusal is None and status == 'unknown' and str(entry.get('status')).strip().lower() in _TRAIL_NOT_DISPATCHABLE_STATUSES:
                # The abstention keeps grading neutral; the marker keeps the FACT, so the run's
                # dispatch-error count can still see a call that never left.
                record['_not_dispatchable'] = True
            if refusal is not None:
                record['_denied'] = True
                record['refusal'] = refusal
            if entry.get('source') in (PRODUCT_SOURCE, MOCK_SOURCE):
                record['source'] = entry['source']
            if isinstance(parent := entry.get(TRAIL_PARENT_KEY), str) and parent.strip():
                record['parent_id'] = parent.strip()
            if isinstance(omitted := entry.get(TRAIL_ARGS_OMITTED_KEY), list) and omitted:
                record['args_omitted'] = [str(name) for name in omitted]
            out.append(record)
    if unusable:
        logger.warning(
            '%d tool_trail entr(ies) were dropped as unusable (possible lost tool calls); each must be a dict with a non-empty %r name. By reason: %s',
            sum(unusable.values()),
            'tool',
            dict(unusable),
        )
    _warn_if_trail_is_cumulative(out)
    return out


def _warn_if_trail_is_cumulative(records: list[RecoveredToolCall]) -> None:
    """Warn when one ``call_id`` appears on more than one turn.

    The lane reads ``agent_state`` as a PER-TURN report, and nothing enforces that. A harness
    whose state accumulates re-lists every earlier call each turn, which inflates the shipped
    count quadratically — the same shape the message lane already defends against with
    ``seen_tool_call_ids`` (see ``orchestrator._record_turn``). Deduping here would diverge from
    the backend, which does not, so this reports rather than repairs: an inflated count is at
    least attributable instead of arriving as a mystery. Ids are optional, so a trail without
    them is simply unchecked.
    """
    turns_by_id: dict[str, set[int]] = {}
    for record in records:
        call_id = record.get('id')
        if call_id:
            turns_by_id.setdefault(call_id, set()).add(record.get('turn', 0))
    repeated = {cid: sorted(turns) for cid, turns in turns_by_id.items() if len(turns) > 1}
    if repeated:
        logger.warning(
            '%d tool_trail call_id(s) appear on more than one turn, so %r looks CUMULATIVE rather than per-turn — '
            'the reported tool-call count is inflated. Report only the calls that turn issued. Repeated ids (turns): %s',
            len(repeated),
            AGENT_STATE_TOOL_TRAIL_KEY,
            dict(list(repeated.items())[:5]),
        )


def _one_row_per_turn_call(records: list[RecoveredToolCall]) -> list[RecoveredToolCall]:
    """``records`` with a repeat of one ``(tool, id)`` on the SAME turn kept once: within a turn a
    call id names one call, so a repeat is two recorders feeding one trail (a harness's own hooks
    beside ``plurai.confine``'s). A repeat across turns is a cumulative trail, which
    :func:`_warn_if_trail_is_cumulative` reports and this keeps. Records with no id are kept."""
    seen: set[tuple[Any, Any, str]] = set()
    kept: list[RecoveredToolCall] = []
    for record in records:
        if call_id := record.get('id'):
            key = (record.get('turn'), record.get('tool'), call_id)
            if key in seen:
                continue
            seen.add(key)
        kept.append(record)
    return kept


def _settle_out_of_band_lanes(message_calls: list[RecoveredToolCall], trail_calls: list[RecoveredToolCall]) -> list[RecoveredToolCall]:
    """Union the two out-of-band lanes, keeping the TRAIL record when both describe one tool.

    The trail is the engine's own log — it carries the RESULT and therefore the true status,
    where a transcript walk of the same tool often has neither. Deduped by tool NAME, which is
    the only key the two lanes share: a trail entry's ``call_id`` is the harness's own, not the
    framework's, so an id match is not expected to hit.

    NAME, not count — which means a tool the trail mentions ONCE discards every message-lane
    record of that tool, and on a hybrid agent (framework-visible calls plus a partial
    self-report) adding this lane can LOWER the reported count. The backend applies the same rule,
    and the two halves counting one payload by different arithmetic is a worse failure than the
    undercount, so this side does not diverge alone: a count-based settle belongs in one
    synchronized change on both sides.
    """
    trail_tool_names = {t.get('tool') for t in trail_calls}
    trail_counts = Counter(t.get('tool') for t in trail_calls)
    dropped = Counter(m.get('tool') for m in message_calls if m.get('tool') in trail_tool_names)
    if outnumbered := {name: (count, trail_counts[name]) for name, count in dropped.items() if count > trail_counts[name]}:
        logger.warning(
            'the two out-of-band lanes disagree on call count for %s (message-lane count, trail count); the trail wins by '
            'name, so the surplus message-lane records are dropped and the reported tool-call count falls.',
            outnumbered,
        )
    return trail_calls + [m for m in message_calls if m.get('tool') not in trail_tool_names]


def verb_args_identity(call: Mapping[str, Any]) -> tuple[str, Any]:
    """WHAT a call is — its verb and its arguments AS PARSED — compared with ``==``.

    ``dict.__eq__`` is recursive, key-order-insensitive and numeric, so the chain's ``tool_calls``
    and the ``agent_actions`` declaration (separate client code, either possibly JSON round-tripped)
    collapse to one identity without a canonicalizer. The verb is ``name`` on a chain call or an
    action and ``tool`` on a record this module built. The backend applies the same rule.
    """
    return (str(call.get('name') or call.get('tool') or ''), call.get('args') or {})


def action_tool_calls(agent_actions_by_turn: Any) -> list[RecoveredToolCall]:
    """``tool_call``-shaped records from the typed ``agent_actions`` channel, turn-indexed.

    What the agent DID beside what it said — a render, a handoff. The transcript renders this
    channel, so a union that omits it counts a render-only run as zero tool activity. The position
    IS the turn, as on the trail lane; ``status`` is the action's own (``unknown`` when the client
    saw no result); ``state_delta`` is empty for the trail lane's reason. An action the harness typed
    ``source: 'product'`` (:data:`PRODUCT_SOURCE`) keeps it; every other record carries no such key.
    """
    if not isinstance(agent_actions_by_turn, list):
        return []
    out: list[RecoveredToolCall] = []
    for turn_idx, actions in enumerate(agent_actions_by_turn):
        for action in actions if isinstance(actions, list) else []:
            if not isinstance(action, Mapping) or not isinstance(action.get('name'), str) or not action['name']:
                continue
            args, status = action.get('args'), action.get('status')
            record = RecoveredToolCall(
                type='tool_call',
                tool=action['name'],
                args=_args_dict(args),
                id='',
                status=status if isinstance(status, str) and status else 'unknown',
                response=None,
                state_delta={},
                new_entities={},
                turn=turn_idx,
                _recovered=True,
                _lane=ACTION_LANE,
            )
            if action.get('source') == PRODUCT_SOURCE:
                record['source'] = PRODUCT_SOURCE
            out.append(record)
    return out


def _actions_not_on(actions: list[RecoveredToolCall], records: list[Any]) -> list[RecoveredToolCall]:
    """The action records no other lane already carries, matched ONE-FOR-ONE by :func:`verb_args_identity`.

    Identity alone settles this lane: an action always carries its parsed args, so it never needs
    the name rule, and a render the chain also carries (or the mock also logged) is one call.
    """
    pool = [verb_args_identity(record) for record in records]
    out: list[RecoveredToolCall] = []
    for action in actions:
        identity = verb_args_identity(action)
        if identity in pool:
            pool.remove(identity)
            continue
        out.append(action)
    return out


def recovered_tool_calls(chatbot_messages: list | None, agent_states: Any = None, *, agent_actions_by_turn: Any) -> list[RecoveredToolCall]:
    """Every tool call recoverable OUTSIDE the mock event log: the transcript walk plus the
    self-reported trail, settled by :func:`_settle_out_of_band_lanes`, plus the typed action lane
    settled against both by call identity. ``agent_actions_by_turn`` is REQUIRED: a caller with no
    action channel passes ``None`` and says so."""
    settled = _settle_out_of_band_lanes(derive_tool_calls_from_messages(chatbot_messages or [], open_calls={}), trail_tool_calls(agent_states))
    return settled + _actions_not_on(action_tool_calls(agent_actions_by_turn), settled)


def _call_identity(record: dict[str, Any]) -> str | None:
    """A call record's DECLARED id, when it carries a usable one.

    Read from ``call_id`` (the field a harness trail and the mock event log write) and then
    ``id`` (the field every record normalized by this module carries), so one predicate answers
    for the raw ``tool_events`` events and for the recovered records alike.

    A blank or non-string id is NOT an identity. A shared id is read below as "the same call",
    so one empty string on each lane would delete a real dispatch; absent is the honest answer,
    and the name rule handles it loudly.
    """
    for key in ('call_id', 'id'):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _logged_call_failed(event: Mapping[str, Any]) -> bool:
    """Whether a LOGGED event is a failed call, read by the SAME rule that stamps a message-lane row
    ``error`` (:func:`_tool_response_errored`, over the response as text), so the union compares the
    two lanes in one vocabulary. A narrower reader on this side would leave a mock-returned failure
    the message lane reads as one (a drifted status, a refusal, a bare error string) out of the
    failure budget, and its message row would then be kept beside it: one dispatch, two calls.
    """
    response = event.get('response')
    try:
        text = response if isinstance(response, str) else '' if response is None else json.dumps(response, default=str, ensure_ascii=False)
    except ValueError:
        # A cyclic response (a record that holds its own parent) has no JSON text; its repr reads the same tokens.
        text = str(response)
    return (
        str(event.get('status') or '').lower() == 'error'
        or _tool_response_errored({}, text)
        # The STAMPED lane's other reader: a drifted top-level status or a refusal reads failed
        # here too, so a row the agent's wrapper stamped ``error`` meets it in the budget.
        or _envelope_status_token(response) in _TRAIL_FAILED_STATUSES
        or (isinstance(response, dict) and response.get('refused') is True)
    )


def union_by_call_identity(events: list[dict[str, Any]], recovered: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[Any, int]]:
    """``(merged tool_call records, {tool: rows the name rule dropped unverified})``.

    IDENTITY FIRST. Where a tool's event log declares ids, a recovered row duplicates a logged
    call only when it NAMES that call, and every other row is a call in its own right. A
    ``mode: mock`` tool whose agent retries internally on a 5xx is the topology that rule exists
    for: the mock logs the one call it saw and the trail carries the other retries. A rule that
    dedupes on the tool NAME deletes all of them; a rule that dedupes on a COUNT deletes one
    arbitrary dispatch that may be none of them. Only a comparison of declared ids can say which
    row is which, so where both lanes declare one, that is the comparison.

    Identity is scoped PER TOOL. A call id is unique within the lane that minted it, not across
    lanes, so a harness that numbers its dispatches per tool would otherwise have one tool's
    ``c1`` delete another's.

    NAME + COUNT is the fallback, and ONLY for rows identity cannot answer: a row carrying no id,
    or one whose tool's event log declares none (a trail id compared against a log that carries
    none matches nothing, so opening the id lane on that comparison would count every logged call
    twice — and an over-count feeds the dead-tool verdict). There the rule drops as many rows as
    there are logged events still unaccounted for and keeps the rest. It cannot tell a duplicate
    from a retry, so a drop can be a suppressed call, and a floor a reader cannot see is
    indistinguishable from a measurement.

    A DROP IS RECORDED IN ``floored`` IN EXACTLY TWO STATES, and is silent otherwise. (a) An
    identity existed on ONE lane and not the other: the lanes could have been matched by id and
    were not, and the remedy is to emit ``call_id`` on both. (b) The tool's trail rows OUTNUMBER
    its logged events: the surplus proves the two lanes do not fully overlap, so the rows the
    count rule dropped may be retries rather than duplicates — the very topology this union was
    built for, arriving with no ids to prove it. Where NEITHER lane declares an id AND the trail
    is no longer than the log, the count rule is the only rule that applies and dropping
    exactly as many rows as the log holds is its documented reconciliation, not a suppression: a
    floor stamped on every ordinary run is noise that hides the real ones, and the next real one
    gets skimmed past.

    A FAILED MESSAGE-LANE ROW spends only a logged FAILURE. The agent framework's own dispatcher
    can fail a call before it reaches any tool — a name its tool node does not carry ("not a
    valid tool"), args its schema rejects — and the mock then logs nothing for it. Charged
    against the count, such a row is dropped in place of the retry that DID run, and the retry
    comes back as a second copy of the logged success: the failed attempt vanishes and the success
    counts twice. So an errored message-lane row is a duplicate only while the tool's log still
    holds an unclaimed errored event, and is otherwise kept as the attempt it was. A row that did
    not fail spends the log's successes first. The trail lane is not read this way: its error is
    the harness's verdict on a dispatch, and its never-left case has its own marker.

    AN ID-LESS LOGGED EVENT IS STILL A CALL ON BOTH LANES. Where a tool's log declares ids for
    only SOME of its events (a call that reached the mock without the active call id, such as one
    made inside a delegated worker thread), a recovered row whose id names no logged event is
    first matched to the earliest unclaimed id-less event of the same tool and args
    (:func:`verb_args_identity`) and is one call with it, not a second one. The match is by
    args, not identity, so its drop is a floor by limb (a). A row that matches none is a call in
    its own right.

    What this rule deliberately does NOT do is dedupe the recovered lane against ITSELF. A
    harness that reports one dispatch on several turns is a CUMULATIVE trail, which already has
    an owner that says so out loud; collapsing it here would be a second owner for that contract,
    quietly contradicting the warning while it is still being printed.
    """
    #: Ids the EVENT LOG declared, per tool. Non-empty for a tool is what opens the id lane.
    logged_ids: dict[Any, set[str]] = {}
    #: Where each logged ``(tool, id)`` sits in ``merged``, so a row naming it can hand its marker over.
    logged_at: dict[tuple[Any, str], int] = {}
    #: Logged events not yet claimed by a recovered row — the budget the name rule spends.
    unaccounted: dict[Any, int] = {}
    #: Of those, the logged FAILURES still unclaimed — the only budget a failed message-lane row spends.
    failures_unaccounted: dict[Any, int] = {}
    #: Logged events that declare NO id and are not yet matched, per tool, in log order.
    idless: dict[Any, list[int]] = {}
    for index, event in enumerate(events):
        name = event.get('tool')
        unaccounted[name] = unaccounted.get(name, 0) + 1
        if _logged_call_failed(event):
            failures_unaccounted[name] = failures_unaccounted.get(name, 0) + 1
        if call_id := _call_identity(event):
            logged_ids.setdefault(name, set()).add(call_id)
            logged_at.setdefault((name, call_id), index)
        else:
            idless.setdefault(name, []).append(index)
    #: How many events each tool logged, kept whole. The floor rule's (b) limb compares the SIZE
    #: of the two lanes, which ``unaccounted`` stops being the moment the first row is charged.
    logged_total: dict[Any, int] = dict(unaccounted)
    #: How many rows the recovered lane carries per tool — the other half of that comparison.
    trailed_total: dict[Any, int] = {}
    for record in recovered:
        trailed_total[record.get('tool')] = trailed_total.get(record.get('tool'), 0) + 1
    seen: dict[Any, set[str]] = {name: set(ids) for name, ids in logged_ids.items()}
    merged: list[dict[str, Any]] = list(events)
    floored: dict[Any, int] = {}
    for record in recovered:
        name = record.get('tool')
        if record.get('_unpaired'):
            # A result the message lane could attach to no call (``derive_tool_calls_from_messages``
            # rule b): the tool fired — that much is a floor — but its args were never seen, and a
            # record with ``args: {}`` in the merged list is graded as a call with EMPTY args. It
            # spends the log's budget first like every other row: a logged call with no id that
            # this result belongs to is accounted, not floored twice.
            if unaccounted.get(name, 0) > 0:
                unaccounted[name] -= 1
            else:
                floored[name] = floored.get(name, 0) + 1
            continue
        call_id = _call_identity(record)
        if record.get('_denied'):
            # A denied call never ran, so no logged EXECUTION is its duplicate, whatever id it shares
            # (a harness that numbers its calls per turn reuses them): it is kept as the attempt it
            # was and never spends the log's budget — spending it would drop the call that DID run.
            # Only a CACHED row of the same call (a trail row the SDK already shipped) is the same
            # record, and the denial is carried onto it. The backend applies the same rule.
            at = logged_at.get((name, call_id)) if call_id is not None else None
            if at is not None and merged[at].get('_recovered'):
                unaccounted[name] = unaccounted.get(name, 0) - 1
                merged[at] = {**merged[at], '_denied': True, **({_REFUSAL_KEY: record[_REFUSAL_KEY]} if _REFUSAL_KEY in record else {})}
            else:
                merged.append(record)
            continue
        if call_id is not None and name in logged_ids:
            if call_id in seen[name]:
                # THE logged call, named by its own id — one dispatch reported on two lanes. The kept
                # row carries ``_not_dispatchable`` when EITHER lane says the call never left.
                unaccounted[name] = unaccounted.get(name, 0) - 1
                if (logged := logged_at.get((name, call_id))) is not None and _logged_call_failed(events[logged]):
                    failures_unaccounted[name] = failures_unaccounted.get(name, 0) - 1
                if record.get('_not_dispatchable') is True and (at := logged_at.get((name, call_id))) is not None:
                    merged[at] = {**merged[at], '_not_dispatchable': True}
                continue
            pending = idless.get(name, [])
            match = next((pos for pos, at in enumerate(pending) if verb_args_identity(events[at]) == verb_args_identity(record)), None)
            if match is not None and unaccounted.get(name, 0) > 0:
                # An id-less logged event with this row's tool and args: the same call (see above).
                at = pending.pop(match)
                unaccounted[name] -= 1
                if _logged_call_failed(events[at]):
                    failures_unaccounted[name] = failures_unaccounted.get(name, 0) - 1
                floored[name] = floored.get(name, 0) + 1
                continue
            seen[name].add(call_id)
            merged.append(record)
            continue
        failed = record.get('_lane') not in (TRAIL_LANE, ACTION_LANE) and str(record.get('status') or '').lower() == 'error'
        if failed and min(failures_unaccounted.get(name, 0), unaccounted.get(name, 0)) <= 0:
            # Failed before the mock saw it (see above): no logged event is its duplicate.
            merged.append(record)
            continue
        if unaccounted.get(name, 0) > 0:
            unaccounted[name] -= 1
            # A failed row claims a logged failure; any other row claims a logged success first.
            failures_unaccounted[name] = failures_unaccounted.get(name, 0) - 1 if failed else min(failures_unaccounted.get(name, 0), unaccounted[name])
            # The two-limbed floor rule stated above: an id on one lane only, or a trail longer
            # than the log. Neither, and this drop is the reconciliation, not a finding.
            if call_id is not None or name in logged_ids or trailed_total.get(name, 0) > logged_total.get(name, 0):
                floored[name] = floored.get(name, 0) + 1
            continue
        merged.append(record)
    return merged, floored


def _decision_key(record: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """What ONE call occurrence is matched on: its verb, its declared id and its args, canonical."""
    call_id, tool = _call_identity(dict(record)), record.get('tool')
    if call_id is None or not isinstance(tool, str):
        return None
    return (tool, call_id, _canonical(record.get('args') or {}))


def _decisions(events: list[Any]) -> dict[tuple[tuple[str, str, str], int], dict[str, Any]]:
    """``{(occurrence key, ordinal): event}`` for every :data:`TOOL_DECISION_EVENT`."""
    out: dict[tuple[tuple[str, str, str], int], dict[str, Any]] = {}
    for event in events:
        if isinstance(event, dict) and event.get('type') == TOOL_DECISION_EVENT and isinstance(event.get('occurrence'), int):
            if (key := _decision_key({'tool': event.get('tool'), 'call_id': event.get('call_id'), 'args': event.get('args')})) is not None:
                out[(key, event['occurrence'])] = event
    return out


def _apply_decisions(records: list[dict[str, Any]], decisions: dict[tuple[tuple[str, str, str], int], dict[str, Any]]) -> list[dict[str, Any]]:
    """Each decision stamped on the message-lane OCCURRENCE it names: the n-th recovered record of that
    verb, id and args in thread order. An id alone names no call — a harness numbering per turn
    reuses it — so a denial of one ``call_0`` never marks a later, approved ``call_0``. A logged
    execution (not ``_recovered``) is never marked: it is evidence the call ran.

    ``denied``: ``unknown`` + ``_denied`` + a ``user`` ``refusal`` (``_not_dispatchable``'s shape: abstain
    on the status, keep the fact on a marker; ``response`` keeps the harness's text). ``undecided``: ``unknown`` +
    ``error_check: 'unchecked'`` + ``_undecided`` — the result arrived, but whether it is the tool's
    or the gate's cannot be read, so it is never ``ok``. ``approved`` leaves the record as it is.
    """
    ordinal: Counter[tuple[str, str, str]] = Counter()
    out: list[dict[str, Any]] = []
    for record in records:
        message_lane = record.get('_recovered') and record.get('_lane') not in (TRAIL_LANE, ACTION_LANE) and not record.get('_unpaired')
        key = _decision_key(record) if message_lane else None
        event = None
        if key is not None:
            event = decisions.get((key, ordinal[key]))
            ordinal[key] += 1
        out.append(_as_decided(record, event) if event is not None else record)
    return out


def _as_decided(record: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """ONE record with the decision that names it applied (see :func:`_apply_decisions`)."""
    decision = event.get('decision')
    if decision == DECISION_DENIED:
        return {**_as_refused(record, _refusal('user', _NOT_APPROVED)), 'decision_turn': event.get('turn')}
    if decision == DECISION_UNDECIDED:
        drop, stamp = ('status_token',), {'status': 'unknown', 'error_check': 'unchecked', '_undecided': True}
    else:
        return record
    return {**{k: v for k, v in record.items() if k not in drop}, **stamp, 'decision_turn': event.get('turn')}


def merged_tool_calls_with_floor(
    tool_events: dict | None, chatbot_messages: list | None, agent_states: Any = None, *, agent_actions_by_turn: Any
) -> tuple[list[dict[str, Any]], dict[Any, int]]:
    """``(merged tool_call records, {tool: rows the name rule dropped unverified})``.

    :func:`merged_tool_calls` with the union's SECOND value kept (:func:`union_by_call_identity`).
    Without it every surface that renders a count would publish a floor as an exact number: one
    logged call plus two unmatched trail rows would ship as "2 tool_call events", with nothing on
    the wire to say 2 is a lower bound. A count
    that is a floor has to say so WHERE IT IS READ, so the floor survives the call here and
    :func:`merged_tool_calls` stays the flat one-liner its consumers want.

    A caller that RENDERS, counts, or asserts on the length of the list wants this function; a
    caller that only inspects the records (which tool, which args, what came back) wants the flat
    one. An empty mapping means the name rule suppressed nothing — the healthy path, and it must
    stay silent, because a floor stamped on every ordinary run is noise that hides the real ones.
    """
    events = (tool_events or {}).get('events') or []
    event_calls: list[dict[str, Any]] = [e for e in events if isinstance(e, dict) and e.get('type') == 'tool_call']
    decisions = _decisions(events)
    # Each recovered record is a tool_call-shaped dict and the event records are the
    # authoritative plain dicts from tool_events, so the union is a list[dict[str, Any]].
    recovered = [dict(m) for m in recovered_tool_calls(chatbot_messages, agent_states, agent_actions_by_turn=agent_actions_by_turn)]
    if any(r.get('_lane') not in (TRAIL_LANE, ACTION_LANE) for r in recovered):
        # A ``_recovered`` MESSAGE-lane record in the event log is a CACHED derivation of the same
        # message chain this call just re-derived — the runner persists its recovery as
        # ``tool_events.events`` on a ``mode: real`` run, so a saved run carries whatever the
        # runner's version of this module made of the messages. Where
        # the messages are here to re-derive from, the fresh derivation supersedes the cached
        # one; where they are not (an empty or text-only chain yields no message-lane record),
        # the cached rows stay, because they are then the only evidence. Trail-lane rows are
        # never cached derivations of the messages and always stay.
        event_calls = [e for e in event_calls if not (e.get('_recovered') and e.get('_lane') != TRAIL_LANE)]
    # The user's answer to an interrupt, carried by call OCCURRENCE onto the record it answered
    # (:func:`_apply_decisions`). BEFORE the union, which must not spend an event's budget on a
    # call that never ran. The whole-chain re-derive above pairs the harness's answer with its call
    # and reads it as the tool's ``ok`` — the executed-denied false green — so the typed decision
    # overrides it here. The backend applies the same rule.
    if decisions:
        event_calls = _apply_decisions(event_calls, decisions)
        recovered = _apply_decisions(recovered, decisions)
    # The action lane never reaches the name rule: settled by identity, in ONE pass, against the union's result.
    # Settled against the chain rows the union CONSUMED as well: a harness mocking below the tool logs other args than the agent declared.
    lanes = [r for r in recovered if r.get('_lane') != ACTION_LANE]
    merged, floored = union_by_call_identity(event_calls, lanes)
    return merged + [
        dict(a) for a in _actions_not_on(action_tool_calls(agent_actions_by_turn), merged + [r for r in lanes if not any(r is m for m in merged)])
    ], floored


def merged_tool_calls(tool_events: dict | None, chatbot_messages: list | None, agent_states: Any = None, *, agent_actions_by_turn: Any) -> list[dict[str, Any]]:
    """Return the UNION of ``tool_call`` records from tool_events.events and from the
    out-of-band lanes (:func:`recovered_tool_calls`), deduped by CALL IDENTITY
    (:func:`union_by_call_identity`) across the event log and the recovered lanes — note the two
    out-of-band lanes are themselves settled by tool NAME in :func:`recovered_tool_calls` first,
    which can already have dropped a hybrid tool's retries.

    The one-liner every consumer wants: "give me the list of tool_call records, regardless of
    whether they came via /mock/execute or not." A mode:mock tool always emits a /mock/execute
    event; a mode:real / storage-adapter tool emits none and is only visible as an AIMessage
    tool_use or on the agent's own ``agent_state`` trail. A tool is not wholly one or the other,
    though — a mock-logged tool whose agent retries internally reports those retries on the trail
    alone — so the dedup compares DECLARED IDS and falls back to the tool name only where an id
    is missing on one of the two lanes. Across the two out-of-band lanes a trail and a transcript
    can both describe one tool; :func:`recovered_tool_calls` settles that in favour of the trail
    before we get here.
    The SDK consumer is the rollup's ``tool_calls`` metric; the backend builds the same union
    server-side, which is why the signature AND the dedup rule have to agree.
    """
    # The floor is dropped HERE and only here, and it is not thereby UNAVAILABLE: a caller that
    # renders a count calls :func:`merged_tool_calls_with_floor`, which is this function with the
    # union's second value kept.
    merged, _floored = merged_tool_calls_with_floor(tool_events, chatbot_messages, agent_states, agent_actions_by_turn=agent_actions_by_turn)
    return merged


def _sequence_events(events: list[Any], added: list[dict[str, Any]]) -> list[Any]:
    """Order the recovered lane INTO the mock lane by conversation turn.

    The two lanes reach this module already chronological within themselves — the mock
    lane is append-ordered by the engine, the recovered lane is message-ordered — so the
    merge is a two-lane interleave, not a global re-sort. The rules differ per lane, and
    deliberately:

    * MOCK LANE: its own order is the authoritative chronology, so its sort key is the
      RUNNING MAX of the turns seen so far. A stamp that is stale (a nested session that
      outlived the turn it was spawned in) or absent (an ``observe`` / ``state_update``
      event, an artifact whose events carry no stamps) therefore inherits its
      predecessor's position and can never reorder this lane against itself.
    * RECOVERED LANE: positioned by its OWN turn, which is the whole point — a real-tool
      call from turn 1 must not be serialized after turn 2's mock calls. A record with no
      turn (whole-transcript recovery) inherits the mock lane's last turn, keeping it at
      the end where plain concatenation put it, rather than being re-bucketed onto turn 0
      ahead of the real turn-0 events.

    The index is part of the key, so equal turns keep insertion order (mock lane first).
    """
    keyed: list[tuple[int, int, Any]] = []
    carried = 0
    for idx, event in enumerate(events):
        raw = event.get('turn') if isinstance(event, dict) else None
        if raw is not None:
            carried = max(carried, _coerce_turn(raw))
        keyed.append((carried, idx, event))
    for idx, record in enumerate(added, start=len(events)):
        raw = record.get('turn')
        keyed.append((_coerce_turn(raw) if raw is not None else carried, idx, record))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in keyed]


#: Marker a mock handler stamps into its response envelope when the handler itself raised.
_HANDLER_ERROR_KEY = '_plurai_handler_error'


def tool_event_errored(event: dict[str, Any]) -> bool:
    """Whether ONE ``tool_call`` event represents a FAILED call, in EITHER lane.

    A ``mode:real`` tool's failure is recorded as a top-level ``status: 'error'`` (recovered
    from the authoritative ``ToolMessage.status``). A ``mode:mock`` tool's failure is not:
    ``/mock/execute`` dispatched the call successfully, so the event carries top-level
    ``status: 'ok'`` and the failure lives one level down in the response envelope.

    The two are the same fact and this is the single predicate for it. A refused attempt (a
    ``refusal`` on the event, or one its mock envelope declares) is a decision, not a failed call.
    """
    if _REFUSAL_KEY in event or mock_refusal(event.get('response')) is not None:
        return False
    if str(event.get('status') or '').lower() == 'error':
        return True
    response = event.get('response')
    if isinstance(response, dict):
        # STRUCTURED envelope only — read the fields, never a repr, and never a bare
        # ``error`` key: mocks routinely carry ``{'error': None}`` on success, and reading
        # that by truthiness marks every clean call failed.
        return str(response.get('status') or '').lower() == 'error' or bool(response.get(_HANDLER_ERROR_KEY))
    return False


def normalize_tool_event_status(events: object) -> list[dict[str, Any]]:
    """``events`` with every ``tool_call``'s top-level ``status`` speaking ONE vocabulary.

    The two recording lanes disagree: a mocked tool's error leaves ``status: 'ok'`` wrapping
    an error envelope while a live tool's error sets ``status: 'error'``. Every consumer that
    reads the field — the rollup metrics, the span renderers and the backend's own readers
    — would otherwise be correct on one lane and silently vacuous on
    the other, with no verdict able to tell the difference from a clean run.

    Normalization is ADDITIVE and reversible: only ``status`` moves, ``response`` is left
    exactly as recorded, and a rewritten event carries ``status_source:
    'response_envelope'`` so the change is attributable and a reader can still recover the
    raw lane value. Events are copied, never mutated in place — the input may be the
    caller's own payload.

    A mocked call the mock REFUSED (:func:`mock_refusal`) is the other rewrite, on the same terms:
    it settles as the refused attempt it was (:func:`_as_refused`, ``status_source: 'refusal'``), so
    every lane records a refusal in one shape and a permission ``no`` never reads as a broken tool.
    """
    out: list[dict[str, Any]] = []
    for event in list(events or []):
        if not isinstance(event, dict):
            out.append(event)
            continue
        if event.get('type') == 'tool_call' and _REFUSAL_KEY not in event and (refusal := mock_refusal(event.get('response'))) is not None:
            out.append({**_as_refused(event, refusal), 'status_source': 'refusal'})
            continue
        if event.get('type') != 'tool_call' or str(event.get('status') or '').lower() == 'error' or not tool_event_errored(event):
            out.append(event)
            continue
        out.append({**event, 'status': 'error', 'status_source': 'response_envelope'})
    return out


#: Payload key :func:`merge_recovered_tool_calls` stamps when the union had to fall back to the
#: tool NAME: ``{'suppressed': <rows dropped>, 'by_tool': {<tool>: <rows dropped>}}``.
#:
#: It counts DROPS, and it is written ONLY when there were some. A row the name rule ADMITTED —
#: because no logged event was left for it to be a duplicate of — is counted nowhere here: it is
#: in ``events``, which is the count. So the key present means "every number in this payload is a
#: floor for these tools"; the key absent means the name rule suppressed nothing, which is the
#: healthy path and must stay silent — a floor stamped on every clean payload is noise, and the
#: next real one gets skimmed past.
#:
#: ``suppressed`` is derived from ``by_tool`` on every write rather than accumulated separately:
#: a consumer asks "is this a floor at all" before it asks which tool, and two counters that can
#: disagree is one counter too many.
TOOL_CALL_COUNT_FLOOR_KEY = 'tool_call_count_floor'

#: Payload key carrying :func:`tool_status_unrecognized` — ``{raw token: rows}`` for statuses
#: neither vocabulary could read. Written ONLY when non-empty, for the same reason the floor key
#: is: a receipt stamped on every clean payload is noise, and the next real one gets skimmed past.
TOOL_STATUS_UNRECOGNIZED_KEY = 'tool_status_unrecognized'

#: Payload key carrying :func:`results_unobservable` — ``{call shape: rows}`` for shapes whose
#: results this reader structurally could not pair. It is the difference between "this agent's
#: calls did not complete" and "this transcript shape does not carry results", and every consumer
#: that grades a completion needs to be able to tell them apart. Same silence rule.
RESULTS_UNOBSERVABLE_KEY = 'results_unobservable'


def merge_recovered_tool_calls(
    tool_events: dict[str, Any] | None,
    messages: list[Any] | None = None,
    *,
    recovered_calls: list[RecoveredToolCall] | None = None,
    agent_states: Any = None,
) -> dict[str, Any] | None:
    """Return ``tool_events`` with out-of-band ``mode:real`` tool calls unioned into ``events``
    by CALL IDENTITY (:func:`union_by_call_identity`) and ordered by conversation turn.

    ABSENCE IS CARRIED, not normalised away. ``tool_events=None`` means no tool channel was
    read at all (the child held no plurai session, so it took no snapshot) — a different claim
    from ``{}`` / ``{'events': []}``, which is a channel that WAS read and recorded nothing.
    When the base is absent and the out-of-band lanes recovered nothing either, this returns
    ``None``: there is no observation to report, and minting ``{'events': []}`` would publish
    "this agent invoked zero tools" on a run where nobody was watching. Recover even one call
    and the answer becomes a dict again — that call IS an observation.

    The finalize-payload counterpart of :func:`merged_tool_calls`: instead of a
    flattened list it returns the SAME ``tool_events`` dict the client ships to
    ``/finalize`` (and writes to ``tool_events.json``), with the recovered records
    appended to ``events`` so the server's tool-trail policies see EVERY tool that
    fired, not just mode:mock ones.

    A ``mode:mock`` tool already emits one authoritative ``tool_call`` event PER call; a
    ``mode:real`` tool emits none and is fully recovered here. WHICH recovered records are the
    same calls as the authoritative ones is decided by their declared ids, not by arithmetic: a
    count rule drops one recovered record per authoritative event whether or not that record is
    the logged call, so a mock-logged tool the agent retried internally would lose a real
    dispatch for every event the mock happens to see. Where an id cannot answer, the count rule remains the
    fallback, and what it dropped is published under :data:`TOOL_CALL_COUNT_FLOOR_KEY` — the
    counts are then a floor, and this is a library, so it says so in the payload rather than on
    a stream nobody is reading.

    The union is ORDERED, not concatenated: ``events`` comes back as the turn-ordered
    interleave of the two lanes (see :func:`_sequence_events`), so a reader of
    ``tool_events.json`` — human or judge — can take the event order as the order the
    calls happened in. Pass ``recovered_calls`` (per-turn recovery, hence turn-stamped)
    to get that ordering; the ``messages`` fallback recovers over the whole transcript,
    which yields turn-less records that can only go at the end.

    ``agent_states`` adds the THIRD lane — the harness's self-reported trail, the only
    channel a self-looping agent has (:func:`trail_tool_calls`). Its records are always
    turn-stamped (the list is built by turn index), and they settle against the message
    lane before the union runs.
    """
    message_calls = recovered_calls if recovered_calls is not None else derive_tool_calls_from_messages(messages or [], open_calls={})
    # A step of the shipped payload only: ``trail_tool_calls`` itself stays one row per entry.
    recovered = _settle_out_of_band_lanes(message_calls, _one_row_per_turn_call(trail_tool_calls(agent_states)))
    # The user's answers to interrupts ride beside the calls, one typed record per decided call: a
    # consumer that re-derives the calls from the chain drops the cached rows, and the decision —
    # which no message spells — would otherwise go with them.
    return _union_into_payload(tool_events, [dict(record) for record in recovered], decisions=tool_decision_events(message_calls), sequence=_sequence_events)


def merge_pipeline_trail(tool_events: dict[str, Any] | None, trail: Any) -> dict[str, Any] | None:
    """``tool_events`` with a one-shot hop's recorded trail (``pipeline_io.outputs.tool_trail``) unioned in.

    A pipeline returns its trail beside its outputs, not on a turn's ``agent_state``, and it is the
    only record of a call that never reached ``/mock/execute`` — an in-process tool, or a wrapper
    tool around a mocked one. Its rows become the SAME ``tool_call`` records the dialog's trail lane
    makes (:func:`trail_tool_calls`), all on turn 0 because the whole hop is one logical turn, and
    are deduped against the mock's log by the same rule (:func:`union_by_call_identity`), so a call
    both lanes recorded counts once. Placed in CALL order (:func:`_in_trail_order`), not turn order.
    """
    records = [dict(record) for record in _one_row_per_turn_call(trail_tool_calls([{AGENT_STATE_TOOL_TRAIL_KEY: trail}]))]
    return _union_into_payload(tool_events, records, decisions=[], sequence=lambda events, added: _in_trail_order(records, events, added))


def _in_trail_order(trail: list[dict[str, Any]], events: list[Any], added: list[dict[str, Any]]) -> list[Any]:
    """``events`` with the ``added`` trail rows placed where the trail says they ran.

    Turn order cannot place them: every row and every logged call of a hop is turn 0, so
    :func:`_sequence_events` would put each admitted row after every logged call, and a
    ``called_before`` over a wrapper and the mocked call it made would read backwards. The trail
    IS the call order: a row the union dropped stands for a logged call (the next unclaimed one of
    its tool), and each admitted row goes right after the logged call its nearest earlier dropped
    row stands for — before the first logged call when no row precedes it.
    """
    admitted = {id(record) for record in added}
    unclaimed: dict[Any, list[int]] = {}
    for index, event in enumerate(events):
        if isinstance(event, dict) and event.get('type') == 'tool_call':
            unclaimed.setdefault(event.get('tool'), []).append(index)
    anchor = min((indices[0] for indices in unclaimed.values()), default=len(events)) - 1
    placed: dict[int, list[Any]] = {}
    for record in trail:
        if id(record) in admitted:
            placed.setdefault(anchor, []).append(record)
            admitted.discard(id(record))
        elif unclaimed.get(record.get('tool')):
            anchor = max(anchor, unclaimed[record.get('tool')].pop(0))
    out: list[Any] = list(placed.get(-1, []))
    for index, event in enumerate(events):
        out.append(event)
        out.extend(placed.get(index, []))
    # Anything admitted that is not a trail row keeps the turn-order rule's place: the end.
    return out + [record for record in added if id(record) in admitted]


def _union_into_payload(
    tool_events: dict[str, Any] | None,
    recovered: list[dict[str, Any]],
    *,
    decisions: list[dict[str, Any]],
    sequence: Callable[[list[Any], list[dict[str, Any]]], list[Any]],
) -> dict[str, Any] | None:
    """The finalize ``tool_events`` payload: ``recovered`` unioned into the mock log by call identity,
    ``sequence`` placing the admitted rows among the logged events, and the receipts stamped.

    The one body behind :func:`merge_recovered_tool_calls` (dialog: turn order) and
    :func:`merge_pipeline_trail` (one-shot hop: the trail's call order).
    """
    absent = not isinstance(tool_events, dict)
    base = dict(tool_events) if isinstance(tool_events, dict) else {}
    events = normalize_tool_event_status(base.get('events'))
    call_events = [e for e in events if isinstance(e, dict) and e.get('type') == 'tool_call']
    # ``merged`` opens with ``call_events`` verbatim (the union appends), so the tail is exactly
    # the records the union admitted — which is what has to be sequenced INTO the mock lane.
    merged, floored = union_by_call_identity(call_events, recovered)
    added = merged[len(call_events) :]
    if absent and not added:
        # Nothing was read and nothing was recovered: report the ABSENCE, not an empty trail.
        # Every receipt below (the floor, the unrecognized statuses, the unobservable results)
        # is computed over rows the union ADMITTED, so with none admitted there is nothing to
        # say here that ``None`` does not say better.
        return None
    # Normalized on EVERY return path, including the nothing-recovered one: this is the
    # recording vocabulary, not an artifact of whether recovery found anything. The RECOVERED
    # records go through the same normalizer as the base ones: ``trail_call_errored`` rules on
    # the dispatch and deliberately ignores the response envelope, so without this a trail
    # record answering with an error envelope would ship ``status: 'ok'`` while a byte-identical
    # mocked record shipped ``status: 'error'`` — the split-brain normalize_tool_event_status
    # exists to end, re-entering through this lane. Normalized AFTER sequencing (the normalizer
    # maps one-to-one and leaves a normalized event as it is), so ``sequence`` sees the very
    # records the union admitted.
    base['events'] = normalize_tool_event_status(sequence(events, added + decisions)) if added or decisions else events
    if floored:
        base[TOOL_CALL_COUNT_FLOOR_KEY] = {'suppressed': sum(floored.values()), 'by_tool': {str(name): n for name, n in floored.items()}}
    # Computed over the rows the union ADMITTED, not over everything recovered. A recovered row
    # the union dropped deduped against an AUTHORITATIVE ``/mock/execute`` event that already
    # says ``ok`` — it is not an abstention, and counting it would publish "this transcript carries
    # no tool results" on a clean mocked run whose result was sitting in the event log. These
    # receipts must be empty on the healthy path or the next real one gets skimmed past.
    admitted = [record for record in added if isinstance(record, dict)]
    if unrecognized := tool_status_unrecognized(admitted):
        base[TOOL_STATUS_UNRECOGNIZED_KEY] = unrecognized
    if unobservable := results_unobservable(admitted):
        base[RESULTS_UNOBSERVABLE_KEY] = unobservable
        blind = {shape: counts for shape, counts in unobservable.items() if not counts['decided']}
        logger.warning(
            'this run recovered tool calls whose outcome never settled (%s). Every such call is status "unknown" and '
            'every completion policy over it ABSTAINS rather than failing. A shape with decided:0 is a BLIND LANE — its '
            'results never reach the transcript at all (%s); emit them as a ToolMessage or an Anthropic tool_result '
            'block, or report them on agent_state.tool_trail. A shape with decided>0 has settled calls beside these, so '
            "these are this run's own unresolved ones — a pause awaiting approval, or a dropped result.",
            unobservable,
            blind or 'none',
        )
    return base


def _coerce_turn(raw: object) -> int:
    """Coerce a ``tool_events`` event ``turn`` field to an int turn index.

    The wire value is an INT in production (the dispatchers stamp the conversation turn
    the child reported, 0 when none was); a string (``"0"``) is also accepted
    defensively. A missing or unparseable value falls back to turn 0 — callers that must
    distinguish "no turn" from turn 0 check for the key themselves before coercing.
    ``bool`` is excluded explicitly (it is an ``int`` subclass, but ``True``/``False``
    are never valid turn indices).
    """
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return 0
    return 0


def _tool_event_to_recovered(event: dict[str, Any]) -> RecoveredToolCall:
    """Shape ONE mock ``tool_events`` event into a :class:`RecoveredToolCall`.

    The span renderers (Langfuse ``_emit_agent_span_with_tools`` / the LangSmith
    turn builder) read ``tool`` / ``args`` / ``response`` / ``status``; the rest of
    the schema is filled from the event. Defensive on every field — a partial event
    still yields a schema-valid record rather than raising mid-emit — and the mutable
    dict fields are shallow-copied so the produced record never aliases the
    ``tool_events`` payload.

    ``_recovered`` / ``_lane`` are CARRIED, not asserted: a ``/mock/execute`` event has
    neither (so the record reads ``_recovered=False``), while a trail record
    merged into ``events`` keeps saying it is one — the flags are how a reader tells an
    authoritative mutation from a self-reported dispatch.
    """
    args = event.get('args')
    state_delta = event.get('state_delta')
    new_entities = event.get('new_entities')
    status = event.get('status')
    record = RecoveredToolCall(
        type='tool_call',
        tool=str(event.get('tool') or 'unknown_tool'),
        args=dict(args) if isinstance(args, dict) else {},
        id=str(event.get('id') or ''),
        state_delta=dict(state_delta) if isinstance(state_delta, dict) else {},
        new_entities=dict(new_entities) if isinstance(new_entities, dict) else {},
        _recovered=event.get('_recovered') is True,
        status=status if isinstance(status, str) else 'ok',
        response=event.get('response'),
    )
    # Carried through rather than re-derived: the turn is the caller's bucket key, and a
    # record that loses it cannot be re-attributed downstream. Absent stays absent.
    turn = event.get('turn')
    if turn is not None:
        record['turn'] = _coerce_turn(turn)
    if event.get('_lane') == TRAIL_LANE:
        # The tag travels WITH its invariants, because this is where a record re-enters the
        # typed world from an untrusted dict: a trail record is never a /mock/execute mutation,
        # never carries an authoritative delta, and must stay turn-stamped — a turn-less one
        # would graft onto turn 0 and mis-attribute a late dispatch to the opening turn.
        record['_lane'] = TRAIL_LANE
        record['_recovered'] = True
        record['state_delta'] = {}
        if 'turn' not in record:
            record['turn'] = 0
    return record


def tool_span_metadata(call: RecoveredToolCall) -> dict[str, bool]:
    """Metadata a sink stamps on a tool span: ``{'mocked': True}`` for a mock-lane record, else ``{}``.

    A record is provably mocked only when it is a ``/mock/execute`` mutation record:
    ``_recovered`` is ``False`` (:func:`_tool_event_to_recovered` reads a bare mock event as
    such) and no ``_lane`` names an out-of-band lane. Message-lane, trail-lane and action-lane
    records carry ``_recovered=True``. A record that does not say stays unmarked, because a
    ``mocked: False`` would assert a live call the record cannot vouch for.
    """
    if call.get('_recovered') is False and '_lane' not in call:
        return {'mocked': True}
    return {}


def attribute_tool_events_to_turns(turns: list[TranscriptTurn], tool_events: dict[str, Any] | None) -> None:
    """Fill empty per-turn ``agent_tool_calls`` from the merged ``tool_events`` log.

    For the Bash-bearing ``code`` / ``cli_http`` action channels the agent's tool
    calls run OUT-OF-BAND (a shadowed import / ``fetch`` in a nested session routes
    each call to ``/mock/execute``), and the simulation deliberately collapses that
    nested session into ONE consolidated ``AIMessage`` per turn. So
    :func:`derive_tool_calls_from_messages` recovers nothing, ``agent_tool_calls``
    is ``[]``, and Langfuse/LangSmith render the per-turn ``agent`` span with NO
    ``tool`` children even though the run clearly called tools. The calls DO survive
    on the mock wire as ``tool_events``; this grafts them onto the matching turn so
    the span renderers materialize one ``tool`` child per event. A self-looping
    agent's calls reach ``events`` the same way, via the trail lane
    (:func:`trail_tool_calls`), and graft here for the same reason.

    Mutates ``turns`` in place. ONLY fills turns whose message-recovery produced NO
    calls: the semantic-``tools`` channel already populates ``agent_tool_calls`` from
    real message ``tool_calls``, so skipping non-empty
    turns is the simplest rule that never double-renders those calls. ``tool_events``
    is the cumulative session log available only AFTER finalize — call this
    post-finalize, before ``record_transcript`` drives the sinks.

    **Turn attribution:** every ``/mock/execute`` POST stamps the conversation turn the
    child reported (``plurai._mocking.turn_context``), so events bucket onto the turn
    that issued them. Two degraded cases still land on turn 0: a harness that drives the
    mocks without ``plurai.child`` (nothing ever reports a turn), and an artifact whose
    events carry no turn stamp at all. Guards the non-dict /
    ``_finalize_error``-marker ``tool_events`` case (no ``events`` list ⇒ no-op).
    """
    if not turns or not isinstance(tool_events, dict):
        return
    events = tool_events.get('events')
    if not isinstance(events, list):
        return
    by_turn: dict[int, list[RecoveredToolCall]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get('type') != 'tool_call':
            continue
        # ``merge_recovered_tool_calls`` runs BEFORE this enrichment and unions BOTH
        # out-of-band lanes into ``events``. Skip only the MESSAGE lane: those records already
        # render on their own turn via ``_record_turn``'s message recovery, so grafting them
        # here too would DUPLICATE them. The trail lane has no such other renderer — it is
        # recovered from ``agent_state``, which ``_record_turn`` never reads — so a self-looping
        # agent's dispatches would otherwise stay invisible as turn-span children, which is the
        # blind spot this module exists to close. Real ``/mock/execute`` code/cli_http events
        # carry neither flag, so they always take the enrichment path.
        if event.get('_recovered') is True and event.get('_lane') != TRAIL_LANE:
            continue
        by_turn.setdefault(_coerce_turn(event.get('turn')), []).append(_tool_event_to_recovered(event))
    if not by_turn:
        return
    consumed: set[int] = set()
    for turn in turns:
        # Never-raise contract: ``Orchestrator._emit_observability`` calls this un-try-wrapped, so a malformed
        # turn must be skipped, not raise. Guard the non-dict case and read ``turn_idx`` via
        # ``.get`` (a turn missing it falls through to ``by_turn.get(None)`` ⇒ no match).
        if not isinstance(turn, dict):
            continue
        if turn.get('agent_tool_calls'):
            continue  # message-recovery already populated this turn — don't double-render
        idx = turn.get('turn_idx')
        bucket = by_turn.get(idx)
        if bucket:
            turn['agent_tool_calls'] = bucket
            consumed.add(idx)
    # This module exists to eliminate tool-call blind spots, so a bucket that found no empty
    # matching turn (no such turn_idx, or the matching turn was already populated) is a silent
    # drop worth a debug breadcrumb — kept debug-level so the happy path stays quiet.
    dropped = {idx: len(bucket) for idx, bucket in by_turn.items() if idx not in consumed}
    if dropped:
        logger.debug(
            'attribute_tool_events_to_turns: %d tool_event(s) had no empty matching turn to graft onto; dropped per turn_idx: %s',
            sum(dropped.values()),
            dropped,
        )
