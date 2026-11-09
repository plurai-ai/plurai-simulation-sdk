"""REST-driven simulation orchestration for plurai.

The orchestrator is a thin HTTP client that drives a turn-by-turn dialog
against the Plurai backend. The agent runs in a subprocess (spawned from the
customer's ``simulate.py``); the orchestrator handles thread creation, the
user-simulation hand-offs (over HTTP), and the final policy evaluation.
Tool-call plumbing inside the agent already flows over HTTP via
:class:`plurai.MockToolNode` to the configured mock-env, so the orchestrator
never touches it directly.

Typical usage::

    from pathlib import Path

    from plurai import Orchestrator
    from plurai.observability import create_observability_sink
    from plurai.orchestrator.client import build_default_client
    from plurai.settings import Settings

    settings = Settings()
    async with Orchestrator(
        simulate_path=Path("path/to/simulate.py"),
        settings=settings,
        observability=create_observability_sink(settings),
        client=build_default_client(settings),
    ) as o:
        result = await o.run(scenario_set_id="set-uuid", scenario_id="scn_01", redrive_available=False, allow_turn_timeout_redrive=False, on_child_closed=None)
        print(result.summary())
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Final, Literal, cast

import httpx

from plurai._endpoints import TOKEN_ENV, URL_ENV, Endpoint, NotConfiguredError
from plurai.content_parts import SILENT_PART_KINDS, content_text, declared_part, document_part_values, opener_content, with_part_values
from plurai.host_effects import redact_value, register_backend_token, trail_recorded
from plurai.module_bindings import _module_scope_bindings
from plurai.run_artifacts import _NON_AGENT_TYPES, REPLAY_CUT_REASON, SEED_RESTORE_KEY, reply_source_counts
from plurai.observability.sink_protocol import NoOpSink, ObservabilitySink, RunContext, TranscriptTurn
from plurai.observability.eval_mapping import filter_scenario_metadata
from plurai.observability.model_calls import ModelCall, derive_model_calls
from plurai.observability.tool_call_recovery import RecoveredToolCall, ThreadCalls, derive_turn_tool_calls
from plurai.observability.dataset_upload import upload_scenarios_from_fetch
from plurai.observability.tool_call_recovery import (
    AGENT_STATE_TOOL_TRAIL_KEY,
    attribute_tool_events_to_turns,
    merge_pipeline_trail,
    merge_recovered_tool_calls,
    strip_seeded_trail,
)
from pydantic import ValidationError

from plurai._protocol import (
    AGENT_RAISED_KIND,
    AGENT_RECURSION_BUDGET_KIND,
    INTERRUPT_ACT,
    PROVIDER_INFRA_KIND,
    TURN_BUDGET_EXHAUSTED_KIND,
    USER_TURN_ENVELOPE,
    _json_shaped,
    AgentRaised,
    AgentTurnTimeout,
    GraphRecursionWrap,
    ProviderInfra,
    FinalizeJobFailed,
    FinalizePollTimeout,
    HarnessChildExited,
    HarnessFaultRaised,
    MockMode,
    OrchestratorAdapterError,
    observed_actions_or_none,
)
from plurai._subprocess import ChildProcess
from plurai.child_env import launch_environment
from plurai.agent_io import AGENT_IO_KEY, DEFAULT_AGENT_IO, INPUT_FIELD, INPUT_PART_KINDS, OPENS_ALWAYS, OPENS_FIELD, AgentIO, declared_agent_io
from plurai.channels import CHANNEL_INPUTS_KEY, parse_channel_plan
from plurai.gates.apply import make_finding
from plurai.gates.catalogue import HARNESS_AGENT_IO_UNKNOWN_FIELD, HARNESS_AGENT_IO_UNKNOWN_PART_KIND, WORLD_UNRESET_EXTERNAL
from plurai.gates.contract import GateFinding
from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, bench_config_roots, discover_config_documents
from plurai.orchestrator._retry import CreditRefused, is_transient_for_idempotent, job_refused_for_credit
from plurai.orchestrator.client import (
    OrchestratorClient,
    UserTurn,
    submit_carrying_overrules,
    take_gate_authority,
)
from plurai.orchestrator.models import CreateThreadResponse, FinalizeResponse
from plurai.orchestrator.redrive import static_trigger, wait_until_ready
from plurai.orchestrator.state_wire import AGENT_STATE_LEAF_CAP_BYTES, map_string_leaves, wire_agent_state
from plurai.orchestrator.world_images import StorageClientFactory, storage_client_for, upload_world_images
from plurai.world.readback import SEED_NOT_LANDED_REASON, seed_readback_at_init
from plurai.settings import FULL_RUN_SCENARIO_TIME_BUDGET, Settings

if TYPE_CHECKING:
    from plurai.ingest.storage_client import StorageClient


# Marker key the child packs into ``final_world_state`` / ``tool_events`` when
# ``plurai.get_full_state`` / ``plurai.get_events`` raise during finalize (see
# ``plurai.child._ChildSession.handle_finalize``). The world-state never
# materialized, so the snapshot is garbage — a scenario carrying this marker
# MUST NOT be scored as a clean run.
_FINALIZE_ERROR_MARKER = '_finalize_error'

#: The wall on the ONE mock-log read taken when the turn loop raises with the child still alive
#: (:meth:`Orchestrator._read_mock_log_on_raise`). The read is evidence for the crash report, never
#: a reason to hold the failure up, so it is bounded well below a turn's wall.
_CRASH_MOCK_LOG_READ_TIMEOUT = 60.0


def _finalize_error_message(final_world_state: dict[str, Any] | None, tool_events: dict[str, Any] | None) -> str | None:
    """Return the child's finalize-error detail when the marker is present, else None.

    The child returns a successful ``FinalizeResponse`` even when state capture
    failed, packing ``{'_finalize_error': '<exc>'}`` into whichever snapshot
    raised. Detect it here so the orchestrator can route the scenario to the
    failure path instead of feeding a never-materialized world-state into the
    server-side policy eval + rollup as if it were a clean run.
    """
    details: list[str] = []
    if isinstance(final_world_state, dict) and _FINALIZE_ERROR_MARKER in final_world_state:
        details.append(f'final_world_state: {final_world_state[_FINALIZE_ERROR_MARKER]}')
    if isinstance(tool_events, dict) and _FINALIZE_ERROR_MARKER in tool_events:
        details.append(f'tool_events: {tool_events[_FINALIZE_ERROR_MARKER]}')
    if not details:
        return None
    return '; '.join(details)


def _fresh_tool_calls(turn_calls: list[RecoveredToolCall], seen_tool_call_ids: set[str]) -> list[Any]:
    """Reduce a turn's recovered calls to the ones it NEWLY issued (mutates the seen-set).

    ``turn_calls`` is the turn's ONE recovery over the thread's call ledger
    (:func:`derive_turn_tool_calls`) — the records ``tool_events`` also carries, so a result
    answering an earlier turn's call is absent here too rather than re-read as a call of its own.
    A MemorySaver-style agent re-lists its CUMULATIVE history every turn, so a naive
    per-turn derive would re-attribute earlier turns' calls; deduping by call id pins
    each call to the turn that ISSUED it. A call with no id is never deduped (kept, so
    an orphan ToolMessage is not silently dropped). Shared by ``_record_turn`` and the
    closing-exchange record so the closing reply's tool calls follow the same rule.
    """
    fresh: list[Any] = []
    for tc in turn_calls:
        if tc.get('_unpaired'):
            # A result nothing could attach (args UNKNOWN): the union floors it; the per-turn
            # record must not render it as a call with empty args either.
            continue
        call_id = tc['id']
        if call_id and call_id in seen_tool_call_ids:
            continue  # already rendered under the turn that issued it
        if call_id:
            seen_tool_call_ids.add(call_id)
        fresh.append(tc)
    return fresh


#: Role spellings that mean "this message is the USER's" once normalised (the LangChain
#: discriminator plus the raw spellings ``child._serialize_message`` maps onto it).
_USER_SIDE_ROLES = frozenset({'human', 'user'})


def _declared_role(msg: Any) -> str | None:
    """A serialized message's declared role, or ``None`` when it names none.

    ``None`` is a real answer and not a default: a message that declares no role has been
    ATTRIBUTED to nobody, and the whole defect below is what happens when something guesses.
    """
    if isinstance(msg, dict):
        raw = msg.get('type') or msg.get('role')
    else:
        raw = getattr(msg, 'type', None) or getattr(msg, 'role', None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().lower().removesuffix('message')


def _message_text(msg: Any) -> str:
    """A serialized message's content as plain text — ``str`` or a LIST of content parts.

    A turn's content may be a list of standard content parts. A list that contributed NOTHING
    here (``''``) would make the echo rule below unable to fire on a parts turn, so an adapter
    echoing the user's own parts message would have it attributed to the AGENT — the exact
    misattribution :func:`split_user_echo` exists to prevent. Both sides of that comparison go
    through the same projection (:func:`plurai.content_parts.content_text`), which is the
    identity on a ``str`` and therefore leaves every text-only harness untouched.
    """
    content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
    return content_text(content)


def split_user_echo(
    chain: list[dict[str, Any]], *, input_user_message: str | list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a turn's chain by the single-role contract, saying WHICH rule moved each message.

    Returns ``(agent side, role-declared user messages, role-less verbatim echoes)`` — three
    lists, not two, because the two rules describe adapters in opposite health. Rule 1 fires on
    the shape this docstring calls NORMAL (a LangGraph agent's first reply is ``[HumanMessage,
    AIMessage]``), so a single lumped report would tell every ordinary LangGraph user, once per
    scenario, to go fix ``simulate.py``. Rule 2 is
    the actual contract breach: a message wearing NO role, carrying the user's words, which is
    the one that would be stamped ``'ai'`` on the wire.

    A customer adapter may legitimately return BOTH roles for a turn: a LangGraph agent's first
    reply is ``[HumanMessage, AIMessage]``, and that is the normal shape, not a defect. The
    defect would be carrying the human half into the AGENT channel: the flat
    ``chatbot_messages`` in ``transcript.json`` and the ``turn_messages`` narration channel,
    which the wire contract itself calls "THIS turn's full AGENT-SIDE chain". An adapter
    returning ``{user, assistant}`` per turn would otherwise put every user turn verbatim at the
    even indices of the agent channel.

    Why it matters even though the backend filters by role: the SDK supplies a role where none
    exists. A message declaring no recognisable role is stamped ``'ai'`` on the wire
    (``_wire_turn_messages``' ``or 'ai'`` default) and is treated as agent output by the
    backend's deliberately EXCLUSION-based selection of agent-side messages — so a user-pasted URL or
    quote would be attributed to the agent, and every check asking "did the AGENT say X" is
    rigged before the run starts.

    TWO rules, both FIRST-HAND — the same membership rule the ``ended_early`` vocabulary keeps,
    and no similarity heuristics:

    * the message DECLARES a user/human role. Nothing is inferred; it said so.
    * the message declares NO role at all and its text is EXACTLY the user message this harness
      just sent. The harness knows what it sent — an identical message coming back inside the
      agent's chain is the adapter echoing it, not the agent speaking.

    Everything else stays on the agent side, including a role-less message with its own text: an
    adapter that names no roles at all is still reporting its agent's replies, and dropping those
    would silence the run to protect it.

    ``turns`` stays the authoritative pairing either way — this only decides what the AGENT
    channel is allowed to contain.
    """
    # The sent message may be a LIST of content parts. ``.strip()`` on it would raise
    # ``AttributeError: 'list' object has no attribute 'strip'`` — not a degraded split, a CRASH
    # on the turn. Projected through the same function the message side uses, so the two are
    # compared on one reading of "the text".
    #
    # ``None`` is NOTHING SENT — an agent-first turn — and the echo rule then has no subject, so
    # it must SKIP rather than match. ``content_text(None)`` is ``''`` and the loop's rule-2 arm
    # below is guarded ``and echoed``, so an empty subject disables it; a ternary here would be a
    # no-op that the next reader takes for the guard. The annotation admits ``None`` for that
    # reason.
    #
    # The arm must stay guarded. Comparing against a bare ``''`` would make every role-less empty
    # message in a chain an "echo" of a turn that never happened, moving real agent output into
    # the user channel.
    echoed = content_text(input_user_message).strip()
    agent_side: list[dict[str, Any]] = []
    declared: list[dict[str, Any]] = []
    verbatim: list[dict[str, Any]] = []
    for msg in chain or []:
        role = _declared_role(msg)
        if role in _USER_SIDE_ROLES:
            declared.append(msg)
        elif role is None and echoed and _message_text(msg).strip() == echoed:
            verbatim.append(msg)
        else:
            agent_side.append(msg)
    return agent_side, declared, verbatim


#: Per-message content cap for the narration channel's REQUEST bodies, for NON-AGENT content. The
#: full chain still lands uncapped in transcript.json; the cap exists because a TOOL RESULT (or a
#: human/system message, or a payload part) beyond this adds payload, not signal — the server clips
#: its own rendering of a tool result far below this anyway.
#:
#: AGENT TEXT IS NEVER CUT. The words of an ``ai`` message on this channel are graded: the backend binds
#: each turn's chain into the deterministic-check sandbox as ``agent_messages`` and renders it to
#: every LLM judge, so a cut there is a different answer graded — a cut inside a citation URL fails a
#: URL check as a fabricated citation. The backend can
#: restore only the turn's LAST agent text (its carrier, ``agent_response``); an earlier message of
#: the same chain — a long report before a tool round-trip, a prior reply the agent re-lists — has
#: no other copy on the wire. Which messages are agent-authored is the one rule this SDK already
#: has (:data:`~plurai.run_artifacts._NON_AGENT_TYPES`, also what ``child._read_suffix`` reads), so
#: the wire and the reply extractor cannot disagree about what is speech. The payload parts of an
#: agent message (base64 ``file`` / ``image_url``, structure, provider-internal blocks) are still
#: budgeted: only its ``text`` rides whole.
#:
#: "Over the cap" is not one action: what an over-budget part COSTS depends on its declared
#: kind, and the per-kind table is :func:`_budgeted_parts`. A ``text`` part keeps its head, as the
#: ``str`` arm below does for a non-agent one: a sentence has a head worth keeping, a base64
#: payload does not.
#:
#: And it is a per-MESSAGE CONTENT cap: it does not bound a BODY, because not every carrier is
#: charged. ``agent_response`` — the string the backend appends to the thread's agent messages — rides
#: whole, and so does every agent message's text on ``turn_messages``. Which carrier is which is
#: NOT decided by a table production consults — production consults one function
#: (:func:`_budgeted_content`) at the call sites that charge.
_WIRE_MESSAGE_CONTENT_CAP = 8_000

#: The ONE spelling of "a producer cut this for the wire", shared by the ``str`` arm of
#: :func:`_wire_turn_messages` and the ``text`` arm of :func:`_budgeted_parts`. The backend knows
#: this literal BY NAME (and re-states it as "(producer-truncated)" when its own tool-result clip
#: would cut the marker off), so a second wording would be a vocabulary the consumer cannot read — and two copies of one
#: literal are two things to keep in step.
_WIRE_TRUNCATION_SUFFIX = ' [TRUNCATED for wire]'


def _budgeted_parts(parts: list[Any], *, cut_words: bool = True) -> list[Any]:
    """A LIST content capped to the same per-message budget a ``str`` content gets.

    Applied to the message's TOTAL rendered size, not per part, so the number a reader has to
    hold is the one that already exists (:data:`_WIRE_MESSAGE_CONTENT_CAP`) rather than a second
    one that interacts with it. Parts are kept whole and in order while the budget lasts; what
    happens to the first part that does not fit is decided by its DECLARED kind, off the
    vocabulary ``content_parts`` already publishes — not by one rule for every kind:

    * ``text`` — rewritten to the projection the SDK's and the backend's renderers read (``type`` and ``text``,
      and nothing else: ``content_parts._render_text`` here and the backend's renderer both read
      exactly those), and TRUNCATED
      to what is left of the budget with :data:`_WIRE_TRUNCATION_SUFFIX` only
      if the words themselves are what does not fit — exactly the way the ``str`` arm truncates a
      long string. The head of a sentence is the part every judge reads; replacing it with a
      marker is the agent's answer deleted and a harness annotation put in the agent's own voice
      in its place. What made the part over-budget is often NOT the words: an Anthropic ``text``
      block carries ``citations`` / ``cache_control`` beside them, and a 9 kB sibling copied
      through while the budget was charged as if the part had been shrunk is a wire body past the
      cap the cap exists to hold. So the siblings go and the charge is the REWRITTEN part's real
      size. When the projection fits, the words are whole and carry NO suffix — the suffix means
      "this text was cut", and the backend reads that literal by name. With ``cut_words=False`` (an
      AGENT message — see :data:`_WIRE_MESSAGE_CONTENT_CAP`) the words are never cut and never
      charged: the part is still rewritten to the projection (siblings stripped), and kept whole
      even after another part has spent the budget.
    * a payload/structure kind (``file``, ``image_url``, ``json``, or an unknown one) — DROPPED
      and NAMED, because a truncated base64 blob is not a smaller image, it is a corrupt one,
      and half a serialized structure is not that structure any more. A part that vanishes
      with no trace is an agent answer that vanished, so the drop leaves a marker.
    * a provider-internal block (:data:`SILENT_PART_KINDS` — ``thinking``, ``tool_use``, …) —
      dropped SILENTLY. The SDK and the backend both render these to NOTHING, so a marker in their place is a
      visible annotation standing where the reader saw nothing before: the same defect the
      silent set exists to prevent, reached through the budget instead of through the renderer.

    The walk does not stop at the first over-budget part. A 9 kB attachment or thinking block
    standing FIRST would otherwise take every part after it — including the one sentence that
    was the agent's actual answer. The drop markers are coalesced into ONE per message, at the
    position of the first drop, naming the count and the kinds: a marker per dropped part would
    spend the very budget it is announcing.

    Untouched when the whole message already fits, which is every text turn and every small
    parts turn — the common case pays one ``len`` and keeps its identity.
    """
    total = 0
    kept: list[Any] = []
    dropped: list[str] = []
    drop_at = 0
    for part in parts:
        size = len(part if isinstance(part, str) else json.dumps(part, ensure_ascii=False, default=str))
        remaining = _WIRE_MESSAGE_CONTENT_CAP - total
        if size <= remaining:
            total += size
            kept.append(part)
            continue
        # The kind is read off the block's DECLARED vocabulary, via the one function that says
        # what a content-list block is (``content_parts.declared_part``): any ``Mapping``, and a
        # bare STRING, which the wire admits beside parts and which MEANS text, so the same words
        # are text on both halves of a turn. A block with no declared kind at all gets the empty
        # string and lands in the default arm below, NAMED by its Python class.
        block: Mapping[str, Any] = declared_part(part) or {}
        kind = str(block.get('type') or '')
        if kind == 'text' and (remaining > 0 or not cut_words):
            # ``remaining > 0`` is load-bearing: once one text part has spent the budget the next
            # one would otherwise append an empty ``_WIRE_TRUNCATION_SUFFIX`` run, one per part.
            # Agent words (``cut_words=False``) are never cut, so that run cannot occur for them.
            #
            # The siblings are dropped with NO marker, for the :data:`SILENT_PART_KINDS` reason:
            # neither renderer reads them, so a marker would stand where the reader saw nothing.
            words = str(block.get('text') or '')
            projected: dict[str, Any] = {'type': 'text', 'text': words}
            if not cut_words:
                # The AGENT's words are graded: kept whole, and not charged against the budget the
                # message's payload parts share.
                kept.append(projected)
                continue
            size = len(json.dumps(projected, ensure_ascii=False, default=str))
            if size > remaining:
                # Now it IS the words that do not fit, so they are cut and the suffix says so.
                projected['text'] = words[:remaining] + _WIRE_TRUNCATION_SUFFIX
                size = len(json.dumps(projected, ensure_ascii=False, default=str))
            # The REWRITTEN part's real size, never an assumed one: charging the whole cap here
            # would let a dropped sibling be paid for twice — once by a budget that thought it
            # had been spent, and once by the wire.
            total += size
            kept.append(projected)
            continue
        if kind in SILENT_PART_KINDS:
            continue
        if not dropped:
            drop_at = len(kept)
        dropped.append(kind or type(part).__name__)
    if dropped:
        named = ', '.join(repr(kind) for kind in dict.fromkeys(dropped))
        marker = f'[{len(dropped)} content part(s) dropped ({named}): over the {_WIRE_MESSAGE_CONTENT_CAP}-char wire budget]'
        kept.insert(drop_at, {'type': 'text', 'text': marker})
    return kept


def _budgeted_content(content: Any, *, cut_words: bool = True) -> Any:
    """A message's CONTENT charged against the wire budget, in either of its two legal spellings.

    The budget is a property of the wire BODY, not of a Python type: the same words spelled as a
    ``str`` or as a ``text`` part get one budget. Both spellings are routine on ``/finalize``'s
    ``user_messages`` — a scenario's opener arrives as ``initial_message``
    (a ``str``) or as ``initial_messages`` (a content-part list), and which one a harness uses is
    not a statement about how much of the user's words the judges should see.

    So the arms live here, once, and the call sites name the CARRIER instead of the shape:

    * ``str`` over the cap — cut to the cap with :data:`_WIRE_TRUNCATION_SUFFIX`, which is the cut
      the narration channel makes for non-agent content. With ``cut_words=False`` (an
      agent message) a ``str`` is returned whole: all of it is words.
    * ``list`` — :func:`_budgeted_parts`, which decides per DECLARED part kind. A bare string
      INSIDE a list is a block and stays that function's business: ``declared_part`` already reads
      one as ``text`` there. This function's ``str`` arm is the MESSAGE level, a different thing.
    * anything else — returned untouched. It has no declared content shape to charge, and the one
      caller that can be handed one (:func:`_wire_turn_messages`) logs it rather than guessing.

    Not every carrier is charged, and that is the other half of the rule — see
    :data:`_WIRE_MESSAGE_CONTENT_CAP`: the narration copy is budgeted, the primary record is whole.
    """
    if isinstance(content, str) and cut_words and len(content) > _WIRE_MESSAGE_CONTENT_CAP:
        return content[:_WIRE_MESSAGE_CONTENT_CAP] + _WIRE_TRUNCATION_SUFFIX
    if isinstance(content, list):
        return _budgeted_parts(content, cut_words=cut_words)
    return content


#: One warning per ``(scenario, declared role)`` whose calls this projection dropped. A
#: per-MESSAGE condition, so an unthrottled log would emit a line per message per turn for a
#: whole run — the same throttle, for the same reason, as :data:`_warned_uncanonical`.
#:
#: Keyed on the SCENARIO too, not the role alone: a suite whose harness emits one unreadable
#: spelling would otherwise report it once, for the first row that hit it, and every later row
#: would lose its calls in silence — the operator reading that row's judge verdicts would have
#: nothing saying the narration channel was empty there. Cleared per run by
#: :func:`reset_warning_ledgers`.
_warned_dropped_calls: set[tuple[str, str]] = set()


def reset_warning_ledgers() -> None:
    """Empty the accumulating THROTTLE ledgers, at run start, before a turn can add to them.

    These are module-level sets whose docstrings promise "one line ... for the whole run", so
    they are cleared per run: a second run in the same interpreter — the shape ``--scenario-id``
    retries and in-process test suites take — would otherwise inherit the first's contents and
    print nothing about a condition it hits on every turn.
    """
    _warned_dropped_calls.clear()
    _warned_uncanonical.clear()
    _warned_omitted_state.clear()
    # These two are worse than a stale warning: the first caches a DISPATCH ROUTE keyed on path
    # with no content invalidation, so a harness that grows ``invoke_pipeline`` between passes of
    # an in-process loop keeps being driven as a dialog; the second permanently exempts a path
    # that was briefly unreadable from the only warning that names the route chosen.
    _INVOKE_HOOK_CACHE.clear()
    _UNREADABLE_HARNESSES.clear()


def _note_dropped_calls(raw: dict[str, Any], *, thread_id: str) -> None:
    """WARN when the canonicaliser refused a message whose tool calls this projection will drop.

    ``_canonical_message(raw) or raw`` is a harmless fallback for a message carrying only prose,
    and a SILENT LOSS for one whose calls are spelled the way the ``name``-only filter below
    cannot read: the turn then ships with ``tool_calls: []`` and reads to every judge as a turn
    in which no tool ran. Without a warning that loss leaves no counter and nothing in the
    artifacts behind, so the fallback says which side of it fired.
    """
    calls = raw.get('tool_calls')
    if not isinstance(calls, list) or not any(isinstance(call, dict) and not call.get('name') for call in calls):
        return
    role = str(raw.get('type') or raw.get('role') or '?')
    if (key := (thread_id, role)) in _warned_dropped_calls:
        return
    _warned_dropped_calls.add(key)
    logging.getLogger(__name__).warning(
        'turn_messages: on %s, a message of role %r carrying tool calls could not be read by the canonical serializer, '
        'and its calls are spelled without a top-level "name" — they are DROPPED from the judges\' narration channel, '
        'which renders this turn as one in which no tool ran. One line per message role per scenario.',
        thread_id or '<unnamed thread>',
        role,
    )


def _wire_turn_messages(chain: list[dict[str, Any]], *, thread_id: str = '') -> list[dict[str, Any]]:
    """A turn's chain projected to the wire shape for the narration channel.

    Keeps exactly the ``SerializedMessage`` fields — type/content/tool_calls/
    tool_call_id/name/status — so provider bookkeeping (response_metadata, signatures,
    usage) never rides the request body, and caps oversized NON-AGENT contents with an
    explicit marker — an agent message's text rides whole (:data:`_WIRE_MESSAGE_CONTENT_CAP`).
    tool_calls entries keep name/args/id only.

    Each message goes through :func:`_canonical_message` FIRST. That is the whole reason a
    harness returning OpenAI role dicts keeps its calls here: the tool-call filter
    below keeps an entry only when it carries a top-level ``name``, and the
    ``{'id', 'type': 'function', 'function': {'name', 'arguments'}}`` spelling that harnesses
    emit has none, so every call-bearing turn would reach the judges with
    ``tool_calls: []``. The filter does not special-case that spelling (a per-provider table
    would follow): ``convert_to_messages`` is the ecosystem's own reader of the wire shapes and
    already knows it, this module already runs the seed compare through it, and every spelling
    IT learns, this learns. A message the canonicaliser cannot read is projected as it arrived —
    and, when that fallback is about to cost the message its calls,
    :func:`_note_dropped_calls` says so rather than dropping them silently.
    """
    out: list[dict[str, Any]] = []
    for raw in chain or []:
        if not isinstance(raw, dict):
            continue
        canonical = _canonical_message(raw)
        if canonical is None:
            _note_dropped_calls(raw, thread_id=thread_id)
        msg = canonical or raw
        content = msg.get('content')
        # ``role`` is the other spelling a serialized message carries, and it has to be read
        # here: stamping such a message 'ai' would render the SIMULATED USER's words as agent
        # narration in the judge view — the mirror of the defect this channel closes.
        #
        # The ``or 'ai'`` tail is the LAST resort and is only defensible because the caller has
        # already run ``split_user_echo``: a message with no declared role that survived that
        # split is the agent's by the harness's own first-hand evidence, not by this default's
        # guess. As the guess, it would stamp 'ai' on an adapter echoing the user's words
        # under an unrecognised role key.
        wire_type = str(msg.get('type') or msg.get('role') or 'ai')
        if isinstance(content, (str, list)):
            # A LIST content is budgeted the way a ``str`` is. The cap is per-MESSAGE; were it
            # string-only, the budget would stop applying the moment a turn carried parts — and
            # the part that makes that expensive is exactly the one a list exists to carry: a
            # base64 ``file`` or ``image_url`` payload would ride the REST body, and
            # transcript.json, at full size. Capping the message as a whole rather than each part
            # keeps ONE number to reason about.
            #
            # Both spellings go through ONE function, so this channel and the ``/finalize``
            # user carrier cannot answer differently about the same words.
            #
            # Agent text is NOT budgeted (see :data:`_WIRE_MESSAGE_CONTENT_CAP`): "is this the
            # agent's" is the same type rule ``child._read_suffix`` reads, off the same type this
            # message is stamped with below.
            content = _budgeted_content(content, cut_words=wire_type in _NON_AGENT_TYPES)
        elif content is not None:
            # The wire accepts str | list; anything else (a customer adapter returning a
            # structured reply — child.py passes such a dict through verbatim) would raise
            # ValidationError inside client.user_turn and kill the whole scenario for a
            # channel that only improves judge context. Same posture as
            # _extract_agent_state: coerce, warn, never fail the turn.
            logging.getLogger(__name__).warning('turn_messages: non-str/list content (%s) stringified for the wire.', type(content).__name__)
            content = str(content)
        wire: dict[str, Any] = {'type': wire_type, 'content': content if content is not None else ''}
        tool_calls = msg.get('tool_calls')
        if isinstance(tool_calls, list) and tool_calls:
            wire['tool_calls'] = [
                {'name': tc.get('name'), 'args': tc.get('args') or {}, 'id': tc.get('id')} for tc in tool_calls if isinstance(tc, dict) and tc.get('name')
            ]
        for key in ('tool_call_id', 'name', 'status'):
            if msg.get(key):
                wire[key] = msg[key]
        out.append(wire)
    return out


def _reply_source_of(response: Any) -> str:
    """An agent response's ``reply_source``, or ``''`` when the producer did not classify.

    Read defensively because the child is a PLUGGABLE protocol (``child_factory``): a duck-typed
    child hands back an envelope without the field, and a missing label must degrade to
    "this producer did not say" rather than crash the turn — or, worse, be filled in as
    ``assistant``, which is the one reading the label exists to prevent.
    """
    label = getattr(response, 'reply_source', '')
    return label if isinstance(label, str) else ''


def _agent_actions_of(response: Any) -> list[dict[str, Any]] | None:
    """What the agent DID this turn, or ``None`` when this producer cannot say.

    Read defensively for the same reason ``_reply_source_of`` is — the child is a PLUGGABLE
    protocol — but the degradation matters more here, because this field is the LEGITIMACY
    signal: a turn counts as delivered when either the speech or the actions are non-empty. A
    duck-typed child that has no such field must therefore send ``None`` ("this
    client cannot observe its agent's actions") and never ``[]`` ("it can, and there were
    none"), or the server would read an unobservable turn as a turn in which nothing happened
    — the exact collapse ``turn_tool_events`` keeps tri-state for the same reason.

    And the SAME collapse at the other end, which this function must not perform itself: the
    ``name`` filter would turn a producer that observed N unnameable actions into ``[]`` — "it can
    observe, and there were none" — which is the reading the tri-state exists to prevent, from
    the frame that exists to protect it. The SDK's own ``child.turn_actions`` always names its calls,
    but ``child_factory`` is a supported extension point, and the narration channel one screen up
    warns (:func:`_note_dropped_calls`) for exactly this drop. A turn whose actions were
    ALL dropped reports ``None``; a partial drop keeps what was read and says what it lost.
    """
    actions = getattr(response, 'agent_actions', None)
    if not isinstance(actions, list):
        return None
    named = [action for action in actions if isinstance(action, dict) and action.get('name')]
    if len(named) < len(actions):
        logging.getLogger(__name__).warning(
            'agent_actions: %d of %d action(s) this turn carry no `name` and were dropped; %s',
            len(actions) - len(named),
            len(actions),
            'this turn reports that its actions could NOT be observed rather than that there were none.'
            if not named
            else 'the turn ships the rest, so its action list is narrower than what the producer saw.',
        )
    # The rule is READ here, not restated: an all-dropped list is "this producer could not be
    # read", never "nothing happened", and an originally EMPTY list stays ``[]`` — that one really
    # is "it can observe, and there were none". One reader keeps this and the child's envelope
    # gate from giving the same turn two different answers.
    return observed_actions_or_none(named, saw_declaration=bool(actions))


def _count_reply_sources(turns: list[TranscriptTurn], *, thread_id: str) -> dict[str, int]:
    """``{reply_source: turns}`` for this scenario, and a WARNING when a reply was not the agent's.

    A per-turn label is a cell nobody opens. The count is what a run summary, an audit or a human
    actually reads, and the warning is what makes a scenario whose "agent answer" was really a
    tool's raw payload visible on the run it happened in rather than three audits later.

    Turns the child did not label are counted under ``unlabelled`` rather than dropped or folded
    into ``assistant``: a producer that says nothing must not be readable as one that said
    "the agent spoke".

    The COUNT itself is :func:`~plurai.run_artifacts.reply_source_counts` — the projection whose
    own docstring says it is public "because the run-LEVEL question is asked by a different
    caller, and both must count it the same way". This is that caller.
    """
    counts = reply_source_counts(list(turns))
    if notable := {label: n for label, n in counts.items() if label not in ('assistant', 'unlabelled')}:
        logging.getLogger(__name__).warning(
            'thread %s: %s of this scenario\'s agent replies did not come from the agent\'s own words — %s. A "tool_result" '
            'reply is a turn that ended on a tool result with no agent words, whose result text was relayed as the reply; '
            'an "unknown" one is a reply shape this SDK could not read. Neither changes a verdict, and both change what a '
            'verdict is about.',
            thread_id,
            sum(notable.values()),
            notable,
        )
    return counts


def _agent_states_by_turn(transcript_turns: list[dict[str, Any]], *, seed_agent_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Each turn's ``agent_state``, at the index of the turn that produced it.

    ``trail_tool_calls`` reads a state's LIST POSITION as its turn number, and every entry here
    already carries the authoritative ``turn_idx`` — so read the key rather than trust append
    order: the two agree only while the loop appends exactly once per iteration and the closing
    turn takes the next index, and reading the key means a filtered or skipped turn shifts no
    trail stamp. Gaps are padded with an
    empty state so position never drifts, and a turn whose index is unusable is skipped rather
    than silently collapsed onto turn 0.
    """
    by_turn: dict[int, dict[str, Any]] = {}
    for entry in transcript_turns:
        if not isinstance(entry, dict):
            continue
        idx = entry.get('turn_idx')
        if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
            by_turn[idx] = entry.get('agent_state') or {}
    states = [by_turn.get(i, {}) for i in range(max(by_turn) + 1)] if by_turn else []
    # The seed's own trail, removed from every turn (``strip_seeded_trail``): ``seed_state``
    # restores the pin's ``agent_state`` verbatim, and a harness that keeps its trail in graph
    # state re-reports the prior session's dispatches as this run's. The rows hold the wire
    # projection (``_wire_state_of``), so the seed is compared in that same shape: an entry with
    # no ``call_id`` is matched by its shape, and a leaf over the cap is a descriptor in the rows.
    states, dropped = strip_seeded_trail(states, wire_agent_state(seed_agent_state, served={}).state)
    if dropped:
        logging.getLogger(__name__).warning(
            "%d tool_trail entries the resume seed already carried were re-reported by the harness; excluded from this run's tool activity.", dropped
        )
    return states


def _safe_wire_turn_messages(chain: list[dict[str, Any]], *, thread_id: str = '') -> list[dict[str, Any]] | None:
    """:func:`_wire_turn_messages`, degraded to ``None`` on any failure or empty chain.

    The narration channel only improves judge context, so nothing about it justifies
    losing a scenario that is already paid for: a projection failure ships the final-reply
    view with a warning. ``None`` rather than ``[]`` on empty too —
    the server reads an empty chain as "this client cannot observe tools", never as
    "zero tools ran", and that distinction selects opposite deterministic judge prompts.
    """
    try:
        return _wire_turn_messages(chain, thread_id=thread_id) or None
    except Exception:  # noqa: BLE001 — an enrichment must never fail a paid turn
        logging.getLogger(__name__).warning(
            "turn_messages: could not project this turn's chain for the wire; judges fall back to the final-reply view.", exc_info=True
        )
        return None


#: One warning per distinct (declared role, failure) pair — a shape the canonicaliser cannot
#: read is a per-MESSAGE condition, so an unthrottled log would emit one line per message per
#: turn for the whole run. See :func:`_canonical_message`.
#:
#: RUN-scoped, not process-scoped: :func:`reset_warning_ledgers` empties it at run start, so
#: "for the whole run" holds for every run in an interpreter. Its key stays
#: ``(role, failure)`` rather than gaining a scenario — :func:`_canonical_message` is also the
#: seed-compare's reader (:meth:`_MessageIdentity.of`), a frame with no thread to name.
_warned_uncanonical: set[tuple[str, str]] = set()

#: One warning per (thread, ``agent_state`` key path) the wire cap replaced: a harness that keeps an
#: oversized value in its state keeps it on every turn, and each scenario's log needs its own line.
#: RUN-scoped like the ledger above.
_warned_omitted_state: set[tuple[str, str]] = set()


def _reference_by_fold(served: Mapping[str, str]) -> dict[str, str]:
    """``served`` (reference → fold) inverted. Byte-identical documents share a fold, and the first reference served for them names it."""
    by_fold: dict[str, str] = {}
    for reference, folded in served.items():
        by_fold.setdefault(folded, reference)
    return by_fold


def _wire_state_of(state: dict[str, Any] | None, reference_of: Mapping[str, str], *, thread_id: str, label: str = 'agent_state') -> dict[str, Any] | None:
    """``state`` as BOTH of its carriers record it: the request body and the transcript row. ``label`` names it in the logs.

    A served document the agent echoes goes back as its ``<scheme>:///`` reference, and a string leaf
    over the wire cap as a descriptor (:mod:`plurai.orchestrator.state_wire`). One value for both
    carriers, so a re-score of the transcript reads what the live judges read. Applies to a turn's
    ``agent_state`` and to a pipeline hop's outputs alike.
    """
    wire = wire_agent_state(state, served=reference_of)
    logger = logging.getLogger(__name__)
    if wire.referenced:
        logger.debug('Thread %s: %s %s sent as the served document reference.', thread_id, label, ', '.join(wire.referenced))
    for path in wire.omitted:
        if (thread_id, f'{label}.{path}') not in _warned_omitted_state:
            _warned_omitted_state.add((thread_id, f'{label}.{path}'))
            logger.warning(
                'Thread %s: %s.%s is over the %s-byte wire cap; it is sent and recorded as a descriptor (size and sha256). '
                'A judge scoring this field sees the descriptor, not the value.',
                thread_id,
                label,
                path,
                f'{AGENT_STATE_LEAF_CAP_BYTES:,}',
            )
    return wire.state


def _for_agent(content: str | list[dict[str, Any]] | None, served: Mapping[str, str]) -> str | list[dict[str, Any]] | None:
    """A user turn as the agent receives it: each served document a part references carries the bytes the client fetched.

    The agent has no bearer for the ``/documents`` route, so a ``<scheme>:///documents/…`` value is
    useless to it. Every other reader (the transcript, ``/finalize``, the judges) records the turn
    as the backend sent it, which is why the swap happens here, at the hand-off, and nowhere else.
    """
    return with_part_values(content, served)


def _echo_references(reference_of: Mapping[str, str], *, sent: str | list[dict[str, Any]] | None, served: Mapping[str, str]) -> dict[str, str]:
    """``reference_of`` (fold → reference), naming for each fold the turn ``sent`` carries the reference that turn used.

    Byte-identical documents share a fold, so ``reference_of`` names the first reference served for
    them; an echo answering a turn that handed over another one is recorded with that one.
    """
    return {**reference_of, **{served[value]: value for value in document_part_values(sent) if value in served}}


def _as_recorded(echoes: list[dict[str, Any]], reference_of: Mapping[str, str]) -> list[dict[str, Any]]:
    """Echoed user messages as the transcript records a user turn: a string leaf equal to a fold is its ``<scheme>:///`` reference.

    An echo is the agent's copy of the turn it received, bytes included. It goes through the projection
    ``agent_state`` rides (:func:`~plurai.orchestrator.state_wire.wire_agent_state`), so a handed-over
    document is not stored a second time as a multi-megabyte ``data:`` URI whatever shape the
    framework gave the echo, and a string leaf over the wire cap is a descriptor. A bare base64
    payload without the ``data:`` header is not a fold and stays.
    """
    return [wire_agent_state(msg, served=reference_of).state or msg for msg in echoes]


def _seed_for_agent(seed: dict[str, Any] | None, served: Mapping[str, str]) -> dict[str, Any] | None:
    """The resume seed as the child restores it: each served document among ``agent_state``'s string leaves carries the bytes the client fetched.

    A trace records a served document as its reference and the agent cannot open it, so the swap
    happens here, where the seed is handed to the child. ``seed`` is not mutated: the transcript
    and the seed-trail comparison read it in the reference form.
    """
    state = seed.get('agent_state') if seed else None
    if seed is None or not served or not isinstance(state, dict):
        return seed
    return {**seed, 'agent_state': map_string_leaves(state, lambda _, leaf: served.get(leaf, leaf))}


def with_unread_parts(unread_inputs: list[str] | None, unread_parts: list[str] | None) -> list[str] | None:
    """The run's record of what never reached the agent, for the backend: the unread ``initial_agent_context`` keys and
    the unread handed-over ``file`` parts (``FinalizeResponse.unread_parts``), each by field.

    Joined only HERE, at the backend boundary: the backend tells the two apart by its own declaration (the scenario's
    handed-over documents) and gives a part the ``turn_documents`` remedy, so a part is never graded as a key. The
    child and every local reader (the setup smoke) keep them apart. ``None`` only when neither side claims anything.
    """
    if unread_inputs is None and not unread_parts:
        return unread_inputs
    return sorted({*(unread_inputs or ()), *(unread_parts or ())})


def _context_for_agent(context: dict[str, Any], served: Mapping[str, str]) -> dict[str, Any]:
    """The scenario's ``initial_agent_context`` as the agent receives it: each served document carries the bytes the client fetched.

    The backend binds a document over its inline cap as a ``<scheme>:///documents/…`` reference and the
    agent cannot open it, so the swap happens where the context is handed over: the child's ``init``
    and the pipeline hop's inputs, at any depth of mappings and lists. ``context`` is not mutated: the
    transcript and ``/finalize`` record it in the reference form.
    """
    if not served:
        return context
    return map_string_leaves(context, lambda _, leaf: served.get(leaf, leaf))


@cache
def _canonicaliser() -> tuple[Any, Any, Any] | None:
    """The canonicalising callables ``(convert_to_messages, _serialize_message,
    _lc_message_type)``, or ``None`` when they are not importable.

    Imported LAZILY and function-locally for the cycle :func:`_seed_baseline` documents
    (``plurai.child`` imports ``plurai``, which imports this module), and cached —
    the ``None`` included — because the alternative is that import machinery runs per message
    per compare.
    """
    try:
        from langchain_core.messages import convert_to_messages  # noqa: PLC0415

        from plurai.child import _lc_message_type, _serialize_message  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — no langchain_core: every compare falls back, none fails
        logging.getLogger(__name__).warning(
            'could not import the canonical message serializer; agent chains and the seed baseline will be '
            'compared on the fields both sides assert rather than on one normalized shape.',
            exc_info=True,
        )
        return None
    return (convert_to_messages, _serialize_message, _lc_message_type)


def _resolved_type(msg: dict[str, Any]) -> Any:
    """The message's LangChain discriminator where the CHILD's own resolver can name one, and
    whatever the message declared where it cannot.

    Reached only for a message :func:`_canonical_message` refused, and it is the half of the
    normalisation that still applies there: ``_lc_message_type`` maps ``assistant`` / ``model`` /
    a class-name spelling onto one discriminator without needing to read the rest of the message,
    so a shape that is unreadable for some OTHER reason — no ``content`` key, a call with no id —
    does not additionally fail on its role being spelled the provider's way.
    """
    canonicaliser = _canonicaliser()
    resolved = canonicaliser[2](msg) if canonicaliser is not None else None  # _lc_message_type
    return resolved or msg.get('type') or msg.get('role')


def _canonical_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """One message as the SEED's own serializer would have written it — ``None`` when that
    serializer cannot read the shape.

    Both sides of the seed compare go through this one serializer. ``_seed_baseline`` builds the
    seeded prefix with ``plurai.child._serialize_message`` over ``convert_to_messages``, but
    ``_serialize_message`` short-circuits on ``isinstance(msg, dict)`` and never reaches
    ``convert_to_messages``, so without this an agent returning OpenAI role dicts would meet the
    seed's LangChain shape in :meth:`_MessageIdentity.of` and differ on three fields at once: a
    ``tool_calls`` entry spelled ``{'id', 'type': 'function', 'function': {'name',
    'arguments'}}`` would project as ``(None, id, None)`` against the seed's ``(name, id, args)``,
    ``content: None`` would meet ``''``, and a ``developer`` role (which ``_lc_message_type``
    resolves to nothing) would meet the converter's ``'system'``. The prefix test would then fail
    on message 0 of every turn, and the seed's tool exchange would be filed as this run's turn-1
    output.

    Running BOTH sides through this makes the compare symmetric by construction, and it is the reason
    this is not a per-provider table: ``convert_to_messages`` is the ecosystem's own reader of
    the wire shapes (it is where the ``function``/``arguments`` unpacking, the ``assistant`` /
    ``model`` / ``developer`` role spellings and the ``None`` content all already live), so
    every shape it learns, this learns. Note what is NOT happening: no field is folded on the
    way in — the rule :class:`_MessageIdentity` states at length. ``content: None`` is not
    rewritten to ``''``; it is READ by the same serializer that read the seed, which is a
    different thing and the only one that cannot invent a match the seed side would not make.

    ``None`` (rather than a partial projection) when the shape is unreadable — a Gemini
    ``role: 'model'`` chain, a ``tool_calls`` entry with no ``id``, a message with no
    ``content`` key. :meth:`_MessageIdentity.matches` then compares on the intersection of what
    both sides actually assert rather than on fields nobody wrote.

    The ``role`` patch is the child's OWN resolution, reused rather than re-derived: the child
    stamps ``type`` from ``_lc_message_type`` and leaves the agent's ``role`` beside it, and
    ``convert_to_messages`` reads ``role`` FIRST — so a Gemini dict stamped ``type: 'ai'`` would
    still be refused on its ``role: 'model'``. Where the child could resolve nothing (a
    ``developer`` role), the dict is passed through untouched, because the converter knows
    spellings ``_lc_message_type`` does not.
    """
    canonicaliser = _canonicaliser()
    if canonicaliser is None:
        return None
    convert, serialize, lc_type = canonicaliser
    try:
        resolved = lc_type(msg)
        converted = convert([{**msg, 'role': resolved} if resolved else msg])
        return serialize(converted[0]) if len(converted) == 1 else None
    except Exception as exc:  # noqa: BLE001 — an unreadable shape is a comparison input, never a run failure
        key = (str(msg.get('type') or msg.get('role') or '?'), type(exc).__name__)
        if key not in _warned_uncanonical:
            _warned_uncanonical.add(key)
            logging.getLogger(__name__).warning(
                'a %s message could not be read by the canonical serializer (%s: %s); messages of this shape are '
                'compared against the seed on the fields both sides assert, which matches less often — a chain that '
                'then fails the prefix test is appended whole and says so.',
                key[0],
                key[1],
                exc,
            )
        return None


@dataclass(frozen=True, slots=True)
class _UnprojectableToolCall:
    """A ``tool_calls`` entry the projection has no shape for, carried verbatim.

    A non-mapping entry has no ``name``/``id``/``args`` to read, so it stands for itself.
    Wrapping it gives it a type of its own, which is what stops it comparing equal to a
    projected call whose three fields happen to all be ``None``.
    """

    value: Any


@dataclass(frozen=True, slots=True)
class _ToolCallIdentity:
    """Which tool, under which id, with which arguments — what tells one call from another.

    The serde's decorative ``type: 'tool_call'`` entry key is dropped: it reads the same on
    every call, so it distinguishes nothing, and keeping it would only make two records of the
    same call compare unequal.
    """

    name: Any
    id: Any
    args: Any

    @classmethod
    def of(cls, entry: Any) -> _ToolCallIdentity | _UnprojectableToolCall:
        if not isinstance(entry, dict):
            return _UnprojectableToolCall(entry)
        return cls(name=entry.get('name'), id=entry.get('id'), args=entry.get('args'))


class _Unasserted:
    """A field the message did not declare AT ALL — distinct from one it declared as ``None``.

    The two are different claims, and one OpenAI-shaped assistant message makes both in the same
    dict: ``content: None`` (declared, empty) beside no ``tool_call_id`` key (never mentioned).
    Reading an absence as an assertion of ``None`` would let an unreadable shape compare equal
    to anything else that happened to be silent on the same fields.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return '<unasserted>'


#: The one instance of :class:`_Unasserted`; compared with ``is``.
UNASSERTED = _Unasserted()


@dataclass(frozen=True, slots=True)
class _MessageIdentity:
    """What two serialized messages have to agree on to be the same message: what was SAID, not
    how it was serialized. :func:`_fresh_turn_tail` compares on this.

    Full-dict equality is defeated on every resumed run. The agent's checkpointer round trip
    re-materializes the seeded messages with metadata defaults filled in —
    ``response_metadata``/``name`` appear, and ``additional_kwargs`` comes back NON-empty on ai
    messages (the serde parks unknown ``model_dump`` fields such as
    ``invalid_tool_calls``/``usage_metadata`` there, so "strip empties then compare" does not fix
    it) — while the fields below stay byte-verbatim.

    One rule holds across every field: project the STRUCTURE, compare the VALUES verbatim.
    Nothing is folded on the way in — no empty-to-``None``, no default stamped over an absent
    key. Normalizing is the direction that invents matches, and a wrong match here drops a real
    message from the transcript, where a wrong MISmatch only re-appends a chain and says so in
    the log.

    ``id`` is what makes a match certain WHEN BOTH SIDES CARRY ONE: every seeded message is
    required to, so two ids that disagree are two messages however alike the text, and a
    rewriting agent that regenerates ids stays a mismatch. An id absent on EITHER side defers to
    the said-content compare (:meth:`matches`) — the record is matched positionally and in FULL,
    so a match there says the agent re-stated the whole record, which is exactly the question
    being asked. Required, not a courtesy: a plain-list agent (OpenAI role dicts) restores the
    seed WITHOUT the seed's ids, and demanding them would re-append its whole cumulative chain on
    every turn and file the seed's replies under each live turn's chain for the judges.
    """

    type: Any
    content: Any
    id: Any
    tool_calls: tuple[_ToolCallIdentity | _UnprojectableToolCall, ...] | None
    tool_call_id: Any
    #: Did :func:`_canonical_message` read this message? ``False`` means the fields below are
    #: the agent's own spelling, unmediated, and :meth:`matches` must not assume the other side
    #: spells the same message the same way — see :meth:`_matches_on_asserted_fields`.
    normalized: bool = True

    @classmethod
    def of(cls, msg: dict[str, Any]) -> _MessageIdentity:
        """Projected from the CANONICAL form of the message wherever one can be built
        (:func:`_canonical_message`), so both sides of a compare are read by one serializer
        rather than each in its own framework's spelling.

        ``role`` is read as the other spelling of ``type`` for the shapes that canonicalisation
        refuses. Unlike the wire projection this stamps no ``'ai'`` default over a message that
        declares neither: a default two different messages would share is a match this must not
        make — hence :data:`UNASSERTED` rather than ``None`` for a key that is simply absent.
        """
        canonical = _canonical_message(msg)
        src = msg if canonical is None else canonical
        tool_calls = src.get('tool_calls', UNASSERTED)
        declared_type = src.get('type') if canonical is not None else _resolved_type(msg)
        return cls(
            type=declared_type if declared_type is not None else UNASSERTED,
            content=src.get('content', UNASSERTED),
            id=src.get('id'),
            # A DECLARED empty list is "this message made no calls", which is the same claim the
            # absence of the key makes on a human message; both compare equal, and neither is
            # evidence of identity (see ``_SAID_FIELDS`` below).
            tool_calls=tuple(_ToolCallIdentity.of(tc) for tc in tool_calls) if isinstance(tool_calls, list) and tool_calls else None,
            tool_call_id=src.get('tool_call_id', UNASSERTED),
            normalized=canonical is not None,
        )

    def matches(self, other: _MessageIdentity) -> bool:
        """Same message: said-content fields verbatim, and ids equal whenever both are present."""
        if self.id is not None and other.id is not None and self.id != other.id:
            return False
        if not (self.normalized and other.normalized):
            return self._matches_on_asserted_fields(other)
        return self.type == other.type and self.content == other.content and self.tool_calls == other.tool_calls and self.tool_call_id == other.tool_call_id

    #: The said-content fields, and whether agreeing on one is EVIDENCE of identity. ``type`` is
    #: not: every ai message in every chain shares it, so a compare that rested on it alone would
    #: match any two replies. The others are, when non-empty — an empty ``content`` distinguishes
    #: nothing either, which is why it is not enough on its own.
    _SAID_FIELDS = (('content', True), ('tool_calls', True), ('tool_call_id', True), ('type', False))

    def _matches_on_asserted_fields(self, other: _MessageIdentity) -> bool:
        """The compare for a shape no serializer here can read, on either side.

        Whole-projection equality is wrong once one side is unmediated: the two dicts describe
        the same message in different vocabularies, and every field the reader could not resolve
        reads as an absence. So compare on the INTERSECTION of what both sides actually assert —
        a disagreement anywhere is still a mismatch — and require at least one DISCRIMINATING
        field to have agreed, the rule ``matches`` already applies to ``id``. Without that floor
        two messages that share only ``type`` (or agree only by both being silent) would match,
        which is the direction that drops a real message from the transcript.

        Ids that both sides carry are settled in ``matches`` before this runs; two agreeing ids
        ARE the discriminating field, so nothing further is required of the payload.
        """
        agreed_on: list[str] = []
        for name, discriminating in self._SAID_FIELDS:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine is UNASSERTED or theirs is UNASSERTED:
                continue
            if mine != theirs:
                return False
            if discriminating and mine not in (None, '', ()):
                agreed_on.append(name)
        return bool(agreed_on) or (self.id is not None and other.id is not None)


def _relisted_run(recorded: list[_MessageIdentity], returned: list[_MessageIdentity]) -> int:
    """The longest stretch of CONSECUTIVE recorded messages that comes back, in order and
    consecutively, anywhere in the returned chain.

    Length, not membership, because an id-less message is matched on its content
    (:meth:`_MessageIdentity.matches`): ONE such hit is as easily a repeated canned line as a
    re-listing, and a run of two is not — the same reading :data:`_SEED_EVIDENCE_MIN_RUN` already
    applies to the seed overlap.
    """
    longest = 0
    for i in range(len(recorded)):
        for j in range(len(returned)):
            run = 0
            while i + run < len(recorded) and j + run < len(returned) and returned[j + run].matches(recorded[i + run]):
                run += 1
            longest = max(longest, run)
    return longest


def _an_id_matched(recorded: list[_MessageIdentity], returned: list[_MessageIdentity]) -> bool:
    """Did a recorded message come back under the SAME id, both sides carrying one?

    That is certain evidence on its own and needs no run: :meth:`_MessageIdentity.matches` rules
    that two ids that disagree are two messages, so two that agree, on a pair that also compares
    equal, are the one message re-listed. A same-id pair whose CONTENT was rewritten does not
    match and is not reported here — that is the wholesale summariser :func:`_fresh_turn_tail`
    declares below as a knowingly unreported loss.
    """
    return any(a.id is not None and any(t.id is not None and t.matches(a) for t in returned) for a in recorded)


def _fresh_turn_tail(accumulated: list[dict[str, Any]], turn_messages: list[dict[str, Any]], *, thread_id: str, seeded: int) -> list[dict[str, Any]]:
    """The messages this turn NEWLY produced, given what earlier turns already recorded.

    The `_fresh_tool_calls` treatment, applied to the message chain itself: a
    MemorySaver-style agent returns its CUMULATIVE history every turn, so blindly
    extending makes the recorded transcript QUADRATIC — every reply re-listed 1..N
    times. Turn-scoped
    evals read the per-turn slice, so those scores would survive — but the flattened
    chain also feeds ``merge_recovered_tool_calls`` into /finalize, and recovery does
    not dedupe by id, so for a ``mode:real`` tool (no authoritative events to dedupe
    against) one call re-listed N times becomes N events, inflating
    the backend's tool-call counts server-side.
    So this guards eval numbers for cumulative agents with real tools, not just the
    record.

    PREFIX-detecting, never a blind slice: when the turn's chain re-states the
    accumulated record as its leading messages, only the tail past it is returned —
    empty when the turn added nothing (a no-op turn re-returning the same history:
    ``>=``, not ``>``, or the whole history would be appended a second time on the
    easiest case there is). A stateless/delta agent — whose chain does NOT re-list
    history — falls through unchanged.

    A TRIMMING agent (a context window: ``[system, the last k messages…, new]``) re-lists
    messages this run's own turns already recorded, out of prefix. Each one that comes back under
    the id it was recorded with, saying the same thing, is that message — certain evidence, no
    run needed (:func:`_an_id_matched`) — and is EXCISED, so a turn's chain is only what the turn
    produced. Without it every earlier reply would be recorded again as part of the latest turn,
    and a turn-scoped check reading "this turn's speech" would read every earlier turn's too. Only LIVE
    messages are excised — the first ``seeded`` of ``accumulated`` are the resume seed, whose
    out-of-prefix re-listing :func:`_seed_restore_reading` reports as its own finding.

    An agent that REWRITES earlier messages (a summarisation/compaction node, anything that
    regenerates ids or the content itself) still falls through, which re-appends what it
    re-listed — so that case is logged, gated on
    EVIDENCE of the re-listing (a RUN of :data:`_SEED_EVIDENCE_MIN_RUN` recorded
    messages comes back consecutively, or a pair matched under ids both sides carry,
    while the chain does not open with the WHOLE record) and NOT on the chain being
    longer than the record. One id-less message matching is NOT that evidence: it is
    matched by content, so a delta agent's repeated canned line reads identical to a
    re-listing — a single id-less content match is a canned line as often as it is a
    rewrite, which is the report
    :data:`_SEED_EVIDENCE_MIN_RUN` exists to refuse.
    And length is no evidence either — it reads as turn INDEX: the same 2-message delta turn is
    longer than a 1-message record and shorter than a 4-message one — and a length gate
    is silent on the trimming agent it names, whose chain is SHORTER than the record
    (record 5, chain [m3, m4, m5] ⇒ three messages recorded twice, and a ``> n`` gate would
    log nothing), which is why the re-listing test runs outside the
    length gate while the SLICE still requires the full prefix at ``>=``. Knowingly
    unreported: a rewrite the identity compare cannot see through — ids regenerated on
    both sides (two disagreeing ids are two messages, per :class:`_MessageIdentity`) or
    content replaced wholesale, even under a surviving id.

    Positional, on the :class:`_MessageIdentity` projection — NOT full-dict equality. The
    dicts are byte-stable across the subprocess JSON boundary alone, but a RESUMED
    run's seeded prefix additionally round-trips the agent's checkpointer, which fills
    metadata defaults full-dict equality trips over while the identity fields stay
    verbatim. Under full-dict equality the compare never matches, so the whole
    cumulative chain is re-appended EVERY turn, every seed id recorded once per turn."""
    n = len(accumulated)
    # An EMPTY chain can match nothing, and projecting the record for it would emit
    # ``_canonical_message``'s own warning on a turn that has nothing to say.
    if n and turn_messages:
        # Projected ONCE per list rather than once per pair: the projection runs the message
        # through the canonical serializer, and the prefix compare would otherwise re-read the same
        # accumulated message on every turn. The WHOLE returned chain, and outside the length gate,
        # because the re-listing test below has to look past the prefix and at a chain SHORTER than
        # the record — the trimming agent, which is the one that actually inflates.
        recorded = [_MessageIdentity.of(a) for a in accumulated]
        returned = [_MessageIdentity.of(t) for t in turn_messages]
        # The slicing branch: prefix match on ``>=`` (not ``>``) or the whole history is
        # appended a second time on a no-op turn.
        if len(turn_messages) >= n and all(t.matches(a) for t, a in zip(returned[:n], recorded, strict=True)):
            return turn_messages[n:]
        live_by_id: dict[Any, list[_MessageIdentity]] = {}
        for a in recorded[seeded:]:
            if a.id is not None:
                live_by_id.setdefault(a.id, []).append(a)
        held = {i for i, t in enumerate(returned) if t.id is not None and any(t.matches(a) for a in live_by_id.get(t.id, ()))}
        rest = [t for i, t in enumerate(returned) if i not in held]
        # What is left re-lists history only on evidence: a run of consecutive re-listed messages,
        # or an id-matched pair (a seeded one) — never a single id-less content match, which is a
        # repeated canned line as often as it is a re-listing.
        if _relisted_run(recorded, rest) >= _SEED_EVIDENCE_MIN_RUN or _an_id_matched(recorded, rest):
            logging.getLogger(__name__).warning(
                'thread %s: agent returned %d messages against %d already recorded; recorded messages re-appear in '
                'the chain without the id they were recorded under, or as seeded history, and the chain does not '
                'open with the WHOLE record, so they are appended. An agent that REWRITES history (re-ordering, '
                'regenerated ids) inflates the flattened transcript and any tool calls recovered from it. A chain '
                'that re-lists NOTHING is a delta agent and is not reported here.',
                thread_id,
                len(turn_messages),
                n,
            )
        return [m for i, m in enumerate(turn_messages) if i not in held]
    return turn_messages


def _accept_agent_chain(
    full_messages: list[dict[str, Any]] | None,
    *,
    sent: str | list[dict[str, Any]] | None,
    raw_seen: list[dict[str, Any]],
    seeded: int,
    thread_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One reply's chain, de-duplicated then split: ``(agent side, declared, verbatim echoes)``.

    The two steps that MUST stay in step, so the driven turns and the closing turn cannot drift
    apart on them:

    * :func:`_fresh_turn_tail` first — a cumulative (MemorySaver) agent re-lists prior turns'
      messages on every reply, and extending verbatim would make the record quadratic.
    * the RAW tail is what accumulates into ``raw_seen``, NOT the split's agent side.
      ``_fresh_turn_tail`` detects a cumulative agent by matching its chain against this exact
      record, so a record with the echoes filtered out stops matching the prefix — which
      re-opens the quadratic transcript the dedup exists to prevent.

    ``raw_seen`` is extended in place; the caller owns what it does with the three lists. Its first
    ``seeded`` messages are the resume seed (:func:`_fresh_turn_tail` excises only live re-listings).
    """
    raw = _fresh_turn_tail(raw_seen, full_messages or [], thread_id=thread_id, seeded=seeded)
    raw_seen.extend(raw)
    return split_user_echo(raw, input_user_message=sent)


#: What the SDK could establish about the seed restore from an agent reply.
#:
#: * ``confirmed`` — the chain opened with the whole baseline. The restore is proven, and this
#:   is the ONE state that publishes nothing: an absent report is the wire's word for a full
#:   restore, so the two already agree.
#: * ``partial`` — seeded messages came back, but not as the chain's leading messages, so they
#:   were recorded (and are judged) as this run's own output.
#: * ``undetermined`` — nothing here could tell those two apart. A delta agent's chain re-lists
#:   no history at all, and the SDK cannot see whether it holds the seed internally. Published
#:   BECAUSE it cannot be told: the plugin's seed-state contract reads an absent report as a
#:   positive full-fidelity claim, so silence here would assert what was never measured.
SeedRestoreState = Literal['confirmed', 'partial', 'undetermined']

#: How many seeded messages must come back CONSECUTIVELY, in the seed's own order, before an
#: overlap is evidence of a restore rather than a coincidence. One message is not: an agent that
#: opens its live leg with a canned line the seed also contains matches it id-lessly, and a
#: single such hit read as a partial restore is the false report this floor exists to refuse.
#: :func:`_fresh_turn_tail` reads it for the same question about the RECORD rather than the seed —
#: one id-less content match is a repeated canned line as often as it is a re-listing.
_SEED_EVIDENCE_MIN_RUN = 2


def _seed_only(msg: _MessageIdentity, *, sent_user_messages: frozenset[str]) -> bool:
    """Could only the SEED have put this message in the agent's chain?

    Two shapes qualify, and neither is a judgement about content:

    * a TOOL EXCHANGE — a tool result, or a call the agent is re-listing rather than making.
      This run records live tool calls from the same chain, so a re-listed one is history.
    * a HUMAN turn this run has not sent. The live user's words are echoed back by most
      frameworks' chains, so a human message matching one the harness sent says nothing; one
      that matches NO message this run sent was put there by the seed.

    An ``ai`` message is deliberately not enough on its own: it is the shape an agent can
    coincidentally re-emit (a canned opener, a repeated apology), which would be a false report.
    A ``system`` message is not either — an agent that prepends its own system prompt every turn
    would otherwise look like it had restored the seed's.
    """
    if msg.type == 'tool' or msg.tool_calls:
        return True
    return msg.type == 'human' and msg.content not in sent_user_messages


def _undetermined(seeded_messages: int, relisted: int, why: str) -> dict[str, Any]:
    """An ``undetermined`` reading, with the one sentence every such report must end on.

    That closing sentence is the whole reason this state exists: the plugin's seed-state
    contract reads an ABSENT report as a positive full-fidelity claim, so a reader who finds
    this key has to be told, in the field itself, that it is the opposite of one.
    """
    return {
        'state': 'undetermined',
        'seeded_messages': seeded_messages,
        'relisted_by_agent': relisted,
        'note': f"{why}. It is NOT a full-fidelity claim: read the harness's own report beside it.",
    }


def _seed_restore_reading(seed_baseline: list[dict[str, Any]], chain: list[dict[str, Any]], *, sent_user_messages: frozenset[str]) -> dict[str, Any] | None:
    """What one agent reply says about the seed — ``None`` only on a thread that was never seeded.

    A reply that carried NO chain is a reading too, and it is ``undetermined``: an adapter that
    surfaces no messages leaves the restore unobservable for the whole run, and returning
    ``None`` for it would publish nothing, which the wire reads as a full-fidelity claim — the one
    thing this must never assert without measuring.

    :func:`_fresh_turn_tail` answers one question — which messages are new — and answers it
    conservatively: a chain that fails the prefix test falls through and is appended whole. That
    is the right default for a delta agent, whose chain never re-lists history and where falling
    through appends exactly the new messages. It is the WRONG outcome for a resumed thread whose
    harness restored only part of the seed: the agent then re-lists the part it did get, the
    prefix test fails on the first message the harness left out, and the re-listed history is
    appended as turn-1 output — scored as this run's opening move. The chain is shorter than the
    record in that shape, so the longer-chain warning cannot see it, and nothing else does either.

    The three states this has to tell apart, all on a thread that WAS seeded — see
    :data:`SeedRestoreState`, which is what the reading is keyed on:

    * a full restore — the chain opens with the whole baseline, prefix-matches, and the seed is
      stripped. ``confirmed``.
    * a delta or stateless agent — the chain re-lists NOTHING of the baseline, or there is no
      chain at all. ``undetermined``, never ``partial``: this is the ordinary shape for an agent
      that returns only new messages, and the SDK cannot see whether such an agent holds the seed
      internally. Reported as the absence of a measurement rather than as a fault, and reported at
      all only because the wire reads silence as a full restore.
    * a partial restore — an overlap that is EVIDENCE. The harness restored part of the seed, or
      the agent rewrote it; either way the messages that DID come back are recorded, and judged,
      as this turn's output. That is the shape reported, and the log says so.

    What makes an overlap evidence, and why bare membership is not (a canned opener would be a false report): three
    rules, none of them about a provider or a customer.

    1. The overlap is a ONE-TO-ONE assignment. Counting "seeded messages some returned message
       matches" would let ONE returned line satisfy several seeded ones — a seed carrying the
       same courtesy twice would read as two restored messages against a chain that re-listed one.
    2. The assigned messages must form a CONTIGUOUS run in the seed's own order, at least
       :data:`_SEED_EVIDENCE_MIN_RUN` long. A restore replays a conversation; a coincidence
       matches a line here and a line there.
    3. That run must contain at least one message only the seed could have produced
       (:func:`_seed_only`). A delta agent whose live leg opens with a canned line the seed also
       contains matches it id-lessly — id-less matching being what a plain-list role-dict agent
       requires — and without this rule that agent is reported as having re-listed history.

    Membership, not position, is what looks for the overlap: a partial restore that kept the
    seed's TAIL re-lists messages the prefix test never reaches. Comparison is by
    :class:`_MessageIdentity` for the reason :func:`_fresh_turn_tail` uses it — a seeded message
    that round-trips the agent's checkpointer comes back with metadata defaults filled in.
    """
    seeded = [_MessageIdentity.of(m) for m in seed_baseline]
    if not seeded:
        return None
    if not chain:
        return _undetermined(
            len(seeded),
            0,
            'this reply carried no message chain at all, so there was nothing here to read the restore from — an '
            'agent whose adapter surfaces no messages leaves this unobservable for the whole run',
        )
    returned = [_MessageIdentity.of(m) for m in chain]
    if len(returned) >= len(seeded) and all(r.matches(s) for r, s in zip(returned[: len(seeded)], seeded, strict=True)):
        return {'state': 'confirmed', 'seeded_messages': len(seeded)}
    assigned = _assign_seeded_messages(seeded, returned)
    run = _longest_ordered_run(assigned)
    if len(run) < _SEED_EVIDENCE_MIN_RUN or not any(_seed_only(seeded[i], sent_user_messages=sent_user_messages) for i in run):
        return _undetermined(
            len(seeded),
            len(assigned),
            "the agent's chain neither opened with the seeded conversation nor re-listed enough of it to tell a "
            'partial restore from a coincidental repeat, so this run could not establish whether the seed was '
            'applied in full',
        )
    return {
        'state': 'partial',
        'seeded_messages': len(seeded),
        'relisted_by_agent': len(assigned),
        'recorded_as_turn_output': len(assigned),
        'contiguous_run': len(run),
        'note': (
            'the agent re-listed part of the seeded conversation but not as the leading messages of its chain, so '
            'the seed could not be stripped from that turn: those messages are recorded, and judged, as this '
            "run's own output."
        ),
    }


def _assign_seeded_messages(seeded: list[_MessageIdentity], returned: list[_MessageIdentity]) -> dict[int, int]:
    """``{seed index: chain index}`` — each returned message satisfying at most ONE seeded one.

    A greedy assignment in the seed's order, taking the earliest chain position still free. The
    one-to-one part matters: a plain ``any(r.matches(s) for r in returned)`` count reads
    a seed that says the same thing twice (a repeated greeting, two identical tool results) as
    twice as much restored history as the chain actually carried.
    """
    taken: set[int] = set()
    assigned: dict[int, int] = {}
    for si, s in enumerate(seeded):
        for ri, r in enumerate(returned):
            if ri not in taken and r.matches(s):
                assigned[si] = ri
                taken.add(ri)
                break
    return assigned


def _longest_ordered_run(assigned: dict[int, int]) -> list[int]:
    """The longest run of CONSECUTIVE seed indices whose chain positions also ascend — the seed
    indices of the longest stretch of the seeded conversation the chain replayed in order.

    Both halves are required. Consecutive-in-the-seed alone would accept a chain that quotes two
    neighbouring seeded messages backwards, which is a rewrite, not a replay; ascending-in-the-
    chain alone would accept two matches from opposite ends of the seed.
    """
    best: list[int] = []
    run: list[int] = []
    for si in sorted(assigned):
        if run and si == run[-1] + 1 and assigned[si] > assigned[run[-1]]:
            run.append(si)
        else:
            run = [si]
        if len(run) > len(best):
            best = list(run)
    return best


def _note_seed_restore(
    seed_baseline: list[dict[str, Any]],
    chain: list[dict[str, Any]],
    *,
    fidelity: dict[str, Any] | None,
    thread_id: str,
    sent_user_messages: frozenset[str],
) -> tuple[dict[str, Any] | None, bool]:
    """``(fidelity with the SDK's reading folded in, is the check still open)``.

    A restore this run could not confirm must not reach finalize as ``None``, because ``None`` is
    the wire's word for "the seed was applied in full" — the state neither a partial restore nor
    an unreadable one is in. Returning a dict costs nothing on the wire (the field is an open
    object) and makes the report say what the run saw rather than what the harness hoped.

    The check stays OPEN until it reaches a conclusive state, rather than being consumed by
    whichever reply arrives first. Consuming it would let an agent whose first chain said nothing
    usable spend the only reading there is, and a partial restore visible on turn 2 would never
    be looked at. Staying armed is safe only because the evidence rule is a contiguous ordered run
    containing a message only the seed could have produced (:func:`_seed_restore_reading`) — the
    later turns of a cumulative agent re-list that agent's OWN live output, which is not seeded
    and cannot form such a run.

    ``confirmed`` CLEARS the key rather than stamping it: a proven full restore and the wire's
    silence say the same thing, and leaving an earlier ``undetermined`` stamp behind would
    contradict a later measurement. The key is overwritten, not merged: a harness that filled it
    in was describing something the SDK measures for itself, and the measurement wins.
    """
    reading = _seed_restore_reading(seed_baseline, chain, sent_user_messages=sent_user_messages)
    if reading is None:  # nothing was seeded — there is no restore to report on
        return fidelity, True
    if reading['state'] == 'confirmed':
        cleared = {k: v for k, v in (fidelity or {}).items() if k != SEED_RESTORE_KEY}
        return (cleared or None), False
    if reading['state'] == 'partial':
        logging.getLogger(__name__).warning(
            'thread %s: the agent re-listed %d of the %d seeded messages (%d of them consecutively), but not as the '
            'leading messages of its chain, so the seed could NOT be stripped from that turn — that history is '
            "recorded as this run's own output and the judges will score it as such. Either the harness restored part "
            'of the seed, or the agent rewrote it. The run continues, with resume_fidelity marked so it can no longer '
            'read as a full restore.',
            thread_id,
            reading['relisted_by_agent'],
            reading['seeded_messages'],
            reading['contiguous_run'],
        )
    # ``partial`` is conclusive and closes the check; ``undetermined`` re-states itself on every
    # reply, so a later turn that CAN be read still overwrites it.
    return {**(fidelity or {}), SEED_RESTORE_KEY: reading}, reading['state'] != 'partial'


# ---------------------------------------------------------------------------
# The ``ended_early`` vocabulary: the typed reasons a dialog ended for something
# other than the user reaching their goal
# ---------------------------------------------------------------------------
#
# There is exactly one rule for membership: a reason belongs in this vocabulary when it is known
# FIRST-HAND — either by this harness, or by the one party that can know it and REPORTS it
# over the wire. A futility heuristic fails that rule: it infers from text similarity a judgement
# the user-simulator LLM already makes with the whole conversation in front of it.
#
# The reported half of that rule is what ``USER_GAVE_UP_REASON`` adds, and it is a widening,
# not a loosening: the simulator does not guess either — it declares. The other two below are
# the harness's own: the loop counted its own turns, and the parent knows which replies it
# manufactured itself. None of the three guesses at what the simulator meant.

#: The typed reason recorded when the dialog ran OUT of turns: the loop exhausted ``max_turns``
#: while the user simulator was still going. This run used every turn it had and was cut anyway,
#: and it maps to this field because a consumer asks one question of every reason here: did the
#: HARNESS end this conversation, or did the agent?
#:
#: Left as ``None``, it would be indistinguishable from a dialog that resolved on its own.
#: A ceiling-hit scenario scores low for every obligation its conversation never reached, and
#: with nothing recorded, the report and the suite critic downstream both read that as the
#: agent's failure — a budget artifact wearing the agent's name.
TURN_CAP_REASON = 'turn_cap_exhausted'

#: The typed reason recorded when EVERY turn of a dialog was emptied by the parent rather than
#: answered by the agent (see ``ChildProcess.degraded_turns``). A mis-wired ``simulate.py`` — a
#: reply returned under a key the harness does not read, a graph that returns its state before
#: the AI message is appended — degrades on every turn, and the run then reports a conversation
#: in which the agent said nothing at all.
#:
#: That is not a measurement of the agent, and it belongs in this vocabulary for exactly the
#: reason the turn cap does: the alternative is a verdict about the harness published under the
#: agent's name. Individual degraded turns stay scoreable (an agent CAN legitimately go silent
#: once, and that is a verdict) — it is the all-or-nothing case that cannot be one.
ADAPTER_DEGRADED_REASON = 'adapter_degraded'

#: The typed reason recorded when the SIMULATED USER declared its goal unreachable and stopped
#: trying. Reported by the backend on ``UserTurnResponse.ended_early`` — the simulator is
#: the only party that can know it, so this harness relays rather than infers.
#:
#: ``should_terminate`` tracks the goal being MET, so without this reason a persona chasing an
#: unanswerable target loops to the budget and this channel reports ``turn_cap_exhausted``. The two
#: route to different owners: one is a finding about the scenario, the other a setting on the run.
#:
#: The string matches the backend's own constant. Read off the wire and NOT validated against this
#: name: the vocabulary is shared and will grow, so a value this build does not recognise is
#: carried through rather than dropped — an unknown reason is still more than no reason.
USER_GAVE_UP_REASON = 'user_gave_up'

#: The simulated user's own stated CAUSE for a give-up, reported beside the reason on the
#: /user-turn response (``UserTurn.ended_early_cause``) and carried on
#: ``ThreadResult.ended_early_cause``. The simulator already names which of its give-up
#: conditions applied — its rubric is "the assistant refused, said it cannot, or repeated the
#: same non-answer" and its ``reasoning`` is told to say which. Without a typed cause every give-up
#: routes to ONE remedy ("look at the SCENARIO (or at what the agent refused)"); the cause separates
#: an agent that refused from a harness that did not surface a render, and only the party that gave
#: up can state it — so, like the reason, it is RELAYED, never inferred here.
#:
#: These are the values this build knows how to ROUTE. Read off the wire and never validated
#: against them: an unknown cause is carried through and rendered verbatim with no remedy
#: asserted, the same open-vocabulary posture as the reason.
GIVE_UP_CAUSE_AGENT_REFUSED = 'agent_refused'
GIVE_UP_CAUSE_AGENT_CANNOT = 'agent_cannot'
GIVE_UP_CAUSE_AGENT_REPEATED = 'agent_repeated'
#: The assistant answered SOMETHING ELSE and did not come back when pushed. The three above all
#: describe an assistant that ENGAGED with the ask and declined it; this one talks past it, which
#: is a case the persona could not otherwise stop on at all.
#:
#: IT IS NOT `agent_refused` WITH A DIFFERENT NAME, which is why it does not reuse that remedy: a
#: diverged agent refused nothing, so "look at what it refused" sends the reader looking for
#: something that is not in the transcript.
GIVE_UP_CAUSE_AGENT_DIVERGED = 'agent_diverged'
#: The harness-shaped cause: the simulated user could not see the agent's reply — blank, or
#: produced somewhere the user is never shown (a render the harness did not surface). A finding
#: about ``simulate.py``.
GIVE_UP_CAUSE_REPLY_NOT_VISIBLE = 'reply_not_visible'

#: The typed reason recorded when the dialog ran out of TIME rather than out of turns: the
#: per-scenario time budget (``--scenario-timeout`` / ``NETWORK__SCENARIO_TIMEOUT``, defaulted
#: from the run phase) was spent while the user simulator was still going.
#:
#: It is the harness's own first-hand knowledge — this loop holds the clock — and it belongs
#: beside ``turn_cap_exhausted`` for the same reason that one does: the conversation was cut by
#: a budget the operator set, so every obligation it never reached scores as unmet, and with
#: nothing recorded the report and the suite critic both read that as the agent's failure.
#:
#: ``max_turns`` cannot express this. One real-mode turn can legitimately cost minutes (a real
#: generator and full-table renders, with no hang and no retries), so a turn count is not a time
#: bound. The HARD ceiling
#: (``--scenario-timeout``) is not this either: it CANCELS the attempt, so the row ends as
#: ``error.json`` with nothing graded. This ends the dialog through the ordinary channel — the
#: closing turn is still delivered and /finalize still runs — so budget exhaustion becomes a
#: scored outcome instead of silent wall clock. That ceiling is DERIVED and internal
#: (``derive_scenario_hard_ceiling``): it catches a hang, while ``--scenario-timeout`` IS this
#: budget, defaulted per run phase.
#:
#: Reported to the backend on ``FinalizeRequest.ended_early`` — the only reason in the vocabulary
#: the server cannot observe for itself, because the dialog clock is ours. It matters there
#: because a censored transcript is what gates absence-shaped judges to ``not_measured``
#: instead of letting the clock manufacture "the agent never did X" out of turns nobody took.
TIME_BUDGET_REASON = 'time_budget_exhausted'

#: The typed reasons a dialog LANDS on when a turn failed after the conversation had started: it is
#: finalized and graded on the turns it reached, instead of the failure throwing those turns away
#: (see ``Orchestrator.run``'s ``redrive_available``). Each names who ended the dialog:
#:
#: * ``agent_raised`` — the agent's own code raised on its turn and produced no reply. A fact
#:   about the agent, measured like ``agent_ended``: the judges decide whether the goal was met
#:   before the raise.
#: * ``agent_recursion_budget`` — the agent's graph hit its recursion/step limit. Measured too;
#:   the attempt's pre-crash tool activity rides ``turn_failure`` so attribution keeps its split.
#: * ``provider_infra`` — the agent's model provider refused or failed the call. Infrastructure,
#:   never the agent: the dialog is CUT by something outside it, so it is censored like
#:   ``time_budget_exhausted`` and the trial is invalid unless the caller overrules that.
#: * ``turn_budget_exhausted`` — one agent turn blew the per-turn wall. The apparatus' budget,
#:   censored for the same reason as the time budget, and the trial invalid unless overruled, as for
#:   ``provider_infra``.
#:
#: The strings are the raising classes' own kinds where one exists, so a reader of ``error.json``
#: and a reader of a landed transcript share one vocabulary.
AGENT_RAISED_REASON = AGENT_RAISED_KIND
AGENT_RECURSION_REASON = AGENT_RECURSION_BUDGET_KIND
PROVIDER_INFRA_REASON = PROVIDER_INFRA_KIND
TURN_BUDGET_REASON = TURN_BUDGET_EXHAUSTED_KIND

#: The reasons on which the HARNESS or its infrastructure cut the dialog, not the agent or the user:
#: what the agent never reached is unmeasured, not failed. The backend declares the same membership as
#: its censored set; the run summary files these rows ``incomplete``.
HARNESS_CUT_REASONS: frozenset[str] = frozenset({TIME_BUDGET_REASON, PROVIDER_INFRA_REASON, TURN_BUDGET_REASON})

#: Every client-measured reason this harness relays to ``/finalize`` (the replay cut aside): the
#: ones only this side can know. Anything else on ``ended_early`` is the server's own or local.
RELAYED_REASONS: frozenset[str] = frozenset({TIME_BUDGET_REASON, AGENT_RAISED_REASON, AGENT_RECURSION_REASON, PROVIDER_INFRA_REASON, TURN_BUDGET_REASON})


def _turn_failure_record(exc: BaseException, *, turn_idx: int, closing: bool) -> dict[str, Any]:
    """``ThreadResult.turn_failure``: the failed turn, typed by its raiser (``kind`` is ``None`` for an
    untyped raise), with the exception's own words — the ``error.json`` field names."""
    return {
        'turn_idx': turn_idx,
        'closing': closing,
        'kind': getattr(type(exc), 'kind', None),
        'exception': type(exc).__name__,
        'child_exc_class': getattr(exc, 'child_exc_class', None),
        'error': str(exc) or repr(exc),
    }


def landed_reason(exc: BaseException) -> str | None:
    """The ``ended_early`` reason a failed TURN lands the dialog on, or ``None`` when the failure must
    stay an error. Typed by the raising class (and, for the turn wall, by the envelope it guarded),
    never by message text. Harness faults, a dead child, an untyped raise and every other kind are
    ``None``: their owner is the harness, or unknown."""
    if isinstance(exc, AgentTurnTimeout):
        return TURN_BUDGET_REASON if exc.envelope_type == USER_TURN_ENVELOPE else None
    if isinstance(exc, ProviderInfra):
        return PROVIDER_INFRA_REASON
    if isinstance(exc, GraphRecursionWrap):
        return AGENT_RECURSION_REASON
    if isinstance(exc, AgentRaised):
        return AGENT_RAISED_REASON
    return None


#: The typed reason recorded when the AGENT ended the conversation itself, and its twin for the
#: agent handing the conversation to somebody else. Reported by the backend on
#: ``UserTurnResponse.ended_early``: the agent says so by returning an ``end_conversation`` /
#: ``handoff`` action on its turn, the server reads that off ``agent_actions`` and ends the
#: dialog, and this harness RELAYS the value like every other reported reason.
#:
#: They belong in this vocabulary under its one membership rule — known first-hand by the party
#: that can know it, and reported. The agent is that party here, in the same way the simulated
#: user is for ``USER_GAVE_UP_REASON``: nothing in this loop infers a closure from a goodbye, a
#: silence or a turn count, and a harness that never emits the verbs never produces these values.
#:
#: They are the answer to the question every consumer asks of this field — did the HARNESS end
#: this conversation, or did the agent? — and without them the second answer could not be
#: given. An agent-closed dialog would be ``ended_early=None``, byte-identical to one the user
#: finished satisfied, while its evals read low for every obligation the closure cut short.
#:
#: The strings match the backend's own constants and, like the rest of the vocabulary, are read off
#: the wire and never validated against these names.
AGENT_ENDED_REASON = 'agent_ended'
HANDED_OFF_REASON = 'handed_off'

#: The reasons on which the dialog's other end is GONE — the agent closed the session or handed
#: it to somebody else. The loop reads this to decide one thing: whether to deliver the closing
#: turn (see the closing-turn block, which exists because an ordinary terminating /user-turn
#: leaves its own user message unanswered). After these two there is nobody left to answer it,
#: so the "unanswered" message is a line sent into a session the agent already left and its
#: reply is a measurement of a conversation that was over.
#:
#: A SET rather than a test on ``ended_early`` being set at all: a give-up, a turn cap and a
#: time budget all end a dialog whose agent is still there, and their closing exchange is the
#: one turn several evals are written against.
AGENT_CLOSED_REASONS: frozenset[str] = frozenset({AGENT_ENDED_REASON, HANDED_OFF_REASON})

#: The typed reason recorded when the SERVER's user simulator produced no user turn at all — its
#: draw came back unusable, its one re-draw came back unusable too, and the dialog stopped
#: because there was nothing to send. Reported by the backend on ``UserTurnResponse.ended_early``,
#: relayed here like every other reported reason.
#:
#: The string matches the backend's own constant and is spelled ONCE, here: this is the one hop where
#: this SDK has to recognise the backend's exact wire spelling, the same posture
#: ``AGENT_ENDED_REASON`` already has.
SIM_DEGENERATE_REASON = 'sim_degenerate'

#: The two remaining reasons the backend emits, both BACKEND decisions relayed like the others: its
#: scenario schedule ran out (every planted beat delivered) or one of its policies declared the run
#: unsalvageable. Spelled here so the run summary routes each to its owner rather than printing it
#: as unknown; a parity check holds the backend's vocabulary to this build's remedy table.
PLAN_EXHAUSTED_REASON = 'plan_exhausted'
POLICY_TERMINATED_REASON = 'policy_terminated'

#: The reasons on which THIS RESPONSE CARRIES NO USER TURN — the second question the loop asks of
#: ``ended_early``, beside ``AGENT_CLOSED_REASONS``' "is the other end gone?". The two questions
#: stay two, and they are still asked at two different sites: "is anybody left to answer" decides
#: whether a closing turn is delivered, "was a line produced at all" decides whether this response
#: contributes a user turn to anything. On an agent close the backend produces no line either: it
#: reads the close off the turn's ``agent_actions`` BEFORE
#: it draws, and there is nobody left to speak to, so nobody is drawn. So this set is the
#: degenerate reason UNION the agent-closed ones — composed from the declared set rather than
#: re-listing its two strings, so a reason added there is answered here with no second edit.
#:
#: The backend draws no user turn on a turn it knows is the last, so no undeliverable line reaches the
#: graded transcript. A backend without the server-side skip answers an agent close with a real
#: ``user_message``; the reason is declared on both, so reading the declaration keeps the carrier
#: clean against either.
#:
#: KEYED ON THE REASON, NEVER ON THE VALUES. The backend ships this absence as ``user_message == ''``
#: beside ``user_messages == null``, and it would be tempting to read the pair — but that pair is
#: also what an ordinary turn that carried no TEXT looks like, and such a turn is delivered and
#: occupies a row. The backend's own wire description says the same thing
#: from the other side ("THE ABSENCE IS DECLARED BY ``ended_early`` AND IS NEVER INFERRED FROM
#: THAT PAIR"). So the declaration is what this reads, and a reason is what it holds.
NO_USER_LINE_REASONS: frozenset[str] = frozenset({SIM_DEGENERATE_REASON}) | AGENT_CLOSED_REASONS


def _opens_a_decision(actions: list[dict[str, Any]] | None) -> bool:
    """Whether a reply leaves a decision awaiting the user that a typed choice can answer: it carries the
    ``interrupt`` act and that act offers ``choices``. A pause offering none is resumed by free text, which
    no re-entered turn can resolve — the same keying the backend holds its stop on."""
    return any(action.get('name') == INTERRUPT_ACT and (action.get('args') or {}).get('choices') for action in actions or ())


def _user_action_kwarg(action: dict[str, Any] | None) -> dict[str, Any]:
    """``{'user_action': action}`` for a turn that carries the simulated user's act, else ``{}``.

    Passed only when there is one: a ``child_factory`` extension that does not know user acts
    keeps working for every dialog in which the user took none.
    """
    return {} if action is None else {'user_action': action}


def _saved_channel_inputs(response: Any) -> dict[str, Any]:
    """``{'channel_inputs': [...]}`` for a turn the scenario's channel plan planted into (:mod:`plurai.channels`), else ``{}``.

    Written on the transcript row only when set, so a plan-free run's rows are byte-identical. A producer without the
    field (a test double, an extension child) planted nothing.
    """
    results = getattr(response, 'channel_results', None)
    return {CHANNEL_INPUTS_KEY: list(results)} if results else {}


def _with_choice_text(actions: list[dict[str, Any]] | None, choice_text: Mapping[str, str] | None) -> list[dict[str, Any]] | None:
    """``actions`` with the declared ``agent.io.choice_text`` copied onto every ``interrupt`` act's args.

    The fact is the HARNESS's (what its resume code does with the user's words beside each choice), so
    it rides the act the server already drives the user's choice from — the server resolves the choice
    and looks its verdict up there, never re-reading a config it cannot see. Undeclared (``None``) leaves
    the wire unchanged: the list is returned as given. An act whose args already carry a
    ``choice_text`` keeps its own; a copy is made, so the response the child handed back is untouched.
    """
    if not choice_text or not actions:
        return actions
    return [
        {**action, 'args': {**action['args'], 'choice_text': dict(choice_text)}}
        if action.get('name') == INTERRUPT_ACT and isinstance(action.get('args'), dict) and 'choice_text' not in action['args']
        else action
        for action in actions
    ]


def _paused_by(response: Any, *, choice_text: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The interrupt contract a turn ended paused on (``{prompt, choices, actions, approve?}``), else ``None``.

    Two spellings of one pause, read in the order the child reads them: the framework's own interrupt
    (``interrupt_context``), else the ``interrupt`` act the harness DECLARED on the envelope's
    ``agent_actions`` — the per-turn action the server already drives the user's choice from. On the
    act, ``approve`` counts only when it is one of its ``choices``, the rule the child applies to the
    framework's payload.

    Read defensively for the reason :func:`_reply_source_of` is: a ``child_factory`` extension may
    hand back an envelope without the interrupt fields, which is a turn that did not pause.
    """
    context = getattr(response, 'interrupt_context', None)
    if getattr(response, 'is_interrupted', False) is True and isinstance(context, dict):
        return context
    acts = [a for a in (_with_choice_text(_agent_actions_of(response), choice_text) or []) if isinstance(a, dict) and a.get('name') == INTERRUPT_ACT]
    args = acts[-1].get('args') if acts else None
    if not isinstance(args, dict):
        return None
    raw_choices = args.get('choices')
    choices: list[Any] = raw_choices if isinstance(raw_choices, list) else []
    declared = {key: args[key] for key in ('prompt', 'actions', 'choice_text') if key in args}
    return {**declared, 'choices': choices, **({'approve': args['approve']} if args.get('approve') in choices else {})}


def _user_world_of(response: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(turn_world, user_action_result)`` off an agent response; ``None`` for a producer without them.

    Read defensively for the reason :func:`_reply_source_of` is: a ``child_factory`` extension may
    hand back an envelope without both fields, which is a producer that reports neither, never an error.
    """
    turn_world, result = getattr(response, 'turn_world', None), getattr(response, 'user_action_result', None)
    return (turn_world if isinstance(turn_world, dict) else None), (result if isinstance(result, dict) else None)


def _saved_turn_world(turn_world: dict[str, Any] | None) -> dict[str, Any]:
    """``{'turn_world': ...}`` for a transcript row, or ``{}`` when there is nothing to save.

    A bench that declares no surface saves nothing, so its rows carry no ``turn_world`` key:
    for a re-score, ``not_declared`` and an absent key are one answer (no view).
    """
    if turn_world is None or turn_world.get('reason') == 'not_declared':
        return {}
    return {'turn_world': turn_world}


def _raised_turn_user_action_result(child: Any) -> dict[str, Any] | None:
    """How the user's act went on the child's last turn that raised; ``None`` for a producer without the record."""
    result = getattr(child, 'raised_turn_user_action_result', None)
    return result if isinstance(result, dict) else None


def _file_user_action_result(transcript_turns: list[dict[str, Any]], result: dict[str, Any] | None) -> None:
    """Put how the simulated user's act was applied on the transcript row of the turn she took it.

    The act rides that row's ``user_action``; its result arrives with the NEXT agent turn (the act
    is applied at its open), and carries the asking ``turn_idx``, so it is filed beside the act.
    """
    if result is None:
        return
    row = next((row for row in reversed(transcript_turns) if row.get('turn_idx') == result.get('turn_idx')), None)
    (row if row is not None else transcript_turns[-1])['user_action_result'] = result


def user_turn_content(turn: UserTurn) -> str | list[dict[str, Any]] | None:
    """This turn's USER LINE, or ``None`` when the response carries no user turn at all.

    ONE reading of the wire's user side, for the three consumers that must agree about it: the
    carrier the next agent call is taken from, the artifact row's ``output_user_message``, and
    the closing-turn decision. As three separate expressions, the first two could disagree with
    the third about whether a turn happened.

    The content itself is the same tri-state projection the opener uses — the message LIST when
    the server sent one, else the string — so a parts-carrying turn reaches the agent as parts
    and a text turn reaches it as the string.
    """
    if turn.ended_early in NO_USER_LINE_REASONS:
        return None
    return opener_content(turn.user_messages) or turn.user_message


# ---------------------------------------------------------------------------
# Finalize-poll resume: how long to wait between attempts to reach a job we
# already have an id for
# ---------------------------------------------------------------------------

#: Ceiling on the resume wait between two poll attempts lost to transport. The job is finished
#: or nearly so; a long wait after a blip would spend the scenario's remaining budget doing
#: nothing. Policy, not config — the CONFIGURED knob is the poll wallclock, which still bounds
#: the whole loop.
_FINALIZE_RESUME_BACKOFF_CAP = 30.0

#: How many doublings the resume wait may take before it flattens at the cap. Bounded so the
#: exponent cannot overflow on a long outage (``2 ** 6`` = 64 x the poll interval, already past
#: the cap for every realistic interval).
_FINALIZE_RESUME_BACKOFF_SHIFT_CAP = 6


def _finalize_resume_backoff(poll_interval: float, failures: int, remaining: float) -> float:
    """How long to wait before resuming a poll that transport lost — never past ``remaining``.

    Doubles per consecutive failure, flattens at :data:`_FINALIZE_RESUME_BACKOFF_CAP`, and is
    clamped to what is left of the poll's own wallclock so a back-off can never outlive the
    budget it is waiting inside.
    """
    doublings = min(failures - 1, _FINALIZE_RESUME_BACKOFF_SHIFT_CAP)
    return min(poll_interval * 2**doublings, _FINALIZE_RESUME_BACKOFF_CAP, remaining)


def _resume_seed_of(thread: CreateThreadResponse) -> dict[str, Any] | None:
    """The resume seed the backend put on the thread; ``None`` for a fresh scenario.

    A non-dict value reads as absent: the seed's shape is the customer's and the backend's to
    agree on, but "is there one at all" is a question this seam has to answer, and a
    scalar cannot be one.
    """
    seed = thread.resume_seed
    return seed if isinstance(seed, dict) and seed else None


#: Shared tail for both :func:`_seed_baseline` refusals — why an unreadable seed stops the run.
_SEED_REFUSAL = (
    "without a turn-0 baseline the seeded conversation would be recorded as the agent's own first turn, and the "
    'run would go GREEN having scored a conversation the agent never had. Refusing to run it.'
)


def _seed_baseline(resume_seed: dict[str, Any] | None, *, thread_id: str) -> list[dict[str, Any]]:
    """The RAW message prefix a RESUMED scenario's record starts from (``[]`` when fresh).

    Turn-0 excision reuses the prefix test of :func:`_fresh_turn_tail`. A resumed agent has the
    seeded conversation in its history, so a cumulative (MemorySaver-style) agent re-lists
    it in the chain it returns for turn 1; with an empty record :func:`_fresh_turn_tail`
    would read that whole prefix as newly produced and the seed would land in turn 1's
    per-turn payload, where judges would score history the agent never uttered as its live
    answer. Pre-populating the record with exactly the messages the seed carried makes the
    prefix-equality test strip them, and makes the longer-but-mismatched warning fire when a
    resumed agent rewrites its history. Delta agents (whose chains never re-list history)
    fall through unchanged.

    RAW is the load-bearing word: the record :func:`_fresh_turn_tail` matches against is
    ``raw_full_messages``, what the adapter actually returned, echoes and all — not the
    single-role agent channel :func:`split_user_echo` derives from it. So this baseline seeds
    that raw record. The agent channel gets the agent SIDE of the same baseline, which is a
    presentation choice; seeding only the agent channel would leave the excision itself
    broken, and a resumed run would go green having scored the seed as turn 1.

    The prefix is built through the SAME serializer, in the same package and process, that
    the per-turn path uses on the child side — ``plurai.child._serialize_message`` over
    ``convert_to_messages``, the contracted formula — so the two sides agree on shape
    wherever they can. Byte-stability alone is NOT what the excision rests on, though: the
    agent's checkpointer round trip enriches the seeded messages with metadata defaults,
    which is why ``_fresh_turn_tail`` compares on the
    :class:`_MessageIdentity` projection rather than full dicts.

    Fails CLOSED, for the reason in :data:`_SEED_REFUSAL`; raising reaches
    ``run_simulation``'s failure path, which writes ``error.json`` and calls ``/fail``. The
    imports stay INSIDE the ``try`` so a missing ``langchain_core`` refuses the run too
    rather than escaping raw, and function-local because ``plurai.child`` imports
    ``plurai``, which imports this module. A seed carrying no ``messages`` is not a failure
    at all: an ``agent_state``-only seed is a legitimate shape, its baseline is genuinely
    empty, and the child still receives the seed in full.
    """
    messages = (resume_seed or {}).get('messages')
    if not messages:
        return []
    if not isinstance(messages, list):
        raise OrchestratorAdapterError(
            f'resume_seed.messages on thread {thread_id} is {type(messages).__name__}, not a list, so this '
            f'trace-resumed scenario cannot be given one — {_SEED_REFUSAL}',
            thread_id=thread_id,
        )
    try:
        from langchain_core.messages import convert_to_messages  # noqa: PLC0415

        from plurai.child import _serialize_message  # noqa: PLC0415

        return [_serialize_message(m) for m in convert_to_messages(messages)]
    except Exception as exc:
        raise OrchestratorAdapterError(
            f'Could not build the turn-0 baseline from resume_seed.messages on thread {thread_id} ({type(exc).__name__}: {exc}) — {_SEED_REFUSAL}',
            thread_id=thread_id,
        ) from exc


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec='milliseconds')


# ---------------------------------------------------------------------------
# The bench's SHAPE: one envelope (pipeline) or a conversation (dialog).
# ---------------------------------------------------------------------------

#: The closed set of bench shapes, as a type like ``egress.ProbeMode``. The value is compared
#: by literal at several sites, and the backend reads it back out of
#: ``transcript.json`` (``scenario_shape == 'pipeline'``). The
#: RUNTIME checks stay: the value also arrives from YAML and from argv, where no type reaches.
Shape = Literal['dialog', 'pipeline']
#: A conversation: the /user-turn loop, the shape of a bench that
#: declares nothing. The default at every rung.
DIALOG_SHAPE: Shape = 'dialog'
#: ONE structured call: ``invoke_pipeline(inputs) -> dict``, no user simulator, no turns. The
#: agent shapes the plugin calls "first-class, NOT a degraded case" — a classification head, a
#: summariser, a structured-output extractor, a document/vision extractor.
PIPELINE_SHAPE: Shape = 'pipeline'
#: The bench's own root-key declaration in ``plurai.config.yaml``. A structural self-description,
#: beside ``agent.framework`` / ``tools`` / ``render_actions`` — NOT a machine-local posture, so
#: it is the committed config and never a ``.run_mode``-style sidecar (those are documented as
#: "machine-local … it describes one workspace"; a shape is the same in every clone).
SHAPE_KEY = 'shape'

# ---------------------------------------------------------------------------
# The bench's WORLD: who owns the state this suite scores.
# ---------------------------------------------------------------------------

#: The other root-key declaration in ``plurai.config.yaml``, beside ``shape:`` and read by the
#: same discovery. ``world_state: {owner: …}`` — a structural self-description of WHERE the state
#: a scenario mutates actually lives, which is not something a run may guess: the whole of what
#: the backend resets between scenarios is the mock session it opened, so a world kept anywhere else
#: is carried forward unless the harness itself clears it.
WORLD_STATE_KEY = 'world_state'
WORLD_OWNER_KEY = 'owner'
#: The deployed mock owns every record this agent reads and writes. The DEFAULT, and what an
#: undeclared bench means: the backend opens a session per scenario and seeds it, so there is nothing
#: for the harness to reset and no hook is required.
WORLD_OWNER_MOCK = 'mock'
#: The AGENT owns it — a store inside the customer's own process or beside it (its checkpointer's
#: database, a file, a local index). Reachable from the harness, and so resettable by it.
WORLD_OWNER_AGENT = 'agent'
#: An EXTERNAL system owns it — a real CRM, a shared database, a server-side conversation store.
#: the backend cannot see it and cannot reset it; only the harness knows where it is.
WORLD_OWNER_EXTERNAL = 'external'
#: The closed set. ``agent`` and ``external`` differ in WHERE the store is, not in what this SDK
#: must do about it — both are worlds the mock does not own — so they share their two
#: requirements (a ``reset`` hook to CLEAR it, a ``world_snapshot`` hook to OBSERVE it) while
#: staying two words, because the remedy an operator reaches for is different and a report that
#: called a customer's CRM "the agent's state" would be wrong.
WORLD_OWNERS = (WORLD_OWNER_MOCK, WORLD_OWNER_AGENT, WORLD_OWNER_EXTERNAL)
#: ``world_state.slots: N`` — an EXTERNAL or AGENT-owned world the bench runs as N identical copies
#: (N stacks, N store paths), so the runner may take up to N attempts at once, each leasing one copy
#: for its child's whole life. ``slots: 1`` is one shared world, run one attempt at a time. Absent, an
#: external world is still run one at a time, while an agent-owned one keeps full ``--concurrency``
#: (a store each child holds in its own process needs no copies). Illegal beside ``owner: mock``:
#: a mock-owned world gets a fresh session per scenario, so the key would be ignored.
WORLD_SLOTS_KEY = 'slots'
#: The environment variable the child of a slotted attempt is spawned with: the index (``0..N-1``) of
#: the world copy it leased. The harness indexes its slot table by it, defaulting to ``0``.
WORLD_SLOT_ENV = 'PLURAI_WORLD_SLOT'
#: ``world_state.shared_world_ok: true`` — the operator's EXPLICIT opt-in to live writes on a world no
#: ``reset`` returns to the same state before each conversation. The scan mocks every
#: write surface by default, so this key is written only when the user chose to keep writes live. With
#: it the run is NOT held at :func:`unreset_external_world_finding`: it proceeds and carries the named
#: risk :func:`shared_world_risk` returns. A strict ``bool``; anything else is refused. ``false`` and
#: absent mean the same thing — no opt-in, the hold stands.
WORLD_SHARED_OK_KEY = 'shared_world_ok'
#: The named-risk ``kind`` a run on an opted-in shared world carries, and the reason its stability
#: retries and trial agreement are marked not comparable (``stability_comparable: false``).
SHARED_WORLD_RISK_KIND = 'shared_world'
#: The hook a non-mock-owned world must export: ``def reset() -> None``, called in the child
#: before every scenario (``plurai.child._ChildSession.handle_init``).
RESET_HOOK = 'reset'
#: The hook that makes a world OBSERVABLE: ``def world_snapshot() -> dict[str, Any]``, called in
#: the child at finalize (``plurai.child._ChildSession.handle_finalize``) and returned VERBATIM as
#: this scenario's ``final_world_state.state`` — the ``{entity_id: {...}}`` map
#: ``plurai.get_full_state`` puts under that key, which is the shape the backend's ``entities_of_type``
#: / ``find_entity`` read.
#:
#: It is the ONLY way a world the mock does not own becomes observable. Nothing else in this SDK
#: can reach such a store: the child snapshots the MOCK's session, so a bench whose world lives in
#: its own process, a file or a real system behind it reports the mock's session as if it were the
#: world it declared — an empty one scores ``len(entities_of_type('x')) == 0`` as a CONCLUSIVE PASS
#: about a store nobody read. Export this hook, or the world channel says ``None`` and the backend
#: demotes every world-reading policy to ``world_unobservable`` instead.
#:
#: DISPATCH IS ON THE HOOK, never on the owner: a bench may declare ``agent`` and still keep part
#: of its world in the mock (``world_state.owner`` is a declaration about RESETTABILITY, not about
#: observability), and suppressing the mock read on the declaration alone would throw that world
#: away. The owner decides only what the hook's ABSENCE means — see
#: :func:`unobservable_world_warning` and ``child._ChildSession.handle_finalize``.
#:
#: ``world_snapshot``, not ``snapshot``: both readers of this name are blind to intent — the child
#: ``getattr``\ s it off the imported module and the parent walks module-scope BINDINGS — so a
#: customer variable that happens to be called ``snapshot`` would be called as a hook.
WORLD_SNAPSHOT_HOOK = 'world_snapshot'

#: ``(has_invoke_agent, has_invoke_pipeline)`` per resolved ``simulate.py``, so a batch of N
#: scenarios parses the file once.
_INVOKE_HOOK_CACHE: dict[Path, tuple[bool, bool]] = {}
#: Harnesses the probe could not READ (missing, unreadable, unparseable), mapped to the NAME OF
#: THE FAILURE CLASS that stopped it. They answer ``(True, True)`` above so dispatch falls through
#: to dialog, which is the safe route — but "could not look" is not "exports both", and only the
#: second is the ambiguity worth warning a customer about. Without this distinction every
#: directly-constructed Orchestrator whose ``simulate_path`` is not a real file on THIS machine
#: earns a warning naming two hooks that nobody wrote.
#:
#: The VALUE is what :func:`unreset_external_world_finding` names in its read-failure refusal.
#: It is recorded HERE, in the one place the exception was actually caught, rather than
#: re-derived by a second read at the refusal — a second read is a second answer to one question.
_UNREADABLE_HARNESSES: dict[Path, str] = {}


def _detect_invoke_hooks(simulate_path: Path) -> tuple[bool, bool]:
    """``(has_invoke_agent, has_invoke_pipeline)`` for ``simulate.py``, read by AST.

    A HINT that names the key to add; NEVER a route. It is consulted only for a bench that
    declared no ``shape:``, and even there it decides nothing: :func:`resolve_scenario_shape`
    returns ``DIALOG_SHAPE`` unconditionally past the declaration rung, and the one gate that
    acts on this answer (:func:`undeclared_pipeline_refusal`) STOPS a run rather than routing
    it.

    The module-scope binding set it reads — AST, never import, and wider than a ``FunctionDef``
    walk — is :func:`_module_scope_bindings_of_file`; see it for both reasons. What this function adds is
    the two hooks' NAMES, the per-path cache (a batch of N scenarios parses the file once) and
    the unreadable answer: a harness the parser could not read reports BOTH hooks present, which
    resolves to dialog — the safe answer, because the child's own
    import error then surfaces with a real traceback instead of the parent guessing a route off
    unreadable source.
    """
    key = Path(simulate_path).expanduser().resolve()
    cached = _INVOKE_HOOK_CACHE.get(key)
    if cached is not None:
        return cached
    bound = _module_scope_bindings_of_file(key)
    if bound is None:
        _INVOKE_HOOK_CACHE[key] = (True, True)
        return (True, True)
    result = ('invoke_agent' in bound, 'invoke_pipeline' in bound)
    _INVOKE_HOOK_CACHE[key] = result
    return result


def _module_scope_bindings_of_file(simulate_path: Path) -> frozenset[str] | None:
    """Every name a ``simulate.py`` binds at MODULE SCOPE, or ``None`` when it could not be read.

    The one AST reading of a harness, so the questions asked of it — which invoke hook it exports
    (:func:`_detect_invoke_hooks`), whether it exports ``reset``
    (:func:`unreset_external_world_finding`) — are answered off ONE parse with one set of rules.
    Two walks would be two implementations of "does this file export X", and where they disagreed
    one gate would fire on a harness the other cleared.

    AST, never import: importing the customer's module in the PARENT process would run its
    module-level side effects (LLM client construction, env reads, network calls) in a process
    that is not the child and may not even share its interpreter.

    Deliberately WIDER than a ``FunctionDef`` walk — an assignment, an annotated assignment and
    an ``from … import reset`` all bind the name — because the CHILD resolves every hook with
    ``getattr`` (``child.py``) and accepts all of them.

    NOT cached: its per-RUN callers are the two world gates
    (:func:`unreset_external_world_finding` and :func:`unobservable_world_warning`), which parse
    the file at most twice per run and only for a non-mock declaration, plus
    :func:`shared_world_risk`, which parses it only beside an explicit ``shared_world_ok`` opt-in; its per-SCENARIO
    caller (:func:`_detect_invoke_hooks`) owns the cache — and its invalidation
    (:func:`reset_warning_ledgers`) — already. A second cache would be a second thing to
    invalidate, and the one that was forgotten would answer from a harness that has since changed
    on disk.

    DECODED THE WAY THE CHILD DECODES IT, for the same reason the walk is wider than a
    ``FunctionDef`` one: ``importlib.util.decode_source`` is the function the child's own
    ``SourceLoader.get_source`` calls (``child._load_simulate_module`` →
    ``spec_from_file_location`` → ``exec_module``), so it honours a UTF-8 BOM and a PEP 263
    ``# -*- coding: … -*-`` cookie exactly as the child does. A hard-coded
    ``read_text(encoding='utf-8')`` would not: on either of those a file the child imports
    perfectly would reach ``None`` here — through ``SyntaxError`` on the BOM and
    ``UnicodeDecodeError`` on the cookie — and the gates that read this would fall open on a world
    nobody resets. Whoever moves this line must move the child's loader with it.

    Returns ``None`` — never an empty set — when the file is missing, unreadable, or unparseable
    BY THIS INTERPRETER, because "could not look" and "looked and found nothing" are different
    answers. What a caller does about the first is the CALLER's to decide, and it differs:
    :func:`_detect_invoke_hooks` is guessing a route, which a refusal cannot improve, so
    it fails open; :func:`unreset_external_world_finding` is proving a HOOK beside a declared world
    the backend cannot reset, and an unprovable one there is refused.
    """
    key = Path(simulate_path).expanduser().resolve()
    try:
        tree = ast.parse(importlib.util.decode_source(key.read_bytes()), filename=str(key))
    except (OSError, SyntaxError, ValueError) as exc:
        # "Could not look" earns the same volume as "looked and found both". Silence here would
        # compound: the same unreadability would then suppress the only line naming the route.
        #
        # ONCE PER HARNESS PER REASON PER RUN, off the ledger this arm already writes. The walk is
        # uncached on purpose (see above) and ONE run start makes several — the shape gate, the
        # world-reset gate, the turn-input gate and the ``run_summary.json`` observability record —
        # so one unparseable ``simulate.py`` would otherwise print the identical sentence several
        # times. Keyed on the reason too, so a harness whose failure MODE changes mid-run still
        # reports the new one; and keyed on the RUN-scoped ledger rather than a process-level
        # ``_warned`` set, so the second run in one interpreter says it again (what
        # :func:`reset_warning_ledgers` exists for).
        if _UNREADABLE_HARNESSES.get(key) != type(exc).__name__:
            logging.getLogger(__name__).warning(
                'could not read %s to see which hooks it exports (%s); the shape gate falls back to the declaration alone and refuses nothing, '
                'and the world-reset gate refuses the run if a `%s:` beside it declares a world the Plurai backend cannot reset.',
                key,
                type(exc).__name__,
                WORLD_STATE_KEY,
            )
        _UNREADABLE_HARNESSES[key] = type(exc).__name__
        return None
    return frozenset(_module_scope_bindings(tree))


def _declared_workspace_value(simulate_path: Path, *, keys: tuple[str, ...], noun: str, legal: tuple[str, ...]) -> tuple[str, Path] | None:
    """The value a ``plurai.config.yaml`` beside this harness declares at ``keys``, or ``None``.

    ONE reader for every env-scoped declaration this module consults — the ``shape:`` a bench
    dispatches on and the ``world_state.owner`` it is reset by — because the three rules that
    make such a declaration trustworthy are the same rules whichever key is being read, and the
    second copy of them is where they drift:

    * **WHERE.** The two rungs every workspace declaration is read at — the harness directory and
      the repo root above it — through the SAME ``discover_config_documents`` that ``anonymize:``
      and ``render_actions:`` use, so a workspace has one layout to learn and a config that
      EXISTS and cannot testify raises rather than silently defaulting.
    * **AN UNKNOWN VALUE IS A REFUSAL**, never a silent default: a typo'd ``shape: pipline`` or
      ``owner: externl`` that fell back to the default would reproduce exactly the failure the
      declaration exists to end.
    * **COLLECT AND REFUSE ON DISAGREEMENT**, as ``discover_render_surface`` / ``resolve_posture``
      / ``discover_closed_class_overrides`` do: the declaration is env-scoped, and a discovery
      ORDER does not get to settle which answer the env meant.

    ``keys`` walks into nested mappings (``('world_state', 'owner')``), so a declaration can be
    grouped under a root key without needing its own reader. An intermediate that exists and is
    not a mapping is a config that cannot testify, and raises like any other.
    """
    label = '.'.join(keys)
    # Root first, then ``plurai_sim/`` — the workspace order ``workspace_config.py:14-18``
    # documents. With collect-and-refuse below, the ORDER decides nothing: two
    # declarations that disagree are refused whichever is seen first. It is kept so the reported
    # order matches the documented one, and so a future single-hit reader inherits the right
    # precedence rather than the accident of which directory was walked first.
    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in bench_config_roots(simulate_path):
        for path, document in discover_config_documents(root):
            if (resolved := path.resolve()) in seen:
                continue
            seen.add(resolved)
            raw: Any = document
            for key in keys:
                if raw is None:
                    break
                if not isinstance(raw, dict):
                    raise WorkspaceConfigError(f'{path}: `{label}` cannot be read — `{key}` is nested under a {type(raw).__name__}, not a mapping.')
                raw = raw.get(key)
            if raw is None:
                continue
            declared = str(raw).strip().lower()
            if declared not in legal:
                raise WorkspaceConfigError(f'{path}: `{label}: {raw!r}` is not {noun}; it must be {" or ".join(repr(value) for value in legal)}.')
            found.append((declared, path))
    if not found:
        return None
    if len({declared for declared, _ in found}) > 1:
        conflict = ', '.join(f'{path}: `{label}: {declared}`' for declared, path in found)
        raise WorkspaceConfigError(f'{label} is declared more than once and the declarations disagree ({conflict}). Remove one, or make them agree.')
    return found[0]


def _declared_shape(simulate_path: Path) -> tuple[Shape, Path] | None:
    """The ``shape:`` a ``plurai.config.yaml`` beside this harness declares, or ``None``.

    Where it is read, what an unknown value does and what two disagreeing files do are
    :func:`_declared_workspace_value`'s, shared with the world-owner declaration below.

    The ``cast`` states what ``legal=`` already ENFORCES one line down — the shared reader
    raises on any other value — rather than re-checking it here, which would be a second answer
    to the question that reader owns. Same shape as ``run_simulation._observability_posture``.
    """
    declared = _declared_workspace_value(simulate_path, keys=(SHAPE_KEY,), noun='a shape', legal=(DIALOG_SHAPE, PIPELINE_SHAPE))
    return None if declared is None else (cast('Shape', declared[0]), declared[1])


def _declared_world_owner(simulate_path: Path) -> tuple[str, Path] | None:
    """The ``world_state.owner`` a ``plurai.config.yaml`` beside this harness declares, or ``None``.

    ``None`` means the bench declared nothing, which means :data:`WORLD_OWNER_MOCK`. A
    declaration that says nothing costs nothing: every reader below treats ``None`` and ``mock``
    as one answer, and every one of them is silent on it.

    THREE readers, and each acts on the SAME word for a different reason — which is why the word
    is resolved here once rather than re-derived per gate:

    * :func:`unreset_external_world_finding` — a world nothing can CLEAR, held at run start;
    * :func:`unobservable_world_warning` — a world nothing can OBSERVE, warned about at run start;
    * ``child._ChildSession.mock_owns_the_world`` — the same read, in the child, deciding what
      the ABSENCE of :data:`WORLD_SNAPSHOT_HOOK` means for the world channel. It reads the config
      beside the harness exactly as this does rather than being told on ``Init``, so the two
      answers cannot disagree and the IPC envelope does not grow a field.

    None of them reads the word as a statement about OBSERVABILITY on its own: see
    :data:`WORLD_SNAPSHOT_HOOK` for why the dispatch is on the hook.
    """
    return _declared_workspace_value(simulate_path, keys=(WORLD_STATE_KEY, WORLD_OWNER_KEY), noun='a world owner', legal=WORLD_OWNERS)


def _declared_world_slots(simulate_path: Path) -> int | None:
    """The ``world_state.slots`` a ``plurai.config.yaml`` beside this harness declares, or ``None``.

    Read over the same two rungs as :func:`_declared_workspace_value`, with its collect-and-refuse
    rule; a separate reader only because that one's string ``legal`` tuple cannot express an integer.
    Every refusal is a :class:`WorkspaceConfigError` (exit 2 at run start, before any spend):

    * a value that is not a positive int (``bool`` included, ``"3"`` included) — never coerced;
    * two configs that declare different counts;
    * ``slots`` beside an owner other than ``external`` or ``agent`` (undeclared included): a
      mock-owned world gets a fresh session per scenario, so the key would be silently ignored;
    * ``slots >= 2`` while no ``.py`` file directly beside ``simulate.py`` names
      :data:`WORLD_SLOT_ENV` (:func:`_names_world_slot`) — every attempt would reach copy 0 and the
      parallel attempts would share one world.

    ``slots: 1`` is accepted. Beside ``external`` it means exactly what an undeclared count means
    (one attempt at a time); beside ``agent`` it is the only way to serialize the run, since an
    undeclared agent-owned world keeps full ``--concurrency``.
    """
    label = f'{WORLD_STATE_KEY}.{WORLD_SLOTS_KEY}'
    found: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for root in bench_config_roots(simulate_path):
        for path, document in discover_config_documents(root):
            if (resolved := path.resolve()) in seen:
                continue
            seen.add(resolved)
            world = document.get(WORLD_STATE_KEY)
            if world is None:
                continue
            if not isinstance(world, dict):
                raise WorkspaceConfigError(f'{path}: `{label}` cannot be read — `{WORLD_STATE_KEY}` is a {type(world).__name__}, not a mapping.')
            raw = world.get(WORLD_SLOTS_KEY)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
                raise WorkspaceConfigError(f'{path}: `{label}: {raw!r}` is not a slot count; it must be a whole number of world copies, 1 or more.')
            found.append((raw, path))
    if not found:
        return None
    if len({slots for slots, _ in found}) > 1:
        conflict = ', '.join(f'{path}: `{label}: {slots}`' for slots, path in found)
        raise WorkspaceConfigError(f'{label} is declared more than once and the declarations disagree ({conflict}). Remove one, or make them agree.')
    slots, config_path = found[0]
    owner = _declared_world_owner(simulate_path)
    if owner is None or owner[0] not in (WORLD_OWNER_EXTERNAL, WORLD_OWNER_AGENT):
        said = 'declares no owner' if owner is None else f'declares `{WORLD_STATE_KEY}.{WORLD_OWNER_KEY}: {owner[0]}`'
        raise WorkspaceConfigError(
            f'{config_path}: `{label}: {slots}` applies only to `{WORLD_STATE_KEY}.{WORLD_OWNER_KEY}: {WORLD_OWNER_EXTERNAL}` or '
            f'`{WORLD_STATE_KEY}.{WORLD_OWNER_KEY}: {WORLD_OWNER_AGENT}`, but the bench {said} — a mock-owned world gets a fresh session per '
            f'scenario. Remove `{WORLD_SLOTS_KEY}`, or declare the owner {WORLD_OWNER_EXTERNAL} or {WORLD_OWNER_AGENT}.'
        )
    if slots >= 2 and not _names_world_slot(simulate_path):
        raise WorkspaceConfigError(
            f'{config_path}: `{label}: {slots}` declares {slots} world copies, but {simulate_path} never names `{WORLD_SLOT_ENV}` '
            "in a .py file beside it, so every attempt would reach copy 0 and the parallel attempts would share one world. Index the harness's "
            f"per-copy settings by `int(os.environ.get('{WORLD_SLOT_ENV}', '0'))`, or remove `{WORLD_SLOTS_KEY}` to run one attempt at a time."
        )
    return slots


#: ``harness.single_instance``: the resources only ONE agent process can hold at a time (a fixed bound
#: port, a lock file, a device), as the scan recorded them — each a short cite, e.g. ``tcp port 8123``.
HARNESS_KEY: Final = 'harness'
SINGLE_INSTANCE_KEY: Final = 'single_instance'


def _declared_single_instance(simulate_path: Path) -> list[str]:
    """The ``harness.single_instance`` resources a ``plurai.config.yaml`` beside this harness declares, unioned in
    declaration order; ``[]`` when none does. Read over the same two rungs as :func:`_declared_world_slots`; a value
    that is not a list of non-empty strings raises :class:`WorkspaceConfigError` (exit 2 before any spend), never
    coerced — a typo here decides whether two agent processes fight over one resource."""
    label = f'{HARNESS_KEY}.{SINGLE_INSTANCE_KEY}'
    found: dict[str, None] = {}
    seen: set[Path] = set()
    for root in bench_config_roots(simulate_path):
        for path, document in discover_config_documents(root):
            if (resolved := path.resolve()) in seen:
                continue
            seen.add(resolved)
            harness = document.get(HARNESS_KEY)
            raw = harness.get(SINGLE_INSTANCE_KEY) if isinstance(harness, dict) else None
            if raw is None:
                continue
            if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
                raise WorkspaceConfigError(f'{path}: `{label}: {raw!r}` must be a list of the resources one agent process holds, e.g. ["tcp port 8123"].')
            found.update(dict.fromkeys(item.strip() for item in raw))
    return list(found)


def _declared_shared_world_ok(simulate_path: Path) -> bool:
    """Whether a ``plurai.config.yaml`` beside this harness declares ``world_state.shared_world_ok: true``.

    Read over the same two rungs, with the same collect-and-refuse rule, as :func:`_declared_world_slots`;
    a separate reader only because the string ``legal`` tuple of :func:`_declared_workspace_value` cannot
    express a ``bool``. A value that is not a ``bool`` (``"yes"``, ``1``) and two configs that disagree
    raise :class:`WorkspaceConfigError` — never coerced, because a typo here decides whether a run on a
    shared world is held."""
    label = f'{WORLD_STATE_KEY}.{WORLD_SHARED_OK_KEY}'
    found: list[tuple[bool, Path]] = []
    seen: set[Path] = set()
    for root in bench_config_roots(simulate_path):
        for path, document in discover_config_documents(root):
            if (resolved := path.resolve()) in seen:
                continue
            seen.add(resolved)
            world = document.get(WORLD_STATE_KEY)
            if world is None:
                continue
            if not isinstance(world, dict):
                raise WorkspaceConfigError(f'{path}: `{label}` cannot be read — `{WORLD_STATE_KEY}` is a {type(world).__name__}, not a mapping.')
            raw = world.get(WORLD_SHARED_OK_KEY)
            if raw is None:
                continue
            if not isinstance(raw, bool):
                raise WorkspaceConfigError(f'{path}: `{label}: {raw!r}` is not an opt-in; it must be true or false.')
            found.append((raw, path))
    if len({ok for ok, _ in found}) > 1:
        conflict = ', '.join(f'{path}: `{label}: {str(ok).lower()}`' for ok, path in found)
        raise WorkspaceConfigError(f'{label} is declared more than once and the declarations disagree ({conflict}). Remove one, or make them agree.')
    return bool(found) and found[0][0]


def _declared_live_writers(simulate_path: Path) -> tuple[str, ...]:
    """The ``tools.<name>`` beside this harness with ``mode: real`` and ``descriptor.write_strategy: real``:
    the tools that write the customer's live system. The same predicate as the plugin's
    ``verify_simulate._declared_live_writers``, so the runner and the verifier count the same writers."""
    names: set[str] = set()
    for root in bench_config_roots(simulate_path):
        for _path, document in discover_config_documents(root):
            tools = document.get('tools')
            if not isinstance(tools, dict):
                continue
            for name, entry in tools.items():
                descriptor = entry.get('descriptor') if isinstance(entry, dict) else None
                if isinstance(entry, dict) and entry.get('mode') == 'real' and isinstance(descriptor, dict) and descriptor.get('write_strategy') == 'real':
                    names.add(str(name))
    return tuple(sorted(names))


def shared_world_risk(simulate_path: Path | str) -> dict[str, Any] | None:
    """The ``shared_world`` named risk a run carries, or ``None``.

    Carried when the operator EXPLICITLY opted in (``world_state.shared_world_ok: true``) AND no
    module-scope ``reset`` is proven AND the world is one conversations can share: a non-mock owner,
    or at least one live writing tool (:func:`_declared_live_writers`) whatever the owner says. Each
    conversation after the first may then run on its predecessors' writes, so the verdicts may depend
    on run order: the suite caps at qualified (the kind is not rung-neutral) and stability retries /
    trial agreement are not comparable. A harness this interpreter could not read proves no ``reset``,
    so it carries the risk too. Without the opt-in this is ``None`` and :func:`unreset_external_world_finding`
    holds.

    The entry carries the keys a ``release_gate.named_risks[]`` entry carries (``policy``,
    ``display_name``, ``kind``, ``verdict``, ``catastrophic``, ``detail``) plus the typed facts behind it.
    """
    path = Path(simulate_path)
    if not _declared_shared_world_ok(path):
        return None
    bound = _module_scope_bindings_of_file(path)
    if bound is not None and RESET_HOOK in bound:
        return None
    declared = _declared_world_owner(path)
    owner = declared[0] if declared is not None else WORLD_OWNER_MOCK
    writers = _declared_live_writers(path)
    if owner == WORLD_OWNER_MOCK and not writers:
        return None
    text = f'Conversations share one world: {len(writers)} live writing tools with no reset between conversations; verdicts may depend on run order.'
    return {
        'policy': None,
        'display_name': text,
        'kind': SHARED_WORLD_RISK_KIND,
        'verdict': None,
        'catastrophic': False,
        'detail': text,
        'live_writers': list(writers),
        'owner': owner,
        'reset_bound': None if bound is None else False,
    }


def _names_world_slot(simulate_path: Path) -> bool:
    """Whether any ``.py`` file DIRECTLY beside the harness names :data:`WORLD_SLOT_ENV` — a tripwire,
    not a guard: the read may live in a helper module the harness imports (``_storage.py``), and
    subdirectories (``probes/``, ``artifacts/``, vendored trees) are never read. Unreadable files are skipped."""
    needle = WORLD_SLOT_ENV.encode()
    for path in Path(simulate_path).expanduser().parent.glob('*.py'):
        try:
            if needle in path.read_bytes():
                return True
        except OSError:
            continue
    return False


def _declared_native_tools(simulate_path: Path) -> list[str]:
    """``agent.native_tools`` beside this harness: tools the agent runs in its own runtime, which no
    plurai lane wraps, so the mock log cannot see them. Empty when nothing is declared."""
    names: list[str] = []
    for root in bench_config_roots(simulate_path):
        for _path, document in discover_config_documents(root):
            agent = document.get('agent')
            declared = agent.get('native_tools') if isinstance(agent, dict) else None
            if isinstance(declared, list):
                names += [str(name) for name in declared if name]
    return list(dict.fromkeys(names))


def _unread_if_native_unobserved(tool_events: dict[str, Any] | None, simulate_path: Path, *, trail_declared: bool, thread_id: str) -> dict[str, Any] | None:
    """``None`` when declared native tools went unwatched: no turn declared the trail lane (a
    ``tool_trail`` beside the host-effects record that says ``plurai.confine`` recorded it,
    :func:`plurai.host_effects.trail_recorded` — a returned list alone is not a recording). Sends the
    unread channel, never "invoked zero tools". Mock events do not lift it: they observe the mocked
    tools, not the native ones, and one ``None`` is the only way the wire says part of the channel
    went unread (a per-lane marker is the backend's to read first)."""
    native_tools = _declared_native_tools(simulate_path)
    if not native_tools or trail_declared or not isinstance(tool_events, dict):
        return tool_events
    logging.getLogger(__name__).warning(
        'thread %s: agent.native_tools %s run outside every plurai lane and no turn returned a tool_trail plurai.confine '
        "recorded, so the tool channel is reported unread (tool-trail checks abstain). Pass the thread's trail to "
        'plurai.confine(..., trail=...), return its per-turn slice as tool_trail and its record as host_effects to grade them.',
        thread_id,
        native_tools,
    )
    return None


def resolve_scenario_shape(simulate_path: Path | str, *, override: str | None = None) -> Shape:
    """``'dialog'`` or ``'pipeline'`` for this harness — DECLARED first, derived second.

    The precedence, matching the SDK's own house rule for every other declaration
    (``run_simulation._default_agent_version``: *"DECLARED first (plurai.config.yaml), derived
    second"*)::

        --shape pipeline|dialog        operator override, for debugging — read FIRST
          > plurai.config.yaml  shape:  the bench's own DECLARATION — and the only other rung
          > dialog                      the default

    TWO RUNGS DISPATCH, not three: a declaration, or the operator's override of one. The AST
    probe of ``simulate.py`` is a HINT — it warns when a harness exports ``invoke_pipeline`` and
    nothing declares a shape, naming the key to add — and it never chooses a route. It reads every
    name the executed module would expose at MODULE SCOPE — including one bound inside an ``if`` /
    ``try`` / ``with`` / ``for`` that runs at import, never one bound inside a function body, the
    same line the child's own ``getattr`` draws — so it answers what a file EXPORTS and still not
    what its author MEANT by exporting it: a bench exporting BOTH hooks holds a conversation and a
    one-shot hop equally well, and a route derived from that is a guess with no rung above it.

    Every declaration is COLLECTED and a disagreement is REFUSED (``_declared_shape``), the way
    ``discover_render_surface`` / ``resolve_posture`` / ``discover_closed_class_overrides`` do:
    the declaration is env-scoped, and a discovery order does not get to settle which one the env
    meant. Resolved in ``Orchestrator.__init__``, BEFORE any thread exists — not inside the
    per-scenario try. The CLI constructs one orchestrator per scenario (and again per retry), so
    this is not once per RUN; what it buys is that a bad declaration is refused before a thread,
    an opener and an error.json are minted for it.
    """
    path = Path(simulate_path)
    log = logging.getLogger(__name__)
    # The OVERRIDE is consulted FIRST, before any config is read. ``_declared_shape`` RAISES on a malformed
    # declaration, and the one escape hatch an operator has must be able to rescue a run from the
    # very config that broke it.
    if override is not None:
        if override not in (DIALOG_SHAPE, PIPELINE_SHAPE):
            raise WorkspaceConfigError(f'`{override!r}` is not a shape; it must be {DIALOG_SHAPE!r} or {PIPELINE_SHAPE!r}.')
        # Said out loud every time: the override silences the bench's own declaration, and a run
        # dispatched against the operator's word rather than the workspace's should never be a
        # thing someone has to reconstruct from a shell history.
        log.warning('--%s %s is in force, so this run ignores any `%s:` the bench declares.', SHAPE_KEY, override, SHAPE_KEY)
        return override
    if (declared := _declared_shape(path)) is not None:
        return declared[0]
    # THE DECLARATION ALONE DISPATCHES; the AST probe is a HINT and never a route. It reads every
    # name the executed module would expose at module scope — one bound inside an ``if`` / ``try``
    # / ``with`` / ``for`` at module level included, one bound inside a function body never — so it
    # answers what the file EXPORTS and still not what its author MEANT by it, which is the whole
    # reason it does not route. Naming the key is the whole job here.
    has_agent, has_pipeline = _detect_invoke_hooks(path)
    if has_pipeline and path.expanduser().resolve() not in _UNREADABLE_HARNESSES:
        # THE CONSEQUENCE DEPENDS ON THE OTHER HOOK, and the two are opposites — so it is said
        # per case rather than asserted once and wrong half the time. A bench that ALSO exports
        # ``invoke_agent`` holds a complete conversation, and only its pipeline-scoped policies
        # go unscored: zero verdicts, clean exit code, the quiet failure this warning exists
        # to surface. A bench that exports ``invoke_pipeline`` ALONE has no hook for the turn loop to
        # drive and dies on turn 1 in the child ("Received user_turn but no invoke_agent is
        # configured") — loud, not quiet. The CLI refuses that second case at run start
        # (:func:`undeclared_pipeline_refusal`), so this branch reaches it only for an embedder
        # constructing an Orchestrator directly.
        log.warning(
            '%s exports invoke_pipeline but no %s declares a root `%s:` key, so this run is dispatched as %s. Add `%s: %s` to %s to '
            'dispatch it as a one-shot hop — %s.',
            path,
            CONFIG_FILENAME,
            SHAPE_KEY,
            DIALOG_SHAPE,
            SHAPE_KEY,
            PIPELINE_SHAPE,
            CONFIG_FILENAME,
            'this bench also exports invoke_agent, so the dialog itself runs to completion and every pipeline-scoped policy is scored '
            'against an empty `pipeline_outputs`: zero verdicts with a clean exit code'
            if has_agent
            else 'this bench exports no invoke_agent, so the turn loop has no hook to drive and every scenario fails on turn 1',
        )
    return DIALOG_SHAPE


def undeclared_pipeline_refusal(simulate_path: Path | str, *, override: str | None = None) -> str | None:
    """Why this run must not start, or ``None`` — the $0 preflight for an UNDECLARED pipeline bench.

    Refused on all four conjuncts, because each one owns a case that must NOT be refused: an
    ``--shape`` override is the operator's escape hatch and is read first everywhere else
    (:func:`resolve_scenario_shape`); a DECLARED shape dispatches on its own word (a declared
    ``dialog`` over a pipeline harness is a different defect, owned by the config verifier); a
    bench exporting BOTH hooks can hold a conversation, so dialog is a defensible default there
    and refusing it would be a false red; and a harness the probe could not READ answers
    ``(True, True)`` — "could not look" must never become a refusal. That last conjunct is
    belt-and-braces against the first, and it stays because the two facts are independent: if the
    unreadable fallback answer ever moves, silence must still not turn into a refusal.

    HOOKS BEFORE THE CONFIG READ, so a bench that is not pipeline-only never reaches
    ``_declared_shape`` here and a malformed ``shape:`` on any other bench keeps raising in
    ``Orchestrator.__init__``.

    The message is the whole point. Without it such a bench would be dispatched into the turn
    loop and die on turn 1 in the child ("Received user_turn but no invoke_agent is configured") — naming a
    hook its author never wrote and cannot add — once per scenario, AFTER the scenario set has
    been generated and one ``POST /threads`` per scenario has been paid for. The probe does not
    ROUTE the bench here either; it stops it, and names the one line that would route it.
    """
    if override is not None:
        return None
    path = Path(simulate_path)
    if _detect_invoke_hooks(path) != (False, True) or path.expanduser().resolve() in _UNREADABLE_HARNESSES:
        return None
    if _declared_shape(path) is not None:
        return None
    return (
        f'{path} exports invoke_pipeline and no invoke_agent, so the turn loop has no hook to drive and every scenario would fail on turn 1 '
        f'— after the scenario set has been paid for. Declare the bench: add `{SHAPE_KEY}: {PIPELINE_SHAPE}` as a ROOT key of '
        f'{CONFIG_FILENAME}, beside `workspace:` and `project:` (a `{SHAPE_KEY}:` nested under `agent:` is not read — move it to the root). '
        f'If this bench really is a dialog whose invoke_agent the probe cannot see at module scope — bound inside a function body, or '
        f'attached to the module only once something has run — declare `{SHAPE_KEY}: {DIALOG_SHAPE}` instead, or pass '
        f'`--{SHAPE_KEY} {DIALOG_SHAPE}` for one run.'
    )


def contradicted_shape_declaration(simulate_path: Path | str, *, override: str | None) -> tuple[Shape, Path] | None:
    """The VALID ``shape:`` declaration an ``--shape`` override contradicts, or ``None``.

    The override WINS — :func:`resolve_scenario_shape` reads it first — and the flag is the operator's
    explicit decision, so the runner reports the contradiction as a finding and runs. ``None`` when there
    is no override, it agrees, or there is no valid declaration to contradict: a typo'd ``shape: pipline``
    is not a declaration (``_declared_shape`` raises), and the override is what rescues a run from it,
    so "could not establish a declaration" never becomes "contradicted one".
    """
    if override is None:
        return None
    try:
        declared = _declared_shape(Path(simulate_path))
    except (WorkspaceConfigError, OSError):
        return None
    if declared is None or declared[0] == override:
        return None
    return declared


def _bench_relative(path: Path, simulate_path: Path) -> str:
    """``path`` relative to the bench root (the folder above the harness's), for stable finding evidence."""
    root = simulate_path.expanduser().resolve().parent.parent
    try:
        return str(path.expanduser().resolve().relative_to(root))
    except ValueError:
        return path.name


def unreset_external_world_finding(simulate_path: Path | str) -> GateFinding | None:
    """The holding finding for a world nothing resets between scenarios, or ``None``.

    Exactly two things are reset per scenario: the mock session the backend opens, and the child process
    the runner spawns. A world kept anywhere else — a file, a real database, a CRM, a server-side
    conversation store — is not reset by either, so scenario 2 may open on scenario 1's writes. It
    does not crash anything, and whether it contaminates depends on whether the agent WRITES, which
    only a reader of the agent's tools can say — so it HOLDS the run until an overrule that states
    what was verified (read-only tools, a reset bound elsewhere) lifts it.

    THE DECLARATION is read first: an undeclared or ``mock``-owned world needs no hook. THE WALK then
    answers one of three things: it found ``reset`` (``None``); it found no ``reset``; or it could not
    read the harness at all (missing, unreadable, unparseable to this interpreter — a child pinned to
    another Python through ``PLURAI_CHILD_PYTHON`` included). The last two are two sentences, never
    one: "could not look" and "looked and found no hook" send an author to different halves.

    THE OPT-IN lifts it: a bench that declares ``world_state.shared_world_ok: true`` chose a
    shared world explicitly, so the run proceeds and carries :func:`shared_world_risk` instead of holding.

    An unknown owner, or two configs that disagree, raise :class:`WorkspaceConfigError` (invariant).
    """
    path = Path(simulate_path)
    declared = _declared_world_owner(path)
    if declared is None or declared[0] == WORLD_OWNER_MOCK:
        return None
    owner, config_path = declared
    bound = _module_scope_bindings_of_file(path)
    if bound is not None and RESET_HOOK in bound:
        return None
    if _declared_shared_world_ok(path):
        return None
    unreadable_as = None if bound is not None else _UNREADABLE_HARNESSES.get(path.expanduser().resolve(), 'unreadable')
    head = (
        f'{config_path} declares `{WORLD_STATE_KEY}.{WORLD_OWNER_KEY}: {owner}`, '
        "so the world this suite scores is not the mock's and the Plurai backend cannot reset it"
    )
    if bound is None:
        remedy = (
            f'{head} — but this interpreter could not read {path} at all ({unreadable_as}), so whether it resets that world could not be '
            "established either way. If the hook is in fact missing, every scenario after the first would run on its predecessor's "
            f'writes. Re-run against a harness this interpreter can read (a child pinned through PLURAI_CHILD_PYTHON: run the parent on '
            f'that interpreter too), or declare `{WORLD_STATE_KEY}: {{{WORLD_OWNER_KEY}: {WORLD_OWNER_MOCK}}}` if the deployed mock really '
            f'owns every record this agent reads.'
        )
    else:
        remedy = (
            f"{head} — but {path} exports no `{RESET_HOOK}`. Every scenario after the first may run on its predecessor's writes, and the "
            f'verdicts would be about the run order. Export `def {RESET_HOOK}() -> None:` at module scope and clear the store this harness '
            f'owns, or declare `{WORLD_STATE_KEY}: {{{WORLD_OWNER_KEY}: {WORLD_OWNER_MOCK}}}` if the deployed mock really owns every record '
            'this agent reads. If the agent never writes that store, overrule this finding by its fingerprint. To run on this shared '
            f'world anyway, declare `{WORLD_STATE_KEY}.{WORLD_SHARED_OK_KEY}: true`: the run then carries the named risk '
            f'`{SHARED_WORLD_RISK_KIND}` and its stability retries are not comparable.'
        )
    return make_finding(
        WORLD_UNRESET_EXTERNAL,
        subject=_bench_relative(path, path),
        evidence={
            'owner': owner,
            'config': _bench_relative(config_path, path),
            'harness': _bench_relative(path, path),
            'reset_bound': None if bound is None else False,
            'unreadable_as': unreadable_as,
        },
        observed={'unreadable': unreadable_as},
        remedy=remedy,
        draft_ref=None,
    )


def unobservable_world_warning(simulate_path: Path | str) -> str | None:
    """What this run cannot MEASURE, or ``None`` — the $0 preflight for an UNOBSERVABLE world.

    The second sentence beside :func:`unreset_external_world_finding`, read off the same two local
    files at the same rung, and kept a SEPARATE function for the reason that one's docstring gives
    about its own two arms: they send an author to different halves of the problem. That one is
    about CLEARING the world between scenarios; this one is about READING it at the end of one.

    A WARNING, AND DELIBERATELY NOT A REFUSAL. Its sibling refuses because an unreset world
    corrupts every scenario after the first, transcript included — nothing in the run is
    salvageable. An unobservable world costs only the world-reading policies: the dialog happened,
    the transcript is honest, and every ``llm_as_judge`` policy scores exactly as it would have.
    Refusing would throw those away, and would silently revoke the decision the reset gate made
    when it let this population RUN. The proportionate answer is to warn here and abstain there —
    the child emits ``None`` on the world channel and the backend resolves every world-reading policy
    to ``world_unobservable`` rather than to a conclusive pass about a store nobody snapshotted.

    THE DECLARATION FIRST, exactly as its sibling reads it and for the same reason: almost every
    bench declares no owner, and an undeclared or ``mock``-owned world needs no hook at all —
    the backend opens a session per scenario, seeds it, and the child snapshots it.

    SILENT WHEN IT COULD NOT LOOK (``bound is None``). Naming a hook to an author whose file
    nobody could parse is a lie if they exported it, and the case is already covered one function
    up, which HOLDS it — so on the CLI path this arm is reached only once that finding is overruled, and otherwise exists so an
    embedder calling this function alone cannot be told something untrue.
    """
    path = Path(simulate_path)
    declared = _declared_world_owner(path)
    if declared is None or declared[0] == WORLD_OWNER_MOCK:
        return None
    owner, config_path = declared
    bound = _module_scope_bindings_of_file(path)
    if bound is None or WORLD_SNAPSHOT_HOOK in bound:
        return None
    return (
        f"{config_path} declares `{WORLD_STATE_KEY}.{WORLD_OWNER_KEY}: {owner}`, so the world this suite scores is not the mock's — but "
        f'{path} exports no `{WORLD_SNAPSHOT_HOOK}`, so nothing in this run can read that store. The dialog and every llm_as_judge policy '
        f'are unaffected; every WORLD-READING policy is demoted to not-measured, because the only world this run could snapshot is the '
        f"mock's session and reporting that one as the world you declared would score a conclusive pass about a store nobody read. Export "
        f'`def {WORLD_SNAPSHOT_HOOK}() -> dict:` at module scope, returning `{{entity_id: {{...}}}}` for the store this harness owns. If '
        f'part of that world really is in the mock, declare `{WORLD_STATE_KEY}.owned_types: [<the entity types your store holds>]` and '
        f"return only your store's rows: the run composes them with the mock's rows of every other type. Or declare "
        f'`{WORLD_STATE_KEY}: {{{WORLD_OWNER_KEY}: {WORLD_OWNER_MOCK}}}` if the deployed mock really owns every record this agent reads.'
    )


def turn_input_findings(simulate_path: Path | str) -> tuple[GateFinding, ...]:
    """The ``agent.io`` words this build dropped, as findings — read through the SAME lenient parser
    every reader uses (:func:`plurai.agent_io.declared_agent_io`), so the run-start rung, the
    per-scenario reader and the child agree.

    An unknown part kind ships with its finding: the parent never projects parts, and a newer child
    reads the kind itself. An unknown sub-key HOLDS: a misspelled turn-0 key (``opens``, ``may_open``)
    would drive every scenario the wrong way. A malformed value of a KNOWN key still raises
    :class:`~plurai.agent_io.AgentIOConfigError` (invariant), and this runs AFTER the world rung, which
    owns the "config could not be read at all" posture the same exception class also covers.
    """
    path = Path(simulate_path)
    surface = declared_agent_io(path)
    out: list[GateFinding] = []
    for defect in surface.defects if surface is not None else ():
        source = _bench_relative(defect.source, path) if defect.source is not None else None
        if defect.code == 'unknown_field':
            out.append(
                make_finding(
                    HARNESS_AGENT_IO_UNKNOWN_FIELD,
                    subject=defect.key,
                    evidence={'key': defect.key, 'value': defect.value, 'source': source},
                    observed={},
                    remedy=f'`{defect.key}` declares `{defect.value}`, which this build does not read, so '
                    'nothing acts on it. Fix the spelling in the config, or overrule this finding if the key is meant for a newer build.',
                    draft_ref=None,
                )
            )
        else:
            out.append(
                make_finding(
                    HARNESS_AGENT_IO_UNKNOWN_PART_KIND,
                    subject=defect.key,
                    evidence={'key': defect.key, 'value': defect.value, 'source': source},
                    observed={},
                    remedy=f'`{defect.key}` declares the part kind {defect.value!r}, which this build cannot put on a turn; it is left off '
                    'the declared surface. Remove it, or upgrade plurai on both sides if a newer build carries it.',
                    draft_ref=None,
                )
            )
    return tuple(out)


#: The typed failure kind of a scenario that has no opening user message on a harness that never declared
#: it can open. ``harness_`` makes it harness-owned (``_protocol.is_harness_owned``): the row is not
#: valid, never a failed agent trial.
HARNESS_NO_OPENER_KIND: Final = 'harness_no_opener'


class HarnessNoOpener(WorkspaceConfigError):  # noqa: N818  # named for the fact, like its siblings
    """A scenario with no opener on a harness that does not declare ``agent.io.may_open``: the scenario set
    and the config do not describe the same bench. Remedy: declare ``may_open``, or regenerate the scenario."""

    kind: ClassVar[str] = HARNESS_NO_OPENER_KIND


#: The typed failure kind of a scenario whose user hands over a content part the harness never declared
#: it reads. Harness-owned, like :data:`HARNESS_NO_OPENER_KIND`.
HARNESS_UNDECLARED_INPUT_KIND: Final = 'harness_undeclared_input'


class HarnessUndeclaredInput(WorkspaceConfigError):  # noqa: N818  # named for the fact, like its siblings
    """A user line carries a content part (a file, an image) of a kind ``agent.io.input`` does not declare:
    the scenario set and the config do not describe the same bench. Remedy: declare the kind and read it in
    ``invoke_agent``, or drop the hand-over from the scenario."""

    kind: ClassVar[str] = HARNESS_UNDECLARED_INPUT_KIND


def undeclared_input_kinds(content: str | list[dict[str, Any]] | None, agent_io: AgentIO) -> tuple[str, ...]:
    """The declarable part kinds ``content`` carries that ``agent_io`` does not accept, in first-seen order.

    A string, an absent line and a part whose kind this build cannot carry give nothing: the child owns
    the refusal of a kind it has never heard of, and a string is the text every harness takes.
    """
    if not isinstance(content, list):
        return ()
    kinds = (str(part.get('type')) for block in content if (part := declared_part(block)) is not None)
    return tuple(dict.fromkeys(kind for kind in kinds if kind in INPUT_PART_KINDS and not agent_io.accepts_part(kind)))


def _refuse_undeclared_input(content: str | list[dict[str, Any]] | None, declared: AgentIO | None, *, scenario_id: str, simulate_path: Path, turn: str) -> None:
    """Raise :class:`HarnessUndeclaredInput` when ``content`` carries a part kind the harness does not declare.

    ``declared`` is what :func:`~plurai.agent_io.declared_agent_io` read: ``None`` when the config has no
    ``agent.io``, in which case the harness is held to :data:`~plurai.agent_io.DEFAULT_AGENT_IO` and the
    message says the text-only surface is that default rather than a declaration. A declared ``agent.io``
    may omit ``input``, so its surface is stated as what the harness accepts, not as an ``input`` it wrote.
    """
    agent_io = DEFAULT_AGENT_IO if declared is None else declared
    if not (missing := undeclared_input_kinds(content, agent_io)):
        return
    if declared is None:
        surface = f'declares no `{AGENT_IO_KEY}` in its {CONFIG_FILENAME}, so it takes the default of {" and ".join(agent_io.input_parts)} only'
    else:
        surface = f'declares `{AGENT_IO_KEY}` in its {CONFIG_FILENAME}, which accepts the part kinds {list(agent_io.input_parts)}'
    raise HarnessUndeclaredInput(
        f'refusing to run scenario {scenario_id!r}: the {turn} hands the agent a content part of kind {", ".join(map(repr, missing))}, '
        f'but {simulate_path} {surface}. '
        f'Scoring it would grade an agent that was never shown what the user sent. Declare the kind in `{AGENT_IO_KEY}.{INPUT_FIELD}` '
        f'(and read it in `invoke_agent`), or remove the hand-over from the scenario.'
    )


# ---------------------------------------------------------------------------
# High-level driver: ``Orchestrator.run()`` runs one scenario end-to-end.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThreadResult:
    """Outcome of a single ``Orchestrator.run()`` invocation."""

    thread_id: str
    turns: int
    finalize: FinalizeResponse
    user_messages: list[str]
    chatbot_messages: list[str]
    # Every tool call ``/mock/execute`` did not answer during this scenario, so the agent was
    # handed a stub (``_protocol.FinalizeResponse.unanswered_mock_calls``). A measurement fact, never
    # a verdict: the runner grades the scenario and records a ``mock.tool_call_unanswered``
    # finding, and re-drives only when a call failed the way a mock restart does. ``None`` = the
    # child claimed nothing.
    unanswered_mock_calls: list[dict[str, Any]] | None = None
    # The agent's full per-turn message chains (serialized dicts incl. tool-call
    # AIMessages + ToolMessages), flattened across turns. Lets the client write a
    # rich transcript.json (with tool_calls) for the diagnose flow. The subprocess
    # path always populates this from the child's per-turn message chains.
    chatbot_full_messages: list[dict[str, Any]] = field(default_factory=list)
    # The recorded tool-event log from the mock service (child.finalize()), so the
    # client can write tool_events.json — the diagnose flow's primary signal.
    #
    # ``None`` means NO tool channel was read on this scenario, and it reaches
    # ``tool_events.json`` as a literal ``null`` rather than as ``{}``: the re-score lane reads
    # that file back and routes an absent channel to "not measured" instead of grading the agent
    # for having invoked nothing. ``{}`` / ``{'events': []}`` stays the answer of a run that DID
    # read a channel and found it empty.
    tool_events: dict[str, Any] | None = None
    # Turn-ordered, role-tagged conversation records (one per turn: user_message,
    # agent_message, the turn's turn_messages chain incl. tool calls, the next
    # output_user_message, should_terminate). Persisted ADDITIVELY as transcript.json's
    # ``turns`` field — free of the two-positional-array off-by-one that the parallel
    # arrays carry, and making tool calls attributable per turn. The two flat arrays remain
    # for readers of the flat shape.
    transcript_turns: list[dict[str, Any]] = field(default_factory=list)
    # ``{reply_source: turns}`` over this scenario — how many of its replies were the agent's own
    # words, a tool result a graph ended on, a typed interrupt, an empty turn, or a suffix the
    # classifier could not read (``plurai.child.REPLY_SOURCES``). Counted rather than left per-turn
    # because a non-``assistant`` reply is only actionable if somebody sees it: the per-turn label
    # is a cell nobody opens, and this is the number a run summary or an audit reads. Empty only
    # for a scenario with no turns.
    reply_source_counts: dict[str, int] = field(default_factory=dict)
    # The TYPED reason this dialog ended for something OTHER than the user reaching their goal:
    # ``USER_GAVE_UP_REASON`` (the simulated user abandoned it — reported by the server),
    # ``TURN_CAP_REASON`` (the budget ran out mid-conversation), or ``ADAPTER_DEGRADED_REASON``
    # (every turn was emptied by the parent, so nothing about the agent was measured). ``None``
    # means none of the three. A consumer must branch on this rather than on a short turn count:
    # a run that exhausted its budget is not a run that finished, a run whose every reply the
    # harness manufactured is not a measurement of the agent at all, and a run the user walked
    # out of is a finding about the SCENARIO rather than about either.
    #
    # The membership rule for this channel lives with the constants above; a value may also
    # arrive here unrecognised, from a newer backend, and is carried through rather than dropped.
    #
    # The THIRD reason, ``USER_GAVE_UP_REASON``, is the simulator's own and arrives REPORTED:
    # the backend returns it on ``UserTurnResponse.ended_early``. It closes the case this
    # channel would otherwise swallow — ``should_terminate`` tracks the goal being MET, so a
    # persona chasing an unanswerable target loops to the budget, and "the budget ran
    # out" is all the harness can see. The backend also records it server-side, which is where
    # the JUDGES are: nothing on this channel ever reached them, because no ``FinalizeRequest``
    # field carries it.
    #
    # PRECEDENCE, most specific last-word first: ``adapter_degraded`` outranks both (it says
    # nothing about the agent was measured AT ALL, which is a statement about the whole run);
    # then ``user_gave_up`` (a declaration); then ``turn_cap_exhausted`` (a budget).
    ended_early: str | None = None
    # The simulated user's own stated CAUSE for the reason above, when that reason is its
    # give-up and the server reported one (see ``GIVE_UP_CAUSE_*``). Travels WITH the reason —
    # displaced when the reason is — and ``None`` from a backend that reports none: a give-up
    # with no stated cause.
    ended_early_cause: str | None = None
    # The resume seed this scenario ran from, VERBATIM as the backend sent it, or None
    # for a fresh scenario. Carried so the client can mirror it into the run artifacts: it is
    # the only record of where a resumed conversation started, so without it an offline
    # re-score rebuilds a different turn-0 baseline than the live run excised against and
    # scores a different conversation.
    resume_seed: dict[str, Any] | None = None
    # The turns the PARENT emptied on the agent's behalf, ``{turn_idx, subkind}`` each (see
    # ``ChildProcess.degraded_turns``). Empty on every ordinary run. Non-empty means this
    # transcript contains replies the agent did not give: a judge reading those turns is
    # scoring the harness, and a consumer that cannot see this cannot know to discount them.
    degraded_turns: list[dict[str, Any]] = field(default_factory=list)
    # The turns whose reply the CHILD trimmed to fit one IPC line, ``{turn_idx}`` each (see
    # ``ChildProcess.ipc_trimmed_turns``). The reply text is the agent's, its message chain is not on the
    # record: a judge reading those turns' tool trail reads less than the agent did. Empty on every ordinary run.
    ipc_trimmed_turns: list[dict[str, Any]] = field(default_factory=list)
    # The messages the single-role split took OUT of the agent channel, by the rule that moved
    # them: ``{'role_declared': [...], 'verbatim_echo': [...]}`` (see ``split_user_echo``). The
    # SIBLING of ``degraded_turns`` and carried for the same reason — both say that this
    # transcript's agent channel is not simply what the adapter returned — and it rides the
    # result rather than only a log line because a log line is not an artifact: the run's own
    # summary, the transcript and the screen could all say nothing while a user-pasted URL sat
    # in the channel a judge reads as the agent's.
    #
    # ``role_declared`` is the ORDINARY LangGraph shape and is recorded, not warned about;
    # ``verbatim_echo`` is the contract breach. Kept apart here so a consumer can tell them
    # apart without re-deriving the rule.
    user_echoes_dropped: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # The server-side finalize JOB this scenario submitted. Recorded so the run's own artifacts
    # account for one finalize per scenario — a backend finalize row with no artifact naming it
    # (or a scenario naming none) is then attributable instead of merely noticed.
    finalize_job_id: str | None = None
    # The turn failure this dialog LANDED on, when one did: ``{turn_idx, closing, kind, exception,
    # child_exc_class, error}`` (+ ``pre_crash_tool_activity`` for a recursion). ``closing`` says
    # whether it was the closing call (the dialog had already ended, so no ``ended_early`` is set) or
    # a turn mid-dialog (``ended_early`` names it — see ``landed_reason``). An untyped closing-call
    # failure lands too, with ``kind: None``. ``None`` when every turn answered.
    turn_failure: dict[str, Any] | None = None
    # The finalize snapshot channels that could not be read even after one readiness-gated re-read
    # (``{'world'?: detail, 'events'?: detail}``); each shipped as ``None`` (unread). ``None`` when
    # both were read.
    capture_failed: dict[str, str] | None = None
    # ``{path: source type}`` for each value of the harness's own world snapshot carried in its JSON
    # form (``_protocol._json_shaped``); ``None`` when nothing was coerced.
    world_values_coerced: dict[str, str] | None = None
    # The files routing the agent's vendor calls to the mock that changed during the session
    # (``ChildProcess.interception_changed``); ``None`` when none changed.
    interception_changed: list[str] | None = None
    # ``seed_state``'s own account of what it restored from ``resume_seed`` — ``None``
    # for full fidelity (every fresh scenario, and every resumed one whose harness didn't
    # report otherwise), or the harness's freeform description of a reduced restore.
    # Carried so the client can mirror it into the run artifacts, the same as ``resume_seed``
    # (``run_simulation._write_success_artifacts``): the backend echoes the report back on the
    # finalize response, but the artifacts persist that payload's world state, tool events
    # and eval block only — never its resume fields — so the mirror taken off THIS field is
    # the run's only on-disk record of restore fidelity.
    resume_fidelity: dict[str, Any] | None = None
    # The two INJECTED INPUTS this scenario started from: the server's ``initial_world_state`` and
    # its ``initial_agent_context`` as the backend sent it. A served document is the
    # ``<scheme>:///documents/…`` reference, not the bytes the child receives
    # (``_context_for_agent`` folds them at the hand-off), so ``transcript.json`` stays small
    # enough to re-post. Carried for the same reason as
    # ``resume_seed`` and ``resume_fidelity``: the only other record is the observability sink's
    # ``example_inputs``, which a run without a sink never writes, so without these a finished
    # run could not be audited for what was actually injected into it.
    # Empty dicts mean "the server sent none", which is what an ordinary seedless scenario gets.
    initial_world_state: dict[str, Any] = field(default_factory=dict)
    initial_agent_context: dict[str, Any] = field(default_factory=dict)
    # The keys of ``initial_agent_context`` the harness never read (``ChildProcess.unread_inputs``);
    # ``None`` = not recorded. Carried beside them so the transcript records what the agent did NOT get.
    unread_inputs: list[str] | None = None
    # The world right after ``seed_state`` (``ChildProcess.post_seed_world_state``); ``None`` = not read.
    # The world the run started from: the backend decides origins against it, live and on a re-score.
    post_seed_world_state: dict[str, Any] | None = None
    # WHICH ROUTE drove this scenario — ``'dialog'`` (the turn loop) or ``'pipeline'`` (ONE
    # ``invoke_pipeline`` envelope). Carried beside the injected inputs above and for the same
    # reason: a hop's transcript has ``turns: []`` and ``turn_count: 0``, which is also what a
    # dialog that died before its first turn looks like, and only this field tells them apart.
    scenario_shape: Shape = DIALOG_SHAPE
    # The hop itself: ``{'outputs', 'error', 'duration_s', 'timed_out'}`` — what the component
    # RETURNED as /finalize carries it (a served document it echoes is its reference, a leaf over
    # the wire cap a descriptor), the structured ``pipeline_error`` when it raised / returned a non-dict / timed
    # out, how long it took, and whether it ever came back. ``None`` on every dialog scenario, so
    # nothing about a dialog run changes.
    #
    # NO ``inputs`` key. THE READ RULE, once, for every consumer:
    #
    #     pipeline_inputs  := transcript['injected_inputs']['initial_agent_context']
    #     pipeline_outputs := transcript['pipeline_io']['outputs']
    #     pipeline_error   := transcript['pipeline_io']['error']
    #
    # The hop's inputs ARE ``initial_agent_context`` above, in its recorded form: a served document is
    # its reference, and the bridge folds it to the bytes only in the copy it hands the component
    # (``_context_for_agent``). /finalize carries that same recorded dict, so the two are equal BY
    # CONSTRUCTION, not by two writers agreeing — and it already reaches
    # ``transcript.json`` under ``injected_inputs`` (unconditionally, empty dicts included, so a
    # reader never branches on presence). One carrier for one truth. The standalone
    # ``pipeline_io.json`` forensic twin restates them because it is a file opened on its own with
    # no re-score role, not because the value lives twice in the file that IS re-scored.
    pipeline_io: dict[str, Any] | None = None
    # What the component returned, before the wire projection ``pipeline_io['outputs']`` carries
    # (served-document echoes as references, leaves over the cap as descriptors). Written only to the
    # standalone ``pipeline_io.json`` twin, the one file a customer opens to read the real output;
    # ``transcript.json`` keeps the projection so a re-post stays small. ``None`` on every dialog scenario.
    pipeline_raw_outputs: dict[str, Any] | None = None

    def summary(self) -> str:
        # pass_rate is nullable on the wire (explicit JSON null validates to None);
        # coerce to 0.0 before the :.3f format, which would raise on None.
        pass_rate = float(self.finalize.pass_rate or 0.0)
        # ``user_turns=``, not a bare ``turns=``: this counts /user-turn CYCLES, while
        # transcript.json's ``turn_count`` counts entries in ``turns`` — which includes the
        # closing reply, so the two legitimately differ by one. Printed side by side under one
        # word, that difference reads as a bug and makes the log useless for cross-checking.
        return (
            f'thread={self.thread_id} user_turns={self.turns} pass_rate={pass_rate:.3f} '
            f'aggregate_score={self.finalize.aggregate_score:.3f} status={self.finalize.status}'
        )


@dataclass(slots=True)
class ScenarioProgress:
    """What the latest dialog :meth:`Orchestrator.run` has produced, readable after it FAILS.

    A run that raises returns no :class:`ThreadResult`, so everything it accumulated would die
    with its locals. This record holds the SAME list objects the turn loop appends to, which
    makes it exact at the moment of failure wherever that is — a turn that raised, the hard
    ceiling's cancellation, a finalize that failed — with no bookkeeping per step. The runner
    reads it to write the partial evidence beside ``error.json``.
    """

    thread_id: str
    initial_world_state: dict[str, Any]
    initial_agent_context: dict[str, Any]
    #: The world right after ``seed_state`` (``ChildProcess.post_seed_world_state``); ``None`` = not read.
    post_seed_world_state: dict[str, Any] | None
    resume_seed: dict[str, Any] | None
    user_messages: list[str | list[dict[str, Any]]]
    chatbot_messages: list[str]
    chatbot_full_messages: list[dict[str, Any]]
    transcript_turns: list[dict[str, Any]]
    recovered_message_calls: list[RecoveredToolCall]
    #: The mock's event log as ``child.finalize()`` returned it. ``mock_log_read`` travels beside
    #: it because ``None`` is ALSO what a child with no tool channel returns: before the child
    #: finalized, the log was never asked for, which is a different fact from an absent channel.
    mock_tool_events: dict[str, Any] | None = None
    mock_log_read: bool = False
    #: The merge ``run()`` shipped to ``/finalize``, once it has. A failure after that point (the
    #: finalize itself, a stub refusal) then reports exactly what was shipped, and the merge's
    #: warnings do not print a second time. ``tool_events_shipped`` tells a shipped ``None`` apart.
    shipped_tool_events: dict[str, Any] | None = None
    tool_events_shipped: bool = False

    def tool_events(self) -> dict[str, Any] | None:
        """The tool events so far: the mock's log once it was read, unioned with the calls
        recovered from the agent's messages and trail.

        The ONE merge :meth:`Orchestrator.run` ships to ``/finalize``, so the partial evidence
        and ``tool_events.json`` cannot describe the same calls two ways. ``None`` when nothing
        was read and nothing recovered.
        """
        if self.tool_events_shipped:
            return self.shipped_tool_events
        return merge_recovered_tool_calls(
            self.mock_tool_events,
            recovered_calls=self.recovered_message_calls,
            agent_states=_agent_states_by_turn(self.transcript_turns, seed_agent_state=(self.resume_seed or {}).get('agent_state')),
        )


def _releasing_served_documents[**P, R](
    run: Callable[Concatenate[Orchestrator, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Orchestrator, P], Coroutine[Any, Any, R]]:
    """``Orchestrator.run`` that releases the client's served documents for its thread when the scenario ends, however it ends.

    One client serves every scenario of a run and a fold is up to several MB, so a thread's documents
    are dropped once its result is built. Nothing reads them afterwards: a re-drive creates a new thread,
    and the artifacts are written from the :class:`ThreadResult`, which holds the recorded forms.
    """

    @wraps(run)
    async def released(self: Orchestrator, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await run(self, *args, **kwargs)
        finally:
            if self.thread_id is not None:
                self.client.forget_thread(self.thread_id)

    return released


class Orchestrator:
    """End-to-end driver for a single scenario.

    The driver:
      1. Creates a thread on the backend (server returns the seed user message
         + initial world state + agent context).
      2. Spawns the agent subprocess (``python -m plurai.child``), which OWNS the
         mock client and session: in ``mock_mode='local'`` the child hosts its own
         in-process mock on a per-run loopback port (handlers from the customer's
         ``simulate.py``); in ``'connect'`` mode it connects to the deployed mock
         URL the server returned (raising if unreachable). The orchestrator never
         configures a mock client itself — the posture comes from ``settings``.
      3. Drives the local agent loop turn-by-turn, asking the backend for the
         next user message after each agent response.
      4. Finalizes — ships the world-state + event snapshots + transcript to
         the backend, which runs end-to-end policy evaluation.
    """

    #: Builds the storage client that takes the finalize world's images; ``None`` is a client of this run's own backend and bearer.
    storage_client_factory: StorageClientFactory | None = None

    def __init__(
        self,
        *,
        simulate_path: Path,
        settings: Settings,
        observability: ObservabilitySink,
        client: OrchestratorClient,
        shape: Shape | None = None,
    ):
        """Construct an orchestrator for one customer agent.

        ``shape`` is the OPERATOR's override (``--shape``), for debugging only; ``None`` — the
        ordinary case — means the bench's own declaration decides. The AST probe of
        ``simulate.py`` is a HINT, not a gap-filler: a bench that exports ``invoke_pipeline`` and
        NO ``invoke_agent`` and declares no ``shape:`` is REFUSED at run start (the CLI's
        :func:`undeclared_pipeline_refusal` preflight; an embedder calling this constructor
        directly keeps the warning); one that exports BOTH is warned and dispatched as
        ``dialog``, because a bench that can hold a conversation has a defensible route and
        refusing it would be a false red. See :func:`resolve_scenario_shape`.

        Ownership rule: the orchestrator NEVER closes the injected ``observability``
        sink — whoever built the sink closes it (the run strategies in
        ``plurai.run_simulation``, or the embedder). The sink is routinely SHARED by
        many short-lived orchestrators (one per scenario), so a per-orchestrator
        close would shut the process-global provider down mid-run and DROP every
        later scenario's spans ("Shutdown called, ignoring Span"). ``aclose`` drains
        only this orchestrator's OWN HTTP client.
        """
        self.settings = settings
        if settings.resolved_endpoint is not None:
            register_backend_token(settings.resolved_endpoint.token)
        self.simulate_path = simulate_path
        self.observability = observability
        self.client = client
        self.shape_override = shape
        # RESOLVED BEFORE ANY THREAD EXISTS. Inside the per-scenario try, a single typo'd
        # `shape:` would mint a thread, a server-side opener and an error.json for EVERY scenario —
        # for a condition knowable before the first one. Here it raises out of construction
        # instead. NOT once per run: the CLI builds one orchestrator per scenario, and again per
        # retry, so the config is still read once per construction.
        self.scenario_shape = resolve_scenario_shape(simulate_path, override=shape)
        # The thread_id of the active scenario, set the moment ``POST /threads``
        # succeeds — setting it any later would leave every earlier failure with
        # no id to report or to /fail. Read by ``_cancel_inflight_finalize`` and
        # by the runner's failure + re-drive paths; ``None`` means create has not
        # succeeded in this run.
        self.thread_id: str | None = None
        # The turns the active attempt has completed, for the same reason and the same readers:
        # a raise unwinds ``run()``, and the failure path keeps what the attempt reached.
        self.turns_reached: list[dict[str, Any]] = []
        # How many entities the server's ``initial_world_state`` seeded for the active attempt: a world that
        # served nothing before any turn completed, over a seed this size, is a seed fault (``redrive.seed_fault``).
        self.seeded_entities = 0
        # The mock URL the SERVER handed this scenario, resolved to absolute. The
        # re-drive readiness gate probes this one — the CLI's --mock-service-url is
        # routinely omitted (the server picks the env), so gating on the CLI value
        # would skip the mock phase in the common case and re-drive straight into a
        # mock that is still redeploying.
        self.mock_service_url: str | None = None
        # Set to the job_id immediately after a successful async finalize submit;
        # cleared to None on a terminal server response (done / failed / schema error).
        # Remains set on FinalizePollTimeout so cancel-on-abandon can DELETE the job.
        self._inflight_finalize_job_id: str | None = None
        # The LAST finalize job this orchestrator submitted, kept past the terminal response
        # the field above clears. It is what puts the job id in the scenario's own artifacts,
        # so "one finalize per scenario" is a property a finished run can be checked for
        # rather than one an operator has to take on faith from the backend's job table.
        self.last_finalize_job_id: str | None = None
        # Facts about the RUN the caller hands ``run()`` for the sinks' record metadata (a trial's
        # ``trial_index`` / ``trial_count`` / ``trial_group`` — ``eval_mapping.RUN_FACT_KEYS``).
        self._run_facts: dict[str, Any] = {}
        # The latest dialog run's accumulators (see ScenarioProgress), reset by the next ``run()``
        # and set once its child is up and seeded. ``None`` for a one-shot hop, and for a run that
        # failed before its first turn could reach the agent (at create, at a refusal, at child
        # spawn or init), which has no conversation to report.
        self.progress: ScenarioProgress | None = None
        # The active run's ``on_setup_done``, called ONCE by :meth:`_setup_done` (see ``run``).
        self._on_setup_done: Callable[[], None] | None = None
        # The active run's ``on_finalize_start``, called ONCE when the attempt starts its finalize (see ``run``).
        self._on_finalize_start: Callable[[], None] | None = None
        # The active run's ``on_child_closed``, called ONCE by :meth:`_child_closed` (see ``run``).
        self._on_child_closed: Callable[[], None] | None = None
        # The world copy the active run leased (``world_state.slots``), or ``None`` for one shared world.
        self.world_slot: int | None = None

    def _child_env(self) -> dict[str, str]:
        """The child's environment: the allowlisted parent environment and the harness's declared variables
        (:func:`plurai.child_env.launch_environment`), :data:`WORLD_SLOT_ENV` for a slotted attempt, and the
        run's resolved ``PLURAI_URL`` / ``PLURAI_TOKEN`` pair, so the agent's ``code`` channel reaches the
        hosted mock whichever way the pair was configured."""
        extra = {} if self.world_slot is None else {WORLD_SLOT_ENV: str(self.world_slot)}
        try:
            endpoint = self._endpoint()
        except NotConfiguredError:
            # A run on an injected client and a local mock names no backend. The child then gets no pair at all,
            # never the parent's raw values. Any other resolver error (a half-set pair, a broken saved login, an unusable
            # url) is a configuration the operator meant to use, and propagates.
            env = launch_environment(self.simulate_path, extra=extra)
            return {name: value for name, value in env.items() if name not in (URL_ENV, TOKEN_ENV)}
        return launch_environment(self.simulate_path, url=endpoint.url, token=endpoint.token, extra=extra)

    def _run_storage_client(self) -> StorageClient:
        """A storage client for the run's own backend and bearer, resolved when the first image is uploaded."""
        endpoint = self._endpoint()
        return storage_client_for(endpoint.url, endpoint.token)()

    def _endpoint(self) -> Endpoint:
        """The run's resolved backend, its token registered for redaction."""
        endpoint = self.settings.endpoint
        register_backend_token(endpoint.token)
        return endpoint

    def _setup_done(self) -> None:
        """Report that this attempt's setup is over: the thread exists and the child is initialised and seeded."""
        callback, self._on_setup_done = self._on_setup_done, None
        if callback is not None:
            callback()

    def _child_closed(self) -> None:
        """Report that this attempt's agent process is gone: its world and mock log were read inside its scope, and only
        the ``/finalize`` of what was captured is left."""
        callback, self._on_child_closed = self._on_child_closed, None
        if callback is not None:
            callback()

    async def aclose(self) -> None:
        """Drain this orchestrator's OWN HTTP client.

        The injected observability sink is deliberately NOT closed here — sink
        lifecycle belongs to whoever created it (see ``__init__``); per-scenario
        span draining already happens via the sink's ``drain_run`` scenario scope.
        """
        await self.client.aclose()

    async def __aenter__(self) -> Orchestrator:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    @_releasing_served_documents
    async def run(
        self,
        *,
        scenario_set_id: str,
        scenario_id: str,
        eval_set_id: str | None = None,
        agent_name: str = 'plurai-agent',
        agent_version: str = 'sim-1',
        max_turns: int = 5,
        experiment_name: str | None = None,
        mock_service_url: str | None = None,
        dataset_name: str | None = None,
        scenario_time_budget: float | None = None,
        run_facts: Mapping[str, Any] | None = None,
        on_setup_done: Callable[[], None] | None = None,
        on_finalize_start: Callable[[], None] | None = None,
        replay_turn_only: bool = False,
        redrive_available: bool,
        allow_turn_timeout_redrive: bool,
        on_child_closed: Callable[[], None] | None,
        world_slot: int | None = None,
    ) -> ThreadResult:
        """Drive a complete simulation for one scenario.

        ``on_child_closed`` is called once, right after the agent process closed and before ``/finalize`` (the world
        snapshot and the mock log are read inside the child's scope, so the finalize needs nothing the process held). A
        runner that leases what only one agent process may hold gives it back there, so the next attempt's process starts
        while this one is judged. It is not called when the attempt raises; the caller releases on its own exit too.
        REQUIRED (``None`` when the caller holds nothing for the child's life), so no caller forgets to say.

        ``world_slot`` is the world copy the caller leased for this attempt (``world_state.slots``):
        the child is spawned with ``PLURAI_WORLD_SLOT=<world_slot>`` (:data:`WORLD_SLOT_ENV`) so the
        harness reaches that copy. ``None`` spawns it with the parent's environment unchanged.

        ``redrive_available`` / ``allow_turn_timeout_redrive`` are the caller's RE-DRIVE policy, and
        REQUIRED: a turn that fails after the dialog started (the agent raised, its recursion budget,
        its provider, the per-turn wall — :func:`landed_reason`) LANDS — finalized and graded on the
        turns it reached — unless the caller will re-drive this attempt for that failure instead
        (:func:`~plurai.orchestrator.redrive.static_trigger`). Landing and re-driving are exclusive,
        and only the caller knows its budget and phase. A failure before the first answered turn, a
        harness fault, and anything on a single-hop replay still raise.

        ``on_setup_done`` is called once, when SETUP is over: ``POST /threads`` answered and the child
        finished its init (configure, session, reset, seed), before the first turn. A runner that limits
        how many scenarios may be in setup at once frees its slot there. It is not called when setup fails.

        ``on_finalize_start`` is called once, when the attempt submits its finalize job. A runner that stops the
        whole run learns here that this attempt is past its dialog, where a cancel would waste a finished measurement.

        ``replay_turn_only`` is the SINGLE-HOP regression replay (``--replay-turn-only``): the thread
        is opened with one turn, the scenario's opener (a regression cut's issue-turn user message) is
        delivered through the closing-turn call, the agent answers it — tool calls included — and the
        run stops there. The user simulator is never called (no ``/user-turn``), no turn cap is
        recorded, and ``/finalize`` is told ``ended_early=REPLAY_CUT_REASON`` so its end-of-run judges
        grade the partial rollout knowing it was cut. ``max_turns`` is ignored. A scenario with no
        opener is refused: a regression cut always lifts a user message.

        ``run_facts`` are facts about the RUN (``eval_mapping.RUN_FACT_KEYS``) this scenario's sink
        records carry beside the ones the orchestrator itself injects — a trial's identity.

        ``dataset_name`` is the durable scenario-set DATASET identity — the same name
        the dataset-upload path targets — handed to the sink's ``prepare_run`` so
        per-scenario tracing registers against the SAME provider dataset the upload
        creates. When omitted, falls back to the per-invocation experiment identity
        (the ``experiment`` metadata key) and then the scenario-set id.

        ``agent_version`` is the operator's REQUIRED agent-build tag (CLI
        ``--agent-version``, defaulting to ``git describe`` / ``"unknown"``): it is BOTH
        sent on the create-thread wire (so the server stamps it into
        ``scenario_metadata``) AND forwarded to the sink's ``prepare_run`` so observability
        records carry it. Never ``None``.
        """
        # A re-driven scenario calls run() again on this same orchestrator, so the
        # previous attempt's identity must not linger: clear before create, set the
        # instant create succeeds.
        self.thread_id = None
        self.mock_service_url = None
        self.turns_reached = []
        self.seeded_entities = 0
        self.last_finalize_job_id = None
        self.progress = None
        self._redrive_available = redrive_available
        self._allow_turn_timeout_redrive = allow_turn_timeout_redrive
        self._run_facts = dict(run_facts or {})
        self._on_setup_done = on_setup_done
        self._on_finalize_start = on_finalize_start
        self._on_child_closed = on_child_closed
        self.world_slot = world_slot
        # The bench's turn-input declaration, read ONCE for both of its readers: ``/threads`` (a
        # host that ALWAYS opens asks for an agent-opened thread) and the opener gate below.
        # A pipeline hop has no conversation to open, so it is never asked.
        declared_io = None if self.scenario_shape == PIPELINE_SHAPE else declared_agent_io(self.simulate_path)
        agent_io = DEFAULT_AGENT_IO if declared_io is None else declared_io
        agent_opens = agent_io.opens == OPENS_ALWAYS
        # `agent.io.choice_text`, copied onto every interrupt act this run ships (both the per-turn and
        # the closing path) so the server can tell an answer whose words reached the agent from one the
        # resume code dropped. ``None`` when undeclared: the acts cross the wire exactly as today.
        choice_text = agent_io.choice_text
        thread = await self.client.create_thread(
            scenario_set_id,
            scenario_id,
            eval_set_id=eval_set_id,
            agent_name=agent_name,
            agent_version=agent_version,
            max_turns=1 if replay_turn_only else max_turns,
            experiment_name=experiment_name,
            mock_service_url=mock_service_url,
            **({'agent_opens': True} if agent_opens else {}),
            # The run's declared shape, on every thread: a dialog takes the documents the user hands over as `file`
            # parts on her turns (the opener's included); a one-shot pipeline takes them as the hop's inputs.
            shape=self.scenario_shape,
        )
        # A live view of the documents the client fetched for this thread: one a user turn hands over is in it
        # as soon as ``user_turn`` returns, before the next hand-off to the agent folds it in (``_for_agent``).
        served_documents = self.client.served_documents(thread.thread_id)
        # Record the thread id HERE, not at finalize submit: every failure between
        # those two points — a child crash, an agent timeout, a raw httpx error —
        # would leave ``thread_id`` unset, so the runner would write
        # ``thread_id: "unknown"`` into error.json and never call /fail on a thread
        # that is still very much alive. It is also what tells the re-drive
        # classifier that create SUCCEEDED, a precondition of every trigger but
        # CREATE_TRANSIENT (create itself answering a gateway status).
        self.thread_id = thread.thread_id
        # Already resolved to absolute by the client's receive seam, so the readiness
        # gate can probe it directly.
        self.mock_service_url = thread.mock_service_url

        # Mock posture is a DECLARED policy (settings), never inferred from a failed
        # connect. The CHILD owns the mock end to end: in 'local' mode it hosts its own
        # in-process mock on a fresh loopback port IT picks itself (isolated per run, no
        # cross-contamination through a shared port, and where the registered @plurai.mock bodies
        # actually execute); in 'connect' mode it connects to the server-supplied
        # deployed URL (raising if unreachable). The orchestrator just forwards the
        # posture + that URL — it never allocates a port or configures a client.
        mock_mode = self.settings.network.mock_mode

        # The trace-resume seed, read ONCE here and used three ways: the child hands it to
        # the customer's seed_state, the turn-0 baseline is built from it, and it is mirrored
        # verbatim into the run artifacts. None for every fresh scenario — the presence of
        # the seed IS the resumed-mode discriminator, there is no parallel flag.
        resume_seed = _resume_seed_of(thread)

        # The two injected inputs, normalized ONCE — here, above every reader in this class, and
        # written back onto ``thread`` so that ``_scenario_scope``'s ``example_inputs`` (a
        # different method, which only ever receives the response object) reads the SAME dict
        # the child and the ThreadResult get instead of minting its own ``{}`` from a second
        # ``or {}``. That is what makes "the artifact records what the agent was given" true by
        # construction rather than by two reads happening to agree; the wire degrades both
        # fields to ``| None`` (models.py has no default_factory), which is what ``or {}`` is
        # for. Taken off ``thread`` AFTER ``_resolve_server_paths`` ran at the wire-receive
        # seam: a served document stays the reference the backend sent, and the agent's copy
        # is folded at the hand-off (``_context_for_agent``). The ``async with`` below is the ONLY call site of
        # ``_scenario_scope``, so this line always runs first.
        initial_world_state = thread.initial_world_state = thread.initial_world_state or {}
        initial_agent_context = thread.initial_agent_context = thread.initial_agent_context or {}
        seeded = initial_world_state.get('entities')
        self.seeded_entities = len(seeded) if isinstance(seeded, dict) else 0

        # Per-scenario observability lifecycle as a single contextlib scope: enter opens
        # the active sink (prepare_run -> run_ctx); exit GUARANTEES the per-scenario drain
        # on the normal path AND on any exception (child crash / HTTP error), so an
        # eagerly-opened sink span (e.g. the Braintrust root) is always ended. The
        # ``return`` lives INSIDE the scope so the drain runs before the result is handed
        # back.
        async with self._scenario_scope(
            thread,
            agent_name=agent_name,
            scenario_set_id=scenario_set_id,
            # ``scenario_id`` IS the backend scenario FOLDER (the runner enumerates
            # ``s.folder_name``) — the key every sink's dataset upload stamps, hence the
            # observability join key. Renamed at this seam so the identity claim is
            # readable; deliberately NOT the inner display name read below.
            scenario_folder=scenario_id,
            dataset_name=dataset_name,
            agent_version=agent_version,
        ) as run_ctx:
            # The scenario's channel plan, checked against the harness's ``agent.io.channels`` before any child
            # is up: a scenario that plants into a channel this harness never declared is the same mismatch as an
            # undeclared content part, and is refused the same way. The CHILD applies it (``plurai.child``).
            declared_by = f'{self.simulate_path} ({CONFIG_FILENAME})' if declared_io is None or declared_io.source is None else str(declared_io.source)
            channel_plan = parse_channel_plan(thread.scenario_metadata, declared_io, scenario_id=scenario_id, declared_by=declared_by)
            # THE DISPATCH. A pipeline bench is one structured call, not a
            # conversation: it has no ``invoke_agent`` for the loop below to drive, so driving it
            # anyway would die in ``_ChildSession.handle_user_turn`` ("Received user_turn but no
            # invoke_agent is configured") before the component was invoked once. Resolved from
            # the DECLARATION (``plurai.config.yaml``'s ``shape:``), never sniffed, with the AST
            # probe only as a HINT that names the key to add and never a route; ``dialog`` at
            # every other rung, so nothing about a dialog bench — which cannot reach this
            # branch — is touched.
            if self.scenario_shape == PIPELINE_SHAPE:
                # A hop has no conversation to resume into, so a seeded scenario is REFUSED here
                # rather than run without its seed. Dropping it silently would run the component against
                # a blank precondition AND write ``resume_seed_present: false`` into the record —
                # a false statement in the block that exists to say what was injected. The flag is
                # only honest because this refusal makes it true by construction.
                if resume_seed is not None:
                    raise OrchestratorAdapterError(
                        f'scenario {thread.thread_id} carries a resume_seed but this bench is declared `{PIPELINE_SHAPE}`: a one-shot hop '
                        f'has no conversation to resume into. Generate this scenario set without a resume seed, or declare the bench `{DIALOG_SHAPE}`.',
                        thread_id=thread.thread_id,
                    )
                return await self._run_pipeline_hop(
                    thread, scenario_id=scenario_id, run_ctx=run_ctx, initial_agent_context=initial_agent_context, mock_mode=mock_mode
                )
            # A turn's user content is a ``str`` OR a LIST of standard content parts, so this
            # carrier holds both and ``[-1]`` feeds ``child.user_turn`` either spelling. The opener
            # is a TRI-STATE and the order is the contract: the server's ``initial_messages`` (the
            # message list) when it sent one, else the ``initial_message`` projection string —
            # which is all a server without ``initial_messages`` sends.
            #
            # BOTH fields are codegen-defaulted to ``None``, so a server that omits them and one
            # that sends an explicit null read IDENTICALLY here. That is the contract for this hop
            # and it is why neither arm needs a presence test.
            opener: str | list[dict[str, Any]] | None = opener_content(thread.initial_messages) or thread.initial_message
            # A BLANK opener is the SAME FACT as no opener, because that is the
            # PREDICATE the backend resolves its own side with: "first_talk absent OR empty after
            # strip". Agreeing with the predicate rather than with any particular VALUE is the
            # right coupling — the SDK and the backend must classify the same scenario folder the same way,
            # or they disagree about which benches are agent-first.
            #
            # The arm belongs because the WIRE TYPE admits ``''`` from any producer that sends one.
            #
            # A ``str`` only. A LIST opener is a real turn whatever its text projects to — an
            # opener that is only a document carries no text and is still an opener — so it is
            # never blanked. And this is the OPENER's rule alone: on a LATER turn ``''`` stays a
            # user turn that carried no text, because there the user demonstrably spoke.
            if isinstance(opener, str) and not opener.strip():
                opener = None
            if agent_opens and opener is not None and resume_seed is None:
                # The server published a user opener on a thread this bench asked it to open with
                # the AGENT: a backend that does not implement ``agent_opens`` ignores the field. Driving it
                # would be the user-first run the declaration exists to prevent, silently. A RESUMED
                # thread is exempt: its conversation is under way, so its opener is the user's
                # continuation line and the server ignores ``agent_opens`` there by design.
                raise WorkspaceConfigError(
                    f'refusing to run scenario {scenario_id!r}: {self.simulate_path} declares `agent.io: {{{OPENS_FIELD}: {OPENS_ALWAYS}}}` '
                    f'in its {CONFIG_FILENAME}, but the server answered `agent_opens` with a user opener. The backend predates the '
                    f'field — upgrade the Plurai backend, or drop the declaration to run first_talk as turn 0.'
                )
            if replay_turn_only and opener is None:
                # A single-hop replay delivers the scenario's OPENER as the turn under test; with none
                # there is no turn to replay, and an agent-opened turn is not a regression's issue turn.
                raise WorkspaceConfigError(
                    f'refusing to replay scenario {scenario_id!r} turn-only: it has no opening user message. A single-hop '
                    f"replay delivers the regression cut's issue-turn user message — run this scenario without --replay-turn-only."
                )
            if opener is None and not agent_io.may_open:
                # THE SCENARIO-LEVEL REFUSAL. `ChildProcess.drive_turn` refuses the
                # same fact (`_subprocess.py`'s may_open gate) and must keep refusing it — it guards
                # the IPC boundary against every caller. But reaching it means the refusal arrives as
                # a turn-0 ADAPTER error, after the thread is open and the child is up, and an
                # operator reads that as their agent having crashed on the first turn. Both halves of
                # the condition are known HERE, the moment the opener is resolved and before any turn
                # is driven: this scenario has no opening user message, and this harness never
                # declared that it opens. Refuse as a workspace-configuration error, which is what it
                # is — a scenario set and a `plurai.config.yaml` that do not describe the same bench.
                #
                # Not before `/threads`, though that is where a scenario-level refusal would ideally
                # sit: whether the scenario HAS an opener is the server's answer, and it is `/threads`
                # that gives it. This is the earliest point the question can be asked.
                #
                # `declared_agent_io` is the SAME reader the child gates on, never a
                # second table: two readers of one declaration would eventually disagree about which
                # benches may open, and the disagreement would surface as this refusal firing on a
                # bench the child would have driven perfectly well.
                raise HarnessNoOpener(
                    f'refusing to run scenario {scenario_id!r}: it has no opening user message (the server sent neither '
                    f'`initial_messages` nor `initial_message`), '
                    f'but {self.simulate_path} does not declare `agent.io: {{may_open: true}}` in its {CONFIG_FILENAME}. '
                    f'An agent that opens the conversation is called with `{{"messages": []}}` on turn 0, which a harness that never '
                    f'declared it would raise on. Declare it (and make invoke_agent greet on an empty message list), or give the '
                    f'scenario a first_talk so the simulated user speaks first.'
                )
            # A missing opener is a POSITIVE statement — the scenario has no user first line and
            # the AGENT opens (the declared ``agent.io.may_open`` shape).
            #
            # So the opener is the list's first element ONLY WHEN THERE IS ONE, and nothing is ever
            # coerced in to stand for its absence. Every reader downstream — the ``/finalize``
            # payload, ``transcript.json``, the dataset Example — COUNTS what is in this list, so a
            # placeholder here would be a phantom user turn as far as all of them are concerned.
            #
            # Note the asymmetry with the rest of the list, which is deliberate: the OPENER is
            # normalized above (blank means absent, the predicate the backend shares), while a LATER
            # ``''`` appended by ``_record_turn`` stays a user turn that carried no text — there the
            # user demonstrably spoke.
            # A user line that hands the agent a part kind the harness never declared is refused HERE, at the
            # earliest point the line is known: the child would refuse it too, but as a turn error the runner
            # records as an empty agent turn and flags degraded. A pipeline hop takes no user parts.
            check_user_parts = self.scenario_shape != PIPELINE_SHAPE
            if check_user_parts:
                _refuse_undeclared_input(opener, declared_io, scenario_id=scenario_id, simulate_path=self.simulate_path, turn='opening message')
            user_messages: list[str | list[dict[str, Any]]] = [] if opener is None else [opener]
            # The user's typed decision on the interrupt the NEXT agent turn answers (the backend's
            # ``UserTurnResponse.interrupt_choice``); ``None`` until a /user-turn carries one.
            interrupt_choice: str | None = None
            # The act the simulated user took with her last reply (``UserTurnResponse.user_action``),
            # applied by the child at the open of the NEXT agent turn; ``None`` when she took none.
            pending_user_action: dict[str, Any] | None = None
            chatbot_messages: list[str] = []
            # The seed's conversation, serialized once and shared by the two records below.
            # ``[]`` for a fresh scenario, so nothing here changes a fresh run.
            seed_baseline = _seed_baseline(resume_seed, thread_id=thread.thread_id)
            # The AGENT channel: the flat chain persisted as transcript.json's
            # ``chatbot_messages`` and shipped as the per-turn narration. Single-role by
            # construction — see ``split_user_echo`` — which is why a RESUMED scenario opens it
            # with the AGENT SIDE of the seed and not the seed itself: the seeded user's words
            # are as misplaced in a channel judges read as the agent's as a live echo is, and
            # the seed is persisted whole under ``resume_seed`` either way, so nothing is lost
            # by keeping this channel to one role. ``input_user_message=''`` disables the
            # verbatim-echo rule, which has nothing to say about history: the seed is not a
            # reply to anything this harness sent, so only the declared-role rule can apply.
            chatbot_full_messages: list[dict[str, Any]] = split_user_echo(seed_baseline, input_user_message='')[0]
            # The UNFILTERED accumulation, kept only as ``_fresh_turn_tail``'s prefix baseline
            # (a cumulative agent re-lists what it returned, echoes and all, so the record it is
            # matched against has to be what it actually returned). This is therefore the record
            # a resumed scenario has to seed, and seeding it is the whole of turn-0 excision: a
            # resumed agent re-lists the seeded conversation in the chain it returns for turn 1,
            # and only a prefix match against THIS list strips it. Seeding the agent channel
            # instead would leave the excision broken while every assertion about the transcript
            # still passed. Note what does NOT happen here: the baseline is never fed to
            # ``derive_tool_calls_from_messages`` and never reaches ``_record_turn``, so per-turn
            # attribution and recovered tool calls stay LIVE-only — a resumed tool call is
            # history, not a call this run made.
            raw_full_messages: list[dict[str, Any]] = list(seed_baseline)
            # Is the SDK's reading of the seed restore still open? Armed for every resumed thread
            # and closed by the first CONCLUSIVE reading (a confirmed full restore, or an
            # evidence-grade partial one) — not by the first reply. A chain that says nothing
            # usable leaves it armed, so a restore fault visible only on a later turn is still
            # read, and a run that never resolves it ships the explicit "could not be determined"
            # rather than the silence the wire reads as a full restore. Either call site can
            # close it: the turn loop's, or the closing exchange's on a thread whose loop drove
            # no turn.
            seed_check_pending = bool(seed_baseline)
            # What the split took OUT of the agent channel, so the run can say so once at the end
            # instead of per turn. Kept in TWO lists because the rules describe different adapter
            # health: a declared user role is the normal LangGraph shape, a role-less verbatim
            # echo is the contract breach (see ``split_user_echo``). Seeded history contributes to
            # neither — these report what the ADAPTER returned, and the seed is not its output.
            role_declared_echoes: list[dict[str, Any]] = []
            verbatim_echoes: list[dict[str, Any]] = []
            turns: list[TranscriptTurn] = []
            # Tool-call ids already attributed to an earlier turn's tree. A LangGraph
            # MemorySaver agent returns the CUMULATIVE message history every turn, so
            # ``full_messages`` re-lists prior turns' tool calls; deduping by call id
            # pins each call to the turn that ISSUED it (first appearance) instead of
            # re-rendering it on every later turn. Harmless for stateless agents (no
            # repeats to drop).
            seen_tool_call_ids: set[str] = set()
            # The same rule for model calls, by assistant-message id: a re-listed message is
            # context for this turn's calls, not a call of its own (its usage would count twice).
            seen_model_message_ids: set[str] = set()
            # ``mode:real`` tool calls recovered from the MESSAGE lane, PER TURN so each record
            # carries the turn that issued it. Recovering once over the whole transcript at
            # finalize would produce the same records with no turn at all, leaving the finalize
            # merge nothing to order them by — they would land after every mock event regardless
            # of when they actually fired. The trail lane needs no accumulator: it is read off
            # ``transcript_turns`` at the merge, where the list position already IS the turn.
            recovered_message_calls: list[RecoveredToolCall] = []
            # The thread's unresolved calls by id, and the interrupt contract they wait on when the
            # last turn paused (``None`` otherwise): a call and its result straddle a turn exactly
            # when the agent paused for approval, so pairing is thread-scoped
            # (:func:`derive_turn_tool_calls`). Never seeded from the resume baseline — see above.
            open_calls = ThreadCalls()
            paused_by: dict[str, Any] | None = None
            # Turn-ordered, role-tagged transcript records (additive — the persisted transcript.json
            # also keeps the flat user_messages/chatbot_messages arrays). Each entry pairs a turn's
            # user prompt with the agent reply + that turn's message chain, so the two-positional-
            # array off-by-one does not apply and tool calls are attributable per turn.
            transcript_turns: list[dict[str, Any]] = self.turns_reached
            # The typed reason this dialog ended for something other than the user reaching
            # their goal — reported by the server, or recorded by this loop below. ``None`` ⇒
            # nothing recorded one, which is the ordinary case.
            ended_early: str | None = None
            # The simulated user's stated cause for the reason above, when the server reported
            # one. Bound beside the reason so the two are set — and displaced — together.
            ended_early_cause: str | None = None
            # The parent-emptied turns and the count of turns driven, read off the child before
            # its scope closes. Bound here so the all-degraded check below cannot depend on how
            # far the child block got.
            degraded_turns: list[dict[str, Any]] = []
            ipc_trimmed_turns: list[dict[str, Any]] = []
            turns_driven = 0

            async with (
                ChildProcess(
                    self.simulate_path,
                    agent_turn_timeout=self.settings.network.agent_turn_timeout,
                    env=self._child_env(),
                ) as child,
                contextlib.AsyncExitStack() as on_raise,
            ):
                resume_fidelity = await child.init(
                    thread_id=thread.thread_id,
                    mock_service_url=thread.mock_service_url,
                    mock_mode=mock_mode,
                    initial_world_state=initial_world_state,
                    initial_agent_context=_context_for_agent(initial_agent_context, served_documents),
                    scenario_metadata=thread.scenario_metadata or {},
                    resume_seed=_seed_for_agent(resume_seed, served_documents),
                )
                # THE SEED READ-BACK, at Init and before any turn: did the world the child reads back hold the seed it
                # was handed? When it did not (``voids_scenario``), the backend voids the scenario at finalize on this same
                # pair (``undelivered_seed``), so nothing is driven: no opener, no turn, no closing turn. The scenario
                # is finalized with zero turns and the backend records the void; every other scenario and trial runs.
                seed_not_landed = seed_readback_at_init(scenario_id, getattr(child, 'post_seed_world_state', None), initial_world_state).voids_scenario
                if seed_not_landed:
                    # Nothing reached the agent, the opener included: the transcript must not claim a user turn was sent.
                    user_messages.clear()
                self._setup_done()
                if channel_plan and not getattr(child, 'applies_channel_inputs', False):
                    raise OrchestratorAdapterError(
                        f'scenario {scenario_id!r} plants channel inputs (`turn_inputs` / `between_turn_actions`), but the agent process '
                        f'runs a plurai SDK that does not apply them; update plurai in the agent environment.',
                        thread_id=thread.thread_id,
                    )
                # The accumulators above, by reference, for a runner that has to write evidence
                # after this run raised. Published only once the child is up and seeded: before
                # that no turn reached the agent, and ``user_messages`` already holds an opener it
                # never received, which a replay check would read as a turn that was sent.
                progress = self.progress = ScenarioProgress(
                    thread_id=thread.thread_id,
                    initial_world_state=initial_world_state,
                    initial_agent_context=initial_agent_context,
                    post_seed_world_state=getattr(child, 'post_seed_world_state', None),
                    resume_seed=resume_seed,
                    user_messages=user_messages,
                    chatbot_messages=chatbot_messages,
                    chatbot_full_messages=chatbot_full_messages,
                    transcript_turns=transcript_turns,
                    recovered_message_calls=recovered_message_calls,
                )
                # A raise out of this scope with the child still alive reads the mock's log once
                # before the child closes, so the crash evidence carries the calls the agent made.
                # Exits before ``ChildProcess`` (entered after it), so the child is still up.
                on_raise.push_async_exit(
                    self._read_mock_log_on_raise(child, progress, thread_id=thread.thread_id, mock_service_url=thread.mock_service_url, mock_mode=mock_mode)
                )

                # The SOFT per-scenario budget's clock. Started after create + child init, so it
                # measures the CONVERSATION rather than the setup around it, and read only
                # between turns: a budget that interrupted a turn in flight would throw away the
                # reply that turn already paid for, and the hard ceiling is what exists for a
                # turn that never returns.
                dialog_clock = time.monotonic()
                # Per-ATTEMPT override, else the configured budget, else the full-run default.
                # The override exists for the turn-cap re-run, which doubles the turn allowance:
                # a re-run held to the ORIGINAL clock would be cut at the same wall-clock second
                # as the attempt it is meant to extend, converting ``turn_cap_exhausted`` into
                # ``time_budget_exhausted`` and quietly disabling the re-drive. The final
                # fallback keeps a directly-constructed Orchestrator bounded rather than
                # unbounded, since the CLI is not the only thing that fills this in.
                time_budget = scenario_time_budget or self.settings.network.scenario_timeout or FULL_RUN_SCENARIO_TIME_BUDGET
                # The LAST /user-turn's user line, or ``None`` when that response carried no user
                # turn (``user_turn_content``). Read after the loop by the closing-turn block, so
                # it is hoisted here rather than derived from ``turn``: on the ``max_turns == 0``
                # path the loop body never runs and ``turn`` is not bound at all.
                pending_user_line: str | list[dict[str, Any]] | None = None
                # THE CLOSING CALL IS THE NEXT ITERATION'S, while turns remain after a stop the USER
                # took (see the stop below). A closing reply that opens a DECISION — an ``interrupt``
                # nobody would answer — is then an ordinary turn: it goes to /user-turn, the
                # simulator answers it with a typed choice, and the agent's result is the next reply.
                # Any other closing reply leaves the loop here, drawn, for the closing block.
                closing_pending = False
                closing_reply: tuple[Any, str, float] | None = None
                closing_raised: Exception | None = None
                # The turn failure the dialog LANDS on (``ThreadResult.turn_failure``), and whether
                # that failure took the child with it (the per-turn wall reaps it): a reaped child
                # cannot be asked to finalize, so its snapshots ship unread.
                turn_failure: dict[str, Any] | None = None
                child_reaped = False
                # A single-hop replay drives NO /user-turn: its one turn is the closing call below. A scenario whose seed
                # did not land drives none either, nor the closing call.
                for turn_idx in range(0 if replay_turn_only or seed_not_landed else thread.max_turns):
                    # Never on the closing call: the closing turn is delivered whatever the clock says.
                    if turn_idx and not closing_pending and (spent := time.monotonic() - dialog_clock) >= time_budget:
                        # ``turn_idx`` guard: the budget may never cut a dialog before its FIRST
                        # turn. A zero-turn transcript is not a short conversation, it is no
                        # conversation, and shipping one to the judges would score the agent for
                        # a dialog it was never given.
                        # ``or``, not an assignment: a reason the SIMULATOR reported on the last
                        # turn is more specific than "the clock ran out", the same way the turn
                        # cap's own floor below defers to one.
                        ended_early = ended_early or TIME_BUDGET_REASON
                        logging.getLogger(__name__).warning(
                            'thread %s: the declared per-scenario time budget (%.0fs) was spent after %d turn(s) (%.0fs) with the user '
                            'simulator still going. The dialog is ENDED here and still finalized, so this is a scored outcome — anything '
                            'it never reached scores as unmet. Raise --scenario-timeout before reading this as an agent failure.',
                            thread.thread_id,
                            time_budget,
                            turn_idx,
                            spent,
                        )
                        break
                    # An agent-first run has NO user message for turn 0, so the carrier is empty and
                    # there is no ``[-1]`` to take. ``None`` is what the child's own contract spells
                    # for it (`_subprocess.py`'s ``may_open`` gate): it opens, and refuses by name on
                    # a harness that declares no ``agent.io: {may_open: true}``. Indexing by ROLE —
                    # is there a user turn? — rather than by position.
                    input_user_message = user_messages[-1] if user_messages else None
                    # The agent-form input is computed once and is what the agent receives, what
                    # the echo rule compares against, and what the seed check treats as sent.
                    agent_input = _for_agent(input_user_message, served_documents)
                    reference_of = _reference_by_fold(served_documents)
                    turn_started_at = _now()
                    agent_clock = time.monotonic()
                    # The agent runs in the child, driven over IPC; the child also
                    # surfaces the rich per-turn message chains.
                    try:
                        agent_response = await child.user_turn(
                            turn_idx, agent_input, interrupt_choice=interrupt_choice, **_user_action_kwarg(pending_user_action)
                        )
                    except Exception as exc:
                        if closing_pending:
                            # The closing block owns what a failed closing call means.
                            closing_raised = exc
                            break
                        # LAND the dialog on a failure its owner caused after it started — the agent,
                        # its provider, the per-turn wall — instead of throwing away every turn it
                        # reached. Never before the first answered turn (a zero-turn transcript is no
                        # conversation, the rule the time budget keeps), never for a harness fault,
                        # and never when the caller will re-drive this attempt instead.
                        reason = self._landing_reason(exc)
                        if reason is None or not transcript_turns:
                            raise
                        turn_failure = _turn_failure_record(exc, turn_idx=turn_idx, closing=False)
                        ended_early = ended_early or reason
                        child_reaped = isinstance(exc, AgentTurnTimeout)
                        # The act she took with her last message was applied (or refused) before the
                        # agent failed: it is still reported, never left for the server to guess.
                        if (raised := _raised_turn_user_action_result(child)) is not None:
                            _file_user_action_result(transcript_turns, raised)
                        logging.getLogger(__name__).warning(
                            'thread %s: turn %d failed (%s: %s) after the dialog started; it LANDS — finalized and graded on the %d '
                            'turn(s) it reached, with ended_early=%r.',
                            thread.thread_id,
                            turn_idx,
                            type(exc).__name__,
                            exc,
                            len(transcript_turns),
                            reason,
                        )
                        break
                    agent_seconds = round(time.monotonic() - agent_clock, 3)
                    if closing_pending:
                        closing_pending = False
                        closing_acts = _agent_actions_of(agent_response)
                        closing_acts = _with_choice_text(closing_acts, choice_text)
                        if not _opens_a_decision(closing_acts) or time.monotonic() - dialog_clock >= time_budget:
                            closing_reply = (agent_response, turn_started_at, agent_seconds)
                            break
                        logging.getLogger(__name__).info(
                            'thread %s: the closing reply opened a decision the user has not answered, and turns remain — it is '
                            'recorded as an ordinary turn and the conversation continues so the user can answer it.',
                            thread.thread_id,
                        )
                    ai_text = agent_response.ai_text
                    chatbot_messages.append(ai_text)
                    if seed_check_pending:
                        resume_fidelity, seed_check_pending = _note_seed_restore(
                            seed_baseline,
                            agent_response.full_messages or [],
                            fidelity=resume_fidelity,
                            thread_id=thread.thread_id,
                            sent_user_messages=frozenset(content_text(_for_agent(m, served_documents)) for m in user_messages),
                        )
                    turn_messages, declared_echoes, verbatim = _accept_agent_chain(
                        agent_response.full_messages, sent=agent_input, raw_seen=raw_full_messages, seeded=len(seed_baseline), thread_id=thread.thread_id
                    )
                    echo_reference_of = _echo_references(reference_of, sent=input_user_message, served=served_documents)
                    role_declared_echoes.extend(_as_recorded(declared_echoes, echo_reference_of))
                    verbatim_echoes.extend(_as_recorded(verbatim, echo_reference_of))
                    chatbot_full_messages.extend(turn_messages)
                    turn_calls = derive_turn_tool_calls(turn_messages, turn=turn_idx, open_calls=open_calls, paused_by=paused_by, choice=interrupt_choice)
                    recovered_message_calls.extend(turn_calls)
                    paused_by = _paused_by(agent_response, choice_text=choice_text)

                    # ONE value, read once, for BOTH of this turn's carriers — the wire below and
                    # the saved row further down. The closing block does the same, for the reason
                    # stated there: two reads of the same turn are two things that can be changed
                    # apart, and the artifact and the server then disagree about one turn.
                    turn_agent_actions = _agent_actions_of(agent_response)
                    turn_agent_actions = _with_choice_text(turn_agent_actions, choice_text)
                    turn_agent_state = _wire_state_of(agent_response.agent_state or None, reference_of, thread_id=thread.thread_id)
                    turn_world, action_result = _user_world_of(agent_response)
                    turn = await self._record_turn(
                        thread,
                        turn_idx=turn_idx,
                        input_user_message=input_user_message,
                        ai_text=ai_text,
                        agent_messages=turn_messages,
                        turn_calls=turn_calls,
                        agent_state=turn_agent_state,
                        reply_source=_reply_source_of(agent_response),
                        agent_actions=turn_agent_actions,
                        turn_world=turn_world,
                        user_action_result=action_result,
                        seen_tool_call_ids=seen_tool_call_ids,
                        seen_model_message_ids=seen_model_message_ids,
                        user_messages=user_messages,
                        turns=turns,
                    )
                    # ONE reading of this response's user side, for both readers below — the row
                    # this iteration writes and, after the loop, the closing-turn decision. Read
                    # HERE rather than in each, for the same reason ``turn_agent_actions`` is: two
                    # reads of one turn are two things that can be changed apart.
                    pending_user_line = user_turn_content(turn)
                    # Checked on arrival rather than at the call that delivers it: the last line of the dialog is
                    # delivered by the closing block, which degrades a failure to "no closing reply".
                    if check_user_parts:
                        _refuse_undeclared_input(
                            pending_user_line, declared_io, scenario_id=scenario_id, simulate_path=self.simulate_path, turn=f'user line of turn {turn_idx + 1}'
                        )
                    interrupt_choice = turn.interrupt_choice
                    pending_user_action = None if turn.user_action is None else turn.user_action.model_dump(mode='json')
                    should_terminate = turn.should_terminate
                    if turn.ended_early:
                        # REPORTED, not inferred: the backend names the reason only where it actually
                        # ended the dialog, past its own veto rules. Taken verbatim
                        # even if this build does not recognise the value — the vocabulary is
                        # shared and will grow, and an unknown reason still beats no reason.
                        #
                        # This is also the only route by which a turn-cap reason survives: the
                        # ``for``/``else`` below cannot fire against a live server, which forces
                        # should_terminate=True once turn_count reaches max_turns — so the loop
                        # always BREAKS on the last budgeted turn and the else-branch is skipped.
                        ended_early = turn.ended_early
                        ended_early_cause = turn.ended_early_cause
                    # Additive turn-ordered record: the prompt, the agent reply + its FULL per-turn
                    # message chain (tool-call AIMessages + ToolMessages — so tool calls attribute to
                    # THIS turn), and the next user message _record_turn just appended.
                    transcript_turns.append(
                        {
                            'turn_idx': turn_idx,
                            # The STRING is the projection: every reader of a committed transcript
                            # keeps working unchanged.
                            # The message list rides BESIDE it, and only when the turn actually
                            # carried one — so a text-only run's transcript carries no extra key.
                            # ``None`` STAYS ``None`` here. ``content_text(None)`` is
                            # ``''``, and an empty string in this row is a user turn that said
                            # nothing — the phantom, in the artifact every audit re-derives from.
                            'user_message': None if input_user_message is None else content_text(input_user_message),
                            **({'user_messages': input_user_message} if isinstance(input_user_message, list) else {}),
                            'agent_message': ai_text,
                            # ``turn_messages`` is the turn's NEW CHAIN, agent side only: the
                            # single-role split (``split_user_echo``) takes the user's own messages
                            # out, and this row carries them once, under ``user_message``. Named a
                            # chain because it is narration + tool calls + tool results + reply,
                            # not the bare reply. The row carries it under this one name and never
                            # as ``agent_messages``: the backend's eval namespace binds that name to the
                            # agent's replies alone, role-filtered, and every eval built on it
                            # (``len(agent_messages)`` turn counts, URL extraction) assumes so. A
                            # reader who took the two for one thing would count every narration
                            # step, tool call and tool result here as a reply.
                            'turn_messages': turn_messages,
                            # PROJECTED, like ``user_message`` eight lines up and for the same rule —
                            # this key is typed ``str`` (``sink_protocol.TranscriptTurn``), so the
                            # SAME message reads as a ``str`` here and on turn N+1's row. Nothing is
                            # lost: turn N+1's row carries the list itself, beside its own
                            # projection.
                            #
                            # THIS TURN'S OWN reply, read off ``pending_user_line`` and NOT off
                            # ``user_messages[-1]``. The two are the same value when every
                            # response appends one, but on a response that carries no user turn
                            # nothing is appended, so ``[-1]`` would silently be the PREVIOUS
                            # turn's line and this row would claim the simulator repeated itself.
                            # ``content_text(None)`` is ``''``, so the row is empty on that turn.
                            'output_user_message': content_text(pending_user_line),
                            'should_terminate': should_terminate,
                            # The agent's NON-message return state for this turn — which model,
                            # which retrieved ids, which suggestions. Persisted here so a finished
                            # run can be re-examined for what the agent actually returned, rather
                            # than needing a hand-written side manifest to answer it.
                            # SCOPE: the state is not transcript-only.
                            # It ALSO rides the wire — per-turn state goes out on /user-turn (see
                            # _record_turn), where the server files it under THAT turn's 0-based
                            # index and accumulates it across the thread, so a turn_based judge
                            # sees only its own turn's state and re-scoring this artifact's
                            # ``turns[].agent_state`` grades identically.
                            # The CLOSING turn's state rides /finalize as ``closing_agent_state``
                            # instead, because that reply never passes through /user-turn.
                            # Nothing here can fail a turn: child._extract_agent_state already
                            # JSON-round-trips the dict per key and drops an unserializable key
                            # with a warning, so what reaches this line is serializable by
                            # construction. Empty state ⇒ the field is omitted, on the wire and here.
                            # The SAME projected value /user-turn carried (``_wire_state_of``): a
                            # served document the agent echoes is its reference, and a leaf over the
                            # wire cap is a descriptor; every other value is what the customer's
                            # simulate.py returned. No further filter here, so this file and the
                            # server record stay one record, and a redaction allowlist here would
                            # silently drop the fields an operator is reading this to see.
                            # ALWAYS written, empty included: the child returns a dict for every
                            # captured turn, so an absent key means the state was not captured
                            # at all, while ``{}`` means it captured and the agent returned none.
                            'agent_state': turn_agent_state or {},
                            # WHAT the agent DID this turn — the same typed list ``/user-turn``
                            # carried two screens up, from the SAME local, so the artifact and the
                            # server cannot describe one turn two ways. Persisted because the backend's
                            # re-score reads this key off the row and reads absent as ``[]``, and
                            # three judge-facing things are built from what it reads — all three
                            # silently, with no refusal and no warning. (1) The judge-facing and
                            # user-facing RENDERS and the end-to-end judge transcript
                            # lose every action the narration CHAIN
                            # does not already carry, and an ``interrupt`` is the whole question a
                            # turn stopped on — so the LLM judges and the persona re-read a
                            # conversation the live run never had. (2) The ``agent_actions_by_turn``
                            # and per-turn ``agent_actions`` keys the re-score publishes come back
                            # empty for a turn that acted. (3) The re-score builds NO judge
                            # transcript at all for a run whose answers were RENDERED rather than
                            # narrated — with no chain, this key is the only thing that opens it.
                            # What this is NOT is a code_eval name: ``agent_actions`` is bound in
                            # neither sandbox scope, so an expression naming it ERRORS rather than
                            # passing quietly, and the damage is to what the judges READ.
                            # Written ONLY when the turn carried actions — the rule
                            # ``user_messages`` and ``reply_source`` follow in this literal, for
                            # the same stated reason: a run whose agent acted on no turn writes
                            # no such key. The wire's tri-state
                            # does NOT survive here and is not meant to: ``[]`` ("observed, and
                            # none") and absent collapse to ``[]`` in the one reader, whose own
                            # docstring declares that collapse deliberate, and the action channel
                            # has no re-score twin of the tool channel's UNSUPPLIED marker to give
                            # the third state a consumer.
                            # ``_agent_actions_of`` ADMITS without NORMALIZING, while the wire
                            # projects through ``AgentAction`` with ``exclude_none``. For the
                            # SDK's own child the two are identical; for a ``child_factory``
                            # extension emitting ``status: None`` or an extra key, this copy is
                            # strictly WIDER than the wire's. A second projection here would
                            # duplicate the one admission gate both carriers read.
                            **({'agent_actions': turn_agent_actions} if turn_agent_actions else {}),
                            # WHEN this turn ran and how long the AGENT took inside it, so
                            # "which turn was the slow one" is a question a finished run can
                            # answer without falling back to the file's mtime. The agent's
                            # share is separated from the cycle's because the rest of the wall
                            # is the user simulator — a different system, a different remedy.
                            'started_at': turn_started_at,
                            'ended_at': _now(),
                            'agent_seconds': agent_seconds,
                            # WHERE ``agent_message`` came from — the agent's own words, a tool
                            # result a graph ended on, a typed interrupt, an empty turn, or a
                            # suffix the classifier could not read. Carried onto the saved
                            # transcript because that file is what a re-score and every audit
                            # read, and there a raw tool payload is otherwise indistinguishable
                            # from an answer. Omitted when the child did not classify.
                            **({'reply_source': source} if (source := _reply_source_of(agent_response)) else {}),
                            # What the simulated user could see after this turn, and the act she took
                            # with her reply — as they went to /user-turn, so a re-score re-folds and
                            # re-projects exactly what the live run showed her. Written only when set.
                            **_saved_turn_world(turn_world),
                            **({'user_action': pending_user_action} if pending_user_action is not None else {}),
                            # What the scenario's channel plan planted at this turn's open, with how each went.
                            **_saved_channel_inputs(agent_response),
                        }
                    )
                    _file_user_action_result(transcript_turns, action_result)
                    if should_terminate:
                        # A stop the USER took (nothing recorded a reason) with a turn left: the
                        # closing call is the next iteration's, so a decision it opens can be answered.
                        if ended_early is None and pending_user_line is not None and turn_idx + 1 < thread.max_turns:
                            closing_pending = True
                            continue
                        break
                else:
                    if seed_not_landed:
                        # Not driven: the seed did not read back (above). Its own reason, never the turn cap's, so the
                        # turn-cap re-run does not re-drive it; local only, as the backend reaches the void on its own.
                        ended_early = SEED_NOT_LANDED_REASON
                    elif replay_turn_only:
                        # The loop drove nothing BY DESIGN: the replay's one turn is the closing call
                        # below, and the run ends after it. Not a turn cap — that reason is what the
                        # turn-cap re-run keys on, and nothing here ran out.
                        ended_early = REPLAY_CUT_REASON
                    else:
                        # ``for``/``else``: no ``break`` fired, so the loop ran out of TURNS while the
                        # user simulator was still asking for more. Recording that is what stops a
                        # budget artifact from reading as an agent that stopped short.
                        #
                        # UNREACHABLE against the Plurai backend, and kept anyway. That server forces
                        # ``should_terminate=True`` once ``turn_count`` reaches ``max_turns``, so the
                        # loop always BREAKS on the last budgeted turn and this branch is skipped —
                        # which is why the cap reason arrives REPORTED, on the /user-turn response,
                        # and why the same value is used on both sides. This stays as the
                        # floor for a server that returns no reason and a client whose loop genuinely
                        # runs out: it can only ever ADD a reason where there would be none.
                        ended_early = ended_early or TURN_CAP_REASON
                        logging.getLogger(__name__).warning(
                            'thread %s: the conversation hit its turn cap (%d) with the user simulator still going. '
                            'Anything its dialog never reached scores as unmet — raise --max-turns before reading this '
                            'as an agent failure.',
                            thread.thread_id,
                            thread.max_turns,
                        )

                # The /user-turn CYCLE count is frozen HERE, before the closing
                # exchange appends to ``turns`` below: ``ThreadResult.turns`` must
                # keep meaning "outer /user-turn cycles" (transcript.json pins it
                # as ``user_turn_cycles``) while the observability ``turns`` list
                # gains the closing entry — decoupling the two counts.
                user_turn_cycles = len(turns)

                # Closing-turn delivery: the loop always exits with the LAST user
                # message unanswered — the /user-turn that produced it also said
                # should_terminate. Dropping that exchange makes any behavior
                # observable only in the reply to a closing turn (no next-step
                # nudge after a goodbye, a final memory-probe answer) structurally
                # untestable. So deliver it, record the reply locally, and ship it
                # to the server via FinalizeRequest.closing_agent_response so the
                # e2e judges grade a transcript that doesn't end mid-exchange.
                # Best-effort: a failure here must not kill an otherwise-complete
                # scenario, which then finalizes with no closing reply.
                closing_agent_response: str | None = None
                # The closing reply's non-message state. It rides /finalize (not
                # /user-turn) for the same reason ``closing_agent_response`` does:
                # this turn never makes a /user-turn round trip, so it is the ONE
                # agent turn whose state has no per-turn wire to ride.
                closing_agent_state: dict[str, Any] | None = None
                # The closing reply's narration chain — rides /finalize for the same
                # no-other-way-in reason as its state.
                closing_turn_messages: list[dict[str, Any]] | None = None
                # ...and the closing turn's two TYPED channels, for the same no-other-way-in
                # reason again. The orchestrator computes both for the ARTIFACT, and
                # ``FinalizeRequest`` carries both; leaving them ``null`` on the wire costs what
                # the fields' own docstring names: a turn whose answer is a render verb would close the dialog with an
                # empty ``closing_agent_response`` and be graded as the agent's silence, and a turn
                # the RUNNER emptied could not be told from one the agent left silent. ONE value each,
                # read once and used for BOTH surfaces, so the artifact and the wire cannot
                # disagree about the same turn — the rule the DRIVEN turn's record follows
                # too (``turn_agent_actions``, hoisted above its own ``_record_turn`` call): both
                # surfaces of both turn shapes read one local, never the reader twice.
                closing_agent_actions: list[dict[str, Any]] | None = None
                closing_reply_source: str | None = None
                # How the act the dialog ended on was applied — by the closing turn, which has no
                # /user-turn of its own, so it rides /finalize.
                closing_user_action_result: dict[str, Any] | None = None
                # ...unless the AGENT is the one that ended it. On an `end_conversation` /
                # `handoff` the other end of this dialog is GONE, so the "unanswered" last user
                # message is a line into a session the agent already left: delivering it would
                # measure a conversation that was over, and ship that reply to /finalize as the
                # transcript's closing exchange for every e2e judge to grade. Keyed on the two
                # AGENT-side reasons and not on `ended_early` being set, because a give-up, a turn
                # cap and a spent time budget all leave an agent that is still there to answer.
                #
                # FIRST, and the ordering is what the operator READS. Both arms below are true
                # at once on an agent close — ``AGENT_CLOSED_REASONS`` is in
                # ``NO_USER_LINE_REASONS``, so ``pending_user_line`` is ``None`` there too — and
                # whichever comes first is the sentence the log carries. The no-line arm's sentence
                # ends "the harness is what broke", which on an agent close is false and reads as
                # an instruction to go fix the apparatus. This one names what actually happened, so
                # it is the more specific fact and it goes first.
                if seed_not_landed:
                    # FIRST: nothing was driven, so there is no line to close. Without this arm the zero-turn path below
                    # would hand the agent the opener as a "closing turn".
                    logging.getLogger(__name__).info(
                        'thread %s: not driven (its seed did not read back at Init), so no closing turn was delivered.', thread.thread_id
                    )
                elif turn_failure is not None:
                    # The dialog LANDED on a failed turn: the agent (or its provider, or the wall) is
                    # what stopped answering, so there is no closing reply to ask for — and after the
                    # per-turn wall there is no child left to ask.
                    logging.getLogger(__name__).info(
                        'thread %s: the dialog landed on a failed turn (%s), so no closing turn was delivered.',
                        thread.thread_id,
                        ended_early,
                    )
                elif ended_early in AGENT_CLOSED_REASONS:
                    logging.getLogger(__name__).info(
                        'thread %s: the assistant ended this conversation (%s), so no closing turn was delivered. '
                        'The transcript ends where the agent left it; obligations it never reached score as unmet.',
                        thread.thread_id,
                        ended_early,
                    )
                # ...and the OTHER way there is nothing to close: NO LAST USER MESSAGE to answer.
                # The closing turn exists to give the dialog's final user line a reply; when the
                # last response carried no user turn at all (``NO_USER_LINE_REASONS``), there is no
                # line, and every value this block would build is about a turn that did not happen:
                # an agent call on the EMPTY string, a phantom ``HumanMessage('')`` in the graded
                # transcript, and a ``closing_agent_response`` every end-to-end judge reads as how
                # the conversation ended.
                #
                # SECOND, and what reaches it is exactly what the arm above does not claim: the
                # SERVER's simulator produced nothing while the agent was still there to answer. It
                # is the only shape whose disclosure is a harness finding.
                #
                # ``and turns`` is load-bearing. On the ``max_turns == 0`` path the loop never ran,
                # ``pending_user_line`` is still its hoisted ``None``, and the documented
                # degradation for that shape is the ``IndexError`` the ``else`` arm takes into its
                # own ``except``. This guard keeps that shape there rather than quietly
                # re-labelling it.
                elif pending_user_line is None and turns:
                    logging.getLogger(__name__).warning(
                        'thread %s: the last /user-turn carried no user message (%s), so no closing turn was delivered. '
                        'There is no line to answer: a reply to one would be the agent responding to nothing, graded as '
                        "this dialog's ending. Nothing here is a verdict about the agent — the harness is what broke.",
                        thread.thread_id,
                        ended_early,
                    )
                else:
                    # On the ``user_messages[-1]`` reads in this block and the one in the turn loop's
                    # record: they are NOT guarded, and that is deliberate rather than an oversight
                    # the by-role comment 200 lines above should be read as covering.
                    # Reaching here means a turn was driven AND ``_record_turn`` appended the next
                    # user message, so the carrier is non-empty by construction. The one shape that
                    # escapes that is ``max_turns == 0`` on an agent-first thread — the loop never
                    # runs, nothing is appended, and ``[-1]`` raises into this ``except``, which
                    # degrades to "closing-turn delivery failed" with no closing reply and a run
                    # that still finalizes. A scenario budgeted zero turns has no conversation to
                    # close, so that IS the right outcome.
                    try:
                        closing_turn_idx = len(transcript_turns)
                        if closing_raised is not None:
                            raise closing_raised
                        closing_agent_input = _for_agent(user_messages[-1], served_documents)
                        closing_reference_of = _reference_by_fold(served_documents)
                        if closing_reply is not None:
                            closing, closing_started_at, closing_agent_seconds = closing_reply
                        else:
                            closing_started_at = _now()
                            closing_clock = time.monotonic()
                            closing = await child.user_turn(
                                closing_turn_idx,
                                closing_agent_input,
                                interrupt_choice=interrupt_choice,
                                **_user_action_kwarg(pending_user_action),
                            )
                            closing_agent_seconds = round(time.monotonic() - closing_clock, 3)
                        closing_turn_world, closing_user_action_result = _user_world_of(closing)
                        # Filed on the row she acted on NOW, before this side's own processing of the
                        # reply: a raise below must not leave /finalize and the saved row disagreeing.
                        _file_user_action_result(transcript_turns, closing_user_action_result)
                        closing_agent_response = closing.ai_text
                        closing_agent_state = _wire_state_of(closing.agent_state or None, closing_reference_of, thread_id=thread.thread_id)
                        closing_agent_actions = _agent_actions_of(closing)
                        closing_agent_actions = _with_choice_text(closing_agent_actions, choice_text)
                        # ``or None``: ``''`` is "this producer did not classify", and the wire
                        # field must carry the absence rather than an empty label — the same rule
                        # ``_record_turn`` applies to the per-turn twin.
                        closing_reply_source = _reply_source_of(closing) or None
                        chatbot_messages.append(closing.ai_text)
                        if seed_check_pending:
                            resume_fidelity, seed_check_pending = _note_seed_restore(
                                seed_baseline,
                                closing.full_messages or [],
                                fidelity=resume_fidelity,
                                thread_id=thread.thread_id,
                                sent_user_messages=frozenset(content_text(_for_agent(m, served_documents)) for m in user_messages),
                            )
                        closing_msgs, closing_declared, closing_verbatim = _accept_agent_chain(
                            closing.full_messages,
                            sent=closing_agent_input,
                            raw_seen=raw_full_messages,
                            seeded=len(seed_baseline),
                            thread_id=thread.thread_id,
                        )
                        closing_echo_reference_of = _echo_references(closing_reference_of, sent=user_messages[-1], served=served_documents)
                        role_declared_echoes.extend(_as_recorded(closing_declared, closing_echo_reference_of))
                        verbatim_echoes.extend(_as_recorded(closing_verbatim, closing_echo_reference_of))
                        chatbot_full_messages.extend(closing_msgs)
                        closing_turn_messages = _safe_wire_turn_messages(closing_msgs, thread_id=thread.thread_id)
                        closing_calls = derive_turn_tool_calls(
                            closing_msgs, turn=closing_turn_idx, open_calls=open_calls, paused_by=paused_by, choice=interrupt_choice
                        )
                        recovered_message_calls.extend(closing_calls)
                        transcript_turns.append(
                            {
                                'turn_idx': closing_turn_idx,
                                # THE SAME PROJECTION AS EVERY OTHER ROW: writing the carrier VERBATIM
                                # would put a LIST in ``user_message`` on a terminating turn that
                                # carries parts. The backend's artifact reader admits ``None`` there
                                # and REJECTS every other non-``str`` — a list included — for the
                                # WHOLE turn array, so one parts-carrying closing turn would
                                # silently drop that scenario out of the re-score lane with no
                                # warning from this side. The string is the projection and the
                                # list rides BESIDE it, exactly as the per-turn row above.
                                'user_message': content_text(user_messages[-1]),
                                **({'user_messages': user_messages[-1]} if isinstance(user_messages[-1], list) else {}),
                                'agent_message': closing.ai_text,
                                'turn_messages': closing_msgs,
                                'output_user_message': None,
                                'should_terminate': True,
                                'closing_turn': True,
                                # Always present, empty included — see the per-turn record above.
                                'agent_state': closing_agent_state or {},
                                # The closing turn's actions, from the SAME local ``/finalize``
                                # was handed, and written only when the turn carried some — the
                                # per-turn twin above states the rule and why the wire's
                                # tri-state stops here. Separately spelled rather than shared
                                # because this row is a second literal over a second carrier:
                                # the reply that ends the dialog is the one agent turn with no
                                # ``/user-turn`` round trip, and a render verb IS its whole
                                # answer, so a re-score that could not see it would grade the
                                # agent's silence.
                                **({'agent_actions': closing_agent_actions} if closing_agent_actions else {}),
                                'started_at': closing_started_at,
                                'ended_at': _now(),
                                'agent_seconds': closing_agent_seconds,
                                **({'reply_source': closing_reply_source} if closing_reply_source else {}),
                                **_saved_turn_world(closing_turn_world),
                                **_saved_channel_inputs(closing),
                            }
                        )
                        # The closing exchange is a REAL turn to observability: without
                        # this record no sink emits a span for the closing reply (it
                        # never passes through ``_record_turn`` — there is no /user-turn
                        # round trip) and every sink's turn tree ends one exchange
                        # short. Fields are honest: the terminating user message is the
                        # user side, the closing reply the agent side; no next user
                        # message exists (``output_user_message`` empty) and
                        # ``turn_count`` carries the last cycle's server count.
                        # ``closing_turn=True`` lets sinks/consumers distinguish it.
                        turns.append(
                            TranscriptTurn(
                                turn_idx=user_turn_cycles,
                                input_user_message=user_messages[-1],
                                agent_response=closing.ai_text,
                                output_user_message='',
                                should_terminate=True,
                                turn_count=turns[-1]['turn_count'] if turns else 0,
                                agent_tool_calls=_fresh_tool_calls(closing_calls, seen_tool_call_ids),
                                agent_model_calls=self._model_calls(
                                    closing_msgs, input_user_message=user_messages[-1], seen_message_ids=seen_model_message_ids
                                ),
                                closing_turn=True,
                                **({'reply_source': closing_reply_source} if closing_reply_source else {}),
                            )
                        )
                    except Exception as closing_exc:
                        # A HARNESS fault (or any typed failure the caller will re-drive) takes the path it
                        # takes on any other turn: re-driven or recorded, never scored with the closing turn
                        # missing. The agent's own failures (``landed_reason``) and an untyped raise finalize
                        # the complete dialog without its closing reply — the dialog had already ended.
                        # On a single-hop replay the closing turn IS the turn under test: without it the run
                        # measured nothing, so any failure is the scenario's failure, never a finalize.
                        if replay_turn_only or (getattr(type(closing_exc), 'kind', None) is not None and self._landing_reason(closing_exc) is None):
                            raise
                        # The act she took with her last message was applied (or refused) before the
                        # agent failed: it is still reported, never left for the server to guess.
                        if closing_user_action_result is None and (raised := _raised_turn_user_action_result(child)) is not None:
                            closing_user_action_result = raised
                            _file_user_action_result(transcript_turns, raised)
                        turn_failure = _turn_failure_record(closing_exc, turn_idx=closing_turn_idx, closing=True)
                        child_reaped = isinstance(closing_exc, AgentTurnTimeout)
                        logging.getLogger(__name__).warning(
                            'closing-turn delivery failed on thread %s; finalizing without a closing reply',
                            thread.thread_id,
                            exc_info=True,
                        )

                # The run-local world snapshot + tool events come from the child's
                # finalize() over IPC, with the tool calls the mock never answered. A channel whose
                # read failed ships as ``None`` (unread) after one readiness-gated re-read — never as
                # a failure of the dialog. A child the per-turn wall reaped is not asked at all.
                if child_reaped:
                    final_world_state, tool_events, unanswered_mock_calls, capture_failed = None, None, None, None
                else:
                    final_world_state, tool_events, unanswered_mock_calls, capture_failed = await self._capture_finalize(
                        child, thread_id=thread.thread_id, mock_service_url=thread.mock_service_url, mock_mode=mock_mode
                    )
                world_values_coerced: dict[str, str] | None = getattr(child, 'world_values_coerced', None)
                interception_changed: list[str] | None = getattr(child, 'interception_changed', None)
                progress.mock_tool_events = tool_events
                # A reaped child was never asked: its ``None`` is unread, not an absent channel.
                progress.mock_log_read = not child_reaped
                # ``getattr``: a ``child_factory`` child need not record reads, and then claims nothing.
                unread_inputs: list[str] | None = with_unread_parts(getattr(child, 'unread_inputs', None), getattr(child, 'unread_parts', None))
                post_seed_world_state: dict[str, Any] | None = getattr(child, 'post_seed_world_state', None)

                # The turns the PARENT emptied on the agent's behalf, read INSIDE the child
                # scope (the closing turn can degrade too, so this is the last honest moment).
                degraded_turns = child.degraded_turns
                # ``getattr``: a ``child_factory`` child need not record trims.
                ipc_trimmed_turns = list(getattr(child, 'ipc_trimmed_turns', None) or [])
                turns_driven = child.turns_driven
            self._child_closed()

            # Reported BY RULE, at the level each one deserves. Both are dropped from the agent
            # channel and both ride the result, but only one is a defect to act on.
            if role_declared_echoes:
                # The NORMAL shape: a LangGraph agent's first reply is [HumanMessage, AIMessage].
                # Reporting it as an ERROR would tell every ordinary LangGraph user to go fix
                # simulate.py, once per scenario, for behaving exactly as its own docstring says
                # is correct — a report about the harness dressed as a finding about the agent.
                logging.getLogger(__name__).debug(
                    'thread %s: %d message(s) declaring a user/human role were kept out of the agent channel. This is the '
                    'ordinary LangGraph shape and needs no action; ``turns`` still pairs each prompt with its reply.',
                    thread.thread_id,
                    len(role_declared_echoes),
                )
            if verbatim_echoes:
                # LOUD, once per scenario, and only for the contract breach: a message that
                # declares NO role and is verbatim what this harness just sent. The run is not
                # wrong — ``turns`` still pairs each prompt with its reply — but such a message
                # would be stamped 'ai' on the wire, and the backend's deliberately exclusion-based
                # selection of agent-side messages reads a role-less message as agent output. Left in, a
                # user-pasted URL or quote is scored as something the agent said.
                logging.getLogger(__name__).error(
                    'thread %s: the adapter returned %d message(s) carrying NO role whose text is verbatim what this harness '
                    'just sent; they were dropped from the agent channel. simulate.py must return only what the AGENT '
                    'produced this turn, and must give every message it returns a role. Left in, a user-pasted URL or quote '
                    'is scored as something the agent said.',
                    thread.thread_id,
                    len(verbatim_echoes),
                )

            if degraded_turns and turns_driven and len(degraded_turns) == turns_driven:
                # EVERY turn was emptied by the parent: this transcript contains no reply the
                # agent actually gave, so it cannot be a measurement OF the agent. Scoring goes
                # ahead — the judges' verdicts on an empty dialog are still the fastest way to
                # see what happened — but the run carries the typed reason, so the report and
                # the suite critic read "the harness never got a reply out of simulate.py"
                # rather than "the agent said nothing for 12 turns". Same misattribution
                # TURN_CAP_REASON exists to prevent, one layer down.
                # Unconditional BECAUSE it is the top of the precedence order — it outranks a
                # server-reported reason, since there was no conversation for the persona to have
                # an intent about. Name what it displaces: the backend stamped its own reason into
                # this run's transcript, so the two artifacts would otherwise disagree in silence.
                if ended_early is not None and ended_early != ADAPTER_DEGRADED_REASON:
                    logging.getLogger(__name__).warning(
                        'thread %s: the run also recorded ended_early=%r, which is superseded here — every turn was '
                        'emptied by the harness, so there was no conversation for that reason to be about. The saved '
                        "transcript's own dialog_outcome still carries it.",
                        thread.thread_id,
                        ended_early,
                    )
                ended_early = ADAPTER_DEGRADED_REASON
                # The cause rode with the reason it explained; with that reason displaced it
                # would explain nothing, and left in place it would read as the cause of the
                # harness fault.
                ended_early_cause = None
                logging.getLogger(__name__).error(
                    'thread %s: ALL %d turn(s) were emptied (%s) — nothing about the agent was measured. Check '
                    'the owners in the order the per-turn `scoring an EMPTY agent turn` warnings give (the agent, '
                    'simulate.py, the runner) before acting on this row.',
                    thread.thread_id,
                    turns_driven,
                    ', '.join(sorted({str(entry.get('subkind')) for entry in degraded_turns})),
                )

            # mode:real tools execute natively in the agent and leave no /mock/execute
            # event. Union what the two out-of-band lanes recovered into tool_events so the
            # server's tool-trail policies (and tool_events.json, and the rollup's
            # tool_calls metric, which reads this same merged payload) see EVERY tool that
            # fired, not just mode:mock ones. Count-based dedup keeps repeated real calls
            # intact without double-counting the authoritative mode:mock events already
            # present; the per-turn stamps on both lanes are what let the merge interleave
            # them in conversation order instead of appending one after the other.
            #
            # The trail lane is the ONLY one that sees an agent running its own tool loop:
            # it puts no tool_use block in the transcript, so message recovery finds nothing
            # and a failed dispatch reads identically to a call that never happened.
            #
            # ``None`` survives this call when the child read no channel AND neither lane
            # recovered anything: an unread trail is not an empty one, and only the ``None``
            # reaches the server as the ``null`` its finalize ingress reads as "not supplied".
            # ``progress.tool_events()`` is the recovered-call merge (the ONE merge the partial
            # evidence reads too); the unread-channel rule then applies to what is shipped, and the
            # shipped value is what the partial evidence reports after this point.
            tool_events = _unread_if_native_unobserved(
                progress.tool_events(),
                Path(self.simulate_path),
                trail_declared=any(trail_recorded(turn.get('agent_state')) for turn in transcript_turns),
                thread_id=thread.thread_id,
            )
            progress.shipped_tool_events = tool_events
            progress.tool_events_shipped = True

            finalize = await self._finalize_async(
                thread.thread_id,
                # INDEXED BY ROLE: one row per user turn that happened. The carrier
                # holds no ``None``, so an agent-first run contributes NO human row here —
                # rather than a ``{'content': None}`` one, which the backend would count as a turn and
                # every judge would read as the user having said nothing when the user was never
                # asked to speak. The two arrays are independent projections of their own channels;
                # nothing pairs them by index.
                # BUDGETED, like the agent channel beside it. Capping a LIST content in
                # ``_wire_turn_messages`` alone would leave the USER-side carrier shipping a base64
                # ``file`` part at full size in this REQUEST BODY.
                #
                # The WIRE only. ``transcript.json`` keeps the part uncapped, and that is the rule
                # :data:`_WIRE_MESSAGE_CONTENT_CAP` states about itself ("the full chain still
                # lands uncapped in transcript.json; the wire copy only feeds judge rendering").
                # The deciding reason is that the ARTIFACT is read and this body is not kept: a
                # re-score replays transcript.json, and a truncated base64 there means it can no
                # longer see what the user sent, while this body only feeds judge and simulator
                # rendering, where the payload is bytes rather than signal. Precisely: the carrier
                # a re-score reads back is the TOP-LEVEL ``user_messages`` of ``transcript.json``,
                # which is this same list and is also uncapped.
                #
                # Capping only the user half of the artifact would also
                # leave the two channels of one file disagreeing about what a turn contained.
                #
                # Through :func:`_budgeted_content`, so BOTH spellings of one message are charged:
                # a ``_budgeted_parts(m) if isinstance(m, list) else m`` would let the same long
                # opener ride this body whole as a ``str`` and capped as a ``text`` part.
                user_messages=[{'type': 'HumanMessage', 'content': _budgeted_content(m)} for m in user_messages],
                # WHOLE — and the line above is the one shrink this rule causes: a
                # legitimately long PLAIN-TEXT user turn (a persona that pastes a policy document)
                # is cut here, as its content-part spelling is. Nothing
                # observable is lost — the server never reads ``user_messages``, and
                # ``transcript.json`` plus ``ThreadResult.user_messages`` keep it whole — but it is
                # a shrink of real user text, and the two spellings agreeing is the point of it.
                #
                # This carrier is not charged. The server declares it UNREAD in as many words
                # (``FinalizeRequest.chatbot_messages``: turn-boundary-free,
                # filled with final-reply strings) and nothing in the server reads
                # it, so budgeting it would shorten bytes nobody looks at while manufacturing a
                # THIRD copy of the agent's words that disagrees with the two that ARE read —
                # ``agent_response`` per turn and ``closing_agent_response`` below.
                chatbot_messages=[{'type': 'AIMessage', 'content': m} for m in chatbot_messages],
                final_world_state=final_world_state,
                tool_events=tool_events,
                # WHOLE, and this is the row that must not be "finished" later. The backend appends
                # this string to the thread's agent messages and binds THAT into the
                # deterministic-check sandbox as the ``chatbot_messages`` namespace the plugin's
                # own contract declares — the name a customer writes ``chatbot_messages[-1]`` and
                # ``contains_term`` against. Capping it would cut what the customer's own check
                # reads, and would cut it for the CLOSING turn alone while every other turn was
                # graded whole. Its narration COPY (``closing_turn_messages``) is charged instead,
                # on this same body, which is the whole shape of the rule.
                closing_agent_response=closing_agent_response,
                closing_agent_state=closing_agent_state,
                closing_turn_messages=closing_turn_messages,
                # WHOLE for the reason the per-kind budget already gives for ``file`` / ``json``:
                # half a serialized structure is not a smaller structure. The backend files these per
                # turn and grades them as actions, so a truncated ``args`` is
                # a corrupt action rather than a shorter one.
                closing_agent_actions=closing_agent_actions,
                closing_reply_source=closing_reply_source,
                # ONLY the client-measured reasons cross the wire (``RELAYED_REASONS``: the time budget
                # and the reasons a landed turn failure ends on). The loop can also end on
                # ``adapter_degraded``, which is ours alone: the backend has no prose for it (its
                # judges would be told nothing while the server logged an unknown-reason
                # warning every run), and its documented precedence is client-supersedes-server,
                # which contradicts the fill-only-a-gap rule the finalize ingress is built on.
                # Relaying the rest is a decision to make deliberately, not a side effect.
                # A single-hop replay relays its cut whatever the local label (an emptied turn reads
                # ``adapter_degraded`` here): the backend reads it to arm the end-of-run cut route, grade a
                # tool-only issue turn and skip the goal judges.
                ended_early=REPLAY_CUT_REASON if replay_turn_only else ended_early if ended_early in RELAYED_REASONS else None,
                resume_fidelity=resume_fidelity,
                user_action_result=closing_user_action_result,
                unread_inputs=unread_inputs,
                post_seed_world_state=post_seed_world_state,
            )
            await self._emit_observability(
                thread,
                run_ctx=run_ctx,
                turns=turns,
                finalize=finalize,
                final_world_state=final_world_state,
                tool_events=tool_events,
                ended_early=ended_early,
                ended_early_cause=ended_early_cause,
            )
            reply_source_counts = _count_reply_sources(turns, thread_id=thread.thread_id)

            return ThreadResult(
                thread_id=thread.thread_id,
                # Outer-conversation turn count (the /user-turn cycles), FROZEN
                # before the closing exchange was appended to ``turns`` — the
                # observability list carries N+1 entries (closing turn included)
                # while this count stays N. NOT len(chatbot_messages) either —
                # that also carries the closing reply.
                turns=user_turn_cycles,
                finalize=finalize,
                user_messages=user_messages,
                chatbot_messages=chatbot_messages,
                chatbot_full_messages=chatbot_full_messages,
                tool_events=tool_events,
                unanswered_mock_calls=unanswered_mock_calls,
                transcript_turns=transcript_turns,
                reply_source_counts=reply_source_counts,
                ended_early=ended_early,
                ended_early_cause=ended_early_cause,
                resume_seed=resume_seed,
                resume_fidelity=resume_fidelity,
                initial_world_state=initial_world_state,
                initial_agent_context=initial_agent_context,
                unread_inputs=unread_inputs,
                post_seed_world_state=post_seed_world_state,
                degraded_turns=degraded_turns,
                ipc_trimmed_turns=ipc_trimmed_turns,
                # Only the rules that FIRED. An empty dict is "the agent channel is exactly what
                # the adapter returned", which is a different claim from "this key was written
                # with two empty lists in it" only in verbosity — but it keeps every ordinary
                # transcript free of a block that says nothing.
                user_echoes_dropped={
                    **({'role_declared': role_declared_echoes} if role_declared_echoes else {}),
                    **({'verbatim_echo': verbatim_echoes} if verbatim_echoes else {}),
                },
                finalize_job_id=self.last_finalize_job_id,
                turn_failure=turn_failure,
                capture_failed=capture_failed,
                world_values_coerced=world_values_coerced,
                interception_changed=interception_changed,
            )

    def _landing_reason(self, exc: BaseException) -> str | None:
        """The ``ended_early`` reason a failed turn LANDS on, or ``None`` when it must raise: a failure
        :func:`landed_reason` does not own, or one the caller will re-drive (the shared
        :func:`~plurai.orchestrator.redrive.static_trigger`). ``concurrency`` there only gates a
        reaped child, which never lands, so it is not this decision's input."""
        reason = landed_reason(exc)
        if reason is None:
            return None
        if self._redrive_available and static_trigger(exc, allow_turn_timeout_redrive=self._allow_turn_timeout_redrive, concurrency=1) is not None:
            return None
        return reason

    def _read_mock_log_on_raise(
        self, child: Any, progress: ScenarioProgress, *, thread_id: str, mock_service_url: str | None, mock_mode: MockMode
    ) -> Callable[[type[BaseException] | None, BaseException | None, Any], Any]:
        """The exit callback that reads the mock's event log when the turn loop RAISES.

        Without it a run that fails before finalizing (a zero-turn agent raise, a re-driven harness
        fault) leaves ``progress`` unread, and the crash evidence reports the agent's tool activity as
        unknown although the child still holds the log. One read through :meth:`_capture_finalize`
        (its readiness-gated single re-read included), bounded by ``_CRASH_MOCK_LOG_READ_TIMEOUT``.

        Never on ``CancelledError`` or another ``BaseException`` (only an ``Exception`` reads), never
        on a child the per-turn wall reaped, never when the log was already read. A failed read is
        logged and dropped: the original raise always propagates (the callback returns ``False``).
        """

        async def _on_exit(_exc_type: type[BaseException] | None, exc: BaseException | None, _tb: Any) -> bool:
            if not isinstance(exc, Exception) or isinstance(exc, AgentTurnTimeout) or progress.mock_log_read:
                return False
            try:
                _, events, _, _ = await asyncio.wait_for(
                    self._capture_finalize(child, thread_id=thread_id, mock_service_url=mock_service_url, mock_mode=mock_mode),
                    timeout=_CRASH_MOCK_LOG_READ_TIMEOUT,
                )
            except Exception as read_exc:  # noqa: BLE001 - evidence only; the turn loop's raise is what propagates
                logging.getLogger(__name__).warning(
                    'thread %s: the mock event log could not be read after the run raised (%s: %s); the crash evidence reports it unread.',
                    thread_id,
                    type(read_exc).__name__,
                    read_exc,
                )
                return False
            progress.mock_tool_events = events
            progress.mock_log_read = events is not None
            return False

        return _on_exit

    async def _capture_finalize(
        self, child: Any, *, thread_id: str, mock_service_url: str | None, mock_mode: MockMode
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]] | None, dict[str, str] | None]:
        """The child's finalize snapshots: ``(world, events, unanswered_mock_calls, capture_failed)``.

        A snapshot read can fail while the dialog is complete (a slow ``/state`` read, a mock that
        restarted under it). That is NOT a failure of the dialog — only world- and trail-reading
        policies lack their input — so it never raises. When the failure was one a re-read can clear (a
        mock restart, a read that timed out), the read is asked ONCE more after the backend proves ready (``wait_until_ready``, its own bound); a
        channel still unread ships as ``None``, which ``/finalize`` reads as unobserved, and
        ``capture_failed`` names it with the child's detail.
        """
        world, events, unanswered, retryable = await child.finalize()
        if _finalize_error_message(world, events) is not None and retryable:
            if await wait_until_ready(self.client, mock_service_url=mock_service_url, mock_mode=mock_mode):
                again_world, again_events, more, _ = await child.finalize()
                if isinstance(world, dict) and _FINALIZE_ERROR_MARKER in world:
                    world = again_world
                if isinstance(events, dict) and _FINALIZE_ERROR_MARKER in events:
                    events = again_events
                if more:
                    unanswered = [*(unanswered or []), *more]
        failed: dict[str, str] = {}
        if isinstance(world, dict) and _FINALIZE_ERROR_MARKER in world:
            failed['world'], world = str(world[_FINALIZE_ERROR_MARKER]), None
        if isinstance(events, dict) and _FINALIZE_ERROR_MARKER in events:
            failed['events'], events = str(events[_FINALIZE_ERROR_MARKER]), None
        if failed:
            logging.getLogger(__name__).warning(
                'thread %s: the finalize snapshot could not be read (%s); the dialog is finalized with %s unread, so the policies that read it abstain.',
                thread_id,
                '; '.join(f'{channel}: {detail}' for channel, detail in failed.items()),
                ' and '.join(failed),
            )
        # The world's images become storage references here, once, so ``/finalize`` and the transcript hold the same rows.
        # They go to the backend and bearer this run talks to, never to whichever the environment names.
        world = await upload_world_images(
            world,
            client_factory=self.storage_client_factory or self._run_storage_client,
        )
        return world, events, unanswered, failed or None

    async def _run_pipeline_hop(
        self,
        thread: CreateThreadResponse,
        *,
        scenario_id: str,
        run_ctx: RunContext,
        initial_agent_context: dict[str, Any],
        mock_mode: MockMode,
    ) -> ThreadResult:
        """Drive ONE ``invoke_pipeline`` envelope and finalize it as a scored row.

        The sibling of the turn loop in :meth:`run`, and the caller of the pipeline transport:
        ``_protocol.PipelineInvoke`` / ``child.handle_pipeline_invoke`` /
        ``ChildProcess.pipeline_invoke``.

        The bridge is the injected agent context: the backend already generates the hop's typed
        inputs (``agent_context.yaml`` → ``CreateThreadResponse.initial_agent_context``), the
        client already fetches its ``<scheme>:///`` documents at the wire-receive seam, and that
        dict, each served document folded to its bytes, IS what ``invoke_pipeline`` takes. The
        recorded form keeps the references, so ``pipeline_inputs`` and ``pipeline_outputs`` stay
        small on ``/finalize``. No second input carrier, no regeneration.

        What is DELIBERATELY absent:

        * **the conversation channels stay EMPTY, not synthetic.** ``/threads`` mints an LLM
          opener for a set with no ``first_talk.txt``; on a hop that message is delivered to
          nothing, so forwarding it would put a user turn that never happened in front of every
          judge and make ``len(user_messages) == 1`` true for every count-shaped guard.
        * **no invented turn.** ``turns=0`` and no ``transcript_turns``: "one-shot is always
          one turn" would be wrong, because the artifact indexes dense per-turn
          lists by that count, so a fabricated turn makes ``agent_states[-1]`` a claim about a
          turn that never ran. The hop is represented by ``pipeline_io`` and ``scenario_shape``.
        * **no new timeout knob.** The hop is bounded by ``agent_turn_timeout`` — 1560s /
          2160s by phase default, well past a 300s default — and the runner's hard ceiling is
          still the hang detector.

        An agent RAISE is an OUTCOME, a harness fault is an ERROR: a handler raise, a non-dict
        return and a wall-clock timeout are all recorded in ``pipeline_error`` and finalized, so a
        deliberate-refusal policy can credit them; a harness fault (``kind='harness_fault'``: the
        two envelope refusals and a ``plurai.HarnessFault``) and a child that exited propagate, so
        the runner writes ``error.json`` and calls ``/fail`` with no verdict.

        Both policy kinds read it: a judge's prompt renders it, and the code_eval sandbox binds
        ``pipeline_error`` (``None`` when the hop returned) beside ``pipeline_inputs`` /
        ``pipeline_outputs``.
        """
        initial_world_state = thread.initial_world_state or {}
        served_documents = self.client.served_documents(thread.thread_id)
        agent_inputs = _context_for_agent(initial_agent_context, served_documents)
        outputs: dict[str, Any] = {}
        duration_s = 0.0
        pipeline_error: dict[str, Any] | None = None
        # The TURN budget (``agent_turn_timeout``) reaps the child before raising, so the hop can
        # still be scored but must be finalized WITHOUT it. The SCENARIO budget
        # (``harness_scenario_timeout``) stays a harness fault and never reaches here.
        child_reaped = False
        # A PROVIDER failure is infrastructure, never the component: the hop lands with its error
        # recorded and ``ended_early='provider_infra'`` (censored, the trial invalid) — the same rule a
        # dialog turn follows — instead of being credited to the component as its own outcome.
        ended_early: str | None = None
        turn_failure: dict[str, Any] | None = None
        async with ChildProcess(self.simulate_path, agent_turn_timeout=self.settings.network.agent_turn_timeout, env=self._child_env()) as child:
            await child.init(
                thread_id=thread.thread_id,
                mock_service_url=thread.mock_service_url,
                mock_mode=mock_mode,
                initial_world_state=initial_world_state,
                initial_agent_context=agent_inputs,
                scenario_metadata=thread.scenario_metadata or {},
            )
            # THE SEED READ-BACK, as :meth:`run` asks it: a seed that did not land voids the hop at finalize
            # (``undelivered_seed``), so the component is not invoked and the hop is finalized with no outputs.
            seed_not_landed = seed_readback_at_init(scenario_id, getattr(child, 'post_seed_world_state', None), initial_world_state).voids_scenario
            if seed_not_landed:
                ended_early = SEED_NOT_LANDED_REASON
            self._setup_done()
            hop = None
            started = time.monotonic()
            try:
                if not seed_not_landed:
                    hop = await child.pipeline_invoke(agent_inputs, timeout=self.settings.network.agent_turn_timeout)
            except (HarnessChildExited, HarnessFaultRaised):
                # A harness fault is typed by its RAISER (the child's ``kind``), never read back
                # out of a message: it propagates to error.json + /fail with no verdict.
                raise
            except OrchestratorAdapterError as exc:
                # A structured shape that a policy can read. ``child_exc_class`` is the
                # CUSTOMER's exception class, without which every hop failure looks like
                # ``OrchestratorAdapterError``; ``timed_out`` rides ``AgentTurnTimeout``, which is
                # a subclass of the caught type — a hop that never returns is a fact ABOUT the
                # component, so it is scored rather than reported as infra.
                # The COMPONENT's own words reach the judge; the SDK's wrapper prose
                # ("pipeline_invoke failed on thread …") stays in the log below.
                pipeline_error = {'exception': type(exc).__name__, 'error': getattr(exc, 'child_message', None) or str(exc)}
                if child_exc_class := getattr(exc, 'child_exc_class', None):
                    pipeline_error['child_exc_class'] = child_exc_class
                if isinstance(exc, ProviderInfra):
                    ended_early = PROVIDER_INFRA_REASON
                    turn_failure = _turn_failure_record(exc, turn_idx=0, closing=False)
                if getattr(exc, 'timed_out', False):
                    pipeline_error['timed_out'] = True
                    # ``_send_and_wait`` SIGTERM/SIGKILLs the child before raising the turn
                    # budget, so asking it to finalize would raise ``HarnessChildExited`` and
                    # discard this scored outcome — the failure this branch exists to record.
                    child_reaped = True
                logging.getLogger(__name__).warning(
                    'thread %s: invoke_pipeline did not return outputs (%s). This is a SCORED outcome, not a harness '
                    'failure — the run finalizes with `pipeline_error` set, which a judge and a code_eval policy '
                    'can both read.',
                    thread.thread_id,
                    pipeline_error['error'],
                )
            else:
                if hop is not None:
                    outputs = hop.outputs
            finally:
                # Present on EVERY outcome: the child's own clock when it returned one, the
                # parent's elapsed otherwise. A hop that raised early or hung to the budget
                # must not report 0.0 — indistinguishable from an instantaneous failure in the
                # one field the artifact offers for "how long did this take".
                duration_s = hop.duration_s if hop is not None else time.monotonic() - started
            if child_reaped:
                # Nothing to ask: the snapshots were never taken, so they are recorded as
                # ABSENT rather than as an empty world, which would be a measurement claim.
                #
                # ``None``, not ``{}``: ``{}`` is the second state ("we looked, and there was
                # nothing"). That is
                # the one reading trusted downstream: ``/finalize`` carries a literal ``null`` for
                # an unread channel and the server demotes a tool-trail policy to not-measured on
                # it, but a ``{}`` tells it this agent invoked zero tools and every world-reading
                # judge that the world was empty. A hop that HUNG TO ITS BUDGET is exactly the
                # population where nobody was watching, so it is the last place to publish a
                # measurement. Both halves: the world snapshot was never taken either.
                final_world_state, tool_events, unanswered_mock_calls, capture_failed = None, None, None, None
            else:
                # Same rule as the dialog tail: a snapshot that cannot be read ships unread, never fails the hop.
                final_world_state, tool_events, unanswered_mock_calls, capture_failed = await self._capture_finalize(
                    child, thread_id=thread.thread_id, mock_service_url=thread.mock_service_url, mock_mode=mock_mode
                )
            world_values_coerced: dict[str, str] | None = getattr(child, 'world_values_coerced', None)
            interception_changed: list[str] | None = getattr(child, 'interception_changed', None)
            # A reaped child recorded nothing, so it claims nothing.
            unread_inputs: list[str] | None = getattr(child, 'unread_inputs', None)
            post_seed_world_state: dict[str, Any] | None = getattr(child, 'post_seed_world_state', None)
        self._child_closed()

        # The hop's recorded trail, returned beside its outputs: the only record of a call that never
        # reached ``/mock/execute``. Projected into the dialog lane's ``tool_call`` records (turn 0,
        # call order), so structured checks, ``tool_call_count`` and the hop's tool spans count it.
        if AGENT_STATE_TOOL_TRAIL_KEY in outputs:
            tool_events = merge_pipeline_trail(tool_events, outputs[AGENT_STATE_TOOL_TRAIL_KEY])
        # A returned trail is not a ``plurai.confine`` recording, so a declared native tool is never observed here.
        tool_events = _unread_if_native_unobserved(tool_events, Path(self.simulate_path), trail_declared=False, thread_id=thread.thread_id)

        # The outputs as the wire and the transcript carry them: a served document the pipeline echoes is its
        # reference, so ``pipeline_outputs`` stays under the request limit beside ``pipeline_inputs``.
        raw_outputs = outputs
        outputs = _wire_state_of(outputs, _reference_by_fold(served_documents), thread_id=thread.thread_id, label='pipeline_outputs') or {}

        # Shaped BEFORE /finalize is submitted, not at the TranscriptTurn below: ``_json_shaped``
        # is the only raise site left in this tail, and a raise AFTER the submit would leave the
        # row carrying a verdict on the server and an ``error.json`` on the customer's disk for
        # the same scenario. Shaping first makes that failure abort with nothing finalized.
        # ``_json_shaped`` raises a NAMED ``TypeError``, but the values it walks can also raise
        # from inside a customer object's own ``model_dump`` (``ValueError`` /
        # ``PydanticSerializationError``) or bottom out on a self-referential dict
        # (``RecursionError``). Left bare, those would escape the one raise site this tail is
        # allowed to have and take the scenario out as an untyped traceback rather than as a labelled adapter
        # failure — for the SHAPE this function exists to serve (a structured-output extractor
        # returning a Pydantic model).
        try:
            # The record of what was coerced lives on the child's own shaping of the outputs
            # (``pipeline_io['values_coerced']``); this is the observability rendering of the same values.
            hop_input_json = json.dumps(_json_shaped(initial_agent_context, coerced={}))
            hop_response_json = json.dumps(_json_shaped(pipeline_error or outputs, coerced={}))
        except Exception as exc:
            raise OrchestratorAdapterError(
                f'The pipeline hop on thread {thread.thread_id} returned a value that cannot be carried as JSON, '
                f'so it could not be scored: {type(exc).__name__}: {exc}',
                thread_id=thread.thread_id,
            ) from exc
        finalize = await self._finalize_async(
            thread.thread_id,
            user_messages=[],
            chatbot_messages=[],
            final_world_state=final_world_state,
            tool_events=tool_events,
            pipeline_inputs=initial_agent_context,
            pipeline_outputs=outputs,
            pipeline_error=pipeline_error,
            unread_inputs=unread_inputs,
            post_seed_world_state=post_seed_world_state,
            # Only the client-measured reasons cross the wire (``RELAYED_REASONS``); ``seed_not_landed`` is local.
            ended_early=ended_early if ended_early in RELAYED_REASONS else None,
        )
        # ONE record for the hop, through the SAME sink seam a dialog's turns take. With
        # ``turns=[]`` the scenario anchor would render with no agent span at all, and every
        # pipeline row would show zero turns, no output and no error, beside eval scores with
        # nothing to read them against. ``reply_source`` labels it so a reader is
        # never told a hop's JSON was something an agent said.
        hop_turn = TranscriptTurn(
            turn_idx=0,
            input_user_message=hop_input_json,
            agent_response=hop_response_json,
            output_user_message='',
            should_terminate=True,
            turn_count=1,
            reply_source='pipeline_error' if pipeline_error else 'pipeline',
        )
        await self._emit_observability(
            thread,
            run_ctx=run_ctx,
            # A hop never invoked (its seed did not land) records no turn.
            turns=[] if seed_not_landed else [hop_turn],
            finalize=finalize,
            final_world_state=final_world_state,
            tool_events=tool_events,
        )
        return ThreadResult(
            thread_id=thread.thread_id,
            turns=0,
            finalize=finalize,
            user_messages=[],
            chatbot_messages=[],
            tool_events=tool_events,
            unanswered_mock_calls=unanswered_mock_calls,
            initial_world_state=initial_world_state,
            initial_agent_context=initial_agent_context,
            unread_inputs=unread_inputs,
            post_seed_world_state=post_seed_world_state,
            scenario_shape=PIPELINE_SHAPE,
            # ``timed_out`` is lifted OUT of the error dict as well as left in it: "did this hop
            # hang?" is the first question asked of a slow suite, and a reader should not have to
            # know that the answer lives inside an optional key of an optional block. It is
            # ``False`` on every hop that returned, including one that raised.
            pipeline_io={
                'outputs': outputs,
                'error': pipeline_error,
                'duration_s': duration_s,
                'timed_out': bool(pipeline_error and pipeline_error.get('timed_out')),
                **({'values_coerced': coerced} if hop is not None and (coerced := getattr(hop, 'values_coerced', None)) else {}),
            },
            pipeline_raw_outputs=raw_outputs,
            finalize_job_id=self.last_finalize_job_id,
            ended_early=ended_early,
            turn_failure=turn_failure,
            capture_failed=capture_failed,
            world_values_coerced=world_values_coerced,
            interception_changed=interception_changed,
        )

    async def upload_datasets(
        self,
        *,
        scenario_set_id: str,
        out_dir: Path | str,
        scenario_ids: Sequence[str] | None = None,
    ) -> Any:
        """Fetch a scenario set and upload it to the active observability sink.

        Opt-in (the CLI exposes ``--upload-datasets``). Fetches the set's
        scenario content from the backend server (KEY-FREE), gathers it, dispatches it
        to the SINGLE active CLIENT-side sink, and persists the provider
        idempotency handles next to the run artifacts (``<out_dir>/
        dataset_upload_metadata.json``) so a re-run reuses the prior items.

        Reuses the held active sink: the upload dispatches to ``self.observability``
        (the same single active sink the factory already built and the orchestrator
        records traces against), so there is NO re-resolution from settings here.
        With a :class:`~plurai.observability.sink_protocol.NoOpSink` held (no sink
        configured) this is a graceful no-op. The upload itself is best-effort: a
        sink / transport failure is returned as a failed
        :class:`plurai.observability.dataset_upload.UploadResult`, never raised,
        so it can't abort the surrounding run. The FETCH step DOES raise on a
        non-2xx (e.g. 404 for an unknown set id) — the caller explicitly asked
        to upload it.

        ``scenario_ids`` limits the upload to those scenario folders (a run uploads
        only what it executes); ``None`` uploads the whole set.

        Returns a :class:`plurai.observability.dataset_upload.UploadResult`.
        """

        fetched = await self.client.fetch_scenario_set(scenario_set_id)
        return await upload_scenarios_from_fetch(
            fetched,
            sink=self.observability,
            out_dir=Path(out_dir),
            scenario_ids=scenario_ids,
        )

    # -- async finalize --

    async def _finalize_async(
        self,
        thread_id: str,
        *,
        user_messages: list[dict[str, str]],
        chatbot_messages: list[dict[str, str]],
        # NULLABLE for the same reason ``tool_events`` beside it is: a snapshot nobody took is a
        # THIRD state, and ``client.finalize_async_submit`` already carries it to the wire as a
        # literal ``null`` (its own docstring says so); the pipeline hop's reaped arm passes it.
        final_world_state: dict[str, Any] | None,
        tool_events: dict[str, Any] | None,
        closing_agent_response: str | None = None,
        closing_agent_state: dict[str, Any] | None = None,
        closing_turn_messages: list[dict[str, Any]] | None = None,
        closing_agent_actions: list[dict[str, Any]] | None = None,
        closing_reply_source: str | None = None,
        ended_early: str | None = None,
        resume_fidelity: dict[str, Any] | None = None,
        pipeline_inputs: dict[str, Any] | None = None,
        pipeline_outputs: dict[str, Any] | None = None,
        pipeline_error: dict[str, Any] | None = None,
        user_action_result: dict[str, Any] | None = None,
        unread_inputs: list[str] | None,
        post_seed_world_state: dict[str, Any] | None,
    ) -> FinalizeResponse:
        """Async finalize: POST mode:async → poll → return result.

        The three ``pipeline_*`` arguments are the ONE-SHOT lane's whole scoring contract and
        are ``None`` on every dialog scenario (see :meth:`_run_pipeline_hop`). They reach the
        e2e guards and the judges through the single per-step info dict the backend builds.

        Submit (idempotent via server dedup) → record in-memory → poll every
        ``finalize_poll_interval_seconds`` bounded by
        ``finalize_poll_max_wallclock_seconds``.

        Terminal outcomes:
        - ``done``    → clear ``_inflight_finalize_job_id``; return ``FinalizeResponse``.
        - ``failed``  → clear ``_inflight_finalize_job_id``; raise ``FinalizeJobFailed``, or ``CreditRefused`` when
          the job ended on a credit refusal.
        - Schema drift (``ValidationError``) → clear ``_inflight_finalize_job_id``; raise
          ``OrchestratorAdapterError`` (unrecognised status from server is a fast-fail).
        - Budget exceeded → raise ``FinalizePollTimeout`` (``_inflight_finalize_job_id``
          stays set so cancel-on-abandon can DELETE the still-running server job).

        A poll GET the client could not complete is NOT terminal: a timeout after the connection was made, or an outage
        that spent the request's own retry budget (``_retry``: a connection failure or a 502, 503 or 504). The job is
        server-side and addressed BY ID, and its judges usually already ran, so the poll RESUMES against the same
        ``job_id``, backing off, until the wallclock budget it already has runs out. An answer about the job (a 404, a
        401) propagates at once, the inflight id left set so cancel-on-abandon runs.

        The open invocation's BACKEND gate overrules ride the submit
        (:func:`~plurai.orchestrator.client.submit_carrying_overrules`), and the result's
        ``gate_authority`` answer is merged into that invocation's collector, then cleared off the
        returned result together with every overrule marker on its cells
        (:func:`~plurai.orchestrator.client.take_gate_authority`): overrules belong to
        ``gate_authority.json`` only, and the persisted cells carry the effective verdict alone.
        """
        # Reassert (run() already set it at create) so the in-process embedding
        # path, which can call _finalize_async directly, still gives
        # _cancel_inflight_finalize a thread to name.
        self.thread_id = thread_id
        callback, self._on_finalize_start = self._on_finalize_start, None
        if callback is not None:
            callback()
        net = self.settings.network
        poll_interval = net.finalize_poll_interval_seconds
        max_wallclock = net.finalize_poll_max_wallclock_seconds

        # 1. Submit (idempotent via server submit-dedup).
        # ValidationError propagates when the server body does not match
        # FinalizeJobSubmitResponse (missing job_id, unknown status, etc.) — convert
        # to a typed OrchestratorAdapterError so thread_id is never lost.
        try:
            submit = await submit_carrying_overrules(
                lambda overrules: self.client.finalize_async_submit(
                    thread_id,
                    user_messages=user_messages,
                    chatbot_messages=chatbot_messages,
                    final_world_state=final_world_state,
                    tool_events=tool_events,
                    closing_agent_response=closing_agent_response,
                    closing_agent_state=closing_agent_state,
                    closing_turn_messages=closing_turn_messages,
                    closing_agent_actions=closing_agent_actions,
                    closing_reply_source=closing_reply_source,
                    ended_early=ended_early,
                    resume_fidelity=resume_fidelity,
                    pipeline_inputs=pipeline_inputs,
                    pipeline_outputs=pipeline_outputs,
                    pipeline_error=pipeline_error,
                    user_action_result=user_action_result,
                    unread_inputs=unread_inputs,
                    post_seed_world_state=post_seed_world_state,
                    overrules=overrules,
                ),
                endpoint='/threads/{id}/finalize',
            )
        except ValidationError as exc:
            raise OrchestratorAdapterError(
                f'Async finalize submit for thread {thread_id} returned an invalid response (schema mismatch or missing required field): {exc}',
                thread_id=thread_id,
            ) from exc

        # 2. Record in-memory so cancel-on-abandon can reach it without file I/O.
        # Exception-safety note: if finalize_job_status raises mid-poll (404, retry
        # exhaustion, etc.), _inflight_finalize_job_id is intentionally left set —
        # the error path in _run_one calls _cancel_inflight_finalize(), which
        # best-effort-DELETEs the still-running job.
        self._inflight_finalize_job_id = submit.job_id
        self.last_finalize_job_id = submit.job_id

        # 3. Poll until terminal or wallclock budget exhausted.
        deadline = time.monotonic() + max_wallclock
        # Consecutive poll attempts lost to transport. Counted only to GROW the wait between
        # resumes (and to say how long this has been going on in the log) — it is not a
        # separate budget: the wallclock the poll already had is the honest bound on how long
        # a scenario waits for its own verdict, whatever the reason for the wait.
        transport_failures = 0
        while True:
            # ValidationError here means the server returned a status value not in
            # the Literal set (schema drift / 'cancelled' / 'expired' / etc.).
            # Treat as an immediate terminal error — clear inflight state and raise.
            try:
                job = await self.client.finalize_job_status(thread_id, submit.job_id)
            except ValidationError as exc:
                self._inflight_finalize_job_id = None
                raise OrchestratorAdapterError(
                    f'Async finalize job {submit.job_id!r} on thread {thread_id} '
                    f'returned an invalid status response (schema drift or unrecognised '
                    f'status); treating as terminal error: {exc}',
                    thread_id=thread_id,
                ) from exc
            except httpx.HTTPError as exc:
                if not is_transient_for_idempotent(exc):
                    # An answer ABOUT the job (a 404, a 401): terminal, the inflight id left set so cancel-on-abandon runs.
                    raise
                # The CLIENT could not reach the job: a timeout after the connection was made, a name that stopped
                # resolving, or an outage that spent the request's own budget (a connection failure or a 502, 503 or 504).
                # That says nothing about the job, which is server-side, addressed by id and usually already judged, so
                # the poll resumes against the same id instead of failing the scenario and losing a verdict that is paid
                # for. A poll that never clears ends at its wallclock as FinalizePollTimeout raised from the last failure,
                # so an outage's unavailable block stays on it for the scenario's failure path to read.
                transport_failures += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Same posture as the ordinary budget exit below: leave the inflight id set
                    # so cancel-on-abandon can DELETE a job that may still be billing judges.
                    raise FinalizePollTimeout(
                        f'Async finalize poll exceeded {max_wallclock}s budget while the poll could not reach the job '
                        f'({transport_failures} consecutive transport failure(s)) '
                        f'(job {submit.job_id}, thread {thread_id})',
                        thread_id=thread_id,
                        job_id=submit.job_id,
                    ) from exc
                backoff = _finalize_resume_backoff(poll_interval, transport_failures, remaining)
                logging.getLogger(__name__).debug(
                    'thread %s: finalize job %s could not be polled (%s: %s); resuming by job id in %.1fs '
                    '(failure %d, %.0fs of the %.0fs budget left). The job is still running on the server.',
                    thread_id,
                    submit.job_id,
                    type(exc).__name__,
                    exc,
                    backoff,
                    transport_failures,
                    remaining,
                    max_wallclock,
                )
                await asyncio.sleep(backoff)
                continue
            transport_failures = 0

            if job.status == 'done':
                self._inflight_finalize_job_id = None
                if job.result is None:
                    raise OrchestratorAdapterError(
                        f'Async finalize job {submit.job_id} on thread {thread_id} returned done with no result payload',
                        thread_id=thread_id,
                    )
                return take_gate_authority(job.result)

            if job.status == 'failed':
                self._inflight_finalize_job_id = None
                if job_refused_for_credit(job):
                    # Raised as the same type a 402 on the REST routes raises, and NOT as FinalizeJobFailed: the
                    # re-drive classifier probes and may re-drive that one, and a credit refusal is never re-driven.
                    raise CreditRefused.for_job(job, thread_id=thread_id, job_id=submit.job_id)
                detail = job.error.detail if job.error else 'async finalize job failed on the server'
                # Typed (FinalizeJobFailed) so the re-drive classifier can recognise "the finalize
                # job reported failed" without reading this message. On its own it
                # means nothing about the server's health; the classifier pairs it
                # with a GET /threads/{id} probe before concluding anything.
                raise FinalizeJobFailed(
                    f'Async finalize job {submit.job_id} failed on thread {thread_id}: {detail}',
                    thread_id=thread_id,
                    job_id=submit.job_id,
                )

            # status is 'queued' or 'running' (the only remaining Literal values) —
            # check wallclock budget before sleeping.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Leave _inflight_finalize_job_id set: the job is still running on
                # the server; cancel-on-abandon will DELETE it via
                # _cancel_inflight_finalize().
                raise FinalizePollTimeout(
                    f'Async finalize poll exceeded {max_wallclock}s budget (job {submit.job_id}, thread {thread_id})',
                    thread_id=thread_id,
                    job_id=submit.job_id,
                )
            await asyncio.sleep(min(poll_interval, remaining))

    async def _cancel_inflight_finalize(self) -> None:
        """Best-effort DELETE of any in-flight async finalize job.

        Reads ``_inflight_finalize_job_id`` (set by ``_finalize_async`` after submit,
        cleared on terminal server response). No-ops when no job is in flight. Called
        from the per-scenario error path in ``run_simulation._run_one`` to stop billing
        LLM judges after the SDK has given up on a scenario.
        """
        if self._inflight_finalize_job_id is None:
            return
        job_id = self._inflight_finalize_job_id
        # Clear BEFORE the call: the cancel is best-effort and never raises, and a
        # scenario that is re-driven runs again on this same orchestrator — a job id
        # left behind would be re-cancelled later against the NEW thread.
        self._inflight_finalize_job_id = None
        await self.client.cancel_finalize_job(self.thread_id, job_id)

    # -- per-scenario observability flow (used by run()) --

    @contextlib.asynccontextmanager
    async def _scenario_scope(
        self,
        thread: CreateThreadResponse,
        *,
        agent_name: str,
        scenario_set_id: str,
        scenario_folder: str | None = None,
        dataset_name: str | None = None,
        agent_version: str,
    ) -> AsyncIterator[RunContext]:
        """Per-scenario sink lifecycle as one ``async with`` scope (the contextlib pattern).

        ENTER opens the active sink for this scenario via the sink's ``prepare_run`` and
        yields the sink-specific run context (``enabled=False`` when no sink is active —
        ``prepare_run`` never returns ``None``).
        EXIT calls the sink's idempotent :meth:`~ObservabilitySink.drain_run` in a
        ``finally`` — so the per-scenario drain runs on the normal path AND when the
        ``async with`` body raises (child crash, HTTP error). This is what guarantees an
        eagerly-opened sink span (e.g. the Braintrust thread ROOT span) is ended even on
        the crash path, where ``post_feedback`` never runs and the drains coupled into the
        happy-path seams therefore never fire.

        ``drain_run`` is idempotent (Braintrust's ``slot.ended`` guard / Arize's repeat-safe
        ``force_flush`` / Langfuse's + LangSmith's repeat-safe client flush), so firing it
        here AFTER :meth:`_emit_observability`'s own per-seam drain on the happy path is
        harmless. The drain is also guarded below so a teardown error can NEVER mask an
        exception propagating out of the body.
        """
        # Open the active sink for this scenario. SINK-AGNOSTIC dataset cascade: prefer the
        # EXPLICIT ``dataset_name`` the caller threads in (the backend set's dataset name —
        # the SAME name the dataset-upload path targets, so tracing and upload register
        # against ONE provider dataset); else fall back to the generic experiment identity
        # (the server-side ``experiment`` metadata key) and then the SCENARIO-SET id.
        # Falling back to ``scenario_set_id`` (NOT ``agent_name``) keeps per-scenario
        # prepare_run's dataset aligned with what the set-level experiment / dataset upload
        # use, so ``agent_name`` stays a pure agent label. The LangSmith sink keeps its own
        # ``cfg.dataset`` (= ``LANGSMITH__DATASET``) taking PRIORITY over this passed value,
        # so an explicit config dataset still wins.
        # ``prepare_run`` is async + self-guarding (it runs blocking SDK work via
        # ``asyncio.to_thread`` and swallows its own errors) and ALWAYS returns a context
        # (``enabled=False`` when disabled / on a swallowed setup failure) — never ``None``.
        # The wire model degrades default_factory dict/list fields to ``| None`` (OpenAPI
        # has no default_factory), so guard the factory fields with ``or {}`` at the read
        # sites: the server always populates them, but a sparse/None payload must not break
        # the .get(...) / dict-payload usage below.
        scenario_metadata = thread.scenario_metadata or {}
        scenario_name = scenario_metadata.get('scenario_name', thread.thread_id)
        experiment_name = scenario_metadata.get('experiment', '')
        dataset_name = dataset_name or scenario_metadata.get('experiment') or scenario_set_id
        run_ctx = await self.observability.prepare_run(
            agent_name=agent_name,
            experiment_name=experiment_name,
            # The scenario's TWO identities, both forwarded because they are NOT
            # interchangeable: ``scenario_name`` is the server's DISPLAY label (it names
            # the record ``scenario:<name>``), while ``scenario_folder`` is the backend
            # folder every sink's dataset upload stamps — the dataset JOIN key. Passing
            # only the name empties the vendor's Experiments grid.
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            thread_id=thread.thread_id,
            dataset_name=dataset_name,
            agent_version=agent_version,
            example_inputs={
                # A hop is never sent `first_talk`; advertising it would put a user message the run
                # never delivered in the vendor's Inputs column. The hop's inputs are already
                # carried below as `initial_agent_context` — the same object — so this arm only
                # DROPS the opener rather than restating one dict under two names.
                # And DROPPED again when there is no opening line to advertise. The
                # readers are the observability sinks, where this lands in the dataset Example's
                # INPUTS (`observability/langsmith.py`): a ``first_talk: null`` there is a field
                # claiming a user turn whose text nobody recorded, which is worse than its absence —
                # absent is already the shape every pipeline run ships past these readers. ``''`` is
                # no better: a blank first_talk IS the agent-first shape (the predicate the backend
                # resolves its side with), so advertising one would file an agent-first run as a
                # user turn that said nothing, in the vendor's Inputs column.
                #
                # The question here is NOT quite the turn loop's. That one asks "is there an opener
                # at all" and a document-only opener (text on ``initial_messages``, nothing on
                # ``initial_message``) answers YES. This field carries the opener's TEXT, so it asks
                # whether there IS text — and for a document-only opener there is none to put here.
                **({} if self.scenario_shape == PIPELINE_SHAPE or not (thread.initial_message or '').strip() else {'first_talk': thread.initial_message}),
                # No second ``or {}``: ``run()`` normalized both onto ``thread`` before it
                # entered this scope (its only call site), so these ARE the dicts the child
                # and transcript.json's ``injected_inputs`` carry — one source of truth.
                'initial_world_state': thread.initial_world_state,
                'initial_agent_context': thread.initial_agent_context,
                # ALLOW-LIST filtered: the dataset Example input snapshot is shown in the
                # vendor UI's Inputs column, so the raw scenario_metadata's prompt-engineering
                # internals and the backend's ``experiment`` run handle must NOT
                # ride along — same allow-list the record metadata uses, minus run ids.
                'scenario_metadata': filter_scenario_metadata(scenario_metadata),
            },
        )
        try:
            yield run_ctx
        finally:
            # Defense in depth: a drain failure must NEVER replace an exception
            # propagating out of the scenario body. Each sink's ``drain_run`` is already
            # self-guarding, but guard here too (mirrors ``aclose()``'s guard around
            # ``close()``) so no single sink's discipline can turn a real child crash
            # into a confusing teardown error.
            try:
                await self.observability.drain_run(run_ctx)
            except Exception:
                import logging  # noqa: PLC0415

                logging.getLogger(__name__).exception(
                    'Observability drain_run failed at scenario-scope exit; scenario flow unaffected.',
                )

    def _model_calls(
        self,
        messages: list[dict[str, Any]],
        *,
        input_user_message: str | list[dict[str, Any]] | None,
        seen_message_ids: set[str],
    ) -> list[ModelCall]:
        """The turn's model calls (:func:`derive_model_calls`), or ``[]`` when no sink is held.

        Only the trace sinks read them, and each call copies the turn's messages before it, so
        a run with observability off (the :class:`NoOpSink` itself) does not build them at all. An
        exact type test: a sink that subclasses ``NoOpSink`` to override a role may still read them.
        """
        if type(self.observability) is NoOpSink:
            return []
        return derive_model_calls(messages, input_user_message=input_user_message, seen_message_ids=seen_message_ids)

    async def _record_turn(
        self,
        thread: CreateThreadResponse,
        *,
        turn_idx: int,
        # A turn's user content is a ``str`` OR a list of content parts, and
        # this records it. The transcript keeps the PROJECTION on ``user_message`` and the list
        # beside it, so every existing reader of the artifact is unaffected.
        input_user_message: str | list[dict[str, Any]] | None,
        ai_text: str,
        agent_messages: list[dict[str, Any]],
        turn_calls: list[RecoveredToolCall],
        agent_state: dict[str, Any] | None = None,
        reply_source: str = '',
        agent_actions: list[dict[str, Any]] | None = None,
        turn_world: dict[str, Any] | None,
        user_action_result: dict[str, Any] | None,
        seen_tool_call_ids: set[str],
        seen_model_message_ids: set[str],
        user_messages: list[str | list[dict[str, Any]]],
        turns: list[TranscriptTurn],
    ) -> UserTurn:
        """Drive ONE outer-conversation user-turn over HTTP and accumulate it.

        Posts the agent's reply to ``/user-turn``, appends
        the next user message to BOTH the ``/finalize`` transcript (``user_messages``)
        and the observability turn record (``turns``) BEFORE the terminate check — the
        terminating user_turn message is a real outer-conversation message and must
        appear in both surfaces. Returns the whole ``UserTurnResponse``: the caller needs
        ``should_terminate`` to break AND ``ended_early`` to record WHY, and a bool return
        would throw the second away.

        ``agent_messages`` is the agent's serialized message chain as of this turn
        (``AgentResponse.full_messages``: tool-call AIMessages + their ToolMessage
        results). ``turn_calls`` is its recovery over the thread's call ledger
        (:func:`derive_turn_tool_calls`: ``tool`` / ``args`` / ``response`` / ``status``); ``seen_tool_call_ids``
        (mutated here) drops any call already attributed to an earlier turn, so a
        MemorySaver agent that re-lists its full history each turn renders each tool
        call exactly once, under the turn that issued it. A call with no id is never
        deduped (kept, so an orphan ToolMessage is not silently dropped). Always set
        (``[]`` when the turn issued no new tools), keeping the turn record fully typed.
        ``seen_model_message_ids`` (mutated) does the same for model calls, by assistant-message id.

        ``agent_state`` is the agent's non-message return state for THIS turn, shipped on
        the same call. The server files it under this turn's 0-based ``turn_idx``
        and accumulates it, which is what lets a ``turn_based`` judge score a non-message
        field — and what makes live grading agree with a later re-score off the saved
        transcript's ``turns[].agent_state``. ``None`` (a stateless
        agent, or an empty dict) omits the key entirely, so a stateless agent's request body
        carries no trace of the field.
        """
        turn = await self.client.user_turn(
            thread.thread_id,
            agent_response=ai_text,
            agent_state=agent_state,
            # The TYPED turn beside the reply text. ``agent_response`` may legitimately be
            # empty when these are not — an agent whose output is a render verb list acts
            # without speaking — and the server's user simulator, its ``reply_not_visible``
            # corroboration and its per-turn judge gate all read "did this turn deliver
            # anything" off the pair, never off the reply string alone.
            agent_actions=agent_actions,
            # WHERE ``agent_response`` came from: without it a turn the runner emptied and an
            # agent that genuinely said nothing would be the same bytes server-side, and the
            # judges would grade both as the agent's silence.
            reply_source=reply_source or None,
            # The narration channel: the same chain transcript.json persists, projected to
            # the wire shape so judges can read carrier narration.
            # An EMPTY chain sends nothing rather than ``[]``: a harness that returns a
            # plain string (no ``.messages``) yields no chain, and the server reads an
            # empty one as "this client cannot observe tools" — never as "zero tools ran".
            turn_messages=_safe_wire_turn_messages(agent_messages, thread_id=thread.thread_id),
            # The idempotency key: this turn's 0-based loop index IS the server's
            # per-turn key, so a retried /user-turn replays the stored reply instead of
            # re-driving the dialog — which is what lets the client retry a lost turn.
            turn_idx=turn_idx,
            # What the simulated user can see after this turn, and how her previous act was applied.
            turn_world=turn_world,
            user_action_result=user_action_result,
        )
        # THE LIVE TURN, read the way the OPENER is. The wire carries a turn's user side TWICE
        # and identically at both ends — ``initial_messages`` beside ``initial_message`` on the
        # thread, ``user_messages`` beside ``user_message`` here. Reading only the string would
        # deliver a document on the opener to the agent but not the same document on ANY LATER
        # TURN: the parts would be parsed off the wire and dropped, silently, and for a
        # parts-only turn the projection is ``''`` — an agent answering a question it was never
        # shown.
        #
        # The SAME projection at both halves of one wire field, never a second rule. ``or`` and not
        # a presence test, for the reason the opener's arm spells out: both fields are
        # codegen-defaulted to ``None``, so a server that omits ``user_messages`` and one that
        # sends an explicit null read identically, and an ordinary string turn is untouched.
        #
        # Through ``user_turn_content``, which is that same projection plus ONE arm: a
        # response whose ``ended_early`` declares that no user turn was produced appends NOTHING.
        # An unconditional append would put ``''`` in the carrier — and the carrier is what the
        # next agent call is taken from and what ``/finalize`` is handed as the graded transcript's
        # user side, so the agent would be called with an empty message and a phantom
        # ``HumanMessage('')`` shipped to every end-to-end judge. A turn the simulator never
        # produced is not a turn the user took.
        if (user_line := user_turn_content(turn)) is not None:
            user_messages.append(user_line)
        fresh_tool_calls = _fresh_tool_calls(turn_calls, seen_tool_call_ids)
        record = TranscriptTurn(
            turn_idx=turn_idx,
            input_user_message=input_user_message,
            agent_response=ai_text,
            output_user_message=turn.user_message,
            should_terminate=turn.should_terminate,
            turn_count=turn.turn_count,
            agent_tool_calls=fresh_tool_calls,
            agent_model_calls=self._model_calls(agent_messages, input_user_message=input_user_message, seen_message_ids=seen_model_message_ids),
        )
        if reply_source:
            # Omitted rather than defaulted when the child did not classify: an absent label means
            # "this producer did not say", which must never be readable as ``assistant``.
            record['reply_source'] = reply_source
        turns.append(record)
        return turn

    async def _emit_observability(
        self,
        thread: CreateThreadResponse,
        *,
        run_ctx: RunContext,
        turns: list[TranscriptTurn],
        finalize: FinalizeResponse,
        final_world_state: dict[str, Any] | None,
        tool_events: dict[str, Any] | None,
        ended_early: str | None = None,
        ended_early_cause: str | None = None,
    ) -> None:
        """Emit observability for this scenario in the mandated order (best-effort).

        record_transcript (builds the scenario anchor + 3-level span tree) -> post_feedback
        (evals attach to that anchor). ``record_transcript`` MUST precede ``post_feedback``
        so the anchor exists before evals attach to it. There is NO separate orchestrator
        flush: each sink couples its per-scenario drain into its own seams (Arize
        force_flushes at the end of record_transcript; Langfuse/Braintrust drain in
        post_feedback; LangSmith posts immediately); the process-teardown drain runs once
        via the sink's async ``close()``.

        Both sink methods are async + SELF-GUARDING (``swallow``): they run their
        own blocking SDK work via ``asyncio.to_thread`` and NEVER raise, so the
        orchestrator awaits them directly with no try-except wrapper. Skipped when the
        run context is disabled (``run_ctx.enabled`` is False — no sink active / the
        sink's per-scenario setup did not go live). The rollup consumes the RUN-LOCAL
        ``final_world_state`` + ``tool_events`` (NEVER ``ThreadResult.tool_events``).

        ``ended_early`` rides the scenario metadata both seams already carry (see
        ``eval_mapping.RUN_FACT_KEYS``): a scenario the harness cut short shows up in a sink
        as an ordinary short one, and its evals are low because obligations were never
        reached — a verdict nothing beside it qualified. ``ended_early_cause`` rides with it,
        only when a reason does: a cause with no reason would be a claim about nothing.
        """
        if not run_ctx.enabled:
            return
        # Out-of-band ``code`` / ``cli_http`` tool calls never reach the agent's
        # (consolidated) message chain, so ``_record_turn`` leaves their turns'
        # ``agent_tool_calls`` empty and the sinks would render the ``agent`` span
        # with NO ``tool`` children. The calls survive on the mock wire as
        # ``tool_events``; graft them onto the matching turns HERE (post-finalize,
        # where both ``turns`` and ``tool_events`` are in scope) so Langfuse/LangSmith
        # render one ``tool`` child per event. Mutates ``turns`` in place; only fills
        # turns whose message-recovery produced none, so the semantic-``tools`` channel
        # (already populated) is never double-rendered.
        attribute_tool_events_to_turns(turns, tool_events)
        scenario_metadata = {**(thread.scenario_metadata or {}), **self._run_facts}
        if ended_early:
            scenario_metadata = {**scenario_metadata, 'ended_early': ended_early, **({'ended_early_cause': ended_early_cause} if ended_early_cause else {})}
        # a vendor client uploads what it is handed: no credential rides it
        thread, finalize, turns, final_world_state, tool_events, scenario_metadata = redact_value(
            (thread, finalize, turns, final_world_state, tool_events, scenario_metadata)
        )
        await self.observability.record_transcript(
            run_ctx,
            thread_id=thread.thread_id,
            scenario_name=scenario_metadata.get('scenario_name', thread.thread_id),
            turns=turns,
            scenario_metadata=scenario_metadata,
        )
        await self.observability.post_feedback(
            thread,
            finalize,
            run_ctx,
            final_world_state=final_world_state,
            tool_events=tool_events,
            scenario_metadata=scenario_metadata,
        )
