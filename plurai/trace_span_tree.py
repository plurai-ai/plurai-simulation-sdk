"""OpenInference span trees → the source-shaped projection the resume pin is built from.

A span-tree export records an agent run as a TREE of timed spans: the agent's run, each
reasoning step, every model call and every tool run, nested by parent and stamped with the
OpenInference semantic conventions (``openinference.span.kind``, ``llm.input_messages.N.…``,
``tool.name``, ``output.value``). Nothing in it is a message list. The conversation exists only
as what every MODEL CALL was shown and said, and each call is shown the whole history again —
so the thread has to be RECONSTRUCTED, and the reconstruction has to be a deterministic walk
whose every step either reads a value verbatim or refuses by name. This module is that walk.

What it reads, in one envelope —
``{"trace_id": …, "spans": [<span with recursive child_spans>]}`` (a flat span list carrying
``parent_span_id`` links is the same document without the nesting):

* ``LLM`` spans, in start-instant order, contribute the conversation. A call's INPUT messages
  are its framework replaying the history it keeps: the ``user`` messages in them are the
  CONTEXT, compared as a prefix against the context already carried, and only the tail beyond
  that prefix is new (each new user turn). The system prompt is carried from the first call; a
  later call shown a different one is reported, not carried twice. Every other input role is a
  replay of something an earlier span already produced and is not carried twice — except on the
  FIRST call, where a replay of structured assistant and tool messages is history from before the
  capture, carried verbatim because those are the inputs the model saw. A call's OUTPUT message is
  what the model said, tool calls included, read from the attributes — the only place those calls
  exist with their ids.
* Not every model call is the conversation's. A guardrail, a router or a title generator is a
  model call of its own whose context is not a prefix of the conversation's, so before the walk
  the calls are split by LINEAGE (:func:`select_lineage`): the conversation is the largest chain of
  calls each shown a prefix of what the latest of them was shown. A call off it that made no tool
  call is left out and reported; one that made tool calls is a second conversation with its own
  tool spans, and is refused.
* ``TOOL`` spans contribute the tool results, each paired IN ORDER with the calls the last model
  output made, by name. The pairing is what turns a span tree into a chat-native
  ai(tool_calls) → tool(tool_call_id) exchange; a tool that ran without a model call to answer,
  or a call no tool span answered, has no chat-native shape and is refused rather than invented.
* Every other span (``AGENT``, ``CHAIN``, a harness wrapper with no kind) is structure. Their
  payloads restate what the model calls already carry — message attributes included, so a CHAIN
  span mirroring its child model call's ``llm.input_messages.*`` contributes nothing and the
  mirrored turns cannot enter the thread twice. Not reading them IS the parent/child
  de-duplication.

One tool call is not a step inside a turn but the turn's ENDING. A ``ToolCallingAgent`` closes
every turn by calling ``final_answer(answer=…)``, and the tool's own span echoes that text back
verbatim as its ``output.value``: the pair is the agent's closing message, not an exchange the
agent had to act on. So a reply whose calls are exactly one call to a terminal answer tool is
FOLDED — it becomes a plain assistant message whose content is the echoed answer, the call and
its TOOL span both consumed, no tool message emitted. The fold is taken only when its whole shape
holds — the reply's single call, no text of its own, the single ``answer`` string argument, and an
echo equal to it. Anything else is carried as what it verbatim is: an ordinary call and its result.

Ids are minted for every message, under the same namespace and recipe the adapter layer uses:
OpenInference records none, and the pin requires them.

What it refuses, all by name: a document with no single root (a dangling ``parent_span_id`` is a
side graph, not a tree), spans whose start stamps are only partly present, a convention this
reader does not know (``gen_ai.*`` / OpenLLMetry attributes carry a different message encoding
that has never been measured against a real capture; a closest-fit mapping would be a guess
wearing a verified reader's name), a model call off the lineage that made tool calls (a nested or
managed agent's own conversation, or a rewritten history), replayed history from before the
capture that is only a text rendering of it, and any content shape beyond text.

smolagents renders its memory into later inputs as ``tool-call`` /
``tool-response`` TEXT messages (no structured ids) — which is why the replay roles include them,
and why the structured calls are read from the output attributes and never from a replay.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from plurai.gates.apply import make_finding
from plurai.gates.catalogue import (
    TRACE_ARTIFACT_HOLDS_MANY_TRACES,
    TRACE_SPAN_LINEAGE_AMBIGUOUS,
    TRACE_SPAN_SIDE_CALLS_EXCLUDED,
    TRACE_SPAN_SYSTEM_CONTEXT_VARIED,
)
from plurai.gates.contract import GateFinding
from plurai.trace_adapter import MINTED_ID_NAMESPACE
from plurai.trace_convert import FormatUnsupportedError, SourceProjection, ThreadRefused, TraceConversionError, read_source_text

#: The attribute that says a span was stamped by an OpenInference instrumentor, and what it is.
SPAN_KIND_ATTRIBUTE = 'openinference.span.kind'

#: How deep a span tree may nest. Well past what an agent run produces (steps inside a run inside
#: a harness span, a managed agent or two deeper), and far short of the stack this walk would
#: exhaust — so a document that runs away is refused by name rather than as a RecursionError.
_MAX_SPAN_DEPTH = 200

#: Attribute prefixes of the conventions this reader recognises and does NOT read. Named so the
#: refusal can say which one it saw, rather than "no OpenInference kinds" for a trace that is
#: plainly an LLM trace in a different dialect. ORDERED, and the order is the claim: OpenLLMetry
#: also stamps ``gen_ai.system`` and its legacy ``gen_ai.prompt.N.*`` messages, so its own markers
#: are checked first and a span carrying both is named for the instrumentor that wrote it.
_UNREAD_CONVENTION_PREFIXES: dict[str, tuple[str, ...]] = {
    'OpenLLMetry': ('llm.request.', 'llm.usage.', 'traceloop.'),
    'OpenTelemetry GenAI semantic conventions': ('gen_ai.',),
    # Langfuse's own OTel exporter namespaces everything it writes — observations, trace metadata,
    # experiment attributes — under this one prefix, which this SDK writes elsewhere itself.
    'Langfuse': ('langfuse.',),
}

#: Tools whose call ENDS a turn instead of taking a step inside one. The smolagents convention:
#: a ``ToolCallingAgent`` closes every turn with ``final_answer(answer=…)`` and the tool echoes
#: that text straight back, so the call and its result are one assistant message.
_TERMINAL_ANSWER_TOOLS = frozenset({'final_answer'})

#: The single argument a terminal answer call carries. Anything else is a tool that shares the
#: name and not the convention, and the name alone is not enough to fold on.
_TERMINAL_ANSWER_ARGUMENT = 'answer'

#: How much of a raw tool argument or result a refusal quotes. Enough to tell two calls apart,
#: short enough that a customer payload does not land whole on a terminal or in a CI log.
_QUOTED_LIMIT = 120


def _shorten(text: str) -> str:
    """``text`` cut to the quoting limit, saying how much was left out when it was."""
    if len(text) <= _QUOTED_LIMIT:
        return text
    return f'{text[:_QUOTED_LIMIT]}… (+{len(text) - _QUOTED_LIMIT} more characters)'


@dataclass(frozen=True)
class SpanRecord:
    """One span, flattened out of its document: the fields the walk reads and nothing else."""

    span_id: str
    parent_span_id: str | None
    name: str
    #: ``openinference.span.kind`` when stamped; ``None`` for a harness span no instrumentor made.
    kind: str | None
    #: The parsed start instant, or ``None`` when the span carries no ``timestamp``.
    start: datetime | None
    attributes: Mapping[str, Any]
    scope_name: str
    scope_version: str
    #: Pre-order position in the document — the tiebreak between equal instants, and the whole
    #: order when no span is stamped.
    document_index: int

    def at(self, source: str) -> str:
        return f'{source} span {self.span_id} ({self.name})'


@dataclass(frozen=True)
class _Convention:
    """How one attribute convention spells a message — declared, so the walk stays generic."""

    name: str
    input_messages: str
    output_messages: str
    #: Input roles that carry the CONTEXT (what the user and the operator said), mapped to the
    #: pin's type for each.
    context_roles: Mapping[str, str]
    #: Input roles that replay an earlier model output — the framework's memory of its own turns.
    assistant_replay_roles: frozenset[str]
    #: Input roles that replay an earlier tool result.
    tool_replay_roles: frozenset[str]
    #: The replay roles that carry a STRUCTURED message — an assistant reply with its tool calls and
    #: their ids, a tool result naming the call it answers — rather than a text rendering of one. Only
    #: these can be carried as history from before the capture.
    structured_assistant_role: str
    structured_tool_role: str
    #: Where a structured tool replay names the call it answers.
    tool_call_id: str
    output_role: str
    tool_name: str
    tool_output: str
    #: The span's input recorded as one flat value instead of as messages. On a TOOL span it is
    #: the call that span served — the only way to tell parallel calls to one tool apart. On an
    #: LLM span it is a prompt string: a rendering of a thread rather than the thread.
    flat_input: str


OPENINFERENCE = _Convention(
    name='openinference',
    input_messages='llm.input_messages',
    output_messages='llm.output_messages',
    context_roles={'system': 'system', 'developer': 'system', 'user': 'human'},
    # `tool-call` / `tool-response` are how smolagents renders its memory (measured on the live
    # captures); `assistant` / `tool` are the convention's own spellings of the same replay.
    assistant_replay_roles=frozenset({'assistant', 'tool-call'}),
    tool_replay_roles=frozenset({'tool', 'tool-response'}),
    structured_assistant_role='assistant',
    structured_tool_role='tool',
    tool_call_id='tool_call_id',
    output_role='assistant',
    tool_name='tool.name',
    tool_output='output.value',
    flat_input='input.value',
)


# --------------------------------------------------------------------------- #
# Document → flat, checked, ordered span records                               #
# --------------------------------------------------------------------------- #


def _string_or_none(value: Any, *, at: str, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TraceConversionError(f'{at}: {key} is {type(value).__name__}, not a string.')
    return value


def _flatten(raw_spans: Any, *, source: str, enclosing: str | None, out: list[SpanRecord], trace_ids: set[str], where: str, depth: int = 0) -> None:
    """Pre-order walk of a span list, nested ``child_spans`` included, into ``out``."""
    if depth > _MAX_SPAN_DEPTH:
        raise TraceConversionError(
            f'{source}: {where} nests more than {_MAX_SPAN_DEPTH} levels deep. A real agent run nests a handful; '
            'a document this deep is a generated or malformed one, and following it would exhaust the stack instead '
            'of saying so.'
        )
    if not isinstance(raw_spans, list):
        raise TraceConversionError(f'{source}: {where} is {type(raw_spans).__name__}, not a list of spans.')
    for position, raw in enumerate(raw_spans):
        at = f'{source} {where}[{position}]'
        if not isinstance(raw, Mapping):
            raise TraceConversionError(f'{at}: is {type(raw).__name__}, not a span object.')
        span_id = raw.get('span_id')
        if not isinstance(span_id, str) or not span_id:
            raise TraceConversionError(f'{at}: has no non-empty "span_id"; a span nothing can refer to cannot be placed in a tree.')
        at = f'{source} span {span_id}'
        declared_parent = _string_or_none(raw.get('parent_span_id'), at=at, key='"parent_span_id"')
        if 'parent_span_id' in raw and enclosing is not None and declared_parent != enclosing:
            raise TraceConversionError(
                f'{at}: is nested under span {enclosing} but declares parent_span_id {declared_parent!r} — '
                'the nesting and the links disagree, so the tree is ambiguous.'
            )
        parent = declared_parent if 'parent_span_id' in raw else enclosing
        if isinstance(raw.get('trace_id'), str) and raw['trace_id']:
            trace_ids.add(raw['trace_id'])
        attributes = raw.get('span_attributes', {})
        if not isinstance(attributes, Mapping):
            raise TraceConversionError(f'{at}: "span_attributes" is {type(attributes).__name__}, not an object.')
        name = raw.get('span_name')
        if not isinstance(name, str):
            raise TraceConversionError(f'{at}: has no string "span_name".')
        kind = attributes.get(SPAN_KIND_ATTRIBUTE)
        if kind is not None and not isinstance(kind, str):
            raise TraceConversionError(f'{at}: {SPAN_KIND_ATTRIBUTE} is {type(kind).__name__}, not a string.')
        stamp = raw.get('timestamp')
        start: datetime | None = None
        if stamp is not None:
            if not isinstance(stamp, str):
                raise TraceConversionError(f'{at}: "timestamp" is {type(stamp).__name__}, not an ISO-8601 string.')
            try:
                start = datetime.fromisoformat(stamp)
            except ValueError as e:
                raise TraceConversionError(f'{at}: "timestamp" {stamp!r} is not an ISO-8601 instant ({e}).') from e
            if start.tzinfo is None:
                # A naive stamp has no instant: two spans stamped in different local zones would
                # compare by wall clock, and the thread would come out in the wrong order silently.
                raise TraceConversionError(f'{at}: "timestamp" {stamp!r} carries no UTC offset, so it names no instant to order by.')
        out.append(
            SpanRecord(
                span_id=span_id,
                parent_span_id=parent,
                name=name,
                kind=kind,
                start=start,
                attributes=attributes,
                scope_name=_string_or_none(raw.get('scope_name'), at=at, key='"scope_name"') or '',
                scope_version=_string_or_none(raw.get('scope_version'), at=at, key='"scope_version"') or '',
                document_index=len(out),
            )
        )
        children = raw.get('child_spans')
        if children is not None:
            _flatten(children, source=source, enclosing=span_id, out=out, trace_ids=trace_ids, where=f'span {span_id} child_spans', depth=depth + 1)


def load_span_document(text: str, *, source: str, source_format: str) -> tuple[str | None, list[SpanRecord]]:
    """The document's trace id and its spans, flattened in pre-order — the shape checks only.

    A document is the ``{trace_id, spans}`` envelope or a bare span list. Every span must name
    itself; ids must be unique; a span that names a trace other than the envelope's belongs to
    a different capture, and a file that mixes captures is refused rather than split.
    """
    try:
        document = json.loads(text)
    except json.JSONDecodeError as e:
        raise TraceConversionError(f'{source}: is not valid JSON ({e}). A span-tree export is a JSON document.') from e
    if isinstance(document, list):
        envelope_trace_id: str | None = None
        raw_spans: Any = document
    elif isinstance(document, Mapping):
        envelope_trace_id = _string_or_none(document.get('trace_id'), at=source, key='"trace_id"')
        if 'spans' not in document:
            raise TraceConversionError(
                f'{source}: has no "spans" key (top-level keys: {sorted(map(str, document))[:8]}). '
                'A span-tree export is {"trace_id", "spans": [...]} or a bare span list.'
            )
        raw_spans = document['spans']
    else:
        raise TraceConversionError(f'{source}: is {type(document).__name__}, not a span-tree document.')

    spans: list[SpanRecord] = []
    trace_ids: set[str] = {envelope_trace_id} if envelope_trace_id else set()
    _flatten(raw_spans, source=source, enclosing=None, out=spans, trace_ids=trace_ids, where='spans')
    if not spans:
        raise TraceConversionError(f'{source}: holds no spans; there is no run to read.')
    if len(trace_ids) > 1:
        raise ThreadRefused(
            make_finding(
                TRACE_ARTIFACT_HOLDS_MANY_TRACES,
                subject=source,
                evidence={'source_format': source_format, 'trace_ids': sorted(trace_ids)},
                observed={},
                remedy=(
                    f'{source}: spans name {len(trace_ids)} different trace ids {sorted(trace_ids)}; one export holds one trace. '
                    'Export, or split, one file per trace.'
                ),
                draft_ref=None,
            )
        )

    seen: set[str] = set()
    for span in spans:
        if span.span_id in seen:
            raise TraceConversionError(f'{source}: span id {span.span_id} appears twice; a tree cannot hold the same span in two places.')
        seen.add(span.span_id)
    return next(iter(trace_ids), None), spans


def _root(spans: list[SpanRecord], *, source: str) -> SpanRecord:
    """The tree's single root, or a refusal naming every root and every dangling link."""
    known = {span.span_id for span in spans}
    dangling = [span for span in spans if span.parent_span_id is not None and span.parent_span_id not in known]
    if dangling:
        shown = ', '.join(f'{span.span_id} ({span.name}) → {span.parent_span_id}' for span in dangling[:5])
        raise TraceConversionError(
            f'{source}: {len(dangling)} span(s) name a parent that is not in this document: {shown}. '
            'A span with an absent parent is the root of a SIDE GRAPH, not part of this tree — '
            'export the whole trace, or drop the spans that belong to another one.'
        )
    roots = [span for span in spans if span.parent_span_id is None]
    if len(roots) != 1:
        shown = ', '.join(f'{span.span_id} ({span.name})' for span in roots[:5])
        raise TraceConversionError(
            f'{source}: has {len(roots)} root spans [{shown}]; one run is one tree with one root. '
            'Two roots are two traces in one file, and this reader will not pick between them.'
        )
    return roots[0]


def _ordered(spans: list[SpanRecord], *, source: str) -> tuple[list[SpanRecord], str]:
    """Spans by start INSTANT (document order as tiebreak), and the name of the order used.

    Instants, not strings: ``…T15:00:00+03:00`` is earlier than ``…T12:30:00Z`` and sorts after
    it lexicographically. Tri-state on purpose — every span stamped, or none (document order is
    then the export's own); a PARTIAL stamping leaves two orders and no way to choose.
    """
    stamped = [span for span in spans if span.start is not None]
    if not stamped:
        return list(spans), 'document'
    if len(stamped) != len(spans):
        missing = next(span for span in spans if span.start is None)
        raise TraceConversionError(
            f'{source}: {len(stamped)} of {len(spans)} spans carry a "timestamp" ({missing.at(source)} does not); '
            'stamp every span or none, because a partly stamped tree has two orders.'
        )
    return sorted(spans, key=lambda span: (span.start, span.document_index)), 'start_instant'  # type: ignore[arg-type,return-value]


def _detect_convention(spans: list[SpanRecord], *, source: str) -> _Convention:
    """OpenInference, or a refusal naming the convention that was seen instead."""
    if any(span.kind is not None for span in spans):
        return OPENINFERENCE
    for convention, prefixes in _UNREAD_CONVENTION_PREFIXES.items():
        hit = next((span for span in spans if any(key.startswith(prefixes) for key in span.attributes)), None)
        if hit is not None:
            raise FormatUnsupportedError(
                f'{source}: no span carries {SPAN_KIND_ATTRIBUTE}, but {hit.at(source)} carries {convention} attributes. '
                f'This reader reads OpenInference; {convention} spells messages differently and has not been measured '
                'against a real capture, so it is refused rather than mapped by resemblance.',
                convention=convention,
            )
    raise TraceConversionError(
        f'{source}: no span carries {SPAN_KIND_ATTRIBUTE}. This reader reads OpenInference span trees; '
        'a trace with no instrumented model call has no conversation to reconstruct.'
    )


# --------------------------------------------------------------------------- #
# Attribute decoding (OpenInference flattens nested values into dotted keys)   #
# --------------------------------------------------------------------------- #


def _unflatten(attributes: Mapping[str, Any], *, at: str) -> dict[str, Any]:
    """``{'a.b.0.c': v}`` → ``{'a': {'b': {'0': {'c': v}}}}``; list indices stay string keys here."""
    root: dict[str, Any] = {}
    for key in sorted(attributes):
        parts = key.split('.')
        node = root
        for depth, part in enumerate(parts[:-1]):
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise TraceConversionError(f'{at}: attribute {key!r} nests under {".".join(parts[: depth + 1])!r}, which is itself a value.')
            node = child
        if isinstance(node.get(parts[-1]), dict):
            raise TraceConversionError(f'{at}: attribute {key!r} is a value and also a prefix of other attributes.')
        node[parts[-1]] = attributes[key]
    return root


def _lookup(tree: Mapping[str, Any], dotted: str) -> Any:
    node: Any = tree
    for part in dotted.split('.'):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _indexed(node: Any, *, at: str) -> list[Any]:
    """An unflattened ``N.…`` group as a list — contiguous from 0, or a refusal."""
    if node is None:
        return []
    if not isinstance(node, Mapping):
        raise TraceConversionError(f'{at}: is {type(node).__name__}, not an indexed group of attributes.')
    by_index: dict[int, Any] = {}
    for key, value in node.items():
        # `isdecimal`, not `isdigit`: superscripts and other digit-ish characters pass `isdigit`
        # and then raise out of `int()`, which would leave the reader raising ValueError instead of
        # refusing by name. And `str(int(key)) == key` on top of it, because '00' and '٣' are both
        # decimal: two spellings of one index would collide here and drop a message without a word.
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise TraceConversionError(f'{at}: index {key!r} is not a plain 0-based number, so the group holds no readable order.')
        by_index[int(key)] = value
    expected = list(range(len(by_index)))
    if sorted(by_index) != expected:
        raise TraceConversionError(f'{at}: indices {sorted(by_index)} are not contiguous from 0; a gap means a message this export lost.')
    return [by_index[index] for index in expected]


def _messages(tree: Mapping[str, Any], dotted: str, *, at: str) -> list[Mapping[str, Any]]:
    out = []
    for index, entry in enumerate(_indexed(_lookup(tree, dotted), at=f'{at} {dotted}')):
        message = entry.get('message') if isinstance(entry, Mapping) else None
        if not isinstance(message, Mapping):
            raise TraceConversionError(f'{at} {dotted}.{index}: has no "message" group.')
        out.append(message)
    return out


def _role(message: Mapping[str, Any], *, at: str) -> str:
    role = message.get('role')
    if not isinstance(role, str) or not role:
        raise TraceConversionError(f'{at}: has no "message.role".')
    return role


def _content(message: Mapping[str, Any], *, at: str, allow_missing: bool = False) -> str | list[dict[str, Any]]:
    """``message.content`` (a string) or ``message.contents.N`` (text blocks), verbatim.

    ``allow_missing`` is for the one message that legitimately carries none: a model reply whose
    whole content is its tool calls. Everywhere else an absent content is an absent message, and
    carrying it as an empty string would put a silently blank turn into the seed.
    """
    has_text, has_blocks = 'content' in message, 'contents' in message
    if has_text and has_blocks:
        raise TraceConversionError(f'{at}: carries both "message.content" and "message.contents"; one message with two contents is not read.')
    if has_blocks:
        blocks: list[dict[str, Any]] = []
        for index, entry in enumerate(_indexed(message['contents'], at=f'{at} contents')):
            block = entry.get('message_content') if isinstance(entry, Mapping) else None
            if not isinstance(block, Mapping):
                raise TraceConversionError(f'{at} contents.{index}: has no "message_content" group.')
            kind, text = block.get('type'), block.get('text')
            if kind != 'text' or not isinstance(text, str):
                raise TraceConversionError(
                    f'{at} contents.{index}: block type {kind!r} is not carried. This reader carries text blocks; '
                    'a dropped block would be a message the model saw and the pin does not.'
                )
            blocks.append({'type': 'text', 'text': text})
        return blocks
    if has_text:
        text = message['content']
        if not isinstance(text, str):
            raise TraceConversionError(f'{at}: "message.content" is {type(text).__name__}, not a string.')
        return text
    if not allow_missing:
        raise TraceConversionError(
            f'{at}: carries neither "message.content" nor "message.contents". A message with no content is one '
            'this export did not record; an empty turn in its place would be a silent hole in the seed.'
        )
    return ''


def _tool_calls(message: Mapping[str, Any], *, at: str) -> list[dict[str, Any]]:
    """``message.tool_calls.N.tool_call.{id, function.{name, arguments}}`` as source-shaped calls.

    ``arguments`` stays the JSON string the attribute holds: the shared canonical projection is
    the ONE place tool args are parsed, for every format alike.
    """
    calls: list[dict[str, Any]] = []
    for index, entry in enumerate(_indexed(message.get('tool_calls'), at=f'{at} tool_calls')):
        call = entry.get('tool_call') if isinstance(entry, Mapping) else None
        where = f'{at} tool_calls.{index}'
        if not isinstance(call, Mapping):
            raise TraceConversionError(f'{where}: has no "tool_call" group.')
        function = call.get('function')
        if not isinstance(function, Mapping):
            raise TraceConversionError(f'{where}: has no "tool_call.function" group.')
        name, call_id, arguments = function.get('name'), call.get('id'), function.get('arguments')
        if not isinstance(name, str) or not name:
            raise TraceConversionError(f'{where}: has no non-empty "function.name".')
        if not isinstance(call_id, str) or not call_id:
            raise TraceConversionError(f'{where} ({name!r}): has no non-empty "tool_call.id" — a call with no id cannot be paired with its result.')
        if not isinstance(arguments, str):
            raise TraceConversionError(f'{where} ({name!r}): has no "function.arguments" string; the call the model made cannot be replayed without them.')
        if any(seen['id'] == call_id for seen in calls):
            raise TraceConversionError(
                f'{where} ({name!r}): repeats tool call id {call_id!r}, which an earlier call in this same reply already '
                'carries. The id is what pairs a result with the call it answers, so two calls sharing one would take '
                'turns claiming every result the tools returned.'
            )
        calls.append({'name': name, 'args': arguments, 'id': call_id})
    return calls


# --------------------------------------------------------------------------- #
# The walk's state                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PendingAnswer:
    """A closing answer call, and the reply it becomes once a TOOL span echoes its text."""

    at: str
    call: dict[str, Any]
    answer: str
    #: The reply message already in the thread, whose content the echo fills in.
    message: dict[str, Any]


@dataclass
class _Thread:
    """The reconstruction in progress: what has been carried, and what is still owed."""

    messages: list[dict[str, Any]]
    #: The (type, content) NON-SYSTEM context prefix every later model call has to replay.
    context: list[tuple[str, Any]]
    #: The system messages the first model call was shown — the ones carried.
    system: list[tuple[str, Any]] | None = None
    ai_count: int = 0
    #: Tool results the spans so far produced — a FOLDED closing answer counted among them, because
    #: the framework's memory still holds it as a tool result and replays it to every later call.
    tool_count: int = 0
    #: Tool calls the last model output made that no TOOL span has answered yet.
    pending: deque[dict[str, Any]] | None = None
    #: The closing answer call waiting for the TOOL span that says what it answered.
    folding: _PendingAnswer | None = None
    #: Replayed messages from before the capture the first model call carried.
    history_before_capture: int = 0
    folded: int = 0
    #: Lineage calls shown a system prompt other than the first one's.
    system_varied: list[SpanRecord] | None = None


# --------------------------------------------------------------------------- #
# Lineage: which model calls are the conversation's                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LineageCall:
    """One model call as lineage sees it: its id, the NON-system context it was shown, and whether it made tool calls.

    ``context`` is the ordered conversation the call was shown with every system message left out,
    as values compared by ``==`` (a reader's own ``(type, content)`` pairs, for instance).
    """

    call_id: str
    context: tuple[Any, ...]
    makes_tool_calls: bool


@dataclass(frozen=True)
class Lineage:
    """The conversation's calls and the calls beside it, as call ids in input order."""

    #: The latest call on the largest chain — the conversation as far as the trace holds it.
    anchor: str
    on_lineage: tuple[str, ...]
    #: Off-lineage calls that made no tool call: a guardrail, a router, a title — left out.
    side_calls: tuple[str, ...]
    #: Off-lineage calls that made tool calls: a second conversation with tool results of its own.
    side_calls_with_tool_calls: tuple[str, ...]
    #: Every call whose chain is as large as the anchor's, the anchor included, when there is more
    #: than one: their chains are incomparable, so which of them is the conversation cannot be told.
    #: Empty when the anchor is the only call with the largest chain.
    tied: tuple[str, ...]


def _is_prefix(shorter: tuple[Any, ...], longer: tuple[Any, ...]) -> bool:
    return len(shorter) <= len(longer) and longer[: len(shorter)] == shorter


def select_lineage(calls: Sequence[LineageCall]) -> Lineage:
    """Which of ``calls`` (in start order) are one conversation's — pure, over what each call was shown.

    A call's chain is every call up to and including it whose context is a PREFIX of its own: the
    calls a conversation makes as it grows. The anchor is the call with the largest chain; its chain
    is the lineage. A call that is not on it — one shown a different conversation, or a call after
    the anchor — is a side call. Choosing the anchor by chain size and not as the last call is what
    keeps a trailing side call (a title generated after the answer) from taking the place of a
    conversation that made more than one call.

    When two or more calls share the largest chain size — one answer and one title, a router and one
    answer — their chains are incomparable and nothing in the contexts says which is the
    conversation: recency picks a trailing title, and anything else picks a leading router. They are
    returned as :attr:`Lineage.tied`, for the caller to refuse rather than guess.
    """
    if not calls:
        raise ValueError('select_lineage needs at least one call')
    chains = [
        [earlier for earlier in range(index + 1) if _is_prefix(tuple(calls[earlier].context), tuple(anchor.context))] for index, anchor in enumerate(calls)
    ]
    largest = max(len(chain) for chain in chains)
    at_largest = [index for index, chain in enumerate(chains) if len(chain) == largest]
    best_anchor = at_largest[-1]
    best_chain = chains[best_anchor]
    on = set(best_chain)
    side = [call for index, call in enumerate(calls) if index not in on]
    return Lineage(
        anchor=calls[best_anchor].call_id,
        on_lineage=tuple(calls[index].call_id for index in best_chain),
        side_calls=tuple(call.call_id for call in side if not call.makes_tool_calls),
        side_calls_with_tool_calls=tuple(call.call_id for call in side if call.makes_tool_calls),
        tied=tuple(calls[index].call_id for index in at_largest) if len(at_largest) > 1 else (),
    )


# --------------------------------------------------------------------------- #
# The walk                                                                     #
# --------------------------------------------------------------------------- #


def _terminal_answer(call: Mapping[str, Any]) -> str | None:
    """The text a closing answer call carries, or ``None`` when its arguments are not the single ``answer`` string."""
    parsed = _parsed_arguments(call)
    if not isinstance(parsed, Mapping) or set(parsed) != {_TERMINAL_ANSWER_ARGUMENT}:
        return None
    answer = parsed[_TERMINAL_ANSWER_ARGUMENT]
    return answer if isinstance(answer, str) else None


def _inputs(tree: Mapping[str, Any], span: SpanRecord, convention: _Convention, *, at: str) -> list[tuple[str, Mapping[str, Any], str]]:
    """The call's input messages as ``(role, message, where)`` — refused when it records none, or only a flat prompt."""
    inputs = _messages(tree, convention.input_messages, at=at)
    if not inputs:
        if span.attributes.get(convention.flat_input) is None:
            raise TraceConversionError(
                f'{at}: carries no {convention.input_messages} — the model was called with nothing on its input. '
                'What the model was shown is the thread; a call with no input records none of it.'
            )
        raise TraceConversionError(
            f'{at}: records its input flat, under "{convention.flat_input}", and carries no {convention.input_messages}. '
            'A prompt string is one rendering of a thread, not the turns it was rendered from; cutting it back into '
            'speakers would be a guess at who said what.'
        )
    out = []
    for index, message in enumerate(inputs):
        where = f'{at} {convention.input_messages}.{index}'
        role = _role(message, at=where)
        if role not in convention.context_roles and role not in convention.assistant_replay_roles | convention.tool_replay_roles:
            raise TraceConversionError(
                f'{where}: role {role!r} is not one this reader knows (context: {sorted(convention.context_roles)}, '
                f'replay: {sorted(convention.assistant_replay_roles | convention.tool_replay_roles)}). '
                'An unknown speaker is never mapped by resemblance.'
            )
        out.append((role, message, where))
    return out


def _lineage_call(span: SpanRecord, convention: _Convention, *, source: str) -> LineageCall:
    """The span as :func:`select_lineage` reads it: its non-system context and whether its reply made calls."""
    at = span.at(source)
    tree = _unflatten(span.attributes, at=at)
    context = tuple(
        (convention.context_roles[role], _content(message, at=where))
        for role, message, where in _inputs(tree, span, convention, at=at)
        if role in convention.context_roles and convention.context_roles[role] != 'system'
    )
    outputs = _messages(tree, convention.output_messages, at=at)
    return LineageCall(call_id=span.span_id, context=context, makes_tool_calls=any(output.get('tool_calls') for output in outputs))


def _replayed_history(inputs: list[tuple[str, Mapping[str, Any], str]], convention: _Convention, *, at: str) -> list[dict[str, Any]]:
    """The FIRST call's inputs, in order, as messages — its replayed history from before the capture included.

    Those replays are the verbatim inputs the model saw, so they are carried when each one is
    structurally a message: an assistant reply with its text and its calls' ids, a tool result
    naming a call an earlier replayed reply made. A replay that is only a text rendering of a turn
    has no such structure, and is refused.
    """
    messages: list[dict[str, Any]] = []
    asked: set[str] = set()
    for role, message, where in inputs:
        if role in convention.context_roles:
            messages.append({'type': convention.context_roles[role], 'content': _content(message, at=where)})
        elif role == convention.structured_assistant_role:
            calls = _tool_calls(message, at=where)
            replayed: dict[str, Any] = {'type': 'ai', 'content': _content(message, at=where, allow_missing=bool(calls))}
            if calls:
                replayed['tool_calls'] = calls
                asked.update(call['id'] for call in calls)
            messages.append(replayed)
        elif role == convention.structured_tool_role:
            call_id = message.get(convention.tool_call_id)
            if not isinstance(call_id, str) or call_id not in asked:
                raise TraceConversionError(
                    f'{where}: replays a tool result from before the capture that names {call_id!r}, not a call an earlier '
                    'replayed reply made, so it cannot be carried as the result of one.'
                )
            messages.append({'type': 'tool', 'content': _content(message, at=where), 'tool_call_id': call_id})
        else:
            raise TraceConversionError(
                f'{where}: the model was shown a {role!r} message from before the capture. The conversation began before '
                'this capture, and that part of it is only a rendering the framework made of it — not readable verbatim.'
            )
    return messages


def _read_llm_span(span: SpanRecord, convention: _Convention, thread: _Thread, *, source: str) -> None:
    at = span.at(source)
    tree = _unflatten(span.attributes, at=at)
    if thread.folding is not None:
        raise TraceConversionError(
            f'{at}: the model was called again, but the closing {thread.folding.call["name"]!r} call '
            f'{thread.folding.call["id"]} made by {thread.folding.at} has no TOOL span echoing its answer. '
            'That echo is the text of the reply the call is folded into, so without it the turn has no words to carry.'
        )
    if thread.pending:
        owed = [call['id'] for call in thread.pending]
        raise TraceConversionError(
            f'{at}: the model was called again while {len(owed)} tool call(s) {owed} had no TOOL span answering them. '
            'A call without a result has no chat-native shape; instrument the tools, or read a trace that ran them.'
        )

    inputs = _inputs(tree, span, convention, at=at)
    context = [(convention.context_roles[role], _content(message, at=where)) for role, message, where in inputs if role in convention.context_roles]
    system = [entry for entry in context if entry[0] == 'system']
    conversation = [entry for entry in context if entry[0] != 'system']
    if thread.system is None:
        # The first call on the lineage: everything it was shown is carried, in order.
        history = _replayed_history(inputs, convention, at=at)
        thread.messages.extend(history)
        thread.history_before_capture = sum(message['type'] in ('ai', 'tool') for message in history)
        thread.ai_count = sum(message['type'] == 'ai' for message in history)
        thread.tool_count = sum(message['type'] == 'tool' for message in history)
        thread.system = system
        thread.context = conversation
    else:
        replayed_assistant = sum(role in convention.assistant_replay_roles for role, _message, _where in inputs)
        replayed_tool = sum(role in convention.tool_replay_roles for role, _message, _where in inputs)
        if replayed_assistant > thread.ai_count or replayed_tool > thread.tool_count:
            raise TraceConversionError(
                f'{at}: the model was shown {replayed_assistant} assistant and {replayed_tool} tool message(s), but the spans before it '
                f'produced {thread.ai_count} and {thread.tool_count}. The conversation began before this capture, and the part '
                'that did is only a rendering the framework made of it — not readable verbatim.'
            )
        if conversation[: len(thread.context)] != thread.context:
            diverged = next(
                (index for index, (seen, replayed) in enumerate(zip(thread.context, conversation, strict=False)) if seen != replayed), len(conversation)
            )
            raise TraceConversionError(
                f'{at}: replays a context that diverges from the thread at context message {diverged}. '
                'A model call inside a nested or managed agent, or a rewritten history, is a different conversation; '
                'this reader carries one thread and will not merge two.'
            )
        if system != thread.system:
            thread.system_varied = [*(thread.system_varied or []), span]
        for kind, content in conversation[len(thread.context) :]:
            thread.messages.append({'type': kind, 'content': content})
        thread.context = conversation

    outputs = _messages(tree, convention.output_messages, at=at)
    if len(outputs) != 1:
        raise TraceConversionError(
            f'{at}: carries {len(outputs)} output messages; one model call is one reply. '
            + ('A call that produced nothing has no turn to carry.' if not outputs else 'Several candidates are not a conversation.')
        )
    where = f'{at} {convention.output_messages}.0'
    role = _role(outputs[0], at=where)
    if role != convention.output_role:
        raise TraceConversionError(f'{where}: the output role is {role!r}, not {convention.output_role!r}; a reply that is not the assistant is not read.')
    calls = _tool_calls(outputs[0], at=where)
    content = _content(outputs[0], at=where, allow_missing=bool(calls))
    reply: dict[str, Any] = {'type': 'ai', 'content': content}
    # Folded only when the whole convention holds: one call, to a terminal answer tool, with no text
    # of its own and the single answer string. Any other reply is carried as the calls it made.
    answer = _terminal_answer(calls[0]) if len(calls) == 1 and calls[0]['name'] in _TERMINAL_ANSWER_TOOLS and not content else None
    if answer is not None:
        thread.folding = _PendingAnswer(at=where, call=calls[0], answer=answer, message=reply)
    elif calls:
        reply['tool_calls'] = calls
    thread.messages.append(reply)
    thread.ai_count += 1
    thread.pending = deque(calls)


def _recorded_arguments(span: SpanRecord, convention: _Convention) -> Mapping[str, Any] | None:
    """The named arguments the TOOL span says it served, or None when it records none by name.

    The convention writes ``{"args": [...], "kwargs": {...}}``. Only ``kwargs`` names its
    arguments; a positionally called tool records values with no names, which nothing can be
    compared against a model call's JSON arguments.
    """
    raw = span.attributes.get(convention.flat_input)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, Mapping):
        return None
    positional, named = raw.get('args'), raw.get('kwargs')
    if positional or not isinstance(named, Mapping):
        return None
    return named


def _answered_call(span: SpanRecord, convention: _Convention, thread: _Thread, *, at: str, name: str) -> dict[str, Any]:
    """The unanswered call this TOOL span answers — by name, and by arguments when the name repeats.

    Parallel calls to one tool are the case position cannot settle: a framework may run them
    concurrently and record them in any order, so a queue would attach a result to the wrong
    call without a word. Two calls this reader cannot tell apart are refused, never guessed.
    """
    if not thread.pending:
        raise TraceConversionError(
            f'{at}: tool {name!r} ran, but the last model reply made no tool call for it to answer. '
            'A tool the framework ran on its own — a code-executing agent, or a tool call the instrumentor '
            'did not record — has no chat-native shape, and this reader will not invent one.'
        )
    candidates = [call for call in thread.pending if call['name'] == name]
    if not candidates:
        owed = ', '.join(f'{call["name"]!r} ({call["id"]})' for call in thread.pending)
        raise TraceConversionError(
            f'{at}: tool {name!r} ran, but no unanswered model call names it — the calls still owed are {owed}. '
            'A result is paired with the call it answers by name; a result for a call nobody made means the '
            'trace lost the call, and the thread cannot be rebuilt around the gap.'
        )
    if len(candidates) == 1:
        return candidates[0]

    served = _recorded_arguments(span, convention)
    shown = ', '.join(f'{call["id"]} {_shorten(call["args"])}' for call in candidates)
    if served is None:
        raise TraceConversionError(
            f'{at}: {len(candidates)} calls to {name!r} are unanswered [{shown}] and this span records no named '
            f'arguments under "{convention.flat_input}" to say which one it served. Pairing them by position '
            'would attach a result to the wrong call silently, so this reader stops here instead.'
        )
    matched = [call for call in candidates if _parsed_arguments(call) == served]
    if len(matched) != 1:
        raise TraceConversionError(
            f'{at}: this result of {name!r} served {_shorten(json.dumps(served, sort_keys=True, default=str))}, which matches '
            f'{len(matched)} of the {len(candidates)} unanswered calls to it [{shown}]. One result answers exactly '
            'one call; anything else is a pairing this reader would have to guess at.'
        )
    return matched[0]


def _parsed_arguments(call: Mapping[str, Any]) -> Any:
    """A call's ``arguments`` JSON as a value; unparsable arguments match nothing."""
    try:
        return json.loads(call['args'])
    except json.JSONDecodeError:
        return None


def _read_tool_span(span: SpanRecord, convention: _Convention, thread: _Thread, *, source: str) -> None:
    at = span.at(source)
    name = span.attributes.get(convention.tool_name)
    if not isinstance(name, str) or not name:
        raise TraceConversionError(f'{at}: is a TOOL span with no "{convention.tool_name}".')
    call = _answered_call(span, convention, thread, at=at, name=name)
    output = span.attributes.get(convention.tool_output)
    # An empty string IS a result: a tool that returned nothing recorded that it returned nothing,
    # and the message carries that empty text. An ABSENT attribute is an export that recorded no
    # result at all, and an empty message in its place would tell a resumed agent the tool answered.
    if not isinstance(output, str):
        raise TraceConversionError(f'{at}: tool {name!r} has no string "{convention.tool_output}"; a tool result the export did not record cannot be replayed.')
    thread.pending.remove(call)  # type: ignore[union-attr]
    folding = thread.folding
    thread.tool_count += 1
    if folding is not None and folding.call is call:
        thread.folding = None
        if output == folding.answer:
            folding.message['content'] = output
            thread.folded += 1
            return
        # The echo is not the answer the call passed, so the pair is not the convention's closing
        # reply: it is carried as the call and the result the trace recorded.
        folding.message['tool_calls'] = [call]
    thread.messages.append({'type': 'tool', 'content': output, 'tool_call_id': call['id']})


@dataclass(frozen=True)
class Reconstruction:
    """The conversation the spans record, source-shaped, and what the walk did to get it."""

    messages: list[dict[str, Any]]
    llm_spans: int
    tool_spans: int
    #: Model calls off the lineage that made no tool call, left out of the thread.
    side_calls: tuple[SpanRecord, ...]
    #: Lineage calls shown a system prompt other than the first call's, which is the one carried.
    system_varied: tuple[SpanRecord, ...]
    history_before_capture: int
    folded_terminal_answers: int


def reconstruct_thread(spans: list[SpanRecord], convention: _Convention, *, source: str) -> Reconstruction:
    """The conversation the ordered spans record, source-shaped, plus what the walk did.

    ``convention`` is one of this module's declared conventions — today ``OPENINFERENCE``, the
    only one measured against a real capture.
    """
    llm = [span for span in spans if span.kind == 'LLM']
    if not llm:
        kinds = sorted({span.kind for span in spans if span.kind})
        raise TraceConversionError(f'{source}: no LLM span among kinds {kinds}; a trace with no model call has no conversation to resume.')
    lineage = select_lineage([_lineage_call(span, convention, source=source) for span in llm])
    if lineage.tied:
        tied = [span for span in llm if span.span_id in lineage.tied]
        named = ', '.join(f'{span.span_id} ({span.name})' for span in tied)
        raise ThreadRefused(
            make_finding(
                TRACE_SPAN_LINEAGE_AMBIGUOUS,
                subject=source,
                evidence={'spans': _span_list(tied)},
                observed={},
                remedy=(
                    f'{source}: model calls {named} were each shown a context the other does not extend, and each is as '
                    'much the conversation as the other — an answer beside a title or a router, say. Pinning either would '
                    "drop the other's words as a side call. Export the conversation's run on its own."
                ),
                draft_ref=None,
            )
        )
    if lineage.side_calls_with_tool_calls:
        named = ', '.join(f'{span.span_id} ({span.name})' for span in llm if span.span_id in lineage.side_calls_with_tool_calls)
        raise TraceConversionError(
            f'{source}: model call(s) {named} replay a context that diverges from the thread the conversation carries, '
            'and made tool calls. A model call inside a nested or managed agent, or a rewritten history, is a different '
            'conversation with tool results of its own; this reader carries one thread and will not merge two.'
        )
    excluded = set(lineage.side_calls)
    thread = _Thread(messages=[], context=[])
    tool_spans = 0
    for span in spans:
        if span.kind == 'LLM' and span.span_id not in excluded:
            _read_llm_span(span, convention, thread, source=source)
        elif span.kind == 'TOOL':
            tool_spans += 1
            _read_tool_span(span, convention, thread, source=source)
    if thread.folding is not None:
        raise TraceConversionError(
            f'{source}: {thread.folding.at} ends the trace by calling {thread.folding.call["name"]!r} '
            f'({thread.folding.call["id"]}), and no TOOL span echoes its answer. The echo is the text of the reply that call '
            'is folded into, so the capture stops on a turn whose words it does not hold.'
        )
    if thread.pending:
        owed = [call['id'] for call in thread.pending]
        raise TraceConversionError(
            f'{source}: the trace ends with {len(owed)} tool call(s) {owed} that no TOOL span answered. '
            'A resume seed ending on an unanswered call would make the agent run that tool on its first turn.'
        )
    return Reconstruction(
        messages=thread.messages,
        llm_spans=len(llm),
        tool_spans=tool_spans,
        side_calls=tuple(span for span in llm if span.span_id in excluded),
        system_varied=tuple(thread.system_varied or ()),
        history_before_capture=thread.history_before_capture,
        folded_terminal_answers=thread.folded,
    )


# --------------------------------------------------------------------------- #
# The front end                                                                #
# --------------------------------------------------------------------------- #

MINTED_ID_RECIPE = 'uuid5(minted_id_namespace, "<source_file>:<message index>")'


def _span_list(spans: Sequence[SpanRecord]) -> list[dict[str, str]]:
    return [{'span_id': span.span_id, 'name': span.name} for span in spans]


def read_span_tree(source_path: Path) -> SourceProjection:
    """One span-tree export → its source-shaped projection, or a named refusal."""
    source = source_path.name
    text = read_source_text(source_path)
    trace_id, spans = load_span_document(text, source=source, source_format='openinference')
    root = _root(spans, source=source)
    convention = _detect_convention(spans, source=source)
    ordered, ordered_by = _ordered(spans, source=source)
    walk = reconstruct_thread(ordered, convention, source=source)
    messages = walk.messages

    for index, message in enumerate(messages):
        message['id'] = str(uuid.uuid5(MINTED_ID_NAMESPACE, f'{source}:{index}'))

    scopes = sorted({f'{span.scope_name} {span.scope_version}'.strip() for span in spans if span.kind == 'LLM'})
    provenance: dict[str, Any] = {}
    if trace_id:
        provenance['trace_id'] = trace_id
        # The trace IS the thread: the file is named however the exporter chose.
        provenance['thread_id'] = trace_id
    provenance.update(
        {
            'convention': convention.name,
            'instrumentation_scope': ', '.join(scopes),
            'root_span': root.name,
            'root_span_id': root.span_id,
            'span_count': len(spans),
            'llm_span_count': walk.llm_spans,
            'tool_span_count': walk.tool_spans,
            'ordered_by': ordered_by,
            'minted_message_ids': len(messages),
            'minted_id_namespace': str(MINTED_ID_NAMESPACE),
            'minted_id_recipe': MINTED_ID_RECIPE,
        }
    )
    # Each only when it happened: a capture read the plain way carries neither key, so its pin does not depend on them.
    if walk.history_before_capture:
        provenance['history_before_capture'] = walk.history_before_capture
    if walk.folded_terminal_answers:
        provenance['folded_terminal_answers'] = walk.folded_terminal_answers
    findings: list[GateFinding] = []
    if walk.side_calls:
        findings.append(
            make_finding(
                TRACE_SPAN_SIDE_CALLS_EXCLUDED,
                subject=source,
                evidence={'spans': _span_list(walk.side_calls)},
                observed={},
                remedy=(
                    f'{len(walk.side_calls)} model call(s) shown a context off the lineage of the conversation, and making no tool '
                    f'call, were left out of the thread: {", ".join(f"{span.span_id} ({span.name})" for span in walk.side_calls)}. '
                    'If one of them is the conversation, export that run on its own.'
                ),
                draft_ref=None,
            )
        )
    if walk.system_varied:
        findings.append(
            make_finding(
                TRACE_SPAN_SYSTEM_CONTEXT_VARIED,
                subject=source,
                evidence={'spans': _span_list(walk.system_varied)},
                observed={},
                remedy=(
                    f'{len(walk.system_varied)} model call(s) on the conversation were shown a system prompt other than the '
                    "first call's; the pin carries the first one."
                ),
                draft_ref=None,
            )
        )
    return SourceProjection(messages=messages, agent_state={}, provenance=provenance, findings=tuple(findings))


__all__ = [
    'MINTED_ID_RECIPE',
    'OPENINFERENCE',
    'SPAN_KIND_ATTRIBUTE',
    'Lineage',
    'LineageCall',
    'Reconstruction',
    'SpanRecord',
    'load_span_document',
    'read_span_tree',
    'reconstruct_thread',
    'select_lineage',
]
