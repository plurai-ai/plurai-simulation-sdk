"""The adapter executor: a workspace-persisted MAPPING document, executed by fixed code.

:mod:`plurai.trace_convert` keeps a registry of first-class front ends, and that registry has two
members — LangSmith and OpenInference span trees, each reader verified live against a real export.
This module is the registry's other half: the answer to every format that is NOT worth a
first-class reader,
which is most of them. A customer's traces are whatever their stack happened to write, and a
reader per stack is a reader per stack to keep verified. An unverified reader is a standing
liability.

So the two tiers divide the work by what they are good at. **The registry absorbs RECURRENCE** — a
format seen often enough to earn a live-verified reader and the maintenance that implies. **An
adapter absorbs NOVELTY** — one customer's one format, mapped once, at the cost of a file.

**The LLM authors the mapping; this code executes it.** The adapter is a declarative YAML document
describing WHERE in a source record each pinned field lives — never code. That is the whole safety
argument, and it is structural rather than promised:

* the document is read with ``yaml.safe_load`` and validated into the closed pydantic models
  below, so there is no ``exec``, no import hook, and no expression to evaluate. A model that
  authors a malicious adapter gets a validation error, not a shell;
* the executor is ordinary tested SDK code, so the same document over the same bytes produces the
  same pin on every machine — a property no model-in-the-loop can offer;
* **message bytes never pass through a model.** A model decides that ``message.content`` is where
  the text lives; the text itself is copied by :func:`_extract` and carried verbatim. This is the
  fidelity bar, kept by construction: an LLM in the verbatim path is exactly what a true resume
  forbids;
* a mapping is REVIEWABLE. A human reading twenty lines of YAML can see what a format was claimed
  to be; the same human reading twenty lines of generated Python cannot see what it does.

**There is no closest-fit anywhere.** Unlike a statistics pipeline, where a slightly-miscategorised
turn only moves a rate, this executor never maps a record by resemblance. A
speaker outside the pin's closed type set is a refusal at load time; an id field that selects two
candidates is a refusal rather than a pick. A content field that selects several values carries
them all, verbatim, as one message's content list — never joined. A record that NO rule accounts
for is left out of the conversation, COUNTED in the pin's provenance with the shapes it had, and
reported as a finding: format drift after the adapter was committed (a new metrics or debug record
kind) should not kill every conversion, and a record that was really a turn shows up by shape in
the finding, for a human to map. A structural loss — a dropped tool result — is still caught by
the shared seed-shape rule, which refuses the thread.

**The executor produces SOURCE-shaped messages, not pin messages.** It hands
:class:`~plurai.trace_convert.SourceProjection` back to the shared half, and
:func:`~plurai.trace_convert.canonical_messages` applies every rule — required non-blank ids, the
closed type set, ``[]`` content normalized to ``''``, tool args parsed from a JSON string, tool
results paired to their call. Those rules live in ONE place for every format alike; an adapter
that canonicalized its own output would be a second place for them to be almost-right.

**Two guard shapes are implemented here rather than imported from the backend**, so this module
pulls no backend package into a customer's environment: the bidirectional capability check and the
strictly-increasing ordering invariant. One of the two sits at a different scope. The backend
judges a capability across a run, which is right where the run is the corpus; here a run is
whichever files one invocation was pointed at, so judging it there would make the same bytes pass
or fail on how the command was typed. The equivalent claim is decidable against the document alone
— see :meth:`AdapterDocument._check_tool_call_declaration` — and what is NOT decidable at
conversion time is reported into pin provenance as counts rather than guessed at with a threshold.
The cost of the copy is a second implementation of the same two invariants.
"""

from __future__ import annotations

import json
import uuid
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from plurai.gates.apply import make_finding
from plurai.gates.catalogue import TRACE_ADAPTER_UNRECOGNISED_RECORDS
from plurai.gates.contract import GateFinding
from plurai.ingest.workspace_config import WORKSPACE_CANDIDATE_ROOTS
from plurai.resume_pin import CANONICAL_MESSAGE_TYPES, NonBlankStr
from plurai.trace_convert import (
    CONVERTER_PROVENANCE_KEYS,
    TRACE_FORMAT_REGISTRY,
    SourceProjection,
    ThreadRefused,
    TraceConversionError,
    _first_problem,
    _FormatSpec,
    read_source_text,
)

#: Directory holding adapter documents, resolved relative to ``--workspace-root``. Adapters are
#: COMMITTED client-side, exactly like ``plurai.config.yaml``: an adapter is a durable statement
#: about what a customer's traces are, reviewed once and reused by every run, not a per-run flag.
ADAPTER_DIR_NAME = 'plurai_adapters'

#: Discovery order, DERIVED from the workspace layout ``plurai.config.yaml`` is found by
#: (:data:`~plurai.ingest.workspace_config.WORKSPACE_CANDIDATE_ROOTS`) rather than restated beside
#: it, so a workspace has ONE layout to learn — and gaining a third root moves both at once.
_ADAPTER_CANDIDATE_RELDIRS: tuple[Path, ...] = tuple(root / ADAPTER_DIR_NAME for root in WORKSPACE_CANDIDATE_ROOTS)

#: Namespace for ids MINTED when a source records none. Fixed forever: minting is
#: ``uuid5(namespace, f'{source_file}:{index}')``, so the same artifact mints the same ids on every
#: machine and every re-run — which is what lets a re-converted pin re-seed the same conversation
#: instead of appending a second copy of it. It is recorded in provenance beside the count.
MINTED_ID_NAMESPACE = uuid.UUID('6f3b1c8e-9a2d-5e47-b0c1-7d4e2a9f5b83')

#: Provenance keys the executor stamps itself, as evidence ABOUT the adapter — which file ran, over
#: how many records, how many ids it had to mint. The whole argument for the two-tier split is that
#: an adapter cannot quietly substitute itself for a built-in reader; if a document could write
#: these, the on-disk record of that property would be authored by the thing it is supposed to be
#: evidence about.
_EXECUTOR_PROVENANCE_KEYS = frozenset(
    {
        'adapter',
        'adapter_file',
        'record_count',
        'discarded_record_count',
        'minted_message_ids',
        'minted_id_namespace',
        'minted_id_recipe',
        'ordinal_resolved_count',
        'tool_call_rule_matches',
        'tool_call_resolved_count',
        'merged_record_count',
    }
)

#: Provenance keys a document may not declare: the executor's own stamps, and every key the
#: converter computes about the pin it builds (:data:`~plurai.trace_convert.CONVERTER_PROVENANCE_KEYS`).
#: A declared value for one of the converter's keys survives exactly when the converter stamps
#: nothing, so the pin would claim a cut, a trim, a state or a turn numbering that no step produced.
#: ``thread_id`` is absent on purpose: it identifies the ARTIFACT, not the reader, and a format that
#: records its own thread identity is entitled to say so (see ``convert_thread_export``).
RESERVED_PROVENANCE_KEYS = _EXECUTOR_PROVENANCE_KEYS | CONVERTER_PROVENANCE_KEYS


class AdapterError(TraceConversionError):
    """An adapter document is unusable, or an artifact does not match what it declares.

    A subclass rather than a bare :class:`~plurai.trace_convert.TraceConversionError` so a caller
    can tell "your mapping is wrong" from "your export is wrong" — the two have different owners
    and, once a skill authors adapters, different remedies.
    """


class AdapterThreadRefused(ThreadRefused, AdapterError):  # noqa: N818 - a ThreadRefused, named as one
    """An artifact this adapter cannot project: THAT thread is refused, and the run goes on.

    Both at once on purpose: it is one thread's refusal, carrying its finding, and it is still an
    :class:`AdapterError` — the export does not match the mapping — rather than a defect of the
    reader that ran it.
    """


#: How many distinct shapes of unrecognised record a pin's provenance and finding name.
UNRECOGNISED_SHAPES_SHOWN = 8


# --------------------------------------------------------------------------- #
# The path language                                                            #
# --------------------------------------------------------------------------- #


def _resolve_path(value: Any, path: str) -> list[Any]:
    """Every value reachable at ``path``, in document order — ``[]`` when nothing is.

    Dotted segments, and three kinds of them: a dict key, an integer index into a list, and ``*``
    which maps over a list. A LIST is returned even for a path that can only match once, because
    "absent", "present once" and "present many times" are three different answers and each has a
    different correct handling upstream — collapsing them to ``None`` is how a missing field turns
    into a null one field later.

    Deliberately not JSONPath or jq: this resolves, it does not compute. There is no arithmetic, no
    function call, no slicing and no recursive descent, so a mapping cannot express a
    transformation of the bytes it names — only where they are.
    """
    current = [value]
    for segment in path.split('.'):
        nxt: list[Any] = []
        for item in current:
            if segment == '*':
                if isinstance(item, list):
                    nxt.extend(item)
            elif isinstance(item, dict):
                # Checked BEFORE the integer branch, so a format whose keys are digit strings is
                # read as the mapping it is rather than indexed as a list it is not.
                if segment in item:
                    nxt.append(item[segment])
            elif isinstance(item, list) and _is_index(segment):
                index = int(segment)
                if -len(item) <= index < len(item):
                    nxt.append(item[index])
        if not nxt:
            return []
        current = nxt
    return current


def _is_index(segment: str) -> bool:
    """Whether ``segment`` is an integer index this resolver may apply to a list.

    Narrower than ``segment.lstrip('-').isdigit()``, which accepts strings ``int()`` then rejects
    (``'--1'`` strips to ``'1'``) and non-ASCII digits ``int()`` reinterprets (``'٣'``). Either
    would escape as a bare ``ValueError`` — a traceback rather than a refusal, from a module whose
    whole posture is that a bad document produces a sentence naming what is wrong with it.

    A segment that is not an index and not a key simply resolves to nothing, and an absent value is
    already refused by :func:`_extract` unless the document declared ``missing:``.
    """
    body = segment.removeprefix('-')
    return bool(body) and body.isascii() and body.isdigit()


def _equal(value: Any, expected: Any) -> bool:
    """``==``, except that numeric types must agree.

    Python has ``True == 1 == 1.0``, so a predicate written ``isMeta: true`` would match a record
    carrying ``isMeta: 1``, and ``version: 1`` would match ``1.0`` — in both cases a different claim
    about the record, matched by accident. JSON distinguishes these lexically (``json.loads`` gives
    ``int`` for ``1`` and ``float`` for ``1.0``), so an author who writes one and means the other is
    making a mistake worth a refusal rather than a silent match.
    """
    if isinstance(value, bool) or isinstance(expected, bool):
        return value is expected
    if isinstance(value, (int, float)) and isinstance(expected, (int, float)):
        return type(value) is type(expected) and value == expected
    return value == expected


def _matches(record: Any, predicate: dict[str, Any]) -> bool:
    """Whether ``record`` satisfies every entry of ``predicate`` (entries are ANDed).

    A matcher is a scalar (equals), a list (membership), or ``{present: bool}`` (the path resolves
    to a non-null value / to nothing). For a path containing ``*`` the scalar and list forms mean
    "ANY value at this path matches", which is what makes ``message.content.*.type: tool_result``
    read the way it looks — *this record has a tool-result block somewhere in its content*.

    ``{present: true}`` deliberately does NOT count a resolved-but-``null`` value, which makes it
    agree with ``missing:`` and :func:`_extract`: in JSON an absent key and a null one both spell
    "the source did not record this". Two notions of presence in one document language is a trap
    for the author, who cannot see from the YAML which of them a given rule got — and the shapes
    that hit it are ordinary, ``"tool_calls": null`` on a non-calling OpenAI assistant turn above
    all, where a truthiness test matches the rule and then refuses the record it just claimed.
    """
    for path, matcher in predicate.items():
        found = _resolve_path(record, path)
        if isinstance(matcher, dict):
            if any(value is not None for value in found) is not matcher['present']:
                return False
        elif isinstance(matcher, list):
            if not any(any(_equal(value, option) for option in matcher) for value in found):
                return False
        elif not any(_equal(value, matcher) for value in found):
            return False
    return True


def _validate_predicate(predicate: dict[str, Any], *, allow_empty: bool) -> dict[str, Any]:
    """Refuse a matcher form the executor would have to interpret at run time."""
    if not predicate and not allow_empty:
        raise ValueError('must name at least one path — an empty `when` matches every record')
    for path, matcher in predicate.items():
        if not path or not isinstance(path, str):
            raise ValueError(f'predicate key {path!r} must be a non-empty path')
        if isinstance(matcher, dict):
            if set(matcher) != {'present'} or not isinstance(matcher['present'], bool):
                raise ValueError(f'{path!r}: a mapping matcher must be exactly {{present: true}} or {{present: false}}, got {matcher!r}')
        elif isinstance(matcher, list):
            if not matcher or any(isinstance(option, (dict, list)) for option in matcher):
                raise ValueError(f'{path!r}: a list matcher must be a non-empty list of scalars, got {matcher!r}')
    return predicate


#: A predicate that must NAME something: an empty ``where`` is a filter that filters nothing, and an
#: empty ``DiscardRule.when`` would discard the whole artifact and report success on a pin that cannot
#: exist. THREE fields carry it — :attr:`DiscardRule.when` directly, and :attr:`FieldSpec.where` and
#: :attr:`ToolCallSpec.where` through :data:`_ElementFilter`. The fourth predicate field,
#: :attr:`MessageRule.when`, carries :data:`_OpenPredicate` instead — the opposite rule, so counting it
#: here would read as a claim that a catch-all is refused everywhere. Written as an annotated type so
#: the rule is stated ONCE and those three fields cannot drift into three spellings of it.
_Predicate = Annotated[dict[str, Any], AfterValidator(lambda value: _validate_predicate(value, allow_empty=False))]
#: :data:`_Predicate`, but a catch-all IS legal — see :attr:`MessageRule.when`, where "everything
#: left is a human turn" is a claim about the format rather than a silent fallback.
_OpenPredicate = Annotated[dict[str, Any], AfterValidator(lambda value: _validate_predicate(value, allow_empty=True))]
#: The optional ``where:`` filter shared by :class:`FieldSpec` and :class:`ToolCallSpec`.
_ElementFilter = _Predicate | None


# --------------------------------------------------------------------------- #
# The adapter document                                                         #
# --------------------------------------------------------------------------- #


class _AdapterModel(BaseModel):
    """Closed and alias-aware. ``extra='forbid'`` is what turns a typo in a hand-edited adapter
    into a named error instead of a silently ignored key — and, once a skill writes these, what
    stops a model inventing a directive the executor would not honour."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class FieldSpec(_AdapterModel):
    """Where ONE pinned field's value lives in a source record.

    Written as a bare path most of the time (``id: uuid``); the mapping form adds a filter for the
    block-structured formats, where the field is inside one element of a content list:

    .. code-block:: yaml

        content: {from: message.content, where: {type: tool_result}, take: content}

    ``missing`` is the ONLY way a field is allowed to be absent, and it covers a null value as
    well as an absent key — in JSON both spell "the source did not record this". Requiring it to be
    declared is the fail-closed half: without it, a path that stops matching because the format
    drifted produces a refusal naming the path, not a pin full of empty strings.
    """

    source: str = Field(alias='from', min_length=1)
    #: Keep only the elements of ``from`` matching this predicate. The result must name exactly one
    #: element — except for ``content``, where several are one message's content list, carried
    #: verbatim and never joined.
    where: _ElementFilter = None
    #: A path read from the selected element — the field inside the block, once ``where`` found it.
    take: str | None = None
    #: Value to use when the path resolves to nothing or to ``null``. Its PRESENCE is what matters;
    #: ``missing: null`` is not the same as omitting the key.
    missing: Any = None

    @model_validator(mode='before')
    @classmethod
    def _accept_bare_path(cls, data: Any) -> Any:
        return {'from': data} if isinstance(data, str) else data


class ToolCallSpec(_AdapterModel):
    """Where an agent record's tool calls live, and where ``{name, args, id}`` sit inside each.

    The three paths are extracted and handed on AS FOUND — a missing name, an unparseable ``args``
    string, an id that pairs with nothing are all refused by
    :func:`~plurai.trace_convert.canonical_messages`, which already says exactly what is wrong and
    what it breaks. Re-checking them here would be a second, worse copy of those sentences.

    That is also why there is no "the arguments are a JSON string" switch: the shared projection
    parses a string into the object the pin schema requires, so an OpenAI-shaped
    ``function.arguments`` needs no declaration at all.
    """

    source: str = Field(alias='from', min_length=1)
    where: _ElementFilter = None
    name: str = Field(min_length=1)
    args: str = Field(min_length=1)
    id: str = Field(min_length=1)


class DiscardRule(_AdapterModel):
    """A record kind declared NOT to be conversation, with the reason it is not.

    Every source carries records that are not turns — a session log's bookkeeping lines, a harness
    caveat, a private reasoning block. Dropping them is correct and dropping them silently is not,
    so a discard is a rule with a ``because``: an enumerated, reviewable claim about the format
    rather than a filter that happened to match. What is NOT declared here and is not a message is
    a refusal, which is the property that makes the whole document exhaustive.
    """

    #: Rules are ordered and a broad one shadows every rule after it, so a catch-all is refused
    #: here: see :data:`_Predicate`.
    when: _Predicate
    because: NonBlankStr


class MessageRule(_AdapterModel):
    """One record shape → one pinned message. Rules are ordered; the first match wins.

    ``speak`` is checked against :data:`~plurai.resume_pin.CANONICAL_MESSAGE_TYPES`, DERIVED from
    the pin's discriminated union, so the closed speaker set is the schema's rather than a copy of
    it that drifts. **There is no default and no nearest-neighbour**: a format whose fifth speaker
    kind has no honest home among the four is a format this SDK refuses to pin, and a human decides
    what it should have been.
    """

    #: A catch-all IS legal here, and useful as a last rule — see :data:`_OpenPredicate`.
    when: _OpenPredicate = Field(default_factory=dict)
    speak: str
    #: Omitted when the source records no id, which mints one. Never omitted as a convenience: a
    #: minted id is stamped into provenance and a source id is not, so the two are told apart on
    #: disk months later.
    id: FieldSpec | None = None
    content: FieldSpec
    tool_calls: ToolCallSpec | None = None
    tool_call_id: FieldSpec | None = None

    @field_validator('speak')
    @classmethod
    def _check_speak(cls, value: str) -> str:
        if value not in CANONICAL_MESSAGE_TYPES:
            raise ValueError(f'{value!r} is not a pinnable speaker; expected one of {sorted(CANONICAL_MESSAGE_TYPES)}')
        return value

    @model_validator(mode='after')
    def _check_shape(self) -> MessageRule:
        if self.tool_calls and self.speak != 'ai':
            raise ValueError(f'speak: {self.speak} may not declare tool_calls — only an ai message carries them')
        if self.speak == 'tool' and not self.tool_call_id:
            raise ValueError('speak: tool must declare tool_call_id — a tool result that names no call cannot be replayed')
        if self.speak != 'tool' and self.tool_call_id:
            raise ValueError(f'speak: {self.speak} may not declare tool_call_id — only a tool message carries one')
        return self


class MergeSpec(_AdapterModel):
    """Consecutive agent records that are ONE reply, told by a key they share.

    Some logs write each content block of one model reply as its own record — a reply that calls
    two tools in parallel arrives as two agent records, then the two results. Projected one record
    to one message, that is ``ai, ai, tool, tool``: a call whose results do not follow it, which a
    provider refuses and the seed-shape rule refuses with it. ``by`` names the path holding the
    reply's own id (a Claude Code record's ``message.id``), and consecutive ``ai`` messages whose
    value there is equal and non-null become one. See :func:`_merge_replies` for what merging does.
    """

    by: str = Field(min_length=1)

    @field_validator('by')
    @classmethod
    def _check_by(cls, value: str) -> str:
        if not value.strip() or any(not segment for segment in value.split('.')):
            raise ValueError(f'must be a non-empty dotted path with no empty segment, got {value!r}')
        return value


class AdapterDocument(_AdapterModel):
    """A whole adapter: how to read one source format, and what it claims that format is.

    This class IS the adapter contract an adapter-authoring skill writes against. Every field is
    either a location in the source or a declaration about the format; none of them is behaviour.
    """

    #: Must equal the file's stem. Two names for one adapter is one name too many to keep true.
    format: NonBlankStr
    #: One line, shown when a ``--source-format`` fails to resolve.
    description: NonBlankStr
    #: ``json`` — one artifact is one JSON document. ``jsonl`` — one line is one record.
    container: Literal['json', 'jsonl'] = 'json'
    #: ``json`` only: path from the document to the record list. Omitted when the document IS one.
    records: str | None = None
    #: Which extensions a DIRECTORY of this format holds, so a `.jsonl` corpus is not reported
    #: empty by a `*.json` glob.
    suffixes: tuple[str, ...] = ('.json',)
    #: Whether this format carries tool calls at all. Must agree with whether any rule below maps
    #: them — see :meth:`_check_tool_call_declaration`.
    emits_tool_calls: bool
    #: Optional path to a per-record ordinal. Where declared, it must be strictly increasing.
    order: str | None = None
    #: Path to the state values to ferry alongside the history. Omitted means ``{}``, which is the
    #: ordinary answer: a transcript is not a graph checkpoint and has no state but its messages.
    agent_state: str | None = None
    #: Extra provenance keys, each a path read from the artifact root. The pin's provenance is its
    #: one open shape, so a format may record what identifies it without a lockstep release.
    provenance: dict[str, FieldSpec] = Field(default_factory=dict)
    discard: list[DiscardRule] = Field(default_factory=list)
    messages: list[MessageRule] = Field(min_length=1)
    #: Omitted when every record is a whole message, which is the ordinary case.
    merge: MergeSpec | None = None

    @field_validator('suffixes')
    @classmethod
    def _check_suffixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not s.startswith('.') or len(s) < 2 for s in value):
            raise ValueError(f'must be a non-empty list of extensions including the dot, got {list(value)}')
        return value

    @field_validator('provenance')
    @classmethod
    def _check_provenance(cls, value: dict[str, FieldSpec]) -> dict[str, FieldSpec]:
        if not (reserved := sorted(set(value) & RESERVED_PROVENANCE_KEYS)):
            return value
        # One refusal naming every offender, each group with its own reason: an author fixing them
        # one refusal at a time learns the rule slowest.
        reasons = []
        if executor := sorted(set(value) & _EXECUTOR_PROVENANCE_KEYS):
            reasons.append(
                f'the executor stamps {executor} as evidence ABOUT this adapter (which file read the artifact, over how many '
                'records, how many ids it had to mint), and a document that could write them could claim on disk to be a '
                'different reader than the one that ran'
            )
        if converter := sorted(set(value) & CONVERTER_PROVENANCE_KEYS):
            reasons.append(
                f'the converter owns {converter}: it stamps them itself to record what it did to the pin (its source file and '
                'version, a cut, a trim, which state it carries, where its turn numbering starts), and a declared value would '
                'claim a conversion step that never ran'
            )
        # No closing period: `_read_document` ends the sentence when it wraps this one.
        raise ValueError(f'may not declare {reserved} — {"; ".join(reasons)}. Rename the key')

    @model_validator(mode='after')
    def _check_container(self) -> AdapterDocument:
        if self.container == 'jsonl' and self.records is not None:
            raise ValueError('container: jsonl has no document to descend into — every line is a record, so `records` cannot apply')
        return self

    @model_validator(mode='after')
    def _check_tool_call_declaration(self) -> AdapterDocument:
        """``emits_tool_calls`` and the rules that map them are one claim, so they must agree.

        A capability declaration contradicted by the document declaring it is decidable with no
        artifact at all: `tool_calls` reaches a pin only from a rule that maps it, so "declares the
        format carries calls, maps none" cannot be true of any corpus, and "maps calls, denies the
        capability" mislabels every one it extracts. Both are properties of these bytes.

        Checked here rather than against a batch of conversions, which is the tempting place for
        it: such an audit can only fire on a run of more than one artifact — otherwise a single
        tool-free conversation, which is entirely legitimate, fails the build — and so the SAME
        artifacts give different verdicts depending on whether the CLI is pointed at their
        directory or invoked once per file into the same ``--out-dir``. A verdict that turns on how
        the command was typed is not a claim about the format. What is genuinely decidable is not
        about the data at all; it is settled here, once, at load.
        """
        maps_calls = any(rule.tool_calls is not None for rule in self.messages)
        if self.emits_tool_calls and not maps_calls:
            raise ValueError(
                'declares emits_tool_calls but no message rule maps `tool_calls` — nothing in this document can ever produce '
                "one, so every resumed scenario would silently lose the agent's tool exchanges. Map them on the rule that "
                'matches assistant records, or set emits_tool_calls: false'
            )
        if not self.emits_tool_calls and maps_calls:
            raise ValueError(
                'maps `tool_calls` on a message rule but declares emits_tool_calls: false — anything keying off the '
                'declaration would skip calls this adapter is extracting. Declare the capability or drop the mapping'
            )
        return self


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def _candidate_paths(name: str, workspace_root: str | Path) -> list[Path]:
    root = Path(workspace_root)
    return [root / reldir / f'{name}.yaml' for reldir in _ADAPTER_CANDIDATE_RELDIRS]


def available_adapters(workspace_root: str | Path = '.') -> list[str]:
    """Adapter names persisted under ``workspace_root``, sorted and de-duplicated.

    Used to answer "then what CAN I convert?" at the moment a ``--source-format`` fails, which is
    the moment the answer is worth anything.
    """
    root = Path(workspace_root)
    directories = [directory for directory in (root / reldir for reldir in _ADAPTER_CANDIDATE_RELDIRS) if directory.is_dir()]
    return sorted({path.stem for directory in directories for path in directory.glob('*.yaml') if path.is_file()})


def _read_document(name: str, workspace_root: str | Path) -> tuple[AdapterDocument, Path]:
    """The adapter named ``name``, parsed and validated, plus the file it came from."""
    candidates = _candidate_paths(name, workspace_root)
    present = [path for path in candidates if path.is_file()]
    if not present:
        known = sorted(TRACE_FORMAT_REGISTRY) + available_adapters(workspace_root)
        searched = ', '.join(str(path) for path in candidates)
        raise AdapterError(
            f'unknown source format {name!r}. This SDK can read {known}. An adapter for a format not on that list is a YAML '
            f'mapping document committed to the workspace — none was found at: {searched}.'
        )
    if len(present) > 1:
        # Mirrors the posture rule: two config files that disagree fail closed rather than letting
        # a discovery order nobody remembers decide what the customer's traces are.
        raise AdapterError(f'adapter {name!r} is defined twice ({", ".join(str(p) for p in present)}) — keep one and delete the other.')

    path = present[0]
    try:
        # `yaml.safe_load`, never `yaml.load`: safe_load builds only plain scalars, lists and dicts,
        # so a document cannot name a Python type to construct. That is the line between a mapping
        # a model may author and code a model may run, and it is enforced here rather than trusted.
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as e:
        raise AdapterError(f'{path}: is not parseable YAML ({e}).') from e
    except (UnicodeDecodeError, OSError) as e:
        raise AdapterError(f'{path}: could not be read ({e}).') from e
    if not isinstance(raw, dict):
        raise AdapterError(f'{path}: must be a YAML mapping, got {type(raw).__name__}.')

    try:
        document = AdapterDocument.model_validate(raw)
    except ValidationError as e:
        raise AdapterError(f'{path}: is not a valid adapter — {_first_problem(e)}.') from e
    if document.format != name:
        raise AdapterError(f'{path}: declares format {document.format!r} but is filed as {name!r}. The stem and the declaration must agree.')
    return document, path


def load_adapter(name: str, *, workspace_root: str | Path = '.') -> _FormatSpec:
    """The workspace-persisted adapter ``name``, as a registry-shaped front end.

    Called only after :data:`~plurai.trace_convert.TRACE_FORMAT_REGISTRY` misses, so a built-in
    always wins: an adapter cannot shadow — or quietly substitute itself for — the live-verified
    LangSmith reader.
    """
    document, path = _read_document(name, workspace_root)
    return _FormatSpec(
        front_end=partial(_project, document, path),
        description=document.description,
        name=document.format,
        suffixes=document.suffixes,
    )


# --------------------------------------------------------------------------- #
# Execution                                                                    #
# --------------------------------------------------------------------------- #


def _extract(record: Any, spec: FieldSpec, *, where: str, field: str, many: bool) -> Any:
    """One field's value, copied verbatim out of ``record`` — or a refusal saying why not.

    ``many`` is for a message's ``content`` alone: several selected values (text blocks, a block per
    attachment) are that message's content LIST, in document order. Every other field names one
    value, and two candidates for it are a question this executor does not answer.
    """
    found = _resolve_path(record, spec.source)

    if spec.where is not None:
        blocks: list[Any] = []
        for value in found:
            if not isinstance(value, list):
                raise AdapterError(f'{where}: {field} filters {spec.source!r}, so it must hold a list; it holds {type(value).__name__}.')
            blocks.extend(value)
        found = [block for block in blocks if _matches(block, spec.where)]

    if spec.take is not None:
        found = [taken for value in found for taken in _resolve_path(value, spec.take)]

    if len(found) > 1:
        if many:
            return found
        raise AdapterError(
            f'{where}: {field} selects {len(found)} values from {spec.source!r}, and a pinned field is carried VERBATIM — never joined, '
            'never the first of several. Narrow the path (or its `where`) until it names exactly one.'
        )

    value = found[0] if found else None
    if value is None:
        if 'missing' in spec.model_fields_set:
            return spec.missing
        raise AdapterError(
            f'{where}: {field} found no value at {spec.source!r} (the path is absent or null). If this field is legitimately '
            'absent for some records, declare what stands in for it with `missing:`; otherwise the path is wrong.'
        )
    return value


def _tool_calls(record: Any, spec: ToolCallSpec, *, where: str) -> tuple[list[dict[str, Any]], bool]:
    """The record's tool calls in ``{name, args, id}`` shape, extracted but NOT validated.

    Returns the calls AND whether ``spec.source`` structurally resolved on this record, which the
    calls alone cannot say: a record holding ``tool_calls: []`` and a record where the path names
    nothing both yield no calls, and only the second is evidence the mapping is wrong. Neither is a
    refusal here — an assistant turn that called nothing is ordinary — so the distinction is
    counted into provenance rather than judged. See :func:`_project`.
    """
    found = _resolve_path(record, spec.source)
    calls: list[Any] = []
    for value in found:
        if not isinstance(value, list):
            raise AdapterError(f'{where}: tool_calls reads {spec.source!r}, which must hold a list; it holds {type(value).__name__}.')
        calls.extend(value)
    if spec.where is not None:
        calls = [call for call in calls if _matches(call, spec.where)]

    def one(call: Any, path: str) -> Any:
        found = _resolve_path(call, path)
        if len(found) > 1:
            # Distinguished from absence rather than collapsed into it, matching `_extract`. Both
            # end in a refusal, but "missing a name" sends the author looking for a field that is
            # right there, when the defect is a path broad enough to have found two of them.
            raise AdapterError(f'{where}: tool call path {path!r} selects {len(found)} values from one call; narrow it until it names exactly one.')
        # `None` when absent, deliberately: canonical_messages refuses a nameless or unpaired call
        # in terms of what it breaks downstream, which is the sentence an operator needs.
        return found[0] if found else None

    return [{'name': one(call, spec.name), 'args': one(call, spec.args), 'id': one(call, spec.id)} for call in calls], bool(found)


def _ordering_violation(ordinals: list[Any], *, source: str, path: str) -> str | None:
    """The ordering invariant.

    What a type cannot express: an optional ordinal, where present, carries a total order.
    Duplicate or descending values leave the transcript ambiguous to anything that sorts by them —
    and unlike a malformed field, all three failures produce a plausible-looking pin with no
    exception anywhere.

    Tri-state on purpose. A format that records no ordinal is entitled to omit it (order is then
    the record list's own); a PARTIAL declaration is the ambiguous case, because a consumer cannot
    tell which of the two orderings to trust.
    """
    declared = [value for value in ordinals if value is not None]
    if not declared:
        # Legitimate PER ARTIFACT: some exports timestamp every message and others timestamp
        # none, both convert, and order is then the record
        # list's own. "Declared but never resolved ANYWHERE" would be the real defect, but nothing
        # here can see it and no batch of artifacts can either: a run is whichever files one
        # invocation was pointed at, so judging it there makes the same bytes pass or fail on how
        # the command was typed. It is reported instead of guessed — `ordinal_resolved_count` is
        # stamped into every pin, and `0 of 7` is the sentence an operator needs to read.
        return None
    if len(declared) != len(ordinals):
        return (
            f'{source}: {path} is declared on {len(declared)} of {len(ordinals)} messages — declare it on every message or none '
            '(a partial ordinal leaves the order ambiguous)'
        )
    try:
        if any(a >= b for a, b in pairwise(declared)):
            return f'{source}: {path} is not strictly increasing: {declared}'
    except TypeError:
        kinds = sorted({type(value).__name__ for value in declared})
        return f'{source}: {path} mixes {kinds}, so the messages cannot be ordered by it'
    return None


def _artifact_root(document: AdapterDocument, source_path: Path) -> Any:
    """The parsed artifact: its JSON document, or the list of its JSONL lines."""
    source = source_path.name
    text = read_source_text(source_path)
    if document.container == 'json':
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise AdapterError(f'{source}: not parseable JSON ({e}). Adapter {document.format!r} declares container: json.') from e

    lines: list[Any] = []
    # `split('\n')`, never `splitlines()`: the latter also breaks on U+2028, U+2029, U+0085, \x0b
    # and \x0c, every one of which is legal INSIDE a JSON string and common in text scraped from the
    # web. A message carrying one would be split into two unparseable fragments and a well-formed
    # export refused — a fidelity hole in the container of a converter whose subject is fidelity.
    for number, raw_line in enumerate(text.split('\n'), start=1):
        line = raw_line.removesuffix('\r')
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise AdapterError(f'{source} line {number}: not parseable JSON ({e}). Adapter {document.format!r} declares container: jsonl.') from e
    return lines


def _records(document: AdapterDocument, root: Any, *, source: str) -> list[Any]:
    """The artifact's records: the root itself, or the list ``records:`` points at."""
    if document.records is None:
        found: list[Any] = [root]
    else:
        found = _resolve_path(root, document.records)
        if len(found) != 1:
            raise AdapterError(f'{source}: records path {document.records!r} matched {len(found)} values; it must name exactly one list.')
    if not isinstance(found[0], list):
        at = f'records path {document.records!r}' if document.records else 'the document'
        raise AdapterError(f'{source}: {at} holds {type(found[0]).__name__}, not a list of records.')
    return found[0]


def _shape_of(record: Any) -> str:
    """A record's top-level key set, for an error that has to describe an unforeseen shape."""
    if not isinstance(record, dict):
        return type(record).__name__
    keys = sorted(record)
    shown = ', '.join(keys[:8])
    return f'{{{shown}{", …" if len(keys) > 8 else ""}}}'


def _merge_key(record: Any, spec: MergeSpec, *, where: str) -> Any:
    """The record's reply id at ``spec.by`` — ``None`` (never merged) when absent or null; two candidates refuse."""
    found = [value for value in _resolve_path(record, spec.by) if value is not None]
    if len(found) > 1:
        raise AdapterError(f'{where}: merge path {spec.by!r} selects {len(found)} values; a record belongs to one reply or none.')
    if found and isinstance(found[0], (dict, list)):
        raise AdapterError(f'{where}: merge path {spec.by!r} holds a {type(found[0]).__name__}, not a reply id.')
    return found[0] if found else None


def _content_blocks(content: Any) -> list[Any]:
    """One record's content as blocks, verbatim: a string is one text block, a list is its own items."""
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    if isinstance(content, list):
        return [{'type': 'text', 'text': item} if isinstance(item, str) else item for item in content if item != '']
    return [content]


def _merge_replies(entries: list[tuple[dict[str, Any], Any, Any, bool]]) -> tuple[list[tuple[dict[str, Any], Any]], int]:
    """``(messages with each reply's records merged, how many records were merged into an earlier one)``.

    ``entries`` are ``(message, ordinal, reply key, breaks)`` in record order, ``breaks`` set on the
    entry FOLLOWING an unrecognised record: a record no rule accounts for may be a turn, so a reply
    is never merged across one. A discarded record is not an entry at all, so a reasoning block
    between two blocks of one reply does not split it. Only ``ai`` messages merge, and only with
    the ``ai`` message right before them carrying the same non-null key — a tool result or a user
    turn between them always ends the reply. Merging is lossless:

    * the id and ordinal are the first record's;
    * the tool calls are every record's, in record order;
    * the content is the one non-empty string when exactly one record said something in a string,
      ``''`` when none said anything (a reply that is only calls, spelled as an unmerged one is), and
      otherwise every record's blocks in order — never joined.
    """
    merged: list[tuple[dict[str, Any], Any]] = []
    keys: list[Any] = []
    contents: list[list[Any]] = []
    folded = 0
    for message, ordinal, key, breaks in entries:
        if not breaks and key is not None and message['type'] == 'ai' and merged and merged[-1][0]['type'] == 'ai' and keys[-1] == key:
            head = merged[-1][0]
            calls = [*(head.get('tool_calls') or []), *(message.get('tool_calls') or [])]
            if 'tool_calls' in head or 'tool_calls' in message:
                head['tool_calls'] = calls
            contents[-1].append(message['content'])
            folded += 1
            continue
        merged.append((message, ordinal))
        keys.append(key)
        contents.append([message['content']])
    for (message, _ordinal), said in zip(merged, contents, strict=True):
        if len(said) == 1:
            continue
        spoken = [content for content in said if content not in ('', [])]
        if len(spoken) == 1 and isinstance(spoken[0], str):
            message['content'] = spoken[0]
        else:
            # A reply that said nothing beside its calls keeps the spelling an unmerged one has: ''.
            message['content'] = [block for content in said for block in _content_blocks(content)] or ''
    return merged, folded


def _project(document: AdapterDocument, adapter_path: Path, source_path: Path) -> SourceProjection:
    """One artifact → its source-shaped projection, per the adapter's mapping.

    Every record is accounted for exactly once: discarded by a declared rule, projected by the
    first message rule that matches, or — matching neither — left out and COUNTED, with its shape,
    in provenance and in a finding. Counting is what keeps the adapter a complete claim about a
    format: drift shows on every pin it touches instead of either killing the conversion or
    vanishing. There is no catch-all discard to hide it behind, since discards are settled before
    message rules and a ``when: {}`` discard would drop every message.
    """
    source = source_path.name
    root = _artifact_root(document, source_path)
    records = _records(document, root, source=source)

    entries: list[tuple[dict[str, Any], Any, Any, bool]] = []
    after_unrecognised = False
    # The messages whose id was minted, by object identity: a merge folds records away, and the
    # count is of the messages the pin carries.
    minted_messages: set[int] = set()
    discarded = 0
    unrecognised: list[str] = []
    # Records matched by a rule that DECLARES tool_calls, and how many of those had the declared
    # path actually resolve. Two different facts, and neither is the number of calls extracted.
    tool_call_rule_matches = 0
    tool_call_resolved = 0

    for index, record in enumerate(records):
        at = f'{source} record[{index}]'
        # Discards are settled first, so a rule may carve an exception out of a broader message
        # rule ("an assistant line whose only block is private reasoning is not a turn").
        if any(_matches(record, rule.when) for rule in document.discard):
            discarded += 1
            continue

        rule = next((candidate for candidate in document.messages if _matches(record, candidate.when)), None)
        if rule is None:
            # Never guessed at — a mislabelled speaker replays a conversation nobody had — and never
            # silently gone: counted with its shape, and named in a finding.
            unrecognised.append(_shape_of(record))
            after_unrecognised = True
            continue

        message: dict[str, Any] = {'type': rule.speak, 'content': _extract(record, rule.content, where=at, field='content', many=True)}
        if rule.id is None:
            message['id'] = str(uuid.uuid5(MINTED_ID_NAMESPACE, f'{source}:{index}'))
            minted_messages.add(id(message))
        else:
            message['id'] = _extract(record, rule.id, where=at, field='id', many=False)
        if rule.tool_calls is not None:
            tool_call_rule_matches += 1
            message['tool_calls'], resolved = _tool_calls(record, rule.tool_calls, where=at)
            tool_call_resolved += int(resolved)
        if rule.tool_call_id is not None:
            message['tool_call_id'] = _extract(record, rule.tool_call_id, where=at, field='tool_call_id', many=False)

        ordinal = None
        if document.order is not None:
            found = _resolve_path(record, document.order)
            if len(found) > 1:
                raise AdapterError(
                    f'{at}: order path {document.order!r} selects {len(found)} values, and a message has one position or none. '
                    'Narrow the path until it names exactly one — two candidates is a question about what orders this '
                    'transcript, not an answer to it.'
                )
            ordinal = found[0] if found else None
        key = _merge_key(record, document.merge, where=at) if document.merge is not None else None
        entries.append((message, ordinal, key, after_unrecognised))
        after_unrecognised = False

    paired, merged_records = _merge_replies(entries)
    messages = [message for message, _ordinal in paired]
    ordinals = [ordinal for _message, ordinal in paired]
    minted = sum(id(message) in minted_messages for message in messages)
    if document.order is not None and (violation := _ordering_violation(ordinals, source=source, path=document.order)):
        raise AdapterError(violation)

    state: dict[str, Any] = {}
    if document.agent_state is not None:
        found = _resolve_path(root, document.agent_state)
        if len(found) != 1 or not isinstance(found[0], dict):
            got = type(found[0]).__name__ if len(found) == 1 else f'{len(found)} values'
            raise AdapterError(f'{source}: agent_state path {document.agent_state!r} holds {got}, not a single object.')
        state = found[0]

    declared = {key: _extract(root, spec, where=source, field=f'provenance.{key}', many=False) for key, spec in document.provenance.items()}
    provenance: dict[str, Any] = {
        # Splatted FIRST so the executor's own stamps win any collision. `RESERVED_PROVENANCE_KEYS`
        # already refuses these at load time, naming the offending line of YAML; this is the second
        # lock on the same door, because the cost of the two disagreeing is a pin that misreports
        # which reader produced it, and that is the one fact a reader cannot check against anything.
        **declared,
        'adapter': document.format,
        'adapter_file': adapter_path.name,
        'record_count': len(records),
        'discarded_record_count': discarded,
        # Always stated, including as 0: "the source recorded every id" is a fact worth reading off
        # a pin, and an absent key would only mean nobody wrote it down.
        'minted_message_ids': minted,
    }
    # The two declarations whose failure mode is SILENCE, each recorded as a count rather than
    # judged. A declared path that resolves nowhere is legitimate on one artifact and suspicious
    # across a corpus, and nothing at conversion time can tell those apart: what a single run holds
    # is whichever files one invocation named. So the pin states the evidence and the reader draws
    # the conclusion — `ordinal_resolved_count: 0` on every pin of a 400-artifact corpus is a wrong
    # path, and it is visible without the converter having to guess at a threshold.
    if document.order is not None:
        provenance['ordinal_resolved_count'] = len([value for value in ordinals if value is not None])
    # `emits_tool_calls`, not a re-scan of the rules: `_check_tool_call_declaration` already made
    # the two agree at load, so the declaration IS the answer to "does this document map calls".
    # Stated whenever the document declares a merge, 0 included, like the other declared-path counts.
    if document.merge is not None:
        provenance['merged_record_count'] = merged_records
    if document.emits_tool_calls:
        provenance['tool_call_rule_matches'] = tool_call_rule_matches
        provenance['tool_call_resolved_count'] = tool_call_resolved
    if minted:
        provenance['minted_id_namespace'] = str(MINTED_ID_NAMESPACE)
        provenance['minted_id_recipe'] = 'uuid5(minted_id_namespace, "<source_file>:<record index>")'
    findings: list[GateFinding] = []
    if unrecognised:
        shapes = sorted(set(unrecognised))[:UNRECOGNISED_SHAPES_SHOWN]
        provenance['unrecognised_record_count'] = len(unrecognised)
        provenance['unrecognised_shapes'] = shapes
        findings.append(
            make_finding(
                TRACE_ADAPTER_UNRECOGNISED_RECORDS,
                subject=source,
                evidence={'count': len(unrecognised), 'shapes': shapes},
                observed={},
                remedy=(
                    f'{len(unrecognised)} record(s) no rule in adapter {document.format!r} accounts for were left out of the '
                    f'conversation, of shapes {shapes}. If one of them is a turn, add a message rule for it to {adapter_path.name}; '
                    'if none is, declare each kind as a `discard` with its reason.'
                ),
                draft_ref=None,
            )
        )

    return SourceProjection(messages=messages, agent_state=state, provenance=provenance, findings=tuple(findings))


__all__ = [
    'ADAPTER_DIR_NAME',
    'MINTED_ID_NAMESPACE',
    'RESERVED_PROVENANCE_KEYS',
    'UNRECOGNISED_SHAPES_SHOWN',
    'AdapterDocument',
    'AdapterError',
    'AdapterThreadRefused',
    'DiscardRule',
    'FieldSpec',
    'MergeSpec',
    'MessageRule',
    'ToolCallSpec',
    'available_adapters',
    'load_adapter',
]
