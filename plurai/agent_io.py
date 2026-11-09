"""The agent's TURN-INPUT surface, as ``plurai.config.yaml`` declares it.

Without a declaration the wire between the simulated user and the customer harness carries one
text string per turn: ``child.py`` builds ``{'messages': [HumanMessage(content=<the string>)]}``.
Two real bench shapes do not fit that: an agent whose user sends a FILE or an IMAGE with their
words, and an agent that OPENS the conversation (an IVR, an outbound call, a kiosk) and so is
driven with no user message at all on turn 0. A harness of the second shape that indexes
``payload['messages'][-1]`` raises the moment the list is empty.

This module is the SDK's reading of the key that declares both. It is the INPUT half only:
the OUTPUT half is ``render_actions:`` (:mod:`plurai.render_actions`) — the verbs the
agent emits — and declaring output twice is how two declarations of one fact start to disagree.

* File: ``plurai.config.yaml``. Key: top-level ``agent.io`` — a ROOT key whose name contains a dot, the
  spelling the plugin's ``verify_config`` gate authors and checks, unlike the ``agent.version`` /
  ``agent.native_tools`` keys that are written nested. ``agent: {io: …}`` is read by nobody, and a
  declaration that is ignored degrades a run silently (the first ``file`` part is refused at the turn), so
  it is an :class:`AgentIOConfigError` naming the root spelling; so is a document that carries both
  spellings. Found under a root at the two rungs every workspace declaration is looked for
  (:data:`plurai.ingest.workspace_config.CONFIG_CANDIDATE_RELPATHS`).
* ABSENT key: the config says nothing, and the run takes the defaults —
  :data:`DEFAULT_INPUT_PARTS` (``text``) and no empty opener. :func:`parse_agent_io` returns
  ``None`` so a consumer can still tell "said nothing" from ``agent.io: {}``, which declares the
  same defaults deliberately.
* MALFORMED — not a mapping, a non-list ``input``, a blank or duplicate part kind, a non-boolean
  ``may_open``, an unknown ``opens``: :class:`AgentIOConfigError`, fail closed. Each of those
  values decides how turn 0 is driven, and a declaration that cannot testify must never degrade
  to "undeclared": undeclared is the state this key exists to end.
* UNKNOWN — a sub-key this build does not read, or a part kind outside :data:`INPUT_PART_KINDS`:
  dropped from the parse and recorded as an :class:`AgentIODefect` the runner reports. A newer
  build may read either, and every reader of this module (the run-start rung, the per-scenario
  reader, the child) sees the same lenient parse, so one unknown word never becomes N refusals.
* Two configs under one root that both declare and disagree: an error, not a precedence puzzle.
  The surface is env-scoped, the same rule ``anonymize`` / ``render_actions`` / ``shape`` follow.

SCOPE. ``agent.io.input`` says what may ride the user's own message (content parts).
``agent.io.channels`` declares the PER-TURN inputs the agent takes outside it (a file id into a
store, a host payload field, a UI change between turns, a host-injected message), which scenarios
plant through :mod:`plurai.channels`. Neither describes the scenario's STARTUP state: an
``agent_context`` precondition field reaches the agent through ``child.py``'s
``payload.update(self.initial_agent_context)`` on turn 1 and is declared once already under
``agent_context:``.

Read by :mod:`plurai.child` (the projection that builds the turn payload, and the typed refusal
of a part this harness did not declare) and by :mod:`plurai._subprocess` (the ``may_open`` gate:
the parent sends an empty opener ONLY to a harness that declared it can serve one).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, bench_config_roots, discover_config_documents
from plurai.scan_contract.input_channels import CHANNEL_KINDS, ChannelKind

#: Top-level ``plurai.config.yaml`` key. ONE vocabulary with the plugin's ``verify_config`` gate
#: that authors it — a ROOT key whose NAME contains a dot, not an ``io:`` nested under ``agent:``.
AGENT_IO_KEY = 'agent.io'
#: The mapping the other ``agent.*`` keys live under; an ``io`` inside it is a misplaced declaration.
AGENT_KEY = 'agent'
_NESTED_IO_KEY = 'io'

#: The sub-keys ``agent.io`` may carry. An unknown one is a DEFECT, reported (and held by the runner
#: until overruled): a mis-spelled ``may_open`` or ``opens`` that is silently ignored is a harness
#: driven against what it meant to declare.
INPUT_FIELD = 'input'
MAY_OPEN_FIELD = 'may_open'
#: ``opens: always`` — production's HOST opens every conversation, even when a scenario authors
#: a user first line. Stronger than ``may_open`` (which only PERMITS an empty opener) and implies
#: it: the run asks the server for an agent-opened thread, and first_talk becomes turn 1.
OPENS_FIELD = 'opens'
OPENS_ALWAYS = 'always'
_OPENS_VALUES = (OPENS_ALWAYS,)
#: ``choice_text: {<choice>: delivered|dropped}`` — what becomes of the words the user types beside an
#: interrupt choice, per choice, as the harness's RESUME code handles it. ``dropped`` is a resume path
#: that discards the text (an approval that resumes with no new input); ``delivered`` is one that hands
#: it to the agent. Read off the resume code by the scan, never inferred from tool names, and never
#: defaulted: an absent key means every choice's text is taken as delivered.
CHOICE_TEXT_FIELD = 'choice_text'
#: ``choice_text_source: <path>:<line>`` — the resume code the ``choice_text`` fact was read off.
#: Required beside it: a fact that changes what the simulator is credited with must name its evidence.
CHOICE_TEXT_SOURCE_FIELD = 'choice_text_source'
CHOICE_TEXT_DELIVERED = 'delivered'
CHOICE_TEXT_DROPPED = 'dropped'
#: The closed vocabulary of a ``choice_text`` value.
CHOICE_TEXT_VALUES: tuple[str, ...] = (CHOICE_TEXT_DELIVERED, CHOICE_TEXT_DROPPED)
#: ``channels: [{name, kind, field, source, handover}]`` — the inputs the agent takes OUTSIDE the user's
#: message content: an id into a store, a per-turn host payload field, a UI change between turns, a
#: message the host adds to history. Applied per scenario by :mod:`plurai.channels`.
CHANNELS_FIELD = 'channels'
_KNOWN_FIELDS = (INPUT_FIELD, MAY_OPEN_FIELD, OPENS_FIELD, CHOICE_TEXT_FIELD, CHOICE_TEXT_SOURCE_FIELD, CHANNELS_FIELD)

# The closed vocabulary of a channel ``kind`` is NOT defined here: :data:`ChannelKind` / :data:`CHANNEL_KINDS` are
# the scan contract's (:mod:`plurai.scan_contract.input_channels`, vendored from the backend), the one definition the
# backend's scenario generation also reads — re-exported below so ``plurai.agent_io.CHANNEL_KINDS`` keeps working.
# Unlike a content-part kind, an unknown one is REFUSED, not dropped: a channel names a field the harness reads, and
# a scenario that plants into it must not run against a harness whose declaration was silently narrowed.
#: Every key of one channel entry; the first four are required.
_CHANNEL_REQUIRED = ('name', 'kind', 'field', 'source')
_CHANNEL_KEYS = (*_CHANNEL_REQUIRED, 'handover')

#: The content-part vocabulary this build can carry, part ``type`` → the key that part carries
#: its payload under. It is OpenAI's, not ours: ``{'type': 'text', 'text': …}``,
#: ``{'type': 'image_url', 'image_url': {…}}``, ``{'type': 'file', 'file': {…}}`` — self-keyed in
#: all three, which is what makes one table enough.
#:
#: A LOOKUP, never a prefix test or an if/elif chain, and that is the dispatch worth pinning: a
#: kind this build has never heard of (``text_v2``, a streaming chunk, a voice frame) is named as a
#: defect at the declaration and refused at the turn, never flattered into ``text`` because it
#: starts with the same five letters. Kinds the runner cannot yet TEST — streaming, media out,
#: voice — are absent on purpose; they become declarable when something can confirm them.
INPUT_PART_KINDS: Mapping[str, str] = MappingProxyType({'text': 'text', 'image_url': 'image_url', 'file': 'file'})

#: What a harness that declares nothing accepts — the default, stated rather than implied.
DEFAULT_INPUT_PARTS: tuple[str, ...] = ('text',)


class AgentIOConfigError(WorkspaceConfigError):
    """``agent.io`` exists but cannot testify to a turn-input surface (fail closed)."""


@dataclass(frozen=True, slots=True)
class AgentIODefect:
    """One part of ``agent.io`` this build does not read: dropped from the parse, reported by the runner."""

    code: Literal['unknown_part_kind', 'unknown_field']
    #: The dotted key it was found under (``agent.io``, ``agent.io.input[<i>]`` or ``agent.io.channels[<i>]``).
    key: str
    #: The unknown word as written.
    value: str
    source: Path | None


@dataclass(frozen=True, slots=True)
class AgentIOChannel:
    """One declared ``agent.io.channels`` entry: an input the agent takes outside the message content."""

    #: The name a scenario's ``turn_inputs`` / ``between_turn_actions`` address it by.
    name: str
    #: One of :data:`CHANNEL_KINDS`.
    kind: ChannelKind
    #: The request / payload field the agent reads it from (for ``between_turn_action``, the UI action it models).
    field: str
    #: ``<path>:<line>`` of the agent code that reads it.
    source: str
    #: How production hands it over, verbatim; ``None`` when undeclared.
    handover: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'kind': self.kind,
            'field': self.field,
            'source': self.source,
            **({'handover': self.handover} if self.handover is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class AgentIO:
    """The declared turn-input surface: the content parts, the opener, and who declared them."""

    #: The content-part kinds the harness accepts on the user's message, in declaration order.
    #: ``()`` is "none, by declaration" — the agent-first bench, which is spoken to but is
    #: handed no parts.
    input_parts: tuple[str, ...] = DEFAULT_INPUT_PARTS
    #: Whether ``invoke_agent`` can be called with ``{'messages': []}`` — the agent OPENS.
    may_open: bool = False
    #: ``'always'`` when the host opens every conversation; ``None`` (undeclared) leaves turn 0 to
    #: the scenario — its first_talk when it has one, else the agent (the ``may_open`` shape).
    opens: str | None = None
    #: ``{<choice>: 'delivered' | 'dropped'}`` — what the resume code does with the user's words beside
    #: each interrupt choice; ``None`` when undeclared (nothing is marked dropped).
    choice_text: Mapping[str, str] | None = field(default=None, hash=False)
    #: ``<path>:<line>`` of the resume code ``choice_text`` was read off; set exactly when it is.
    choice_text_source: str | None = None
    #: The declared non-message input channels (:class:`AgentIOChannel`), in declaration order.
    channels: tuple[AgentIOChannel, ...] = ()
    #: The config file the declaration was read from; ``None`` for an in-memory document.
    source: Path | None = None
    #: What this build dropped from the declaration (:class:`AgentIODefect`). Not part of the surface.
    defects: tuple[AgentIODefect, ...] = ()

    def accepts_part(self, kind: str) -> bool:
        """Whether a content part of ``kind`` may be put on this harness's turn."""
        return kind in self.input_parts

    def channel(self, name: str) -> AgentIOChannel | None:
        """The declared channel called ``name``, or ``None``."""
        return next((channel for channel in self.channels if channel.name == name), None)

    def as_record(self) -> dict[str, Any]:
        """The wire shape. A mapping even when everything in it is the default, so a consumer
        that tests truthiness still sees a DECLARATION rather than reading it as silence."""
        return {
            INPUT_FIELD: list(self.input_parts),
            MAY_OPEN_FIELD: self.may_open,
            **({OPENS_FIELD: self.opens} if self.opens is not None else {}),
            **(
                {CHOICE_TEXT_FIELD: dict(self.choice_text), CHOICE_TEXT_SOURCE_FIELD: self.choice_text_source}
                if self.choice_text is not None
                else {}
            ),
            **({CHANNELS_FIELD: [channel.as_record() for channel in self.channels]} if self.channels else {}),
            'source': str(self.source) if self.source is not None else None,
        }


#: The surface a harness that declares nothing is driven with. Returned by name (rather than
#: rebuilt at each call site) so "undeclared" has exactly one spelling.
DEFAULT_AGENT_IO = AgentIO()


def parse_agent_io(document: Mapping[str, Any], *, source: Path | None = None) -> AgentIO | None:
    """The turn-input surface one loaded config declares: ``None`` when the key is absent.

    Raises :class:`AgentIOConfigError` on a malformed value of a known key. ``source`` is named in
    every error so the operator is sent to the file, not to a key. An unknown sub-key or part kind is
    dropped and carried in :attr:`AgentIO.defects`.
    """
    where = f'{source}: ' if source is not None else ''
    agent = document.get(AGENT_KEY)
    if isinstance(agent, Mapping) and _NESTED_IO_KEY in agent:
        raise AgentIOConfigError(
            f'{where}`{AGENT_KEY}: {{{_NESTED_IO_KEY}: …}}` declares nothing: only the ROOT key `{AGENT_IO_KEY}:` is read, spelled with a '
            f'literal dot and placed beside `workspace:` / `project:`, not nested under `{AGENT_KEY}:`. Move the block to the root of '
            f'{CONFIG_FILENAME}; an ignored declaration would run every scenario that hands the agent a file or an image against a harness '
            f'that never declared it.'
        )
    if AGENT_IO_KEY not in document:
        return None
    raw = document[AGENT_IO_KEY]
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}` must be a mapping declaring `{INPUT_FIELD}:` and/or `{MAY_OPEN_FIELD}:`, got {type(raw).__name__}. '
            f'Omit the key entirely when the agent takes a plain text turn and never opens the conversation.'
        )
    defects = [AgentIODefect('unknown_field', AGENT_IO_KEY, name, source) for name in sorted(set(map(str, raw)) - set(_KNOWN_FIELDS))]
    opens = _parse_opens(raw, where=where)
    may_open = _parse_may_open(raw, where=where)
    if opens == OPENS_ALWAYS:
        if MAY_OPEN_FIELD in raw and not may_open:
            raise AgentIOConfigError(
                f'{where}`{AGENT_IO_KEY}` declares `{OPENS_FIELD}: {OPENS_ALWAYS}` beside `{MAY_OPEN_FIELD}: false` — a host that always '
                f'opens is driven with an empty message list on turn 0. Drop `{MAY_OPEN_FIELD}: false` (`{OPENS_FIELD}: {OPENS_ALWAYS}` implies it).'
            )
        may_open = True
    parts = _parse_input_parts(raw, where=where, source=source, defects=defects)
    choice_text, choice_text_source = _parse_choice_text(raw, where=where)
    return AgentIO(
        input_parts=parts,
        may_open=may_open,
        opens=opens,
        choice_text=choice_text,
        choice_text_source=choice_text_source,
        channels=_parse_channels(raw, where=where, source=source, defects=defects),
        source=source,
        defects=tuple(defects),
    )


def _parse_channels(raw: Mapping[str, Any], *, where: str, source: Path | None, defects: list[AgentIODefect]) -> tuple[AgentIOChannel, ...]:
    """``agent.io.channels``, validated against :data:`CHANNEL_KINDS`; ``()`` when undeclared.

    Fail closed on every malformed entry — a missing or blank required key, a kind outside the vocabulary,
    a non-string ``handover``, a duplicate name: a scenario plants into a channel BY NAME, so a declaration
    that cannot testify must never degrade to "no such channel". An unknown key in an entry is a defect.
    """
    if CHANNELS_FIELD not in raw:
        return ()
    declared = raw[CHANNELS_FIELD]
    if not isinstance(declared, list):
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}.{CHANNELS_FIELD}` must be a list of {{name, kind, field, source, handover}} entries, got {type(declared).__name__}.'
        )
    channels: list[AgentIOChannel] = []
    for index, entry in enumerate(declared):
        key = f'{AGENT_IO_KEY}.{CHANNELS_FIELD}[{index}]'
        if not isinstance(entry, Mapping):
            raise AgentIOConfigError(f'{where}`{key}` must be a mapping {{name, kind, field, source, handover}}, got {type(entry).__name__}.')
        defects.extend(AgentIODefect('unknown_field', key, name, source) for name in sorted(set(map(str, entry)) - set(_CHANNEL_KEYS)))
        for required in _CHANNEL_REQUIRED:
            value = entry.get(required)
            if not isinstance(value, str) or not value.strip():
                raise AgentIOConfigError(f'{where}`{key}.{required}` must be a non-blank string, got {value!r}.')
        kind = entry['kind'].strip()
        if kind not in CHANNEL_KINDS:
            raise AgentIOConfigError(f'{where}`{key}.kind` must be one of {list(CHANNEL_KINDS)}, got {kind!r}.')
        handover = entry.get('handover')
        if handover is not None and not isinstance(handover, str):
            raise AgentIOConfigError(f'{where}`{key}.handover` must be a string when set, got {handover!r}.')
        name = entry['name'].strip()
        if any(channel.name == name for channel in channels):
            raise AgentIOConfigError(
                f'{where}`{AGENT_IO_KEY}.{CHANNELS_FIELD}` declares the channel {name!r} more than once — scenarios address a channel by name.'
            )
        channels.append(AgentIOChannel(name=name, kind=kind, field=entry['field'].strip(), source=entry['source'].strip(), handover=handover))
    return tuple(channels)


def _parse_choice_text(raw: Mapping[str, Any], *, where: str) -> tuple[Mapping[str, str] | None, str | None]:
    """``agent.io.choice_text`` + ``agent.io.choice_text_source``, or ``(None, None)`` when undeclared.

    Fail closed on every malformed shape: the fact decides whether a turn's words are credited as heard,
    so a value outside :data:`CHOICE_TEXT_VALUES`, a blank choice, a missing or blank source, or a source
    with no fact is refused by name rather than read as "undeclared" or as either verdict.
    """
    declared, source = raw.get(CHOICE_TEXT_FIELD), raw.get(CHOICE_TEXT_SOURCE_FIELD)
    if CHOICE_TEXT_FIELD not in raw:
        if CHOICE_TEXT_SOURCE_FIELD in raw:
            raise AgentIOConfigError(
                f'{where}`{AGENT_IO_KEY}.{CHOICE_TEXT_SOURCE_FIELD}` is set without `{CHOICE_TEXT_FIELD}:` — it cites the resume code a '
                f'`{CHOICE_TEXT_FIELD}` fact was read off. Declare the fact beside it, or drop both.'
            )
        return None, None
    if not isinstance(declared, dict) or not declared:
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}.{CHOICE_TEXT_FIELD}` must be a non-empty mapping of interrupt choice → one of '
            f'{list(CHOICE_TEXT_VALUES)}, got {declared!r}. Omit the key when the resume code was not read.'
        )
    facts: dict[str, str] = {}
    for choice, verdict in declared.items():
        if not isinstance(choice, str) or not choice.strip():
            raise AgentIOConfigError(f'{where}`{AGENT_IO_KEY}.{CHOICE_TEXT_FIELD}` keys must be non-blank choice spellings, got {choice!r}.')
        if verdict not in CHOICE_TEXT_VALUES:
            raise AgentIOConfigError(
                f'{where}`{AGENT_IO_KEY}.{CHOICE_TEXT_FIELD}.{choice}` must be one of {list(CHOICE_TEXT_VALUES)}, got {verdict!r}.'
            )
        facts[choice] = str(verdict)
    if not isinstance(source, str) or not source.strip():
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}.{CHOICE_TEXT_SOURCE_FIELD}` is required beside `{CHOICE_TEXT_FIELD}:` — `<path>:<line>` of the '
            f'resume code the fact was read off, got {source!r}.'
        )
    return MappingProxyType(facts), source.strip()


def _parse_input_parts(raw: Mapping[str, Any], *, where: str, source: Path | None, defects: list[AgentIODefect]) -> tuple[str, ...]:
    """``agent.io.input``, validated against :data:`INPUT_PART_KINDS`.

    An absent ``input:`` is :data:`DEFAULT_INPUT_PARTS` — the harness declared something about
    its opener and nothing about its parts, so it takes the default text turn. An
    ``input: []`` is a DECLARATION of none and is not the same fact.
    """
    if INPUT_FIELD not in raw:
        return DEFAULT_INPUT_PARTS
    declared = raw[INPUT_FIELD]
    if not isinstance(declared, list):
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}.{INPUT_FIELD}` must be a list of content-part kinds, got {type(declared).__name__}. '
            f'Known kinds: {sorted(INPUT_PART_KINDS)}.'
        )
    parts: list[str] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, str) or not entry.strip():
            raise AgentIOConfigError(f'{where}`{AGENT_IO_KEY}.{INPUT_FIELD}[{index}]` must be a non-blank content-part kind, got {entry!r}.')
        kind = entry.strip()
        if kind not in INPUT_PART_KINDS:
            # BY NAME, never flattered into `text`: a kind this build cannot put on a turn is dropped
            # from the surface and named as a defect; a turn that carries one is still refused by name.
            defects.append(AgentIODefect('unknown_part_kind', f'{AGENT_IO_KEY}.{INPUT_FIELD}[{index}]', kind, source))
            continue
        if kind in parts:
            raise AgentIOConfigError(f'{where}`{AGENT_IO_KEY}.{INPUT_FIELD}` lists {kind!r} more than once — the surface is a set of kinds.')
        parts.append(kind)
    return tuple(parts)


def _parse_may_open(raw: Mapping[str, Any], *, where: str) -> bool:
    """``agent.io.may_open`` — strictly a YAML boolean.

    Not ``bool(raw)``: ``may_open: "false"`` is a true string, and a harness that would crash on
    an empty opener must never be told it can serve one by a quoting accident.
    """
    if MAY_OPEN_FIELD not in raw:
        return False
    declared = raw[MAY_OPEN_FIELD]
    if not isinstance(declared, bool):
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}.{MAY_OPEN_FIELD}` must be true or false, got {declared!r}. '
            f'It declares whether `invoke_agent` can be called with an EMPTY message list (the agent opens the conversation); '
            f'a harness that cannot serve one crashes on turn 0, so this is not a value to guess at.'
        )
    return declared


def _parse_opens(raw: Mapping[str, Any], *, where: str) -> str | None:
    """``agent.io.opens`` — ``always`` or absent; any other value is refused by name."""
    if OPENS_FIELD not in raw:
        return None
    declared = raw[OPENS_FIELD]
    if declared not in _OPENS_VALUES:
        raise AgentIOConfigError(
            f'{where}`{AGENT_IO_KEY}.{OPENS_FIELD}` must be one of {list(_OPENS_VALUES)}, got {declared!r}. '
            f'Declare `{OPENS_FIELD}: {OPENS_ALWAYS}` when the production host speaks first in EVERY conversation; omit it otherwise.'
        )
    return str(declared)


def discover_agent_io(roots: Sequence[Path | str]) -> AgentIO | None:
    """The surface declared under ``roots``, or ``None`` when no config under them declares one.

    Each root is read the way every workspace declaration is (two rungs; a config that exists
    and cannot be read raises, whether or not it carries this key). A file reached from two
    roots is read once. Two DIFFERENT files that both declare and disagree raise: the surface is
    env-scoped, and a discovery order does not get to settle which one the env meant.
    """
    seen: set[Path] = set()
    declared: list[AgentIO] = []
    for root in roots:
        for path, document in discover_config_documents(root, error=AgentIOConfigError):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            surface = parse_agent_io(document, source=path)
            if surface is not None:
                declared.append(surface)
    if not declared:
        return None
    if len({(frozenset(io.input_parts), io.may_open, io.opens, frozenset((io.choice_text or {}).items()), frozenset(io.channels)) for io in declared}) > 1:
        sources = '; '.join(f'{io.source} → {io.as_record()}' for io in declared)
        raise AgentIOConfigError(
            f'conflicting `{AGENT_IO_KEY}` declarations ({sources}); the turn-input surface is env-scoped and must agree. '
            f'Reconcile the {CONFIG_FILENAME} files (or remove one) so the env declares a single surface.'
        )
    # Agreement is judged on the surface AFTER dropping; every file's defects are reported.
    return replace(declared[0], defects=tuple(d for io in declared for d in io.defects))


def declared_agent_io(simulate_path: Path | str) -> AgentIO | None:
    """The declaration beside ``simulate.py``, or ``None`` when nothing declares one.

    The two rungs ``orchestrator._declared_shape`` resolves ``shape:`` at — the harness
    directory and the repo root above it — so a workspace has one layout to learn and the two
    declarations that describe the same bench are never found in different places.
    """
    path = Path(simulate_path)
    return discover_agent_io(bench_config_roots(path))


__all__ = [
    'AGENT_IO_KEY',
    'AGENT_KEY',
    'CHANNELS_FIELD',
    'CHANNEL_KINDS',
    'CHOICE_TEXT_DELIVERED',
    'CHOICE_TEXT_DROPPED',
    'CHOICE_TEXT_FIELD',
    'CHOICE_TEXT_SOURCE_FIELD',
    'CHOICE_TEXT_VALUES',
    'DEFAULT_AGENT_IO',
    'DEFAULT_INPUT_PARTS',
    'INPUT_FIELD',
    'INPUT_PART_KINDS',
    'MAY_OPEN_FIELD',
    'OPENS_ALWAYS',
    'OPENS_FIELD',
    'AgentIO',
    'AgentIOChannel',
    'AgentIOConfigError',
    'AgentIODefect',
    'ChannelKind',
    'declared_agent_io',
    'discover_agent_io',
    'parse_agent_io',
]
