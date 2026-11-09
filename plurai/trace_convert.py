"""Client-side trace → scenario-bundle conversion (the trace-resume entry path).

Turns a **trace export** on disk into ONE scenario-bundle folder carrying an authoritative
**resume pin** — the artifact that makes the generated scenario resume the customer's
conversation from the end of the trace instead of opening a fresh one.

Sibling of :mod:`plurai.trace_extract` (which COLLECTS traces from the sink); this module
CONVERTS a collected export into the bundle shape ``plurai ingest-bundle`` uploads. Its default
input is exactly ``trace_extract``'s output file shape — ``traces/<trace_id>.json``, a JSON list
of LangSmith run dicts — so an ``ingest-traces --out-dir X`` run and a hand-collected export
directory are the same input here.

**Two tiers read a format, and the split is by RECURRENCE.** The module keeps a registry,
:data:`TRACE_FORMAT_REGISTRY`, mirroring ``trace_extract``'s ``TRACE_SOURCE_REGISTRY``, and it has
two members: LangSmith, and the OpenInference span-tree reader. That is the first tier and it is
deliberately hard to join, because an unverified reader is a standing liability rather than a feature.
Everything else — a customer's own stack, a benchmark corpus, a session log
— is read by an ADAPTER: a YAML mapping document committed to the workspace, saying only WHERE in
that format each pinned field lives, executed by fixed code in :mod:`plurai.trace_adapter`. So
novelty costs a file rather than a release, and a model may author the mapping without ever
touching the bytes.

**The seam is :class:`SourceProjection`.** A front end — built-in or adapter — answers only the
question its format can answer: where the conversation, the ending state and the identifying
metadata are. It returns messages in their SOURCE shape, and everything below
:func:`canonical_messages` is shared by every format alike. The first-class LangSmith reader is
never re-expressed (see :func:`_langsmith_front_end`), so its bytes stay identical structurally
rather than by assertion: it is reached through one more indirection and nothing else.

**Pure code, no LLM, ever.** An LLM anywhere in this path is precisely what the true-resume
fidelity bar forbids: the whole point of the pin is that the verbatim message history and agent
state reach the customer's checkpointer without passing through a generation model that could
paraphrase them.

Emitted folder (one per thread export, under ``--out-dir``)::

    <thread-folder>/ROLES.md          # required by the bundle store; describes the two files below, and
                                      # under a cut of a production source ends on the WORLD FACTS
    <thread-folder>/resume_pin.json   # THE PIN — byte-preserved into the scenario's resume_seed.yaml
    <thread-folder>/transcript.md     # deterministic rendering; authoring conditioning only

The pin envelope (the contracted shape, declared by :mod:`plurai.resume_pin` and never
re-derived here)::

    {"schema_version": 2,
     "provenance": {...},                 # source file, thread id, trace timestamps
     "messages": [ {type, content, id, ...} ],
     "agent_state": {...},                # the final `update_state` values, minus `messages`
     "resume_point": {"node": "..."} }    # or null — the operator's `--start-node`

``messages`` + ``agent_state`` are the two keys the backend lifts VERBATIM into the scenario folder's
``resume_seed.yaml``, which is why they are spelled the same on both sides. ``resume_point`` is
carried the same way and read by nothing here: it names the top-level graph node the customer's
harness enters on turn 1 of the resumed leg, and only that harness knows the graph, so its ONE
source is the ``--start-node`` flag — never a guess derived from the trace. One invocation pins
one node across every thread it converts.

Every format question is resolved HERE; format knowledge never enters the backend, and a bundle
gives away nothing about which tier read it.

**The pin is BORN from :mod:`plurai.resume_pin`**, the shared model the backend declares its own
identical copy of — so this module PROJECTS a source export onto that schema rather than
restating it. Each message keeps exactly ``type``, ``content`` and ``id``; ai messages add
``tool_calls: [{name, args, id}]`` with ``args`` as parsed dicts; tool messages add
``tool_call_id``. Nothing else — ``additional_kwargs``, ``response_metadata``, and a tool
message's ``name``/``status`` are dropped, which ``extra='forbid'`` on the models makes structural
rather than a rule this file remembers to apply. ``id`` is REQUIRED on every message: LangGraph's
``add_messages`` reconciles by id, and that is what makes re-seeding idempotent, so a message
without one fails the export rather than getting a synthesized id. A front end whose format leaves
ids out may MINT them — an adapter for a format that records none, the LangSmith reader for a
message serialized without one, or the simulation-run reader, since a run records no id for the
user side and a harness may return agent messages without one — deterministically (keyed on the
thread and the message's SOURCE position, never on a random value) and counted in provenance. It
does so upstream of this guard, which therefore still never invents one. ``type`` is emitted
from the CLOSED set :data:`CANONICAL_MESSAGE_TYPES` — the aliases LangChain accepts (``user``/``assistant``/``developer``) are folded to the spelling it
deserializes them to, so the reader on the backend side has four values to code against rather
than seven.

Because the pin is the model's ``model_dump()``, its key order is the models' FIELD DECLARATION
order and an ai message that made no calls carries an explicit ``tool_calls: null`` rather than
omitting the key. Both are deliberate: one shape per message type, identical on both sides of the
contract, which is what makes byte-verbatim carriage checkable rather than hopeful.

**Fail-closed per thread.** Every shape the converter cannot faithfully carry — an unknown message
type, an unparseable ``tool_calls`` arg, a tool message with no ``tool_call_id``, a seed the backend
refuses — refuses THAT THREAD with a :class:`ThreadRefused` carrying its gate finding, instead of
emitting a lossy pin. A pin that quietly drops half a tool exchange would leave the run green while
violating the fidelity bar, which is the same reasoning that makes the backend fail a malformed pin
closed. One thread's defect is not the directory's: the other threads of a directory run are still
written, and the run reports every skipped thread beside the folders it wrote. The same stance
covers the destination: a thread that would REPLACE a conversation already staged in ``--out-dir``
is skipped rather than written over it. What a front end may do instead of refusing is NORMALISE a
lossless spelling difference — a role-keyed message, a null assistant content beside its calls, a
byte-order mark — and every normalisation is counted in the pin's provenance. Nothing relaxes the
seed-shape mirror below: a relaxation is a normalisation upstream of it, never a looser rule.

**Posture gate.** The bundle lane bypasses the SDK's ingest posture scrubber, and ``anonymize:
true`` is a once-per-env human consent act that the server never holds verbatim content. So the
converter reads the env's declared posture (:mod:`plurai.ingest.posture`) — under
``--workspace-root`` AND in the workspace the export itself sits in — and REFUSES to write anything
when either is anonymized: trace-to-scenario is raw-only. That refusal is the invocation's, never a
thread's, and no overrule lifts it.
"""

from __future__ import annotations

import io
import json
import logging
import re
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from plurai import __version__
from plurai.gates.apply import make_finding
from plurai.gates.catalogue import (
    TRACE_CUT_OUT_OF_RANGE,
    TRACE_EXPORT_MALFORMED,
    TRACE_FOLDER_CONFLICT,
    TRACE_FORMAT_UNSUPPORTED,
    TRACE_INVALID_TOOL_CALLS_DROPPED,
    TRACE_LANGSMITH_ROOT_SELECTION,
    TRACE_MESSAGE_UNREPRESENTABLE,
    TRACE_MESSAGES_CHANNEL_INFERRED,
    TRACE_ORPHAN_TOOL_RESULTS_DROPPED,
    TRACE_POSTURE_ANONYMIZED,
    TRACE_SEED_SHAPE_REFUSED,
)
from plurai import readers
from plurai.gates.contract import GateFinding
from plurai.ingest.posture import resolve_posture
from plurai.ingest.workspace_config import CONFIG_CANDIDATE_RELPATHS
from plurai.observability.tool_call_recovery import AGENT_STATE_TOOL_TRAIL_KEY, TRAIL_PARENT_KEY
from plurai.resume_pin import (
    CANONICAL_MESSAGE_TYPES,
    RESUME_PIN_SCHEMA_VERSION,
    AIPinMessage,
    HumanPinMessage,
    ResumePinDocument,
    ResumePinMessage,
    ResumePinProvenance,
    ResumePoint,
    ResumeToolCall,
    SystemPinMessage,
    ToolPinMessage,
)
from plurai.run_artifacts import RESUME_STATE_DROPPED_KEY, SCHEMA_NOT_LOADED, SEED_RESTORE_KEY, declared_message_type, load_transcript

logger = logging.getLogger(__name__)

#: The pin's filename inside each scenario-bundle folder. JSON, following the
#: ``agent_context.json`` precedent for structured authoritative values in a bundle folder (the
#: backend-side artifact is YAML — ``resume_seed.yaml`` — because scenario folders are YAML).
#: **Wire contract**: the backend reads this exact name.
RESUME_PIN_FILENAME = 'resume_pin.json'

#: The bundle store's required per-folder description file.
ROLES_FILENAME = 'ROLES.md'

#: Deterministic human/LLM-readable rendering of the conversation, staged as the folder's
#: authoring conditioning (the store inlines only ``ROLES.md``; materials are staged on disk).
TRANSCRIPT_FILENAME = 'transcript.md'

#: Longest ``ROLES.md`` opening line carried into the overview's hand-authored table. The cell is
#: an identifying line, not the folder's documentation; the folder's own file is what an agent
#: reads for the rest.
PLAIN_DESCRIPTION_MAX_CHARS = 120

#: The set-level overview at the bundle root. Must equal ``cli.BUNDLE_SET_OVERVIEW`` — that
#: constant is what ``bundle_units`` excludes from the upload units and passes inline instead.
#: It is restated rather than imported because importing it here would close an import cycle with the CLI.
BUNDLE_OVERVIEW_FILENAME = 'BUNDLE.md'

#: Input spellings folded to their canonical form on the way into the pin. This is exactly the
#: mapping ``convert_to_messages`` applies when it deserializes, so folding here changes no
#: reconstruction — it just means the pin states the type the backend will see rather than an alias a
#: raw-dict reader on that side could miss. The aliases are the converter's INPUT vocabulary and
#: are deliberately NOT members of the pin's union: they must never reach a pin.
_TYPE_ALIASES = {'user': 'human', 'assistant': 'ai', 'developer': 'system'}

#: Message ``type`` values accepted on INPUT — the pin's own closed set plus the aliases folded
#: into it. DERIVED, so a member joining the shared model's union is accepted here without a
#: second edit that could be forgotten. Validated at conversion so an unknown type fails beside
#: the trace that caused it rather than inside the customer's agent process at seed time.
#: ``function`` is absent from both halves on purpose: ``convert_to_messages`` builds a
#: ``FunctionMessage`` from it, which REQUIRES a ``name`` the closed pin schema has no room for,
#: so a legacy function-calling trace fails closed here.
SUPPORTED_MESSAGE_TYPES = CANONICAL_MESSAGE_TYPES | frozenset(_TYPE_ALIASES)

#: The backend validates the scenario folder's ``resume_seed.yaml`` against a 256 KB
#: file-size cap at authoring. The converter cannot enforce that cap —
#: the YAML it warns about is serialized on the other side — but an oversized pin is worth saying
#: out loud here, where the operator can still pick a shorter thread, rather than a stage later.
SEED_SIZE_WARN_BYTES = 256 * 1024

#: Provenance keys whose meaning the converter fixes: each is a statement about the PIN it builds —
#: the file and version that produced it, what kind of source it came from, the cut and trim it
#: applied, which state it carries, and where its turn numbering starts and how many turns that
#: numbering holds — whether the value is computed during conversion or counted by a built-in
#: reader. Most are stamped only when the thing they record happened, and code reads them back to
#: act on them — the ``ROLES.md`` state phrase, the bundle overview's origin, the tail warning, the
#: CLI summary — so a value from anywhere else is a claim about the pin that nothing checked. An
#: adapter maps an artifact's own fields, so it may declare none of these:
#: ``plurai.trace_adapter.RESERVED_PROVENANCE_KEYS`` is built from this set, which is what keeps a
#: key the converter stamps from being declarable there.
CONVERTER_PROVENANCE_KEYS = frozenset(
    {
        'source_file',
        'converted_by',
        'origin',
        'cut_turn',
        'agent_state_at',
        'state_withheld',
        'trimmed_mid_turn_tail',
        'resume_seed_messages',
        'user_turns',
        # What a normalisation or a drop did to the pin — each stamped only when it happened.
        'messages_channel',
        'normalized_content',
        'dropped_invalid_tool_calls',
        'dropped_orphan_tool_results',
        'reordered_user_turns',
        'history_before_capture',
        'unrecognised_record_count',
        'unrecognised_shapes',
        'folded_terminal_answers',
    }
)

#: The title of the ``ROLES.md`` section that carries the world facts. Exact, because a caller
#: copies the folder's world facts from this heading to the end of the file.
WORLD_FACTS_HEADING = 'World the live leg needs'

#: Longest rendering of one tool call's arguments, or of one result, carried into the world facts.
#: A tool can return a whole document or a base64 image, and the section is read by an agent in
#: full, so a value past this is cut and the cut is stated beside it.
WORLD_FACT_VALUE_MAX_BYTES = 8 * 1024

#: Longest world-facts section, heading to EOF and closing note included, at either depth it is
#: written at. Whole entries only, in recorded order; what does not fit is named in the closing note
#: rather than cut through the middle of an exchange.
WORLD_FACTS_MAX_BYTES = 64 * 1024


class TraceConversionError(Exception):
    """A trace export could not be faithfully converted (fail-closed, user-actionable).

    Raised for every shape the converter cannot carry verbatim rather than emitting a lossy pin:
    an export with no root run, a message missing its required ``id``, an unknown message type, a
    tool message with no ``tool_call_id``, or tool-call args that are not a parsed object.

    Raised bare, it refuses the whole INVOCATION — an operator's flag, an empty directory, an
    unloadable adapter. A refusal of one thread is a :class:`ThreadRefused`."""


class ThreadRefused(TraceConversionError):  # noqa: N818 - a refusal of one thread, not an error of the run
    """ONE thread cannot be converted faithfully; every other thread of the invocation proceeds.

    Carries the thread's :class:`~plurai.gates.contract.GateFinding`, an invariant one: no overrule
    lifts it, because what it guards is the verbatim pin. Its message is the finding's remedy, which
    says what is wrong with the thread and what would make it convert.
    """

    def __init__(self, finding: GateFinding) -> None:
        super().__init__(finding.remedy)
        self.finding = finding


class FormatUnsupportedError(TraceConversionError):
    """The artifact is written in a convention no reader of this SDK reads — named, never mapped by resemblance."""

    def __init__(self, message: str, *, convention: str) -> None:
        super().__init__(message)
        self.convention = convention


class PostureRefusedError(TraceConversionError):
    """The env declares ``anonymize: true`` and trace-to-scenario is raw-only.

    A distinct type because this is a CONSENT refusal, not a malformed input: nothing about the
    trace is wrong, the env simply has not consented to verbatim content leaving it. It refuses the
    invocation before any export is read, and carries its security finding, which no overrule lifts."""

    def __init__(self, message: str, *, finding: GateFinding) -> None:
        super().__init__(message)
        self.finding = finding


#: What a folder's ``ROLES.md`` calls the thing a conversation was read from when nothing else
#: says. The first-class reader's own word, and the reading a pin that records no origin gets:
#: absent means the pin was written from a trace export.
_DEFAULT_ORIGIN = 'production trace'

#: The remedy offered for a history captured mid-turn when the format can be captured again.
_DEFAULT_RECAPTURE_REMEDY = 'Re-export the thread once the turn has completed'


@dataclass(frozen=True)
class WorldFact:
    """One tool exchange a source recorded after the cut: a call and its result, paired by call id."""

    #: The turn it happened on, numbered as ``--cut-turn`` numbers turns.
    turn: int
    call_id: str
    #: The call's tool name and arguments. ``None`` for a result whose call the post-cut part does
    #: not hold — kept as its own entry, since it is still something a tool returned.
    name: str | None = None
    args: dict[str, Any] | None = None
    #: Whether the source recorded a result. Apart from ``result`` because a recorded result can
    #: itself be empty, and "the tool returned nothing" is a different fact from "no reply arrived".
    answered: bool = False
    result: Any = None


@dataclass(frozen=True)
class WorldFacts:
    """The tool exchanges a production source recorded after a cut, in recorded order."""

    cut_turn: int
    #: The source's last turn, so a section with nothing to show can still name the turns it covers.
    last_turn: int
    entries: tuple[WorldFact, ...] = ()


@dataclass(frozen=True)
class ThreadConversion:
    """One converted thread export: the folder name to write it under, plus the pin envelope."""

    #: Folder name for this thread inside the bundle dir (the bundle store's ``<scenario>`` segment).
    folder: str
    #: The full :data:`RESUME_PIN_FILENAME` envelope, as the SHARED model rather than a loose dict —
    #: so the discriminated union narrows each message for the renderers below, and the on-disk
    #: bytes are the model's own serialization rather than a second, hand-kept spelling of it.
    pin: ResumePinDocument
    #: The source export path, kept for error messages and the CLI summary.
    source: Path
    #: The ``--source-format`` that read it; ``None`` for a folder read back off disk, whose pin does
    #: not record which reader wrote it.
    source_format: str | None
    #: What ``ROLES.md`` and ``BUNDLE.md`` call the thing this conversation was read from — the
    #: format's own :attr:`_FormatSpec.origin`, recorded in the pin so a folder read back off disk
    #: (:func:`pinned_folders`) recovers it rather than guessing from the source file name.
    #: Defaulted because a pin with no origin key came from a trace export.
    origin: str = _DEFAULT_ORIGIN
    #: What to tell an operator to do about a history captured mid-turn, as an imperative clause —
    #: :attr:`_FormatSpec.recapture_remedy`. Format-owned because the remedy differs in kind: a
    #: live sink can be exported again once the turn completes, while a finished run has no later
    #: capture to take and its transcript is all there will ever be.
    recapture_remedy: str = _DEFAULT_RECAPTURE_REMEDY
    #: How many user turns the SOURCE recorded, numbered as ``--cut-turn`` numbers them and counted
    #: before any cut or trim — the range a caller picks a cut from, which a cut must not shrink.
    #: ``None`` for a folder read back off disk, where only the pin survives and the source does not.
    source_user_turns: int | None = None
    #: How many leading messages are a resume seed, as the READER counted them
    #: (:attr:`SourceProjection.resume_seed_messages`). Carried beside the pin rather than read out
    #: of its provenance, which is an open shape a format may write to: the rendering numbers turns
    #: from here, and a count taken from anywhere else would number them differently from the cut.
    resume_seed_messages: int = 0
    #: What the source's tools returned after the cut, for ``ROLES.md`` — never for the pin, whose
    #: bytes are the same with or without it. ``None`` unless this is a cut of a source whose format
    #: carries world facts (:attr:`_FormatSpec.world_facts`).
    world_facts: WorldFacts | None = None
    #: The advisory findings the conversion made about this thread — a root it did not pin, a channel it
    #: inferred, what it dropped. Never part of the pin, whose provenance already counts what changed;
    #: the run reports them. Empty for a folder read back off disk.
    findings: tuple[GateFinding, ...] = ()

    @property
    def messages(self) -> list[ResumePinMessage]:
        return self.pin.messages

    @property
    def agent_state(self) -> dict[str, Any]:
        return self.pin.agent_state

    @property
    def resume_point(self) -> ResumePoint | None:
        """The pinned start node, or ``None`` when the operator named none."""
        return self.pin.resume_point

    @cached_property
    def provenance(self) -> dict[str, Any]:
        """Provenance as a plain mapping — the pin's ONE open shape.

        Its trace window, run count and converter stamp are EXTRAS on an ``extra='allow'`` model
        (that is what lets this converter record more of them without a lockstep backend release),
        so they are keys rather than fields and are read as keys.

        Cached, and read-only by contract like :attr:`payload`: the two renderers, the bundle table
        and the CLI summary all ask for it, and a caller that mutated the mapping would be editing
        every other reader's copy rather than its own snapshot.
        """
        return self.pin.provenance.model_dump()

    @cached_property
    def payload(self) -> dict[str, Any]:
        """The pin exactly as it reaches disk — key order, explicit nulls and all.

        Every place that SERIALIZES or COMPARES the pin goes through this, or through
        :attr:`pin_text` on top of it, so "what is written" and "what a previous run wrote" are
        the same rendering by construction rather than by two code paths agreeing. Cached because
        several of them ask per thread — the re-run idempotence comparison, the on-disk text and
        the seed-size warning — and a whole-pin ``model_dump`` of a long conversation is not free.
        Read-only by contract, like the frozen dataclass around it: a caller that mutated it would
        be editing every other reader's copy.
        """
        return self.pin.model_dump()

    @cached_property
    def pin_text(self) -> str:
        """The pin's on-disk TEXT — byte-for-byte what :func:`write_scenario_folder` writes.

        The ONE place :attr:`payload` is turned into characters, so a caller that wants to
        MEASURE the file (the CLI summary's ``pin_bytes``) reads the same rendering the writer
        emits instead of restating its ``json.dumps`` arguments at a second site. Two such sites
        drift: a compact dump with no ``indent`` and no trailing newline under-reports the file
        size, which is invisible in review because a size taken off a different rendering is still
        a plausible number.
        """
        return json.dumps(self.payload, indent=2, ensure_ascii=False) + '\n'


@dataclass(frozen=True)
class SourceProjection:
    """What a per-format FRONT END produces — the seam every source format meets the pin at.

    The front end answers only the question its format can answer: *where in this artifact is the
    conversation, the state it ended on, and what identifies it?* It hands back the messages in
    their SOURCE shape, not the pin's, so the canonical projection
    (:func:`canonical_messages` — required ids, the closed type set, parsed tool args, paired tool
    results) runs ONCE, in the shared half, for every format alike. A front end that canonicalized
    its own messages would be a second place for those rules to be almost-right.
    """

    #: The thread's messages as the SOURCE spells them — LangSmith-serialized message dicts, or an
    #: adapter's projection onto that same shape. :func:`canonical_messages` is what turns these
    #: into pin messages, and it is the only thing that does.
    messages: list[Any]
    #: The state values to ferry alongside the history. ``{}`` is legal and ordinary: an agent
    #: whose whole state lives in ``messages`` — every source that is a transcript rather than a
    #: graph checkpoint — has nothing else to carry, and the pin schema requires the key, not
    #: content in it.
    agent_state: dict[str, Any] = field(default_factory=dict)
    #: Provenance keys identifying THIS artifact, merged into the envelope's provenance between
    #: ``thread_id`` and ``converted_by``. Extras on the pin's one open shape, so a format may
    #: record what it knows (a run window, an adapter's digest) without a lockstep backend release.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: The state the agent ended each USER TURN on, entry *i* for turn *i* — supplied only by a
    #: format that records state per turn. It is what makes :func:`cut_at_turn` honest about the
    #: other half of the seed: a cut at turn *k* pins the state as of turn *k-1*, so the resumed
    #: agent does not open holding channels its own seeded history has not reached yet. A format
    #: with only a final state leaves this empty and a cut carries :attr:`agent_state` unchanged,
    #: which is the most that source can say.
    agent_state_by_turn: list[dict[str, Any]] = field(default_factory=list)
    #: How many of :attr:`messages`' LEADING entries are a resume seed — conversation the source
    #: started from rather than recorded. Non-zero only for a recording that was itself resumed,
    #: whose own turns begin after them. A cut counts those turns and never the seed's: the turns an
    #: operator can name are the ones the source recorded, and the seed rides along whole, which is
    #: also why a cut at turn 0 is legal here — the seed is the history it keeps.
    resume_seed_messages: int = 0
    #: The state the source's first recorded turn STARTED from — the resume seed's own state — so a
    #: cut at turn 0 pins what the agent held at that moment rather than a state it reached on a
    #: turn the cut replays. ``None`` when the source was not resumed.
    resume_seed_state: dict[str, Any] | None = None
    #: The name the bundle FOLDER takes when the operator names none, in place of the artifact's
    #: file stem. For a format whose artifacts all share one filename — a run's per-scenario
    #: ``transcript.json`` — the stem identifies nothing, so every conversion would land in one
    #: folder named after the filename. ``None`` keeps the stem, which is what every format whose
    #: artifact is named for its thread wants.
    folder_stem: str | None = None
    #: Advisory findings the front end made while reading — each about this artifact, each lossless
    #: or counted in :attr:`provenance`, so the thread still converts.
    findings: tuple[GateFinding, ...] = ()


@dataclass(frozen=True)
class _FormatSpec:
    """How to read one source format + what to tell an operator it is."""

    front_end: Callable[[Path], SourceProjection]
    #: One line, shown when a `--source-format` fails to resolve. The registry is the only place
    #: that can enumerate what this SDK can read, so it also has to be able to say what each is.
    description: str
    #: The ``--source-format`` name this spec is registered or filed under — the evidence a thread's
    #: finding names its reader by.
    name: str
    #: What the folder's ``ROLES.md`` calls the thing the conversation was read from, as a noun
    #: phrase. The generation agent authors from that file alone, so a folder built from a
    #: simulation run must not tell it the conversation is production traffic: the two carry
    #: opposite weight when the agent decides how literally to take what was said.
    origin: str = _DEFAULT_ORIGIN
    #: What an operator can do about a conversation this format captured mid-turn, as an imperative
    #: clause completed by the refusal around it. Format-owned because a sink still holding the
    #: thread can be exported again, while a run that has finished cannot be captured a second
    #: time — its transcript is the whole record, so the way past a half turn is to stop before it.
    recapture_remedy: str = _DEFAULT_RECAPTURE_REMEDY
    #: Which file extensions a DIRECTORY of this format holds. Format-owned rather than fixed:
    #: a Claude Code session log is `.jsonl`, and a `*.json` glob would report an obviously
    #: non-empty directory as holding nothing convertible.
    suffixes: tuple[str, ...] = ('.json',)
    #: Whether a cut writes the WORLD FACTS — the tool calls and results the source recorded after
    #: the cut — into ``ROLES.md``. True for a production source, whose results are the only record
    #: of what the real tools answered. False for a simulation run: its post-cut results came from
    #: the mock world itself, and handing them back to the world author as constraints is circular.
    world_facts: bool = True


# --------------------------------------------------------------------------- #
# Run-tree selection (LangSmith)                                               #
# --------------------------------------------------------------------------- #


def _started_at(run: Mapping[str, Any]) -> tuple[int, float, str]:
    """Sort key for a run's ``start_time``: the parsed instant, with ``dotted_order`` as tiebreak.

    Compared as instants, NOT as strings. ``…T15:00:00+03:00`` is half an hour EARLIER than
    ``…T12:30:00Z`` yet sorts after it lexicographically, and ``Z`` vs ``+00:00`` suffixes and
    present-vs-absent fractional seconds mis-sort the same way. LangSmith is self-consistent
    today, so this only bites a hand-merged or mixed-provenance export — but there it would pick
    the wrong final state silently, which is the failure this module exists to prevent.

    A stamp that will not parse ranks BELOW every parseable one, so a malformed run can never win
    selection on a technicality.
    """
    raw = str(run.get('start_time') or '')
    tiebreak = str(run.get('dotted_order') or '')
    try:
        return (1, datetime.fromisoformat(raw).timestamp(), tiebreak)
    except ValueError:
        return (0, 0.0, raw + tiebreak)


#: The state channel a LangGraph app keeps its conversation in, read without inference when present.
DEFAULT_MESSAGES_CHANNEL = 'messages'


def _root_runs(runs: list[Any], *, source: str) -> list[dict[str, Any]]:
    """The export's root runs, or a refusal: a run tree with no root has no final state to pin."""
    roots = [r for r in runs if isinstance(r, dict) and r.get('parent_run_id') is None]
    if not roots:
        raise TraceConversionError(
            f'{source}: no root run (every run has a parent_run_id), so the thread has no final state to pin. '
            'A thread export must contain the run tree, not a subtree — re-export the whole trace.'
        )
    return roots


def _is_message_like(value: Any) -> bool:
    """Whether ``value`` is spelled as a message: a ``content`` key beside a type or role the converter reads."""
    if not isinstance(value, Mapping) or 'content' not in value:
        return False
    declared = value.get('type') if value.get('type') not in (None, 'chat') else value.get('role')
    return isinstance(declared, str) and declared in SUPPORTED_MESSAGE_TYPES


def resolve_messages_channel(roots: Sequence[Mapping[str, Any]], *, source: str) -> tuple[str, bool]:
    """``(the output-state channel holding the conversation, whether it was inferred)``, or a refusal.

    ``messages`` whenever a root's output state has it. Otherwise the ONE channel, across the roots'
    output states, whose value is a non-empty list of message-spelled dicts: an app that keeps its
    conversation under ``chat_history`` holds the same history under another name. None, or two such
    channels, is a question this reader will not answer by preference — which of two histories is
    the conversation is a fact about the app.
    """
    states = [root['outputs'] for root in roots if isinstance(root.get('outputs'), dict)]
    if not states:
        final = max(roots, key=_started_at)
        raise TraceConversionError(
            f'{source}: the final root run {final.get("id")!r} has no output state (outputs is '
            f'{type(final.get("outputs")).__name__}) — status {final.get("status")!r}, error {_shorten_error(final.get("error"))!r} — '
            'and no earlier root has one either. A trace that never produced a state cannot seed a resume.'
        )
    if any(DEFAULT_MESSAGES_CHANNEL in state for state in states):
        return DEFAULT_MESSAGES_CHANNEL, False
    candidates = sorted(
        {name for state in states for name, value in state.items() if isinstance(value, list) and value and all(_is_message_like(item) for item in value)}
    )
    if len(candidates) == 1:
        return candidates[0], True
    channels = sorted({str(name) for state in states for name in state})[:12]
    found = f'{len(candidates)} channels hold one ({candidates})' if candidates else 'no channel holds one'
    raise TraceConversionError(
        f'{source}: the root runs\' output states have no "{DEFAULT_MESSAGES_CHANNEL}" channel, and {found} — a non-empty '
        f'list of messages, each with "content" and a type or role. Channels: {channels}. Which history is the conversation '
        'is a fact about the app, not one to guess; author an adapter for this export with /author-trace-adapter.'
    )


def _shorten_error(error: Any) -> Any:
    """A run's recorded error, cut to a line an operator can read in a refusal."""
    if not isinstance(error, str):
        return error
    first = error.strip().splitlines()[0] if error.strip() else error
    return first if len(first) <= 160 else f'{first[:160]}…'


def _channel_messages(run: Mapping[str, Any], channel: str) -> list[Any]:
    """The run's output-state ``channel`` when it is a non-empty list, else ``[]``."""
    outputs = run.get('outputs')
    messages = outputs.get(channel) if isinstance(outputs, dict) else None
    return messages if isinstance(messages, list) else []


def _message_ids(messages: Sequence[Any]) -> set[str]:
    return {m['id'] for m in messages if isinstance(m, Mapping) and isinstance(m.get('id'), str) and m['id'].strip()}


@dataclass(frozen=True)
class RootSelection:
    """The root run whose output state is pinned, and — when that choice was not the only reading — why."""

    run: dict[str, Any]
    #: :data:`~plurai.gates.catalogue.TRACE_LANGSMITH_ROOT_SELECTION` evidence, or ``None`` when every
    #: other root agrees with the choice.
    evidence: dict[str, Any] | None


def select_final_run(runs: list[Any], *, source: str, channel: str) -> RootSelection:
    """The root run whose ``channel`` state holds the thread — chosen by LINEAGE, not by recency alone.

    A thread export is a flat list of the thread's runs (roots plus their descendants). Under a
    checkpointed LangGraph each root run's output state carries the conversation ACCUMULATED so far,
    so the roots of one conversation SHARE message ids: each later state re-carries the earlier
    messages under their ids. That shared id is what ties a root to the conversation, so:

    * only roots whose state holds a non-empty ``channel`` list are candidates — a crashed or timed-out
      last turn has no state, and the turn before it is the conversation as far as it got;
    * the pin is the LATEST candidate that shares a message id with another candidate. A side graph
      (title or summary generation) that finishes last shares none, and loses; a summarising app whose
      latest state is shorter still shares ids, and wins;
    * when no two candidates share an id (one root, ids absent, or an app whose roots each output only
      their own turn), nothing ties the roots to one conversation, so recency is the only reading left:
      the latest candidate is pinned only when it holds at least as many messages as the longest. A
      SHORTER latest root is the side graph finishing last over a conversation whose roots cannot be
      told apart from it, and pinning it would silently truncate the conversation — the thread is
      refused instead.

    Every other choice another reading of the export would have made differently is reported as
    evidence for a finding rather than refused: a later root skipped for lack of state, roots sharing
    no id with the pinned one, and a pinned state shorter than the longest (only ever a latest state
    that shares ids with an earlier, longer one — a summarised history).
    """
    roots = sorted(_root_runs(runs, source=source), key=_started_at)
    usable = [index for index, root in enumerate(roots) if _channel_messages(root, channel)]
    if not usable:
        final = roots[-1]
        raise TraceConversionError(
            f'{source}: no root run\'s output state carries a non-empty "{channel}" list — the final root run '
            f'{final.get("id")!r} has status {final.get("status")!r}, error {_shorten_error(final.get("error"))!r}. '
            'There is no conversation to resume.'
        )
    ids = {index: _message_ids(_channel_messages(roots[index], channel)) for index in usable}
    sharing = [index for index in usable if any(ids[index] & ids[other] for other in usable if other != index)]
    chosen = sharing[-1] if sharing else usable[-1]
    counts = {index: len(_channel_messages(roots[index], channel)) for index in usable}
    longest = max(usable, key=lambda index: counts[index])
    if not sharing and counts[chosen] < counts[longest]:
        raise ThreadRefused(
            _thread_finding_malformed(
                source,
                'langsmith',
                'root_selection',
                f'{source}: the latest root run {roots[chosen].get("id")!r} carries {counts[chosen]} messages, but root '
                f'{roots[longest].get("id")!r} carries {counts[longest]}, and no message id ties either to the other — so the '
                'latest state is not the fullest one and pinning it would silently truncate the conversation. This usually '
                'means the export mixes a side-graph (title or summary generation) into the thread; export the conversation '
                'graph on its own.',
            )
        )
    skipped_stateless = [str(roots[index].get('id')) for index in range(chosen + 1, len(roots)) if index not in counts]
    disjoint = [str(roots[index].get('id')) for index in usable if index != chosen and not ids[index] & ids[chosen]]
    evidence: dict[str, Any] | None = None
    if skipped_stateless or disjoint or counts[longest] > counts[chosen]:
        evidence = {
            'chosen': {'id': str(roots[chosen].get('id')), 'messages': counts[chosen]},
            'skipped_stateless': skipped_stateless,
            'disjoint_roots': disjoint,
            'longest': {'id': str(roots[longest].get('id')), 'messages': counts[longest]},
        }
    return RootSelection(run=roots[chosen], evidence=evidence)


# --------------------------------------------------------------------------- #
# Message projection                                                          #
# --------------------------------------------------------------------------- #


def _render_location(loc: tuple[int | str, ...]) -> str:
    """A pydantic error location as a path an operator can find in the JSON: ``messages[3].id``."""
    if not loc:
        return ''
    rendered = str(loc[0])
    for part in loc[1:]:
        rendered += f'[{part}]' if isinstance(part, int) else f'.{part}'
    return rendered


def _first_problem(error: ValidationError) -> str:
    """The model's first complaint, rendered for an operator rather than for a stack trace.

    The shared model defines the SHAPE; the words around a violation stay the SDK's. That split
    is the whole point of sharing a model rather than a module: the two repos fail in different
    places — here in front of an operator holding the export, there over one folder of an uploaded
    bundle — so a single shared error string would be wrong on one side of it.
    """
    first = error.errors()[0]
    location = _render_location(first['loc'])
    return f'{location}: {first["msg"]}' if location else str(first['msg'])


def _parse_tool_args(raw: Any, *, where: str) -> dict[str, Any]:
    """Tool-call ``args`` as a parsed dict, or fail closed.

    LangChain normally stores args already parsed, but a provider that streamed them can leave a
    JSON string behind — the schema says parsed dicts, so a string is parsed here rather than
    shipped for the customer's agent to choke on at seed time.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise TraceConversionError(f'{where}: tool_call args is a string that is not JSON ({e}); the pin schema requires parsed object args.') from e
        if not isinstance(parsed, dict):
            raise TraceConversionError(f'{where}: tool_call args parsed to {type(parsed).__name__}, not an object; the pin schema requires parsed object args.')
        return parsed
    raise TraceConversionError(f'{where}: tool_call args is {type(raw).__name__}, not an object; the pin schema requires parsed object args.')


def _canonical_tool_calls(message: dict[str, Any], *, where: str) -> list[ResumeToolCall]:
    """The message's ``tool_calls`` projected to exactly ``[{name, args, id}]``."""
    calls = []
    for position, call in enumerate(message.get('tool_calls') or []):
        at = f'{where} tool_call[{position}]'
        if not isinstance(call, dict):
            raise TraceConversionError(f'{at}: expected an object, got {type(call).__name__}.')
        name, call_id = call.get('name'), call.get('id')
        if not isinstance(name, str) or not name:
            raise TraceConversionError(f'{at}: missing a non-empty "name".')
        if not isinstance(call_id, str) or not call_id:
            # The id is what pairs this call with its ToolMessage; without it the tool exchange
            # cannot be reconstructed, and a synthesized id would pair with nothing.
            raise TraceConversionError(f'{at} ({name!r}): missing a non-empty "id" — a tool call with no id cannot be paired with its tool result.')
        calls.append(ResumeToolCall(name=name, args=_parse_tool_args(call.get('args'), where=at), id=call_id))
    return calls


def canonical_message(message: Any, *, where: str) -> ResumePinMessage:
    """One LangSmith-serialized message → its member of the shared pin union, or fail closed.

    Keeps ``type``/``content``/``id`` (+ ``tool_calls`` on ai, ``tool_call_id`` on tool) and drops
    everything else — ``additional_kwargs``, ``response_metadata``, and a tool message's
    ``name``/``status`` are deliberately NOT carried, which the models' ``extra='forbid'``
    enforces rather than merely documents. ``content`` is carried verbatim whatever its shape: a
    plain string or a list of content blocks both survive ``convert_to_messages`` unchanged, and
    the model annotates it ``Any`` so pydantic carries the value instead of rebuilding it.

    The checks below stay hand-written on purpose even though the model would refuse most of these
    inputs too: they name the trace position and say what an operator should do about it, which a
    raw ``ValidationError`` does not.
    """
    if not isinstance(message, dict):
        raise TraceConversionError(f'{where}: expected a message object, got {type(message).__name__}.')

    declared = message.get('type')
    if not isinstance(declared, str) or declared not in SUPPORTED_MESSAGE_TYPES:
        raise TraceConversionError(f'{where}: unsupported message type {declared!r}; expected one of {sorted(SUPPORTED_MESSAGE_TYPES)}.')
    kind = _TYPE_ALIASES.get(declared, declared)

    if 'content' not in message:
        raise TraceConversionError(f'{where} ({kind}): has no "content" key.')

    content = message['content']
    if not isinstance(content, (str, list)):
        # A BaseMessage's content is `str | list` — anything else never came from a serialized
        # message, and both of the ways it goes wrong are invisible here: `None` reconstructs as
        # '' (a pin that does not rebuild to itself), while a number or object raises a
        # ValidationError inside the customer's process at seed time.
        raise TraceConversionError(f"{where} ({kind}): content is {type(content).__name__}; a message's content must be a string or a list of content blocks.")
    if content == []:
        # `convert_to_messages` coerces an empty content list to '', so carrying [] would make the
        # pin differ from what re-seeding it produces. Both spell "no content"; '' is the one that
        # survives the round trip.
        content = ''

    message_id = message.get('id')
    if not isinstance(message_id, str) or not message_id:
        # Ids are REQUIRED because add_messages reconciles by id, which is what makes re-seeding
        # idempotent. Synthesizing one here would produce a pin that re-seeds as a DUPLICATE
        # conversation on the second update_state — silently.
        raise TraceConversionError(
            f'{where} ({kind}): missing a non-empty "id". Message ids are required — LangGraph reconciles by id, '
            'and a synthesized id would make re-seeding append a duplicate conversation instead of reconciling.'
        )

    if invalid := message.get('invalid_tool_calls'):
        # LangChain parked these because it could not parse them into tool calls. Dropping them
        # would silently lose a call the agent actually made, and carrying them is outside the
        # schema — so the export fails and a human decides.
        names = [c.get('name') for c in invalid if isinstance(c, dict)]
        raise TraceConversionError(
            f'{where} ({kind}): carries {len(invalid)} invalid_tool_calls {names} that LangChain could not parse. '
            'The pin schema has nowhere to put them and dropping them would lose a call the agent made.'
        )

    if kind != 'ai' and message.get('tool_calls'):
        raise TraceConversionError(f'{where} ({kind}): carries tool_calls, which only ai messages may have.')

    if kind == 'ai':
        # `[]` and an absent key both mean "made no calls", and `convert_to_messages` rebuilds
        # both as `[]` — so the pin gives that state ONE spelling, an explicit null. Two spellings
        # for one meaning is how two byte-identical conversations produce two different pins.
        return AIPinMessage(type='ai', content=content, id=message_id, tool_calls=_canonical_tool_calls(message, where=where) or None)
    if kind == 'tool':
        tool_call_id = message.get('tool_call_id')
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise TraceConversionError(f'{where} ({kind}): missing a non-empty "tool_call_id" — a tool result that names no call cannot be replayed.')
        return ToolPinMessage(type='tool', content=content, id=message_id, tool_call_id=tool_call_id)
    if kind == 'human':
        return HumanPinMessage(type='human', content=content, id=message_id)
    if kind == 'system':
        return SystemPinMessage(type='system', content=content, id=message_id)
    # Reachable only if the shared union grows a member: SUPPORTED_MESSAGE_TYPES is DERIVED from
    # it, so a new type is accepted on input the moment it is declared. Failing here — rather than
    # falling through to whichever class happens to be last — is what makes that a loud
    # one-line fix instead of silently mislabelled messages.
    raise TraceConversionError(f'{where}: message type {kind!r} is in the pin schema but this converter has no projection for it.')


def canonical_messages(messages: list[Any], *, source: str) -> list[ResumePinMessage]:
    """The thread's message list projected to the pin schema, in order."""
    if not messages:
        raise TraceConversionError(f'{source}: the final run state has an empty "messages" list — there is no conversation to resume.')
    return [canonical_message(m, where=f'{source} messages[{i}]') for i, m in enumerate(messages)]


# --------------------------------------------------------------------------- #
# Source normalisation (shared by every front end that reads chat messages)    #
# --------------------------------------------------------------------------- #

#: How a front end mints the id a source message was serialized without: keyed on the THREAD and on
#: the message's position in the SOURCE list, so a later drop, reorder or cut never shifts an id and
#: a re-converted pin re-seeds the same conversation instead of appending a copy of it.
SOURCE_INDEX_MINTED_ID_RECIPE = 'uuid5(minted_id_namespace, "<thread_id>:<source message index>")'


def mint_message_id(thread_id: str, source_index: int) -> str:
    """The deterministic id :data:`SOURCE_INDEX_MINTED_ID_RECIPE` gives the message at ``source_index``."""
    return str(uuid.uuid5(_minted_id_namespace(), f'{thread_id}:{source_index}'))


def _unnested_call(call: Any) -> Any:
    """An OpenAI-shaped call ``{id, function: {name, arguments}}`` as ``{name, args, id}``; any other shape as given."""
    if isinstance(call, Mapping) and 'name' not in call and isinstance(function := call.get('function'), Mapping):
        return {'name': function.get('name'), 'args': function.get('arguments'), 'id': call.get('id')}
    return call


def normalize_role_keyed_message(message: Any) -> tuple[Any, bool]:
    """``(a COPY of message in the type-keyed vocabulary, whether its content was normalised)``.

    Pure and lossless, applied BEFORE :func:`canonical_message` so every check there still runs:

    * a message keyed by ``role`` (OpenAI's spelling), or a LangChain ``chat`` message, takes its
      ``role`` as its ``type`` — the aliases among them fold as they always have, and a role the pin
      has no type for (``function``) is left for :func:`canonical_message` to refuse by name;
    * an OpenAI call ``{id, function: {name, arguments}}`` becomes ``{name, args, id}``; the
      arguments string is parsed by the shared projection, like every other format's;
    * an ai message's ``null`` content — and an absent one beside its tool calls, which is how a
      provider spells a reply that is only calls — becomes ``''``, which is what a LangChain message
      rebuilds it as anyway;
    * a tool result recorded as a JSON value rather than text becomes its JSON text, verbatim.

    Anything not a mapping is returned as given, for :func:`canonical_message` to refuse.
    """
    if not isinstance(message, Mapping):
        return message, False
    out = dict(message)
    declared = out.get('type')
    if (declared is None or declared == 'chat') and isinstance(role := out.get('role'), str):
        out['type'] = declared = role
    kind = _TYPE_ALIASES.get(declared, declared) if isinstance(declared, str) else None
    if isinstance(calls := out.get('tool_calls'), list):
        out['tool_calls'] = [_unnested_call(call) for call in calls]
    if kind == 'ai' and (('content' in out and out['content'] is None) or ('content' not in out and out.get('tool_calls'))):
        out['content'] = ''
        return out, True
    if kind == 'tool' and 'content' in out and not isinstance(out['content'], (str, list)):
        out['content'] = json.dumps(out['content'], ensure_ascii=False)
        return out, True
    return out, False


@dataclass(frozen=True)
class NormalizedMessages:
    """A source message list after :func:`normalize_source_messages`, with a count of each change."""

    messages: list[Any]
    minted: int
    normalized_content: int
    #: ``(source index, how many)`` for each message whose unparseable calls were dropped.
    dropped_invalid_tool_calls: tuple[tuple[int, int], ...]


def normalize_source_messages(messages: Sequence[Any], *, thread_id: str) -> NormalizedMessages:
    """``messages`` normalised for the shared projection — copies, never the export's own objects.

    Each message goes through :func:`normalize_role_keyed_message`. Then, over the whole list:

    * ``invalid_tool_calls`` — calls the source recorded but could not parse — are dropped from a
      message when NO tool message answers one of them: the pin has nowhere to put them, and an
      unanswered call carried anyway would be refused by a provider. When a tool message DOES answer
      one, dropping the call would orphan that result, so the thread is refused instead;
    * a message with no id — absent, ``null`` or blank — gets :func:`mint_message_id` for its source
      index. A present, non-string id is a malformed export, left for :func:`canonical_message` to refuse.
    """
    normalized: list[Any] = []
    changed = 0
    for message in messages:
        out, content_changed = normalize_role_keyed_message(message)
        normalized.append(out)
        changed += content_changed
    answered = {m['tool_call_id'] for m in normalized if isinstance(m, Mapping) and m.get('type') == 'tool' and isinstance(m.get('tool_call_id'), str)}
    dropped: list[tuple[int, int]] = []
    minted = 0
    for index, message in enumerate(normalized):
        if not isinstance(message, dict):
            continue
        invalid = message.get('invalid_tool_calls')
        if isinstance(invalid, list) and invalid:
            if paired := [c['id'] for c in invalid if isinstance(c, Mapping) and isinstance(c.get('id'), str) and c['id'] in answered]:
                raise TraceConversionError(
                    f'messages[{index}]: carries invalid_tool_calls {paired} that the source could not parse, and a tool message '
                    'answers them. The pin has nowhere to put an unparsed call, and dropping it would leave its result answering '
                    'nothing — cut before this message with --cut-turn.'
                )
            del message['invalid_tool_calls']
            dropped.append((index, len(invalid)))
        recorded = message.get('id')
        if recorded is None or (isinstance(recorded, str) and not recorded.strip()):
            message['id'] = mint_message_id(thread_id, index)
            minted += 1
    return NormalizedMessages(messages=normalized, minted=minted, normalized_content=changed, dropped_invalid_tool_calls=tuple(dropped))


def drop_leading_orphan_results(messages: list[ResumePinMessage]) -> tuple[list[ResumePinMessage], list[str]]:
    """``(messages without the tool results that open the history, the call ids those answered)``.

    After any leading system messages, a run of tool results that no earlier message asks for is
    what ``trim_messages``, summarisation or a ``RemoveMessage`` leaves behind when it cut a call
    away from its results. No cut can repair a prefix, and the results answer nothing the pin
    holds, so they are dropped and counted. A result orphaned anywhere LATER in the history is a
    different defect, inside the recorded conversation, and the seed-shape rule still refuses it.
    """
    start = 0
    while start < len(messages) and isinstance(messages[start], SystemPinMessage):
        start += 1
    end = start
    while end < len(messages) and isinstance(messages[end], ToolPinMessage):
        end += 1
    dropped = [message.tool_call_id for message in messages[start:end] if isinstance(message, ToolPinMessage)]
    return [*messages[:start], *messages[end:]], dropped


# --------------------------------------------------------------------------- #
# Mid-turn capture                                                             #
# --------------------------------------------------------------------------- #


def _message_shape(message: Any) -> tuple[str | None, bool]:
    """A message read as ``(type, carries tool calls)`` — from a pin model or a raw mapping alike.

    Both spellings reach the tail rule. The converter holds validated models; a pin read back off
    disk on the way to an upload is a mapping nothing has validated yet, and a hand-edited one is
    exactly the case that read exists for. One reader, so the rule an operator satisfies locally is
    the rule the upload applies.
    """
    if isinstance(message, Mapping):
        message_type = message.get('type')
        return (message_type if isinstance(message_type, str) else None), bool(message.get('tool_calls'))
    return getattr(message, 'type', None), bool(getattr(message, 'tool_calls', None))


#: The ``resume_seed_shape`` rule ids the backend publishes (its client contract), each mirrored
#: by :func:`mid_turn_tail_problem`, and the rule-set version they come from. A backend on a
#: higher version refuses seeds this converter still accepts.
SEED_SHAPE_RULES_MIRRORED: frozenset[str] = frozenset(
    {
        'ends_on_completed_agent_turn',
        'tool_tail_closes_last_call',
        'tool_result_follows_its_call',
        'tool_call_has_result',
        'tool_result_once',
        'open_call_ids_unique',
        'tool_results_adjacent',
        'seed_has_content',
        'message_ids_unique',
    }
)
SEED_SHAPE_RULES_VERSION = 2


def _history_and_lift(messages: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """``messages`` split into the SEEDED HISTORY and the trailing user turn lifted out of it.

    The split is the backend's, not this module's invention: a pin ending on a user message has
    that message taken off the history and used as the opening line of the resumed conversation,
    so the tail every rule below judges is the message BEFORE it.
    """
    lifted = list(messages[-1:]) if messages and _message_shape(messages[-1])[0] == 'human' else []
    return list(messages[: len(messages) - len(lifted)]), lifted


def _tool_exchange(message: Any) -> tuple[list[Any], Any]:
    """``(the call ids an ai message asks for, the call id a tool message answers)``.

    Read from a pin model or a raw mapping alike, for the reason :func:`_message_shape` gives.
    Unlike :func:`_exchange_ids` it keeps an id of any value: the tail rule compares ids, and a
    malformed one is still an id that fails to match.
    """
    if isinstance(message, Mapping):
        calls = message.get('tool_calls')
        requested = [call.get('id') if isinstance(call, Mapping) else None for call in calls] if isinstance(calls, list) else []
        return requested, message.get('tool_call_id')
    return [getattr(call, 'id', None) for call in getattr(message, 'tool_calls', None) or []], getattr(message, 'tool_call_id', None)


def _tool_tail_problem(seed: Sequence[Any]) -> tuple[int, str] | None:
    """``(index, why)`` a seed ending on tool results does not end on a completed agent turn, or ``None``.

    A MIRROR of the backend's rule ``tool_tail_closes_last_call``. The trailing
    run of tool messages closes the turn only when it answers the agent message right before it,
    and answers all of it once: every call that message asks for has exactly one result, and no
    result in the run answers anything else. The other half of the rule, that a user message
    follows the results, is the caller's.
    """
    start = len(seed)
    while start > 0 and _message_shape(seed[start - 1])[0] == 'tool':
        start -= 1
    if start == len(seed):
        return None
    caller_index = start - 1
    caller_type, caller_calls = _message_shape(seed[caller_index]) if caller_index >= 0 else (None, False)
    if caller_type != 'ai' or not caller_calls:
        return start, f'the seeded history ends on tool result(s) at messages[{start}:] with no agent tool call right before them, so they close no agent turn.'
    requested, _ = _tool_exchange(seed[caller_index])
    if asked_twice := sorted({call_id for call_id in requested if requested.count(call_id) > 1}, key=requested.index):
        return caller_index, (
            f'the seeded history ends on the results of messages[{caller_index}], which asks for tool call id(s) {asked_twice} '
            'more than once, so no run of results can answer each call exactly once.'
        )
    answered = [_tool_exchange(message)[1] for message in seed[start:]]
    if pending := [call_id for call_id in requested if call_id not in answered]:
        return caller_index, (
            f'the seeded history ends on the results of messages[{caller_index}], but its tool call(s) {pending} have no result — '
            'the trace appears to have been captured mid-turn, while the agent was still waiting on them.'
        )
    if stray := [call_id for call_id in answered if call_id not in requested]:
        return start, (
            f'the seeded history ends on the result of tool call(s) {stray}, which messages[{caller_index}] — the agent message '
            'right before the results — does not ask for, so the results close no single agent turn.'
        )
    if repeated := sorted({call_id for call_id in answered if answered.count(call_id) > 1}, key=answered.index):
        return start, (
            f'the seeded history ends on more than one result for tool call(s) {repeated} of messages[{caller_index}], '
            'and a call takes exactly one result — the trace recorded a result twice.'
        )
    return None


def _unpaired_tool_call(seed: Sequence[Any]) -> tuple[int, bool, str, str] | None:
    """``(index, in_flight, rule_id, why)`` for the first broken tool exchange in the seed, or ``None``.

    A MIRROR of the backend's unpaired-tool-call refusal, read over the WHOLE seed, and
    deliberately no stricter. It holds four rules: every result answers a call an earlier message
    made (``tool_result_follows_its_call``), every call has a result after it
    (``tool_call_has_result``), a call takes one result (``tool_result_once``), and no call reuses
    the id of an earlier call that is still open (``open_call_ids_unique``). A later message that
    reuses an answered id makes a new call, which takes one more result. A fifth,
    ``tool_results_adjacent``, is reported only when those four hold: no message but a tool result
    sits between a call and its result.

    ``in_flight`` is true for a call whose only followers are tool results: the capture stopped
    while the agent waited on it. Every other break lies inside the recorded conversation, where
    exporting again records the same exchange.
    """
    requested: dict[Any, int] = {}
    unanswered: dict[Any, int] = {}
    interrupted: tuple[int, bool, str, str] | None = None
    for index, message in enumerate(seed):
        calls, answered = _tool_exchange(message)
        if unanswered and interrupted is None and (message_type := _message_shape(message)[0]) != 'tool':
            call_id, at = next(iter(unanswered.items()))
            interrupted = (
                index,
                False,
                'tool_results_adjacent',
                f'messages[{index}] is a {message_type} message between tool call {call_id!r} (messages[{at}]) and its result — '
                "a provider accepts a call's results only directly after it; re-convert the pin from its source.",
            )
        if _message_shape(message)[0] == 'tool':
            if answered not in requested:
                return (
                    index,
                    False,
                    'tool_result_follows_its_call',
                    f'messages[{index}] carries the result of tool call {answered!r}, which no earlier agent message asks for — '
                    'the trace appears to have been captured mid-turn, from a point after the request that produced this result.',
                )
            if answered not in unanswered:
                return (
                    index,
                    False,
                    'tool_result_once',
                    f'messages[{index}] carries a second result for tool call {answered!r}, which messages[{requested[answered]}] '
                    'asks for once and an earlier message already answers — a call takes exactly one result.',
                )
            unanswered.pop(answered)
        for call_id in calls:
            if call_id in unanswered:
                return (
                    index,
                    False,
                    'open_call_ids_unique',
                    f'messages[{index}] asks for tool call {call_id!r} while the call messages[{unanswered[call_id]}] made under that id '
                    'has no result yet — two open calls cannot share an id, since a result could answer either.',
                )
            requested[call_id] = index
            unanswered[call_id] = index
    if unanswered:
        call_id, index = next(iter(unanswered.items()))
        in_flight = all(_message_shape(message)[0] == 'tool' for message in seed[index + 1 :])
        return (
            index,
            in_flight,
            'tool_call_has_result',
            f'messages[{index}] asks for tool call {call_id!r} and no later message carries its result — '
            'the trace appears to have been captured mid-turn, while that call was still in flight.',
        )
    return interrupted


def _results_before_user_turn(messages: list[ResumePinMessage]) -> tuple[list[ResumePinMessage], int]:
    """``(messages with each user message recorded inside an open tool call moved after its results, how many moved)``.

    A run that records each turn as ``[user message, *agent chain]`` writes an approval turn as
    ``ai(tool_use) → human → tool``: the agent paused on a call, the user answered, and the call's
    result was produced only after. No provider accepts that order — a call's results must be the
    very next thing it reads (rule ``tool_results_adjacent``). So a user message that sits while
    calls are open and is directly followed by results answering them moves to just after those
    results. Nothing is dropped and nothing else moves, so ids and the user-turn count are unchanged.

    Applied AFTER a cut, never before: a cut at the approval turn keeps the open call and lifts the
    user message, with its result past the cut, and the in-flight rule still refuses it.
    """
    moved = list(messages)
    moves = 0
    open_calls: set[Any] = set()
    index = 0
    while index < len(moved):
        message = moved[index]
        message_type = _message_shape(message)[0]
        if message_type == 'human' and open_calls:
            end = index + 1
            while end < len(moved) and _message_shape(moved[end])[0] == 'tool' and _tool_exchange(moved[end])[1] in open_calls:
                end += 1
            # Not when the results run to the END: an uncut pin that stops right after the approval's result
            # would become an accepted pin whose approval is re-said live as the opener (the answer
            # already applied). Left as recorded, the in-flight rule refuses it.
            if end > index + 1 and end < len(moved):
                moved[index:end] = [*moved[index + 1 : end], message]
                moves += 1
                continue
        calls, answered = _tool_exchange(message)
        if message_type == 'tool':
            open_calls.discard(answered)
        open_calls.update(calls)
        index += 1
    return moved, moves


def _flatten_content(content: Any) -> str:  # noqa: ANN401 - message content of any shape
    """A MIRROR of the backend's content flattening: text blocks joined, other blocks dropped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(text := block.get('text'), str):
                parts.append(text)
        return '\n'.join(part for part in parts if part)
    return '' if content is None else str(content)


def _message_field(message: Any, name: str) -> Any:  # noqa: ANN401 - a field of any shape
    return message.get(name) if isinstance(message, Mapping) else getattr(message, name, None)


@dataclass(frozen=True)
class SeedShapeProblem:
    """Why the backend refuses a seed: the published rule broken, where, and the sentence that says so."""

    #: One of :data:`SEED_SHAPE_RULES_MIRRORED`.
    rule_id: str
    message_index: int
    text: str


def _contentless_seed_problem(seed: Sequence[Any]) -> SeedShapeProblem | None:
    """A MIRROR of the backend's rule ``seed_has_content``.

    Whole-seed, never per message: an agent message carrying only tool calls has empty content.
    """
    if any(_flatten_content(_message_field(message, 'content')).strip() or _tool_exchange(message)[0] for message in seed):
        return None
    return SeedShapeProblem(
        'seed_has_content',
        0,
        'the seeded history carries no content — no message has text and none carries a tool call, so there is no conversation to resume from.',
    )


def _duplicate_message_id_problem(messages: Sequence[Any]) -> SeedShapeProblem | None:
    """A MIRROR of the pin model's id check (rule ``message_ids_unique``), for a pin read back as mappings."""
    seen: dict[Any, int] = {}
    for index, message in enumerate(messages):
        message_id = _message_field(message, 'id')
        if message_id in seen:
            return SeedShapeProblem(
                'message_ids_unique',
                index,
                f'messages[{index}].id {message_id!r} duplicates messages[{seen[message_id]}] — '
                'add_messages reconciles by id, so the two would collapse into one.',
            )
        seen[message_id] = index
    return None


def _unpaired_tool_call_problem(seed: Sequence[Any]) -> SeedShapeProblem | None:
    """Why the seed carries a broken tool exchange (:func:`_unpaired_tool_call`), or ``None``."""
    found = _unpaired_tool_call(seed)
    return SeedShapeProblem(found[2], found[0], found[3]) if found is not None else None


def _tool_tail_shape_problem(seed: Sequence[Any]) -> SeedShapeProblem | None:
    found = _tool_tail_problem(seed)
    return SeedShapeProblem('tool_tail_closes_last_call', found[0], found[1]) if found is not None else None


def seed_shape_problem(messages: Sequence[Any]) -> SeedShapeProblem | None:
    """Why the backend refuses this history as a resume seed, or ``None`` if it accepts it.

    A MIRROR of the seed normalization the backend applies, and deliberately no stricter: the trailing
    user turn is lifted out, and what remains must end on a completed agent turn. That is an agent
    message, or the tool results of the agent's last call when a user turn was lifted — many agents
    close a reply with a status or bookkeeping call, and the lifted turn is the proof the reply was
    over. :func:`_tool_tail_problem` holds the conditions on the results themselves. A seed whose
    tail passes is then read whole, as the backend reads it: the tool pairing rules
    (:func:`_unpaired_tool_call`) and then the content rule (:func:`_contentless_seed_problem`).
    Message ids are checked first, as the backend's pin model checks them before any of this.
    Every rule id the backend publishes in its ``resume_seed_shape`` contract is mirrored here;
    :data:`SEED_SHAPE_RULES_MIRRORED` names them.

    The shape this catches is not exotic — it is where a mid-thread export of any tool-using graph
    naturally lands. The run is captured between the tool answering and the agent reading the
    answer, so the history ends on a tool message with nothing after it, and the backend refuses it
    during generation: after the bundle has been uploaded and a job has started. Reading the same
    rule here moves that answer to the converter, on the operator machine, before any spend.

    Returns the CAUSE alone, with the rule id it breaks and where; each call site adds the remedy it
    is in a position to offer. :func:`mid_turn_tail_problem` is its sentence.
    """
    if problem := _duplicate_message_id_problem(messages):
        return problem
    seed, lifted = _history_and_lift(messages)
    tail_rule = 'ends_on_completed_agent_turn'
    if lifted and not _flatten_content(_message_field(lifted[0], 'content')).strip():
        return SeedShapeProblem(
            tail_rule,
            len(messages) - 1,
            'the trace ends on a user message with no text content, so there is no opener to lift from it — '
            'the trailing turn is an attachment or an empty message.',
        )
    if not seed:
        return SeedShapeProblem(
            tail_rule,
            0,
            'the pin holds no agent history — its only message is the trailing user turn, which the backend lifts out '
            'to open the resumed conversation, leaving nothing to resume from.',
        )
    tail = len(seed) - 1
    tail_type, _tool_calls = _message_shape(seed[-1])
    if tail_type is None:
        return SeedShapeProblem(
            tail_rule,
            tail,
            'the last message of the seeded history declares no "type", so nothing here can tell whether the history ends on a completed agent turn.',
        )
    if tail_type == 'tool':
        if not lifted:
            return SeedShapeProblem(
                tail_rule,
                tail,
                'the seeded history ends on a tool result and no user message follows it, so the trace was captured mid-turn, '
                'after the tool answered and before the agent read the result.',
            )
        return _tool_tail_shape_problem(seed) or _unpaired_tool_call_problem(seed) or _contentless_seed_problem(seed)
    if tail_type == 'human':
        return SeedShapeProblem(
            tail_rule,
            tail,
            'the seeded history must end on a completed agent turn, but the trace ends on two user messages in a row — only '
            'the last one can be lifted as the opener, and the one before it would reach the agent unanswered.',
        )
    if tail_type != 'ai':
        return SeedShapeProblem(
            tail_rule,
            tail,
            f'the seeded history must end on a completed agent turn, but ends on a {tail_type!r} message — the trace appears to have been captured mid-turn.',
        )
    return _unpaired_tool_call_problem(seed) or _contentless_seed_problem(seed)


def mid_turn_tail_problem(messages: Sequence[Any]) -> str | None:
    """Why the backend refuses this history as a resume seed, or ``None`` if it accepts it — :func:`seed_shape_problem`'s sentence."""
    problem = seed_shape_problem(messages)
    return problem.text if problem is not None else None


def _trim_to_completed_turn(messages: list[ResumePinMessage]) -> tuple[list[ResumePinMessage], list[ResumePinMessage]] | None:
    """``messages`` cut back to the last completed agent turn as ``(kept, dropped)``, or ``None``.

    ``None`` when no cut point leaves a history the backend accepts: there is no point to cut back
    to, and a trim would leave nothing to resume from.

    A cut point is judged by :func:`mid_turn_tail_problem`, the rule the write boundary applies,
    so a trimmed pin is never refused after the trim. A trailing user turn is KEPT. It is the
    opening line of the resumed conversation rather than part of the history being judged, and
    dropping it would discard the very thing the customer asked next. What goes is everything
    between the last completed agent turn and it — the half-issued tool calls and their replies,
    which is what makes the tail incomplete.

    A turn that closed on the results of its tool calls is a point to cut back to only where the
    RECORDING shows a user message right after those results. A trailing user turn, when there is
    one, does not count: it follows the cut in the pin, not in the conversation. When there is
    none, that recorded user message is kept as the opener, because a tool tail with no user
    message after it is a capture the backend refuses.
    """
    seed, lifted = _history_and_lift(messages)
    for index in range(len(seed) - 1, -1, -1):
        tail_type = _message_shape(seed[index])[0]
        if tail_type not in ('ai', 'tool'):
            continue
        kept = seed[: index + 1]
        if tail_type == 'tool':
            if index + 1 >= len(seed) or _message_shape(seed[index + 1])[0] != 'human':
                continue
            if not lifted:
                kept = seed[: index + 2]
        if mid_turn_tail_problem(kept + lifted) is None:
            return kept + lifted, seed[len(kept) :]
    return None


def _apply_trim(
    messages: list[ResumePinMessage], *, source: str, trim: bool, cut_turn: int | None = None, resume_seed_messages: int = 0
) -> tuple[list[ResumePinMessage], dict[str, Any] | None]:
    """The history to pin, plus the provenance record of a trim that ran.

    The trim is never implicit: without the flag the history is carried exactly as projected, and
    a mid-turn tail is REFUSED at the write boundary rather than quietly repaired here. With the
    flag the cut is recorded in the pin PROVENANCE, because a pin whose history no longer matches
    its source export has to carry that fact to everyone who reads it later, not only to the
    terminal of the run that wrote it.

    A trim never reaches into a resume seed (``resume_seed_messages``). The seed is carried whole
    by contract and its length is recorded in provenance, so a trim into it would leave a pin
    whose recorded seed length counts messages that are not the seed's.
    """
    if not trim or mid_turn_tail_problem(messages) is None:
        return messages, None
    trimmed = _trim_to_completed_turn(messages)
    if trimmed is None:
        raise TraceConversionError(
            f'{source}: --trim-mid-turn-tail has nothing to trim back to — no message in this history ends a completed '
            "agent turn the backend would seed from (an ai message carrying no tool calls, or tool results answering the "
            "agent's last call with a user message right after them, and every earlier tool call answered), so the trim "
            'would leave no conversation to resume from.'
        )
    kept, dropped = trimmed
    dropped_types = [_message_shape(message)[0] for message in dropped]
    if (into_seed := resume_seed_messages - (len(kept) - len(_history_and_lift(messages)[1]))) > 0:
        raise TraceConversionError(
            f'{source}: --trim-mid-turn-tail would drop {into_seed} message(s) of the resume seed this run started from '
            '— the last completed agent turn lies inside the seed, and the pin carries the seed whole. Convert without '
            '--trim-mid-turn-tail, or convert the pin the resumed scenario was built from.'
        )
    # Under a cut the two flags can contradict each other, and the contradiction is not visible in
    # the folder: `cut_turn` says the pin seeds turns 0..K-1, while a trim that reaches back past
    # one of those turns' human messages leaves a pin holding fewer. Refused rather than recorded,
    # and refused HERE, so nothing is written — a pin whose provenance overstates what it seeds is
    # read by every later stage as the conversation the customer asked to resume from.
    #
    # Only under a cut. Without one there is no claim to contradict: dropping the trailing user
    # turn along with the half-finished agent turn it opened is what the flag is FOR, and it is the
    # shape a mid-thread export of a tool-using graph naturally lands in.
    if cut_turn is not None and (eaten := dropped_types.count('human')):
        raise TraceConversionError(
            f'{source}: --cut-turn {cut_turn} seeds turns 0..{cut_turn - 1}, but --trim-mid-turn-tail would drop '
            f'{eaten} of those user turn(s) — no message between them and the cut is a completed agent turn, so the '
            'pin would seed less of the conversation than its own provenance claims. Cut at an earlier turn, one the '
            'conversation completed.'
        )
    return kept, {'dropped': len(dropped), 'dropped_types': dropped_types}


def mid_turn_refusal(conversion: ThreadConversion) -> str | None:
    """Why this conversion must not be WRITTEN into a bundle, or ``None`` if it may.

    Read at the write boundary rather than during projection, and the boundary is the point: a
    projection is a reading of someone else's bytes, while a bundle is the thing that gets uploaded
    and paid for. So a history the backend cannot seed from is refused exactly where it would
    otherwise be staged for upload.
    """
    problem = mid_turn_tail_problem(conversion.messages)
    if problem is None:
        return None
    seed, _ = _history_and_lift(conversion.messages)
    broken = _unpaired_tool_call(seed)
    if broken is not None and broken[3] == problem and not broken[1]:
        remedy = 'The broken exchange lies inside the recorded conversation, so a re-export records the same exchange; cut before it with --cut-turn'
    else:
        remedy = conversion.recapture_remedy
    trim = ', or convert with --trim-mid-turn-tail to drop the incomplete tail, which is then recorded in the pin.' if _trim_applies(conversion) else '.'
    return (
        f'{conversion.source.name}: {problem} The backend applies this same rule during generation — after the bundle '
        f'has been uploaded and a job has started — so the pin is refused here instead. {remedy}{trim}'
    )


def _trim_applies(conversion: ThreadConversion) -> bool:
    """Whether ``--trim-mid-turn-tail`` would produce a pin from this conversion, cut and seed included.

    Offered only when it would: a refusal naming a flag that then refuses in turn sends the
    operator round in a circle.
    """
    cut_turn = conversion.provenance.get('cut_turn')
    try:
        _apply_trim(
            list(conversion.messages),
            source=conversion.source.name,
            trim=True,
            cut_turn=cut_turn if isinstance(cut_turn, int) else None,
            resume_seed_messages=conversion.resume_seed_messages,
        )
    except TraceConversionError:
        return False
    return True


# --------------------------------------------------------------------------- #
# The cut                                                                      #
# --------------------------------------------------------------------------- #


def user_turn_positions(messages: Sequence[Any]) -> list[int]:
    """Where each USER TURN begins — the index of every ``human`` message, in order.

    Turn numbering is defined here and nowhere else: turn *k* is the k-th human message and
    everything up to the next one is that turn. The definition sits in the shared half because a
    cut has to mean the same thing whichever reader produced the history — a run transcript
    already knows its own turn boundaries, a span tree does not, and a cut whose meaning depended
    on that would be a different feature per format.
    """
    return [index for index, message in enumerate(messages) if _message_shape(message)[0] == 'human']


def _cuttable_turn_positions(messages: Sequence[Any], resume_seed_messages: int = 0) -> list[int]:
    """Where each turn ``--cut-turn`` can NAME begins: the user turns after any resume seed.

    Entry *k* is where a cut at *k* lands. The cut, the rendering's turn numbers and the summary's
    turn count all read this one list, so the number a caller reads off a rendering is the number
    the cut applies.
    """
    return [index for index in user_turn_positions(messages) if index >= resume_seed_messages]


def cut_at_turn(messages: Sequence[Any], cut_turn: int, *, source: str, resume_seed_messages: int = 0) -> list[Any]:
    """``messages`` cut so turn ``cut_turn`` is the FIRST live turn of the resumed conversation.

    The result keeps turns ``0..cut_turn - 1`` with their agent replies and ends on turn
    ``cut_turn``'s human message — which :func:`_history_and_lift` then lifts out as the scripted
    opener, exactly as it lifts a trailing user turn from an uncut history. So the seed is the
    conversation up to the moment being replayed, and the message that triggered it is said again
    live rather than pinned as something the agent has already answered.

    Both refusals are raised HERE, during projection, so a directory of exports settles every cut
    before the run writes its first file — a cut is an operator's number rather than a property of
    the artifact, and a wrong one should not leave half a bundle on disk. They name the turn count
    because that is the one fact the operator is missing: the number they typed is wrong only in
    relation to how many turns this conversation actually has.

    A cut at turn 0 is refused rather than emptied. It asks for a seed of nothing, and a
    conversation that opens on its first message is not a resumed scenario at all — it is an
    ordinary one, authored from the opener alone with no pin.

    ``resume_seed_messages`` is the one exception, and it changes where counting starts rather than
    what a turn is. The first that many messages are a resume seed the recording started from
    (:attr:`SourceProjection.resume_seed_messages`), so turns are counted after them and the seed is
    kept whole under every cut. Turn 0 is then the recording's own first turn, and a cut there keeps
    the seed as its history, which is something to resume from.
    """
    turns = _cuttable_turn_positions(messages, resume_seed_messages)
    if cut_turn < 1 and not resume_seed_messages:
        raise TraceConversionError(
            f'{source}: --cut-turn {cut_turn} would seed nothing — turn 0 is the first thing the user said, so cutting '
            f'there leaves no conversation to resume from. This conversation holds {len(turns)} user turn(s); cut at 1 '
            'or later, or author an opener-only scenario, which needs no pin.'
        )
    if not 0 <= cut_turn < len(turns):
        after_seed = f' after the {resume_seed_messages}-message resume seed it started from' if resume_seed_messages else ''
        raise TraceConversionError(
            f'{source}: --cut-turn {cut_turn} is {"past the last user turn" if cut_turn > 0 else "not a turn"} — this '
            f'conversation holds {len(turns)} user turn(s){after_seed}, numbered 0..{len(turns) - 1}, so there is no turn '
            f'{cut_turn} to lift as the opener.'
        )
    return list(messages[: turns[cut_turn] + 1])


def world_facts_after_cut(messages: Sequence[ResumePinMessage], cut_turn: int, *, resume_seed_messages: int = 0) -> WorldFacts:
    """The tool exchanges ``messages`` recorded AFTER a cut at ``cut_turn`` — pure, no LLM.

    Everything from turn ``cut_turn``'s user message on is what the live leg replays, so every
    tool call made there, with the result recorded for it, is what the world the live leg runs
    against has to be able to answer. The agent's prose and the user's later messages are not world
    facts and are left out.

    Calls pair with results by ``tool_call_id``, never by position — parallel calls are answered in
    any order — and only within the post-cut part. A call with no result stays an entry marked
    unanswered; a result with no call there is an entry of its own. A call id repeated by the
    source pairs first-come, first-answered.
    """
    turns = _cuttable_turn_positions(messages, resume_seed_messages)
    entries: list[WorldFact] = []
    waiting: dict[str, list[int]] = {}
    turn = cut_turn
    for message in messages[turns[cut_turn] + 1 :]:
        if isinstance(message, HumanPinMessage):
            turn += 1
        elif isinstance(message, AIPinMessage):
            for call in message.tool_calls or []:
                waiting.setdefault(call.id, []).append(len(entries))
                entries.append(WorldFact(turn=turn, call_id=call.id, name=call.name, args=call.args))
        elif isinstance(message, ToolPinMessage):
            if pending := waiting.get(message.tool_call_id):
                position = pending.pop(0)
                entries[position] = replace(entries[position], answered=True, result=message.content)
            else:
                entries.append(WorldFact(turn=turn, call_id=message.tool_call_id, answered=True, result=message.content))
    return WorldFacts(cut_turn=cut_turn, last_turn=len(turns) - 1, entries=tuple(entries))


# --------------------------------------------------------------------------- #
# Conversion                                                                   #
# --------------------------------------------------------------------------- #


def _folder_name(stem: str) -> str:
    """A bundle-folder name for a thread export stem (the tar member key's first segment)."""
    safe = ''.join(c if (c.isalnum() or c in '._-') else '-' for c in stem).strip('-') or 'thread'
    return f'thread-{safe}'[:96]


#: Byte-order marks that mean UTF-16. A UTF-8 file cannot open with either: both are invalid UTF-8.
_UTF16_BOMS = (b'\xff\xfe', b'\xfe\xff')
_GZIP_MAGIC = b'\x1f\x8b'


def read_source_text(source_path: Path) -> str:
    """A source artifact's text, or fail closed — the ONE read every front end shares.

    Decoding is format-independent, so it lives here rather than being restated per format: a gzip
    (known by its magic, whatever the name) is inflated first, under :mod:`plurai.readers`' cap; then
    UTF-8, with a UTF-8 byte-order mark read past (Windows tools write one, and ``json.loads`` refuses
    it), or UTF-16 when the text opens on a UTF-16 mark. Anything else that will not decode is refused.
    ``UnicodeDecodeError`` is a ``ValueError``, not an ``OSError``: without its own clause it escapes
    the fail-closed family and reaches the operator as a traceback. PARSING stays with the front
    end, because what a malformed file should have looked like is exactly the thing only the format
    knows.
    """
    try:
        raw = source_path.read_bytes()
    except OSError as e:
        raise TraceConversionError(f'{source_path.name}: could not be read ({e}).') from e
    if raw[:2] == _GZIP_MAGIC:
        gz_name = source_path.name if source_path.name.lower().endswith(readers.GZIP_SUFFIX) else f'{source_path.name}{readers.GZIP_SUFFIX}'
        try:
            _, raw = readers.decompress(gz_name, raw)
        except readers.UnreadableError as e:
            raise TraceConversionError(f'{source_path.name}: is gzip-compressed but could not be read ({e}).') from e
    encoding = 'utf-16' if raw[:2] in _UTF16_BOMS else 'utf-8-sig'
    try:
        return io.TextIOWrapper(io.BytesIO(raw), encoding=encoding).read()  # universal newlines, as read_text
    except UnicodeDecodeError as e:
        raise TraceConversionError(
            f'{source_path.name}: is not UTF-8 text ({e}). A trace artifact must be a UTF-8 text file (UTF-16 only with its byte-order mark).'
        ) from e


def _thread_stem(source_path: Path) -> str:
    """The thread a source file is named for: its stem once any ``.gz`` is set aside (``t1.json.gz`` -> ``t1``)."""
    return Path(readers.inner_name(source_path.name)).stem


def _langsmith_front_end(source_path: Path) -> SourceProjection:
    """The FIRST-CLASS front end: a LangSmith thread export → its source-shaped projection.

    The conversation is the message channel of a root run's output
    state, and every other channel is ``agent_state``. The reader adds NORMALISATION in front of the
    shared checks, never a looser check: which root is pinned (:func:`select_final_run`), which channel
    holds the conversation (:func:`resolve_messages_channel`), and the spellings
    :func:`normalize_source_messages` folds — each counted in provenance, and each choice another
    reading would have made differently reported as a finding.
    """
    source = source_path.name
    try:
        raw = json.loads(read_source_text(source_path))
    except json.JSONDecodeError as e:
        raise TraceConversionError(f'{source}: not parseable JSON ({e}). A thread export is a JSON list of LangSmith run dicts.') from e

    if isinstance(raw, dict) and 'observations' in raw:
        # `ingest-traces` writes Langfuse traces into the same `traces/<trace_id>.json` layout, so
        # this default reader meets them. Named, because "a dict is not a list" sends the operator
        # hunting for a malformed LangSmith export that does not exist.
        raise FormatUnsupportedError(
            f'{source}: is a Langfuse trace (a trace object carrying an "observations" tree), and no reader converts '
            'Langfuse yet. This reader takes a LangSmith thread export — a JSON list of run dicts.',
            convention='langfuse',
        )
    if not isinstance(raw, list):
        raise TraceConversionError(f'{source}: expected a JSON list of run dicts, got {type(raw).__name__}. This is the `traces/<trace_id>.json` export shape.')

    channel, inferred = resolve_messages_channel(_root_runs(raw, source=source), source=source)
    selection = select_final_run(raw, source=source, channel=channel)
    run = selection.run
    # The final `update_state` values, ferried opaquely: every channel the graph ended on EXCEPT the
    # conversation's, which is carried canonically in its own half so the client seeds one
    # reconciled history rather than two competing ones.
    state = dict(run['outputs'])
    normalized = normalize_source_messages(state.pop(channel), thread_id=_thread_stem(source_path))

    provenance: dict[str, Any] = {
        'root_run_id': run.get('id'),
        'trace_id': run.get('trace_id'),
        'langsmith_session_id': run.get('session_id'),
        'trace_start_time': run.get('start_time'),
        'trace_end_time': run.get('end_time'),
        'run_count': len(raw),
    }
    findings: list[GateFinding] = []
    # Each stamped only when it happened, so a pin read the plain way carries none of these keys.
    if inferred:
        provenance['messages_channel'] = channel
        findings.append(
            make_finding(
                TRACE_MESSAGES_CHANNEL_INFERRED,
                subject=source,
                evidence={'channel': channel},
                observed={},
                remedy=(
                    f'The conversation was read from the "{channel}" channel, the only one holding messages. The resumed '
                    f"agent is seeded from the pin's messages: its harness must write them into \"{channel}\", not into "
                    '"messages", or the seed does not restore.'
                ),
                draft_ref=None,
            )
        )
    if normalized.minted:
        provenance['minted_message_ids'] = normalized.minted
        provenance['minted_id_namespace'] = str(_minted_id_namespace())
        provenance['minted_id_recipe'] = SOURCE_INDEX_MINTED_ID_RECIPE
    if normalized.normalized_content:
        provenance['normalized_content'] = normalized.normalized_content
    if dropped := normalized.dropped_invalid_tool_calls:
        provenance['dropped_invalid_tool_calls'] = sum(count for _index, count in dropped)
        findings.append(
            make_finding(
                TRACE_INVALID_TOOL_CALLS_DROPPED,
                subject=source,
                evidence={'messages': [{'index': index, 'count': count} for index, count in dropped]},
                observed={},
                remedy=(
                    f'{provenance["dropped_invalid_tool_calls"]} tool call(s) the source could not parse were left out of the '
                    'pin; no tool message answers them, so the history around them is unchanged.'
                ),
                draft_ref=None,
            )
        )
    if selection.evidence is not None:
        evidence = selection.evidence
        findings.append(
            make_finding(
                TRACE_LANGSMITH_ROOT_SELECTION,
                subject=source,
                evidence=evidence,
                observed={},
                remedy=(
                    f'Pinned root run {evidence["chosen"]["id"]} ({evidence["chosen"]["messages"]} messages). Later roots with '
                    f'no state: {evidence["skipped_stateless"]}; roots sharing no message with it: {evidence["disjoint_roots"]}; '
                    f'longest state: root {evidence["longest"]["id"]} ({evidence["longest"]["messages"]} messages). If the '
                    'pinned state is not the conversation to resume, export that conversation on its own.'
                ),
                draft_ref=None,
            )
        )
    return SourceProjection(messages=normalized.messages, agent_state=state, provenance=provenance, findings=tuple(findings))


def _minted_id_namespace() -> uuid.UUID:
    from plurai.trace_adapter import MINTED_ID_NAMESPACE  # deferred: the adapter layer imports this module

    return MINTED_ID_NAMESPACE


def _openinference_front_end(source_path: Path) -> SourceProjection:
    """The span-tree reader, imported on first use: ``trace_span_tree`` imports this module."""
    from plurai.trace_span_tree import read_span_tree

    return read_span_tree(source_path)


#: The message keys the run reader carries. A turn's chain holds serialized ``BaseMessage`` dumps,
#: which also carry ``additional_kwargs``, ``response_metadata`` and ``usage_metadata``; those are
#: dropped HERE rather than left for :func:`canonical_message` to pass over, so the projection
#: hands on the pin's own vocabulary and a field a future LangChain dump adds cannot arrive inside
#: it unnoticed. ``invalid_tool_calls`` is deliberately not on the list: whether it is dropped
#: depends on what it holds, which :func:`_run_message` settles.
_RUN_MESSAGE_KEYS = ('type', 'content', 'id', 'tool_calls', 'tool_call_id')

#: How the run reader mints the ids a transcript does not record: every user message, since the
#: user side of a run records none, and any agent-side message a harness returned without one.
#: Keyed on the THREAD rather than on the file name every run transcript shares, so two scenarios
#: of one run converted into one bundle do not mint the same ids for different conversations; and
#: on the message's position in the pin, so a cut, which keeps a prefix, keeps its ids too.
RUN_MINTED_ID_RECIPE = 'uuid5(minted_id_namespace, "<thread_id>:<message index>")'


def _run_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """One agent-side dump pruned to :data:`_RUN_MESSAGE_KEYS`, plus a non-empty ``invalid_tool_calls``.

    Every ai dump carries ``invalid_tool_calls``, and in a healthy run it is ``[]`` — bookkeeping
    the pin has no use for, pruned with the rest. A non-empty one is a call the agent made that
    LangChain could not parse: the pin cannot carry it and must not lose it. That refusal belongs
    to :func:`canonical_message`, which applies it to every format, so the run reader hands the
    field on for that check to see rather than judging it here and growing a second copy of the
    rule that could drift from the first.
    """
    pruned = {key: message[key] for key in _RUN_MESSAGE_KEYS if key in message}
    if invalid := message.get('invalid_tool_calls'):
        pruned['invalid_tool_calls'] = invalid
    return pruned


def _run_turn_chain(turn: Mapping[str, Any], *, where: str) -> list[tuple[str, dict[str, Any]]]:
    """One turn's AGENT-side messages as ``(where, message)``, each projected by :func:`_run_message`.

    ``where`` names the message's place in the transcript (``turns[i].turn_messages[j]``), which a
    refusal of it further on has to be able to say.

    The writer puts the turn's agent side under ``turn_messages`` on every turn, ``[]`` included
    for a turn whose agent returned no chain. A turn without that list was therefore not written
    by this SDK, and reading it as a turn whose agent said nothing would drop every reply and
    tool call of it from the pin without a word.

    The chain is single-role by construction: the orchestrator's ``split_user_echo`` takes the
    user's own words out of it, and the turn records them once, under ``user_message``. A user
    message found inside it is refused rather than skipped. Carried, it may put the user's words
    in the pin twice; skipped, the reader would be guessing that it was only an echo, and a
    message the run never paired with a turn would vanish from the conversation.
    """
    chain = turn.get('turn_messages')
    if not isinstance(chain, list):
        found = 'is missing' if 'turn_messages' not in turn else f'is {type(chain).__name__}, not a list'
        raise TraceConversionError(
            f'{where}: "turn_messages" {found}, so the agent side of this turn cannot be read. Every turn this SDK '
            "writes carries that list, so this transcript was not written by it; convert a scenario folder's "
            'transcript.json from a run of this SDK.'
        )
    projected: list[tuple[str, dict[str, Any]]] = []
    for position, message in enumerate(chain):
        at = f'{where}.turn_messages[{position}]'
        if not isinstance(message, Mapping):
            raise TraceConversionError(f'{at}: expected a message object, got {type(message).__name__}.')
        declared = declared_message_type(message)
        if _TYPE_ALIASES.get(declared, declared) == 'human':
            raise TraceConversionError(
                f'{at}: is a {declared!r} message inside the agent side of the turn. This SDK records what the user '
                'said once, under "user_message", and keeps it out of "turn_messages", so this transcript was not '
                'written by it, and the message cannot be placed in the conversation without guessing.'
            )
        projected.append((at, _run_message(message)))
    return projected


def _object_args(args: Any) -> bool:
    """Whether a trail entry's ``args`` is one a pin tool call takes: absent, an object, or a JSON object string."""
    if isinstance(args, str):
        try:
            return isinstance(json.loads(args), dict)
        except ValueError:
            return False
    return args is None or isinstance(args, dict)


def _fold_tool_trail(
    chain: list[tuple[str, dict[str, Any]]], agent_state: Any, *, where: str, known_call_ids: set[str] | frozenset[str]
) -> list[tuple[str, dict[str, Any]]]:
    """``chain`` with the turn's own ``agent_state.tool_trail`` folded in as the tool exchange it records.

    A harness whose agent runs its own tool loop (the bridge, the Claude Agent SDK lane) returns one
    text reply per turn and reports its calls only on the trail, so a pin built from the chain alone
    seeds a history with no tool results and charges the resumed agent for ids it had seen. Folded only
    when the chain carries no tool exchange of its own: one ai call and its tool result per entry, in
    trail order, ahead of the turn's last ai message. Skipped: an entry run inside another call
    (``parent_call_id``), which the agent's model never issued, and one whose ``call_id`` the pin
    already carries or the resume seed's trail holds, which a cumulative trail or a re-reported seed
    trail repeats. A trail entry without a ``tool`` or ``call_id`` cannot be paired or told apart from
    a repeat, and one whose ``args`` is no object cannot be carried, so that turn is left as recorded.
    """
    trail = agent_state.get(AGENT_STATE_TOOL_TRAIL_KEY) if isinstance(agent_state, Mapping) else None
    kinds = [_TYPE_ALIASES.get(declared, declared) for declared in (declared_message_type(message) for _, message in chain)]
    if not isinstance(trail, list) or not trail or 'tool' in kinds or any(message.get('tool_calls') for _, message in chain):
        return chain
    entries = [(position, entry) for position, entry in enumerate(trail) if not (isinstance(entry, Mapping) and entry.get(TRAIL_PARENT_KEY))]
    if not all(
        isinstance(entry, Mapping) and all(isinstance(entry.get(key), str) and entry[key] for key in ('tool', 'call_id')) and _object_args(entry.get('args'))
        for _, entry in entries
    ):
        logger.warning(
            '%s: agent_state.tool_trail has an entry without a "tool", a "call_id" or object "args", so the pin carries none of its tool calls.', where
        )
        return chain
    known = set(known_call_ids)
    folded: list[tuple[str, dict[str, Any]]] = []
    for position, entry in entries:
        call_id, args, result = entry['call_id'], entry.get('args'), entry.get('result')
        if call_id in known:
            continue
        known.add(call_id)
        at = f'{where}.agent_state.tool_trail[{position}]'
        folded.append((at, {'type': 'ai', 'content': '', 'tool_calls': [{'name': entry['tool'], 'args': {} if args is None else args, 'id': call_id}]}))
        content = result if isinstance(result, str) else '' if result is None else json.dumps(result, ensure_ascii=False)
        folded.append((at, {'type': 'tool', 'content': content, 'tool_call_id': call_id}))
    last_ai = max((index for index, kind in enumerate(kinds) if kind == 'ai'), default=len(chain))
    return [*chain[:last_ai], *folded, *chain[last_ai:]]


def _exchange_ids(message: Any) -> tuple[list[str], str | None]:
    """``(tool-call ids an ai message issues, the call id a tool message answers)``.

    Read leniently — only well-formed string ids count — because the shape itself is judged later,
    by :func:`canonical_message`, and this reading only has to find ids that repeat.
    """
    if not isinstance(message, Mapping):
        return [], None
    declared = declared_message_type(message)
    kind = _TYPE_ALIASES.get(declared, declared)
    if kind == 'ai':
        calls = message.get('tool_calls')
        if not isinstance(calls, list):
            return [], None
        return [call['id'] for call in calls if isinstance(call, Mapping) and isinstance(call.get('id'), str) and call['id']], None
    answered = message.get('tool_call_id') if kind == 'tool' else None
    return [], answered if isinstance(answered, str) and answered else None


def _run_reduced_restore(raw: Mapping[str, Any], *, resumed: bool, source: str) -> None:
    """Refuse a RESUMED run that recorded a reduced restore of its resume seed.

    Read only for a resumed run, once :func:`_run_resume_seed` has settled that the seed and its
    flag agree. A fresh run carries ``resume_fidelity`` only when a harness's ``seed_state``
    returned a report with no seed to restore; there is nothing to restore short of, the live
    conversation is recorded whole, and the report is not read.

    On a resumed run ``resume_fidelity`` carries two kinds of report. Every key but
    :data:`SEED_RESTORE_KEY` is the harness's own account from ``seed_state``, which returns one
    only for a restore short of the whole seed. :data:`SEED_RESTORE_KEY` is the runner's reading
    of the restore from the agent's chains: ``partial`` when the agent re-listed part of the seed
    where it could not be stripped, and ``undetermined`` when the runner could not tell.
    ``undetermined`` alone is the ordinary reading for an agent that returns only its new
    messages, beside a harness that claimed nothing short of a full restore, so that run converts.
    Every other report means the run did not start from the seed it records — the harness
    restored only part of it, or the agent rewrote its seeded history — so its live turns answered
    a different history, and no pin built from it, cut or uncut, is the conversation the agent
    had. A report of a shape this reader does not know is refused for the same reason.
    """
    fidelity = raw.get('resume_fidelity')
    if not resumed or not fidelity:
        return
    # State keys the harness had no slot for are a skip, not a reduction: the conversation was
    # restored whole (``RESUME_STATE_DROPPED_KEY``), so it is read past like an undetermined reading.
    if isinstance(fidelity, Mapping):
        fidelity = {key: value for key, value in fidelity.items() if key != RESUME_STATE_DROPPED_KEY}
        if not fidelity:
            return
    if isinstance(fidelity, Mapping) and set(fidelity) == {SEED_RESTORE_KEY}:
        reading = fidelity[SEED_RESTORE_KEY]
        if isinstance(reading, Mapping) and reading.get('state') == 'undetermined':
            return
    raise TraceConversionError(
        f'{source}: records a reduced restore of its resume seed (resume_fidelity), so the run did not start from the '
        'seed it records: the harness restored only part of it, or the agent rewrote its seeded history. Its live turns '
        'answered a different history, and no pin built from it, cut or uncut, is the conversation the agent had. To '
        "convert this scenario, make the harness's seed_state replay the whole seed and the agent keep it, then re-run "
        'the scenario.'
    )


def _run_identity(metadata: Mapping[str, Any], key: str, *, source: str) -> str:
    """The name the run recorded for itself under ``metadata[key]``, or a refusal.

    Never derived from the path. The scenario names the bundle folder and the thread id keys every
    minted id, and a scenario folder can be moved or renamed after the run, so a name read off
    wherever the file sits would give one conversation a different folder and different ids.
    Every transcript this SDK writes records both.
    """
    declared = metadata.get(key)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    raise TraceConversionError(
        f'{source}: metadata.{key} is missing or blank. Every transcript this SDK writes records it, and the pin '
        "takes the conversation's name from the run rather than from where the file sits; convert a scenario "
        "folder's transcript.json from a run of this SDK."
    )


def _plurai_run_front_end(source_path: Path) -> SourceProjection:
    """A simulation run's per-scenario ``transcript.json`` → its source-shaped projection.

    Read through :func:`~plurai.run_artifacts.load_transcript` rather than by parsing the file
    here: that artifact carries six overlapping views of one conversation, and a second reading of
    it is how two consumers end up describing the same run differently.

    The projection walks ``turns``, which is the ONLY view that pairs what the user said with the
    reply and tool calls it produced. Each entry contributes its user message followed by that
    turn's agent-side chain, so the history is turn-ordered and every prefix of it ends where an
    agent turn ended — which is what a resume pin needs, and what the flat ``user_messages`` /
    ``chatbot_messages`` arrays cannot reconstruct without re-deriving the pairing by position.
    Each user message is emitted exactly once even though the record carries it twice: a turn's
    ``output_user_message`` is the NEXT turn's ``user_message``, and the closing turn's
    ``user_message`` is the last one already said, so only ``user_message`` is read.

    The user side of a transcript records no ids — nothing in a run needs them — and a harness may
    return the agent's messages without one, so every message a turn records with no id, or a
    blank one, gets an id minted by :data:`RUN_MINTED_ID_RECIPE`: deterministically, so a
    re-converted pin re-seeds the same conversation, and counted in provenance. A message that
    carries an id keeps it, since those are the ids the agent's graph reconciles by. Only the
    message id is minted; a tool message's ``tool_call_id`` pairs it with a call, and a minted one
    would pair with nothing. ``agent_state`` is carried per turn as well as at the end, so a cut
    pins the state the conversation had reached rather than the state it finished in. A turn whose
    chain carries no tool exchange gets the calls its own ``agent_state.tool_trail`` reports folded
    in ahead of its reply (:func:`_fold_tool_trail`), with ids minted the same way.

    A RESUMED scenario's ``turns`` hold only its live leg: the conversation it resumed is kept
    apart, verbatim, under the top-level ``resume_seed`` key. That seed is put back in FRONT of the
    live turns, because the pin is the conversation the agent had and the live leg alone opens on
    words that answered a history the pin would then leave out. Prepending is sound because of what
    the seed already is: the pin-shaped messages the backend lifted out of a pin, ending on a completed
    agent turn (the pin's trailing user message became the live leg's turn 0, so a seed that closes
    on the results of the agent's last tool calls is complete too), with the ids the agent's graph
    reconciled by. It passes through the same :func:`canonical_messages` as every other history, and
    nothing is minted into it: the seed is trusted as the history the agent had only because it
    arrives exactly as the backend sent it, so a seed message without an id is refused there like any
    other. The run excised the seed from the live turns' chains; a chain that re-lists it anyway is
    the re-listed history the next paragraph covers. See :func:`_run_resume_seed` for how the seed
    is read and refused, and :func:`_run_reduced_restore` for why a resumed run that did not start
    from its whole seed is refused whole.

    **Re-listed history is refused where it can be recognized.** A harness can return messages from
    earlier in the conversation again — a rewritten seed the excision found no prefix to strip, or
    a cumulative agent whose earlier turns the runner could not match. Re-listed WITH their ids,
    the pin schema's duplicate-id check refuses them. Re-listed WITHOUT ids, each would get a fresh
    minted id and pin as new, so a message whose id is minted is refused when it issues a tool call
    an earlier message already issued, or answers one an earlier tool message already answered: a
    call id names one exchange, and a second copy of it is the repeat. A minted message with no
    tool call carries nothing that names it, so a re-listed line of plain text cannot be told apart
    from the agent saying the same words again, and it converts. The check applies to minted
    messages only, and only here: other formats reach the shared projection with provider call
    ids that a conversation may legitimately reuse.

    **``--cut-turn K`` counts the run's own turns** — ``turns[K]``, the index its judges report —
    and never the seed's user messages, which no run artifact numbers. The seed rides along whole
    under every cut, so ``K = 0`` cuts at the seed boundary: the seed is the history, turn 0's
    words are the opener, and the pinned state is the seed's own. To cut INSIDE the seed, convert
    the pin the resumed scenario was built from; this artifact holds no per-turn state for it.

    **Shapes the runner never writes are refused rather than guessed at**: a turn with no
    ``turn_messages`` list or a user message inside one, a transcript that does not name its
    scenario and thread in ``metadata``, and a seed that disagrees with its flag. That is not the
    same as reading everything the runner writes. An agent-first run (turn 0's ``user_message`` is
    ``None``) and a turn whose user text is empty are both runner output, and both are refused here
    as a turn with no user message, because the pin pairs every reply with the words that prompted
    it.

    ``load_transcript`` never raises — a missing or foreign file loads as an empty transcript —
    which is right for an audit script and wrong here: this module fails closed, so the two shapes
    that mean "there is no conversation here" are turned back into refusals.
    """
    from plurai.trace_adapter import MINTED_ID_NAMESPACE  # deferred: the adapter layer imports this module

    source = source_path.name
    transcript = load_transcript(source_path)
    if transcript.schema == SCHEMA_NOT_LOADED:
        raise TraceConversionError(
            f'{source}: could not be read as a run transcript — it is missing, not UTF-8 JSON, or not a JSON object. '
            "Point --export at one scenario folder's transcript.json inside a finished run's output directory."
        )
    if not transcript.turns:
        raise TraceConversionError(
            f'{source}: records no turns, so there is no conversation to resume. A scenario that failed before its '
            'first turn writes error.json instead of a transcript; convert a scenario that ran.'
        )

    metadata = transcript.raw.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    scenario = _run_identity(metadata, 'scenario', source=source)
    thread_id = _run_identity(metadata, 'thread_id', source=source)

    seed_messages, seed_state = _run_resume_seed(transcript.raw, source=source)
    _run_reduced_restore(transcript.raw, resumed=seed_state is not None, source=source)
    messages: list[dict[str, Any]] = list(seed_messages)
    minted = 0
    # Every tool-call id issued and answered so far, seed included: how a minted message that
    # repeats an earlier exchange is recognized (see the docstring's re-listing paragraph).
    issued: set[str] = set()
    answered: set[str] = set()
    for seeded in seed_messages:
        calls, answer = _exchange_ids(seeded)
        issued.update(calls)
        if answer:
            answered.add(answer)
    # The resume seed's own trail calls: a harness that keeps its trail in state re-reports them on the live turns.
    seed_trail = (seed_state or {}).get(AGENT_STATE_TOOL_TRAIL_KEY)
    seed_trail_ids = (
        {entry['call_id'] for entry in seed_trail if isinstance(entry, Mapping) and isinstance(entry.get('call_id'), str)}
        if isinstance(seed_trail, list)
        else set()
    )
    agent_state_by_turn: list[dict[str, Any]] = []
    for index, turn in enumerate(transcript.turns):
        where = f'{source} turns[{index}]'
        user_message = turn.get('user_message')
        if not isinstance(user_message, str) or not user_message.strip():
            raise TraceConversionError(
                f'{where}: records no "user_message", so this turn\'s agent reply cannot be paired '
                'with what prompted it. The pin carries the conversation as it was said, and a turn missing its user '
                'side is not that conversation.'
            )
        chain = _fold_tool_trail(_run_turn_chain(turn, where=where), turn.get('agent_state'), where=where, known_call_ids=issued | seed_trail_ids)
        for at, message in ((where, {'type': 'human', 'content': user_message}), *chain):
            calls, answer = _exchange_ids(message)
            # Only an id the recording lacks (absent, null or blank) is filled. One that is present
            # but not a string is a malformed recording, left for `canonical_message` to refuse
            # rather than replaced here.
            recorded = message.get('id')
            if recorded is None or (isinstance(recorded, str) and not recorded.strip()):
                repeat = next((f'issues tool call {call!r}, which an earlier message already issued' for call in calls if call in issued), None)
                if repeat is None and answer in answered:
                    repeat = f'answers tool call {answer!r}, which an earlier tool message already answered'
                if repeat:
                    raise TraceConversionError(
                        f'{at}: carries no id and {repeat}. That is re-listed history: the harness returned messages from '
                        'earlier in the conversation again, without the ids that would let the pin schema see the repeat, '
                        'and a pin carrying them would replay that exchange as if the agent had made it twice. Make the '
                        "harness return only each turn's new messages, or return them with their ids, then re-run the "
                        'scenario.'
                    )
                message['id'] = str(uuid.uuid5(MINTED_ID_NAMESPACE, f'{thread_id}:{len(messages)}'))
                minted += 1
            issued.update(calls)
            if answer:
                answered.add(answer)
            messages.append(message)
        agent_state = turn.get('agent_state')
        agent_state_by_turn.append(agent_state if isinstance(agent_state, dict) else {})

    # A run writes one `transcript.json` per scenario folder, so the basename alone names every
    # scenario in every run alike. The folder above it is what tells them apart, and it is the path
    # segment a reader can carry back to the run directory. Overrides the shared default in place:
    # the key already exists on the envelope below, so the splat replaces its VALUE and the pin's
    # byte layout is the same either way. Blank only for a transcript converted from the directory
    # it sits in, where there is no parent segment to name and the basename is the whole path.
    folder = source_path.parent.name
    provenance: dict[str, Any] = {
        'source_file': f'{folder}/{source_path.name}' if folder else source_path.name,
        'thread_id': thread_id,
        'scenario': scenario,
    }
    # Only when the artifact recorded one: the writer records it only when the invocation resolved
    # one. The run handle is what joins this pin back to the run it came from, and a value derived
    # from whichever directory the file happens to sit in would answer that question with a guess.
    run_name = metadata.get('run_name')
    if isinstance(run_name, str) and run_name.strip():
        provenance['run_name'] = run_name.strip()
    provenance.update(
        {
            'transcript_schema': transcript.schema,
            'user_turns': len(agent_state_by_turn),
            'minted_message_ids': minted,
            'minted_id_namespace': str(MINTED_ID_NAMESPACE),
            'minted_id_recipe': RUN_MINTED_ID_RECIPE,
        }
    )
    # Only for a resumed run, so a fresh run's pin carries no such key. It is what tells a reader
    # of the pin that `user_turns` and `cut_turn` count from after this many messages, not from the
    # pin's first one — and, at 0, that the run resumed from a seed of state alone.
    if seed_state is not None:
        provenance['resume_seed_messages'] = len(seed_messages)
    return SourceProjection(
        messages=messages,
        agent_state=agent_state_by_turn[-1],
        provenance=provenance,
        agent_state_by_turn=agent_state_by_turn,
        resume_seed_messages=len(seed_messages),
        resume_seed_state=seed_state,
        folder_stem=scenario,
    )


def _run_resume_seed(raw: Mapping[str, Any], *, source: str) -> tuple[list[Any], dict[str, Any] | None]:
    """A run transcript's resume seed as ``(messages, agent_state)``; ``([], None)`` for a fresh run.

    Two keys say whether a run was resumed, and they must agree: a non-empty top-level
    ``resume_seed``, the seed itself, is present exactly when ``injected_inputs.resume_seed_present``
    is ``true``. An absent flag, or an absent ``injected_inputs`` block, reads as ``false``. The
    writer sets both from one value, so a transcript where they disagree was not written by this
    SDK, and the file cannot say which half is wrong. Either reading of it may be wrong: as a
    fresh run, the pin would be the live leg alone with nothing saying so; as a resumed one, it
    would put a history in front of the live leg that the run may never have started from. So
    every mismatch is refused.

    A seed of state alone is a legitimate shape (``[]`` messages), and so is a seed with no
    ``agent_state`` key (``{}``) — both are what the run started from. A value of the wrong TYPE is
    not: the pin would carry a history or a state the agent never had.
    """
    seed = raw.get('resume_seed')
    injected = raw.get('injected_inputs')
    flagged = isinstance(injected, dict) and injected.get('resume_seed_present') is True
    if not seed:
        if flagged:
            raise TraceConversionError(
                f'{source}: records that it was resumed (injected_inputs.resume_seed_present) but carries no resume_seed, '
                'so the conversation it resumed is not in this file and a pin built from the live turns alone would drop it. '
                'Convert a transcript that carries its seed.'
            )
        return [], None
    if not flagged:
        raise TraceConversionError(
            f'{source}: carries a resume_seed but does not record that it was resumed (injected_inputs.resume_seed_present '
            'is not true). This SDK writes the two together, so this transcript was not written by it, and whether the '
            "run started from that seed cannot be told from the file. Convert a scenario folder's transcript.json from a "
            'run of this SDK.'
        )
    if not isinstance(seed, dict):
        raise TraceConversionError(f'{source}: resume_seed is {type(seed).__name__}, not an object, so the conversation this run resumed cannot be read.')
    messages = seed.get('messages', [])
    if not isinstance(messages, list):
        raise TraceConversionError(
            f'{source}: resume_seed.messages is {type(messages).__name__}, not a list, so the conversation this run resumed cannot be read.'
        )
    state = seed.get('agent_state', {})
    if not isinstance(state, dict):
        raise TraceConversionError(
            f'{source}: resume_seed.agent_state is {type(state).__name__}, not an object, so the state this run resumed from cannot be read.'
        )
    return messages, state


#: Maps the customer-facing ``--source-format`` → the front end that reads it, mirroring
#: ``trace_extract.TRACE_SOURCE_REGISTRY`` (the sibling module's registry of sink READERS; this one
#: registers artifact PARSERS). A first-class reader is a standing liability until a real export
#: of its format exists to pin, so the registry absorbs RECURRENCE (a format earns an entry once
#: such an export exists) while novelty is absorbed by an authored adapter, which needs no SDK
#: release at all. Span trees are the one shape an adapter cannot express — its grammar is a record list —
#: which is why that reader is built in rather than authored.
TRACE_FORMAT_REGISTRY: dict[str, _FormatSpec] = {
    'langsmith': _FormatSpec(
        front_end=_langsmith_front_end,
        name='langsmith',
        description="LangSmith thread export: a JSON list of run dicts, pinned from the last root run's output state.",
    ),
    'openinference': _FormatSpec(
        front_end=_openinference_front_end,
        name='openinference',
        description=(
            'OpenInference span tree (a TRAIL/Patronus `{trace_id, spans}` export with nested `child_spans`): '
            'the thread reconstructed from the LLM and TOOL spans in start-instant order.'
        ),
    ),
    'plurai-run': _FormatSpec(
        front_end=_plurai_run_front_end,
        name='plurai-run',
        description=(
            "A simulation run's per-scenario `transcript.json`: the conversation the simulated user and the agent "
            'actually had, turn by turn, pinned from the state the last turn ended on.'
        ),
        origin='simulation run',
        recapture_remedy='A finished run holds the whole conversation it will ever hold, so cut before that turn with --cut-turn',
        world_facts=False,
    ),
}

#: What ``convert-trace`` reads when nothing says otherwise. Pinned to the first-class reader, so
#: an invocation naming no ``--source-format`` converts a LangSmith export — which is what every
#: caller of the bare verb means by it.
DEFAULT_SOURCE_FORMAT = 'langsmith'


def resolve_source_format(source_format: str, *, workspace_root: str | Path = '.') -> _FormatSpec:
    """The front end for ``source_format``: a registry entry, else a workspace-persisted adapter.

    Lookup order is deliberate. A built-in wins, so an adapter cannot shadow (or silently
    substitute itself for) a verified built-in reader. Everything else is an ADAPTER name,
    resolved from the workspace — which is what makes a novel format an authoring act in the
    customer's repo rather than an SDK release.

    A name that is neither fails closed and names BOTH vocabularies, because at the moment of the
    error the operator cannot tell which half they got wrong: a typo'd built-in and an unauthored
    adapter look identical from the command line.
    """
    if spec := TRACE_FORMAT_REGISTRY.get(source_format):
        return spec
    from plurai.trace_adapter import load_adapter  # deferred: the adapter layer imports this module

    return load_adapter(source_format, workspace_root=workspace_root)


def convert_thread_export(
    path: str | Path,
    *,
    source_format: str = DEFAULT_SOURCE_FORMAT,
    workspace_root: str | Path = '.',
    start_node: str | None = None,
    trim_mid_turn_tail: bool = False,
    cut_turn: int | None = None,
) -> ThreadConversion:
    """One trace artifact → its :class:`ThreadConversion`, resolving ``source_format`` itself.

    The one-artifact entry point. A DIRECTORY run goes through :func:`convert_trace_export`, which
    resolves the front end once and hands it to :func:`_convert_one` per file — resolving an
    adapter re-reads and re-validates its YAML, so doing it here per artifact would parse the same
    document once per file and let a mid-run edit change what later threads in the same bundle mean.

    A thread that cannot be converted raises :class:`ThreadRefused` carrying its finding; an
    operator-level defect (a blank ``--start-node``, an unknown format) raises the plain error first.
    """
    resume_point = _resume_point(start_node)
    return _convert_one(
        Path(path),
        resolve_source_format(source_format, workspace_root=workspace_root),
        resume_point=resume_point,
        trim_mid_turn_tail=trim_mid_turn_tail,
        cut_turn=cut_turn,
    )


def _resume_point(start_node: str | None) -> ResumePoint | None:
    """``--start-node`` as the pin's shape, or ``None`` — refusing a blank one in CLI grade.

    The value comes from an operator's command line, so its one failure mode is a name that is
    empty or all whitespace, and the flag is what they can act on. Left to the envelope
    validation below, the same input reports a field path (``resume_point.node``) that names
    neither the flag they typed nor what to type instead.
    """
    if start_node is None:
        return None
    try:
        return ResumePoint(node=start_node)
    except ValidationError as exc:
        raise TraceConversionError(f'--start-node {start_node!r} is blank; name the top-level graph node the resumed leg starts at, or omit the flag.') from exc


def _state_entering_turn(projection: SourceProjection, turn: int) -> dict[str, Any] | None:
    """The state the agent held ENTERING ``turn``, or ``None`` when the source did not record it."""
    if turn == 0:
        return projection.resume_seed_state
    if turn - 1 < len(projection.agent_state_by_turn):
        return projection.agent_state_by_turn[turn - 1]
    return None


def _convert_one(
    source_path: Path, spec: _FormatSpec, *, resume_point: ResumePoint | None, trim_mid_turn_tail: bool, cut_turn: int | None
) -> ThreadConversion:
    """One trace artifact → its :class:`ThreadConversion` (pure, no I/O beyond the read).

    The FORMAT half is ``spec``'s front end; everything from here down is shared by every
    format, which is the whole point of the seam: the canonical projection, the id-reconciliation
    rules and the envelope are properties of the PIN, not of where it was read from, so a new
    format cannot acquire its own dialect of them.

    Every refusal here is THIS thread's: each step turns what it refuses into a :class:`ThreadRefused`
    under the gate that step is (:func:`_thread_step`). The operator's own flags were settled by the
    caller before any artifact was read, so nothing raised here is about the invocation.
    """
    source = source_path.name
    with _thread_step(lambda remedy: _thread_finding_malformed(source, spec.name, 'read', remedy)):
        try:
            projection = spec.front_end(source_path)
        except FormatUnsupportedError as exc:
            raise ThreadRefused(
                make_finding(
                    TRACE_FORMAT_UNSUPPORTED,
                    subject=source,
                    evidence={'source_format': spec.name, 'convention': exc.convention},
                    observed={},
                    remedy=str(exc),
                    draft_ref=None,
                )
            ) from exc
    findings = list(projection.findings)

    # The history is settled BEFORE the envelope is validated: a cut and a trim both shorten the
    # messages, and the record of either is provenance — a key on the mapping validated just below.
    #
    # Canonical first, then cut: the whole conversation is validated even when only its opening
    # turns are kept, so an artifact this converter cannot represent is refused the same way
    # whatever cut is asked of it, rather than passing at one number and failing at the next.
    #
    # Cut, then trim, in that order. The cut chooses WHICH turns the seed covers; the trim is
    # about the shape of whatever the seed ends on, so it has to see the cut history — a cut that
    # lands after an agent's tool call has an incomplete tail of its own, and trimming the full
    # history first would trim a tail the seed no longer contains.
    with _thread_step(lambda remedy: _thread_finding_unrepresentable(source, spec.name, 'message', remedy)):
        canonical = canonical_messages(projection.messages, source=source)
    # Before the cut, so the turns it numbers are the ones the operator reads; never into a resume
    # seed, which the backend already accepted whole and which the run reader carries whole.
    orphans: list[str] = []
    if not projection.resume_seed_messages:
        canonical, orphans = drop_leading_orphan_results(canonical)
    if orphans:
        findings.append(
            make_finding(
                TRACE_ORPHAN_TOOL_RESULTS_DROPPED,
                subject=source,
                evidence={'tool_call_ids': orphans},
                observed={},
                remedy=(
                    f'{len(orphans)} tool result(s) opening the history answer no call the history holds (a trimmed or '
                    'summarised history leaves these behind) and were left out of the pin.'
                ),
                draft_ref=None,
            )
        )
    source_user_turns = len(_cuttable_turn_positions(canonical, projection.resume_seed_messages))
    agent_state = projection.agent_state
    agent_state_at: str | None = None
    state_withheld: list[str] = []
    world_facts: WorldFacts | None = None
    if cut_turn is not None:
        with _thread_step(lambda remedy: _thread_finding_cut(source, cut_turn, source_user_turns, remedy)):
            if cut_turn == 0 and projection.resume_seed_state is not None and not projection.resume_seed_messages:
                # Refused in the resumed run's own terms: the generic turn-0 refusal offers an
                # opener-only scenario, which drops the seed's state — the loss a resumed run's cut exists
                # to prevent.
                raise TraceConversionError(
                    f'{source}: --cut-turn 0 cuts at the resume seed this run started from, and that seed holds agent state '
                    'but no messages — there is no history for a pin to resume from, and an opener-only scenario would drop '
                    'the state. Cut at 1 or later.'
                )
            recorded = canonical
            canonical = cut_at_turn(recorded, cut_turn, source=source, resume_seed_messages=projection.resume_seed_messages)
        # Read off the WHOLE recorded conversation, which the cut has just replaced: the part after
        # the cut is exactly what the pin leaves out. After the cut, so its refusals come first.
        if spec.world_facts:
            world_facts = world_facts_after_cut(recorded, cut_turn, resume_seed_messages=projection.resume_seed_messages)
        # The state as of the LAST SEEDED turn, from a format that records one per turn. Turn
        # `cut_turn` is the one being replayed live, so the state the agent resumes holding is the
        # one it ended turn `cut_turn - 1` on — or, at turn 0 of a recording that was itself
        # resumed, the seed's state, which is what it held entering its first turn. A format with
        # only a final state reached it AFTER the cut — the replayed turn and every turn past it
        # wrote it — so that state is never seeded: the pin carries the conversation alone, and
        # the channels it left out are named in provenance.
        at_cut = _state_entering_turn(projection, cut_turn)
        if at_cut is not None:
            agent_state = at_cut
            agent_state_at = 'cut'
        else:
            state_withheld = sorted(agent_state)
            agent_state = {}
            agent_state_at = 'withheld'
    # After the cut, which judges turns in recorded order (see `_results_before_user_turn`).
    canonical, reordered = _results_before_user_turn(canonical)
    # A trim is refused only for a history the seed rule already refuses: the finding names that rule.
    with _thread_step(lambda remedy: _thread_finding_seed_shape(source, seed_shape_problem(canonical), remedy)):
        messages, trim_record = _apply_trim(
            canonical, source=source, trim=trim_mid_turn_tail, cut_turn=cut_turn, resume_seed_messages=projection.resume_seed_messages
        )

    # Validated from a mapping rather than called with keywords: everything past `thread_id` is an
    # EXTRA on the pin's one open shape, so these are keys, not fields — which is also how
    # `ThreadConversion.provenance` hands them back. They dump after the declared fields in the
    # order written here, so the envelope's byte layout stays stable while remaining extensible.
    # Inside the guard, not above it: `source_file` and `thread_id` are `NonBlankStr`, and a front
    # end may supply `thread_id` from document content, so a trace value of the wrong type — or a
    # blank one — reaches this validation. Outside it, that escapes as a raw `ValidationError`,
    # which is a traceback out of the CLI rather than a refusal, and contradicts `resume_pin`'s own
    # promise that the module never surfaces one.
    try:
        provenance = ResumePinProvenance.model_validate(
            {
                'source_file': source,
                # The ARTIFACT's thread identity: the file is named for it, and for a LangSmith
                # export it matches the root run id. Deliberately NOT read from the state's own
                # `thread_id` channel — that is the customer graph's internal id, a DIFFERENT uuid
                # that is often absent, and it stays where it belongs, inside agent_state. A front
                # end may override it by returning its own `thread_id` key; `**` replaces the VALUE
                # while leaving the key in position, so the envelope's byte layout is either way.
                'thread_id': _thread_stem(source_path),
                **projection.provenance,
                # Both present ONLY when they happened, so a pin converted the plain way carries
                # neither and each key means what it says wherever it appears. The cut is recorded
                # because the pin no longer holds the whole recorded
                # conversation, and nothing else on disk would say where it was stopped.
                **({'cut_turn': cut_turn} if cut_turn is not None else {}),
                # Only alongside a cut, where the two answers diverge. An uncut pin carries the
                # state the conversation ended on by construction, and a key restating that on
                # every pin would change the bytes of every folder to say what the absence of
                # `cut_turn` already says.
                **({'agent_state_at': agent_state_at} if agent_state_at is not None else {}),
                **({'state_withheld': state_withheld} if state_withheld else {}),
                **({'trimmed_mid_turn_tail': trim_record} if trim_record is not None else {}),
                **({'dropped_orphan_tool_results': len(orphans)} if orphans else {}),
                **({'reordered_user_turns': reordered} if reordered else {}),
                # Only when it is not the default, on the same terms: a pin that names no origin
                # came from a trace export, and that is what `pinned_folders` reads absence as.
                **({'origin': spec.origin} if spec.origin != _DEFAULT_ORIGIN else {}),
                'converted_by': f'plurai {__version__}',
            }
        )
    except ValidationError as exc:
        raise ThreadRefused(
            _thread_finding_unrepresentable(
                source, spec.name, 'pin_envelope', f'{source}: the projected provenance is not a valid pin envelope — {_first_problem(exc)}.'
            )
        ) from exc
    try:
        pin = ResumePinDocument(
            schema_version=RESUME_PIN_SCHEMA_VERSION,
            messages=messages,
            agent_state=agent_state,
            resume_point=resume_point,
            provenance=provenance,
        )
    except ValidationError as exc:
        # The per-message checks above cover everything a message can get wrong on its own; what
        # is left here are properties of the WHOLE history — duplicate ids above all, which
        # `add_messages` would silently collapse into one turn at seed time.
        raise ThreadRefused(
            _thread_finding_unrepresentable(
                source, spec.name, 'pin_envelope', f'{source}: the projected pin does not satisfy the resume-pin schema — {_first_problem(exc)}.'
            )
        ) from exc
    return ThreadConversion(
        folder=_folder_name(projection.folder_stem or _thread_stem(source_path)),
        pin=pin,
        source=source_path,
        source_format=spec.name,
        origin=spec.origin,
        recapture_remedy=spec.recapture_remedy,
        source_user_turns=source_user_turns,
        resume_seed_messages=projection.resume_seed_messages,
        world_facts=world_facts,
        findings=tuple(findings),
    )


# --------------------------------------------------------------------------- #
# Per-thread refusals                                                          #
# --------------------------------------------------------------------------- #


@contextmanager
def _thread_step(to_finding: Callable[[str], GateFinding]) -> Iterator[None]:
    """Run one step of a thread's conversion; what it refuses becomes that thread's :class:`ThreadRefused`.

    ``to_finding`` builds the finding from the refusal's text, under the gate the step is: reading
    the artifact, projecting a message, cutting, trimming. A refusal that already carries its own
    finding passes through unchanged.
    """
    try:
        yield
    except ThreadRefused:
        raise
    except TraceConversionError as exc:
        from plurai.trace_adapter import AdapterError, AdapterThreadRefused  # deferred: the adapter layer imports this module

        # An adapter's refusal stays an AdapterError: "the export does not match the mapping" keeps its own owner.
        refused = AdapterThreadRefused if isinstance(exc, AdapterError) else ThreadRefused
        raise refused(to_finding(str(exc))) from exc


def _thread_finding_malformed(source: str, source_format: str, step: str, remedy: str) -> GateFinding:
    return make_finding(
        TRACE_EXPORT_MALFORMED, subject=source, evidence={'source_format': source_format, 'step': step}, observed={}, remedy=remedy, draft_ref=None
    )


def _thread_finding_unrepresentable(source: str, source_format: str, step: str, remedy: str) -> GateFinding:
    return make_finding(
        TRACE_MESSAGE_UNREPRESENTABLE, subject=source, evidence={'source_format': source_format, 'step': step}, observed={}, remedy=remedy, draft_ref=None
    )


def _thread_finding_cut(source: str, cut_turn: int, user_turns: int, remedy: str) -> GateFinding:
    return make_finding(
        TRACE_CUT_OUT_OF_RANGE, subject=source, evidence={'cut_turn': cut_turn, 'user_turns': user_turns}, observed={}, remedy=remedy, draft_ref=None
    )


def _thread_finding_seed_shape(source: str, problem: SeedShapeProblem | None, remedy: str) -> GateFinding:
    """The seed-shape finding; ``problem`` is ``None`` only if a caller refused a seed the rules accept, which is a bug here."""
    if problem is None:
        raise AssertionError(f'{source}: refused under the seed-shape gate, but the seed breaks no seed-shape rule: {remedy}')
    return make_finding(
        TRACE_SEED_SHAPE_REFUSED,
        subject=source,
        evidence={'rule_id': problem.rule_id, 'rules_version': SEED_SHAPE_RULES_VERSION, 'message_index': problem.message_index},
        observed={},
        remedy=remedy,
        draft_ref=None,
    )


def _thread_finding_folder(source: str, folder: str, kind: str, remedy: str) -> GateFinding:
    return make_finding(TRACE_FOLDER_CONFLICT, subject=source, evidence={'folder': folder, 'kind': kind}, observed={}, remedy=remedy, draft_ref=None)


@dataclass(frozen=True)
class SkippedThread:
    """A thread the invocation did not write, and the finding that says why."""

    source: Path
    finding: GateFinding


@dataclass(frozen=True)
class ConversionResult:
    """What one directory run did: the threads it wrote (or rendered), and every thread it skipped.

    ``findings`` is every finding the run made, in the order it made them — each skipped thread's
    refusal, and the advisory findings of every thread it converted, written or not.
    """

    conversions: tuple[ThreadConversion, ...]
    skipped: tuple[SkippedThread, ...]
    findings: tuple[GateFinding, ...]


def _skip(skipped: list[SkippedThread], findings: list[GateFinding], source: Path, finding: GateFinding) -> None:
    skipped.append(SkippedThread(source=source, finding=finding))
    findings.append(finding)


# --------------------------------------------------------------------------- #
# Folder emission                                                              #
# --------------------------------------------------------------------------- #


def _render_content(content: Any) -> str:
    """A message's content as display text (a content-block list is flattened for READING only)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get('text', '') if isinstance(block, dict) else str(block) for block in content]
        return ''.join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


#: Provenance keys every format records, in the order the envelope declares them. Everything else
#: is a format's own EXTRA and is rendered generically — see :func:`_render_provenance_extras`.
_UNIVERSAL_PROVENANCE_KEYS = ('source_file', 'thread_id', 'converted_by')


def _render_provenance_extras(provenance: dict[str, Any]) -> list[str]:
    """Every format-contributed provenance key as its own bullet, in recorded order.

    The renderers below must not name LangSmith's extras — ``run_count``, ``trace_id``, the trace
    window — in their prose. That reads well and is wrong twice over: it puts format knowledge
    BELOW the seam that exists to keep it out, and it fails with a bare ``KeyError`` (not a
    :class:`TraceConversionError`) the moment a conversion comes from anywhere else, so a working
    adapter surfaces as a traceback from the render step after the pin has already been built.

    Keys are shown as the pin spells them rather than prettified into prose, so a line here is
    greppable in ``resume_pin.json``; a recorded ``null`` shows as ``—`` rather than vanishing,
    because "this format has this slot and this artifact left it empty" is worth reading.

    The KEY is escaped as well as the value. ``RESERVED_PROVENANCE_KEYS`` constrains which names a
    document may declare but says nothing about their SHAPE, so an unescaped key carrying a newline
    forges the same block structure a value would — including a second ``## Provenance`` naming a
    reader that never ran, which is the exact claim reserving those names exists to prevent. Escaping at the
    render site rather than validating at the load site is what makes that a property of every
    rendered key, not of the ones a validator anticipated.
    """
    return [f'* `{_provenance_display(key)}`: {_provenance_display(value)}' for key, value in provenance.items() if key not in _UNIVERSAL_PROVENANCE_KEYS]


def _provenance_display(value: Any) -> str:
    """One provenance value as a single, structurally inert line of markdown.

    A provenance value is trace-derived, and ``ROLES.md`` is the one bundle file the store inlines
    into the generation agent's context. A newline inside a value would forge headings and bullets
    there — a trace could write ``## Provenance`` into the agent's own instructions. So every
    non-printable character is escaped to its source spelling: the value stays legible, and cannot
    become structure. Spaces are printable and survive; only the characters that end a line do not.

    EVERY rendered provenance field goes through this, not just the format-contributed extras.
    Escaping only the extras leaves ``thread_id`` — interpolated into both rendered files —
    carrying trace bytes straight into ``ROLES.md``, and ``thread_id`` is deliberately the one key
    an adapter MAY declare (it identifies the artifact, not the reader). The single key exempted
    from reservation is then the one that reaches the agent unescaped: a hole opened by two
    defensible decisions that are only safe apart. Escaping is applied where the text becomes
    markdown, because that is the only place that cannot be reached around.
    """
    if value is None:
        return '—'
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ''.join(char if char.isprintable() else repr(char)[1:-1] for char in text)


def _writable(text: str) -> str:
    """``text`` with every character UTF-8 cannot encode written as its backslash escape.

    A JSON source can carry an unpaired UTF-16 surrogate (``"\\ud83d"``, usually half of an emoji a
    producer cut in two), which decodes to a string no UTF-8 file can hold. The rendered files are
    for reading, so the character is shown the way the source spelled it, deterministically, rather
    than failing the write.
    """
    return text.encode('utf-8', 'backslashreplace').decode('utf-8')


def _fenced(text: str, info: str) -> str:
    """``text`` in a code fence one backtick longer than the longest backtick run inside it.

    Message and tool-result text is whatever the traced system said or returned, and both rendered
    files are read by an agent: ``ROLES.md`` is inlined into the generation agent's context, and
    ``transcript.md`` is what a caller reads to choose a cut. A fence only closes on a line of at
    least as many backticks as opened it, so sizing it past every run in the content means no text
    can close its block — and so none can forge a heading, a numbered turn or a second world-facts
    section.
    """
    longest = max((len(run) for run in re.findall(r'`+', text)), default=0)
    fence = '`' * max(3, longest + 1)
    return f'{fence}{info}\n{text}\n{fence}'


def _turn_count_line(conversion: ThreadConversion, turns: int) -> str:
    """The header line that says how many turns are numbered, and whether ``--cut-turn`` agrees.

    The claim holds for every pin but one: a trim that dropped a user message out of an uncut
    history. Its trailing user turn is kept and lands on a lower number than the source gave it, so
    the header says the numbers are this history's own rather than the ones a cut counts.
    """
    seeded = conversion.resume_seed_messages
    after_seed = f' after a {seeded}-message resume seed' if seeded else ''
    trim = conversion.provenance.get('trimmed_mid_turn_tail')
    if isinstance(trim, Mapping) and 'human' in (trim.get('dropped_types') or []):
        return (
            f'{turns} user turn(s){after_seed}, numbered from 0 over this trimmed history. --trim-mid-turn-tail dropped a '
            'user message, so these are not the numbers `--cut-turn` counts on the source.'
        )
    return f'{turns} user turn(s){after_seed}, numbered from 0 as `--cut-turn` counts them.'


def render_transcript(conversion: ThreadConversion, *, pinned: bool = True) -> str:
    """The conversation as deterministic markdown — authoring conditioning, never the source of truth.

    The pin is what the backend byte-preserves; this rendering exists because the generation agent
    authors the persona, the scripted opener and the policies by READING the conversation, and the
    pin's id-bearing JSON is a poor read. Nothing parses this file.

    Every user message a cut can name carries its turn number — ``--cut-turn`` numbering, from
    :func:`_cuttable_turn_positions` — because this is also what a caller reads to CHOOSE a cut, and
    a count taken by eye over the headings is off by one on any history that opens on a system
    prompt or a resume seed. A resume seed's own user messages are marked as seed and left
    unnumbered: a cut never counts them. Every message body is fenced (:func:`_fenced`), so text the
    traced system produced cannot write a heading of its own among the numbered ones.

    ``pinned`` is False for a rendering written with no pin beside it (:func:`render_trace_export`),
    whose header points at the pin a conversion would write instead of one in the same folder.
    """
    provenance = conversion.provenance
    turn_at = {position: turn for turn, position in enumerate(_cuttable_turn_positions(conversion.messages, conversion.resume_seed_messages))}
    authority = (
        'Rendered verbatim from the trace for authoring context. The authoritative copy is '
        f'`{RESUME_PIN_FILENAME}`; this file is prose for reading, and differs from it '
        '(content blocks are flattened, ids are omitted).'
        if pinned
        else 'Rendered verbatim from the trace to choose a `--cut-turn`. No pin is written beside a rendering: the '
        f'authoritative copy is the `{RESUME_PIN_FILENAME}` that a conversion of this export writes, and this file '
        'differs from it (content blocks are flattened, ids are omitted).'
    )
    lines = [
        f'# Trace transcript — thread `{_provenance_display(provenance["thread_id"])}`',
        '',
        f'Source export: `{_provenance_display(provenance["source_file"])}` · {len(conversion.messages)} messages',
        '',
        _turn_count_line(conversion, len(turn_at)),
        '',
        authority,
        '',
    ]
    for index, message in enumerate(conversion.messages):
        # The discriminated union does the narrowing, so the per-type fields are reached only on
        # the members that HAVE them — no `.get` on a shape that may or may not carry the key.
        if isinstance(message, ToolPinMessage):
            lines.append(f'### tool result (for call `{_provenance_display(message.tool_call_id)}`)')
        elif index in turn_at:
            lines.append(f'### human · turn {turn_at[index]}')
        elif isinstance(message, HumanPinMessage):
            lines.append('### human · resume seed')
        else:
            lines.append(f'### {message.type}')
        if text := _render_content(message.content).strip():
            lines.extend(['', _fenced(text, 'text')])
        if isinstance(message, AIPinMessage):
            for call in message.tool_calls or []:
                arguments = _provenance_display(json.dumps(call.args, ensure_ascii=False))
                lines.extend(['', f'*calls* `{_provenance_display(call.name)}`({arguments})'])
        lines.append('')
    return _writable('\n'.join(lines).rstrip() + '\n')


def _turn_span(first: int, last: int) -> str:
    """``turn 3`` or ``turns 3–7`` — a turn range as the world-facts prose names it."""
    return f'turn {first}' if first == last else f'turns {first}–{last}'


def _world_fact_value(label: str, value: Any, *, verbatim: bool) -> str:
    """One labelled, fenced value, capped at :data:`WORLD_FACT_VALUE_MAX_BYTES`.

    A string is shown ``verbatim``, since that is how the tool returned it; anything else is shown
    as indented JSON, which is also how a list of content blocks reads. The cut falls on a
    character boundary, and the note beside it says how much of the value is missing.
    """
    text = _writable(value if verbatim else json.dumps(value, indent=2, ensure_ascii=False))
    encoded = text.encode('utf-8')
    shown = encoded[:WORLD_FACT_VALUE_MAX_BYTES].decode('utf-8', errors='ignore')
    block = f'{label}:\n\n{_fenced(shown, "text" if verbatim else "json")}'
    if len(encoded) <= WORLD_FACT_VALUE_MAX_BYTES:
        return block
    return f'{block}\n\n*Truncated: the first {len(shown.encode("utf-8"))} of {len(encoded)} bytes are shown.*'


def _world_fact_entry(fact: WorldFact, *, origin: str) -> tuple[str, str]:
    """One exchange as ``(title, body)`` — the title is the heading text, unmarked."""
    call = f'call `{_provenance_display(fact.call_id)}`'
    parts: list[str] = []
    if fact.name is None:
        title = f'Turn {fact.turn} · result for {call}'
        parts.append('*No call with this id was recorded after the cut.*')
    else:
        title = f'Turn {fact.turn} · `{_provenance_display(fact.name)}` · {call}'
        parts.append(_world_fact_value('Arguments', fact.args, verbatim=False))
    if fact.answered:
        parts.append(_world_fact_value('Result', fact.result, verbatim=isinstance(fact.result, str)))
    else:
        parts.append(f'*Unanswered: the {origin} recorded no result for this call.*')
    return title, '\n\n'.join(parts)


def _world_facts_intro(facts: WorldFacts, *, origin: str) -> str:
    """What the section is, said to the one reader it is for: the world author."""
    return f"""Every tool call the agent made from turn {facts.cut_turn} — the turn this scenario replays live — to the end of
the recorded conversation, with the result the {origin} recorded for it, copied verbatim.
These are constraints for the world author: the world the live leg runs against should be able
to return what these results show, so the resumed conversation meets the situation it met when
it was recorded. They are not a frozen world and not a script — the live agent may call other
tools, or these in another order. The agent's own replies after the cut, and the user's later
messages, are left out."""


#: How many tool names the omitted-entries note spells out before summing the rest, and how many
#: characters of each: the note is charged to the section cap, so its length has to be bounded
#: whatever the trace called its tools.
_OMITTED_NOTE_TOOLS = 10
_OMITTED_NOTE_NAME_CHARS = 48

#: Bytes held back from :data:`WORLD_FACTS_MAX_BYTES` for the omitted-entries note once entries have
#: to be left out. The note's own bounds keep it under this.
_OMITTED_NOTE_RESERVE = 4 * 1024


@dataclass(frozen=True)
class _WorldFactsLayout:
    """Which entries fit under :data:`WORLD_FACTS_MAX_BYTES`, rendered, and which did not."""

    shown: tuple[tuple[str, str], ...]
    omitted: tuple[WorldFact, ...]


def _world_facts_head(facts: WorldFacts, *, origin: str, heading: str) -> str:
    """The section's heading, intro and — when nothing was recorded after the cut — the sentence saying so."""
    blocks = [f'{heading} {WORLD_FACTS_HEADING}', _world_facts_intro(facts, origin=origin)]
    if not facts.entries:
        blocks.append(f'The {origin} recorded no tool calls after the cut ({_turn_span(facts.cut_turn, facts.last_turn)}).')
    return '\n\n'.join(blocks)


def _layout_world_facts(facts: WorldFacts, *, origin: str) -> _WorldFactsLayout:
    """The longest leading run of entries that keeps the WHOLE section within its cap.

    Every byte is charged — heading, intro, separators, heading markers and the omitted-entries
    note — at the deeper of the two depths the section is written at, so one layout fits a
    converted folder and an ``--into`` merge alike. When everything fits, nothing is held back;
    otherwise the note's reserve comes off the budget first. A leading run rather than any subset
    that fits: what is left out is then one contiguous range of turns, which the note can name,
    and nothing later is shown past a gap.
    """
    budget = WORLD_FACTS_MAX_BYTES - len((_world_facts_head(facts, origin=origin, heading='###') + '\n').encode('utf-8'))
    entries = [_world_fact_entry(fact, origin=origin) for fact in facts.entries]
    costs = [len(f'\n\n#### {title}\n\n{body}'.encode()) for title, body in entries]
    if sum(costs) > budget:
        budget -= _OMITTED_NOTE_RESERVE
    shown: list[tuple[str, str]] = []
    for position, (entry, cost) in enumerate(zip(entries, costs, strict=True)):
        if cost > budget:
            return _WorldFactsLayout(tuple(shown), facts.entries[position:])
        budget -= cost
        shown.append(entry)
    return _WorldFactsLayout(tuple(shown), ())


def _omitted_note(omitted: Sequence[WorldFact]) -> str:
    """What the section left out: how many exchanges, from which turns, and of which tools.

    The most frequent tools first, at most :data:`_OMITTED_NOTE_TOOLS` of them and each name cut at
    :data:`_OMITTED_NOTE_NAME_CHARS`, so the note stays inside the reserve the layout held for it.
    """

    def label(fact: WorldFact) -> str:
        if fact.name is None:
            return 'result with no call'
        name = _provenance_display(fact.name)
        return f'`{name if len(name) <= _OMITTED_NOTE_NAME_CHARS else name[: _OMITTED_NOTE_NAME_CHARS - 1] + "…"}`'

    ranked = sorted(Counter(label(fact) for fact in omitted).items(), key=lambda item: (-item[1], item[0]))
    named = [f'{name} ×{count}' for name, count in ranked[:_OMITTED_NOTE_TOOLS]]
    if rest := ranked[_OMITTED_NOTE_TOOLS:]:
        named.append(f'and {len(rest)} other tool(s) ×{sum(count for _name, count in rest)}')
    return (
        f'*Omitted: {len(omitted)} more tool call(s), from {_turn_span(omitted[0].turn, omitted[-1].turn)}, past the '
        f'{WORLD_FACTS_MAX_BYTES}-byte cap on this section: {", ".join(named)}.*'
    )


def _render_world_facts(facts: WorldFacts, *, origin: str, heading: str) -> str:
    """The world-facts section, rooted at ``heading``, its entries one level beneath it."""
    layout = _layout_world_facts(facts, origin=origin)
    blocks = [_world_facts_head(facts, origin=origin, heading=heading)]
    blocks.extend(f'{heading}# {title}\n\n{body}' for title, body in layout.shown)
    if layout.omitted:
        blocks.append(_omitted_note(layout.omitted))
    return '\n\n'.join(blocks) + '\n'


#: What the ROLES.md pin bullet calls the state block, keyed by the pin's own `agent_state_at`.
#: Absent means an uncut conversion, where the state the agent ended on IS the state the resumed
#: leg starts from; ``'withheld'`` is a cut whose source records one state for the whole
#: conversation — reached after the cut, so none is seeded. ``'end'`` marks a pin whose state is the
#: one the agent reached AFTER the cut.
_ROLES_STATE_PHRASES = {
    None: "the agent's final state values",
    'cut': "the agent's state values as of the cut",
    'withheld': 'no state values — this source records one state for the whole conversation, reached after the cut, '
    'so the pin seeds the conversation alone',
    'end': "the agent's state values from the END of the recorded conversation, past the cut — this source "
    'records one state for the whole conversation rather than one per turn',
}


def _render_roles_body(conversion: ThreadConversion, *, heading: str) -> str:
    """The Files + Provenance half of a folder's ``ROLES.md``, rooted at ``heading``.

    Two files carry this half at two depths: a converted folder's whole description IS it, and a
    folder holding hand-authored material as well carries it as a section BENEATH that material's
    own description. One rendering serves both, so the resume half a generation agent reads cannot
    depend on which route wrote the folder.

    The world facts, when the conversion carries them, close it — at the same depth, and so last in
    the file on both routes, since a caller copies them from their heading to the end of the file.
    """
    provenance = conversion.provenance
    final_type = conversion.messages[-1].type
    extras = '\n'.join(_render_provenance_extras(provenance))
    # Named exactly, because under a cut the two are different states and the pin cannot show
    # which it holds. A generation agent that reads "final" over values the agent only reached
    # after the cut authors an opener addressing an arc the resumed leg has not lived yet.
    # Adapters are refused `agent_state_at` at load and no built-in front end returns it, so the
    # value is one the converter stamped, and this table has a sentence for each.
    state_phrase = _ROLES_STATE_PHRASES[conversion.provenance.get('agent_state_at')]
    # Escaped like a provenance value even though an operator typed it: ROLES.md is the one bundle
    # file inlined into the generation agent's context, so anything that can end a line here can
    # forge a heading in the agent's own instructions.
    start_node_line = (
        f'\n  The pin also names a **start node**, `{_provenance_display(conversion.resume_point.node)}` — the graph node the'
        '\n  harness enters on turn 1 of the resumed leg.'
        if conversion.resume_point is not None
        else ''
    )
    body = f"""{heading} Files

* `{RESUME_PIN_FILENAME}` — the **resume pin**: the verbatim message history
  ({len(conversion.messages)} messages, the last of type `{final_type}`) and {state_phrase}, plus
  provenance and a schema version. It is authoritative and machine-read — its content is carried
  into the generated scenario byte-preserved. Do not paraphrase, summarize or regenerate any of
  it.{start_node_line}
* `{TRANSCRIPT_FILENAME}` — the same conversation rendered as prose, for reading while authoring
  the persona, the opening message and the policies. Not authoritative.

{heading} Provenance

* Source export: `{_provenance_display(provenance["source_file"])}`
* Thread id: `{_provenance_display(provenance["thread_id"])}`
* Converted by: {_provenance_display(provenance["converted_by"])}
{extras}
"""
    if conversion.world_facts is None:
        return body
    return body.rstrip('\n') + '\n\n' + _render_world_facts(conversion.world_facts, origin=conversion.origin, heading=heading)


def _roles_resume_heading(conversion: ThreadConversion) -> str:
    """The appended section's heading line — the marker that says this thread is already there.

    Spelled once because it is read as well as written: the append is skipped when this exact
    line is already a line of the file, which is what makes repeating the same ``--into`` a no-op
    rather than a second copy of the section. It names the thread, so a folder carrying one
    resume section does not swallow a second, different one.
    """
    return f'## Resumed from {conversion.origin} `{_provenance_display(conversion.provenance["thread_id"])}`'


def _render_roles_resume_section(conversion: ThreadConversion) -> str:
    """The resume half of a ``ROLES.md``, as a SECTION to append beneath a hand-authored one.

    Rooted an H-level deeper than :func:`render_roles` so the hand-authored description keeps the
    document's single H1 and stays the first thing read. The opening paragraph is the one piece
    that cannot be shared: it has to say the folder is BOTH kinds, because a generation agent that
    read only the resume half would author a fresh conversation about the materials, and one that
    read only the materials half would ignore the conversation it must continue.
    """
    return f"""{_roles_resume_heading(conversion)}

This folder is ALSO a **resumed** scenario: beside the materials described above, it carries a
real conversation that already happened, recorded in a {conversion.origin}, and the scenario
generated from it must CONTINUE that conversation rather than open a new one — grounded in those
same materials.

{_render_roles_body(conversion, heading='###')}"""


def render_roles(conversion: ThreadConversion) -> str:
    """The folder's required ``ROLES.md`` — the only file the bundle store inlines for the agent."""
    return f"""# Resumed scenario from {conversion.origin} `{_provenance_display(conversion.provenance['thread_id'])}`

This folder is a **resumed** scenario: it did not start from nothing. It carries a real
conversation that already happened, recorded in a {conversion.origin}, and the scenario generated
from it must CONTINUE that conversation rather than open a new one.

{_render_roles_body(conversion, heading='##')}"""


def _origins_phrase(conversions: Sequence[ThreadConversion]) -> str:
    """What the folders in this overview were converted from, as a plural noun phrase.

    A bundle dir accumulates one export per invocation and nothing stops those invocations
    naming different formats, so the set can hold folders of several origins at once and the
    overview has to name them all: a set described as production traces alone tells the
    generation agent to weight simulated dialogue as real customer behaviour.

    Empty only for an overview of no converted folders at all, which is described in the reader's
    default terms rather than with a blank.
    """
    plurals = [f'{origin}s' for origin in sorted({c.origin for c in conversions}) or [_DEFAULT_ORIGIN]]
    if len(plurals) == 1:
        return plurals[0]
    return f'{", ".join(plurals[:-1])} and {plurals[-1]}'


def render_bundle_overview(
    conversions: list[ThreadConversion],
    plain: Mapping[str, str] = MappingProxyType({}),
    grounded: Mapping[str, str] = MappingProxyType({}),
) -> str:
    """The set-level ``BUNDLE.md`` — what the whole bundle dir is, for the generation agent.

    Optional to the bundle store (and uploaded inline, not as a unit), but worth writing: without
    it the agent meets N folders of resumed scenarios with no statement of what they have in
    common or that the pins are authoritative.

    ``plain`` maps each pinless folder's NAME to its one-line description (:func:`plain_folders`).
    A set holding both kinds is described as a set holding both kinds: the title, the census and
    the section headings all turn on whether ``plain`` is empty, because an overview that opened "Resumed scenarios
    converted from production traces" over a table of hand-authored folders would misdescribe half
    the set to the one reader it exists for. A set with no pinless folders renders flat — the
    sectioning appears only when there is a second kind to separate.

    ``grounded`` (:func:`grounded_folders`) names the folders that are BOTH — a resume pin beside
    hand-authored material. They are a subsection of the resumed table rather than a third
    section, because they are already counted and tabled there: a folder that carries a pin is a
    resumed folder, and grounding says what else is in it. Naming them is what stops the agent
    reading the resumed table as "these folders hold a conversation and nothing else", which is
    true of every other row.
    """
    mixed = bool(plain)
    origins = _origins_phrase(conversions)
    lines = ['# Scenarios in this bundle' if mixed else f'# Resumed scenarios converted from {origins}', '']
    if mixed:
        lines += [
            f'{len(conversions) + len(plain)} scenario folder(s): {len(conversions)} resumed from {origins} and',
            f'{len(plain)} hand-authored. Both kinds are published as ONE set and generated together; what differs is',
            "where each folder's material came from, and whether its scenario continues a conversation or opens one.",
            '',
            f'## Resumed from {origins}',
            '',
            f'{len(conversions)} folder(s), one per recorded conversation thread. Each was produced',
            'deterministically from the source export — no model rewrote any of it.',
        ]
    else:
        lines += [
            f'{len(conversions)} scenario folder(s), one per recorded conversation thread. Each was produced',
            'deterministically from the source export — no model rewrote any of it.',
        ]
    lines += [
        '',
        f'{"Every folder in this section" if mixed else "Every folder"} carries a **resume pin** (`{RESUME_PIN_FILENAME}`): the verbatim message',
        'history and the agent state the real conversation ended on. Scenarios generated from these',
        'folders CONTINUE their conversation — they do not open a new one — so the pin content is',
        'authoritative and must be carried through unchanged, and the authored opener and policies',
        'must address what remains of the arc rather than what already happened.',
        '',
        '| Folder | Source export | Messages | Ends on |',
        '| --- | --- | --- | --- |',
    ]
    lines.extend(
        f'| `{_table_cell(c.folder)}` | `{_table_cell(c.provenance["source_file"])}` | {len(c.messages)} | `{c.messages[-1].type}` |' for c in conversions
    )
    # A list rather than a fifth column: most sets pin no node at all, and a column would spend a
    # cell of every row on it either way. Attributed per folder rather than stated once for the
    # set: one invocation pins one node and a set accumulates over invocations, so the folders
    # here can carry different nodes, or none, and the agent authors one opener per folder.
    if pinned := [(c.folder, c.resume_point.node) for c in conversions if c.resume_point is not None]:
        lines += [
            '',
            'These folder(s) also name a **start node** — the top-level graph node the harness enters on turn 1 of the',
            "resumed leg, so each folder's authored opener addresses the arc from THAT point rather than from wherever",
            'the traced conversation stopped:',
            '',
        ]
        lines.extend(f'* `{_provenance_display(folder)}` → `{_provenance_display(node)}`' for folder, node in pinned)
        if len(pinned) != len(conversions):
            lines += ['', "Every other folder above names no start node, and resumes at the graph's own entry point."]
    if grounded:
        lines += [
            '',
            '### Grounded in hand-authored materials',
            '',
            f'{len(grounded)} of the folder(s) above also hold hand-authored materials beside their resume pin. Their',
            'scenarios BOTH continue the pinned conversation AND stay grounded in those materials — the',
            "records that conversation refers to. Each folder's own `ROLES.md` carries the authoritative",
            'detail: the hand-authored description first, then the appended resume section.',
            '',
            '| Folder | Description |',
            '| --- | --- |',
        ]
        lines.extend(f'| `{_table_cell(folder)}` | {_table_cell(description)} |' for folder, description in grounded.items())
    if mixed:
        lines += [
            '',
            '## Hand-authored',
            '',
            f'{len(plain)} folder(s) carrying no resume pin. Their material was written for this set rather than taken',
            'from a trace, so each opens its conversation in the ordinary way — there is no prior arc to continue and',
            "nothing in them is byte-preserved. The description below is the opening line of the folder's own",
            f'`{ROLES_FILENAME}`, which is where the authoritative detail for it lives.',
            '',
            '| Folder | Description |',
            '| --- | --- |',
        ]
        lines.extend(f'| `{_table_cell(folder)}` | {_table_cell(description)} |' for folder, description in plain.items())
    return '\n'.join(lines) + '\n'


def _table_cell(value: Any) -> str:
    """One provenance value as a single markdown TABLE cell.

    :func:`_provenance_display` neutralises everything that ends a line, which is all a bullet can
    be broken with. A table row has a second structural character: ``|`` is PRINTABLE, so it
    survives the escape and splits one cell into several — and a filename may legally contain one
    on POSIX, so this is reachable without an adversary.
    """
    return _provenance_display(value).replace('|', r'\|')


def _parse_staged_pin(pin: Any, *, pin_path: Path, destination: Path) -> ResumePinDocument:
    """A staged pin read back as the shared model, or fail closed.

    The whole envelope is validated, not just the parts this module renders, because the
    alternative is not a graceful degrade: an ``AttributeError`` or ``KeyError`` from the render
    escapes the ``TraceConversionError`` family the CLI catches, so a half-plausible pin left in
    the staging directory turns a converted run into a raw traceback. A pin written by an OLDER or
    NEWER converter, or by a run interrupted mid-write, is exactly that shape.

    Validating against the model means this side refuses at staging what the backend would refuse at
    import — same schema, sooner, and naming the folder. The WORDS stay this module's: an operator
    here is looking at a directory they can fix, not at an upload that already happened.
    """
    where = f'{pin_path} is already in {destination} but'
    try:
        return ResumePinDocument.model_validate(pin)
    except ValidationError as exc:
        if any(error['loc'][:1] == ('messages',) and error['type'] in {'missing', 'too_short', 'list_type'} for error in exc.errors()):
            # Said first and said plainly: an envelope with no conversation is the one defect that
            # makes every other complaint about it beside the point.
            raise TraceConversionError(f'{where} carries no message history; it is not a usable resume pin.') from exc
        raise TraceConversionError(f'{where} {_first_problem(exc)}; it is not a usable resume pin.') from exc


def pinned_folders(out_dir: str | Path) -> list[ThreadConversion]:
    """Every resume-pinned scenario folder already under ``out_dir``, read back from its pin.

    The verb is driven ONE export at a time into a shared ``--out-dir`` (that is what the plugin
    skill does), so the set-level overview has to describe the folders ACCUMULATED there rather
    than only the ones an invocation just converted — otherwise the last run's ``BUNDLE.md``
    silently contradicts the set of folders sitting next to it.

    A folder with no pin is skipped, not an error: a bundle may legitimately mix resumed folders
    with regular hand-authored ones. A pin that is present but unreadable IS an error — the backend
    fails the folder closed at import anyway, so it is better surfaced here, naming the folder,
    than after upload.
    """
    destination = Path(out_dir)
    folders = []
    for pin_path in sorted(destination.glob(f'*/{RESUME_PIN_FILENAME}')):
        try:
            raw = json.loads(pin_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise TraceConversionError(f'{pin_path} is already in {destination} but is not readable as a resume pin ({exc}).') from exc
        pin = _parse_staged_pin(raw, pin_path=pin_path, destination=destination)
        # The origin is read back rather than defaulted, so a set accumulated one export at a time
        # describes each folder as what it was converted from. It is an EXTRA on the pin's one open
        # shape, so it is read as a key and type-checked here: a hand-edited pin can carry anything
        # under it, and a folder is described as a trace export unless it says otherwise.
        recorded = pin.provenance.model_dump().get('origin')
        origin = recorded if isinstance(recorded, str) and recorded.strip() else _DEFAULT_ORIGIN
        folders.append(ThreadConversion(folder=pin_path.parent.name, pin=pin, source=Path(pin.provenance.source_file), source_format=None, origin=origin))
    return folders


def _plain_description(roles_path: Path, *, destination: Path) -> str:
    """One line describing a pinless folder, read from its ``ROLES.md``.

    The first non-blank line with its heading marker dropped: that file opens with a title, and a
    title is what tells two hand-authored folders apart in a table. Truncated because the line is
    one table cell — a ``ROLES.md`` whose first line is a whole paragraph would stretch the row
    past reading width without adding meaning, and the folder's own file carries the full text
    either way.
    """
    try:
        text = roles_path.read_text(encoding='utf-8')
    except (OSError, ValueError) as exc:
        raise TraceConversionError(
            f'{roles_path} is already in {destination} but is not readable ({exc}). Every folder in a bundle dir needs a '
            f'readable {ROLES_FILENAME} — the bundle store refuses the whole upload without one.'
        ) from exc
    for line in text.splitlines():
        if stripped := line.strip().lstrip('#').strip():
            return stripped[:PLAIN_DESCRIPTION_MAX_CHARS]
    return f'(no description in {ROLES_FILENAME})'


def plain_folders(out_dir: str | Path) -> dict[str, str]:
    """Every PINLESS scenario folder already under ``out_dir``, mapped NAME → one-line description.

    Keyed by folder name because that is what the bundle store keys on too — two folders in one dir
    cannot share a name — and ordered by it, so the overview lists them the way the dir does.

    The counterpart to :func:`pinned_folders`, and read at the same moment for the same reason: a
    bundle dir may hold hand-authored folders beside converted ones and publish them as ONE mixed
    set, so an overview that named only the converted half would describe a smaller set than the
    one being uploaded.

    A folder with no readable ``ROLES.md`` is an ERROR, not a skip — the same ground the
    unreadable-pin arm of :func:`pinned_folders` stands on. The bundle store REQUIRES that file
    per folder and refuses the whole upload without it, so it is better surfaced here, naming the
    folder while it is still a directory the operator can fix, than after the upload. A file that
    is present but says nothing is DESCRIBED as such and kept: the store accepts it, and refusing
    what the thing being protected accepts would be a rule of this module's own invention.
    """
    destination = Path(out_dir)
    folders = {}
    for folder in sorted(p for p in destination.glob('*') if p.is_dir()):
        if (folder / RESUME_PIN_FILENAME).exists():
            continue
        roles = folder / ROLES_FILENAME
        if not roles.is_file():
            raise TraceConversionError(
                f'{folder} is already in {destination} but holds neither a {RESUME_PIN_FILENAME} nor a {ROLES_FILENAME}, '
                f'so it is neither a converted thread folder nor a usable hand-authored one. The bundle store refuses the '
                f'whole upload over it. Give it a {ROLES_FILENAME}, or move it out of the bundle dir.'
            )
        folders[folder.name] = _plain_description(roles, destination=destination)
    return folders


#: Everything a purely converted folder holds. A pinned folder carrying any entry beyond these was
#: authored by hand as well as converted, which is the whole test :func:`grounded_folders` applies.
_CONVERTED_FOLDER_FILES = frozenset({RESUME_PIN_FILENAME, ROLES_FILENAME, TRANSCRIPT_FILENAME})


def grounded_folders(out_dir: str | Path) -> dict[str, str]:
    """Every pinned folder under ``out_dir`` that ALSO holds hand-authored material, NAME → description.

    Grounding is a property OF a resumed folder, not a third kind of folder: such a folder carries
    a pin, so it is counted and tabled with the resumed ones and appears here as well, never twice.
    What it adds is that the conversation being continued is not the folder's only content — the
    records that conversation refers to are staged beside it, and the generation agent has to
    author against both.

    The test is what is on disk rather than how the folder got there, because the backend applies the
    same test: it counts every file but ``ROLES.md`` as scenario material, so a pin
    beside a PDF is a folder with two materials whatever wrote them.

    The description is the hand-authored one — the same reading :func:`plain_folders` takes, from
    the same opening line — because in a folder holding both kinds that line is what tells the
    folder apart. Its unreadable-``ROLES.md`` refusal comes along with it, which is why this is
    read back BEFORE any write: the bundle store refuses the whole upload over that file.
    """
    destination = Path(out_dir)
    folders = {}
    for folder in sorted(p for p in destination.glob('*') if p.is_dir()):
        if not (folder / RESUME_PIN_FILENAME).exists():
            continue
        if all(entry.name in _CONVERTED_FOLDER_FILES for entry in folder.iterdir()):
            continue
        folders[folder.name] = _plain_description(folder / ROLES_FILENAME, destination=destination)
    return folders


def _folder_conflict(conversion: ThreadConversion, out_dir: Path, *, merge: bool = False) -> tuple[str, str] | None:
    """``(kind, why)`` ``conversion`` must not be written under ``out_dir``, or ``None`` if it may.

    The in-invocation duplicate check cannot see this: the verb is driven ONE export per
    invocation into a shared ``--out-dir``, so a collision is normally between this run and a
    PREVIOUS one, and the write is an unconditional overwrite. Two exports collide whenever their
    file names sanitize to the same folder — which hand-collected exports (``trace.json``,
    ``customer-a.json``) do readily — and the loss is silent: both runs exit 0, and the first
    production conversation is simply gone.

    Re-converting the same export with the same flags stays idempotent (that is what makes the
    verb re-runnable), so the test is whether the pin CONTENT differs, not whether the folder
    exists. A flag whose value lands IN the pin is therefore part of what has to match:
    ``--start-node`` included, so correcting a node is refused rather than overwritten. The guard
    cannot tell a correction from a collision, and only one of the two is safe to lose.

    ``merge`` is the operator naming this folder as the deliberate destination, and it relaxes
    exactly ONE arm: a pinless folder stops being someone else's work and becomes the
    hand-authored half of a folder that carries both kinds. Nothing else relaxes — a folder
    already holding a DIFFERENT conversation is still refused, so a merge aimed at the wrong
    folder is turned away by the guard that was already there rather than by a second one.
    """
    folder = out_dir / conversion.folder
    if not folder.exists():
        return None
    existing = folder / RESUME_PIN_FILENAME
    if not existing.exists():
        if not merge:
            return 'not_a_converted_folder', (
                f'{folder} already exists and is not a converted thread folder (no {RESUME_PIN_FILENAME}), so converting '
                f'{conversion.source.name} into it would overwrite its {ROLES_FILENAME}. Convert into a different --out-dir, '
                'or move that folder aside.'
            )
        if not (folder / ROLES_FILENAME).is_file():
            return 'merge_target_has_no_roles', (
                f'{folder} holds no {ROLES_FILENAME}, so there is no hand-authored description to merge beside. '
                f'Give it a {ROLES_FILENAME} first — the bundle store refuses the whole upload without one.'
            )
        if (folder / TRANSCRIPT_FILENAME).exists():
            return 'merge_target_has_transcript', (
                f'{folder} already holds a {TRANSCRIPT_FILENAME}, and this folder carries no pin — so that file is '
                f'hand-authored, and the merge would replace it with the rendering of `{conversion.source.name}`. '
                f'{ROLES_FILENAME} survives a merge because a section can be appended to it; a transcript is a whole '
                'rendering of one conversation, with nothing to append it to. Rename that file (any other name is '
                'carried into the bundle as material), or drop --into.'
            )
        return None
    try:
        previous = json.loads(existing.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return 'unreadable_pin', (
            f'{folder} already holds a {RESUME_PIN_FILENAME} that cannot be read ({exc}), so it cannot be '
            f'told apart from the one {conversion.source.name} converts to.'
        )
    if previous == conversion.payload:
        return None
    was = previous.get('provenance', {}).get('source_file') if isinstance(previous, dict) else None
    return 'pin_differs', (
        f'{folder} already holds a resume pin (converted from `{was}`) whose content differs from the one '
        f'`{conversion.source.name}` converts to, and writing it would replace that conversation silently. This is either '
        'two exports whose names sanitize to the same folder, the same thread re-exported after it grew, or the same export '
        're-converted with a different --start-node or --cut-turn. Convert into a different --out-dir, or remove that folder first.'
    )


def _roles_text(conversion: ThreadConversion, roles_path: Path, *, merge: bool) -> str | None:
    """The folder's ``ROLES.md`` to write, APPENDED to a description a person already wrote there —
    or ``None`` when the file on disk is already complete and is left exactly as it is.

    Outside a merge the file is this converter's to write whole. Inside one it is not: the
    hand-authored text is the description of the materials the resumed conversation refers to, the
    converter cannot restate it, and overwriting it would destroy the very grounding the merge
    exists to keep. So the resume half is appended and the hand-authored half stays first and
    byte-untouched.

    Two shapes are already complete: a file this converter wrote whole (there is no hand-authored
    description to merge beside) and one already carrying this thread's section. The second is what
    keeps repeating the same merge a no-op instead of stacking a second copy of the section on
    every re-run.
    """
    if not (merge and roles_path.is_file()):
        return render_roles(conversion)
    existing = roles_path.read_text(encoding='utf-8')
    if existing == render_roles(conversion) or _roles_resume_heading(conversion) in existing.splitlines():
        return None
    return existing.rstrip('\n') + '\n\n' + _render_roles_resume_section(conversion)


@dataclass(frozen=True)
class _FolderFiles:
    """Every file one conversion writes, rendered and encoded — so writing them cannot fail halfway."""

    pin: bytes
    #: ``None`` when the ``ROLES.md`` already on disk is left as it is (see :func:`_roles_text`).
    roles: bytes | None
    transcript: bytes


def _folder_files(conversion: ThreadConversion, folder: Path, *, merge: bool) -> _FolderFiles:
    """Render and encode everything ``conversion`` writes into ``folder``, before any of it is written.

    Every failure a file can have — a render error, a character UTF-8 cannot encode — is met here,
    so a refused conversion leaves no pin on disk without the ``ROLES.md`` that describes it. The
    rendered files escape such a character (:func:`_writable`); the pin cannot, since its bytes are
    the conversation verbatim, so it is refused by name instead.
    """
    try:
        pin = conversion.pin_text.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ThreadRefused(
            _thread_finding_unrepresentable(
                conversion.source.name,
                conversion.source_format,
                'encode',
                f'{conversion.source.name}: the conversation holds {exc.object[exc.start : exc.end]!r}, which UTF-8 cannot '
                'encode — an unpaired UTF-16 surrogate, usually half of an emoji the producer cut in two — so the pin cannot '
                'carry it verbatim. Nothing was written for it. Re-export the thread from a producer that writes whole '
                'characters, or cut before the message that holds it.',
            )
        ) from exc
    roles = _roles_text(conversion, folder / ROLES_FILENAME, merge=merge)
    return _FolderFiles(
        pin=pin,
        roles=None if roles is None else roles.encode('utf-8'),
        transcript=render_transcript(conversion).encode('utf-8'),
    )


def _write_folder(folder: Path, files: _FolderFiles) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RESUME_PIN_FILENAME).write_bytes(files.pin)
    if files.roles is not None:
        (folder / ROLES_FILENAME).write_bytes(files.roles)
    (folder / TRANSCRIPT_FILENAME).write_bytes(files.transcript)


def write_scenario_folder(conversion: ThreadConversion, out_dir: str | Path, *, merge: bool = False) -> Path:
    """Write one scenario-bundle folder for ``conversion`` under ``out_dir``; return its path.

    ``merge`` writes the pin INTO a folder that was authored by hand, making it carry both kinds:
    the pin and transcript land as always, and the description is appended to rather than replaced.
    """
    folder = Path(out_dir) / conversion.folder
    if refusal := _write_boundary_refusal(conversion, Path(out_dir), merge=merge):
        raise ThreadRefused(refusal)
    _write_folder(folder, _folder_files(conversion, folder, merge=merge))
    return folder


def _write_boundary_refusal(conversion: ThreadConversion, out_dir: Path, *, merge: bool) -> GateFinding | None:
    """The finding that keeps ``conversion`` out of ``out_dir``, or ``None`` if it may be written.

    The seed rule first — a seed the backend refuses is refused wherever it would land — then the
    destination, which must not lose a conversation or a hand-authored folder to the write.
    """
    source = conversion.source.name
    if refusal := mid_turn_refusal(conversion):
        return _thread_finding_seed_shape(source, seed_shape_problem(conversion.messages), refusal)
    if conflict := _folder_conflict(conversion, out_dir, merge=merge):
        kind, why = conflict
        return _thread_finding_folder(source, conversion.folder, kind, why)
    return None


def discover_thread_exports(trace_export: str | Path, *, suffixes: tuple[str, ...] = ('.json',)) -> list[Path]:
    """The trace artifacts ``trace_export`` names: itself if a file, its ``suffixes`` if a directory.

    A directory is sorted so a run is deterministic, and ``manifest.json`` — the sibling
    ``trace_extract`` writes beside its exports — is skipped rather than failing the run as a
    malformed export.

    A single FILE is taken whatever its name: the operator pointed at it, so second-guessing the
    extension would refuse a correctly-named artifact for a reason the operator has already
    overruled. The suffixes are how a directory is enumerated, not a check on what a file may be.
    """
    path = Path(trace_export)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise TraceConversionError(f'{path} is neither a file nor a directory; point --export at a thread export or a directory of them.')
    scope = (*suffixes, *(f'{s}{readers.GZIP_SUFFIX}' for s in suffixes))  # an export reads the same gzipped
    exports = [p for p in sorted(path.iterdir()) if p.is_file() and readers.admits(p.name, scope) and p.name != 'manifest.json']
    if not exports:
        wanted = '/'.join(f'*{s}' for s in suffixes)
        raise TraceConversionError(
            f'{path} holds no {wanted} thread export, gzipped or not (manifest.json aside). Point --export at the directory the trace export wrote.'
        )
    return exports


def _warn_if_oversized(conversion: ThreadConversion) -> None:
    """Say so when the pin's seed half is heading for the backend's wire cap.

    Not a refusal: the cap applies to the YAML the backend serializes on the other side, so this side
    cannot know the exact figure. It is worth a warning anyway because the weight is
    usually NOT the conversation — an opaque ``agent_state`` channel scales with the customer's
    graph, not with trace length — so the operator would otherwise learn about it a stage later,
    from a folder that failed authoring, with no hint of which channel was to blame.
    """
    seed_bytes = len(json.dumps({'messages': conversion.payload['messages'], 'agent_state': conversion.agent_state}, ensure_ascii=False).encode('utf-8'))
    if seed_bytes <= SEED_SIZE_WARN_BYTES:
        return
    channels = sorted(((len(json.dumps(value, ensure_ascii=False).encode('utf-8')), name) for name, value in conversion.agent_state.items()), reverse=True)
    logger.warning(
        '%s: the resume seed is %d KB, over the %d KB the scenario wire allows; authoring may reject it. Largest agent_state channels: %s',
        conversion.source.name,
        seed_bytes // 1024,
        SEED_SIZE_WARN_BYTES // 1024,
        ', '.join(f'{name} ({size // 1024} KB)' for size, name in channels[:3]) or 'none (the conversation itself is the weight)',
    )


def _warn_about_tail(conversion: ThreadConversion) -> None:
    """Say out loud that a trim ran.

    A trim is the one thing the converter does that the source export does not say — the operator
    asked for it, and the count belongs on the same terminal as the folder it wrote.
    """
    if record := conversion.provenance.get('trimmed_mid_turn_tail'):
        logger.warning(
            '%s: --trim-mid-turn-tail dropped the last %d message(s) (%s); the pin resumes from the completed agent turn '
            'before them, and says so in its provenance.',
            conversion.source.name,
            record.get('dropped'),
            ', '.join(str(message_type) for message_type in record.get('dropped_types', [])),
        )


def _enclosing_workspaces(trace_export: str | Path) -> list[Path]:
    """Every directory at or above ``trace_export`` holding a workspace config, nearest first.

    The export's OWN workspaces, found by where the file sits rather than by ``--workspace-root``: an
    export copied out of an anonymized workspace still belongs to it, and naming another root on the
    command line must not be a way to read it raw. All of them, not the nearest: a ``plurai_sim/``
    config silent on the posture does not override an ``anonymize: true`` declared above it.
    """
    path = Path(trace_export).resolve()
    directories = (path, *path.parents) if path.is_dir() else path.parents
    return [directory for directory in directories if any((directory / relative).exists() for relative in CONFIG_CANDIDATE_RELPATHS)]


def _refuse_anonymized_posture(workspace_root: str | Path, *, trace_export: str | Path, rendering: bool) -> None:
    """Refuse outright when the env declared ``anonymize: true`` — before any export is read.

    Every workspace that can speak for the export is read, and any one declaring it refuses: the one
    ``--workspace-root`` names, and every one the export itself sits in. Every route through this module shares it, a
    rendering included, and the refusal says why in the terms of the route: a conversion would carry
    the conversation out verbatim, while a rendering would read trace files ingest has already
    scrubbed in place under that posture, and only ever prepares a conversion that is refused anyway.
    """
    named = Path(workspace_root).resolve()
    roots = [('workspace_root', named), *(('export_workspace', root) for root in _enclosing_workspaces(trace_export) if root != named)]
    for declared_by, root in roots:
        resolution = resolve_posture(root)
        if resolution.posture != 'anonymized':
            continue
        declared = f' (declared in {resolution.source})' if resolution.source else ''
        why = (
            'Under this posture ingest scrubs the local trace files in place, so a rendering would number a conversation '
            'other than the one that happened, and the conversion it prepares is refused here as well. '
            if rendering
            else 'The resume pin carries the conversation and the agent state VERBATIM — that is what makes the scenario resume '
            'rather than reopen — so converting here would egress unscrubbed production content from an env that has not '
            'consented to it. '
        )
        message = (
            f'this env declares anonymize: true{declared}, and trace-to-scenario is raw-only for now. {why}Nothing was written. Run this in a raw-posture env.'
        )
        raise PostureRefusedError(
            message,
            finding=make_finding(
                TRACE_POSTURE_ANONYMIZED,
                subject=Path(trace_export).name,
                evidence={'declared_by': declared_by},
                observed={'declared_in': str(resolution.source) if resolution.source else None},
                remedy=message,
                draft_ref=None,
            ),
        )


def _render_conflict(conversion: ThreadConversion, out_dir: Path) -> str | None:
    """Why ``conversion``'s rendering must not be written under ``out_dir``, or ``None`` if it may.

    A folder holding only a previous rendering is replaced, which keeps a re-render idempotent. A
    folder holding anything else is someone's work — a bundle folder, a hand-authored one — and a
    rendering is never worth overwriting it for.
    """
    folder = out_dir / conversion.folder
    if not folder.exists():
        return None
    if folder.is_dir() and all(entry.name == TRANSCRIPT_FILENAME and entry.is_file() for entry in folder.iterdir()):
        return None
    return (
        f'{folder} already exists and is not a rendering (it holds more than a {TRANSCRIPT_FILENAME}), so '
        '--render-only will not write over it. Render into a separate --out-dir.'
    )


def _convert_each(
    exports: Sequence[Path],
    spec: _FormatSpec,
    *,
    resume_point: ResumePoint | None,
    trim_mid_turn_tail: bool,
    cut_turn: int | None,
    skipped: list[SkippedThread],
    findings: list[GateFinding],
) -> list[ThreadConversion]:
    """Every export converted, each refused one skipped with its finding; the rest in export order."""
    converted: list[ThreadConversion] = []
    for export in exports:
        try:
            conversion = _convert_one(export, spec, resume_point=resume_point, trim_mid_turn_tail=trim_mid_turn_tail, cut_turn=cut_turn)
        except ThreadRefused as exc:
            _skip(skipped, findings, export, exc.finding)
            continue
        findings.extend(conversion.findings)
        converted.append(conversion)
    return converted


def _without_collisions(
    conversions: Sequence[ThreadConversion], *, verb: str, skipped: list[SkippedThread], findings: list[GateFinding]
) -> list[ThreadConversion]:
    """``conversions`` minus every one that shares its folder name with another, each skipped.

    Both members of a collision are skipped: neither can be chosen over the other, and either one
    written would be published as the only conversation under that name.
    """
    counts = Counter(c.folder for c in conversions)
    kept: list[ThreadConversion] = []
    for conversion in conversions:
        if counts[conversion.folder] == 1:
            kept.append(conversion)
            continue
        others = sorted(c.source.name for c in conversions if c.folder == conversion.folder and c is not conversion)
        remedy = (
            f'{conversion.source.name} and {others} {verb} to the same folder name {conversion.folder!r} — they would collide '
            'on upload, and neither can be chosen over the other. Rename the source exports, or convert them in separate runs '
            'into separate --out-dir directories.'
        )
        _skip(skipped, findings, conversion.source, _thread_finding_folder(conversion.source.name, conversion.folder, 'name_collision', remedy))
    return kept


def render_trace_export(
    trace_export: str | Path,
    out_dir: str | Path,
    *,
    source_format: str = DEFAULT_SOURCE_FORMAT,
    workspace_root: str | Path = '.',
) -> ConversionResult:
    """Write ONLY ``<out_dir>/<folder>/transcript.md`` for each trace artifact — the read before a cut.

    A caller that chooses ``--cut-turn K`` by reading the conversation reads this: the same posture
    gate, reader and projection as :func:`convert_trace_export`, so the turns it numbers are the
    turns a conversion of the same artifact cuts, and nothing else. No pin, ``ROLES.md`` or
    ``BUNDLE.md`` is written, so the output is never a bundle, and a directory already holding a
    ``BUNDLE.md`` is refused: a transcript-only folder there has no ``ROLES.md``, and the bundle
    store refuses the whole upload over it.

    The mid-turn refusal does not apply. It guards what gets uploaded and seeded; a rendering is
    only read, and a capture that stops mid-turn is often the one a caller needs to find a cut in.
    Per thread like a conversion: a thread that cannot be read, that collides with another, or whose
    folder holds more than a rendering is skipped with its finding, and every other one is rendered.
    """
    _refuse_anonymized_posture(workspace_root, trace_export=trace_export, rendering=True)
    spec = resolve_source_format(source_format, workspace_root=workspace_root)
    exports = discover_thread_exports(trace_export, suffixes=spec.suffixes)
    destination = Path(out_dir)
    if (destination / BUNDLE_OVERVIEW_FILENAME).exists():
        raise TraceConversionError(
            f'{destination} holds a {BUNDLE_OVERVIEW_FILENAME}, so it is a scenario bundle dir, and --render-only writes '
            'transcripts to read, not bundle folders — a folder holding only a transcript would make the bundle store '
            'refuse the whole upload. Render into a separate --out-dir.'
        )
    skipped: list[SkippedThread] = []
    findings: list[GateFinding] = []
    converted = _convert_each(exports, spec, resume_point=None, trim_mid_turn_tail=False, cut_turn=None, skipped=skipped, findings=findings)
    rendered: list[tuple[ThreadConversion, bytes]] = []
    for conversion in _without_collisions(converted, verb='render', skipped=skipped, findings=findings):
        if conflict := _render_conflict(conversion, destination):
            _skip(skipped, findings, conversion.source, _thread_finding_folder(conversion.source.name, conversion.folder, 'not_a_rendering', conflict))
            continue
        rendered.append((conversion, render_transcript(conversion, pinned=False).encode('utf-8')))

    for conversion, transcript in rendered:
        folder = destination / conversion.folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / TRANSCRIPT_FILENAME).write_bytes(transcript)
    return ConversionResult(conversions=tuple(c for c, _ in rendered), skipped=tuple(skipped), findings=tuple(findings))


def convert_trace_export(
    trace_export: str | Path,
    out_dir: str | Path,
    *,
    source_format: str = DEFAULT_SOURCE_FORMAT,
    workspace_root: str | Path = '.',
    into_folder: str | None = None,
    start_node: str | None = None,
    trim_mid_turn_tail: bool = False,
    cut_turn: int | None = None,
) -> ConversionResult:
    """Convert a trace artifact (or a directory of them) into scenario-bundle folders under ``out_dir``.

    ``source_format`` selects the front end: a :data:`TRACE_FORMAT_REGISTRY` built-in, or the name
    of an adapter persisted under ``workspace_root``. It is resolved ONCE for the whole run — a
    directory is one corpus in one format, and letting it be settled per file would mean a
    directory could convert half in one dialect and half in another.

    ``into_folder`` names an EXISTING folder under ``out_dir`` to convert into, so one folder
    carries the pin beside material someone authored by hand — a scenario that resumes a real
    conversation AND stays grounded in the records that conversation refers to. It is an explicit
    flag rather than a match on folder name because two exports whose names converge is an
    accident, and auto-merging one would turn the collision this verb refuses into a silent merge
    at exit 0. It pairs ONE export with ONE folder, and it never CREATES the folder: the other
    must be authored first, which is a precondition rather than a nicety.

    ``start_node`` pins the top-level graph node the harness enters on turn 1 of the resumed leg,
    onto every thread this invocation converts. It is carried verbatim and interpreted nowhere in
    this SDK: only the customer's harness holds the graph, so only it can say whether the name
    resolves, and it validates that at seed time. Nothing derives it from the trace — a node the
    traced conversation never reached is a legitimate target, so a trace-shaped guess would refuse
    correct input while blessing a stale one.

    ``trim_mid_turn_tail`` is the operator overruling the mid-turn refusal for a trace that was
    captured between an agent tool call and the tool reply: the incomplete tail is dropped, the pin
    resumes from the completed agent turn before it, and the cut is recorded in the pin provenance.
    Opt-in, and never a default — a pin the converter silently shortened would carry a conversation
    that never happened under the name of one that did.

    ``cut_turn`` stops each conversation short: turns ``0..cut_turn - 1`` are seeded and turn
    ``cut_turn`` becomes the scripted opener, so a recorded conversation yields a scenario that
    replays from the moment the operator is interested in rather than only from its end. It
    applies to every export in the run; a thread it names no turn of is skipped — see
    :func:`cut_at_turn`.

    PER THREAD. A thread that cannot be converted faithfully, whose seed the backend would refuse,
    or whose folder would replace another conversation or collide with another thread of this run
    is SKIPPED with its finding, and every other thread is written. Within that, nothing is
    half-written: every thread is converted and settled, and every pin already staged in ``out_dir``
    read back, BEFORE the first file is written, and ``BUNDLE.md`` is then rendered from what is on
    disk — so it never misdescribes the folders beside it. A run that writes nothing writes no
    ``BUNDLE.md`` either. What refuses the whole INVOCATION, before any folder is written, is the
    operator's own input: an ``anonymized`` posture, a bad ``--into`` / ``--start-node``, an
    unresolvable format, an empty export directory, or a folder already staged in ``out_dir`` that
    the bundle store would refuse.
    """
    _refuse_anonymized_posture(workspace_root, trace_export=trace_export, rendering=False)

    # A folder NAME, never a path: --into addresses one directory that is already a bundle unit,
    # and a segmented value would either escape --out-dir or name a nested directory the bundle
    # store does not see as a folder at all. Checked on the argument, before anything reads disk.
    if into_folder is not None and (Path(into_folder).name != into_folder or '\\' in into_folder or into_folder in {'.', '..'}):
        raise TraceConversionError(f'--into must name a folder directly under --out-dir; {into_folder!r} is a path, not a folder name.')
    # Before any read: a blank --start-node is wrong whatever the exports hold, and a directory of
    # exports must not answer a defect in the operator's own command line with whatever its first
    # artifact happens to be wrong about.
    resume_point = _resume_point(start_node)

    # Resolved BEFORE discovery: the format decides which files in a directory are artifacts at
    # all, and an unresolvable --source-format should say so rather than first reporting an empty
    # directory it enumerated with the wrong glob.
    spec = resolve_source_format(source_format, workspace_root=workspace_root)
    exports = discover_thread_exports(trace_export, suffixes=spec.suffixes)
    if into_folder is not None and len(exports) > 1:
        raise TraceConversionError(
            f'--into pairs ONE export with ONE existing folder, but {trace_export} holds {len(exports)} exports. '
            f'Point --export at the single export that belongs in {into_folder!r}, and convert the rest without --into.'
        )
    destination = Path(out_dir)
    if into_folder is not None and not (destination / into_folder).is_dir():
        raise TraceConversionError(
            f'{destination / into_folder} does not exist, so there is no hand-authored folder to convert into. '
            '--into names a folder ALREADY authored under --out-dir — author it first, or drop --into to write a new '
            'thread-<name> folder.'
        )
    # Read back what is ALREADY staged here before writing anything. Doing it afterwards would let
    # a corrupt pin left in the staging dir abort the render with the new folder already written —
    # leaving a BUNDLE.md that describes a smaller set than the folders sitting beside it, which
    # is precisely the contradiction pinned_folders exists to prevent. All three read-backs run
    # here for their refusals; the render below takes its values from the post-write disk instead.
    pinned_folders(destination)
    plain_folders(destination)
    grounded_folders(destination)

    skipped: list[SkippedThread] = []
    findings: list[GateFinding] = []
    converted = _convert_each(
        exports, spec, resume_point=resume_point, trim_mid_turn_tail=trim_mid_turn_tail, cut_turn=cut_turn, skipped=skipped, findings=findings
    )
    if into_folder is not None:
        # Re-rooted here, once, so the destination name is what every later stage sees: the
        # collision guard, the writes, and the summary's `folder` field, which is how an operator
        # reads back where the conversation actually landed.
        converted = [replace(conversion, folder=into_folder) for conversion in converted]

    merge = into_folder is not None
    staged: list[tuple[ThreadConversion, _FolderFiles]] = []
    for conversion in _without_collisions(converted, verb='convert', skipped=skipped, findings=findings):
        # Every refusal runs for every surviving thread before the first write, and every file is
        # rendered and encoded here too, so no failure is left for the write loop to meet.
        if refusal := _write_boundary_refusal(conversion, destination, merge=merge):
            _skip(skipped, findings, conversion.source, refusal)
            continue
        try:
            files = _folder_files(conversion, destination / conversion.folder, merge=merge)
        except ThreadRefused as exc:
            _skip(skipped, findings, conversion.source, exc.finding)
            continue
        staged.append((conversion, files))

    if staged:
        destination.mkdir(parents=True, exist_ok=True)
    for conversion, files in staged:
        written = destination / conversion.folder
        _write_folder(written, files)
        logger.info('converted %s → %s (%d messages)', conversion.source.name, written, len(conversion.messages))
        _warn_about_tail(conversion)
        _warn_if_oversized(conversion)
    if staged:
        # Rendered from what is NOW on disk, not from `conversions` alone — see pinned_folders. All
        # three halves are re-read, because a merge moves a folder between them: the folder it wrote
        # into carried no pin a moment ago, so the pre-write reads would list it as hand-authored AND
        # as resumed, describing one folder as two scenarios.
        (destination / BUNDLE_OVERVIEW_FILENAME).write_text(
            render_bundle_overview(pinned_folders(destination), plain_folders(destination), grounded_folders(destination)), encoding='utf-8'
        )
    return ConversionResult(conversions=tuple(c for c, _ in staged), skipped=tuple(skipped), findings=tuple(findings))


__all__ = [
    'BUNDLE_OVERVIEW_FILENAME',
    'CANONICAL_MESSAGE_TYPES',
    'CONVERTER_PROVENANCE_KEYS',
    'DEFAULT_MESSAGES_CHANNEL',
    'DEFAULT_SOURCE_FORMAT',
    'SOURCE_INDEX_MINTED_ID_RECIPE',
    'ConversionResult',
    'FormatUnsupportedError',
    'NormalizedMessages',
    'RootSelection',
    'SeedShapeProblem',
    'SkippedThread',
    'ThreadRefused',
    'drop_leading_orphan_results',
    'mint_message_id',
    'normalize_role_keyed_message',
    'normalize_source_messages',
    'resolve_messages_channel',
    'seed_shape_problem',
    'PLAIN_DESCRIPTION_MAX_CHARS',
    'RESUME_PIN_FILENAME',
    'RESUME_PIN_SCHEMA_VERSION',
    'ROLES_FILENAME',
    'RUN_MINTED_ID_RECIPE',
    'SEED_SIZE_WARN_BYTES',
    'SUPPORTED_MESSAGE_TYPES',
    'TRACE_FORMAT_REGISTRY',
    'TRANSCRIPT_FILENAME',
    'WORLD_FACTS_HEADING',
    'WORLD_FACTS_MAX_BYTES',
    'WORLD_FACT_VALUE_MAX_BYTES',
    'PostureRefusedError',
    'ResumePinDocument',
    'ResumePinMessage',
    'SourceProjection',
    'ThreadConversion',
    'TraceConversionError',
    'WorldFact',
    'WorldFacts',
    'canonical_message',
    'canonical_messages',
    'convert_thread_export',
    'convert_trace_export',
    'cut_at_turn',
    'discover_thread_exports',
    'grounded_folders',
    'mid_turn_refusal',
    'mid_turn_tail_problem',
    'SEED_SHAPE_RULES_MIRRORED',
    'SEED_SHAPE_RULES_VERSION',
    'pinned_folders',
    'plain_folders',
    'read_source_text',
    'render_bundle_overview',
    'render_roles',
    'render_trace_export',
    'render_transcript',
    'resolve_source_format',
    'select_final_run',
    'user_turn_positions',
    'world_facts_after_cut',
    'write_scenario_folder',
]
