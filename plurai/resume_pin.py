"""The resume pin's schema, DECLARED — the producer half of a client/backend contract.

``resume_pin.json`` is written here, by :mod:`plurai.trace_convert`, and read by the Plurai
backend. Neither side depends on the other, so the schema cannot be imported across the
boundary: it is DUPLICATED, the same model declared twice, and what keeps the two copies honest
is ``schemas/resume_pin.schema.json`` — a normalized
:meth:`~pydantic.BaseModel.model_json_schema` regenerated and diffed by CI on both sides
(``scripts/regen_resume_pin_schema.py``).

**This package is the source of truth** — it PRODUCES the artifact. The models below are the contract written
out as types rather than as prose a reader keeps in sync by hand:

* a discriminated union on ``type``, so the closed message-type set is STRUCTURAL. The alternative
  — a ``frozenset`` of names checked by hand — is a second copy of the same fact, and
  :data:`CANONICAL_MESSAGE_TYPES` is DERIVED from the union rather than restated beside it;
* ``extra='forbid'`` throughout, so "exactly these keys and nothing else" is the type instead of a
  sentence in a docstring. :class:`ResumePinProvenance` is the ONE deliberately open shape;
* required-and-non-empty expressed once, on :data:`NonBlankStr`, rather than at each use.

**What the shared shape does NOT share is the error rendering.** The models define fields, types,
the discriminator and the closed/open split; each side keeps its own operator-facing text, because
the two sides fail in different places — this one refuses at convert time, in front of an operator
holding the export; the backend refuses one folder of an uploaded bundle. :mod:`plurai.trace_convert`
therefore keeps its own CLI-grade refusals and does not surface a raw ``ValidationError``.

**The pin is BORN from these models**, so ``model_dump()`` is what reaches disk — key order and an
explicit ``tool_calls: null`` on an agent message that made no calls included. Never hand-build
that dict: dumping the model is what makes the pin's bytes deterministic and IDENTICAL on both
sides, which matters because byte-verbatim carriage is load-bearing all the
way downstream — ``convert_to_messages`` is not idempotent on its own serialization, and the
client's turn-0 excision matches the seeded prefix against its own. Any re-serialization that
changes bytes scores turn 1 against words the agent never said.

**The fence proves the two current schemas agree; it cannot fix runtime version skew.** A customer can
run an older SDK against a newer backend, which is what :data:`RESUME_PIN_SCHEMA_VERSION` covers. And
because ``extra='forbid'`` fails closed, an ADDITIVE field still needs a version bump — a shared
model does not make additions free.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

#: Envelope version of ``resume_pin.json``. Bump for ANY envelope change, an additive field
#: included: every shape but :class:`ResumePinProvenance` is ``extra='forbid'``, so a reader of an
#: older version refuses a pin carrying a key it has never heard of. The backend validates the version
#: and fails a folder whose value it does not know. This — NOT the drift fence — is what covers a
#: customer running an SDK and a backend of different ages.
RESUME_PIN_SCHEMA_VERSION = 2


def _reject_blank(value: str) -> str:
    """``min_length=1`` accepts ``'   '``; an id of three spaces is not an id.

    Measured, not assumed: pydantic's length constraint counts characters, so a whitespace-only
    id satisfies it and then reaches ``add_messages`` as a reconciliation key that collides with
    every other blank one.
    """
    if not value.strip():
        raise ValueError('must be a non-empty string')
    return value


#: A required identifier: present, a string, and not whitespace.
NonBlankStr = Annotated[str, Field(min_length=1), AfterValidator(_reject_blank)]


class _ClosedModel(BaseModel):
    """Every shape in the pin is closed. ``extra='forbid'`` is the whole reason these models
    replace hand-written key checks: "exactly these keys" stops being a sentence a reader has to
    keep in sync with a ``frozenset`` and becomes the type itself."""

    model_config = ConfigDict(extra='forbid')


class ResumeToolCall(_ClosedModel):
    """A tool call as the pin carries it: ``{name, args, id}``, nothing else.

    ``args`` is a mapping deliberately: LangSmith stores tool arguments as a JSON *string*, and a
    string that reached the customer's tool node would arrive as an opaque blob the node cannot
    address. The converter parses it on the way in; this refuses the unparsed form.

    ``id`` is what pairs the call with its ``ToolMessage``. Without it the exchange cannot be
    reconstructed, and a synthesized id would pair with nothing.
    """

    name: NonBlankStr
    args: dict[str, Any]
    id: NonBlankStr


class _PinMessage(_ClosedModel):
    """The fields every pinned message carries.

    ``content`` is ``Any`` on purpose and must stay that way. It is a plain string for most models
    and a list of typed blocks for Anthropic-style ones; both survive ``convert_to_messages``
    unchanged, and the converter flattens neither.

    What forbids a narrower annotation is not re-serialization: an element-blind ``str |
    list[Any]`` validates only the container and hands the same block objects back out, measured.
    It is that the set of shapes a seeded message may carry is not a static type. ``ToolMessage``
    stringifies whatever it is given, so no content can fail on a tool message; and falsy content
    of any type is coerced to ``''`` rather than refused. A union tight enough to read as
    documentation therefore refuses shapes the seed loader reads happily, which kills a scenario
    folder for no reason. Each end states the rule where it can also name what went wrong instead:
    :func:`_pin_message` on the way in, and the backend reader by probing every message with
    ``convert_to_messages`` itself. ``Any`` still makes the key REQUIRED when absent, which is the
    half this model wants.

    ``id`` is required and not a formality: LangGraph's ``add_messages`` reconciles by id, and
    that is the whole reason re-seeding the client's checkpointer is idempotent. An id-less
    history appends a duplicate copy of the entire conversation on every re-seed.
    """

    content: Any
    id: NonBlankStr


class HumanPinMessage(_PinMessage):
    type: Literal['human']


class SystemPinMessage(_PinMessage):
    type: Literal['system']


class AIPinMessage(_PinMessage):
    type: Literal['ai']
    #: ``None`` — dumped as an explicit ``null`` — when the agent only spoke. The key is always
    #: present rather than omitted on a call-free message: one shape for every agent message is
    #: what lets both sides emit the same bytes.
    tool_calls: list[ResumeToolCall] | None = None


class ToolPinMessage(_PinMessage):
    type: Literal['tool']
    #: A tool result that names no call cannot be replayed — the pairing is what makes the
    #: exchange reconstructible, and ``convert_to_messages`` reports its absence as a bare
    #: ``KeyError`` far away from the trace that caused it.
    tool_call_id: NonBlankStr


#: Discriminated on ``type``, which is what makes the closed set STRUCTURAL rather than a
#: ``frozenset`` restated on both sides. The aliases LangChain accepts on INPUT
#: (``user``/``assistant``/``developer``) are folded by the converter before a pin is written and
#: are deliberately NOT members here: admitting one would forgive no real producer while writing a
#: pin the backend's reader refuses at run time. ``function`` is absent for a different reason
#: — it rebuilds as a ``FunctionMessage`` needing a ``name`` this schema has no field for.
ResumePinMessage = Annotated[
    HumanPinMessage | AIPinMessage | ToolPinMessage | SystemPinMessage,
    Field(discriminator='type'),
]

#: DERIVED from the union above, never hand-listed. A ``frozenset`` written out beside the union
#: is a second copy of the same fact, and the two only agree until one of them is edited; deriving
#: it means a member joining or leaving the union moves both ends at once. The backend derives its own
#: set the same way, from its own copy of the union — the pair the schema fence keeps
#: equal.
CANONICAL_MESSAGE_TYPES = frozenset(get_args(member.model_fields['type'].annotation)[0] for member in get_args(get_args(ResumePinMessage)[0]))


class ResumePinProvenance(BaseModel):
    """Where the pinned conversation came from. **The one open shape in the pin**, deliberately:
    provenance never reaches the runtime carrier (the backend drops it when it writes the seed), so
    this converter may record more about an export without a lockstep backend release. Everything
    else is closed."""

    model_config = ConfigDict(extra='allow')

    source_file: NonBlankStr
    thread_id: NonBlankStr


class ResumePoint(_ClosedModel):
    """Where the resumed leg STARTS: the top-level graph node the harness enters on turn 1.

    Operator-supplied and carried verbatim, exactly like ``agent_state``. The SDK shape-validates
    the name and nothing more — the graph is known only to the customer's harness, which is also
    the only side that can say whether the node exists, so a name this converter cannot recognise
    is not thereby wrong. A node the traced conversation never entered is a legitimate target: the
    operator is choosing where the next leg begins, not describing where the last one ended.
    """

    node: NonBlankStr


class ResumePinDocument(_ClosedModel):
    """``resume_pin.json`` as a type — the five-key envelope both sides code against."""

    schema_version: int
    messages: Annotated[list[ResumePinMessage], Field(min_length=1)]
    #: May legitimately be empty: an agent whose whole state lives in ``messages`` seeds an empty
    #: values dict. Present-but-not-a-mapping is still a defect.
    agent_state: dict[str, Any]
    #: ``None`` — dumped as an explicit ``null`` — unless an operator pinned a start node. The key
    #: is always present rather than omitted, the same rule an agent message's ``tool_calls``
    #: follows: one envelope shape, so both sides of the contract emit the same bytes.
    resume_point: ResumePoint | None = None
    provenance: ResumePinProvenance

    @model_validator(mode='after')
    def _check_version(self) -> ResumePinDocument:
        # Not ``Literal[1]``: the version mismatch is the one envelope defect with an obvious
        # operator action, and it deserves a sentence that names it rather than a bare literal
        # error. It fires when a pin staged by an OLDER or NEWER converter is read back.
        if self.schema_version != RESUME_PIN_SCHEMA_VERSION:
            raise ValueError(
                f'declares schema_version {self.schema_version!r}, but this converter writes version '
                f'{RESUME_PIN_SCHEMA_VERSION} — regenerate it with a matching converter'
            )
        return self

    @model_validator(mode='after')
    def _check_unique_ids(self) -> ResumePinDocument:
        # The quiet killer: ``add_messages`` reconciles BY id, so two messages sharing one id
        # collapse into a single message at seed time. The history would be short by a turn with
        # nothing anywhere reporting it.
        seen: dict[str, int] = {}
        for i, message in enumerate(self.messages):
            if message.id in seen:
                raise ValueError(
                    f'messages[{i}].id {message.id!r} duplicates messages[{seen[message.id]}] — '
                    f'add_messages reconciles by id, so the two would collapse into one'
                )
            seen[message.id] = i
        return self


__all__ = [
    'CANONICAL_MESSAGE_TYPES',
    'RESUME_PIN_SCHEMA_VERSION',
    'AIPinMessage',
    'HumanPinMessage',
    'NonBlankStr',
    'ResumePinDocument',
    'ResumePinMessage',
    'ResumePinProvenance',
    'ResumePoint',
    'ResumeToolCall',
    'SystemPinMessage',
    'ToolPinMessage',
]
