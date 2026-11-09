"""The scenario half of ``agent.io.channels``: what a scenario plants through each channel, and when.

The feature has one owner per concern: the kind VOCABULARY (``ChannelKind``, ``CHANNEL_KINDS``, which schedule
each kind rides) is the scan contract's, :mod:`plurai.scan_contract.input_channels`, vendored from the backend so the
backend that authors scenarios and this runner read one definition; the harness's DECLARATION (the
``agent.io.channels`` entries in ``plurai.config.yaml``) is parsed by :mod:`plurai.agent_io` with the rest of
``agent.io``; this module checks a scenario's PLAN against that declaration.

A harness declares its non-message inputs once (``agent.io.channels``, :mod:`plurai.agent_io`). A scenario
plants into them through two lists the backend authors on its metadata — ``CreateThreadResponse.scenario_metadata``,
which the parent forwards to the child as ``Init.scenario_metadata``:

* ``turn_inputs: [{turn, channel, value}]`` — planted on turn ``turn`` (0-based, the transcript row's
  ``turn_idx``) through a per-turn channel (``file_by_reference``, ``structured_context``,
  ``host_injected_message``);
* ``between_turn_actions: [{after_turn, channel, mutation}]`` — a user/UI change written to the world after
  turn ``after_turn`` and before the next agent call, through a ``between_turn_action`` channel.

:func:`parse_channel_plan` checks both against the declaration and fails closed: a scenario that plants into
an undeclared channel, through the wrong kind, or with a value its kind cannot carry is refused at scenario
start (:class:`HarnessUndeclaredChannel`), never run against a harness that would drop the input. The parent
refuses; the child applies each entry by kind (``plurai.child``) and reports it as an
:class:`AppliedChannelInput`, which the turn's transcript row records under ``channel_inputs``.

What ``invoke_agent`` receives on a turn that carries any: each per-turn channel's value under its declared
``field`` (a file's id — or id list — for ``file_by_reference``, the value itself for ``structured_context``),
host-injected messages ahead of the user's message in ``messages``, and the typed records under
:data:`CHANNEL_INPUTS_KEY`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

from plurai.agent_io import AGENT_IO_KEY, CHANNELS_FIELD, AgentIO, AgentIOChannel
from plurai.ingest.workspace_config import WorkspaceConfigError
from plurai.scan_contract.input_channels import SCHEDULE_OF_KIND, ChannelKind

#: The scenario-metadata keys of the two lists — the values of the scan contract's ``SCHEDULE_OF_KIND``, which says
#: which list each channel kind is planted through.
TURN_INPUTS_KEY: Final = 'turn_inputs'
BETWEEN_TURN_ACTIONS_KEY: Final = 'between_turn_actions'
#: The payload key a turn that carries channel inputs hands ``invoke_agent``: a tuple of :class:`AppliedChannelInput`.
#: Also the transcript row key the same records are filed under, as dicts.
CHANNEL_INPUTS_KEY: Final = 'channel_inputs'
#: The roles a host-injected message may take.
HOST_MESSAGE_ROLES: tuple[str, ...] = ('system', 'user', 'assistant')
#: Payload keys a channel ``field`` may not name: the runner writes them itself.
_RESERVED_PAYLOAD_KEYS = frozenset({'messages', 'interrupt_choice', CHANNEL_INPUTS_KEY})
#: The kinds whose value is written into a payload field.
_PAYLOAD_FIELD_KINDS: frozenset[ChannelKind] = frozenset({'file_by_reference', 'structured_context'})

#: The typed failure kind of a scenario that plants into a channel the harness does not declare (or through the
#: wrong kind, or with a value the kind cannot carry). Harness-owned: the row is not valid, never a failed trial.
HARNESS_UNDECLARED_CHANNEL_KIND: Final = 'harness_undeclared_channel'

ChannelStatus = Literal['applied', 'refused', 'unexecutable']


class HarnessUndeclaredChannel(WorkspaceConfigError):  # noqa: N818  # named for the fact, like its siblings
    """A scenario's ``turn_inputs`` / ``between_turn_actions`` do not fit the harness's ``agent.io.channels``: the
    scenario set and the config do not describe the same bench. Remedy: declare the channel, or regenerate the scenario."""

    kind: ClassVar[str] = HARNESS_UNDECLARED_CHANNEL_KIND


@dataclass(frozen=True, slots=True)
class PlannedChannelInput:
    """One scenario entry, resolved to its declared channel and the turn whose agent call it reaches."""

    #: The turn (0-based) at whose open it is applied.
    turn: int
    channel: AgentIOChannel
    #: A turn input's ``value``, or a between-turn action's ``mutation``.
    value: Any
    #: Its position in the scenario list it came from (mints a created row's id).
    index: int
    #: Set for a between-turn action: the turn it follows.
    after_turn: int | None = None


@dataclass(frozen=True, slots=True)
class AppliedChannelInput:
    """What one planted entry did — handed to ``invoke_agent`` and recorded on the turn's transcript row."""

    channel: str
    kind: ChannelKind
    field: str
    #: ``applied``; ``refused`` (the world refused the write); ``unexecutable`` (nothing could write it).
    status: ChannelStatus
    #: What reached the agent or the world: the id(s) put in ``field``, the payload value, the message, or the mutation.
    value: Any = None
    after_turn: int | None = None
    detail: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            'channel': self.channel,
            'kind': self.kind,
            'field': self.field,
            'status': self.status,
            'value': self.value,
            **({'after_turn': self.after_turn} if self.after_turn is not None else {}),
            **({'detail': self.detail} if self.detail else {}),
        }


@dataclass(frozen=True, slots=True)
class ChannelPlan:
    """A scenario's planted channel inputs, in application order within each turn."""

    entries: tuple[PlannedChannelInput, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)

    def for_turn(self, turn_idx: int) -> tuple[PlannedChannelInput, ...]:
        """The entries applied at the open of ``turn_idx``: the between-turn actions first (they happened after
        the previous turn), then the turn's own inputs, each in scenario order."""
        due = [entry for entry in self.entries if entry.turn == turn_idx]
        return tuple(sorted(due, key=lambda entry: entry.after_turn is None))


EMPTY_PLAN = ChannelPlan()


def scenario_plants_channels(metadata: Mapping[str, Any] | None) -> bool:
    """Whether the scenario metadata carries either list at all (so the declaration has to be read)."""
    if not metadata:
        return False
    return any(metadata.get(key) not in (None, []) for key in (TURN_INPUTS_KEY, BETWEEN_TURN_ACTIONS_KEY))


# --- value checks, one per kind: ``None`` when the value fits, else what is wrong -------------------------


def _item_problem(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return f'an item must be a mapping {{entity_id, <fields…>}}, got {type(item).__name__}'
    entity_id = item.get('entity_id')
    if not isinstance(entity_id, str) or not entity_id.strip():
        return f'an item must carry a non-blank `entity_id` (the id passed to the agent), got {entity_id!r}'
    return None


def _row_type_problem(row: Mapping[str, Any]) -> str | None:
    """A row the plan may CREATE must name its ``_entity_type``: the world refuses a write to a key no entity
    holds unless it declares one, so an untyped new row would never reach the store the agent reads."""
    declared = row.get('_entity_type')
    return None if isinstance(declared, str) and declared.strip() else f'a new row must carry its `_entity_type`, got {declared!r}'


def _file_by_reference_problem(value: Any) -> str | None:
    items = value if isinstance(value, list) else [value]
    if not items:
        return 'the item list is empty'
    return next((problem for item in items if (problem := _item_problem(item) or _row_type_problem(item)) is not None), None)


def _structured_context_problem(value: Any) -> str | None:
    return 'the value is missing (null)' if value is None else None


def _host_message_problem(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return f'the value must be a message {{role, content}}, got {type(value).__name__}'
    if value.get('role') not in HOST_MESSAGE_ROLES:
        return f'`role` must be one of {list(HOST_MESSAGE_ROLES)}, got {value.get("role")!r}'
    if not isinstance(value.get('content'), str | list):
        return f'`content` must be a string or a content-part list, got {type(value.get("content")).__name__}'
    return None


def _mutation_problem(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return f'the mutation must be {{entity_id, <fields…>}} or {{create: {{…}}}}, got {type(value).__name__}'
    if 'create' in value:
        created = value['create']
        if len(value) != 1 or not isinstance(created, Mapping) or not created:
            return 'a create is `{create: {<fields…>}}` alone, with at least one field'
        if (problem := _row_type_problem(created)) is not None:
            return problem
        entity_id = created.get('entity_id')
        if entity_id is None or (isinstance(entity_id, str) and entity_id.strip()):
            return None
        return f'`create.entity_id` must be non-blank when set, got {entity_id!r}'
    if (problem := _item_problem(value)) is not None:
        return problem
    return None if len(value) > 1 else 'the mutation names a row but no field to change'


#: A channel kind → the check of the value a scenario plants through it. Keyed on the contract's
#: :data:`~plurai.scan_contract.input_channels.ChannelKind`, so the type checker rejects a key outside the vocabulary.
_VALUE_PROBLEMS: Mapping[ChannelKind, Callable[[Any], str | None]] = MappingProxyType(
    {
        'file_by_reference': _file_by_reference_problem,
        'structured_context': _structured_context_problem,
        'host_injected_message': _host_message_problem,
        'between_turn_action': _mutation_problem,
    }
)


@dataclass(frozen=True, slots=True)
class _ListSpec:
    """How one scenario list is read: its turn key, its value key, the kinds it may name, the turn it lands on."""

    turn_key: str
    value_key: str
    kinds: frozenset[ChannelKind]
    #: Added to the entry's turn key to give the turn whose open applies it.
    lands_after: int


def _kinds_scheduled_through(list_key: str) -> frozenset[ChannelKind]:
    """The kinds the scan contract's ``SCHEDULE_OF_KIND`` plants through ``list_key`` — read, never restated."""
    return frozenset(kind for kind, schedule in SCHEDULE_OF_KIND.items() if schedule == list_key)


_LISTS: Mapping[str, _ListSpec] = MappingProxyType(
    {
        TURN_INPUTS_KEY: _ListSpec('turn', 'value', _kinds_scheduled_through(TURN_INPUTS_KEY), 0),
        BETWEEN_TURN_ACTIONS_KEY: _ListSpec('after_turn', 'mutation', _kinds_scheduled_through(BETWEEN_TURN_ACTIONS_KEY), 1),
    }
)


def parse_channel_plan(metadata: Mapping[str, Any] | None, agent_io: AgentIO | None, *, scenario_id: str, declared_by: str) -> ChannelPlan:
    """The scenario's channel plan, checked against ``agent_io``; :data:`EMPTY_PLAN` when it plants nothing.

    Raises :class:`HarnessUndeclaredChannel` naming the scenario, the entry and ``declared_by`` (where the
    declaration in force came from) on the first entry that does not fit.
    """
    if metadata is None or not scenario_plants_channels(metadata):
        return EMPTY_PLAN

    def refuse(where: str, problem: str) -> HarnessUndeclaredChannel:
        return HarnessUndeclaredChannel(
            f'refusing to run scenario {scenario_id!r}: `{where}` {problem}. {declared_by} declares the channels '
            f'{[channel.name for channel in agent_io.channels] if agent_io is not None else []} under `{AGENT_IO_KEY}.{CHANNELS_FIELD}`. '
            f'Declare the channel (and read it in `invoke_agent`), or regenerate the scenario against this config.'
        )

    entries: list[PlannedChannelInput] = []
    seen: set[tuple[int, str]] = set()
    for list_key, spec in _LISTS.items():
        declared = metadata.get(list_key)
        if declared is None:
            continue
        if not isinstance(declared, list):
            raise refuse(list_key, f'must be a list, got {type(declared).__name__}')
        for index, entry in enumerate(declared):
            where = f'{list_key}[{index}]'
            if not isinstance(entry, Mapping):
                raise refuse(where, f'must be a mapping {{{spec.turn_key}, channel, {spec.value_key}}}, got {type(entry).__name__}')
            turn = entry.get(spec.turn_key)
            if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
                raise refuse(where, f'`{spec.turn_key}` must be a 0-based turn index, got {turn!r}')
            name = entry.get('channel')
            channel = agent_io.channel(name) if agent_io is not None and isinstance(name, str) else None
            if channel is None or not isinstance(name, str):
                raise refuse(where, f'plants into the channel {name!r}, which is not declared')
            if channel.kind not in spec.kinds:
                raise refuse(where, f'plants into {name!r}, a `{channel.kind}` channel; `{list_key}` takes {sorted(spec.kinds)}')
            if channel.kind in _PAYLOAD_FIELD_KINDS and channel.field in _RESERVED_PAYLOAD_KEYS:
                raise refuse(where, f'plants into {name!r}, whose field {channel.field!r} is a payload key the runner writes itself')
            if spec.value_key not in entry:
                raise refuse(where, f'carries no `{spec.value_key}`')
            if (problem := _VALUE_PROBLEMS[channel.kind](entry[spec.value_key])) is not None:
                raise refuse(where, f'cannot be planted through the `{channel.kind}` channel {name!r}: {problem}')
            lands = turn + spec.lands_after
            if list_key == TURN_INPUTS_KEY:
                if (lands, name) in seen:
                    raise refuse(where, f'is a second entry for channel {name!r} on turn {turn}; put every item of one turn in one `value`')
                seen.add((lands, name))
            entries.append(
                PlannedChannelInput(
                    turn=lands,
                    channel=channel,
                    value=entry[spec.value_key],
                    index=index,
                    after_turn=turn if list_key == BETWEEN_TURN_ACTIONS_KEY else None,
                )
            )
    return ChannelPlan(tuple(entries))


def item_write(item: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(entity_id, fields)`` of one ``file_by_reference`` item or ``{entity_id, …}`` mutation."""
    return str(item['entity_id']).strip(), {key: value for key, value in item.items() if key != 'entity_id'}


def mutation_write(entry: PlannedChannelInput) -> tuple[str, dict[str, Any], bool]:
    """``(entity_id, fields, created)`` of a between-turn mutation. A create with no ``entity_id`` gets the
    deterministic id ``<channel>-after-<after_turn>-<index>``."""
    mutation: Mapping[str, Any] = entry.value
    if 'create' not in mutation:
        return (*item_write(mutation), False)
    created: Mapping[str, Any] = mutation['create']
    if isinstance(created.get('entity_id'), str):
        return (*item_write(created), True)
    return f'{entry.channel.name}-after-{entry.after_turn}-{entry.index}', dict(created), True


__all__ = [
    'BETWEEN_TURN_ACTIONS_KEY',
    'CHANNEL_INPUTS_KEY',
    'EMPTY_PLAN',
    'HARNESS_UNDECLARED_CHANNEL_KIND',
    'HOST_MESSAGE_ROLES',
    'TURN_INPUTS_KEY',
    'AppliedChannelInput',
    'ChannelPlan',
    'HarnessUndeclaredChannel',
    'PlannedChannelInput',
    'item_write',
    'mutation_write',
    'parse_channel_plan',
    'scenario_plants_channels',
]
