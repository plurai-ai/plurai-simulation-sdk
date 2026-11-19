"""``agent.io.channels``: the agent's declared NON-MESSAGE input channels, and the CLOSED vocabulary of their kinds.

An agent takes input beyond the user's message text: a file the user attaches BY ID into a store the agent reads, a
per-turn payload field the host fills, a change the user makes in the UI between turns, a message the host adds to the
history. The scan writes one entry per such channel under ``agent.io.channels`` in ``plurai.config.yaml``
(``{name, kind, field, source, handover}``); message content parts stay in ``agent.io.input``.

Scenario generation reads the declaration to deal each channel into the scenarios that use it, and the authoring gate
checks every scheduled use against it. The SDK plants each kind.

Served to clients as the ``input_channel_kinds`` contract block; the SDK and the plugin vendor it rather than retype
it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, get_args

#: ``file_by_reference``: an id into a store the agent reads; the harness writes the item into the mocked store and
#: passes the id in the channel's ``field``. ``structured_context``: a per-turn host payload field.
#: ``between_turn_action``: a user/UI change applied to the world between turns. ``host_injected_message``: a message
#: the host adds to the history.
ChannelKind = Literal['file_by_reference', 'structured_context', 'between_turn_action', 'host_injected_message']

CHANNEL_KINDS: tuple[ChannelKind, ...] = get_args(ChannelKind)

#: Which scenario schedule each kind is used through: ``turn_inputs`` ride a user turn, ``between_turn_actions`` change
#: the world after the agent's reply. Keyed on the kind, so a new kind is one row and no rule changes.
SCHEDULE_OF_KIND: Mapping[ChannelKind, Literal['turn_inputs', 'between_turn_actions']] = MappingProxyType(
    {
        'file_by_reference': 'turn_inputs',
        'structured_context': 'turn_inputs',
        'between_turn_action': 'between_turn_actions',
        'host_injected_message': 'turn_inputs',
    }
)


#: The ROOT ``plurai.config.yaml`` key the channels are declared under: its name contains a literal dot.
AGENT_IO_KEY = 'agent.io'
#: The keys a CONFIG channel entry must carry as non-blank strings.
_CONFIG_CHANNEL_REQUIRED = ('name', 'kind', 'field', 'source')
#: The ``agent.io`` sub-keys the SDK reads (``plurai.agent_io._KNOWN_FIELDS``); any other is a field it ignores.
AGENT_IO_FIELDS = ('input', 'may_open', 'opens', 'choice_text', 'choice_text_source', 'channels')


class MalformedInputChannelError(ValueError):
    """A declared channel or ``agent.io`` declaration that cannot be read; the cases are listed on :func:`declared_channels`."""


@dataclass(frozen=True)
class InputChannel:
    """One declared channel: its ``name`` (what a scenario schedules), its ``kind`` and the payload ``field`` it fills."""

    name: str
    kind: ChannelKind
    field: str = ''

    def to_wire(self) -> dict[str, str]:
        return {'name': self.name, 'kind': self.kind, 'field': self.field}


def channel_from_wire(raw: Any) -> InputChannel:
    """One ``{name, kind, field?}`` mapping as an :class:`InputChannel`. RAISES :class:`MalformedInputChannelError`."""
    if not isinstance(raw, Mapping):
        raise MalformedInputChannelError(f'carried input_channels entry {raw!r} is not a mapping with `name` and `kind`')
    name = str(raw.get('name') or '').strip()
    kind = str(raw.get('kind') or '').strip().lower()
    if not name:
        raise MalformedInputChannelError(f'carried input_channels entry {dict(raw)!r} names no channel')
    if kind not in CHANNEL_KINDS:
        raise MalformedInputChannelError(f'carried input_channels entry `{name}`: kind {raw.get("kind")!r} is not one of {list(CHANNEL_KINDS)}')
    return InputChannel(name=name, kind=kind, field=str(raw.get('field') or '').strip())  # type: ignore[arg-type]  # narrowed by the membership test


def channels_from_wire(entries: Any) -> tuple[InputChannel, ...]:
    """The job's CARRIED channel copy (:meth:`InputChannel.to_wire` mappings, as held in job inputs, graph state and the
    redrive contract), validated, in order. Lenient on purpose: that copy has no ``source``/``handover``, and its kind
    is lowercased. The strict read of the customer's config is :func:`declared_channels`.
    """
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise MalformedInputChannelError(f'carried input_channels is a {type(entries).__name__}, not a list of channel entries')
    out: list[InputChannel] = []
    for entry in entries:
        channel = channel_from_wire(entry)
        if any(seen.name == channel.name for seen in out):
            raise MalformedInputChannelError(f'carried input_channels names `{channel.name}` twice')
        out.append(channel)
    return tuple(out)


def declared_channels(plurai_config_yaml: str | None) -> tuple[InputChannel, ...]:
    """Every ``agent.io.channels`` entry of a ``plurai.config.yaml``, in config order: the STRICT config read.

    Read from the ROOT key ``agent.io`` (a literal dot, beside ``workspace:``), the spelling the SDK requires. ``()`` for
    an absent, blank or comments-only config, a config with no ``agent.io`` (or ``agent.io:`` null), and an ``agent.io``
    with no ``channels:``.

    RAISES :class:`MalformedInputChannelError` (each message names its fix), never reading the config as declaring
    none, for:

    * a config that does not parse, or whose top level is not a mapping;
    * the nested ``agent: {io: …}`` form, including a document that also carries the root ``agent.io:``;
    * an ``agent.io`` that is not a mapping;
    * a ``channels`` that is not a list, or an entry that is not a mapping, lacks a non-blank string
      ``name``/``kind``/``field``/``source``, has a ``kind`` outside :data:`CHANNEL_KINDS` (case-sensitive), a
      non-string ``handover``, or repeats a name.

    Values are compared after trimming whitespace. An ``agent.io`` sub-key the SDK does not read is not refused here:
    :func:`agent_io_field_warnings` names it.
    """
    io = _agent_io_block(plurai_config_yaml)
    return _config_channels(io) if io is not None else ()


def agent_io_field_warnings(plurai_config_yaml: str | None) -> list[str]:
    """One warning per ``agent.io`` sub-key the SDK does not read (it records them as ``unknown_field`` and ignores
    them), naming the known ones: a misspelled ``channels:`` declares nothing. Never raises: an unreadable config is
    :func:`declared_channels`' to refuse.
    """
    if not plurai_config_yaml or AGENT_IO_KEY not in plurai_config_yaml:
        return []
    try:
        io = _agent_io_block(plurai_config_yaml)
    except MalformedInputChannelError:
        return []
    known = ', '.join(AGENT_IO_FIELDS)
    return [
        f'`{AGENT_IO_KEY}.{key}` is not a field the SDK reads (known: {known}); a misspelled `channels:` declares nothing.'
        for key in sorted(set(map(str, io or {})) - set(AGENT_IO_FIELDS))
    ]


def _agent_io_block(plurai_config_yaml: str | None) -> dict[str, Any] | None:
    """The config's ROOT ``agent.io`` mapping, or ``None`` when it declares none. RAISES as :func:`declared_channels`."""
    import yaml  # noqa: PLC0415

    from plurai.scan_contract._safe_yaml import safe_yaml_load  # noqa: PLC0415

    if not plurai_config_yaml or not plurai_config_yaml.strip():
        return None
    try:
        raw = safe_yaml_load(plurai_config_yaml)
    except yaml.YAMLError as exc:
        raise MalformedInputChannelError(
            f'plurai.config.yaml does not parse, so its {AGENT_IO_KEY} channels cannot be read: {exc}. Fix the YAML syntax at that line/column.'
        ) from exc
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MalformedInputChannelError(
            f'plurai.config.yaml must be a YAML mapping to declare anything; got {type(raw).__name__}. Write it as `key: value` lines at the root.'
        )
    agent = raw.get('agent')
    if isinstance(agent, Mapping) and 'io' in agent:
        raise MalformedInputChannelError(
            f'`agent: {{io: …}}` declares nothing: only the ROOT key `{AGENT_IO_KEY}:` (a literal dot, beside `workspace:`) is read. '
            f'Move the block to the root of plurai.config.yaml (or delete it, if a root `{AGENT_IO_KEY}:` already holds the declaration).'
        )
    io = raw.get(AGENT_IO_KEY)
    if io is None:
        return None
    if not isinstance(io, dict):
        raise MalformedInputChannelError(
            f'`{AGENT_IO_KEY}` must be a mapping declaring `channels:`, `input:` and/or `may_open:`, got {type(io).__name__}. '
            'Omit the key when the agent declares none of them.'
        )
    return io


def _config_channels(io: Mapping[str, Any]) -> tuple[InputChannel, ...]:
    """``agent.io.channels`` held to the SDK's own read (``plurai.agent_io._parse_channels``). An unknown key in an entry
    is ignored (the SDK reports it as a defect, never refuses it)."""
    if 'channels' not in io:
        return ()
    declared = io['channels']
    entry_shape = '{name, kind, field, source, handover}'
    if not isinstance(declared, list):
        raise MalformedInputChannelError(
            f'`{AGENT_IO_KEY}.channels` must be a list of {entry_shape} entries, got {type(declared).__name__}. '
            'Write one `- {name: …, kind: …, field: …, source: …}` item per channel, or drop the key.'
        )
    out: list[InputChannel] = []
    for index, entry in enumerate(declared):
        key = f'{AGENT_IO_KEY}.channels[{index}]'
        if not isinstance(entry, Mapping):
            raise MalformedInputChannelError(f'`{key}` must be a mapping {entry_shape}, got {type(entry).__name__}. Rewrite it as one.')
        for required in _CONFIG_CHANNEL_REQUIRED:
            value = entry.get(required)
            if not isinstance(value, str) or not value.strip():
                raise MalformedInputChannelError(f'`{key}.{required}` must be a non-blank string, got {value!r}. Set it (re-run the scan).')
        kind = entry['kind'].strip()
        if kind not in CHANNEL_KINDS:
            raise MalformedInputChannelError(f'`{key}.kind` must be one of {list(CHANNEL_KINDS)}, got {kind!r}. Use one of them, spelled exactly.')
        handover = entry.get('handover')
        if handover is not None and not isinstance(handover, str):
            raise MalformedInputChannelError(f'`{key}.handover` must be a string when set, got {handover!r}. Quote it as text, or drop it.')
        name = entry['name'].strip()
        if any(seen.name == name for seen in out):
            raise MalformedInputChannelError(
                f'`{AGENT_IO_KEY}.channels` declares the channel {name!r} more than once: scenarios address a channel by name. Rename or merge the duplicates.'
            )
        out.append(InputChannel(name=name, kind=kind, field=entry['field'].strip()))  # type: ignore[arg-type]  # narrowed by the membership test
    return tuple(out)
