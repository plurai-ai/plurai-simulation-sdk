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


class MalformedInputChannelError(ValueError):
    """A declared ``agent.io.channels`` entry that is not ``{name, kind}`` with a contract kind, or a duplicate name."""


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
        raise MalformedInputChannelError(f'agent.io.channels entry {raw!r} is not a mapping with `name` and `kind`')
    name = str(raw.get('name') or '').strip()
    kind = str(raw.get('kind') or '').strip().lower()
    if not name:
        raise MalformedInputChannelError(f'agent.io.channels entry {dict(raw)!r} names no channel')
    if kind not in CHANNEL_KINDS:
        raise MalformedInputChannelError(f'agent.io.channels.{name}.kind {raw.get("kind")!r} is not one of {list(CHANNEL_KINDS)}')
    return InputChannel(name=name, kind=kind, field=str(raw.get('field') or '').strip())  # type: ignore[arg-type]  # narrowed by the membership test


def channels_from_wire(entries: Any) -> tuple[InputChannel, ...]:
    """A list of channel mappings (the job's carried copy, or the config's block), validated, in order."""
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise MalformedInputChannelError(f'agent.io.channels is a {type(entries).__name__}, not a list of channel entries')
    out: list[InputChannel] = []
    for entry in entries:
        channel = channel_from_wire(entry)
        if any(seen.name == channel.name for seen in out):
            raise MalformedInputChannelError(f'agent.io.channels declares `{channel.name}` twice')
        out.append(channel)
    return tuple(out)


def declared_channels(plurai_config_yaml: str | None) -> tuple[InputChannel, ...]:
    """Every ``agent.io.channels`` entry, in config order. Empty for an absent or unparseable config, or one that
    declares none.

    RAISES :class:`MalformedInputChannelError` for an entry outside the contract: a channel no scenario could be
    checked against is refused, never dropped.
    """
    import yaml  # noqa: PLC0415

    from plurai.scan_contract._safe_yaml import safe_yaml_load  # noqa: PLC0415

    if not plurai_config_yaml or not plurai_config_yaml.strip():
        return ()
    try:
        raw = safe_yaml_load(plurai_config_yaml)
    except yaml.YAMLError:
        return ()
    agent = raw.get('agent') if isinstance(raw, dict) else None
    io = agent.get('io') if isinstance(agent, dict) else None
    return channels_from_wire(io.get('channels') if isinstance(io, dict) else None)
