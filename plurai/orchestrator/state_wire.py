"""``agent_state`` projected for the wire: what ``/user-turn`` and ``/finalize`` carry of the agent's state.

The child returns the agent's whole non-message state every turn, and a stateful agent keeps the
scenario's ``agent_context`` in it. Two rules bound what that costs on the wire:

* **A served document goes back as its reference.** the backend binds a document over its inline cap as
  a ``<scheme>:///documents/…`` reference; the client folds it into a ``data:`` URI for the agent,
  which cannot reach the bearer-guarded ``/documents`` route. A state leaf that is byte-identical
  to that fold is the backend's own document, so the reference is sent in its place. Exact equality is
  sound because a ``str`` survives the child's per-key JSON round trip and the IPC encoding
  unchanged; a value the agent altered in any way is not the served document and stays.
* **A string leaf over :data:`AGENT_STATE_LEAF_CAP_BYTES` is replaced by a descriptor**: its size in bytes, its
  sha256 and, for a ``data:`` URI, its media type. A cut value is never sent: half a ``data:`` URI is
  not a smaller document, and a descriptor that carries a scheme would be dispatched on by readers
  that branch on ``data:`` and ``<scheme>:///``.

The projection is the ONE value both carriers record: the request body, and ``transcript.json``'s
``turns[].agent_state``, so live grading and a re-score of the artifact read the same thing.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

#: A string leaf whose UTF-8 encoding is longer than this many bytes is sent as a descriptor. The request body
#: is UTF-8 JSON and the proxy limit is in bytes, so the cap is too: a single leaf at the cap stays under the
#: request body a fronting proxy accepts (``client.PROXY_BODY_LIMIT_BYTES``), so no one value can fail a turn on its own; several large
#: leaves can add up past it, which the client's body-size WARNING reports.
AGENT_STATE_LEAF_CAP_BYTES: Final = 900_000

#: What every descriptor starts with. A resume seed leaf that starts with it holds no value.
DESCRIPTOR_PREFIX: Final = '[plurai: agent_state value not sent'

_DATA_URI_MEDIA_TYPE = re.compile(r'^data:(?P<mime>[\w.+-]+/[\w.+-]+)[;,]', re.IGNORECASE)


@dataclass(frozen=True)
class WireAgentState:
    """The projected state and the key paths each rule rewrote, in walk order."""

    state: dict[str, Any] | None
    referenced: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()


def wire_agent_state(state: Mapping[str, Any] | None, *, served: Mapping[str, str]) -> WireAgentState:
    """``state`` with every served-document echo replaced by its reference and every oversized leaf by a descriptor.

    ``served`` maps a folded agent-context value to the ``<scheme>:///documents/…`` reference it was
    folded from. ``None`` stays ``None`` and ``{}`` stays ``{}``: the two mean different things to
    the server and the transcript. Keys and non-string leaves are never rewritten, and ``state`` is
    not mutated.
    """
    if state is None:
        return WireAgentState(state=None)
    referenced: list[str] = []
    omitted: list[str] = []

    def _project(path: str, value: str) -> str:
        reference = served.get(value)
        if reference is not None:
            referenced.append(path)
            return reference
        if len(value.encode('utf-8')) > AGENT_STATE_LEAF_CAP_BYTES:
            omitted.append(path)
            return _descriptor(value)
        return value

    projected = map_string_leaves(state, _project)
    return WireAgentState(state=projected, referenced=tuple(referenced), omitted=tuple(omitted))


def map_string_leaves(value: Any, fn: Callable[[str, str], Any], path: str = '') -> Any:
    """``value`` with every string leaf, at any depth of mappings and lists, replaced by ``fn(path, leaf)``.

    ``path`` spells where a leaf sits (``a.b[2]``). Keys and non-string leaves are never rewritten, and
    ``value`` is not mutated: mappings come back as ``dict`` and lists as ``list``.
    """
    if isinstance(value, str):
        return fn(path, value)
    if isinstance(value, Mapping):
        return {key: map_string_leaves(item, fn, f'{path}.{key}' if path else str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [map_string_leaves(item, fn, f'{path}[{index}]') for index, item in enumerate(value)]
    return value


def _descriptor(value: str) -> str:
    """A one-line stand-in for ``value`` that names it without carrying it, and claims no scheme."""
    encoded = value.encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    match = _DATA_URI_MEDIA_TYPE.match(value)
    kind = f', a {match.group("mime").lower()} data URI' if match else ''
    return f'{DESCRIPTOR_PREFIX}: {len(encoded):,} bytes{kind}, over the {AGENT_STATE_LEAF_CAP_BYTES:,}-byte wire cap; sha256:{digest}]'
