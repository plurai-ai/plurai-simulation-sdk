"""The world's change over one agent turn, as ``/user-turn`` carries it (``UserTurnRequest.turn_world``).

The child captures the world after each agent turn with the same read ``/finalize`` uses, keeps
only the rows of the entity types the declared user surfaces show (:mod:`plurai.user_surfaces`),
and sends what changed since the previous capture. The backend folds the deltas and projects the
user's view of them. :func:`diff_world` computes the same diff as the backend's;
a shared set of test vectors pins the two to each other.

``turn_world`` has three states, and ``{}`` is none of them:

* absent (``None`` on the wire): a client that predates the field;
* ``status: unobservable`` with a typed ``reason``: this client cannot say what changed;
* ``status: observed`` with ``delta`` (and ``seed_delta`` on the first observed report).

``types`` names the entity types the capture covers, on every state.

A delta larger than :data:`TURN_WORLD_MAX_CHARS` is reported ``unobservable`` / ``overflow``,
never truncated: a truncated delta folds into a world that never existed.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any, Final, Literal

#: ``TurnWorld.version``.
TURN_WORLD_VERSION: Final = 1
#: Canonical-JSON size of one delta; above it the turn is ``unobservable`` / ``overflow``.
TURN_WORLD_MAX_CHARS: Final = 250_000
#: The key every world row carries its entity type under.
ENTITY_TYPE_KEY: Final = '_entity_type'

UnobservableReason = Literal['not_declared', 'no_world_route', 'read_failed', 'overflow']


def diff_world(before: Mapping[str, dict[str, Any]], after: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """``{upserted: {id: full row after}, removed: [id, ...]}``: rows added or changed, and ids gone.

    Nothing is sorted: ``upserted`` keeps ``after``'s order and ``removed`` keeps ``before``'s, because
    the order rows first appear in IS the product's stored order, which the server's fold keeps.
    """
    upserted = {entity_id: row for entity_id, row in after.items() if before.get(entity_id) != row}
    removed = [entity_id for entity_id in before if entity_id not in after]
    return {'upserted': upserted, 'removed': removed}


def surface_rows(entities: Mapping[str, Any], types: Collection[str]) -> dict[str, dict[str, Any]]:
    """The rows of ``entities`` whose ``_entity_type`` is one of ``types``: all a capture ever keeps."""
    return {str(entity_id): row for entity_id, row in entities.items() if isinstance(row, dict) and row.get(ENTITY_TYPE_KEY) in types}


def initial_entities(initial_world_state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The entity map of the ``initial_world_state`` a thread starts from (``{entities: {id: row}}``)."""
    entities = initial_world_state.get('entities')
    return entities if isinstance(entities, Mapping) else {}


def delta_chars(delta: Mapping[str, Any]) -> int:
    """The compact-JSON size :data:`TURN_WORLD_MAX_CHARS` is compared with (key order never changes it)."""
    return len(json.dumps(delta, separators=(',', ':'), ensure_ascii=False, default=str))


def observed(*, types: Collection[str], delta: dict[str, Any], seed_delta: dict[str, Any] | None) -> dict[str, Any]:
    """An observed turn. ``seed_delta`` rides only the first observed report of a thread."""
    turn_world: dict[str, Any] = {'version': TURN_WORLD_VERSION, 'status': 'observed', 'types': sorted(types), 'delta': delta}
    if seed_delta is not None:
        turn_world['seed_delta'] = seed_delta
    return turn_world


def unobservable(reason: UnobservableReason, *, types: Collection[str]) -> dict[str, Any]:
    """A turn this client cannot report, and why."""
    return {'version': TURN_WORLD_VERSION, 'status': 'unobservable', 'reason': reason, 'types': sorted(types)}


__all__ = [
    'ENTITY_TYPE_KEY',
    'TURN_WORLD_MAX_CHARS',
    'TURN_WORLD_VERSION',
    'UnobservableReason',
    'delta_chars',
    'diff_world',
    'initial_entities',
    'observed',
    'surface_rows',
    'unobservable',
]
