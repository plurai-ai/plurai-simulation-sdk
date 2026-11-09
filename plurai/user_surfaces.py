"""Which entity types ``plurai.config.yaml`` says the simulated user can see.

``world_state.user_surfaces`` lists the product's views of the world a person sees: a screen in
the app, an inbox, a phone. The scan writes it and the customer reviews it. The backend owns the type
and every semantic rule; the plugin's ``verify_config.py``
applies the same rules at authoring time. The SDK reads only which entity TYPES the surfaces need
(the type prefix of each ``rows`` path, each ``joins[].target`` and each action's ``values_from``
path), because that decides what the child captures after each agent turn: only rows a surface can
show leave the process.

* ABSENT key: :func:`declared_surface_types` returns ``None`` types. Nothing is captured, and each
  turn tells the server ``not_declared``, so an undeclared bench keeps the default user view.
* ``user_surfaces: []``: ``frozenset()``, a declaration that nothing reaches her.
* a non-empty list: the surfaces' row types, their join target types and their ``values_from`` types.
* MALFORMED as a whole (not a list, an entry that is not a mapping, an entry with no ``name``, a name
  listed twice): :class:`UserSurfacesConfigError`. A broken declaration read as "undeclared" would put
  the user's view back where this key exists to take it from.
* ONE ENTRY whose ``rows`` / ``joins`` / ``values_from`` names no entity type: that entry is dropped and
  named as a :class:`SurfaceDefect`; the other surfaces are still captured. The surface ``kind`` is
  the backend's to validate (it owns ``SurfaceKind`` and checks every scenario's surfaces when a thread
  opens), so the SDK does not read it.

Two configs beside one harness that both declare and disagree are an error, as for
``render_actions`` and ``world_state.owner``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from plurai.ingest.workspace_config import WorkspaceConfigError, bench_config_roots, discover_config_documents

#: The group key and the key under it. The backend uses the same vocabulary.
WORLD_STATE_KEY: Final = 'world_state'
USER_SURFACES_KEY: Final = 'user_surfaces'
#: The entity type a ``rows`` path starts with: ``<type>`` or ``<type>[<selector>].<list_field>``.
_ROWS_TYPE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


class UserSurfacesConfigError(WorkspaceConfigError):
    """``world_state.user_surfaces`` exists but cannot testify to a declaration (fail closed)."""


@dataclass(frozen=True, slots=True)
class SurfaceDefect:
    """One surface entry dropped because a path in it names no entity type."""

    index: int
    name: str
    #: ``rows``, ``joins[].target``, ``joins`` (not a list), ``actions`` (not a list) or ``actions[].values_from``.
    code: str
    value: str
    source: Path | None


def declared_surface_types(document: Mapping[str, Any], *, source: Path | None = None) -> tuple[frozenset[str] | None, tuple[SurfaceDefect, ...]]:
    """``(types, defects)``: ``types`` is ``None`` when the key is absent, else the entity types the readable
    surfaces show (``frozenset()`` for ``[]``); ``defects`` names each entry dropped for an untyped path.

    Raises :class:`UserSurfacesConfigError` when the block as a whole cannot be read, naming ``source``.
    """
    where = f'{source}: ' if source is not None else ''
    group = document.get(WORLD_STATE_KEY)
    if group is None:
        return None, ()
    if not isinstance(group, Mapping):
        raise UserSurfacesConfigError(f'{where}`{WORLD_STATE_KEY}` must be a mapping, got {type(group).__name__}.')
    if USER_SURFACES_KEY not in group:
        return None, ()
    label = f'{WORLD_STATE_KEY}.{USER_SURFACES_KEY}'
    raw = group[USER_SURFACES_KEY]
    if not isinstance(raw, list):
        raise UserSurfacesConfigError(
            f'{where}`{label}` must be a list of the views of the world the user sees, got {type(raw).__name__}. '
            f'Omit the key when it is not declared; `{USER_SURFACES_KEY}: []` declares that nothing reaches the user.'
        )
    names: set[str] = set()
    types: set[str] = set()
    defects: list[SurfaceDefect] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise UserSurfacesConfigError(f'{where}`{label}[{index}]` must be a mapping, got {type(entry).__name__}.')
        name = entry.get('name')
        if not isinstance(name, str) or not name.strip():
            raise UserSurfacesConfigError(f'{where}`{label}[{index}]` has no `name`.')
        if name in names:
            raise UserSurfacesConfigError(f'{where}`{label}` lists {name!r} more than once.')
        names.add(name)
        rows = entry.get('rows')
        match = _ROWS_TYPE.match(rows) if isinstance(rows, str) else None
        if match is None:
            defects.append(SurfaceDefect(index, name, 'rows', repr(rows), source))
            continue
        linked = _linked_types(entry)
        if isinstance(linked, tuple):
            defects.append(SurfaceDefect(index, name, linked[0], linked[1], source))
            continue
        types.add(match.group(0))
        types |= linked
    return frozenset(types), tuple(defects)


def _linked_types(entry: Mapping[str, Any]) -> set[str] | tuple[str, str]:
    """The types a surface READS beside its rows: each ``joins[].target`` and each action's
    ``values_from`` path — or ``(code, value)`` for the first one that names no entity type."""
    paths: list[tuple[str, Any]] = []
    joins = entry.get('joins') or []
    if not isinstance(joins, list):
        return 'joins', repr(joins)
    paths += [('joins[].target', join.get('target') if isinstance(join, Mapping) else None) for join in joins]
    actions = entry.get('actions') or []
    if not isinstance(actions, list):
        return 'actions', repr(actions)
    for action in actions:
        values_from = action.get('values_from') if isinstance(action, Mapping) else None
        if values_from is None:
            continue
        if not isinstance(values_from, Mapping):
            return 'actions[].values_from', repr(values_from)
        paths += [('actions[].values_from', path) for path in values_from.values()]
    types: set[str] = set()
    for key, path in paths:
        match = _ROWS_TYPE.match(path) if isinstance(path, str) else None
        if match is None:
            return key, repr(path)
        types.add(match.group(0))
    return types


def discover_surface_types(simulate_path: Path) -> tuple[frozenset[str] | None, tuple[SurfaceDefect, ...]]:
    """The declaration beside this harness, read at the two rungs every declaration is read at.

    ``(None, ())`` when no config declares the key. Two configs whose readable surfaces disagree raise;
    every file's dropped entries are returned.
    """
    found: list[tuple[frozenset[str], Path]] = []
    defects: list[SurfaceDefect] = []
    seen: set[Path] = set()
    for root in bench_config_roots(simulate_path):
        for path, document in discover_config_documents(root, error=UserSurfacesConfigError):
            if (resolved := path.resolve()) in seen:
                continue
            seen.add(resolved)
            declared, dropped = declared_surface_types(document, source=path)
            defects += dropped
            if declared is not None:
                found.append((declared, path))
    if not found:
        return None, tuple(defects)
    if len({declared for declared, _ in found}) > 1:
        conflict = ', '.join(f'{path}: {sorted(declared)}' for declared, path in found)
        raise UserSurfacesConfigError(f'`{WORLD_STATE_KEY}.{USER_SURFACES_KEY}` is declared more than once and the declarations disagree ({conflict}).')
    return found[0][0], tuple(defects)


__all__ = [
    'USER_SURFACES_KEY',
    'SurfaceDefect',
    'UserSurfacesConfigError',
    'declared_surface_types',
    'discover_surface_types',
]
