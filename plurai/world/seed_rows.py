"""A seeded world row as a record of the harness's own store: ``plurai.store_record``.

``seed_state`` param 2 hands the harness the rows the backend seeded, keyed by row id. Each row carries two
ROUTING keys besides its type's fields:

* ``_entity_type``, the row's type (every ``_``-prefixed key is bookkeeping, never a field);
* ``id``, a copy of the row's own id that the backend stamps whenever the type keys by another declared
  ``id_field``, so a by-id reader finds it.

Both stay on the rows the harness is handed, on purpose. Harnesses route on ``_entity_type`` (a
harness-owned store picks the table or the registry by it) and rows a harness applies through
``plurai.world_state`` need it. The backend's landing check (``seed_rows_not_landed``) compares every
non-``_`` key of a seeded row, ``id`` included, against the post-seed read, so a harness whose
``world_snapshot()`` echoes its seeded rows must still hold the ``id`` copy there.

Neither key is a field of the store, though. A store that validates its records (a config schema that
forbids extra keys) refuses a record that carries one: a config built from a row would carry the ``id``
copy into the store's schema and the record would never register.
:func:`store_record` is the one place a harness turns a row into a store record: the row minus both
routing keys, the ``id`` judged against the type's declared schema (``world_state.entity_types`` in the
``plurai.config.yaml`` beside the harness). It never drops a declared field and never guesses: a row with
no ``_entity_type``, or of a type the config does not declare, is refused by name.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from plurai._protocol import HarnessFault
from plurai.ingest.workspace_config import WorkspaceConfigError, bench_config_roots, discover_config_documents

#: The type key every seeded row carries inside its value.
ENTITY_TYPE_KEY = '_entity_type'
#: The backend's by-id copy of a row's own id, stamped when the type keys by another ``id_field``.
ID_COPY_KEY = 'id'

# The harness file the child loaded (``bind_harness``); its config declares the schema.
_HARNESS_PATH: Path | None = None
_DECLARED: dict[Path, dict[str, Any]] = {}


def bind_harness(simulate_path: Path) -> None:
    """Called by the child once it has loaded ``simulate.py``: the schema is read beside that file."""
    global _HARNESS_PATH  # noqa: PLW0603 - one harness per child process
    _HARNESS_PATH = Path(simulate_path).resolve()


def declared_entity_types(simulate_path: Path) -> dict[str, Any]:
    """``world_state.entity_types`` from the config beside ``simulate_path`` (later rungs win per type), read once per path."""
    path = Path(simulate_path).resolve()
    if path not in _DECLARED:
        types: dict[str, Any] = {}
        for root in bench_config_roots(path):
            for config, document in discover_config_documents(root):
                world = document.get('world_state')
                declared = world.get('entity_types') if isinstance(world, Mapping) else None
                if declared is None:
                    continue
                if not isinstance(declared, Mapping):
                    raise WorkspaceConfigError(f'{config}: `world_state.entity_types` is not a mapping of type name -> spec.')
                types.update(declared)
        _DECLARED[path] = types
    return _DECLARED[path]


def _caller_harness() -> Path | None:
    """The file that called :func:`store_record` when no child bound a harness (a smoke run of ``simulate.py``)."""
    frame = sys._getframe(3)  # noqa: SLF001 - caller -> store_record -> _schema -> here
    name = frame.f_globals.get('__file__')
    return Path(name) if isinstance(name, str) else None


def _schema(entity_types: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], str]:
    if entity_types is not None:
        return entity_types, 'the entity_types passed in'
    source = _HARNESS_PATH or _caller_harness()
    if source is None:
        return {}, 'no harness file'
    return declared_entity_types(source), f'world_state.entity_types beside {source}'


def store_record(row: Mapping[str, Any], *, entity_types: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The store record a seeded world row stands for: the row minus its routing keys.

    Drops every ``_``-prefixed key (``_entity_type``) and the backend's ``id`` copy, which is the ``id`` of a row
    whose declared type keys by another ``id_field`` and does not declare ``id`` as a field. Every other key is kept
    as the row holds it. ``entity_types`` defaults to ``world_state.entity_types`` in the config beside the harness.

    Use it wherever a harness builds a record of its own store from a row (``seed_state`` param 2). Keep the row
    itself for routing and for ``world_snapshot()``, which reports each seeded row with both keys.
    """
    if not isinstance(row, Mapping):
        raise HarnessFault(f'plurai.store_record: a world row is a mapping of field -> value, got {type(row).__name__}.')
    entity_type = row.get(ENTITY_TYPE_KEY)
    if not entity_type:
        raise HarnessFault(f'plurai.store_record: the row carries no `{ENTITY_TYPE_KEY}`, so its type schema cannot be read: keys {sorted(map(str, row))}.')
    types, where = _schema(entity_types)
    spec = types.get(entity_type)
    if not isinstance(spec, Mapping):
        raise HarnessFault(
            f'plurai.store_record: entity type {entity_type!r} is not declared in {where}, so whether its `id` is a field or the '
            "backend's routing copy cannot be told. Declare the type (with its `id_field` and `fields`) under world_state.entity_types."
        )
    fields = spec.get('fields')
    id_field = spec.get('id_field')
    id_is_copy = isinstance(id_field, str) and id_field != ID_COPY_KEY and not (isinstance(fields, Mapping) and ID_COPY_KEY in fields)
    return {key: value for key, value in row.items() if not str(key).startswith('_') and not (id_is_copy and key == ID_COPY_KEY)}


__all__ = ['ENTITY_TYPE_KEY', 'ID_COPY_KEY', 'bind_harness', 'declared_entity_types', 'store_record']
