"""Did the scenario's seed land in the world the harness reads back? — the ONE comparator.

Stdlib-only, so the backend's finalize, the SDK's runner (at Init) and the plugin's setup read-back run the same bytes;
the SDK's copy (``plurai/world/seed_landing.py``) is vendored and sha-pinned. The backend re-exports every name here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

#: The key a snapshot carries when the world was NEVER OBSERVED (the harness does not own the store the agent
#: reads, so nobody took a reading) — the third state beside "observed and populated" and "observed and empty".
#: A marker on the snapshot rather than a bare ``null``, because the artifact store reads only JSON objects.
WORLD_UNOBSERVED_MARKER = '_unobserved'

#: The reserved in-entity key naming the fields of THIS entity whose read RAISED: ``{field: "<ExcClass>: <message>"}``,
#: the field itself left ``None``. Published to clients as the ``world_snapshot_keys`` contract block.
UNREADABLE_FIELDS_KEY = '_unreadable'


def unobserved_world() -> dict[str, Any]:
    """A FRESH third-state snapshot: ``entities`` present and empty, beside the marker. A function so no two
    snapshots share an ``entities`` map."""
    return {'entities': {}, WORLD_UNOBSERVED_MARKER: True}


#: The canonical third-state snapshot as a read-only VALUE, for comparison; :func:`unobserved_world` builds one to
#: hand onward.
UNOBSERVED_WORLD: Mapping[str, Any] = MappingProxyType({**unobserved_world(), 'entities': MappingProxyType({})})


def world_measurable(raw: Any) -> bool:
    """WAS THE WORLD OBSERVED AT ALL? — the ONE predicate every ``final_world_state`` reader asks.

    ``False`` for ``None`` (the producer took no reading: no mock session, or a harness that declares a world it
    owns and exports no ``world_snapshot()``) and for a mapping carrying :data:`WORLD_UNOBSERVED_MARKER`. ``True``
    for every other value, ``{}`` and ``{'entities': {}}`` included: the third state is declared, never inferred.
    """
    if raw is None:
        return False
    return not (isinstance(raw, Mapping) and raw.get(WORLD_UNOBSERVED_MARKER) is True)


def world_malformed(raw: Any) -> bool:
    """WAS SOMETHING RECORDED THAT IS NOT A WORLD? — so a wrong-shape payload is never read as an observed, empty one.

    ``True`` for a non-mapping payload and for a mapping whose ``entities``/``state`` values are all non-mappings
    (a row list). ``False`` for an unobserved world (its own answer) and for a bare ``{}``.
    """
    if not world_measurable(raw):
        return False
    if not isinstance(raw, Mapping):
        return True
    recorded = [raw[key] for key in ('entities', 'state') if key in raw]
    return bool(recorded) and not any(isinstance(value, Mapping) for value in recorded)


def normalize_final_world_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    """A world snapshot as ``{entities: {entity_id: {...}}, ...}``.

    The mock's ``{session_id, state: {...}}`` envelope is unwrapped (``session_id`` and ``_``-prefixed markers
    kept); an unobserved world becomes :func:`unobserved_world`; a mock error envelope becomes a ``_finalize_error``
    crash marker, so finalize fails instead of scoring an empty world.
    """
    if not world_measurable(raw):
        return unobserved_world()
    if not isinstance(raw, dict):
        # Malformed, not unobserved: the producer recorded something of the wrong shape.
        return {'entities': {}}
    if 'entities' in raw and isinstance(raw['entities'], dict):
        return raw
    if 'state' in raw and isinstance(raw['state'], dict):
        normalized: dict[str, Any] = {'entities': raw['state']}
        if 'session_id' in raw:
            normalized['session_id'] = raw['session_id']
        for k, v in raw.items():
            if k.startswith('_'):
                normalized[k] = v
        return normalized
    # A mock error envelope ({error, status_code}, no state/entities): the read failed.
    if 'error' in raw and 'entities' not in raw and 'state' not in raw:
        status = raw.get('status_code')
        detail = f'HTTP {status}: {raw["error"]}' if status is not None else str(raw['error'])
        return {'entities': {}, '_finalize_error': f'mock read failed at finalize ({detail})'}
    return raw


#: Which world :func:`run_start_world` took: the post-seed read, the scenario seed because the client
#: reported no read, or the scenario seed because the reported read produced no world.
RunStartBasis = Literal['post_seed_read', 'scenario_seed', 'scenario_seed_fallback']


def run_start_basis(post_seed_world_state: Any) -> RunStartBasis:
    """The :data:`RunStartBasis` of a post-seed read, told by its TYPE (see :func:`run_start_world`)."""
    if post_seed_world_state is None:
        return 'scenario_seed'
    world = normalize_final_world_state(post_seed_world_state)
    read = world_measurable(world) and not world_malformed(world) and not world_state_has_crash_marker(world)
    return 'post_seed_read' if read else 'scenario_seed_fallback'


def seed_landing_gaps(*, post_seed_world_state: Any, scenario_seed: Any) -> list[tuple[str, str | None]]:
    """``(entity_id, None)`` for each seeded row a post-seed read lacks, ``(entity_id, field)`` for each seeded field it
    holds differently (:func:`_seed_field_landed`). Empty with no read that produced a world, or no seed."""
    if run_start_basis(post_seed_world_state) != 'post_seed_read' or scenario_seed is None:
        return []
    declared = normalize_final_world_state(scenario_seed).get('entities')
    read = normalize_final_world_state(post_seed_world_state).get('entities') or {}
    if not isinstance(declared, Mapping):
        return []
    gaps: list[tuple[str, str | None]] = []
    for entity_id, row in sorted(declared.items(), key=lambda item: str(item[0])):
        if entity_id not in read:
            gaps.append((str(entity_id), None))
            continue
        got = read[entity_id]
        if isinstance(row, Mapping):
            gaps.extend(
                (str(entity_id), str(field)) for field, value in row.items() if not str(field).startswith('_') and not _seed_field_landed(field, value, got)
            )
    return gaps


#: The fields a FILE row's reader computes from the file's bytes (the SDK's ``file_row``): the file's own facts,
#: never compared against a seeded value.
FILE_ROW_COMPUTED_FIELDS = frozenset({'sha256', 'size_bytes', 'suffix', 'content_type', 'text_preview'})


def _seed_field_landed(field: Any, seeded: Any, got: Any) -> bool:
    """Did the seeded ``field`` of one row land on the read row ``got``? :func:`_seed_value_landed`, plus the FILE-ROW
    rule: on a row with a ``sha256``, seeded ``content`` lands when its UTF-8 SHA-256 matches (and ``content`` is not
    marked unreadable), and :data:`FILE_ROW_COMPUTED_FIELDS` are not compared."""
    if not isinstance(got, Mapping):
        return _seed_value_landed(seeded, None)
    digest = got.get('sha256')
    if isinstance(digest, str):
        if field in FILE_ROW_COMPUTED_FIELDS:
            return True
        if field == 'content' and isinstance(seeded, str):
            unreadable = got.get(UNREADABLE_FIELDS_KEY)
            if isinstance(unreadable, Mapping) and 'content' in unreadable:
                return False
            return _seed_value_landed(seeded, got.get('content')) or hashlib.sha256(seeded.encode('utf-8')).hexdigest() == digest.lower()
    return _seed_value_landed(seeded, got.get(field))


def seed_rows_not_landed(*, post_seed_world_state: Any, scenario_seed: Any) -> list[str]:
    """:func:`seed_landing_gaps` as subjects: ``<id>`` for a row the read lacks, ``<id>.<field>`` for a field it holds differently."""
    return [
        entity_id if field is None else f'{entity_id}.{field}'
        for entity_id, field in seed_landing_gaps(post_seed_world_state=post_seed_world_state, scenario_seed=scenario_seed)
    ]


def _blank(value: Any) -> bool:
    """``''``, ``None`` and an empty container are one value on the wire: a seed that wrote nothing."""
    return value is None or value == '' or (isinstance(value, (Mapping, list, tuple)) and not value)


def _instant(value: Any) -> datetime | None:
    """An ISO-8601 date-time string as an instant, else ``None``."""
    if not isinstance(value, str) or len(value) < 10 or value[4:5] != '-':
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def _seed_value_landed(seeded: Any, read: Any) -> bool:
    """Does the read hold the seeded value, up to representation?

    A mapping lands when every seeded key does (extra keys in the read are the store's own columns); a list when it
    has the same length and each item lands; blank (:func:`_blank`) matches blank; numbers compare numerically;
    ISO-8601 strings compare as instants (``-04:00`` and ``Z`` name one moment); ``_``-prefixed keys are bookkeeping."""
    if _blank(seeded):
        return _blank(read)
    if isinstance(seeded, Mapping):
        return isinstance(read, Mapping) and all(_seed_value_landed(v, read.get(k)) for k, v in seeded.items() if not str(k).startswith('_'))
    if isinstance(seeded, (list, tuple)):
        return isinstance(read, (list, tuple)) and len(seeded) == len(read) and all(_seed_value_landed(a, b) for a, b in zip(seeded, read, strict=True))
    if isinstance(seeded, bool) or isinstance(read, bool):
        return seeded == read
    if isinstance(seeded, (int, float)) and isinstance(read, (int, float)):
        return float(seeded) == float(read)
    if (a := _instant(seeded)) is not None and (b := _instant(read)) is not None:
        if (a.tzinfo is None) != (b.tzinfo is None):
            return a.replace(tzinfo=None) == b.replace(tzinfo=None)
        return a == b
    return bool(seeded == read)


# Crash markers whose presence means the snapshot is not a world: ``_finalize_error`` (the read raised) and the
# reserved ``_partial``.
_WORLD_ERROR_MARKERS = ('_finalize_error', '_partial')


#: Beside ``_finalize_error``: ``True`` when the read failed because the BACKEND did not answer (a timeout, a dropped
#: connection, a 5xx), typed by the SDK child off the error's own fields. Not the harness failing to read its world,
#: so it never voids a scenario (:attr:`SeedReadback.voids_scenario`).
BACKEND_TRANSPORT_FAILURE_KEY = '_backend_transport_failure'


def world_state_has_crash_marker(normalized: dict[str, Any]) -> bool:
    """True when the snapshot carries a crash marker (a failed read, not an empty world)."""
    return any(k in normalized for k in _WORLD_ERROR_MARKERS)


#: What one read-back says. ``landed``: every seeded row and field read back as seeded. ``gaps``: some did not.
#: ``read_failed``: the read itself failed (a crash marker or a malformed payload), so nothing can be compared.
#: ``no_read_path``: no read was taken (the harness declares a world it does not read back) — never a pass.
SeedReadbackVerdict = Literal['landed', 'gaps', 'read_failed', 'no_read_path']


@dataclass(frozen=True)
class SeedReadback:
    """The seed-vs-read verdict of one post-seed read, split so it names the fix.

    ``rows_missing``: ``<id>`` the read lacks. ``fields_not_read``: ``<id>.<field>`` the read row has no such field
    (the read path never returns it: widen ``world_snapshot``). ``fields_changed``: ``<id>.<field>`` the read holds
    with another value (``seed_state`` did not apply it). ``read_error``: the crash marker's text when the read failed;
    ``backend_transport``: that failure was the backend not answering (:data:`BACKEND_TRANSPORT_FAILURE_KEY`).
    ``subjects`` is exactly :func:`seed_rows_not_landed`'s answer."""

    verdict: SeedReadbackVerdict
    rows_missing: tuple[str, ...] = ()
    fields_not_read: tuple[str, ...] = ()
    fields_changed: tuple[str, ...] = ()
    read_error: str | None = None
    backend_transport: bool = False
    rows_seeded: int = 0
    fields_seeded: int = 0
    subjects: tuple[str, ...] = ()

    @property
    def voids_scenario(self) -> bool:
        """Whether this read-back VOIDS its scenario: ``gaps``, or a harness ``read_failed`` (not a backend transport
        failure) on a scenario that seeds at least one row. The backend's finalize voids on it (``undelivered_seed``) and the
        SDK's runner skips the scenario at Init."""
        return self.verdict == 'gaps' or (self.verdict == 'read_failed' and self.rows_seeded > 0 and not self.backend_transport)

    def line(self) -> str:
        """The one operator sentence every reader prints."""
        if self.verdict == 'landed':
            return f'seed read back as seeded: {self.rows_seeded} row(s), {self.fields_seeded} field(s)'
        if self.verdict == 'no_read_path':
            return 'NO VERDICT: no read path. The harness declares a world it does not read back, so nothing can be compared'
        if self.verdict == 'read_failed':
            cause = ' (the backend did not answer)' if self.backend_transport else ''
            return f'the world could not be read after seed_state{cause}: {self.read_error}'
        parts = []
        if self.rows_missing:
            parts.append(f'rows the read lacks: {", ".join(self.rows_missing)}')
        if self.fields_not_read:
            parts.append(f'fields the read never returns: {", ".join(self.fields_not_read)}')
        if self.fields_changed:
            parts.append(f'fields read back changed: {", ".join(self.fields_changed)}')
        return 'seed not read back as seeded. ' + '; '.join(parts)


def seed_readback(*, post_seed_world_state: Any, scenario_seed: Any) -> SeedReadback:
    """Classify one post-seed read against the scenario's seed: the single verdict every reader prints from.

    ``no_read_path`` for no read or an unobserved one; ``read_failed`` for a crash marker or a malformed payload;
    otherwise :func:`seed_landing_gaps`, split into rows missing / fields the read lacks / fields changed."""
    if post_seed_world_state is None or not world_measurable(post_seed_world_state):
        return SeedReadback(verdict='no_read_path')
    declared = normalize_final_world_state(scenario_seed).get('entities') if scenario_seed is not None else None
    declared = declared if isinstance(declared, Mapping) else {}
    fields = sum(1 for row in declared.values() if isinstance(row, Mapping) for f in row if not str(f).startswith('_'))
    if run_start_basis(post_seed_world_state) != 'post_seed_read':
        world = normalize_final_world_state(post_seed_world_state)
        error = next((str(world[k]) for k in _WORLD_ERROR_MARKERS if k in world), None)
        return SeedReadback(
            verdict='read_failed',
            read_error=error or f'the read is not a world ({type(post_seed_world_state).__name__})',
            backend_transport=world.get(BACKEND_TRANSPORT_FAILURE_KEY) is True,
            rows_seeded=len(declared),
            fields_seeded=fields,
        )
    read = normalize_final_world_state(post_seed_world_state).get('entities') or {}
    gaps = seed_landing_gaps(post_seed_world_state=post_seed_world_state, scenario_seed=scenario_seed)
    missing: list[str] = []
    not_read: list[str] = []
    changed: list[str] = []
    for entity_id, field in gaps:
        if field is None:
            missing.append(entity_id)
            continue
        got = read.get(entity_id)
        (changed if isinstance(got, Mapping) and field in got else not_read).append(f'{entity_id}.{field}')
    return SeedReadback(
        verdict='gaps' if gaps else 'landed',
        rows_missing=tuple(missing),
        fields_not_read=tuple(not_read),
        fields_changed=tuple(changed),
        rows_seeded=len(declared),
        fields_seeded=fields,
        subjects=tuple(entity_id if field is None else f'{entity_id}.{field}' for entity_id, field in gaps),
    )


#: The ``undelivered_seed`` subject of a scenario voided because its post-seed read FAILED (:func:`seed_read_failure`):
#: no seeded row can be named, since nothing was read.
SEED_READ_FAILED_SUBJECT = 'post-seed world read'


def seed_read_failure(*, post_seed_world_state: Any, scenario_seed: Any) -> str | None:
    """The read error that voids the scenario (a harness ``read_failed`` on a scenario that seeds a row), else ``None``: the
    other half of the seed void, beside :func:`seed_rows_not_landed`."""
    readback = seed_readback(post_seed_world_state=post_seed_world_state, scenario_seed=scenario_seed)
    return readback.read_error if readback.verdict == 'read_failed' and readback.voids_scenario else None
