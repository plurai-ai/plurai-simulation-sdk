"""The scrub map an anonymized trace part carries inside its own bytes.

Under ``anonymized`` posture the scrub rewrites trace values and field names, so an observation the
backend reads from the part may be a pseudonym where the agent produced a real value. The scrub map
tells the backend which declared keys that applies to, so it reads them as unseen and never as a
contradiction. It is a ``scrub_map.json`` archive member, hashed with the part, so one part cannot
arrive with two maps::

    {"version": 1, "keys": [{"key": "tool:lookup_order.returns.field:customer_email", "treatment": "pseudonymised"}]}

Keys are written in the ledger's key grammar over the RAW span and field names the scrub saw: the scrub
runs before any alias exists, and the backend applies its alias table before it reads the map. A key
is listed when any unit's value under it was scrubbed. Every replacement the SDK makes is a keyed hash,
so the treatment is always ``pseudonymised``.

The grammar names a tool's top-level returns fields only, so a scrubbed value at or below a top-level field is
listed under that field (``{"order": {"customer_email": ...}}`` lists ``tool:lookup.returns.field:order``); a
member path is never written. The same holds for args. The error text of a tool span is the tool's ``error_channel``.

This module builds keys; it does not parse them. The declared names are the one input from the agent
descriptions: a key or tool-span name the descriptions already hold keeps its spelling through the
scrub, so the map never has to explain a renamed key. A name the scrub itself rewrote is never listed,
because its raw spelling is the thing the scrub is hiding.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final, Protocol

from plurai.ingest.anonymize import TreeHooks

MEMBER_NAME: Final[str] = 'scrub_map.json'
VERSION: Final[int] = 1
TREATMENT: Final[str] = 'pseudonymised'
#: The backend refuses a map naming more keys than it caps a package's declared keys at.
MAX_KEYS: Final[int] = 10_000

_SEPARATOR_RUN = re.compile(r'[-_ ]+')
_ESCAPES: Final[dict[str, str]] = {'%': '%25', '.': '%2E', ':': '%3A'}
#: Between a segment head and its name: ``args`` joins with ``.``, the others with ``:``.
_SEGMENT_SEPARATORS: Final[dict[str, str]] = {'args': '.', 'field': ':', 'fact': ':'}


class ScrubMapError(ValueError):
    """The map cannot be written faithfully, or the declared names cannot be read."""


class _Scrubber(Protocol):
    def anonymize_value(self, value: str) -> str: ...

    def anonymize_obj(
        self,
        obj: Any,  # noqa: ANN401 - decoded JSON
        *,
        exempt_key: Callable[[str], bool] | None = None,
        hooks: TreeHooks | None = None,
    ) -> Any: ...  # noqa: ANN401


@cache
def fold_name(name: str) -> str:
    """NFKC, casefold, every run of ``-``, ``_`` and space becomes one ``_``, leading and trailing ``_`` go."""
    return _SEPARATOR_RUN.sub('_', unicodedata.normalize('NFKC', name).casefold()).strip('_')


def build_key(kind: str, name: str, *segments: tuple[str, ...]) -> str | None:
    """The ledger key of ``kind`` and ``name`` with ``segments`` (``('returns',)``, ``('error_channel',)``,
    ``('args', name)``, ``('field', path)``, ``('fact', predicate)``), or ``None`` when a name folds to nothing.

    Every key the SDK writes is made here, so the fold, the escapes and the separators have one spelling.
    """
    subject = _escaped(fold_name(name))
    if not subject:
        return None
    parts = [f'{kind}:{subject}']
    for head, *names in segments:
        if not names:
            parts.append(head)
        elif text := _escaped(fold_name(names[0])):
            parts.append(f'{head}{_SEGMENT_SEPARATORS[head]}{text}')
        else:
            return None
    return '.'.join(parts)


def _escaped(folded: str) -> str:
    return ''.join(_ESCAPES.get(char, char) for char in folded)


def fold_names(names: frozenset[str] | set[str] | list[str]) -> frozenset[str]:
    return frozenset(folded for name in names if (folded := fold_name(name)))


def load_declared_names(path: str | Path) -> frozenset[str]:
    """The folded declared names in a JSON file: a list of names, or an object with a ``names`` list."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as err:
        raise ScrubMapError(f'declared names {path} cannot be read ({type(err).__name__}); pass a JSON list of names') from None
    names = data.get('names') if isinstance(data, dict) else data
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ScrubMapError(f'declared names {path} must be a JSON list of strings, or an object with a "names" list of strings')
    return fold_names(names)


class ScrubRecorder:
    """Collects the keys whose values the scrub changed, across every unit of one part."""

    def __init__(self) -> None:
        self._keys: set[str] = set()

    def add(self, key: str) -> None:
        self._keys.add(key)

    def member_bytes(self) -> bytes:
        if len(self._keys) > MAX_KEYS:
            raise ScrubMapError(f'the scrub changed {len(self._keys)} distinct keys, past the {MAX_KEYS} a scrub map may name; narrow the export and re-run')
        body = {'version': VERSION, 'keys': [{'key': key, 'treatment': TREATMENT} for key in sorted(self._keys)]}
        return json.dumps(body, sort_keys=True, separators=(',', ':')).encode('utf-8')


# The two span shapes the trace exports write: a LangSmith run (``run_type`` + ``inputs``/``outputs``/``error``)
# and a Langfuse observation (``type`` + ``input``/``output``/``statusMessage``).
_ARGS_FIELDS: Final[frozenset[str]] = frozenset({'inputs', 'input'})
_RETURNS_FIELDS: Final[frozenset[str]] = frozenset({'outputs', 'output'})
_ERROR_FIELDS: Final[frozenset[str]] = frozenset({'error', 'statusMessage'})


def _is_tool_span(obj: dict[str, Any]) -> bool:
    if not isinstance(obj.get('name'), str):
        return False
    kind = obj.get('run_type') if 'run_type' in obj else obj.get('type')
    return isinstance(kind, str) and kind.lower() == 'tool'


@dataclass(frozen=True)
class _Where:
    """A position in a trace tree, as far as the map cares.

    ``tool`` is the raw name of the enclosing tool span, ``None`` outside one or when the scrub rewrote the name
    (a rewritten name is not safe to write into the map). ``role`` is the block of the span, ``field`` the
    top-level member of the block the position sits at or under (``None`` for the block itself), and ``hidden``
    is set when the scrub renamed that member, which then cannot be written into the map. ``kept`` marks the
    span's own declared name, which rides through.
    """

    tool: str | None = None
    role: str = ''
    field: str | None = None
    hidden: bool = False
    kept: bool = False


_OUTSIDE = _Where()


class _TraceHooks:
    """Walks a trace for :meth:`Anonymizer.anonymize_obj`: tracks the tool span a leaf sits in and records its key."""

    def __init__(self, scrub_value: Callable[[str], str], declared: frozenset[str], recorder: ScrubRecorder) -> None:
        self._scrub_value = scrub_value
        self._declared = declared
        self._recorder = recorder
        self._tools: dict[str, str | None] = {}

    def descend(self, parent: dict[str, Any], key: str, key_kept: bool, where: _Where | None) -> _Where:
        if _is_tool_span(parent):
            return self._in_span(parent['name'], key)
        if where is None or where.tool is None:
            return _OUTSIDE
        if where.field is not None:
            return where
        return _Where(where.tool, where.role, key, hidden=not key_kept)

    def leaf(self, where: _Where | None, value: str, scrub_value: Callable[[str], str]) -> str:
        if where is not None and where.kept:
            return value
        new = scrub_value(value)
        if new != value and where is not None:
            self._record(where)
        return new

    def _in_span(self, name: str, key: str) -> _Where:
        if fold_name(name) in self._declared and key == 'name':
            return _Where(kept=True)
        tool = self._tool(name)
        if key in _ARGS_FIELDS:
            return _Where(tool, 'args')
        if key in _RETURNS_FIELDS:
            return _Where(tool, 'returns')
        if key in _ERROR_FIELDS:
            return _Where(tool, 'error')
        return _OUTSIDE

    def _tool(self, name: str) -> str | None:
        if name not in self._tools:
            declared = fold_name(name) in self._declared
            self._tools[name] = name if declared or self._scrub_value(name) == name else None
        return self._tools[name]

    def _record(self, where: _Where) -> None:
        if where.tool is None or where.hidden:
            return
        tool, field = where.tool, where.field
        if where.role == 'returns':
            key = build_key('tool', tool, ('returns',), ('field', field)) if field is not None else build_key('tool', tool, ('returns',))
        elif where.role == 'args':
            key = build_key('tool', tool, ('args', field)) if field is not None else None
        else:
            key = build_key('tool', tool, ('error_channel',))
        if key is not None:
            self._recorder.add(key)


def scrub_trace_tree(data: Any, scrubber: _Scrubber, declared: frozenset[str], recorder: ScrubRecorder) -> Any:  # noqa: ANN401 - decoded JSON
    """``data`` with every string value and field name scrubbed, recording which tool fields changed.

    The result is what ``Anonymizer.anonymize_obj`` returns, except that a dict key or a tool-span name whose folded
    form is in ``declared`` (see :func:`fold_names`) keeps its spelling. ``data`` is scrubbed in place.
    """
    exempt = (lambda key: fold_name(key) in declared) if declared else None
    return scrubber.anonymize_obj(data, exempt_key=exempt, hooks=_TraceHooks(scrubber.anonymize_value, declared, recorder))
