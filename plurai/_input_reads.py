"""Which injected inputs the harness READ — the fact behind ``FinalizeResponse.unread_inputs``.

The child hands the scenario's ``initial_agent_context`` to the harness through three lanes:
``seed_state`` param 1, the turn-1 payload (a one-shot hop's inputs are the same dict), and the mock's
``get_simulation_context().agent_context``. A harness that reads none of them never delivers the key
to the agent, and nothing downstream could tell: the run scored the agent against a scenario it never
received. A read on ANY lane credits a key, so a harness that read every key is recorded clean. A key
nobody read is named only where no lane serves it on demand: a mock handler reads its context only when
the agent calls the tool, and often only for some arguments, so a non-read there proves nothing, and a
deployed mock is out of sight — there the child claims nothing. Except a document the backend bound for the
user to hand over (:func:`is_bound_document`): no mock serves it on demand, its only route is the harness,
so its non-read is named whatever the mock is.

:class:`InputReads` is a ``dict`` that records the keys read off it. A single-key read marks that
key; a whole-mapping read (iteration, ``keys``/``items``/``values``, a copy, a merge, pickling —
which covers ``{**d}``, ``dict(d)``, ``json.dumps``, ``yaml.safe_dump``, ``str``/f-strings,
``deepcopy``, pydantic and LangGraph state construction) marks every key. So "unread" is exact except
for readers with no Python hook: C serializers that walk dict storage directly (``orjson``, ``ujson``,
``msgspec``, ``rapidjson``) and a forked child reading inherited memory — a key read ONLY that way is
reported unread — and a harness that keeps the mapping by reference and reads a key only inside a
tool the agent never called. The converse is deliberately loose — a harness that hands the whole payload to a
framework, or prints or logs it, counts as having read every key — which only ever withholds a
verdict, never invents one.

A FACT, not a verdict: whether an unread key is an undelivered input depends on the key's declared
channel, which the backend owns (a ``store``-channel field rides the seeded world instead).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from typing import Any

import yaml


class InputReads(dict[str, Any]):
    """A ``dict`` that records which of its keys were read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def _read_all(self) -> None:
        self.read.update(dict.keys(self))

    def snapshot(self) -> dict[str, Any]:
        """A plain copy that records NO read — for the child's own bookkeeping, never the harness."""
        return dict(dict.items(self))

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        self.read.add(key)
        return dict.get(self, key, default)

    def pop(self, key: str, *default: Any) -> Any:
        self.read.add(key)
        return dict.pop(self, key, *default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        self.read.add(key)
        return dict.setdefault(self, key, default)

    def __iter__(self) -> Iterator[str]:
        self._read_all()
        return dict.__iter__(self)

    def keys(self) -> Any:
        self._read_all()
        return dict.keys(self)

    def items(self) -> Any:
        self._read_all()
        return dict.items(self)

    def values(self) -> Any:
        self._read_all()
        return dict.values(self)

    def copy(self) -> dict[str, Any]:  # type: ignore[override]
        self._read_all()
        return self.snapshot()

    def __or__(self, other: Any) -> Any:
        self._read_all()
        return self.snapshot() | other

    def __reduce_ex__(self, protocol: Any) -> Any:
        self._read_all()
        return (dict, (self.snapshot(),))

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        self._read_all()
        return copy.deepcopy(self.snapshot(), memo)

    def __repr__(self) -> str:
        self._read_all()
        return repr(self.snapshot())


# PyYAML represents by EXACT type, so a ``dict`` subclass would make ``yaml.safe_dump`` raise inside a
# harness that writes its context to disk. Represented as the plain mapping it is — a whole read.
for _representer in (yaml.representer.SafeRepresenter, yaml.representer.Representer):
    _representer.add_representer(InputReads, lambda dumper, data: dumper.represent_dict(data))


#: The backend's document generation binds a binary under the inline cap as a ``data:`` URI and only an over-cap one as a
#: served ``<scheme>:///documents/…`` reference, so a stored binding is in either form.
_BOUND_DOCUMENT_PREFIXES: tuple[str, ...] = ('data:', 'plurai:///documents/')


def is_bound_document(value: Any) -> bool:
    """True when ``value`` is a document the backend bound for the user to hand over — never served by a mock."""
    return isinstance(value, str) and value.strip().casefold().startswith(_BOUND_DOCUMENT_PREFIXES)


def unread_inputs(declared: Iterable[str], recorders: Iterable[InputReads]) -> list[str]:
    """The declared keys no recorder saw read, sorted."""
    seen: set[str] = set()
    for recorder in recorders:
        seen |= recorder.read
    return sorted(set(declared) - seen)
