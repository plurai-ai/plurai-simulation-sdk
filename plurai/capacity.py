"""The capacity refusal, and its sibling the unavailable stop: the backend declining work, or not being reachable for it.

Capacity: the backend declining work because the provider refused the account it runs on.

The backend owns the contract and sends data only, in the shape of the credit block: ``code`` is always
:data:`CAPACITY_CODE`, and the block may carry ``resets_at``, the UTC time the refusal lifts (``%Y-%m-%dT%H:%M:%SZ``), and
``reset_source``, whether the provider named that time (``named_reset``) or the backend estimated it (``ttl``). Every call
is refused the same way meanwhile, so a scored run stops on the first one, as on a credit refusal, and keeps what it
measured. No sentence for the user crosses the wire; the words belong to the caller. The stop text is fixed
(:data:`CAPACITY_TEXT`) and never reads a reset time: ``resets_at`` and ``reset_source`` are recorded only.

The unavailable stop is the other end of the same run: the backend could not be reached (:data:`UnavailableCause`
``transport`` or ``status``, a transient failure that outlasted the retry budget) or said so itself (``backend``, the typed
``unavailable`` block, recognized by where it sits: under ``unavailable`` beside where ``capacity`` sits, never by its ``code``, so the
backend can add codes without a client release). An outage is recovered first: the scenario that met it is re-driven once
the backend answers again, and the run stops only when it does not. The same block on a hosted mock's answer fails one
scenario for a re-drive and never stops the run. Its text is fixed (:data:`UNAVAILABLE_TEXT`) and the block carries only the
cause: no status, no backend code and no id reach a customer.

A leaf module, like :mod:`plurai.credit`: it imports only the leaf module :mod:`plurai.credit` (and the standard library) at module load.
"""

from __future__ import annotations

import urllib.error
from collections.abc import Mapping
from typing import Final, Literal, NamedTuple, get_args

from plurai.credit import resets_at_of

#: The one ``code`` of a capacity refusal. Disjoint from every credit kind: both ride the same ``{code, resets_at}``
#: shape, and a reader branching on ``code`` must never mistake one for the other.
CAPACITY_CODE: Final = 'capacity_exhausted'

#: What the customer reads, once, when a run stops on a capacity refusal. Product owns the words. It names a fixed wait and
#: nothing of the backend's state: no status, no provider, no reset time.
CAPACITY_TEXT: Final = "Plurai is in high demand right now and we couldn't fit your run in. Please try again in about 20 minutes."

#: What every other line a customer reads at a capacity stop says about the cause (scenario lines, the abort line, the
#: failure records, mock fault messages): the backend could not take the run, and nothing of why.
BACKEND_REFUSED_PHRASE: Final = 'the backend could not take the run'

#: What the customer reads, once, when a run stops because the backend is unavailable. Product owns the words.
UNAVAILABLE_TEXT: Final = 'Plurai is unavailable right now. Please try again in a few minutes.'

#: Why a run stopped as unavailable: ``transport`` (a connection failure outlasted the retry budget), ``status`` (a
#: 502, 503 or 504 did) or ``backend`` (the backend's typed ``unavailable`` block).
UnavailableCause = Literal['transport', 'status', 'backend']


class CapacityBlock(NamedTuple):
    """One capacity refusal: when the backend lifts it, and whether the provider named that time."""

    resets_at: str | None = None
    reset_source: str | None = None

    def to_record(self) -> dict[str, str]:
        """The block as JSON-ready fields, ``code`` first; ``resets_at`` / ``reset_source`` are absent, never ``null``, when unknown."""
        record = {'code': CAPACITY_CODE}
        if self.resets_at is not None:
            record['resets_at'] = self.resets_at
        if self.reset_source is not None:
            record['reset_source'] = self.reset_source
        return record


def capacity_block_from_body(body: object) -> CapacityBlock | None:
    """The block of a body whose ``code`` is exactly :data:`CAPACITY_CODE`, else ``None``. Strict, like the credit reader:
    a code the backend did not send is not a capacity refusal, and no prose is ever read for one."""
    if not isinstance(body, Mapping) or body.get('code') != CAPACITY_CODE:
        return None
    source = body.get('reset_source')
    return CapacityBlock(resets_at_of(body), source if isinstance(source, str) and source.strip() else None)


def capacity_block_of(exc: BaseException) -> CapacityBlock | None:
    """The capacity block *exc* carries (a mock runtime fault the backend refused for capacity), or ``None``."""
    block = getattr(exc, 'capacity', None)
    return block if isinstance(block, CapacityBlock) else None


class UnavailableBlock(NamedTuple):
    """One unavailable stop: its cause, and the backend's ``code`` when the backend named one (recorded, never read for wording)."""

    cause: UnavailableCause
    code: str | None = None

    def to_record(self) -> dict[str, str]:
        """The block as JSON-ready fields, ``cause`` first; ``code`` is absent, never ``null``, when unknown."""
        record = {'cause': self.cause}
        if self.code is not None:
            record['code'] = self.code
        return record


def unavailable_block_from_body(body: object) -> UnavailableBlock | None:
    """The typed unavailable block *body* carries: the mapping under its ``unavailable`` key, else ``None``. Read by placement, like
    the plugin reads it: whatever ``code`` the block names is recorded and never read for wording or for whether it counts, so the
    backend can add codes without a client release. No prose is ever read for one."""
    if not isinstance(body, Mapping) or not isinstance(block := body.get('unavailable'), Mapping):
        return None
    code = block.get('code')
    return UnavailableBlock('backend', code if isinstance(code, str) and code else None)


def unavailable_block_from_record(record: object) -> UnavailableBlock | None:
    """The block :meth:`UnavailableBlock.to_record` wrote (the form it takes between the agent's process and the runner), else ``None``."""
    if not isinstance(record, Mapping) or record.get('cause') not in get_args(UnavailableCause):
        return None
    code = record.get('code')
    return UnavailableBlock(record['cause'], code if isinstance(code, str) else None)


def unavailable_block_of(exc: BaseException) -> UnavailableBlock | None:
    """The unavailable block *exc* carries (a request failure no retry could cure, or a mock runtime fault the backend refused
    as unavailable), or ``None``. A layer that wraps the failure keeps it, so the chain is read.

    The chain follows what an error was raised ``from`` (``__cause__``), and the error being handled when it was raised
    (``__context__``) only while that context is not suppressed (``raise ... from None``) and is itself a transport or status
    failure: a socket error, or a request failure that carries a block. A different error raised ``from None`` inside a handler
    of an outage, or raised while handling one that is no transport failure, is not that outage.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(block := getattr(current, 'unavailable', None), UnavailableBlock):
            return block
        following = current.__cause__
        if following is None and not current.__suppress_context__ and _is_transport_or_status_failure(current.__context__):
            following = current.__context__
        if following is None and isinstance(current, urllib.error.URLError) and isinstance(current.reason, BaseException):
            following = current.reason
        current = following
    return None


def _is_transport_or_status_failure(exc: BaseException | None) -> bool:
    """Whether *exc* is a failure of reaching the backend: a socket error (urllib's ``URLError`` is one), or a request failure
    that carries an unavailable block."""
    return exc is not None and (isinstance(exc, OSError) or isinstance(getattr(exc, 'unavailable', None), UnavailableBlock))


__all__ = [
    'BACKEND_REFUSED_PHRASE',
    'CAPACITY_CODE',
    'CAPACITY_TEXT',
    'UNAVAILABLE_TEXT',
    'CapacityBlock',
    'UnavailableBlock',
    'UnavailableCause',
    'capacity_block_from_body',
    'capacity_block_of',
    'unavailable_block_from_body',
    'unavailable_block_from_record',
    'unavailable_block_of',
]
