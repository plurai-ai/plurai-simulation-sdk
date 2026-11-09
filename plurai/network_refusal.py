"""A request that a network control between this machine and Plurai answered, or held, instead of Plurai.

Two typed causes, read from facts of the exchange and never from a response body:

* ``refused``: the answer carries no :data:`~plurai.client_handshake.FLOOR_HEADER`, which a Plurai backend adds to every
  answer to an identified client, and its status is one a proxy or firewall writes: a redirect (3xx, often to a sign-in
  page), 403 forbidden, 407 proxy authentication required, or 451 unavailable for legal reasons. Every other status
  keeps its own words, whatever its headers: a 401, 402, 404, 400 or 413 can be Plurai's own answer without the header.
* ``undelivered``: an upload whose object Plurai had already created, whose content then failed to arrive (a transport
  failure on a part, after the retries).

:func:`sentence` is what the operator reads. :func:`note` appends one JSON line per refusal to the sink a command sets
(:data:`SINK`); it never raises, so recording a refusal cannot change how a command fails.

Stdlib and httpx only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import httpx

from plurai.client_handshake import FLOOR_HEADER

logger = logging.getLogger(__name__)

#: The schema tag of one line in a command's refusal record.
RECORD_SCHEMA: Final = 'network_refusal/1'

#: The file a command's refusals are appended to, under the workspace's ``.plurai/`` folder.
RECORD_FILENAME: Final = 'network_refusals.jsonl'

#: The statuses other than a redirect that an egress control writes.
EGRESS_STATUSES: Final = frozenset({403, 407, 451})

#: The longest ``Server`` header quoted; a longer one is dropped rather than cut.
_MAX_SERVER_CHARS = 60

Cause = Literal['refused', 'undelivered']


@dataclass(frozen=True)
class NetworkRefusal:
    """One request that did not reach Plurai, or whose content never arrived."""

    cause: Cause
    #: The host the request was sent to.
    host: str
    #: The route template the call site names (``/storage/objects/{id}/parts/{index}``), never a path holding ids.
    request_kind: str
    #: The answer's status, or ``None`` when nothing answered (``undelivered``).
    status: int | None
    #: The answer's ``Server`` header when it is short printable ASCII, else ``None``.
    server: str | None
    #: The exception class of an ``undelivered`` upload, else ``None``.
    failure: str | None
    #: When it happened: ISO-8601 UTC, to the second.
    at: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def is_egress_status(status: int) -> bool:
    """Whether ``status`` is one a proxy or firewall writes: a redirect, or :data:`EGRESS_STATUSES`."""
    return 300 <= status < 400 or status in EGRESS_STATUSES


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def _host(request: httpx.Request | None) -> str:
    return request.url.host if request is not None else ''


def _request_of(response: httpx.Response) -> httpx.Request | None:
    try:
        return response.request
    except RuntimeError:  # a response built without a request
        return None


def _clean_server(value: str | None) -> str | None:
    if value is None or not value or len(value) > _MAX_SERVER_CHARS or not all(' ' <= char <= '~' for char in value):
        return None
    return value


def of_response(response: httpx.Response, *, request_kind: str) -> NetworkRefusal | None:
    """The refusal *response* is when it did not come from Plurai, else ``None``. Reads the status and headers only."""
    if FLOOR_HEADER in response.headers or not is_egress_status(response.status_code):
        return None
    return NetworkRefusal(
        cause='refused',
        host=_host(_request_of(response)),
        request_kind=request_kind,
        status=response.status_code,
        server=_clean_server(response.headers.get('Server')),
        failure=None,
        at=_now(),
    )


def undelivered(error: BaseException, *, request: httpx.Request | None, request_kind: str) -> NetworkRefusal:
    """The record of upload content that failed to arrive after its object was created."""
    return NetworkRefusal(
        cause='undelivered', host=_host(request), request_kind=request_kind, status=None, server=None, failure=type(error).__name__, at=_now()
    )


def sentence(refusal: NetworkRefusal) -> str:
    """The operator's words for *refusal*: what happened and who can fix it."""
    host = refusal.host or 'the Plurai backend'
    if refusal.cause == 'undelivered':
        return (
            f'Plurai accepted the upload but never received its content: sending it to {host} failed ({refusal.failure}). '
            f'If this machine reaches the internet through a proxy or firewall, it may be holding the upload: ask your IT team to allow uploads to {host}.'
        )
    answered_by = f', answered by "{refusal.server}"' if refusal.server else ''
    return (
        f'the request to {host} was blocked before it reached Plurai (HTTP {refusal.status}{answered_by}). '
        f'A network proxy or firewall is the most likely cause: ask your IT team to allow this machine to reach {host}.'
    )


@dataclass(frozen=True)
class RefusalSink:
    """Where a command's refusals are recorded: the file, the command's name, and how to prepare the file's folder."""

    path: Path
    command: str
    prepare: Callable[[Path], None] | None = None


#: The current command's sink, or ``None`` when nothing records (the runner records into ``error.json`` instead).
SINK: ContextVar[RefusalSink | None] = ContextVar('plurai_network_refusal_sink', default=None)


def note(refusal: NetworkRefusal) -> None:
    """Append *refusal* to the current :data:`SINK`, if one is set. Never raises: any failure is logged once and dropped."""
    try:
        sink = SINK.get()
        if sink is None:
            return
        if sink.prepare is not None:
            sink.prepare(sink.path.parent)
        line = json.dumps({'schema': RECORD_SCHEMA, 'command': sink.command, **refusal.to_record()}, ensure_ascii=False)
        with sink.path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
    except Exception as error:  # noqa: BLE001 - the record is best-effort; the refusal it describes is still raised
        logger.warning('could not record a network refusal (%s)', type(error).__name__)


__all__ = [
    'EGRESS_STATUSES',
    'RECORD_FILENAME',
    'RECORD_SCHEMA',
    'SINK',
    'NetworkRefusal',
    'RefusalSink',
    'is_egress_status',
    'note',
    'of_response',
    'sentence',
    'undelivered',
]
