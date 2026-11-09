"""The retry arithmetic every HTTP stack in the SDK shares.

A hosted backend answers 429 when it throttles a caller, and every path that reaches it must treat
that the same way: wait as ``Retry-After`` says, add jitter, never wait longer than the cap, and try
again. The stacks differ in their transport (async httpx, a blocking client in the agent's process,
the Node shims, which mirror these numbers in ``_mocking/node/retry.cjs``), so this module holds
only the pure parts: reading ``Retry-After``, the back-off schedule, and the combination of the two.
Deciding WHICH failures are retried stays with each stack, because that depends on whether a
repeated request can be applied twice.

Stdlib only: the mock client runs inside the customer's agent process.
"""

from __future__ import annotations

import random
import socket
import time
import urllib.error
from collections.abc import Callable, Iterator
from email.utils import parsedate_to_datetime

#: A rate limiter's refusal. It is answered before the request is forwarded, so it was never applied
#: and repeating it cannot double-apply it, whatever the method.
RATE_LIMITED_STATUS = 429

#: Retries after the first attempt that a throttled request gets, and the schedule between them:
#: ``min(cap, base * 2**(n-1) + uniform(0, base))`` for the n-th retry. At the defaults the waits
#: are 1, 2, 4, 8 and 16 seconds plus up to a second of jitter each, about 31 to 36 seconds in all,
#: which outlasts a short throttling window without holding a run for minutes.
RATE_LIMIT_RETRIES = 5

#: The statuses the gateway answers while the backend is not there: a transient failure, waited out under :data:`TRANSIENT_RETRIES`.
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})

#: Retries after the first attempt that a transient failure (502/503/504, a connection failure) gets. The waits are the
#: same schedule, so the sixth is held to the cap: 1, 2, 4, 8, 16 and 30 seconds plus jitter, about 61 to 66 seconds in
#: all, which outlasts the 35 seconds a backend upgrade rejects every request for.
TRANSIENT_RETRIES = 6
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """*exc*, then what it was raised from or while handling, and the ``reason`` of a :class:`urllib.error.URLError`, once each.

    The one walk every reader of a wrapped failure shares: a layer that wraps a failure keeps it on ``__cause__`` or
    ``__context__``, and urllib keeps a connect failure on ``URLError.reason``.
    """
    seen: set[int] = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if (wrapped := current.__cause__ or current.__context__) is not None:
            pending.append(wrapped)
        if isinstance(current, urllib.error.URLError) and isinstance(current.reason, BaseException):
            pending.append(current.reason)


def is_name_not_found(exc: BaseException) -> bool:
    """Whether *exc* failed because the host name did not resolve (a mistyped URL, a DNS outage on the caller's side).

    httpx, httpcore and urllib each wrap the ``socket.gaierror``, so the whole chain is read. A name that does not resolve is not
    the backend being down: it does not clear by waiting, so it is never an outage.
    """
    return any(isinstance(link, socket.gaierror) for link in exception_chain(exc))


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """The wait a ``Retry-After`` header value asks for, in seconds, or ``None`` when it asks for none.

    Both RFC 9110 forms are read: delta-seconds, and an HTTP-date, which is measured against ``now``
    (epoch seconds, the wall clock by default). A date in the past waits nothing; a value that is
    neither form is ``None``, so the ordinary back-off applies.
    """
    raw = (value or '').strip()
    if not raw:
        return None
    if raw.isascii() and raw.isdigit():
        return float(raw)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def backoff_seconds(
    retry_number: int,
    *,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_CAP_SECONDS,
    jitter: Callable[[float, float], float] = random.uniform,
) -> float:
    """The wait before the ``retry_number``-th retry (counting from 1): exponential, additive jitter, capped.

    Additive rather than full jitter: drawing from zero lets a wait collapse or shrink between
    attempts, which hammers a server that asked for room. The floor is ``base`` and the schedule grows.
    """
    return min(cap, base * 2 ** (retry_number - 1) + jitter(0.0, base))


def throttled_wait(backoff: float, retry_after: float | None, *, cap: float = BACKOFF_CAP_SECONDS) -> float:
    """The wait before retrying a 429: the larger of the back-off and what ``Retry-After`` asked, capped."""
    return min(cap, max(backoff, retry_after or 0.0))


def rate_limit_wait(
    retry_number: int,
    retry_after_header: str | None,
    *,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_CAP_SECONDS,
) -> float:
    """:func:`throttled_wait` over the scheduled back-off for ``retry_number`` and a raw ``Retry-After`` value."""
    return throttled_wait(backoff_seconds(retry_number, base=base, cap=cap), parse_retry_after(retry_after_header), cap=cap)
