"""Release the threads a CI run holds open: ``DELETE /threads/{id}`` for each, all at once, under one absolute deadline.

The backend keeps an open thread until its idle time-to-live runs out, so a job that is cancelled releases them itself. The
requests go out from one thread each and the caller waits for all of them together: the wait is the deadline, never the sum of
the answers. ``DELETE`` is idempotent, so a thread that already finished answers success and an id the backend no longer holds
answers 404; both count as released. A cancel that fails with a connection error, a 429 or a 5xx is sent again, after a short pause, until
it is answered or the deadline passes; any other answer is final. A thread whose cancel did not answer in time is counted.

The deadline is a point on ``clock`` and every wait is cut to it, so nothing here outlasts it. ``clock`` and ``pause`` are injectable:
a test drives the deadline logic without wall time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx

from plurai import _http


@dataclass(frozen=True)
class Released:
    """How many of the cancels were answered with success, and how many were not (an error, or no answer before the deadline)."""

    confirmed: int
    unconfirmed: int


#: The pause before a cancel is sent again, doubled after each failure up to the cap.
_RETRY_PAUSE: Final = 0.1
_RETRY_PAUSE_CAP: Final = 0.4
#: How often the wait looks at the cancels.
_POLL_SECONDS: Final = 0.01
#: The status a connection error is treated as: a failure that is sent again.
_TRANSPORT_ERROR: Final = 599
#: The status of a rate limit: sent again like a server error.
_TOO_MANY_REQUESTS: Final = 429

Clock = Callable[[], float]
#: ``pause(event, seconds)``: wait up to ``seconds`` or until ``event`` is set; true when it is set.
Pause = Callable[[threading.Event, float], bool]


def wait_event(event: threading.Event, seconds: float) -> bool:
    return event.wait(seconds)


def deliver(send: Callable[[float], int], *, ends: float, stopped: threading.Event, clock: Clock = time.monotonic, pause: Pause = wait_event) -> bool:
    """Send one cancel until it is answered or ``ends`` (a point on ``clock``) passes; true when it counts as released.

    ``send(timeout)`` makes the request, which may take at most ``timeout`` seconds, and returns the status. A connection error,
    a 429 or a 5xx is sent again after a pause that doubles up to the cap and is cut to what is left of the deadline; any other
    answer is final. Nothing is sent once ``stopped`` is set or the deadline has passed.
    """
    delay = _RETRY_PAUSE
    while (remaining := ends - clock()) > 0 and not stopped.is_set():
        status = send(remaining)
        if status < 500 and status != _TOO_MANY_REQUESTS:
            return status < 300 or status == 404
        pause(stopped, min(delay, max(0.0, ends - clock())))
        delay = min(delay * 2, _RETRY_PAUSE_CAP)
    return False


class Cancels:
    """The cancels in flight. :meth:`wait` returns once every one answered, the deadline passed or ``hurry`` was set.

    ``ends`` is the deadline, a point on ``clock``. ``transport`` replaces the network.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        ends: float,
        clock: Clock = time.monotonic,
        pause: Pause = wait_event,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        #: Thread id to how its cancel ended: ``None`` while it is being sent, ``True`` released, ``False`` not.
        self._outcomes: dict[str, bool | None] = {}
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._ends = ends
        self._clock = clock
        self._pause = pause
        self._client = _http.Client(
            base_url=url.rstrip('/'),
            headers={'Authorization': f'Bearer {token}'},
            follow_redirects=False,
            limits=httpx.Limits(max_connections=None),
            transport=transport,
        )

    def add(self, thread_id: str) -> None:
        """Cancel one more thread, under the deadline of the others. Does nothing for an id already added or once :meth:`wait` has ended."""
        with self._lock:
            if self._closed.is_set() or thread_id in self._outcomes:
                return
            self._outcomes[thread_id] = None
        threading.Thread(target=self._cancel, args=(thread_id,), name='plurai-ci-cancel', daemon=True).start()

    def _status(self, thread_id: str, timeout: float) -> int:
        try:
            return self._client.delete(f'/threads/{thread_id}', timeout=timeout).status_code
        except (httpx.HTTPError, RuntimeError):
            return _TRANSPORT_ERROR

    def _cancel(self, thread_id: str) -> None:
        released = False
        try:
            released = deliver(lambda timeout: self._status(thread_id, timeout), ends=self._ends, stopped=self._closed, clock=self._clock, pause=self._pause)
        finally:
            with self._lock:
                self._outcomes[thread_id] = released

    def _busy(self) -> bool:
        with self._lock:
            return any(outcome is None for outcome in self._outcomes.values())

    def wait(self, hurry: threading.Event) -> Released:
        """Block until every cancel answered, the deadline passed or ``hurry`` is set; then stop the retries and close the client.
        A cancel still unanswered then counts as unconfirmed."""
        while self._busy() and (left := self._ends - self._clock()) > 0 and not self._pause(hurry, min(_POLL_SECONDS, left)):
            pass
        self._closed.set()
        with self._lock:
            confirmed = sum(1 for outcome in self._outcomes.values() if outcome)
            total = len(self._outcomes)
        try:
            self._client.close()
        except Exception:  # noqa: BLE001, S110 - requests still in flight are abandoned with the process
            pass
        return Released(confirmed, total - confirmed)
