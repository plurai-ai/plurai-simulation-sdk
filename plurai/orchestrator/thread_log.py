"""A log of the thread ids a process opens and releases, for a caller that must cancel the ones still open if it is stopped.

``plurai ci run`` installs a :class:`ThreadLog` for the length of a run; :meth:`OrchestratorClient.create_thread` notes each
thread it opens and :meth:`OrchestratorClient.forget_thread` notes it closed: when its scenario has ended, however it ended, or
when ``create_thread`` fails after the backend created it, because the caller never learns that id. A signal handler that
ends the run reads :meth:`ThreadLog.snapshot`, the ids still open, and cancels each (``DELETE /threads/{id}`` is idempotent, so an
id whose thread already finished answers success). A creation still on the wire when the run is stopped is not cancelled by the
client, because its id is not known yet; the server's abandonment sweep ends it once the thread has driven no turn for the thread
time-to-live. With no log installed, noting a thread does nothing.
"""

from __future__ import annotations


class ThreadLog:
    """Thread ids in the order they were opened, and which of them were closed.

    Takes no lock: the signal handler runs on the thread that notes ids and the supervisor then reads the log, so a lock the
    interrupted thread holds would never be released. Each step below is one list or set operation, atomic under the GIL.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._closed: set[str] = set()

    def note(self, thread_id: str) -> None:
        if thread_id not in self._ids:
            self._ids.append(thread_id)

    def close(self, thread_id: str) -> None:
        self._closed.add(thread_id)

    def opened(self) -> tuple[str, ...]:
        """Every id noted, open or closed."""
        return tuple(self._ids)

    def snapshot(self) -> tuple[str, ...]:
        """The ids noted and not closed."""
        return tuple(i for i in tuple(self._ids) if i not in self._closed)


_current: ThreadLog | None = None


def install(log: ThreadLog | None) -> None:
    """Make ``log`` the one :func:`note_opened` writes to; ``None`` removes it."""
    global _current  # noqa: PLW0603
    _current = log


def note_opened(thread_id: str) -> None:
    log = _current
    if log is not None:
        log.note(thread_id)


def note_closed(thread_id: str) -> None:
    log = _current
    if log is not None:
        log.close(thread_id)
