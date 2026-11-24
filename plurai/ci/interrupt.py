"""SIGINT and SIGTERM during ``plurai ci run``: freeze the run, finish it on another thread, then unwind.

GitHub cancels a step with SIGINT, SIGTERM 7.5 seconds later and a kill 2.5 seconds after that. The handler runs on the main
thread, which is inside the runner's event loop. It starts a supervisor thread that does the whole job inside one time budget:
the cancels and the stop of the agent children share a single absolute deadline (:data:`WORK_DEADLINE_SECONDS` after the signal),
and the partial reports are written after it, inside the margin that is left before the kill. The handler blocks the main thread
until the supervisor is done: a frozen loop records nothing more, so the partial results are the ones on
disk at the signal, and no agent child reacts to its own death by writing an error over a scenario that was still running. When the
supervisor is done the handler raises :class:`Interrupted`, which unwinds the main thread as ``KeyboardInterrupt`` does.

A later signal while the supervisor works only sets ``hurry``: the supervisor stops waiting and finishes at once. It never raises, so
a signal cannot cut a file in half (files are written through a temporary file and a rename) and cannot start a second cleanup.
Once the reports are final a timer ends the process after :data:`UNWIND_SECONDS` if the main thread has not unwound by then.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FrameType
from typing import Final

from plurai import _subprocess, exit_codes
from plurai.ci.cancel import Pause, wait_event

#: GitHub kills a step about this long after its first SIGTERM. After a SIGINT there are 7.5 seconds more; the budget below
#: assumes SIGTERM is the first signal, so it holds either way.
GITHUB_KILL_SECONDS: Final = 2.5

#: Kept free at the end of the window for the partial reports: the summary, the comment, the report, ``run.json``, the JUnit file
#: and the transcripts, all written locally. A second is far above what a large run takes and leaves room on a loaded runner.
REPORT_MARGIN_SECONDS: Final = 1.0

#: The one deadline, counted from the first signal, that the thread cancels and the stop of the agent children share. When it
#: passes the supervisor stops waiting, whatever is still unanswered, and writes the reports, so they are on disk before the kill
#: even when the backend never answers.
WORK_DEADLINE_SECONDS: Final = GITHUB_KILL_SECONDS - REPORT_MARGIN_SECONDS

#: How long an agent child has to exit after SIGTERM before it is killed. Runs beside the cancels and ends at the work deadline at
#: the latest.
CHILD_GRACE_SECONDS: Final = 1.0

#: How long the main thread has to unwind once the reports are final. The unwinding closes event-loop tasks and can wait on an agent
#: child; past this the process ends at once. The timer starts after the reports are written, so with the work deadline (1.5 s) and
#: reports that take milliseconds it fires about 2.0 s after the signal, before GitHub's kill at 2.5 s. A report write that uses the
#: whole margin pushes it to 3.0 s, past the kill; the reports are on disk by then either way.
UNWIND_SECONDS: Final = 0.5

#: How often the stop of the agent children looks at them.
_CHILD_POLL_SECONDS: Final = 0.02

_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


class Interrupted(KeyboardInterrupt):
    """Raised on the main thread once the run is finished; carries the exit code. A ``KeyboardInterrupt`` so that the runner treats
    it as the run ending and not as a scenario failing."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class Signalled:
    """The first signal: which one, when (monotonic clock), and the event a later signal sets."""

    number: int
    at: float
    hurry: threading.Event = field(default_factory=threading.Event)


def stop_children(
    grace: float, hurry: threading.Event, *, ends: float | None = None, clock: Callable[[], float] = time.monotonic, pause: Pause = wait_event
) -> int:
    """SIGTERM every agent child, wait up to ``grace`` seconds (or until ``ends``, a point on ``clock``, or ``hurry``), then
    SIGKILL those still alive.

    The child answers SIGTERM by closing its input, which ends its loop and the agent processes under it; a child that does not
    in time is killed. Returns how many were killed.
    """
    pids = _subprocess.live_child_pids()
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    until = clock() + grace if ends is None else min(clock() + grace, ends)
    while (pids := [pid for pid in pids if _exists(pid)]) and (left := until - clock()) > 0 and not pause(hurry, min(_CHILD_POLL_SECONDS, left)):
        pass
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
    return len(pids)


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class SignalGuard:
    """Handle SIGINT and SIGTERM for the length of a ``with`` block. ``finish`` runs once, on a supervisor thread, and returns the
    exit code. ``first`` runs in the handler itself, on the main thread, before the supervisor starts, so it must not take a lock the
    main thread may hold. Outside the main thread the guard installs nothing."""

    def __init__(self, finish: Callable[[Signalled], int], *, first: Callable[[], None] = lambda: None) -> None:
        self._finish = finish
        self._first_signal = first
        self._first: Signalled | None = None
        self._code = exit_codes.NOT_MEASURED
        self._done = threading.Event()
        self._armed = True
        self._previous: dict[int, object] = {}

    def __enter__(self) -> SignalGuard:
        if threading.current_thread() is threading.main_thread():
            for number in _SIGNALS:
                self._previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *_: object) -> None:
        for number, previous in self._previous.items():
            signal.signal(number, previous)  # type: ignore[arg-type]
        self._previous.clear()

    def disarm(self) -> None:
        """The run's files are final: a signal from here on has nothing to add and is ignored."""
        self._armed = False

    def _handle(self, number: int, _frame: FrameType | None) -> None:
        if self._first is not None:
            self._first.hurry.set()
            return
        if not self._armed:
            return
        self._first = Signalled(number, time.monotonic())
        self._first_signal()
        threading.Thread(target=self._supervise, args=(self._first,), name='plurai-ci-interrupt', daemon=True).start()
        self._done.wait()
        raise Interrupted(self._code)

    def _supervise(self, first: Signalled) -> None:
        try:
            self._code = self._finish(first)
        except BaseException:  # noqa: BLE001 - the run must end with a code, whatever the cleanup hit
            traceback.print_exc()
        finally:
            with contextlib.suppress(OSError, ValueError):
                sys.stdout.flush()
                sys.stderr.flush()
            hard_exit = threading.Timer(UNWIND_SECONDS, os._exit, args=(self._code,))
            hard_exit.daemon = True
            hard_exit.start()
            self._done.set()
