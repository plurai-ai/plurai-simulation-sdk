"""The exit codes of every Plurai command line: ``plurai``, ``plurai-run-simulation`` and ``plurai-scan-contract``.

A caller that dispatches on a status (the plugin, a CI job, a shell script) reads this one table. Each code
means one thing in every command, and a command uses only the rows that apply to it. Two rules hold for all of
them:

* A command that ran and found its policies failing exits non-zero (:data:`POLICIES_FAILED`). A zero status
  means every policy that was checked held.
* 2 is never returned. It is the status ``argparse`` uses for a bad command line, and every parser here exits
  with :data:`USAGE` instead, so a 2 can only come from outside the SDK.

The numbers below 64 are specific to Plurai. 64 and above follow the BSD ``sysexits`` convention for the rows
that have a counterpart there.

Stdlib only: a command that never reads a ``simulate.py`` imports this module without the rest of the package.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from typing import Final, NoReturn

from plurai import client_handshake

#: The command did what it was asked and, where it checks anything, everything it checked held.
OK: Final = 0

#: The run completed and at least one scenario failed a policy. The artifacts are written and valid: this is a
#: finding about the agent, not a fault of the run.
POLICIES_FAILED: Final = 1

#: A bad command line: an unknown flag, a missing argument, a value out of range.
USAGE: Final = 64

#: The command declined to proceed on what it checked first: a gate finding holds the run, the target or input is
#: unusable (a malformed overrule file, a directory the command will not overwrite, an unreadable run), or the
#: platform is unsupported. Nothing trustworthy was produced; the message says what to change.
REFUSED: Final = 65

#: A local input file is missing, empty or unreadable.
INPUT_UNREADABLE: Final = 66

#: The backend or the mock was unreachable, timed out, or answered something this client cannot use, after the
#: retries. Running the command again may succeed.
TRANSPORT: Final = 69

#: A defect in the SDK itself, not a finding about the input. Nothing was verified.
INTERNAL_FAULT: Final = 70

#: A temporary refusal: the sign-in of ``plurai login`` timed out, or a server answered HTTP 429. Running the command
#: again later may succeed.
TEMPFAIL: Final = 75

#: No usable backend credential: nothing is configured, the saved login is unusable, or the backend rejected the
#: token (HTTP 401 or 403). ``plurai login`` fixes it.
CREDENTIALS: Final = 77

#: The backend refuses this client version (HTTP 426). Upgrading the SDK, and the plugin that pins it, fixes it.
UPGRADE_REQUIRED: Final = 10

#: A mocked tool call could not be answered, so the run was stopped instead of scoring an agent against a world
#: that did not respond. The run's artifacts say which call and why.
MOCK_UNANSWERED: Final = 11

#: The backend does not have the route this command calls: it predates the feature.
ROUTE_MISSING: Final = 12

#: A backend job ended in failure.
JOB_FAILED: Final = 13

#: ``plurai job wait`` stopped waiting with the job still running (its time budget ran out while the backend was
#: healthy). Invoke it again to keep waiting.
STILL_RUNNING: Final = 14

#: The job, run or artifact asked for does not exist on the backend.
NOT_FOUND: Final = 15

#: The artifact exists but is still a draft, so the command did not fetch it.
NOT_READY: Final = 16

#: The backend refused what was sent (an HTTP 4xx other than a credential or version refusal). Retrying the same
#: input fails the same way.
BACKEND_REFUSED: Final = 17

#: ``python -m plurai.host_toolchain`` measured a gap in what the agent's host needs.
REQUIREMENT_GAP: Final = 19

#: ``plurai-scan-contract`` verified the sources and a finding holds the flow.
SCAN_HELD: Final = 18

#: A command that checks policies stopped, or finished, with nothing it can call a pass or a fail: an id the backend does not
#: have, an infrastructure error, a suite fault, no credit, or gating scenarios it could not measure. Running it again, or
#: fixing the cause the report names, may succeed.
NOT_MEASURED: Final = 20

#: The user interrupted the command (Ctrl-C): 128 plus SIGINT, the status a shell reports for it. What the command had
#: finished stays on disk; nothing was concluded.
INTERRUPTED: Final = 130

#: Name -> (code, one-line meaning), in the order the README prints them.
TABLE: Final[dict[str, tuple[int, str]]] = {
    'OK': (OK, 'done; everything checked held'),
    'POLICIES_FAILED': (POLICIES_FAILED, 'the run completed and at least one scenario failed a policy'),
    'USAGE': (USAGE, 'bad command line'),
    'REFUSED': (REFUSED, 'declined to proceed: a gate holds the run, or the input or platform is unusable'),
    'INPUT_UNREADABLE': (INPUT_UNREADABLE, 'a local input file is missing, empty or unreadable'),
    'TRANSPORT': (TRANSPORT, 'the backend or the mock was unreachable or gave an unusable answer'),
    'INTERNAL_FAULT': (INTERNAL_FAULT, 'a defect in the SDK; nothing was verified'),
    'TEMPFAIL': (TEMPFAIL, 'temporary: the sign-in timed out or a server asked to retry later (HTTP 429)'),
    'CREDENTIALS': (CREDENTIALS, 'no usable credential, or the backend rejected the token'),
    'UPGRADE_REQUIRED': (UPGRADE_REQUIRED, 'the backend refuses this client version'),
    'MOCK_UNANSWERED': (MOCK_UNANSWERED, 'a mocked tool call could not be answered; the run was stopped'),
    'ROUTE_MISSING': (ROUTE_MISSING, 'the backend predates the route this command calls'),
    'JOB_FAILED': (JOB_FAILED, 'a backend job ended in failure'),
    'STILL_RUNNING': (STILL_RUNNING, 'a job is still running when the wait budget ran out; wait again'),
    'NOT_FOUND': (NOT_FOUND, 'the job, run or artifact does not exist'),
    'NOT_READY': (NOT_READY, 'the artifact is still a draft'),
    'BACKEND_REFUSED': (BACKEND_REFUSED, 'the backend refused what was sent'),
    'REQUIREMENT_GAP': (REQUIREMENT_GAP, 'the host toolchain check measured a gap'),
    'SCAN_HELD': (SCAN_HELD, 'the scan verified the sources and a finding holds the flow'),
    'NOT_MEASURED': (NOT_MEASURED, 'the run could not measure what it checks: infra, suite or credits'),
    'INTERRUPTED': (INTERRUPTED, 'interrupted by the user (Ctrl-C)'),
}

#: HTTP statuses that mean the same row in every command.
_CREDENTIAL_STATUSES: Final = frozenset({401, 403})


def for_http_status(status: int) -> int | None:
    """The row an HTTP status means in every command (credentials, upgrade), or ``None`` when it depends on the command."""
    if status in _CREDENTIAL_STATUSES:
        return CREDENTIALS
    if status == client_handshake.UPGRADE_STATUS:
        return UPGRADE_REQUIRED
    return None


class UsageParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose command-line errors exit :data:`USAGE`, not argparse's 2 (which no command returns).

    Sub-parsers created through ``add_subparsers`` use the parent's class, so one root parser covers a whole command tree.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f'{self.prog}: error: {message}', file=sys.stderr)
        raise SystemExit(USAGE)


def guarded(command: str, body: Callable[[], int]) -> int:
    """``body()``'s exit code, or :data:`INTERNAL_FAULT` with the traceback when it raises.

    An exception the interpreter handles itself exits 1, the code of a failed policy, so a defect would read as a
    finding about the agent. ``KeyboardInterrupt`` and ``SystemExit`` pass through untouched.
    """
    try:
        return body()
    except Exception:
        print(f'{command}: internal error. Exit {INTERNAL_FAULT} means nothing was concluded; the traceback below is a defect in the SDK.', file=sys.stderr)
        traceback.print_exc()
        return INTERNAL_FAULT
