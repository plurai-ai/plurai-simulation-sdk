"""``plurai job status <id>`` and ``plurai job wait <id>``: read one backend job, or block until it ends.

Both read ``GET /jobs/{id}``, which returns the envelope the MCP ``job_status`` tool returns
(``kind``, ``status``, ``phase``, ``elapsed_seconds``, ``seconds_since_progress``, ``error``,
``detail``). A body is an envelope only when it is a JSON object with non-empty string ``kind`` and
``status``: a proxy or a framework answers with JSON too. The backend serves the envelope of an id it
does not know (``kind=unknown``, ``status=failed``) as a 404, so the body, not the status, tells an
unknown id from a missing route. Redirects are followed, because an edge that upgrades http to
https redirects a healthy read; the envelope check still refuses a login page at the end of the
chain. Target and bearer resolve from the configuration
(:func:`~plurai.ingest._command.resolve_target`), never from an argument.

Nothing of a body is printed raw. An envelope is projected (:func:`~plurai.customer_safe.project_job`):
only the fields the wire model declares survive, ``error`` is scrubbed, and ``detail`` loses its
internal-only names. A failed read prints the backend's scrubbed message and the ``error_id`` to quote
(:func:`~plurai.customer_safe.describe_failure`); a body that is not JSON adds nothing but its status.

The id is checked locally, before the target resolves or any request leaves (:func:`is_job_id`):
an id that is not a shape the backend issues would change the request path (``a/b`` is a deeper
route, ``a?b`` a query) and the 404 that comes back carries no envelope, which reads as a backend
that predates ``GET /jobs/{id}``.

``job status`` makes one read and prints the envelope as JSON; a non-zero exit prints nothing.

``job wait`` reads every :data:`INTERVAL_SECONDS` until the job is terminal or ``--max-seconds``
of wall clock is spent (clamped to :data:`MIN_BUDGET_SECONDS`..:data:`MAX_BUDGET_SECONDS`). The
ceiling keeps one invocation, plus one in-flight read and one health probe, inside a 600 s
tool-call limit; a job that outlives it is not an error, and a second invocation keeps waiting.
stdout is the transcript the caller reads: one ``[HH:MM:SS] <status> — phase=<phase> (elapsed <n>s
per server)`` line per status or phase change, the stall notes, and the verdict lines; lines of
its own carry the ``job_wait:`` prefix, and backend-state diagnostics go to stderr. The exit code
is the result, so invoke it bare: a pipeline reports the last command's code.

Two readers read that transcript: a person at a shell or in CI, and the model that drives the plugin. Only the model
gets instructions addressed to it (classify the error yourself, the MCP tools, the skill's phase costs): a caller sets
``PLURAI_READER=model`` (:data:`READER_ENV`) for that. Any other value, or none, is a person. An environment variable,
not a flag, so a plugin and an SDK of different versions still agree: an older SDK ignores it, where it would refuse an
unknown flag. Both streams reach both readers, so a split by stream would hide nothing from either.

It escalates on a STALL, not on duration: ``seconds_since_progress`` against a threshold per kind
and phase (:func:`stall_threshold`), because the long phases publish no progress while they work
and a flat bar reads a healthy job as stuck. :data:`BAD_READS_BEFORE_PROBE` bad reads in a row, or
a spent budget, trigger a ``/healthz`` probe, and only a 2xx on that path licenses "the job is
slow"; anything else means the job's state is unknown. A gateway code there (:data:`GATEWAY_CODES`)
is a proxy that cannot reach the backend for now, so the bad reads restart and the wait polls on
until the budget ends.

``POLL_JOB_INTERVAL`` and ``POLL_JOB_MIN_SECONDS`` shorten the nap and the budget floor for
hermetic tests. They are honoured only against a loopback backend: inherited by a real session,
they would turn a minutes-cadence wait into a tight poll of a shared box.

EXIT CODES — job status (see :mod:`plurai.exit_codes`)
   0  OK: the envelope was read
  77  CREDENTIALS: nothing is configured, or the bearer was rejected (401/403)
  65  REFUSED: the configured target is not a usable URL
  69  TRANSPORT: transport failure, a 2xx whose body is not a job envelope, an unexpected client error,
      or any other status: the job's state is unknown
  13  JOB_FAILED: the gateway refused the read for credit: the job's state is unknown
  12  ROUTE_MISSING: a 404/405 with no job envelope; this backend predates ``GET /jobs/{id}``
  15  NOT_FOUND: the backend knows no such job (a 404/405 that carries the envelope)
  10  UPGRADE_REQUIRED: the backend refuses this client version (426)
  64  USAGE: no job id, or one that is not a shape the backend issues (nothing is sent)

EXIT CODES — job wait
   0  OK: job done
  13  JOB_FAILED: the job failed, or the gateway refused the read for credit: the job's state is unknown, and only the
      printed line tells the two apart
  15  NOT_FOUND: the backend knows no such job
  14  STILL_RUNNING: the wait budget is spent, the job is still running, ``/healthz`` answers 2xx: re-invoke to keep waiting
  12  ROUTE_MISSING (as for ``job status``): read the job through the MCP ``job_status`` tool
  77  CREDENTIALS, 65 REFUSED, 10 UPGRADE_REQUIRED: as for ``job status``
  69  TRANSPORT: backend not responding or unhealthy, or up with the job route faulting, or an unexpected
      client error: state unknown
  64  USAGE: no job id, or one that is not a shape the backend issues (nothing is sent)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TextIO

import httpx

from plurai import client_handshake, exit_codes
from plurai._endpoints import CREDENTIAL_FIX, is_loopback_url
from plurai.credit import CREDIT_REFUSED_STATUS, CreditBlock, credit_block_from_body, credit_block_or_unknown, credit_sentence
from plurai.customer_safe import describe_failure, is_job, project_job, response_json
from plurai.ingest._command import (
    CommandError,
    client,
    resolve_target,
    say,
    send_with_retry,
)
from plurai.ingest.backend_commands import HEALTHZ_PATH, PROBE_TIMEOUT_SECONDS, is_alive, probe
from plurai.login._environments import billing_url_for


#: One ``job status`` read.
STATUS_TIMEOUT_SECONDS = 30.0
#: One read inside the wait loop; short enough that four failures reach the probe in about a minute.
READ_TIMEOUT_SECONDS = 20.0
INTERVAL_SECONDS = 15
MIN_BUDGET_SECONDS = 30
MAX_BUDGET_SECONDS = 540
#: Consecutive bad reads that mean a wedged box rather than a slow one.
BAD_READS_BEFORE_PROBE = 4
#: ``/healthz`` codes a proxy answers when it cannot reach the backend: the wait polls on through them.
GATEWAY_CODES = frozenset({'502', '503', '504'})
INTERVAL_ENV = 'POLL_JOB_INTERVAL'
MIN_SECONDS_ENV = 'POLL_JOB_MIN_SECONDS'
#: Set to :data:`MODEL_READER` by a caller whose stdout a model reads; see the module docstring.
READER_ENV = 'PLURAI_READER'
MODEL_READER = 'model'

_STATUS = 'job_status'
_WAIT = 'job_wait'

#: The ids ``GET /jobs/{id}`` resolves: a job id is ``uuid4().hex``; an env definition and an edit
#: turn are keyed by a canonical ``str(uuid4())``. Lowercase only, as the backend mints them.
_JOB_ID = re.compile(r'[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
#: Longest prefix of a rejected id echoed in the usage message.
_ECHO_CHARS = 60


is_envelope = is_job


def _field(envelope: dict[str, Any], key: str) -> str:
    """A lifecycle field as the transcript prints it: absent or null is empty, one line always."""
    value = envelope.get(key)
    return '' if value is None else str(value).replace('\n', ' ').replace('\r', ' ')


def stall_threshold(kind: str, phase: str) -> int | None:
    """Seconds without progress before THIS kind and phase has stalled; ``None`` for a phase with no
    usable progress clock.

    Env-gen publishes nothing to its progress clock while a tool executes or its contract tests
    run (``generating``, ``verifying``; no other kind uses those names). The eval and document
    authoring phases are one uninterrupted SDK session each, quoted at 10-30 minutes by their
    skills. Env ``deploying`` covers the env-contract gate, which ticks once per check and has
    single checks that run for many minutes on a large world. The scenario walk's ``evolving``
    phase publishes no usable clock at all.
    """
    if phase in ('generating', 'verifying') or (kind, phase) in (('env', 'deploying'), ('evals', 'calling_skill'), ('scenarios', 'generating_documents')):
        return 1800
    if phase == 'evolving':
        return None
    return 900


@dataclass(frozen=True)
class JobRead:
    """One ``GET /jobs/{id}``: ``status`` 0 is no response; ``envelope`` is set only for a body
    with the envelope shape, whatever the status. ``text`` is the customer-safe line for a failure; for a 426 it is
    the upgrade message."""

    status: int
    envelope: dict[str, Any] | None
    text: str = ''
    credit: CreditBlock | None = None


def is_job_id(job_id: str) -> bool:
    """True when ``job_id`` has a shape the backend issues: 32 lowercase hex digits, or a canonical
    lowercase dashed UUID. Every such id is one safe path segment."""
    return _JOB_ID.fullmatch(job_id) is not None


def _malformed(job_id: str, usage: str) -> str | None:
    """The one-line usage message for a missing or malformed id, or ``None`` for a good one."""
    if not job_id.strip():
        return usage
    if is_job_id(job_id):
        return None
    shown = repr(job_id if len(job_id) <= _ECHO_CHARS else f'{job_id[:_ECHO_CHARS]}...')
    return f'{shown} is not a job id (32 lowercase hex digits, or a lowercase dashed UUID); nothing was sent. {usage}'


def job_path(job_id: str) -> str:
    """``/jobs/<id>`` for an id that passes :func:`is_job_id`, so it is always one path segment.

    Raises:
        ValueError: the id is not a shape the backend issues.
    """
    if not is_job_id(job_id):
        raise ValueError(f'not a job id: {job_id!r}')
    return f'/jobs/{job_id}'


async def read_job(http: httpx.AsyncClient, job_id: str, timeout: float) -> JobRead:
    """One read, each attempt bounded by ``timeout`` end to end; a 429 is waited out first. Never raises for a transport failure."""
    try:
        response = await send_with_retry(lambda: http.get(job_path(job_id)), attempts=1, delay=0.0, timeout=timeout)
    except (httpx.HTTPError, TimeoutError) as error:
        return JobRead(0, None, repr(error))
    body = response_json(response)
    if (refusal := client_handshake.refusal_from(response.status_code, body)) is not None:
        return JobRead(response.status_code, None, client_handshake.upgrade_message(refusal))
    credit = credit_block_from_body(body) if response.status_code == CREDIT_REFUSED_STATUS else None
    return JobRead(response.status_code, project_job(body) if is_envelope(body) else None, describe_failure(response), credit)


def classify_status_read(read: JobRead, url: str, *, billing_url: str = '') -> dict[str, Any]:
    """The envelope of a ``job status`` read, or the classified failure.

    Raises:
        CommandError: per the ``job status`` exit table.
    """
    if read.status == 0:
        raise CommandError(exit_codes.TRANSPORT, f'transport failure reaching {url}: {read.text}')
    if read.status in (404, 405):
        if read.envelope is not None:
            raise CommandError(
                exit_codes.NOT_FOUND, f'the backend knows no such job ({url} answered {read.status} with the envelope): {_field(read.envelope, "error")}'
            )
        raise CommandError(
            exit_codes.ROUTE_MISSING, f'{url} returned {read.status} with no job envelope: this backend predates GET /jobs/{{id}}; use the MCP job_status tool'
        )
    if read.status in (401, 403):
        raise CommandError(exit_codes.CREDENTIALS, f'{url} returned {read.status}: the bearer was rejected')
    if read.status == client_handshake.UPGRADE_STATUS:
        raise CommandError(exit_codes.UPGRADE_REQUIRED, f'{url}: {read.text}')
    if read.credit is not None:
        # The gateway refused the read itself for credit: no retry changes that, so this is not the transport failure below.
        raise CommandError(exit_codes.JOB_FAILED, credit_sentence(read.credit.code, resets_at=read.credit.resets_at, billing_url=billing_url))
    if not 200 <= read.status < 300:
        raise CommandError(exit_codes.TRANSPORT, f'{url} returned HTTP {read.status}; the job state is unknown: {read.text}')
    if read.envelope is None:
        raise CommandError(exit_codes.TRANSPORT, f'{url} returned HTTP {read.status} with no job envelope in the body: {read.text}')
    return read.envelope


async def _status(job_id: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
    base, token = resolve_target()
    async with client(base, token, STATUS_TIMEOUT_SECONDS, transport, follow_redirects=True) as http:
        read = await read_job(http, job_id, STATUS_TIMEOUT_SECONDS)
    return classify_status_read(read, f'{base}{job_path(job_id)}', billing_url=billing_url_for(base))


def run_job_status(
    job_id: str,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai job status`` command: returns its exit code."""
    err = stderr if stderr is not None else sys.stderr
    if problem := _malformed(job_id, 'usage: plurai job status <job_id>'):
        say(_STATUS, problem, err)
        return exit_codes.USAGE
    try:
        envelope = asyncio.run(_status(job_id, transport))
    except CommandError as error:
        say(_STATUS, str(error), err)
        return error.code
    except Exception as error:  # noqa: BLE001 - an uncaught exception would exit 1, the policies-failed row
        say(_STATUS, f'unexpected error: {error!r}', err)
        return exit_codes.TRANSPORT
    print(json.dumps(envelope, ensure_ascii=False), file=stdout if stdout is not None else sys.stdout)
    return exit_codes.OK


def _seam(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, '')), 1)
    except ValueError:
        return default


def wait_cadence(base: str, max_seconds: int, err: TextIO) -> tuple[int, int]:
    """``(budget, interval)`` for one wait: the test seams on loopback only, then the clamp."""
    floor, interval = MIN_BUDGET_SECONDS, INTERVAL_SECONDS
    if os.environ.get(MIN_SECONDS_ENV) or os.environ.get(INTERVAL_ENV):
        if is_loopback_url(base):
            floor, interval = _seam(MIN_SECONDS_ENV, MIN_BUDGET_SECONDS), _seam(INTERVAL_ENV, INTERVAL_SECONDS)
        else:
            say(_WAIT, f'{MIN_SECONDS_ENV} / {INTERVAL_ENV} ignored: they are test seams, honoured only against a loopback backend.', err)
    return min(max(max_seconds, floor), MAX_BUDGET_SECONDS), interval


def _stamp() -> str:
    return time.strftime('%H:%M:%S')


class _Wait:
    """One ``job wait``: the loop state and the transcript it writes. ``for_model`` adds the lines addressed to a model."""

    def __init__(self, job_id: str, base: str, http: httpx.AsyncClient, out: TextIO, err: TextIO, *, for_model: bool) -> None:
        self.job_id, self.base, self.http, self.out, self.err = job_id, base, http, out, err
        self.for_model = for_model
        self.url = f'{base}{job_path(job_id)}'
        self.envelope: dict[str, Any] = {}
        self.last = JobRead(0, None)
        self.stall_noted = self.quiet_noted = False

    def f(self, key: str) -> str:
        return _field(self.envelope, key)

    def line(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    def note(self, text: str) -> None:
        self.line(f'{_WAIT}: {text}')

    def diag(self, text: str) -> None:
        say(_WAIT, text, self.err)

    async def run(self, budget: int, interval: int, sleep: Callable[[float], Awaitable[Any]], clock: Callable[[], float]) -> int:
        start = clock()
        bad_reads = 0
        last_sig: str | None = None
        while True:
            self.last = await read_job(self.http, self.job_id, READ_TIMEOUT_SECONDS)
            status = self.last.status
            if status in (404, 405):
                if self.last.envelope is not None:
                    self.envelope = self.last.envelope
                    return self.unknown_job()
                return self.route_missing()
            if status in (401, 403):
                self.diag(f'{self.url} returned {status}: the bearer was rejected. The token does not match the backend at {self.base}; {CREDENTIAL_FIX}.')
                return exit_codes.CREDENTIALS
            if status == client_handshake.UPGRADE_STATUS:
                self.diag(f'{self.url}: {self.last.text}')
                return exit_codes.UPGRADE_REQUIRED
            if self.last.credit is not None:
                # The gateway refused the poll itself: waiting longer cannot change what the organization's credit allows.
                self.line(self.refused_read_line(self.last.credit))
                return exit_codes.JOB_FAILED
            if status == 200 and self.last.envelope is not None:
                bad_reads = 0
                self.envelope = self.last.envelope
                sig = f'{self.f("status")}/{self.f("phase")}'
                if sig != last_sig:
                    self.line(self.progress_line())
                    last_sig = sig
                if self.f('status') == 'done':
                    if self.for_model:
                        detail = f'Read job_status({self.job_id}) through MCP for the full detail envelope.'
                    else:
                        detail = f'`plurai job status {self.job_id}` prints the full detail envelope.'
                    self.line(f'[{_stamp()}] done — job {self.job_id} is terminal. {detail}')
                    return exit_codes.OK
                if self.f('status') == 'failed':
                    if self.f('terminal_reason') == 'credit':
                        self.line(self.credit_line(credit_block_or_unknown(self.envelope.get('credit'))))
                        return exit_codes.JOB_FAILED
                    self.line(self.failed_line())
                    if self.for_model:
                        self.note('classify this error YOURSELF before reporting it; never paste it at the user.')
                    return exit_codes.JOB_FAILED
                self.stall_notes()
            else:
                bad_reads += 1

            if bad_reads >= BAD_READS_BEFORE_PROBE:
                code = await self.healthz()
                left = budget - int(clock() - start)
                if code not in GATEWAY_CODES or left <= 0:
                    return self.unknown_state(code, f'failing repeatedly (last HTTP {self.last.status:03d})')
                # A gateway in front of the backend could not reach it: transient, and the job runs on.
                self.note(f'[{_stamp()}] healthz {code} from the gateway in front of the backend (last job read HTTP {self.last.status:03d});')
                self.note(f'still polling, {left}s of budget left.')
                bad_reads = 0

            elapsed = int(clock() - start)
            if elapsed >= budget:
                code = await self.healthz()
                if is_alive(code):
                    return self.slow(code, elapsed)
                return self.unknown_state(code, f'budget of {elapsed}s spent with the job still {self.f("status") or "unread"}')
            await sleep(min(interval, budget - elapsed))

    async def healthz(self) -> str:
        return await probe(self.http, HEALTHZ_PATH, PROBE_TIMEOUT_SECONDS)

    def progress_line(self) -> str:
        text = f'[{_stamp()}] {self.f("status") or "unknown"}'
        if self.f('phase'):
            text += f' — phase={self.f("phase")}'
        if self.f('elapsed_seconds'):
            text += f' (elapsed {self.f("elapsed_seconds")}s per server)'
        return text

    def failed_line(self) -> str:
        reason, error, error_id = self.f('terminal_reason'), self.f('error'), self.f('error_id')
        text = f'[{_stamp()}] failed — job {self.job_id}'
        if reason:
            text += f' ({reason})'
        if error:
            text += f': {error}'
        if error_id and error_id not in error:
            text += f' (error id: {error_id})'
        return text

    def credit_line(self, block: CreditBlock) -> str:
        """The failure line of a job that ended on a credit refusal: the kind's sentence and nothing else.

        The sentence is the one the run prints, so the words are the same wherever the refusal is met. The job's generic
        error text and its error id stay on the backend, and the caller has nothing to classify: the stop is the
        organization's credit, not a fault. A block that names no known kind reads as ``credit-refused``.
        """
        return f'[{_stamp()}] failed — job {self.job_id} (credit): {self.credit_kind_sentence(block)}'

    def refused_read_line(self, block: CreditBlock) -> str:
        """The line of a read the gateway refused for credit. The refusal is about the organization and says nothing of
        the job, which may have finished, so the line never calls it failed. Callers read the sentence after ``(credit): ``,
        the ending it shares with :meth:`credit_line`."""
        return f'[{_stamp()}] could not read the status of job {self.job_id} (credit): {self.credit_kind_sentence(block)}'

    def credit_kind_sentence(self, block: CreditBlock) -> str:
        return credit_sentence(block.code, resets_at=block.resets_at, billing_url=billing_url_for(self.base))

    def stall_notes(self) -> None:
        """Print the stall or no-clock note this read earns, each once per wait: the no-clock
        notice keeps the wait from being silent and points at ``elapsed_seconds``, the one number
        that means anything in such a phase."""
        since = self.f('seconds_since_progress').split('.', 1)[0]
        if self.stall_noted or not (since.isascii() and since.isdigit()):
            return
        threshold = stall_threshold(self.f('kind'), self.f('phase'))
        if threshold is None:
            if not self.quiet_noted:
                elapsed = self.f('elapsed_seconds')
                self.note(f'NOTE — phase={self.f("phase")} does not publish a usable progress clock, so')
                self.note('seconds_since_progress is NOT a stall signal here. Judge this job by')
                cost = ' against what the skill says the phase costs' if self.for_model else ''
                self.note(f'elapsed_seconds{f" ({elapsed}s)" if elapsed else ""}{cost}.')
                self.quiet_noted = True
        elif int(since) >= threshold:
            self.note(f'NOTE — seconds_since_progress={since} (>= {threshold}s for {self.f("kind") or "job"}/{self.f("phase") or "?"}). The job has not MOVED;')
            self.note('that is a stall signal, unlike a merely long elapsed time.')
            self.stall_noted = True

    def unknown_job(self) -> int:
        self.line(self.failed_line())
        if not self.for_model:
            self.note(f'the backend at {self.base} knows no job with this id: GET /jobs/{{id}} answered with the job envelope, so the route')
            self.note('exists. Check that `plurai backend identity` names the deployment that started the job. If it does, the id')
            self.note('is mistyped or expired, or a restart of the backend lost it.')
            return exit_codes.NOT_FOUND
        self.note('this is an UNKNOWN JOB ID on a CURRENT backend, not a missing route: GET /jobs/{id} answered with the job envelope')
        self.note(
            f'(kind={self.f("kind") or "unknown"}, status={self.f("status") or "failed"}). Rule OUT the WRONG BACKEND first: a healthy box that never held'
        )
        self.note(f'this job answers exactly like this. This poll went to {self.base}; the job was launched through the MCP server, whose URL')
        self.note('was fixed at session launch and may differ. Compare `plurai backend identity` with the MCP backend_identity() tool: a')
        self.note('different deployment is your answer; the same deployment with a different instance_id means a restart lost the id;')
        self.note('only if they match was the id mistyped or expired. Never re-deploy the backend for this. Classify it YOURSELF; never paste it at the user.')
        return exit_codes.NOT_FOUND

    def route_missing(self) -> int:
        self.note(f'{self.url} returned {self.last.status} with no job envelope in the body: this backend')
        if not self.for_model:
            self.note('predates GET /jobs/{id}, so `plurai job wait` cannot wait on this job. Re-deploy the backend to get the blocking wait.')
            return exit_codes.ROUTE_MISSING
        self.note('predates GET /jobs/{id}. FALL BACK to the MCP job_status tool, with serial waits between')
        self.note(f'reads: job_status(id={self.job_id}). TERMINAL STATUSES ARE done AND failed, nothing else.')
        self.note('Re-deploy the backend to get the blocking wait.')
        return exit_codes.ROUTE_MISSING

    def slow(self, code: str, elapsed: int) -> int:
        phase = self.f('phase')
        server = ''.join(
            [
                f' Server says elapsed={self.f("elapsed_seconds")}s.' if self.f('elapsed_seconds') else '',
                f' seconds_since_progress={self.f("seconds_since_progress")}.' if self.f('seconds_since_progress') else '',
            ]
        )
        self.line(f'[{_stamp()}] still {self.f("status") or "running"}{f"/{phase}" if phase else ""} after {elapsed}s of waiting in this invocation.')
        self.note(f'backend is UP (healthz {code}) — the job is SLOW, not wedged.{server}')
        self.note(f're-invoke `plurai job wait {self.job_id}` to keep waiting.')
        return exit_codes.STILL_RUNNING

    def unknown_state(self, code: str, cause: str) -> int:
        """Exit TRANSPORT, naming which failure this is. A healthz 2xx does not license "the job route is
        faulting" on its own: the message has to match the class of the job read's last answer.
        Without a 2xx, the message names the host THIS wait called, which is not necessarily where
        the job was launched."""
        last = self.last.status
        if is_alive(code):
            head = f'BACKEND UP (healthz {code}) but {self.url} is {cause} —'
            if last == 0:
                self.diag(f"{head} the job read got NO response (HTTP 000), so the job's state is UNKNOWN. RE-INVOKE ONCE before")
                self.diag(f'treating the backend as broken: 000 is transport (a read slower than the {READ_TIMEOUT_SECONDS:.0f}s cap, or a network drop),')
                self.diag('not proof of a route fault.')
            elif last >= 400:
                self.diag(f"{head} the job route answered HTTP {last}, so the job's state is UNKNOWN. This is a backend-side fault, not a slow job.")
            else:
                self.diag(f"{head} the job route answered HTTP {last}, which is not a job envelope, so the job's state is UNKNOWN.")
                self.diag('That is not an error status: suspect a proxy, a redirect to a login page, or a different service on this port.')
            return exit_codes.TRANSPORT
        if code == '000':
            verdict = f'NOT RESPONDING at {self.base} (healthz {code}; {cause})'
        else:
            verdict = f'DOWN/UNHEALTHY at {self.base} (healthz HTTP {code}; {cause})'
        if code in GATEWAY_CODES:
            self.diag(f"BACKEND {verdict} — the job's state is UNKNOWN. A gateway in front of the backend answered {code}, and the")
            self.diag('job may still be running behind it: RE-INVOKE ONCE before treating the backend as down.')
        else:
            self.diag(f"BACKEND {verdict} — the job's state is UNKNOWN. This is a WEDGED/DOWN backend, NOT a slow job: waiting longer will not resolve it.")
        if self.for_model:
            self.diag(f'Check {self.base} — the host THIS POLL CALLED, which is not necessarily where the job was launched: the MCP server')
            self.diag('dials the URL fixed at session launch, which may differ. Confirm they match before concluding this box is down.')
        else:
            self.diag(f'Check {self.base} — the host THIS POLL CALLED, which is not necessarily where the job was launched. Confirm')
            self.diag('it is the deployment that started the job (`plurai backend identity`) before concluding this box is down.')
        return exit_codes.TRANSPORT


async def wait_job(
    job_id: str,
    max_seconds: int,
    out: TextIO,
    err: TextIO,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Block until the job is terminal or the budget is spent; return the exit code.

    ``sleep`` and ``clock`` are the test seams for the wall clock.

    Raises:
        CommandError: 1 for a refused target.
    """
    base, token = resolve_target()
    budget, interval = wait_cadence(base, max_seconds, err)
    async with client(base, token, READ_TIMEOUT_SECONDS, transport, follow_redirects=True) as http:
        return await _Wait(job_id, base, http, out, err, for_model=os.environ.get(READER_ENV) == MODEL_READER).run(budget, interval, sleep, clock)


def run_job_wait(
    job_id: str,
    max_seconds: int,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai job wait`` command: returns its exit code."""
    err = stderr if stderr is not None else sys.stderr
    if problem := _malformed(job_id, 'usage: plurai job wait <job_id> [--max-seconds N]'):
        say(_WAIT, problem, err)
        return exit_codes.USAGE
    try:
        return asyncio.run(wait_job(job_id, max_seconds, stdout if stdout is not None else sys.stdout, err, transport=transport))
    except CommandError as error:
        say(_WAIT, str(error), err)
        return error.code
    except Exception as error:  # noqa: BLE001 - an uncaught exception would exit 1, the policies-failed row
        say(_WAIT, f"unexpected error: {error!r}; the job's state is UNKNOWN", err)
        return exit_codes.TRANSPORT


__all__ = [
    'MODEL_READER',
    'READER_ENV',
    'JobRead',
    'is_envelope',
    'is_job_id',
    'job_path',
    'read_job',
    'run_job_status',
    'run_job_wait',
    'stall_threshold',
    'wait_job',
]
