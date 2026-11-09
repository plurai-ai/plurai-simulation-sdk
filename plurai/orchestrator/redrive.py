"""Transient-failure scenario re-drive for the REST thread flow.

When the backend box restarts mid-run, every in-flight scenario dies. The server
persists nothing for a thread — threads live in memory only by design, and the
connect-mode mock is a subprocess whose boot revival redeploys it FRESH FROM SEED
— so there is nothing to re-attach to. **The scenario is the unit and the batch is
the checkpoint**: completed scenarios' artifacts are the ledger, and an interrupted
scenario is *re-driven* once by the client as a brand-new thread.

A dead thread is one case, not the only one: the admission test is that the
failure was transient — that nothing about the scenario, the agent or the world caused
it — so a fresh thread has a real chance where this one had none. A stalled agent turn
qualifies on exactly those terms (see ``RedriveTrigger.TURN_TIMEOUT``).

Three rules shape this module:

* **Zero prose matching.** A failure is classified from typed signals only — an
  exception type, an HTTP status, a URL path. Never from message text, which
  drifts silently and would re-drive on unrelated failures.
* **The gate spends nothing.** A re-drive fired the instant the old thread died
  would race the box's own boot redeploy straight into the 503 → stub → integrity
  failure it is trying to avoid, burning the one attempt on a race we can see
  coming. So readiness is proven BEFORE the budget is touched; if readiness never
  arrives, the ORIGINAL error surfaces and the budget is unspent.
* **Re-drives are loud.** One warning log naming the dead thread and the trigger,
  plus a provenance block in the scenario's own artifacts — so an absorbed
  false-positive stays visible in the output, not only in a log that may be gone.

Billing needs nothing built here: a thread bills only on finalize success, so an
interrupted thread provably billed nothing, and a re-drive is a fresh thread with a
new ``run_id``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import httpx

from plurai._protocol import (
    USER_TURN_ENVELOPE,
    AgentTurnTimeout,
    FinalizeJobFailed,
    HarnessChildExited,
    MockInitStateFault,
    MockRuntimeFaultRaised,
    MockUnreachableError,
)
from plurai.capacity import capacity_block_of, unavailable_block_of
from plurai.credit import credit_block_of
from plurai.orchestrator._retry import TRANSIENT_STATUS_CODES, ServerHTTPError

if TYPE_CHECKING:
    from plurai._protocol import MockMode
    from plurai.orchestrator.client import OrchestratorClient

logger = logging.getLogger(__name__)

#: Re-drive attempts per scenario. In-memory and deliberately not configurable:
#: the persisted, tunable re-drive budget belongs to the SERVER, for jobs whose
#: work is expensive to redo. Here a fresh CLI invocation is a human action and is
#: the natural second attempt, so losing this count to a client crash is correct.
REDRIVE_BUDGET = 1

#: ONE fixed wall-clock bound covering the WHOLE readiness gate (server ``/health``
#: plus, in connect mode, the mock route). Generous because it is waiting out a box
#: boot + a mock redeploy; fixed because a knob nobody sets is a knob that rots. An old box with no native ``/health``
#: never satisfies the gate, so the original failure surfaces with the budget unspent.
READINESS_BOUND_SECONDS = 300.0

#: Poll cadence inside the gate.
READINESS_POLL_INTERVAL_SECONDS = 3.0

#: The ``/mock`` 503 that will NOT self-heal: an env held because an interrupted
#: edit session still owes a finalize, i.e. blocked on a HUMAN up to the session
#: TTL. Same body shape as the ``restarting`` 503 that clears in seconds, so
#: without this discriminator the gate would wait out its whole bound on a
#: condition no amount of waiting resolves — and then misattribute the failure.
MOCK_HOLD_CODE = 'awaiting_edit_finalize'


class RedriveTrigger(StrEnum):
    """The typed signal that classified a failure as re-drivable.

    Most members name a dead thread (the box restarted underneath it); the test
    every member must pass is narrower than that and is what governs additions:
    the failure is TRANSIENT — nothing about the scenario, the agent or the world
    made it fail — so a fresh thread has a real chance where this one had none.
    """

    #: ``mock_mode='connect'`` could not reach the deployed mock — at init, or
    #: mid-scenario via tool calls left unanswered the way a restart does. The signature of a box
    #: that went away underneath a running scenario. Also a mock that was dialed
    #: but could not serve the scenario's setup (``MockInitStateFault``: a state
    #: call timing out under load); the provenance keeps that kind.
    CONNECT_EXHAUSTION = 'connect_exhaustion'
    #: A thread-scoped route answered 404 for a thread we ourselves created.
    THREAD_404 = 'thread_404'
    #: The finalize job failed (or its job id 404'd) AND a ``GET /threads/{id}``
    #: probe confirms the thread itself is gone.
    FINALIZE_THREAD_GONE = 'finalize_thread_gone'
    #: A DIALOG TURN blew its wall-clock budget and the child was killed. The thread
    #: here is usually ALIVE — what stalled is the provider behind the customer's agent
    #: — so this is the transient-not-dead member: re-drivable for the same reason as
    #: the rest, and bounded by the same single budget so a genuinely slow agent costs
    #: one extra attempt, never a loop. (An agent that needs longer turns has a knob
    #: for it: ``--agent-turn-timeout``.) Strictly the ``user_turn`` envelope: the same
    #: budget also guards init and finalize, and a stall in those is deterministic
    #: rather than transient — see :func:`classify_restart_failure`.
    #:
    #: The ONE trigger a phase can switch off: a qualification pass passes
    #: ``allow_turn_timeout_redrive=False`` so its per-turn wall is the real wall rather
    #: than half of one. Every other member re-drives in every phase.
    TURN_TIMEOUT = 'turn_timeout'
    #: The backend's mock runtime could not answer a mocked tool and typed the fault TRANSIENT (a
    #: provider 5xx/429, a dropped connection, a lock or budget race). A fresh thread seeds a fresh
    #: mock world, so a re-run cannot double-apply a write. It fails fast (inside the runtime's own
    #: budget), so unlike TURN_TIMEOUT it re-drives in every phase. A deterministic one never does.
    MOCK_RUNTIME_FAULT = 'mock_runtime_fault'
    #: ``POST /threads`` itself answered a gateway status (502/503/504) — the ONE member that fires
    #: with no thread id, because create is what failed. A gateway status is the proxy's answer, not
    #: the app's refusal (a bad scenario or eval set id is a 4xx, an app fault a 500 — neither is
    #: here), so a fresh create has a real chance. The HTTP layer cannot retry it: the POST is not
    #: idempotent, and only a 503 is provably unforwarded (``_retry.retry_unforwarded``). If the 502
    #: came AFTER the app applied the create, the first thread is an orphan: it never finalizes, so
    #: it never bills, and the server's sweep files its terminal.
    CREATE_TRANSIENT = 'create_transient'
    #: The child was SIGKILLed mid-turn while sibling scenarios ran beside it (``--concurrency`` > 1):
    #: the OOM reaper's signature under load the siblings supplied. A fresh attempt runs under
    #: different pressure, so it has a real chance. At concurrency 1 the pressure is this scenario's
    #: own and would recur, and a child that exited on its own exits again — neither re-drives.
    CHILD_KILLED = 'child_killed'
    #: Any other raise the backend TYPED transient (``transient is True`` on the raising object: a
    #: ``ServerHTTPError`` 5xx/429 on a thread route, such as a ``/user-turn`` 500 — and nothing read from its
    #: text). The request was not retried at the HTTP layer (a non-idempotent call may have been applied),
    #: so the scenario is the unit: one fresh thread. A ``False`` or absent ``transient`` never re-drives.
    TRANSIENT_FAULT = 'transient_fault'
    #: The world was EMPTY when the dialog began: the mock runtime faulted before any turn completed and no tool
    #: call had returned data, while the server seeded the scenario's world with entities (:func:`seed_fault`). The
    #: session was handed out before its seed committed — a race, not the scenario — so a fresh thread re-seeds.
    SEED_FAULT = 'seed_fault'
    #: A request to the backend waited out its whole transient budget (:func:`backend_outage`: a refused or reset connection,
    #: or a 502, 503 or 504, for about a minute): the backend is down or restarting, which says nothing about the scenario.
    #: The readiness gate waits for the backend to come back before one fresh thread runs the scenario again. Fires with or
    #: without a thread id, since a create meets an outage like any other request.
    BACKEND_OUTAGE = 'backend_outage'


def backend_outage(exc: BaseException) -> bool:
    """Whether *exc* is a request to the backend that waited out the transient budget (the block ``tag_outage`` puts on it,
    cause ``transport`` or ``status``). A hosted mock's fault is not one, whatever block it carries: it is one scenario's
    mock turn, re-driven as a mock runtime fault (:func:`static_trigger`)."""
    if isinstance(exc, MockRuntimeFaultRaised):
        return False
    block = unavailable_block_of(exc)
    return block is not None and block.cause != 'backend'


@dataclass(frozen=True, slots=True, kw_only=True)
class RedriveProvenance:
    """The record of a re-drive, persisted into the re-driven scenario's artifacts.

    One residual risk is accepted explicitly: a genuine id-mangling bug that 404s
    would be absorbed into one paid re-drive. This block is the mitigation —
    it makes an absorbed anomaly readable in the artifact a human actually opens.

    ``original_kind`` / ``original_transient`` are the first attempt's typed cause, read off its
    raiser as ``error.json`` reads them (``None`` when the raiser declared none), so a consumer of a
    re-drive that also failed never has to regex ``original_error``. They are provenance only: the
    row's own ``kind`` stays the final attempt's.
    """

    #: ``None`` when the first attempt never got a thread (:attr:`RedriveTrigger.CREATE_TRANSIENT`).
    original_thread_id: str | None
    trigger: RedriveTrigger
    original_error: str
    original_kind: str | None
    original_transient: bool | None

    def to_artifact(self) -> dict[str, Any]:
        return {
            'original_thread_id': self.original_thread_id,
            'trigger': self.trigger.value,
            'original_error': self.original_error,
            'original_kind': self.original_kind,
            'original_transient': self.original_transient,
        }


#: The one thread-scoped sub-path whose 404 is NOT self-evident (see below).
_FINALIZE_JOB_SEGMENT = '/finalize/jobs/'


def _is_thread_scoped_404(exc: BaseException) -> bool:
    """Whether *exc* is a 404 from a ``/threads/...`` route, self-evidently.

    Typed twice over — the status code and the request's URL PATH — so a 404 from
    some unrelated endpoint (a mistyped scenario-set id, say) can never be read as
    a dead thread.

    This leans on the server's never-evict guarantee — the server never removes a
    thread — so on a healthy server a THREAD we created cannot 404, and the
    404 IS the restart signal.

    The ``/finalize/jobs/`` sub-path is deliberately excluded even though it lives
    under ``/threads/``: a finalize JOB carries no such guarantee — it can 404 on a
    perfectly healthy server — so its 404 is routed to the probe-gated branch
    instead. Without this carve-out an evicted job would re-drive (and re-bill) a
    scenario whose thread is alive and well.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code != 404:
        return False
    path = exc.request.url.path
    return path.startswith('/threads/') and _FINALIZE_JOB_SEGMENT not in path


def _is_create_gateway_status(exc: BaseException) -> bool:
    """Whether *exc* is ``POST /threads`` answering 502/503/504 — typed by status, method and URL path."""
    if not isinstance(exc, ServerHTTPError) or exc.status not in TRANSIENT_STATUS_CODES:
        return False
    return exc.request.method == 'POST' and exc.request.url.path == '/threads'


def _finalize_failed_terminally(exc: BaseException) -> bool:
    """Whether *exc* is a finalize outcome that a dead server would produce.

    Two shapes qualify, each typed: :class:`FinalizeJobFailed` (the job reported
    terminal ``failed``) and a 404 on the ``/finalize/jobs/`` poll (the job
    registry went with the process). Both are also reachable on a perfectly
    healthy server — a genuinely failed eval, an evicted job — which is exactly
    why this is only ever half the classification: the ``GET /threads/{id}``
    probe in :func:`classify_restart_failure` decides.

    Note what is NOT here: a finalize POLL TIMEOUT means the server is alive and
    still working, and a schema-drift error means it answered something we could
    not parse. Neither is a dead thread; neither re-drives.
    """
    if isinstance(exc, FinalizeJobFailed):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404 and _FINALIZE_JOB_SEGMENT in exc.request.url.path
    return False


#: The signal the OOM reaper sends (``HarnessChildExited.returncode == -9``).
_SIGKILL = 9


def static_trigger(exc: BaseException, *, allow_turn_timeout_redrive: bool, concurrency: int) -> RedriveTrigger | None:
    """The re-drive trigger *exc* matches from its TYPE alone — no probe, no I/O — else ``None``.

    The one pure predicate shared by the runner's classifier (:func:`classify_restart_failure`, which
    adds the probe-gated and count-gated triggers) and the orchestrator, which must know BEFORE it
    lands a failed dialog whether the runner would re-drive it instead: landing and re-driving are
    exclusive, and a second copy of this rule is how they would come to disagree.

    For TURN_TIMEOUT the ENVELOPE decides, not the type: one wall-clock budget guards init,
    user_turn and finalize alike, and only a stalled dialog turn is transient. An init stall is the
    customer's simulate.py being slow to build its agent and a finalize stall is the child hanging
    on its own teardown — both deterministic. No liveness probe pairs with this branch, and none
    should: the thread being ALIVE is the normal case for a stalled turn.
    """
    if isinstance(exc, MockUnreachableError | MockInitStateFault):
        return RedriveTrigger.CONNECT_EXHAUSTION
    if allow_turn_timeout_redrive and isinstance(exc, AgentTurnTimeout) and exc.envelope_type == USER_TURN_ENVELOPE:
        return RedriveTrigger.TURN_TIMEOUT
    if isinstance(exc, MockRuntimeFaultRaised) and (exc.transient or unavailable_block_of(exc) is not None):
        # The backend's unavailable answer on a mock turn is a provider blip behind one tool call: a fresh mock world has
        # the same chance a transient fault has. Credit and capacity refusals are the same answer on every try.
        return RedriveTrigger.MOCK_RUNTIME_FAULT
    if concurrency > 1 and isinstance(exc, HarnessChildExited) and exc.envelope_type == USER_TURN_ENVELOPE and exc.killed_by_signal == _SIGKILL:
        return RedriveTrigger.CHILD_KILLED
    if getattr(exc, 'transient', None) is True:
        return RedriveTrigger.TRANSIENT_FAULT
    return None


def seed_fault(exc: BaseException, *, turns_completed: int, seeded_entities: int, served_data: bool) -> bool:
    """Whether *exc* is an EMPTY WORLD at the start of the dialog (:attr:`RedriveTrigger.SEED_FAULT`), from typed
    facts only: a mock-runtime fault (whatever class the runtime gave it — it cannot tell an unseeded world from
    an empty one), raised before any turn completed, before any tool call returned data, on a scenario whose
    ``initial_world_state`` seeded at least one entity. Never the runtime's prose (``no tables`` is a model's
    sentence). A world that already served a row was not empty; a scenario seeded with nothing has no seed to race."""
    refused = any(read(exc) is not None for read in (credit_block_of, capacity_block_of, unavailable_block_of))  # a refusal is the same answer on every try
    return isinstance(exc, MockRuntimeFaultRaised) and not refused and turns_completed == 0 and seeded_entities > 0 and not served_data


async def classify_restart_failure(
    exc: BaseException,
    *,
    client: OrchestratorClient,
    thread_id: str | None,
    restart_shaped_calls: int,
    allow_turn_timeout_redrive: bool,
    concurrency: int,
) -> RedriveTrigger | None:
    """Return the trigger when *exc* is re-drive-classified, else ``None``.

    ``thread_id`` is the id of the thread this scenario created; ``None`` means
    create never succeeded. Every trigger but two requires "create succeeded": a
    failure BEFORE create is a plain failure (a bad scenario id, a host name that does
    not resolve) that a re-drive would only repeat. The exceptions are a backend outage
    (:func:`backend_outage`, a spent transient budget on any request, create included)
    and create itself answering a GATEWAY status (:func:`_is_create_gateway_status`) —
    typed by status, method and path, so a 4xx refusal or an app 500 on create still declines.

    ``restart_shaped_calls`` carries the mid-scenario connect-exhaustion signal from the child:
    how many tool calls the mock left unanswered the way a restart does (a 503 that outlived the
    retry, a refused connection). REQUIRED — ``0`` where the failure carries no such count — so a
    call site cannot drop the signal by omission. Only consulted when create succeeded.

    ``concurrency`` is the run's ``--concurrency``: a SIGKILLed child re-drives only above 1
    (:attr:`RedriveTrigger.CHILD_KILLED`). Both it and ``allow_turn_timeout_redrive`` are REQUIRED,
    so a call site cannot drop the phase or the load by omission.

    ``allow_turn_timeout_redrive`` gates the TURN_TIMEOUT branch ALONE — a qualification
    pass sets it False. That phase is a fail-fast gate: a per-turn wall that
    silently costs twice its length before it reports has defeated its own purpose. It is
    scoped to this one trigger deliberately, because the others describe a thread the box
    killed underneath us — losing a qualification batch to a box restart is not "failing
    fast", it is just lost work, so those still re-drive in every phase.

    A reaped child (:class:`~plurai._protocol.HarnessChildExited`) re-drives on ONE shape only: a
    SIGKILL during a dialog turn at concurrency above 1, where sibling scenarios supplied the memory
    pressure the OOM reaper answered. At concurrency 1 the pressure is this scenario's own and is
    still there on the re-drive, and a child that exited on its own (a positive code) does so
    deterministically — both decline. The shared budget of one still bounds it.
    """
    if backend_outage(exc):
        return RedriveTrigger.BACKEND_OUTAGE
    if _is_create_gateway_status(exc):
        return RedriveTrigger.CREATE_TRANSIENT
    if not thread_id:
        return None
    if restart_shaped_calls > 0:
        return RedriveTrigger.CONNECT_EXHAUSTION
    if (trigger := static_trigger(exc, allow_turn_timeout_redrive=allow_turn_timeout_redrive, concurrency=concurrency)) is not None:
        return trigger
    if _is_thread_scoped_404(exc):
        return RedriveTrigger.THREAD_404
    if _finalize_failed_terminally(exc) and await _thread_is_gone(client, thread_id):
        return RedriveTrigger.FINALIZE_THREAD_GONE
    return None


async def _thread_is_gone(client: OrchestratorClient, thread_id: str) -> bool:
    """404-probe ``GET /threads/{id}``: is the thread itself gone?

    A finalize failure alone is ambiguous — the eval genuinely failing looks the
    same as the process dying under it. This probe disambiguates: a 200 means the
    server still knows the thread, so the finalize failure is REAL and must
    surface unchanged (no re-drive). Only a 404 — impossible on a healthy server,
    per never-evict — confirms the restart.

    Any other outcome (a 5xx, a transport error, an unexpected body) answers
    ``False``: an inconclusive probe must never authorize spending the budget.
    """
    try:
        await client.get_thread(thread_id)
    except httpx.HTTPStatusError as exc:
        return exc.response.status_code == 404
    except Exception:  # noqa: BLE001 — an unusable probe is a "no", never a "yes"
        logger.debug('thread-liveness probe for %s was inconclusive', thread_id, exc_info=True)
        return False
    return False


async def wait_until_ready(
    client: OrchestratorClient,
    *,
    mock_service_url: str | None,
    mock_mode: MockMode,
    bound_seconds: float | None = None,
    poll_interval: float | None = None,
) -> bool:
    """Block until the backend can actually serve a fresh scenario, within one bound.

    Two phases under ONE deadline:

    1. ``GET /health`` returns 200. Native and pre-auth on a current box; an older
       box without that endpoint never answers 200, the bound expires, and the
       caller surfaces the original failure with the budget unspent.
    2. connect mode only — the scenario's mock route stops answering 503. The box
       redeploys mock envs asynchronously after boot, so ``/health`` can be green
       while the mock the agent needs is still coming up.

    Returns ``True`` when both hold, ``False`` on expiry or on the
    ``awaiting_edit_finalize`` hold — a 503 that is blocked on a human and would
    otherwise consume the entire bound before failing anyway.

    The two timing knobs resolve from the module constants HERE rather than as
    default arguments: a default binds its value at import, which would freeze the
    bound at 300s for every caller and make it untestable without waiting it out.
    """
    bound_seconds = READINESS_BOUND_SECONDS if bound_seconds is None else bound_seconds
    poll_interval = READINESS_POLL_INTERVAL_SECONDS if poll_interval is None else poll_interval
    deadline = time.monotonic() + bound_seconds
    if not await _poll_until(client.health, deadline=deadline, poll_interval=poll_interval, what='server /health'):
        return False
    if mock_mode != 'connect' or not mock_service_url:
        return True

    hold_seen = False

    # Probe the mock's OWN /health, the same endpoint ``plurai.configure`` verifies
    # at scenario start. Probing the route root would answer 404 from a perfectly
    # live mock (it serves no ``/`` route), which reads as "not 503, go" — true, but
    # it proves only that the proxy answered, not that the mock behind it is up.
    mock_health_url = mock_service_url.rstrip('/') + '/health'

    async def _mock_ready() -> bool:
        nonlocal hold_seen
        status, code = await client.probe_url(mock_health_url)
        if status == 503 and code == MOCK_HOLD_CODE:
            hold_seen = True
            return True  # break out of the poll; the caller checks hold_seen
        # ANY non-503 answer means the route has stopped redeploying. A 410 or a
        # 502 here is a real, differently-shaped problem — let the re-driven run
        # hit it and report it properly rather than time out with a vaguer error.
        return status is not None and status != 503

    ready = await _poll_until(_mock_ready, deadline=deadline, poll_interval=poll_interval, what=f'mock route {mock_service_url}')
    if hold_seen:
        logger.warning(
            'mock route %s is holding for an unfinalized edit session (503 %s) — not waiting it out; '
            'that hold clears only when the edit session is finalized, abandoned, or expires.',
            mock_service_url,
            MOCK_HOLD_CODE,
        )
        return False
    return ready


async def _poll_until(
    predicate: Callable[[], Awaitable[bool]],
    *,
    deadline: float,
    poll_interval: float,
    what: str,
) -> bool:
    """Poll *predicate* until it answers truthy or *deadline* passes."""
    while True:
        if await predicate():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning('readiness gate gave up waiting for %s', what)
            return False
        await asyncio.sleep(min(poll_interval, remaining))
