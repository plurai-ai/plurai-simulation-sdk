"""IPC envelope types + decode/encode for the parent↔child subprocess protocol.

This module is the single source of truth for the wire format used between
``Orchestrator`` (parent) and ``python -m plurai.child`` (child). The protocol
is newline-delimited JSON: one envelope per line, each envelope a JSON object
with a ``type`` discriminator and type-specific fields.

The error-classifier helpers (async-footgun, langgraph-recursion) and the
canned ``OrchestratorAdapterError`` message templates live here too so the
child can label adapter errors at classification time and the parent can
re-render the customer-visible message after the IPC handshake.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import decimal
import enum
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Final, Literal, get_args

from plurai._mocking.turn_context import MockRouteCause
from plurai.capacity import CapacityBlock, UnavailableBlock
from plurai.credit import CreditBlock

# Mock posture, defined ONCE and shared by ``NetworkSettings.mock_mode``, the ``Init``
# wire carrier, ``ChildProcess.init``, and ``plurai.configure(mode=...)`` — so the
# Literal isn't restated at each site.
#
# THREE postures, because there are three, and collapsing the last two makes the posture
# line lie:
# * ``'connect'``  — talk to a deployed mock; raise if unreachable.
# * ``'local'``    — host an in-process loopback mock with the registered ``@plurai.mock``
#                    handlers. A real mocked world, hosted here instead of deployed.
# * ``'mockless'`` — NOTHING is mocked. A tool-less agent, or one whose every tool runs
#                    ``mode: real``. Shares local's transport (an idle loopback stub, so
#                    ``plurai.mock_service_url()`` still answers for harness ``seed_state``
#                    templates) but is a DIFFERENT CLAIM: no surface is held constant and
#                    every tool call goes to the real thing. Declaring that as ``'local'``
#                    would tell readers of the run report the exact opposite.
type MockMode = Literal['connect', 'local', 'mockless']

# The two postures served by the in-process stub. Named HERE, beside the type it derives from, so
# the disjunction has one home: every consumer needs one of two DERIVED properties rather than the
# posture itself — "did a deployed mock observe these calls" (no, under either) and "was anything
# held constant" (no, only under mockless) — and hand-retyping `in ('local','mockless')` at each
# site is how the two drift apart.
type StubMode = Literal['local', 'mockless']
_STUB_MODES: frozenset[str] = frozenset(get_args(StubMode.__value__))


# ---------------------------------------------------------------------------
# Customer-facing exception type (kept here so both parent driver
# (``_subprocess.ChildProcess``) and the deprecated in-process path
# (``orchestrator.py``) can ``raise`` and re-export the same class —
# isinstance(..., OrchestratorAdapterError) must work regardless of
# which path produced the error).
# ---------------------------------------------------------------------------


class OrchestratorAdapterError(RuntimeError):
    """Raised when the agent adapter (``invoke_agent`` / ``seed_state`` /
    ``invoke_pipeline``) hits a misuse the orchestrator can detect at runtime.

    Subclasses :class:`RuntimeError` so existing ``except RuntimeError`` callers
    keep working; carries actionable text + a SKILL.md step pointer for the
    user to fix. The original exception is preserved as ``__cause__`` for
    callers that want the underlying traceback.

    Optional structured attributes (default to None/False, so a caller that
    sets none of them still constructs):

    * ``child_exc_class``: when the error originated in the customer's
      handler (carried back via ``PipelineError`` / ``AgentError``
      envelopes), this names the original exception class so consumers
      can filter / route by class without substring-matching the text.
    * ``timed_out``: True when the failure was a wall-clock timeout
      rather than a handler raise. Distinguishes "pipeline hung" from
      "pipeline crashed" in downstream diagnostics.
    * ``thread_id``: the server thread id this error is associated with,
      when known. Lets the runner mark the corresponding server-side
      thread failed without re-deriving the id from message text.
    * ``raised_at``: WHEN in the scenario's lifecycle this raise fired —
      ``'init'`` (before any turn ran) or ``'finalize'`` (after the dialog
      or pipeline hop completed and only the final state capture failed).
      A raiser with only one raise site, or whose sites all fire at the
      same phase, never sets it. It exists because a kind like
      :class:`MockUnreachableError` fires from more than one site with
      different preconditions, and a remedy that is right at one phase
      (e.g. "check your --mock-mode") can be a non-sequitur at the other
      (the agent already dialed the mock for however long it ran) — a
      consumer must be able to tell the phases apart from a TYPED field,
      never by re-deriving it from ``turns_reached`` (structurally 0 on a
      pipeline hop's own finalize-capture raise, since pipeline hops never
      touch that counter) or from the message text.

    ``kind`` is a CLASS attribute, not a constructor argument: the slug is a property of
    the failure class itself, so it is declared once by the class that raises and can never
    disagree with the type. ``None`` here means "no typed classification" — the honest
    answer for a generic adapter error, and the reason downstream consumers treat a missing
    ``kind`` as unknown rather than defaulting it to anything.

    The rule this encodes: **the kind is tagged by the raiser, never re-derived by the
    reader.** Matching a consumer against the emitters' prose is a trap — it silently
    misclassifies real emitters, and every new failure mode becomes one more string the
    reader has to learn.
    """

    #: Typed failure slug, set by the raising class. See the class docstring.
    kind: ClassVar[str | None] = None

    def __init__(
        self,
        message: str = '',
        *,
        child_exc_class: str | None = None,
        child_message: str | None = None,
        timed_out: bool = False,
        thread_id: str | None = None,
        raised_at: Literal['init', 'finalize'] | None = None,
    ) -> None:
        super().__init__(message)
        self.child_exc_class = child_exc_class
        #: The COMPONENT's own message, unwrapped. ``str(self)`` prefixes SDK plumbing prose
        #: ("pipeline_invoke failed on thread …"), which is right for a log and wrong for a
        #: verdict input — a judge grading a refusal should read what the component said.
        self.child_message = child_message
        self.timed_out = timed_out
        self.thread_id = thread_id
        self.raised_at = raised_at


class FinalizePollTimeout(OrchestratorAdapterError):  # noqa: N818  # frozen public name; MUST NOT rename
    """Raised when the async finalize poll loop exceeds its max wallclock budget.

    Carries ``thread_id`` (inherited) and the ``job_id`` of the in-flight
    finalize job as a public informational field (useful for logging and
    diagnostics). The actual cancel-on-abandon issues the DELETE via the
    orchestrator's in-memory ``_inflight_finalize`` state, not this attribute.

    The THIRD of the SDK's own wall-clock budgets, and typed for the same reason as the other
    two: a slow eval set blowing this wall is our budget expiring, not the agent failing. Left
    untyped it reaches ``error.json`` as a bare class name, which the run report and the suite
    critic both read as the agent crashing — off a scenario with no transcript to check that
    reading against. Reachable at qualification defaults, where the scenario ceiling and this
    poll wall are both 1800s.
    """

    #: A SIMULATION budget expiring — same contract as the other ``harness_*`` kinds.
    kind: ClassVar[str] = 'harness_finalize_poll_timeout'

    def __init__(self, message: str, *, thread_id: str, job_id: str) -> None:
        super().__init__(message, thread_id=thread_id, timed_out=True)
        self.job_id = job_id


class FinalizeJobFailed(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """Raised when an async finalize job reports terminal ``failed`` on the server.

    Split out of the generic adapter error so the restart classifier can recognise
    it by TYPE. It is deliberately NOT a restart signal on its own — a finalize
    genuinely failing on a perfectly healthy server raises exactly this — which is
    why the classifier pairs it with a ``GET /threads/{id}`` 404-probe before it
    will call the thread dead.
    """

    def __init__(self, message: str, *, thread_id: str, job_id: str) -> None:
        super().__init__(message, thread_id=thread_id)
        self.job_id = job_id


class MockUnreachableError(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """Raised when a ``mock_mode='connect'`` child could not reach the deployed mock.

    A TYPED specialisation of the message ``_raise_for_error`` renders for
    ``InitError(kind='mock_unreachable')``: same text, with the condition
    isinstance-checkable. The restart classifier re-drives a scenario on
    *connect-exhaustion* and must recognise it from a type, never by
    substring-matching that message — zero prose matching.

    ``thread_id`` is inherited and always populated (the child knows the id it
    was initialised with), so the runner's failure path can call ``/fail`` and stamp
    ``error.json`` with a real id rather than ``thread_id: "unknown"``.
    """

    #: A mock the runner could not reach is never the agent's, whatever the cause (a posture, a
    #: deploy, a blip): a ``harness_*`` kind, so the row is ``not_valid`` and never scored.
    kind: ClassVar[str] = 'harness_mock_unreachable'


class MockInitStateFault(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The mock was dialed successfully (``configure`` / ``start_session``'s own connectivity check
    passed), but a state-shaped call during init/reset/seed (a ``/state`` read, typically) failed
    afterward — e.g. a timeout under backend load.

    Distinct from :class:`MockUnreachableError`, whose message is specifically about connect
    posture / URL / deploy state, which does not fit a mock that answered enough to be dialed.
    ``harness_*`` so a genuinely infra/backend-load symptom is never scored as the agent's or a
    bare adapter bug.

    ``thread_id`` is inherited and always populated (the child knows the id it was initialised
    with), so the runner's failure path can call ``/fail`` and stamp ``error.json`` with a real id
    rather than ``thread_id: "unknown"``.
    """

    #: A mock that answered enough to be dialed but then failed a state operation is still not
    #: the agent's fault, whatever the cause (backend load, a slow /state read): a ``harness_*``
    #: kind, so the row is ``not_valid`` and never scored.
    kind: ClassVar[str] = 'harness_mock_init_state_fault'


class MockStubFallback(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """RE-DRIVE PROVENANCE ONLY: a scenario re-driven because some tool calls failed the way a mock restart does.

    Never raised and never written as an ``error.json`` kind: a scenario whose calls went unanswered is
    graded, with a ``mock.tool_call_unanswered`` finding. This type exists so a re-drive's provenance
    records what the first attempt's cause was (``original_kind``). Re-drive classification keys on the
    restart-shaped count of the child's ``unanswered_mock_calls``, never on this type.
    """

    kind: ClassVar[str] = 'harness_mock_stub_fallback'


class AgentTurnTimeout(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """Raised when a child envelope blew its wall-clock budget and the child was killed.

    A TYPED specialisation of the message ``_await_response_or_exit`` renders: same
    text, with the condition isinstance-checkable. The re-drive classifier can only
    recognise it from this type — the message is prose, and prose matching is exactly
    what the classifier contract rules out. Left untyped, a stall matches no branch of
    the classifier, so a single slow turn costs the whole scenario an ``error.json``.

    ``envelope_type`` is what the timeout was WAITING on, and it is load-bearing rather
    than informational: the same budget guards ``init``, ``user_turn`` and ``finalize``,
    and only a stalled ``user_turn`` is the transient provider hiccup a fresh thread can
    survive. An ``init`` stall is the customer's ``simulate.py`` being slow to construct
    its agent — deterministic, and it would stall again on the re-drive. So consumers
    branch on this field; the type alone does not mean "re-drivable".

    ``thread_id`` is inherited and is populated at the raise site (``init`` records the
    id before the first envelope goes out), so the runner's failure path can call
    ``/fail`` and stamp ``error.json`` with a real id. ``timed_out`` is always True —
    this class only ever describes a hang, never a crash.

    ``budget_seconds`` is the budget actually IN FORCE, carried as a field rather than left
    inside the rendered message. A consumer that wants to tell the operator what to raise
    would otherwise have to regex it back out of prose, and a report that names the wall
    without naming the knob is a dead end.
    """

    #: This is a SIMULATION budget expiring, not the agent producing a wrong answer — on a DIALOG
    #: turn, where the budget bounds one exchange of a conversation the harness is driving, every
    #: consumer must treat it as a harness fault and must not score it.
    #:
    #: A ONE-SHOT HOP is the exception, and it is not a contradiction: there the component IS the
    #: unit under test, so a hop that never returns is a fact ABOUT it. `_run_pipeline_hop`
    #: records it as `pipeline_error{timed_out: true}` and finalizes, per the backend's own
    #: contract for that field (an outcome a refusal policy can credit). The SCENARIO budget (`harness_scenario_timeout`) stays a harness fault in both.
    kind: ClassVar[str] = 'harness_turn_timeout'

    def __init__(self, message: str, *, envelope_type: str, budget_seconds: float | None = None, thread_id: str | None = None) -> None:
        super().__init__(message, thread_id=thread_id, timed_out=True)
        self.envelope_type = envelope_type
        self.budget_seconds = budget_seconds


class ScenarioTimeout(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """Raised when a WHOLE SCENARIO blew its wall-clock ceiling, across all its turns.

    The sibling of :class:`AgentTurnTimeout` one level up: that class bounds ONE envelope,
    this one bounds the scenario. Both are needed, because a scenario can run away without
    any single turn doing so — every turn lands inside the per-turn budget while the
    scenario accumulates them without limit.

    Why it exists at all: a ceiling living ONLY inside the Arize experiment driver's task
    timeout lets the configured OBSERVABILITY BACKEND decide whether a scenario can run
    forever — capped on Arize, unbounded on LangSmith, Langfuse, Braintrust and no-sink
    alike, since only the Arize sink implements ``run_experiment_set``. A tracing choice
    must not decide execution semantics.

    NOT re-drivable, and deliberately so. A re-drive replays the scenario from turn 1 under
    the same ceiling, so it would blow the same budget at roughly the same point — buying a
    second full scenario's cost to re-learn what the first attempt already established.
    """

    #: A SIMULATION budget expiring — the same contract as ``AgentTurnTimeout.kind``: not a
    #: verdict about the agent, and never to be scored as one.
    kind: ClassVar[str] = 'harness_scenario_timeout'

    def __init__(self, message: str, *, budget_seconds: float, thread_id: str | None = None) -> None:
        super().__init__(message, thread_id=thread_id, timed_out=True)
        self.budget_seconds = budget_seconds


class HarnessChildExited(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """Raised when the ``simulate.py`` child process DIED before answering an envelope.

    The third shape of harness fault beside the two budgets, and the one that looks least
    like one: a child reaped by the OS reaches the parent as a bare returncode, so an
    untyped ``error.json`` says only that something exited and every consumer reads that as
    the agent crashing. Concurrency-driven OOM kills otherwise read as
    agent failures.

    ``returncode`` is carried rather than only rendered because its SIGN is the
    classification: a NEGATIVE value is ``-signal`` (POSIX), i.e. the child was killed
    rather than having exited on its own, and ``-9``/SIGKILL under concurrency is the OOM
    reaper. A positive code is the child's own exit — an ImportError in the customer's
    venv, a missing dep — and that one IS the customer's to fix, so the two must stay
    distinguishable from a field.

    NOT re-drivable: see ``orchestrator.redrive.classify_restart_failure``.
    """

    #: The harness lost its child — a simulation fault like the budget kinds, never a
    #: verdict about the agent. Unlike them it is not a timeout, so ``timed_out`` stays False.
    kind: ClassVar[str] = 'harness_child_exited'

    def __init__(self, message: str, *, returncode: int | None, envelope_type: str, thread_id: str | None = None) -> None:
        super().__init__(message, thread_id=thread_id)
        self.returncode = returncode
        self.envelope_type = envelope_type

    @property
    def killed_by_signal(self) -> int | None:
        """The signal that reaped the child, or ``None`` when it exited on its own."""
        return -self.returncode if self.returncode is not None and self.returncode < 0 else None


#: The per-line cap on the JSONL pipes, both directions (the parent's stdout reader and the child's
#: stdin reader). ``asyncio.StreamReader`` defaults to 64 KiB, and ``Init`` /
#: ``FinalizeResponse`` carry arbitrary customer JSON that routinely exceeds that; 16 MiB holds
#: realistic transcripts and tool-event logs while still bounding pathological payloads.
IPC_LINE_LIMIT_BYTES: Final = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class OversizedIpcLine:
    """What :func:`read_ipc_line` returns for a line that overran :data:`IPC_LINE_LIMIT_BYTES`.

    The rest of that line has already been discarded, so the reader is in sync for the next envelope.
    """

    limit: int


async def read_ipc_line(stream: asyncio.StreamReader) -> bytes | OversizedIpcLine:
    """One newline-terminated IPC line, ``b''`` at EOF, or :class:`OversizedIpcLine` when it overran the limit.

    Both ends of the pipe read through this (the parent's stdout reader and the child's stdin loop).
    ``StreamReader.readline`` raises a bare ``ValueError`` on an over-limit line and, when the
    separator was not yet buffered, leaves the rest of that line to be read as the NEXT line. Read
    with ``readuntil`` instead, and on an overrun discard up to and including the next newline, so
    the reader stays in sync for the envelopes after it.
    """
    try:
        return await stream.readuntil(b'\n')
    except asyncio.IncompleteReadError as exc:
        return exc.partial
    except asyncio.LimitOverrunError as exc:
        consumed = exc.consumed
        while True:
            try:
                await stream.readexactly(consumed)
                await stream.readuntil(b'\n')
            except asyncio.LimitOverrunError as again:
                consumed = again.consumed
                continue
            except asyncio.IncompleteReadError:
                pass
            return OversizedIpcLine(limit=IPC_LINE_LIMIT_BYTES)


#: The ``agent_state`` marker on a reply the child TRIMMED to fit one IPC line (its message chain and the bulk of its
#: agent state dropped; ``ai_text`` and ``agent_actions`` kept). The parent records each such turn
#: (``ChildProcess.ipc_trimmed_turns``): a judge reading that turn's tool trail reads less than the agent did.
IPC_TRIMMED_KEY: Final = '_ipc_trimmed'


#: Which side of the pipe an overrunning IPC line came from: ``sent`` by the runner, ``received`` from the child.
IpcOverrunDirection = Literal['sent', 'received']


class HarnessIpcOverrun(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """One JSONL envelope was longer than :data:`IPC_LINE_LIMIT_BYTES`, so it cannot cross the pipe.

    Malformed wire, raised the moment it is seen — not a hang that later reads as a turn timeout.
    Deterministic, so never re-driven: the same envelope would overrun again. The child trims what
    it can before writing (a reply's message chain, a world that will not fit), so reaching this
    means even the trimmed envelope did not fit, or the parent's own ``Init`` did not.

    ``direction`` is which side's line overran (:data:`IpcOverrunDirection`): ``sent`` — the runner's
    outbound envelope (``envelope_type``) was refused before it was written; ``received`` — the
    child's answer to ``envelope_type`` could not be read. ``envelope_type`` alone cannot tell them
    apart: an oversized ``Init`` and an oversized reply to it both name ``init``.
    """

    kind: ClassVar[str] = 'harness_ipc_line_overrun'

    def __init__(self, message: str, *, envelope_type: str, direction: IpcOverrunDirection, limit: int, thread_id: str | None = None) -> None:
        super().__init__(message, thread_id=thread_id)
        self.envelope_type = envelope_type
        self.direction = direction
        self.limit = limit


class HarnessSeedStateContract(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The harness's ``seed_state`` hook could not be called as the seed contract requires.

    Three shapes, all refused by the child at init and all about the TEMPLATE, never the agent:
    the hook is missing on a scenario that resumes from a trace, its signature cannot take the
    three arguments the contract passes, or it returned something other than a mapping. Typed so
    the run report attributes the row to the harness — an untyped ``RuntimeError`` reads
    downstream as the agent crashing — and so the remedy can be named: fix ``simulate.py``'s
    ``seed_state`` and run the plugin's ``verify_simulate.py``. The message keeps the child's own
    sentence, which names the file and the signature to change it to.

    NOT re-drivable and NOT a timeout: see ``orchestrator.redrive.classify_restart_failure``.
    """

    kind: ClassVar[str] = 'harness_seed_state_contract'

    def __init__(self, message: str, *, child_exc_class: str | None = None, thread_id: str | None = None) -> None:
        super().__init__(message, child_exc_class=child_exc_class, thread_id=thread_id)


class HarnessFault(Exception):  # noqa: N818  # public name, raised by simulate.py
    """Raised by ``simulate.py`` for a refusal the HARNESS makes — a path its bench does not cover.

    Any other raise inside ``invoke_agent`` / ``invoke_pipeline`` / ``seed_state`` / ``reset`` is
    the agent's; this one is tagged by its raiser, so the run records it as ``not_valid`` rather
    than scoring it against the agent. It is recorded when it is CONSTRUCTED — so an agent tool
    that catches it does not hide it, and one built inside a hook voids that hook even unraised.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        from plurai._mocking.turn_context import note_harness_fault

        note_harness_fault(f'{type(self).__module__}.{type(self).__name__}', str(self), kind='harness_fault')


class AgentFault(Exception):  # noqa: N818  # public name, raised by a harness on the agent's behalf
    """Raised by a harness ON THE AGENT'S BEHALF: the agent's own code raised where no Python frame can show it
    (a bridge child in another runtime), and the harness has already attributed ownership to the agent.

    ``invoke_agent`` raising it is read as ``agent_raised`` (a failed trial the agent owns) — on the raised
    exception itself, never its chain, and only after the provider check and any harness fault recorded in the
    turn. It records nothing, so it never hides a ``HarnessFault``.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class HarnessFaultRaised(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The parent side of :class:`HarnessFault`: the child reported a raise the harness tagged as its own.

    NOT re-drivable and NOT a timeout: see ``orchestrator.redrive.classify_restart_failure``.
    """

    kind: ClassVar[str] = 'harness_fault'

    def __init__(self, message: str, *, child_exc_class: str | None = None, child_message: str | None = None, thread_id: str | None = None) -> None:
        super().__init__(message, child_exc_class=child_exc_class, child_message=child_message, thread_id=thread_id)


class MockRuntimeFaultRaised(HarnessFaultRaised):  # noqa: N818  # sibling of the frozen public names above
    """The backend's mock runtime could not generate a mocked tool's answer (``kind='mock_runtime_fault'``).

    A ``harness_*`` kind, so the row stays ``not_valid`` everywhere, but NOT the harness's code to fix:
    ``cause`` is the runtime's own reason and ``transient`` says whether one more try could answer.
    ``route_cause`` is the typed cause when the mock never answered the call at all (``plurai.MockRouteFault``),
    ``None`` when the runtime answered with a fault. Re-drivable only when transient: see
    ``orchestrator.redrive.classify_restart_failure``.

    ``credit`` is set when the backend refused the call for credit (``route_cause == 'credit'``): the run stops on it
    like on any credit refusal, and no re-drive rule takes it. :func:`~plurai.credit.credit_block_of` reads it.
    ``capacity`` is set, the same way, when the backend refused it because the provider refused its account
    (``route_cause == 'capacity'``): :func:`~plurai.capacity.capacity_block_of` reads it.
    ``unavailable`` is set when the backend answered with its typed ``unavailable`` block (``route_cause == 'unavailable'``):
    :func:`~plurai.capacity.unavailable_block_of` reads it.
    """

    kind: ClassVar[str] = 'harness_mock_runtime'

    def __init__(
        self,
        message: str,
        *,
        transient: bool,
        cause: str,
        route_cause: MockRouteCause | None,
        credit: CreditBlock | None = None,
        capacity: CapacityBlock | None = None,
        unavailable: UnavailableBlock | None = None,
        child_exc_class: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        super().__init__(message, child_exc_class=child_exc_class, child_message=cause, thread_id=thread_id)
        self.transient = transient
        self.cause = cause
        self.route_cause = route_cause
        self.credit = credit
        self.capacity = capacity
        self.unavailable = unavailable


class HarnessResetContract(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The harness's ``reset`` hook could not be called as the reset contract requires.

    The sibling of :class:`HarnessSeedStateContract`, and typed for the same reason: this is a
    defect in the TEMPLATE, never in the agent, and an untyped ``RuntimeError`` reads downstream
    as the agent crashing. One shape today — a signature that wants an argument the contract does
    not pass — and it fails EVERY scenario of the run, fresh ones included, because ``reset()``
    is called before each. The message keeps the child's own sentence, which names the file and
    the signature to change it to.

    NOT re-drivable and NOT a timeout: see ``orchestrator.redrive.classify_restart_failure``.
    """

    kind: ClassVar[str] = 'harness_reset_contract'

    def __init__(self, message: str, *, child_exc_class: str | None = None, thread_id: str | None = None) -> None:
        super().__init__(message, child_exc_class=child_exc_class, thread_id=thread_id)


def is_harness_owned(kind: object) -> bool:
    """Whether a typed failure ``kind`` (a raising class's attribute, or its ``error.json`` copy) is
    the SIMULATION's own — a budget, a lost child, a hook contract, a ``HarnessFault`` — and never
    an agent outcome. The one spelling of the owner rule: the runner and the plugin's scan smoke read it."""
    return str(kind or '').startswith('harness_')


#: The AGENT raised inside its own turn (``invoke_agent`` raised, untagged by the harness): a failed
#: trial the agent owns, never an invalid one. Tagged by the raising class, like every ``kind``.
AGENT_RAISED_KIND: Final = 'agent_raised'


class AgentRaised(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The agent's own call raised during a turn (``AgentError(kind='agent_raised')``)."""

    kind: ClassVar[str] = AGENT_RAISED_KIND


#: The agent's PROVIDER refused or failed the agent's own call (``child._provider_unavailable``): an
#: invalid trial owned by the infrastructure, never the agent's outcome and never the harness's.
PROVIDER_INFRA_KIND: Final = 'provider_infra'


#: One agent turn blew the per-turn wall (``AgentTurnTimeout`` on a ``user_turn`` envelope) after the
#: dialog started, and the dialog LANDED: the apparatus' budget cut it, so it is censored and the trial is
#: invalid unless overruled — the same standing as a provider cut.
TURN_BUDGET_EXHAUSTED_KIND: Final = 'turn_budget_exhausted'


class ProviderInfra(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The agent's provider failed its call during a turn (``AgentError(kind='provider_infra')``)."""

    kind: ClassVar[str] = PROVIDER_INFRA_KIND


#: The stable prefix the backend puts on every quarantine refusal. A FROZEN wire constant: the
#: backend refuses to open a thread for a scenario whose known-good reference path could not pass
#: its own eval set, and a run report separating "the suite refused to measure this" from "the
#: agent errored" matches on THIS, not on the English sentence after it.
QUARANTINE_REFUSAL_PREFIX: Final = 'scenario_quarantined_unsolvable:'


class ScenarioQuarantined(OrchestratorAdapterError):  # noqa: N818  # sibling of the frozen public names above
    """The SUITE refused to measure this scenario — it is not a failure of the agent.

    Raised when ``POST /threads`` comes back 400 carrying :data:`QUARANTINE_REFUSAL_PREFIX`: the
    scenario's reference path could not pass its own eval set, so scoring an agent against it
    would measure the suite. The backend refuses at the one chokepoint every scored run goes
    through; this is the client half, and its whole job is that the refusal keeps its identity
    all the way to the artifact a report reads.

    ``kind`` is deliberately NOT a ``harness_*`` slug. Those mean "our simulation broke"; this
    means the suite worked exactly as designed and declined to take a measurement it could not
    trust. Merging the two would hide a scenario that needs REPAIR inside a bucket a reader
    dismisses as infrastructure noise.
    """

    #: A SUITE decision, not a harness fault and not an agent verdict.
    kind: ClassVar[str] = 'suite_scenario_quarantined'

    def __init__(self, message: str, *, scenario_id: str | None = None) -> None:
        super().__init__(message)
        self.scenario_id = scenario_id


#: A recursion/step-limit crash — see :data:`_RECURSION_LIMIT_DETECTORS`. Distinct from
#: :data:`AGENT_RAISED_KIND`: none of ``GraphRecursionWrap``'s own A/B/C causes is unconditionally
#: the agent's own (the mock's empty answers and a seed defect loop the graph exactly as an agent
#: bug does), so this kind alone does not promote a trial to measured/FAILED — that additionally
#: requires ``pre_crash_tool_activity == 'real_data'`` on the error record (see
#: ``plurai.trials.cells_from_run_dir`` and the backend's run-error routing). A distinct kind, not
#: left blank: an omitted ``kind`` is indistinguishable from "nobody thought about this" and would
#: default every recursion crash to a harness fault in the backend's attribution and to an invalid
#: (never-measured) trial in :mod:`plurai.trials`, regardless of what the transcript showed.
AGENT_RECURSION_BUDGET_KIND: Final = 'agent_recursion_budget'


class GraphRecursionWrap(OrchestratorAdapterError):  # noqa: N818  # frozen public name (plugin consumer); MUST NOT rename
    """Specialisation raised when the agent's LangGraph hit its recursion
    limit. Carries structured attributes alongside the human message so
    diagnostic tooling can route programmatically (e.g. "auto-run the
    verifier when cause_class == 'local_seed'") without substring-matching
    the message text.

    ``cause_class`` is best-effort:
      - ``'local_seed'`` — the verifier flagged a seed_state defect (set
        by the orchestrator when it shells out to verify_simulate
        before re-raising; not set on the bare wrap from the SDK).
      - ``'remote_mock'`` — local artifacts are clean but the mock keeps
        returning empty data (set when the orchestrator has additional
        evidence).
      - ``'unknown'`` — default; the orchestrator only saw the recursion
        and didn't try to disambiguate. Callers should run the verifier.

    ``kind`` is :data:`AGENT_RECURSION_BUDGET_KIND` — a distinct kind, not
    :data:`AGENT_RAISED_KIND`: unlike an agent raising in its own turn, a recursion crash is not
    unconditionally the agent's fault (the A/B/C ambiguity above is real — cases A and B are the
    harness's / the mock's). A downstream reader (``plurai.trials``, the backend's
    run-error routing) only counts this as a measured, FAILED agent trial when the error
    record's ``pre_crash_tool_activity`` field (stamped by ``run_simulation.py`` from this
    attempt's OWN recorded tool events, never from this class or from ``cause_class``) says
    ``'real_data'`` — evidence the agent was not silent before it crashed. Everything else about
    this kind (``'empty_or_none'``, ``'unknown'``, or an SDK build that does not stamp the field)
    keeps the conservative posture: errored, never counted as the agent's trial.
    """

    kind: ClassVar[str] = AGENT_RECURSION_BUDGET_KIND

    def __init__(
        self,
        message: str,
        *,
        cause_class: Literal['local_seed', 'remote_mock', 'unknown'] = 'unknown',
        thread_id: str | None = None,
        turn_idx: int | None = None,
    ) -> None:
        super().__init__(message, thread_id=thread_id)
        self.cause_class = cause_class
        self.turn_idx = turn_idx


# ---------------------------------------------------------------------------
# Adapter-error classification (used by child to label; parent to re-render).
# ---------------------------------------------------------------------------

AdapterErrorSubkind = Literal[
    'invoke_async_footgun',
    'seed_state_async_footgun',
    'graph_recursion',
    'no_assistant_reply',
    'reply_projection_failed',
    'undeclared_input_part',
    'undeclared_empty_opener',
    'reply_oversized',
]

#: The subkinds whose only correct outcome is a RAISE — the turn cannot be scored because the
#: adapter itself is broken, so every remaining turn would fail identically and a verdict built
#: on them would be a verdict about ``simulate.py``, not about the agent.
#:
#: Everything ELSE — a listed subkind absent from this set, and any subkind this build has never
#: heard of — degrades to a SCOREABLE turn (see :func:`adapter_error_is_scoreable`). The default
#: is that way round on purpose: an agent that produced no usable reply is a FAILING agent, and
#: turning that into an exception loses the measurement the evals exist to take. A new subkind
#: therefore costs a warning line, never a dead scenario.
FATAL_ADAPTER_SUBKINDS: frozenset[str] = frozenset({'invoke_async_footgun', 'seed_state_async_footgun', 'graph_recursion'})


def adapter_error_is_scoreable(subkind: str | None) -> bool:
    """Whether an adapter error must be recorded as an empty assistant turn instead of raised.

    ``no_assistant_reply`` is the case this exists for: the agent ran and produced nothing a
    user could read. That is a verdict — ``response_not_empty`` and every judge downstream of it
    are there to catch exactly it — and raising instead of scoring would turn one silent
    agent into a lost scenario.

    An UNKNOWN subkind takes the same route. The classifier that emits subkinds and the renderer
    that explains them are edited independently, so "the renderer has not heard of this one" is
    a condition that recurs; making it fatal means each recurrence costs a scenario.
    """
    return subkind is not None and subkind not in FATAL_ADAPTER_SUBKINDS


# Substring fragments matched against ``str(exc)`` to detect the
# nested-event-loop footgun. Tolerates wording differences across CPython
# releases / nest_asyncio / uvloop.
_ASYNC_FOOTGUN_FRAGMENTS = (
    'asyncio.run() cannot be called from a running event loop',
    'event loop is already running',
)


def is_async_run_footgun(exc: BaseException) -> bool:
    """True when ``exc`` looks like the nested-event-loop runtime error."""
    msg = str(exc)
    return any(fragment in msg for fragment in _ASYNC_FOOTGUN_FRAGMENTS)


def is_langgraph_recursion_error(exc: BaseException) -> bool:
    """True when ``exc`` is langgraph's ``GraphRecursionError``.

    Matches by class name + module prefix so the SDK doesn't gain a hard
    langgraph dependency.
    """
    cls = type(exc)
    return cls.__name__ == 'GraphRecursionError' and cls.__module__.startswith('langgraph')


#: Every framework's "hit its recursion/step-limit budget" fingerprint, as a TABLE rather than an
#: inline special case — so a second framework (a non-LangGraph agent loop with its own step-cap
#: exception) is a one-line addition here, never a duplicated ``is_x_recursion_error`` scattered
#: through the classification call site (``child.py``'s dispatch over ``invoke_agent``'s raise).
#: :func:`is_langgraph_recursion_error` is row 1. Consulted with ``any(...)``, so a build that
#: registers a second detector needs no change at the call site.
_RECURSION_LIMIT_DETECTORS: tuple[Callable[[BaseException], bool], ...] = (is_langgraph_recursion_error,)


def is_recursion_limit_error(exc: BaseException) -> bool:
    """True when ``exc`` matches ANY registered recursion/step-limit fingerprint.

    The one call site (``child.py``) reads this instead of :func:`is_langgraph_recursion_error`
    directly, so it never needs editing again when a second framework's detector is added to
    :data:`_RECURSION_LIMIT_DETECTORS`.
    """
    return any(detector(exc) for detector in _RECURSION_LIMIT_DETECTORS)


def render_adapter_error_message(
    subkind: AdapterErrorSubkind | str,
    *,
    thread_id: str = '',
    turn_idx: int = 0,
) -> str:
    """Customer-visible message for an OrchestratorAdapterError.

    Keep aligned with the symptom table in
    ``simulation-setup-python/SKILL.md`` Step 10e — these are the messages
    customers see + grep for when their adapter misuses the protocol.

    NEVER raises. An unrecognised subkind renders a generic message naming the subkind
    verbatim; the caller logs it as a warning and scores the turn (see
    :func:`adapter_error_is_scoreable`). Raising here would make every gap between the emitting
    classifier and this renderer cost a whole scenario — the renderer is an EXPLANATION, and failing to explain something
    must not be able to destroy the measurement it was explaining.
    """
    if subkind == 'invoke_async_footgun':
        return (
            'Your invoke_agent is sync but contains asyncio.run(...). '
            'The child process driving your agent already runs an event '
            'loop and awaits this call. Make invoke_agent `async def` and '
            '`await graph.ainvoke(...)`. See the Python sub-skill SKILL.md '
            'Step 8 (adapter rules: simulation-setup-python/SKILL.md).'
        )
    if subkind == 'seed_state_async_footgun':
        return (
            'Your seed_state hook contains asyncio.run(...). The child '
            'process driving your agent awaits this hook the same way it '
            'awaits invoke_agent — declare seed_state as `async def` and '
            '`await` your storage calls directly. See the Python sub-skill '
            'SKILL.md Step 8 (adapter rules: simulation-setup-python/SKILL.md).'
        )
    if subkind == 'graph_recursion':
        # Keep aligned with orchestrator.py's in-process branch — same triage
        # ordering, same wording. The transcript on disk is the FIRST evidence
        # to inspect: it disambiguates between "the agent couldn't act at all"
        # (cases A/B — empty seed state, broken mock dispatch) and "the agent
        # acted N times then got stuck" (case C — multi-step planning failure).
        # Don't pre-judge; tell the operator to read transcript.json first.
        return (
            f'Agent hit GraphRecursionError on thread {thread_id} '
            f'(turn {turn_idx + 1}). Before re-running anything, inspect '
            f'``<out>/<scenario>/transcript.json`` for the dialog up to '
            f'the crash and ``tool_events.json`` for what actually fired. '
            f'Three failure modes look alike from outside but need '
            f'different fixes:\n'
            f'  (A) LOCAL-SIDE empty seed — transcript shows ZERO '
            f'successful ``ToolMessage`` responses. The agent retried '
            f'its first tool until the budget exhausted because seed_state '
            f'in simulate.py applied 0 entities (or ignored '
            f'initial_world_state). Cheapest check: '
            f'``uv run python plurai_sim/verify_simulate.py '
            f'plurai_sim/simulate.py``.\n'
            f'  (B) REMOTE-SIDE empty mock — transcript shows ToolMessages '
            f'returning empty data ({{"results": []}} / status:"ok" with '
            f'no records) regardless of seed. The deployed mock dispatch '
            f'is broken. Use ``/simulation-diagnose`` with the failing '
            f'scenario; see simulation-diagnose/SKILL.md.\n'
            f'  (C) MULTI-STEP PLAN FAILURE — transcript shows several '
            f'SUCCESSFUL ToolMessages then a complex user message after '
            f'which the agent looped. The agent ran out of recursion '
            f'budget trying to fulfill a multi-part / adversarial request. '
            f'Inspect the LAST HumanMessage in transcript.json; it usually '
            f'demands data the tool surface cannot produce, or contradicts '
            f'a workflow-terminal flag. This is an AGENT bug (planning / '
            f'tool catalog), not a simulation-infra bug.\n'
            f'Read the transcript FIRST. Only re-run after you can name '
            f'which of A/B/C this is.'
        )
    if subkind == 'no_assistant_reply':
        # SCOREABLE, not fatal (see FATAL_ADAPTER_SUBKINDS): this text is a WARNING in the run
        # log beside an empty assistant turn, not an exception. The agent ran; it just returned
        # no message a user could read, which is a verdict its evals must be allowed to take.
        # THREE owners look identical from here: the agent, the customer's harness, or this
        # runner's own reply projection. So the message routes rather than blames, and it
        # says out loud where an SDK defect goes.
        return (
            f'Your agent delivered neither speech nor an action on thread {thread_id} '
            f'(turn {turn_idx + 1}). The turn is recorded as an EMPTY agent '
            f'message and scored as such — an agent that says nothing fails '
            f'the checks that require it to say something. Three different things look like '
            f'this; check them in order. (1) THE AGENT genuinely produced nothing — then this '
            f'is the verdict and nothing needs fixing. (2) THE HARNESS returns its reply where '
            f'this runner does not read it: invoke_agent must return the message list under '
            f'``messages`` (or a plain string), not a bare state dict whose reply lives under '
            f'some other key (adapter rules: simulation-setup-python/SKILL.md Step 8). '
            f'(3) THIS RUNNER could not read a shape invoke_agent DOES return — if the reply is '
            f'visibly in the message list, that is an SDK defect: report it at '
            f'https://github.com/plurai-ai/plurai-simulation-sdk/issues (surface: sdk) rather than editing simulate.py around it. Note that an agent whose '
            f'answer is a render/action verb is NOT case 1: a turn with no speech but a tool call '
            f'is a complete turn, carried as ``agent_actions``.'
        )
    if subkind == 'reply_projection_failed':
        # SCOREABLE, and it names the RIGHT owner. This is the SDK's own reply projection
        # raising something other than the condition it declares — over a customer message
        # shape it could not read, not over anything simulate.py did wrong. Routing this to the
        # customer's file is how an SDK defect gets closed by rewriting a harness around it.
        return (
            f'plurai could not project the agent reply on thread {thread_id} '
            f'(turn {turn_idx + 1}): reading the returned message chain raised. The turn is '
            f'recorded as an EMPTY agent message and scored as such, and the underlying '
            f'exception is on the error envelope and the child stderr. This is an SDK defect, '
            f'not a harness one — report it at https://github.com/plurai-ai/plurai-simulation-sdk/issues '
            f'(surface: sdk) with the message '
            f'shape invoke_agent returned, rather than editing simulate.py around it.'
        )
    if subkind == 'undeclared_input_part':
        # The turn was never DELIVERED: the parent put a content part on it that this harness's
        # `agent.io.input` does not declare, so the child refused to hand it over rather than
        # dropping the part and letting the agent answer a question it was not asked. Named the
        # other way round from the two above — the owner here is the DECLARATION, which is a
        # line in the customer's config, and the child's own message (on the envelope) names
        # the kind it saw and the kinds the config declares.
        return (
            f'plurai did not deliver turn {turn_idx + 1} on thread {thread_id}: it carried a message content part '
            f'this harness does not declare it can read. The turn is recorded as an EMPTY agent message — the agent '
            f'never saw it, so nothing about the agent was measured on this turn. Fix ONE of the two: declare the '
            f'kind under `agent.io.input:` in plurai.config.yaml (and make invoke_agent read it), or stop sending it. '
            f'The declared kinds and the kind observed are on the error envelope and the child stderr.'
        )
    if subkind == 'undeclared_empty_opener':
        # The agent-opens turn, at a bench that never said it could serve one. `invoke_agent`
        # would be called with `{'messages': []}`, and a harness that indexes
        # ``payload['messages'][-1]`` raises on exactly that, which is why the declaration
        # gates the send instead of the harness discovering it at runtime.
        return (
            f'plurai did not deliver the EMPTY OPENER on thread {thread_id} (turn {turn_idx + 1}): driving it would call '
            f'invoke_agent with `{{"messages": []}}`, and this harness does not declare `agent.io.may_open: true`. '
            f'The turn is recorded as an EMPTY agent message. If this agent really does open the conversation, declare '
            f'`agent.io: {{may_open: true}}` in plurai.config.yaml AND make invoke_agent handle an empty message list '
            f'(greet, then wait); if it does not, the scenario set must supply the first turn.'
        )
    if subkind == 'reply_oversized':
        # The reply was produced, but even without its message chain and state it is longer than
        # one IPC line can carry, so the parent never saw it.
        return (
            f'the agent reply on thread {thread_id} (turn {turn_idx + 1}) is longer than one IPC line can carry '
            f'({IPC_LINE_LIMIT_BYTES} bytes) even without its message chain and agent state, so it never reached the '
            f'runner. The turn is recorded as an EMPTY agent message and scored as such. Return a shorter reply from '
            f'invoke_agent, or keep bulky data out of the reply text.'
        )
    # UNKNOWN subkind — the emitting classifier is ahead of this renderer. Explain what is
    # known and let the caller decide (it degrades to a scoreable turn); do NOT raise.
    return (
        f'Your agent failed with adapter condition {subkind!r} on thread '
        f'{thread_id} (turn {turn_idx + 1}), which this build of plurai has no '
        f'specific guidance for — the agent subprocess is newer than the runner '
        f'driving it, or a new condition was added without a message. The turn is '
        f'recorded as an EMPTY agent message and scored as such. Upgrade plurai on '
        f'both sides, and read ``<out>/<scenario>/transcript.json`` for what the '
        f'agent actually did.'
    )


# ---------------------------------------------------------------------------
# Envelope types — one frozen dataclass per ``type`` discriminator.
# ---------------------------------------------------------------------------

# 'mock_unreachable' is a dedicated kind for the connect-mode case where
# ``plurai.configure(...)`` could not reach a healthy deployed mock. It is an
# actionable, expected operational condition (wrong/stale mock URL, mock not
# deployed, transient outage) — distinct from 'other', which would otherwise
# collapse it into an opaque start_session-style bug. The parent renders a
# targeted "is your mock URL reachable?" message for this kind.
# 'seed_state_contract' / 'reset_contract' are the two HARNESS-HOOK kinds: a hook the
# contract calls could not be called as the contract calls it. Separate words because the
# remedies are different files' worth of different — one is the hook that seeds, the other the
# hook that clears — and a run report that attributed both to one name would send the reader to
# the wrong half of simulate.py.
# 'harness_fault' is a raise the harness TAGGED as its own (``plurai.HarnessFault``), and the two
# pipeline envelope refusals the child makes itself.
# 'mock_runtime_fault' is the backend's mock runtime failing to answer a mocked tool, with ``transient``
# set on the envelope; any real harness fault in the same hook keeps 'harness_fault'.
# 'provider_infra' is the agent's own call failing at its provider (a 429, a 5xx, a dropped connection).
# 'mock_init_state_fault' is the mock answering enough to be DIALED (unlike 'mock_unreachable', which
# fires before that) but then failing a state-shaped call during init/reset/seed — typically a /state
# read timing out under backend load. Distinct from 'mock_unreachable' because the remedy differs: a
# mock that answered enough to be dialed is not a connect-posture / URL / deploy problem.
ErrorKind = Literal[
    'adapter',
    'other',
    'agent_raised',
    'provider_infra',
    'mock_unreachable',
    'mock_init_state_fault',
    'seed_state_contract',
    'reset_contract',
    'harness_fault',
    'mock_runtime_fault',
]


@dataclass(frozen=True, slots=True)
class Init:
    """Parent → child: scenario setup. Must be the first message sent."""

    thread_id: str
    mock_service_url: str
    mock_mode: MockMode = 'connect'
    initial_world_state: dict[str, Any] = field(default_factory=dict)
    initial_agent_context: dict[str, Any] = field(default_factory=dict)
    scenario_metadata: dict[str, Any] = field(default_factory=dict)
    # The resume seed for a TRACE-RESUMED scenario: ``{messages: [...],
    # agent_state: {...}}`` exactly as the backend put it on ``CreateThreadResponse``.
    # ``None`` — the only value a fresh scenario ever carries — is the discriminator;
    # there is no parallel "resumed" flag. Passed through VERBATIM: the SDK never reads
    # inside it here, it only hands it to the customer's ``seed_state`` (which is what
    # decides what the agent's history and store actually become) and mirrors it into
    # the run artifacts. A dict rather than a typed model for that reason — the schema
    # is the backend's and the customer's to agree on, and the transport must not become a
    # third party that can reject a shape the two ends already accept.
    resume_seed: dict[str, Any] | None = None
    # The bench's declared agent entry as a ``module:Qual.name`` ref, arming the child's entry
    # witness (``plurai._entry_witness``). ``None`` — unarmed — is left off the wire.
    entry_witness: str | None = None
    type: Literal['init'] = 'init'


@dataclass(frozen=True, slots=True)
class InitOk:
    """Child → parent: scenario setup succeeded; ready for user_turn."""

    # ``seed_state``'s own return value, passed through VERBATIM:
    # ``None`` means full fidelity — the default a harness gets by
    # returning nothing — and any other value is the harness's own account of what it
    # could and couldn't restore. Opaque like ``Init.resume_seed`` for the same reason:
    # the transport must not become a third party that can reject a shape the harness
    # and whoever reads the run record already agree on.
    resume_fidelity: dict[str, Any] | None = None
    # ``True`` when this child armed ``plurai.egress_guard`` before loading the harness. A runner
    # that claimed ``guarded`` withdraws the claim on anything else (an older child SDK).
    egress_guard_armed: bool | None = None
    # ``True`` when this child executes ``UserTurn.user_action``. The parent sends an action only
    # to a child that said so: an older child refuses the unknown field and would die on it.
    user_actions: bool | None = None
    # The WHOLE world right after ``seed_state`` (every type, surfaces or not), read by the same table
    # finalize reads with: ``{session_id, state}``, the ``_finalize_error`` marker on a failed read or
    # one with no ``state`` map, ``None`` with no world route. The backend decides each final row's origin against it.
    post_seed_world_state: dict[str, Any] | None = None
    # ``True`` when this child applies the scenario's channel plan (:mod:`plurai.channels`), read off
    # ``Init.scenario_metadata``. Set only when the plan is non-empty, so a plan-free run's envelope is unchanged;
    # the parent refuses a plan-carrying scenario at a child that did not say so (an older child ignores the plan).
    channel_inputs: bool | None = None
    type: Literal['init_ok'] = 'init_ok'


@dataclass(frozen=True, slots=True)
class InitError:
    """Child → parent: scenario setup failed (import error, seed_state crash, ...)."""

    kind: ErrorKind = 'other'
    subkind: AdapterErrorSubkind | None = None
    exc_class: str = ''
    message: str = ''
    #: On ``kind='mock_runtime_fault'`` only: whether the runtime typed it transient. ``None`` (off the wire) otherwise.
    transient: bool | None = None
    #: On ``kind='mock_runtime_fault'`` only, when the mock never answered the call: its typed cause. ``None`` (off the wire) otherwise.
    route_cause: MockRouteCause | None = None
    #: On ``kind='mock_runtime_fault'`` and ``route_cause='credit'`` only: the credit block ``{code, resets_at?}``. ``None`` (off the wire) otherwise.
    credit: dict[str, str] | None = None
    #: On ``kind='mock_runtime_fault'`` and ``route_cause='capacity'`` only: the capacity block ``{code, resets_at?, reset_source?}``.
    #: ``None`` (off the wire) otherwise.
    capacity: dict[str, str] | None = None
    #: On ``kind='mock_runtime_fault'`` and ``route_cause='unavailable'`` only: the unavailable block ``{cause, code?}``.
    #: ``None`` (off the wire) otherwise.
    unavailable: dict[str, str] | None = None
    type: Literal['init_error'] = 'init_error'


#: ``UserTurn``'s wire discriminator, named because consumers outside this module branch
#: on it — the re-drive classifier singles out a stalled dialog TURN from the other
#: envelopes that share one timeout. A ``slots=True`` dataclass exposes no readable class
#: attribute for the field, so the literal is published here rather than restated.
USER_TURN_ENVELOPE = 'user_turn'


@dataclass(frozen=True, slots=True)
class UserTurn:
    """Parent → child: drive one agent turn.

    ``user_message`` is what the user says this turn, in one of the three shapes the harness's
    own ``agent.io`` declaration (:mod:`plurai.agent_io`) decides between:

    * ``str`` — a plain text turn. THE RULE, stated once here for the three readers that defer
      to it (``child._ChildSession._turn_messages`` / ``._agent_io``,
      ``_subprocess.ChildProcess._declared_io``): a ``str`` turn asks nothing of the
      declaration, so it is never refused and never triggers a config read. A parts list and the
      empty opener both do — and both ride SCORED runs (``UserTurnResponse.user_messages`` is a
      nullable list of messages beside the non-nullable ``user_message`` string, and an
      agent-first scenario sends ``None`` on turn 0), so the declaration-reading paths are live
      production paths rather than an unreached widening.
    * ``list[dict]`` — an OpenAI content-part list (``text`` / ``image_url`` / ``file``), for a
      harness declaring ``agent.io.input``. The child projects the DECLARED kinds verbatim and
      refuses an undeclared one; it does not drop it.
    * ``None`` — the EMPTY OPENER, for a harness declaring ``agent.io.may_open``: the agent
      opens the conversation and is called with ``{'messages': []}``. ``None`` is the one
      spelling (an empty LIST is a message that says nothing, which is a different fact), and
      the parent's ``ChildProcess.user_turn`` will not send it to a harness that did not
      declare it.

    This is an ANNOTATION, not a new capability at the boundary: :func:`decode` validates field
    NAMES and never types (see its docstring), so all three already round-trip this IPC — and
    ``UserTurn`` is SDK-internal, absent from the vendored ``wire/openapi.json``, so widening it
    is not a contract regen.
    """

    turn_idx: int
    user_message: str | list[dict[str, Any]] | None
    #: The user's decision on the ``interrupt`` this turn answers — one of its offered ``choices``
    #: (``UserTurnResponse.interrupt_choice``), or ``None`` on a turn that answers none.
    interrupt_choice: str | None = None
    #: The act the simulated user took on what she sees (``UserTurnResponse.user_action``:
    #: ``{turn_idx, surface, action, entity_id, updates}``), applied to the world at turn open, before
    #: the agent runs. Sent only to a child whose ``InitOk.user_actions`` is ``True``.
    user_action: dict[str, Any] | None = None
    type: Literal['user_turn'] = 'user_turn'


#: What ``reply_source`` may say — the CONSUMER contract, so every reader must handle all five.
#: Not a free-text field: a consumer branches on it, and a label nobody can enumerate is a label
#: nobody checks. ``unknown`` is mandatory and load-bearing — any suffix this classifier cannot
#: read lands there rather than being flattered into ``assistant``.
#:
#: ``plurai.child._extract_reply`` emits four of them. ``empty`` belongs to the OTHER producer of an agent
#: response — ``plurai/_subprocess.py``'s scoreable adapter degradation, which returns an empty
#: reply without a message chain to classify. An empty SUFFIX never reaches here as ``empty``: with
#: a typed interrupt it is ``interrupt``, and without one the turn raises
#: :class:`NoAssistantReplyError` rather than inventing a reply.
REPLY_SOURCES = ('assistant', 'tool_result', 'interrupt', 'empty', 'unknown')

#: The label for a turn whose reply is empty BY CONSTRUCTION — the parent emptied it rather than
#: the agent producing nothing. Named here so the producer that stamps it and the vocabulary that
#: declares it cannot drift.
REPLY_SOURCE_EMPTY = 'empty'


#: The trail contract's THIRD dispatch outcome, stamped on a ``agent_state.tool_trail`` entry's
#: ``status``. Declared here for the same reason :data:`REPLY_SOURCE_EMPTY` is — the harness that
#: stamps it and the readers that key on it must not spell it twice.
#:
#: ``status`` needs a THIRD value beside ``ok`` / ``error`` (and ``unavailable``): dispatched-ok,
#: dispatched-and-failed, and **not dispatchable by name**. A harness gating a verb its flow does
#: not expose that could only stamp ``error`` would make the run report read invented verbs as a
#: failing environment (``dead_tools`` counts errors over calls) rather than as an agent asking for
#: capabilities it does not have. Nothing failed there — no endpoint was reached and no capability
#: was blocked — so it needs its own word, not a failure spelling.
#:
#: **NOT** ``unknown_tool``. That token already means an ORPHAN TOOL RESULT on both sides of this
#: seam (``plurai/observability/tool_call_recovery.py`` and the backend's counterpart), and one word meaning
#: two unrelated things in one pipeline is the defect, not the fix.
TRAIL_STATUS_NOT_DISPATCHABLE: Final = 'not_dispatchable'

#: Every ``status`` spelling a reader must treat as "the call never left because the name is not
#: one this agent has". :data:`TRAIL_STATUS_NOT_DISPATCHABLE` is what a harness SHOULD write;
#: ``refused`` rides beside it because harnesses already write it and the plugin's report reader
#: (``skills/simulation-run/scripts/build_run_json.py::_TRAIL_NOT_DISPATCHABLE_STATUSES``) accepts
#: both. A reader that took one spelling and not the other would
#: be half-mirrored, which is worse than either.
#:
#: ONE READER CARVES ``refused`` OUT, DELIBERATELY, and the rule above is not unconditional
#: because of it: ``observability.tool_call_recovery._trail_status`` runs ``trail_call_errored``
#: FIRST, and ``refused`` is in that module's ``_TRAIL_FAILED_STATUSES``, so a ``refused`` entry
#: is settled as ``error`` and never reaches the not-dispatchable branch. That set mirrors the
#: backend's own and is parity-checked against it, so it is not this constant's to move.
TRAIL_NOT_DISPATCHABLE_STATUSES: Final = frozenset({'refused', TRAIL_STATUS_NOT_DISPATCHABLE})

#: The trail contract's FOURTH dispatch outcome: a gate the harness or the user owns said no, so the
#: call never ran. A decision, not a failure — deliberately NOT in any failed-status set, because a
#: confinement deny or a user's "no" read as a failed dispatch inflates the error and
#: dead-tool counts. Readers settle it ``status: 'unknown'`` with the ``_denied`` marker, the same
#: abstain-and-keep-the-fact shape as :data:`TRAIL_STATUS_NOT_DISPATCHABLE`; the message lane stamps
#: the same marker when an interrupt's answer is not its declared ``approve`` choice.
TRAIL_STATUS_DENIED: Final = 'denied'

#: The typed record of a refused ATTEMPT, on every ``tool_call`` a gate said no to: ``{by, reason}``.
#: A refusal for a permission, approval or policy reason is something the agent TRIED, so it stays a
#: call in the trail a judge, a structured check and a finding read — this key is what tells it apart
#: from a call that ran. A harness may declare it on a trail entry; a mock declares it as a top-level
#: sibling of its envelope (``{"status": …, "content": …, "refusal": {"reason": …}}``), which the
#: agent never sees.
TRAIL_REFUSAL_KEY: Final = 'refusal'
#: Who said no: the mocked system (``mock``), the harness's approval gate (``harness``), the SDK's
#: own sandbox (``confinement``), or the simulated user answering an interrupt (``user``).
REFUSED_BY: Final = frozenset({'mock', 'harness', 'confinement', 'user'})

#: The reserved action verb for a PAUSE for a human decision, on the per-turn ``agent_actions`` channel:
#: its ``args`` carry ``prompt``, ``choices``, ``actions`` and ``approve`` (the choice that runs them).
#: The backend's reserved agent-action set holds the same word.
INTERRUPT_ACT: Final = 'interrupt'


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Child → parent: the agent's reply for this turn.

    ``ai_text`` is the final user-visible string. ``full_messages`` carries the
    serialized message chain (incl. intermediate ToolMessages + tool-call
    AIMessages) so the parent can preserve them in transcript.json.
    ``is_interrupted`` + ``interrupt_context`` surface a LangGraph
    ``interrupt()`` for the user-simulator to respond to.

    ``agent_state`` carries the agent's NON-message return state for this turn
    (every key of the returned dict except ``messages`` — e.g. a LangGraph
    agent's ``last_suggestions`` / ``last_retrieved_skus``). Without it those
    keys are discarded at the IPC boundary (only ``messages`` crosses), so any
    eval scoring a non-message state field (e.g. a follow-up-suggestions judge)
    sees nothing even though the agent produced it. JSON-safe values only.

    ``reply_source`` says WHERE ``ai_text`` came from
    (:data:`~plurai.child.REPLY_SOURCES`). ``ai_text`` alone cannot be told apart from a
    tool result's text when a turn with no agent words ends on a tool result — a legitimate
    terminal shape (a tool that ends the graph), whose result is relayed as ``ai_text`` because it
    is what the graph returned, but which is not the agent's own words; ``tool_result`` is what
    lets every downstream reader tell. Labelling only; it changes no verdict.
    ``''`` means the producer did not classify — an older child, never "assistant".

    ``agent_actions`` is what the agent DID this turn, and is TRI-STATE: ``[]`` says "this
    producer can observe its agent's actions and there were none", ``None`` says "it cannot
    observe them at all". The wire keeps the distinction for the same reason
    ``turn_tool_events`` does — the field is what decides whether a turn happened, so an
    unobservable turn read as ``[]`` becomes a turn in which nothing happened. It therefore
    DEFAULTS to ``None``: a producer that did not fill it has said nothing, and only
    :func:`~plurai.child._build_agent_response`, which actually read the chain, may say ``[]``.
    """

    ai_text: str
    full_messages: list[dict[str, Any]] = field(default_factory=list)
    is_interrupted: bool = False
    interrupt_context: dict[str, Any] = field(default_factory=dict)
    agent_state: dict[str, Any] = field(default_factory=dict)
    reply_source: str = ''
    agent_actions: list[dict[str, Any]] | None = None
    #: ``{state, reason}`` from the entry witness, cumulative since Init; ``None`` when unarmed.
    entry_witness: dict[str, str] | None = None
    #: What changed in the world over this agent turn (:mod:`plurai.world_delta`): ``None`` from a
    #: producer that did not capture it, else ``observed`` with the delta or ``unobservable`` with a
    #: typed reason. Rides ``/user-turn`` as ``turn_world``.
    turn_world: dict[str, Any] | None = None
    #: How this turn's ``UserTurn.user_action`` resolved (``{turn_idx, status, detail, refused_by, unexecutable_by}``), ``None`` when
    #: the turn carried none. Rides the next ``/user-turn`` as ``user_action_result``.
    user_action_result: dict[str, Any] | None = None
    #: The scenario's channel inputs applied at the open of this turn (:class:`plurai.channels.AppliedChannelInput`
    #: records), ``None`` when the turn carried none. Filed on the transcript row as ``channel_inputs``.
    channel_results: list[dict[str, Any]] | None = None
    type: Literal['agent_response'] = 'agent_response'


def observed_actions_or_none(admitted: list[dict[str, Any]], *, saw_declaration: bool) -> list[dict[str, Any]] | None:
    """The action channel's tri-state, in ONE place: what a reader ADMITTED becomes what it reports.

    ``[]`` is "this producer can observe its agent's acts and there were none"; ``None`` is "it
    cannot observe them at all". The third reading — the one that must never be produced — is a
    reader that SAW a declaration, could take nothing out of it, and reports ``[]``: "observed, and
    none" for a turn that acted. ``saw_declaration`` is that bit, and every reader on this channel
    passes it, whatever the declaration's shape was.

    Stated on the module that TYPES the field (:class:`AgentResponse`) so the child's envelope gate
    and the parent's defensive re-read of a pluggable child's envelope share one answer; two copies
    could disagree, and which answer a turn got would depend on which path produced it.
    """
    return admitted or (None if saw_declaration else [])


@dataclass(frozen=True, slots=True)
class AgentError:
    """Child → parent: agent invocation failed this turn.

    ``full_messages`` / ``agent_state`` carry the child's RESULT for a condition the parent
    degrades to a scoreable turn (see :func:`adapter_error_is_scoreable`). They are what the
    parent builds that turn from: without them it can only manufacture an empty one, and a turn
    the agent really did run reaches transcript.json with ``turn_messages: []`` and
    ``agent_state: {}`` — the tool trail dropped at the one boundary where it can still be kept.
    Empty on the FATAL subkinds, on ``kind='other'`` and on ``'agent_raised'``: there is no result to carry when the
    invocation itself raised.
    """

    kind: ErrorKind = 'other'
    subkind: AdapterErrorSubkind | None = None
    exc_class: str = ''
    message: str = ''
    turn_idx: int = 0
    full_messages: list[dict[str, Any]] = field(default_factory=list)
    agent_state: dict[str, Any] = field(default_factory=dict)
    #: The world at the end of the turn, on the SCOREABLE conditions only (the agent ran and may have
    #: changed it; see :attr:`AgentResponse.turn_world`). ``None`` on every other error.
    turn_world: dict[str, Any] | None = None
    #: How the simulated user's act that opened this turn was applied (see
    #: :attr:`AgentResponse.user_action_result`), on EVERY error: the act happened whatever the agent did.
    user_action_result: dict[str, Any] | None = None
    #: The channel inputs applied before the agent ran (see :attr:`AgentResponse.channel_results`), on EVERY error.
    channel_results: list[dict[str, Any]] | None = None
    #: As :attr:`InitError.transient`.
    transient: bool | None = None
    #: As :attr:`InitError.route_cause`.
    route_cause: MockRouteCause | None = None
    #: As :attr:`InitError.credit`.
    credit: dict[str, str] | None = None
    #: As :attr:`InitError.capacity`.
    capacity: dict[str, str] | None = None
    #: As :attr:`InitError.unavailable`.
    unavailable: dict[str, str] | None = None
    type: Literal['agent_error'] = 'agent_error'


@dataclass(frozen=True, slots=True)
class PipelineInvoke:
    """Parent → child: drive a one-shot pipeline (no dialog, no tools).

    Used when the customer's simulate.py exports ``invoke_pipeline(inputs)
    -> dict`` for one-shot LLM pipelines (sample generation, classification,
    summarization). Carries the typed input payload the pipeline_io.yaml
    declared. Pairs with ``PipelineResult`` (success) or ``PipelineError``
    (timeout / handler raise / non-dict return). Distinct from ``UserTurn``
    because one-shot has no recursion budget, no turn counter, and a
    different wall-clock timeout policy.
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    type: Literal['pipeline_invoke'] = 'pipeline_invoke'


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Child → parent: one-shot pipeline completed successfully."""

    outputs: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    #: As ``AgentResponse.entry_witness``.
    entry_witness: dict[str, str] | None = None
    #: ``{path: source type}`` for each output value :func:`_json_shaped` coerced to its JSON form
    #: (a datetime, a Decimal, a set …); ``None`` when nothing was coerced.
    values_coerced: dict[str, str] | None = None
    type: Literal['pipeline_result'] = 'pipeline_result'


@dataclass(frozen=True, slots=True)
class PipelineError:
    """Child → parent: one-shot pipeline failed (handler raise / timeout / non-dict return).

    Separate from ``AgentError`` because the dispatch semantics differ —
    no recursion classification, no turn_idx, distinct timeout policy. Used
    by the parent to write ``pipeline_io.json`` with the ``error`` field
    populated for forensic inspection.
    """

    exc_class: str = ''
    message: str = ''
    timed_out: bool = False
    #: ``'harness_fault'`` on a refusal the harness made; ``None`` (off the wire) on an agent outcome.
    kind: ErrorKind | None = None
    #: As :attr:`InitError.transient`.
    transient: bool | None = None
    #: As :attr:`InitError.route_cause`.
    route_cause: MockRouteCause | None = None
    #: As :attr:`InitError.credit`.
    credit: dict[str, str] | None = None
    #: As :attr:`InitError.capacity`.
    capacity: dict[str, str] | None = None
    #: As :attr:`InitError.unavailable`.
    unavailable: dict[str, str] | None = None
    type: Literal['pipeline_error'] = 'pipeline_error'


#: The two ``pipeline_invoke`` refusals that are HARNESS FAULTS rather than agent outcomes.
#:
#: An agent that RAISES — or returns a non-dict, or runs past its wall clock — has told us
#: something about itself: the parent records it as ``pipeline_error`` and finalizes, so a
#: deliberate-refusal policy can credit it — a judge reads it, and the code_eval sandbox binds
#: ``pipeline_error`` beside ``pipeline_inputs`` / ``pipeline_outputs``. These two say nothing about the agent at all: the
#: harness sent an envelope the child was never in a position to answer. The child returns them
#: with ``kind='harness_fault'``, and the parent routes that KIND to ``error.json`` + ``/fail``
#: with no verdict — never these words.
PIPELINE_NO_HOOK_MESSAGE = 'simulate.py does not export invoke_pipeline; received pipeline_invoke for a dialog-only adapter.'
PIPELINE_NOT_INITIALIZED_MESSAGE = 'received pipeline_invoke before init'


@dataclass(frozen=True, slots=True)
class Finalize:
    """Parent → child: dialog complete; return world snapshot + events."""

    type: Literal['finalize'] = 'finalize'


@dataclass(frozen=True, slots=True)
class FinalizeResponse:
    """Child → parent: final world state + recorded tool events.

    ``unanswered_mock_calls`` is every tool call of this session that ``/mock/execute`` did not
    answer, so the agent was handed a stub — one record each (``plurai._mocking.resolution
    .UnansweredCall._asdict()``: tool, turn, status_code, code, transport, served,
    restart_shaped). ``[]`` is a session in which every call was answered; ``None`` claims nothing
    (no session, or a ``child_factory`` peer that does not record them). A fact about the
    measurement, never a verdict: whether the world can be trusted is decided by the world read
    at finalize, and the scenario is graded either way.

    ``world_values_coerced`` is ``{path: source type}`` for each value of the harness's own world
    snapshot that was carried in its JSON form (a datetime as ISO text, a Decimal as a number …),
    see :func:`_json_shaped`; ``None`` when nothing was coerced.

    ``capture_retryable`` is ``True`` when a finalize-time snapshot read failed in a way one more
    read after the mock proves ready can clear: the way a mock restart fails (a 503 that outlived
    the retry, nothing answering) or a read that timed out. It is the ONE retry signal a failed
    read carries — a read is not a tool call and never lands in ``unanswered_mock_calls``. ``None``
    claims nothing, and is what every clean finalize sends.

    ``final_world_state`` and ``tool_events`` are BOTH tri-state, under one rule, and the third
    state is the point. A dict — ``{'events': [...]}``, ``{'state': {...}}``, the empty ones
    included — is an OBSERVED snapshot: a channel this run could read, which happened to hold
    nothing. ``None`` is ABSENT: no channel was there to read, because the child held no plurai
    session, so it took no snapshot and is claiming nothing. Collapsing the two to ``{}`` would
    make "the agent invoked zero tools" and "nobody was watching" one answer — likewise "the agent
    created no booking" and "nobody looked at the store" — and every judge downstream would read
    the first, scoring a CONCLUSIVE pass on a channel no one measured. Both default to ``None`` (not
    ``{}``) so a peer that omits either field on the IPC wire — ``child_factory`` is a supported
    extension point — is read as silent rather than as having measured an empty world.

    The gate is DECLARED, never inferred: *was there a session to read*. Emphatically not "the
    snapshot came back empty, so presumably nobody read it" — that inference is the thing the
    third state replaces.

    ``None`` is how absence rides THIS wire and the ``/finalize`` wire (both fields are
    ``dict | None`` in the wire spec, and the finalize dump is ``exclude_none=False``, so the
    null is carried rather than dropped). It is NOT how it rides the run's artifacts, and the two
    files differ from each other: ``tool_events.json`` records a literal ``null``, while
    ``final_world_state.json`` must always be a JSON OBJECT — the backend normalizes an unread world
    into a typed ``{'entities': {}, '_unobserved': True}`` marker on every ``/finalize`` return
    path, and its upload loader RAISES on a null under that filename. See
    ``run_simulation._write_success_artifacts``.

    ``unread_inputs`` is the Init ``initial_agent_context`` keys the harness never read — through
    ``seed_state`` param 1, the turn-1 payload, a hop's inputs or the in-process mock's
    ``agent_context`` (:mod:`plurai._input_reads`); ``[]`` when every key was read on some lane.
    ``None`` = claims nothing: a child that never initialised, a custom ``child_factory``, or an unread
    key a mock could still serve on demand (a registered handler; a deployed mock) — never "all read".
    A fact: the backend owns whether an unread key is an undelivered input.

    ``unread_parts`` is a SEPARATE fact: the documents handed over as ``file`` content parts on a turn
    (the opener's included) whose ``file_data`` the harness never read, named by field (``<field>.<ext>`` →
    ``field``). Never an ``initial_agent_context`` key — a handed-over document is not one — so it is never folded
    into ``unread_inputs`` here: its remedy is reading the turn's parts (``turn_documents``), not serving a key.
    ``[]`` when every part was read; ``None`` before Init. A read made only through a C or pydantic serializer
    (``model_dump``, ``orjson``) is not seen, exactly as for a key.

    ``interception_changed`` names the files routing the agent's vendor calls to the mock that changed
    during the session (:mod:`plurai._mocking.interception_watch`): calls made after the change may have
    reached the real service, and the world does not hold their effects. ``None`` when none changed, or
    when the harness recorded none.
    """

    final_world_state: dict[str, Any] | None = None
    tool_events: dict[str, Any] | None = None
    unanswered_mock_calls: list[dict[str, Any]] | None = None
    capture_retryable: bool | None = None
    world_values_coerced: dict[str, str] | None = None
    unread_inputs: list[str] | None = None
    unread_parts: list[str] | None = None
    interception_changed: list[str] | None = None
    type: Literal['finalize_response'] = 'finalize_response'


def restart_shaped_calls(calls: list[dict[str, Any]] | None) -> int:
    """How many of ``unanswered_mock_calls`` failed the way a mock restart fails — the one count
    retry policy reads off them. ``None`` (nothing claimed) counts none."""
    return sum(1 for call in calls or () if isinstance(call, dict) and call.get('restart_shaped') is True)


@dataclass(frozen=True, slots=True)
class Shutdown:
    """Parent → child: clean shutdown signal. Child exits 0 after receipt."""

    type: Literal['shutdown'] = 'shutdown'


Message = (
    Init | InitOk | InitError | UserTurn | AgentResponse | AgentError | PipelineInvoke | PipelineResult | PipelineError | Finalize | FinalizeResponse | Shutdown
)


_MESSAGE_TYPES: dict[str, type[Message]] = {
    'init': Init,
    'init_ok': InitOk,
    'init_error': InitError,
    'user_turn': UserTurn,
    'agent_response': AgentResponse,
    'agent_error': AgentError,
    'pipeline_invoke': PipelineInvoke,
    'pipeline_result': PipelineResult,
    'pipeline_error': PipelineError,
    'finalize': Finalize,
    'finalize_response': FinalizeResponse,
    'shutdown': Shutdown,
}


class ProtocolDecodeError(ValueError):
    """A line on the IPC channel could not be decoded as a known envelope."""


#: Fields a newer peer added, left OFF the wire while unset: :func:`decode` refuses unknown field
#: names, so an older peer can only read an envelope that does not carry them.
_OMITTED_WHEN_UNSET: frozenset[str] = frozenset(
    {
        'capacity',
        'capture_retryable',
        'channel_inputs',
        'channel_results',
        'egress_guard_armed',
        'entry_witness',
        'interception_changed',
        'kind',
        'post_seed_world_state',
        'route_cause',
        'transient',
        'turn_world',
        'unanswered_mock_calls',
        'unavailable',
        'unread_inputs',
        'unread_parts',
        'user_action',
        'values_coerced',
        'world_values_coerced',
        'user_actions',
        'user_action_result',
    }
)


#: Fields an OLDER peer may send that this side does not have, per envelope type. Taken off the payload at
#: decode so a child pinned to an older plurai (``PLURAI_CHILD_PYTHON``) still decodes, instead of every
#: envelope failing. Never DROPPED as a claim: ``stub_fallback_count`` / ``mock_unreachable_count`` are an older
#: child's stub counters, so :func:`_unanswered_from_retired_counts` carries what they say onto
#: ``unanswered_mock_calls``.
_RETIRED_FIELDS: dict[str, frozenset[str]] = {
    'finalize_response': frozenset({'stub_fallback_count', 'mock_unreachable_count'}),
}


def _retired_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _unanswered_from_retired_counts(retired: dict[str, Any]) -> list[dict[str, Any]] | None:
    """An older child's stub counters as ``unanswered_mock_calls`` rows, or ``None`` when it sent none readable.

    ``0`` is that child's claim that every call was answered (``[]``). A count above zero becomes that many
    SYNTHESIZED rows with ``served: 'unknown'`` — the child named neither the call nor what the agent was served —
    of which the first ``mock_unreachable_count`` are ``restart_shaped`` (the subset it counted as a mock that
    went away), so the measurement folds them into ``measurement_unclean_because`` exactly as named rows.
    """
    stubbed = _retired_count(retired.get('stub_fallback_count'))
    if stubbed is None:
        return None
    unreachable = min(_retired_count(retired.get('mock_unreachable_count')) or 0, stubbed)
    return [
        {'tool': None, 'turn': None, 'status_code': None, 'code': None, 'transport': None, 'served': 'unknown', 'restart_shaped': index < unreachable}
        for index in range(stubbed)
    ]


def encode(msg: Message) -> bytes:
    """Serialize ``msg`` to a single newline-terminated JSON line (UTF-8)."""
    payload = {k: v for k, v in asdict(msg).items() if not (v is None and k in _OMITTED_WHEN_UNSET)}
    return (json.dumps(payload, default=str) + '\n').encode('utf-8')


def decode(line: str | bytes) -> Message:
    """Parse one JSONL line into the corresponding envelope dataclass.

    Field-name validation (not field-type) is intentional at this IPC
    boundary: the child and parent may run different Python versions, and
    ``dataclass(**payload)`` raises ``TypeError`` on unknown/missing field
    *names* while accepting any compatible *values* — keeping the wire
    format forward-compatible without a schema-registry.
    """
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    line = line.strip()
    if not line:
        raise ProtocolDecodeError('empty line')
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolDecodeError(f'invalid JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise ProtocolDecodeError(f'expected JSON object, got {type(payload).__name__}')
    msg_type = payload.pop('type', None)
    cls = _MESSAGE_TYPES.get(msg_type)
    if cls is None:
        raise ProtocolDecodeError(f'unknown message type: {msg_type!r}')
    retired = {name: payload.pop(name) for name in _RETIRED_FIELDS.get(msg_type, ()) if name in payload}
    if msg_type == 'finalize_response' and retired and payload.get('unanswered_mock_calls') is None:
        payload['unanswered_mock_calls'] = _unanswered_from_retired_counts(retired)
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ProtocolDecodeError(f'invalid fields for {msg_type!r}: {exc}') from exc


__all__ = [
    'FATAL_ADAPTER_SUBKINDS',
    'IPC_LINE_LIMIT_BYTES',
    'IPC_TRIMMED_KEY',
    'PIPELINE_NOT_INITIALIZED_MESSAGE',
    'PIPELINE_NO_HOOK_MESSAGE',
    'QUARANTINE_REFUSAL_PREFIX',
    'AdapterErrorSubkind',
    'AgentError',
    'AgentFault',
    'AgentResponse',
    'ErrorKind',
    'Finalize',
    'FinalizePollTimeout',
    'FinalizeResponse',
    'HarnessFault',
    'HarnessFaultRaised',
    'HarnessIpcOverrun',
    'OversizedIpcLine',
    'MockRuntimeFaultRaised',
    'Init',
    'InitError',
    'InitOk',
    'Message',
    'OrchestratorAdapterError',
    'PipelineError',
    'PipelineInvoke',
    'PipelineResult',
    'ProtocolDecodeError',
    'ScenarioQuarantined',
    'Shutdown',
    'UserTurn',
    'adapter_error_is_scoreable',
    'decode',
    'encode',
    'is_async_run_footgun',
    'is_langgraph_recursion_error',
    'is_recursion_limit_error',
    'read_ipc_line',
    'render_adapter_error_message',
]

# --- shared JSON shaping -----------------------------------------------------
# Lives HERE, not on either side, because BOTH need it: the child shapes a hop's outputs
# before they cross the IPC boundary, and the parent shapes the same payload again for the
# observability record. A parent importing `child` would be the wrong direction — that
# module is the subprocess entry point.


def _json_shaped(outputs: dict[str, Any], *, coerced: dict[str, str], _path: str = 'outputs') -> dict[str, Any]:
    """The hop's outputs as JSON, or ``TypeError`` naming exactly what could not be carried.

    The envelope encoder ships with ``default=str``, which SILENTLY replaces an unserializable
    value with its repr — so a structured-output extractor (the common hop) returning a Pydantic
    model would hand every judge and every ``code_eval`` a STRING. A model is DUMPED; a value with ONE
    obvious JSON form is COERCED to it (:func:`_coerce_scalar`) and its path recorded in
    ``coerced`` (``{path: source type}``, REQUIRED so a caller cannot drop the record); anything
    else that cannot be carried is a NAMED failure, never a repr.
    """

    def _coerce(value: Any, path: str) -> Any:
        if hasattr(value, 'model_dump') and callable(value.model_dump):
            # The dump itself can RAISE — a model holding an opaque object raises
            # ``PydanticSerializationError`` (a ``ValueError``), not a ``TypeError``. Re-raised as
            # the same NAMED ``TypeError`` every other unserializable value gets, so every path out
            # of this function keeps the promise above.
            try:
                return value.model_dump(mode='json')
            except Exception as exc:
                raise TypeError(
                    f'{path} is a {type(value).__name__} whose model_dump failed ({type(exc).__name__}: {exc}), so it cannot be carried as JSON'
                ) from None
        if isinstance(value, dict):
            # KEYS too: ``json.dumps`` accepts only ``str``/``int``/``float``/``bool``/``None``
            # keys, so any other key is carried as its text and recorded, never left to raise
            # from the envelope encoder.
            out: dict[Any, Any] = {}
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, type(None))):
                    coerced[f'{path}.<key {key!r}>'] = type(key).__name__
                    # The key's own JSON form when it has a scalar one (a datetime's ISO text, an
                    # Enum's value), else its text.
                    carried_key = _coerce_scalar(key)
                    key = carried_key if isinstance(carried_key, (str, int, bool)) else str(key)  # noqa: PLW2901
                out[key] = _coerce(item, f'{path}.{key}')
            return out
        if isinstance(value, (list, tuple)):
            return [_coerce(item, f'{path}[{index}]') for index, item in enumerate(value)]
        if (carried := _coerce_scalar(value)) is not _NO_JSON_FORM:
            if carried is value:
                return value
            # The carried form is shaped in turn — an Enum whose value is a datetime, a set of Decimals —
            # and the path records the OUTER type, which is what the harness returned.
            shaped = _coerce(carried, path)
            coerced[path] = type(value).__name__
            return shaped
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            raise TypeError(f'{path} is a {type(value).__name__}, which cannot be carried as JSON') from None
        return value

    return _coerce(outputs, _path)


_NO_JSON_FORM: Final = object()


def _coerce_scalar(value: Any) -> Any:
    """``value`` itself when JSON already carries it, its ONE obvious JSON form when it has one, else
    :data:`_NO_JSON_FORM`. Datetimes become ISO text, a Decimal a number (an int when integral), a
    UUID its text, a set a sorted list, an Enum its value, bytes base64 text, and NaN/Inf ``None``
    (``json.dumps`` would otherwise write a bare ``NaN``, which is not JSON)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            return None
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(value)
        except TypeError:
            return sorted(value, key=repr)
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode('ascii')
    return _NO_JSON_FORM
