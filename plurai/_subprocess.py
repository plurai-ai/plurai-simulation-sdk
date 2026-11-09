"""Parent-side IPC driver: spawn ``python -m plurai.child`` and drive the protocol.

:class:`ChildProcess` owns the subprocess and two background reader tasks
(stdout → response queue, stderr → logger). Each high-level call
(``init`` / ``user_turn`` / ``finalize``) writes one envelope, awaits one
response or the child's exit (whichever comes first), and applies a timeout.

Adapter errors classified by the child (``AgentError(kind="adapter", ...)``)
are translated back into :class:`OrchestratorAdapterError` with the canned
customer-visible message — preserving the error UX of the in-process path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from plurai._protocol import (
    AGENT_RAISED_KIND,
    IPC_LINE_LIMIT_BYTES,
    IPC_TRIMMED_KEY,
    PROVIDER_INFRA_KIND,
    REPLY_SOURCE_EMPTY,
    AgentError,
    AgentRaised,
    AgentResponse,
    AgentTurnTimeout,
    Finalize,
    FinalizeResponse,
    HarnessChildExited,
    Init,
    GraphRecursionWrap,
    HarnessResetContract,
    HarnessFaultRaised,
    HarnessIpcOverrun,
    HarnessSeedStateContract,
    InitError,
    InitOk,
    Message,
    MockInitStateFault,
    MockMode,
    MockRuntimeFaultRaised,
    MockUnreachableError,
    OrchestratorAdapterError,
    OversizedIpcLine,
    PipelineError,
    PipelineInvoke,
    PipelineResult,
    ProtocolDecodeError,
    ProviderInfra,
    Shutdown,
    UserTurn,
    adapter_error_is_scoreable,
    decode,
    encode,
    read_ipc_line,
    render_adapter_error_message,
)
from plurai.agent_io import DEFAULT_AGENT_IO, AgentIO, declared_agent_io
from plurai.child_env import launch_environment
from plurai.capacity import capacity_block_from_body, unavailable_block_from_record
from plurai.credit import credit_block_from_body
from plurai.host_effects import redact_cli_credentials, scrub_parent_record
from plurai.settings import FULL_RUN_AGENT_TURN_TIMEOUT


def _provider_infra(message: str, err: InitError | AgentError | PipelineError, *, thread_id: str | None) -> ProviderInfra:
    """The agent's provider (or its sign-in) failed a hook's call — in setup, a turn or a pipeline: never the agent's, never the harness's."""
    return ProviderInfra(f'{message}: {err.exc_class}: {err.message}', child_exc_class=err.exc_class, child_message=err.message, thread_id=thread_id)


#: A raise inside the agent's own turn, by the child's typed ``kind``: the agent's outcome, or its provider's failure.
_AGENT_TURN_RAISES: dict[str, type[OrchestratorAdapterError]] = {AGENT_RAISED_KIND: AgentRaised, PROVIDER_INFRA_KIND: ProviderInfra}

#: The result of a user act the agent process could not be sent: its plurai SDK predates user acts.
_UNEXECUTABLE_BY_AN_OLDER_CHILD: dict[str, Any] = {
    'status': 'unexecutable',
    'detail': "the agent process runs a plurai SDK that cannot apply the simulated user's acts; update plurai in the agent's environment",
}


@dataclass(frozen=True)
class _DecodeFailure:
    """Sentinel pushed onto the response queue when stdout produces non-JSON.

    Without it, the parent would silently drop the offending line, the
    response for the in-flight envelope would never arrive, and the customer
    would see a misleading "agent turn exceeded {timeout}s" after the full
    ``agent_turn_timeout`` elapsed. With it, ``_await_response_or_exit``
    surfaces the actual cause (usually a stray ``print(...)`` in the
    customer's simulate.py) immediately.
    """

    raw_line: bytes
    error: str


logger = logging.getLogger(__name__)


#: ``_thread_id`` before ``init()`` names the real thread. Only ever rendered into an
#: error message — never handed on as an id, because a caller that resolves
#: ``exc.thread_id or <the id it recorded at create>`` would take this placeholder over
#: the genuine one.
_UNINITIALIZED_THREAD_ID = '(uninitialized)'

#: Each child starts in its own session (POSIX), so it leads a process group, and closing it ends what its harness
#: spawned (:meth:`ChildProcess._reap_group`). A leased resource — a ``harness.single_instance`` port, a world copy — is
#: handed to the next child the moment this one closes, so a server or bridge the harness started must not outlive it:
#: it would still hold the port or the rows the next child binds.
_OWN_PROCESS_GROUP = os.name == 'posix'
#: The child's own copy of that fact (``plurai.child.GROUP_LEADER_ENV``), spelled here so this module never imports the child.
_GROUP_LEADER_ENV = 'PLURAI_CHILD_GROUP_LEADER'

#: Set by the runner while its egress verdict is ``guarded``: called when a child's Init reply does
#: NOT confirm the in-process guard, it withdraws the claim, re-measures, and returns the line naming
#: each egress finding that now holds the run — gate id and fingerprint, so an overrule can lift it on
#: a re-run — or ``None`` when nothing holds. ``None`` = no claim to confirm.
EGRESS_GUARD_UNCONFIRMED: Callable[[], str | None] | None = None


# stderr lines from the child matching this pattern get escalated to WARNING
# in the parent's logger. Everything else stays at DEBUG (keeps default
# customer logs quiet but lets you turn them up with logging.DEBUG).
#
# The alternation catches (after lstripping leading whitespace):
#   - Bare "Traceback" header
#   - Explicit "RuntimeError"
#   - "[plurai…]" prefix lines — both "[plurai.child]" (already handled) and
#     "[plurai]" (from the mock-env HTTP client: "  [plurai] HTTP 401 …", printed
#     with a 2-space indent which lstrip removes before matching)
#   - Python logging level prefixes: "WARNING:…" / "ERROR:…" / "CRITICAL:…"
#     (the default lastResort/StreamHandler format used when no basicConfig is
#     called — these carry mock-env proxy errors like "mock-env proxy returned
#     HTTP 401 … the simulation result is NOT trustworthy")
#   - Standard Python exception class headers ("TypeError:", "ValueError:", ...)
#   - The warnings-module header "<file>:<lineno>: <Category>: <message>" — the
#     line that carries the category, the message AND where it fired. It starts
#     with the path, so none of the alternatives above reach it; without this one
#     the only promoted line is the boilerplate tail ("RuntimeWarning: Enable
#     tracemalloc …"), which says nothing about what actually happened.
# Traceback FRAMES (the indented "  File ..." lines and "    code" lines)
# don't match here — they get promoted via the stateful in-traceback flag
# in ``_read_stderr`` once the header fires.
_EXC_CLASS_NAME = r'[A-Z][A-Za-z_]*(?:Error|Exception|Warning)'
_STDERR_WARN_RE = re.compile(
    r'^('
    r'Traceback'
    r'|RuntimeError'
    r'|\[plurai(?:\.[^\]]+)?\]'
    r'|(?:WARNING|ERROR|CRITICAL):'
    rf'|{_EXC_CLASS_NAME}[: ]'
    rf'|.+:\d+: {_EXC_CLASS_NAME}:'
    r')'
)

# Once "Traceback (most recent call last):" fires, every subsequent stderr
# line is part of the traceback (frames + final exception summary) until
# the summary line matches one of these patterns. We promote those frames
# to WARNING so a default-log-level run shows the full trace, not just the
# header. Without this, debugging swallowed exceptions in the child
# requires re-running with LOGGING__LEVEL=DEBUG.
_TRACEBACK_HEADER_RE = re.compile(r'^Traceback \(most recent call last\):')
_EXCEPTION_SUMMARY_RE = re.compile(rf'^{_EXC_CLASS_NAME}(?::|$)')

# The same stateful treatment for the warnings module's multi-line render: the
# header above, then INDENTED continuation lines (the echoed source line), then
# an optional non-indented "<Category>: …" tail that _STDERR_WARN_RE catches on
# its own. Continuation lines ride along until the indentation stops.
_WARNING_HEADER_RE = re.compile(rf'^.+:\d+: {_EXC_CLASS_NAME}:')


class ChildProcess:
    """Async context manager wrapping the simulate.py child subprocess.

    Typical use::

        async with ChildProcess(simulate_path, agent_turn_timeout=1200.0) as child:
            await child.init(thread)
            for turn_idx, user_msg in enumerate(...):
                response = await child.user_turn(turn_idx, user_msg)
                ai_text = response.ai_text
            final_state, events = await child.finalize()
    """

    def __init__(
        self,
        simulate_path: Path,
        *,
        agent_turn_timeout: float = FULL_RUN_AGENT_TURN_TIMEOUT,
        env: dict[str, str] | None = None,
    ):
        self.simulate_path = Path(simulate_path)
        self.agent_turn_timeout = agent_turn_timeout
        # ``env`` is the child's whole environment when a caller supplies one (the orchestrator, which
        # resolves the pair and the world slot; tests). ``None`` builds it here from this process's
        # environment through the allowlist (:func:`plurai.child_env.launch_environment`), never by
        # inheritance. Per-run mock isolation is owned CHILD-side, not via a parent-supplied
        # PLURAI_MOCK_URL: in ``mock_mode='local'`` the child picks its own fresh loopback port
        # (``_free_loopback_port`` / ``_serve_local_mock`` in ``plurai/__init__.py``) so no two
        # children share one :8199; in ``'connect'`` mode it dials the server-supplied deployed URL.
        self._env = env
        self._proc: asyncio.subprocess.Process | None = None
        # The process group the child leads (:data:`_OWN_PROCESS_GROUP`), reaped once at close. ``None`` before
        # :meth:`start`, on Windows, and once reaped, so the group id is never signalled after its close.
        self._pgid: int | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._response_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._closed = False
        self._thread_id = _UNINITIALIZED_THREAD_ID  # set on init() for error rendering
        # The world right after ``seed_state``, set by :meth:`init` (see ``InitOk.post_seed_world_state``).
        self.post_seed_world_state: dict[str, Any] | None = None
        # Turns the parent emptied on the agent's behalf, and how many it drove in total —
        # the degradation RECORD (see ``degraded_turns`` / ``turns_driven``).
        self._degraded_turns: list[dict[str, Any]] = []
        self._turns_driven = 0
        # Replies the CHILD trimmed to fit one IPC line, ``{turn_idx}`` each (see ``ipc_trimmed_turns``).
        self._ipc_trimmed_turns: list[dict[str, Any]] = []
        # The child's read record, set by :meth:`finalize`; ``None`` until then (see ``FinalizeResponse``).
        self.unread_inputs: list[str] | None = None
        #: The handed-over ``file`` parts the harness never read, by field (``FinalizeResponse.unread_parts``).
        self.unread_parts: list[str] | None = None
        # The finalize world's coerced values (``FinalizeResponse.world_values_coerced``), set by :meth:`finalize`.
        self.world_values_coerced: dict[str, str] | None = None
        # The interception files the agent changed (``FinalizeResponse.interception_changed``), set by :meth:`finalize`.
        self.interception_changed: list[str] | None = None
        # The harness's ``agent.io`` declaration, read at most once and only when a turn needs
        # it — see ``_declared_io``.
        self._agent_io: AgentIO | None = None
        # Whether the child said at init that it applies the simulated user's acts
        # (``InitOk.user_actions``). An older child refuses the unknown field, so it is never sent one.
        self._child_applies_user_actions = False
        # Whether the child said at init that it applies the scenario's channel plan (``InitOk.channel_inputs``).
        # Read by the orchestrator, which refuses a plan-carrying scenario at a child that did not say so.
        self.applies_channel_inputs = False
        # How the user's act went on the last turn that RAISED instead of answering (``None`` when
        # none did, or when the child died without saying). A caller that survives that raise — the
        # closing turn — still reports the act rather than leaving it unreported.
        self.raised_turn_user_action_result: dict[str, Any] | None = None
        # Ring buffer of the last N child stderr lines. When the child dies
        # before responding to ``init`` (most common cause: ImportError or
        # missing dep in the customer's venv), the raw RuntimeError shows
        # only the return code — surfacing the tail of stderr here is what
        # turns "child subprocess exited with code 1" into something a
        # caller can actually act on without grepping a separate log stream.
        self._stderr_tail: deque[str] = deque(maxlen=20)

    # ----- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Spawn ``python -m plurai.child <simulate_path>`` and start readers.

        The child interpreter defaults to ``sys.executable`` (same venv as the
        parent). When the agent under test pins dependencies that conflict with
        the backend's own (e.g. a legacy langchain that can't coexist with
        ``langchain>=1.0``), set ``PLURAI_CHILD_PYTHON`` to an interpreter from
        an isolated agent venv. The parent↔child wire is plain JSON over
        stdin/stdout, so the two processes may run different dependency stacks;
        the child venv only needs the plurai SDK + the agent's own deps.
        """
        child_python = os.environ.get('PLURAI_CHILD_PYTHON') or sys.executable
        if child_python != sys.executable:
            logger.info('[plurai] child interpreter override: %s', child_python)
        self._proc = await asyncio.create_subprocess_exec(
            child_python,
            '-m',
            'plurai.child',
            str(self.simulate_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._spawn_env(),
            # See ``IPC_LINE_LIMIT_BYTES`` — default 64 KiB truncates real
            # world-state payloads.
            limit=IPC_LINE_LIMIT_BYTES,
            start_new_session=_OWN_PROCESS_GROUP,
        )
        if _OWN_PROCESS_GROUP:
            self._pgid = self._proc.pid
        self._stdout_task = asyncio.create_task(self._read_stdout(), name='plurai-child-stdout')
        self._stderr_task = asyncio.create_task(self._read_stderr(), name='plurai-child-stderr')
        logger.info(
            '[plurai] starting child: %s pid=%s',
            self.simulate_path,
            self._proc.pid,
        )

    def _spawn_env(self) -> dict[str, str]:
        """The child's environment (``env``, else the allowlisted launch environment), marked as leading its own process
        group when it is started in its own session, so it ends that group itself if the parent dies before closing it."""
        env = dict(self._env) if self._env is not None else launch_environment(self.simulate_path)
        if _OWN_PROCESS_GROUP:
            env[_GROUP_LEADER_ENV] = '1'
        return env

    async def __aenter__(self) -> ChildProcess:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Send Shutdown, then wait briefly; SIGTERM/SIGKILL if the child hangs. The group is reaped however close ends,
        a cancel included, since ``_closed`` is set first and no later close would reap it."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        try:
            if proc is not None and proc.returncode is None and proc.stdin is not None:
                try:
                    proc.stdin.write(encode(Shutdown()))
                    await proc.stdin.drain()
                    proc.stdin.close()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except TimeoutError:
                    await self._kill()
        finally:
            await self._reap_group()
            for task in (self._stdout_task, self._stderr_task):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 — swallow (correct during teardown) but log so a crashed reader is traceable
                    logger.exception(
                        '[plurai] child reader task %r raised during teardown (swallowed); the reader may have crashed before exit',
                        getattr(task, 'get_name', lambda: '<task>')(),
                    )

    async def _kill(self) -> None:
        """SIGTERM with a 3s grace window, then SIGKILL to the child's process group (to the child alone without one).

        ``proc.wait()`` returns only once the child's pipes close as well, and a grandchild holding the inherited stderr
        keeps them open after the child died. So the hard stop ends the group, and the last wait is bounded."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
            return
        except TimeoutError:
            pass
        await self._reap_group(hard=True)
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except TimeoutError:
            logger.warning('[plurai] child %s still holds its pipes open 3s after SIGKILL: a process outside its group keeps them', proc.pid)

    async def _reap_group(self, *, hard: bool = False) -> None:
        """End what is left in the child's process group: SIGTERM, a 3s grace, SIGKILL (``hard``: SIGKILL at once), so a
        harness-spawned server or bridge never outlives the child that started it. Runs once; a group that is already
        empty costs one failed ``killpg``. A grandchild that started its own session left the group and is not reached."""
        pgid, self._pgid = self._pgid, None
        if pgid is None:
            return
        for sig in (signal.SIGKILL,) if hard else (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                return
            for _ in range(30):
                await asyncio.sleep(0.1)
                try:
                    os.killpg(pgid, 0)
                except (ProcessLookupError, PermissionError):
                    return
        logger.warning('[plurai] process group %s of the closed child still has members after SIGKILL', pgid)

    # ----- protocol methods --------------------------------------------------

    async def init(
        self,
        *,
        thread_id: str,
        mock_service_url: str,
        mock_mode: MockMode = 'connect',
        initial_world_state: dict[str, Any],
        initial_agent_context: dict[str, Any],
        scenario_metadata: dict[str, Any],
        resume_seed: dict[str, Any] | None = None,
        entry_witness: str | None = None,
    ) -> dict[str, Any] | None:
        """Send Init; await InitOk (raises OrchestratorAdapterError on InitError).

        ``resume_seed`` defaults to None so any caller — the runner, and every test that
        drives a child directly — can omit it: a fresh scenario is exactly the no-seed
        case, not a special one.

        ``entry_witness`` is the declared agent entry (``module:Qual.name``) to witness; each
        reply then carries its ``{state, reason}``. ``None`` arms nothing and sends nothing.

        Returns ``seed_state``'s own fidelity report (``None`` for full fidelity,
        or whatever the harness returned to describe a reduced restore).
        """
        self._thread_id = thread_id
        msg = Init(
            thread_id=thread_id,
            mock_service_url=mock_service_url,
            mock_mode=mock_mode,
            initial_world_state=initial_world_state,
            initial_agent_context=initial_agent_context,
            scenario_metadata=scenario_metadata,
            resume_seed=resume_seed,
            entry_witness=entry_witness,
        )
        response = await self._send_and_wait(msg, timeout=self.agent_turn_timeout)
        if isinstance(response, InitOk):
            hook = EGRESS_GUARD_UNCONFIRMED
            if hook is not None and response.egress_guard_armed is not True and (refusal := await asyncio.to_thread(hook)) is not None:
                self._raise_for_error(InitError(exc_class='plurai.egress_guard.NotArmed', message=refusal), turn_idx=0)
            self._child_applies_user_actions = response.user_actions is True
            self.applies_channel_inputs = response.channel_inputs is True
            self.post_seed_world_state = response.post_seed_world_state
            return response.resume_fidelity
        if isinstance(response, InitError):
            self._raise_for_error(response, turn_idx=0)
        raise RuntimeError(f'unexpected child response to init: {type(response).__name__}')

    async def user_turn(
        self,
        turn_idx: int,
        user_message: str | list[dict[str, Any]] | None,
        *,
        interrupt_choice: str | None = None,
        user_action: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Send UserTurn; await AgentResponse (raises adapter / generic errors).

        Returns the full envelope so the caller can read ``full_messages``
        (intermediate ToolMessages + tool-call AIMessages preserved for
        transcript / post-hoc auditing) and ``is_interrupted`` /
        ``interrupt_context`` (HITL signal for the user-simulator). A caller
        that wants only the final user-visible string reads
        ``response.ai_text``.

        ``user_message=None`` is the EMPTY OPENER — the agent speaks first and this turn carries
        no user words. It is SENT only to a harness whose ``plurai.config.yaml`` declares
        ``agent.io.may_open: true`` (:mod:`plurai.agent_io`); to any other harness this refuses
        here, before the envelope is written, because ``invoke_agent({'messages': []})`` is a
        CRASH at a harness that indexes the last message, and a crashed child takes the whole
        conversation with it. A content-part LIST is sent as-is — the child owns
        that projection, against the same declaration.

        ``interrupt_choice`` is the user's typed decision on the interrupt this turn answers
        (``None`` on every other turn, and for a caller that never answers one).

        ``user_action`` is the act the simulated user took on the previous turn
        (``UserTurnResponse.user_action``), which the child applies before the agent runs. A child
        that did not say ``InitOk.user_actions`` is never sent one; the turn then reports the act
        ``unexecutable`` itself, so the server is told rather than left to guess.
        """
        if user_message is None and not self._declared_io().may_open:
            raise OrchestratorAdapterError(
                f'refusing to drive turn {turn_idx + 1} with an EMPTY OPENER on thread {self._thread_id}: '
                f'{self.simulate_path} does not declare `agent.io: {{may_open: true}}` in its plurai.config.yaml. '
                f'An agent that opens the conversation is called with `{{"messages": []}}` on turn 0, which a harness that '
                f'never declared it would raise on. Declare it (and make invoke_agent greet on an empty message list), or '
                f'let the scenario set supply the first turn.',
                thread_id=None if self._thread_id == _UNINITIALIZED_THREAD_ID else self._thread_id,
            )
        sent_action = user_action if self._child_applies_user_actions else None
        unsent = None if user_action is None or sent_action is not None else _UNEXECUTABLE_BY_AN_OLDER_CHILD | {'turn_idx': user_action.get('turn_idx')}
        msg = UserTurn(turn_idx=turn_idx, user_message=user_message, interrupt_choice=interrupt_choice, user_action=sent_action)
        response = await self._send_and_wait(msg, timeout=self.agent_turn_timeout)
        reply: AgentResponse | None = None
        if isinstance(response, AgentResponse):
            reply = response
            if isinstance(response.agent_state, dict) and response.agent_state.get(IPC_TRIMMED_KEY) is True:
                # Every turn this child answers comes through here, the closing turn included.
                self._ipc_trimmed_turns.append({'turn_idx': turn_idx})
        elif isinstance(response, AgentError):
            reply = self._scoreable_empty_turn(response, turn_idx=turn_idx)
            if reply is None:
                self.raised_turn_user_action_result = unsent or response.user_action_result
                self._raise_for_error(response, turn_idx=turn_idx)
        if reply is None:
            raise RuntimeError(f'unexpected child response to user_turn: {type(response).__name__}')
        self._turns_driven += 1
        # Both scored shapes — a reply, or a turn this parent emptied — report the act the same way.
        return reply if unsent is None else replace(reply, user_action_result=unsent)

    def _declared_io(self) -> AgentIO:
        """The ``agent.io`` declared beside this harness, resolved at most once.

        LAZILY, and only from the one caller that needs it: a ``str`` turn asks nothing of the
        declaration (the rule, once, on :class:`plurai._protocol.UserTurn`), so a run driven
        with strings alone never reads a config file here at all. The empty opener does — and an
        agent-first scenario sends one on turn 0 of a scored run — which is what makes this a
        deferral rather than a path nothing reaches. The child resolves the SAME declaration for
        itself, from the same two rungs, which is why neither side needs a new ``Init`` field to
        agree with the other.
        """
        if self._agent_io is None:
            self._agent_io = declared_agent_io(self.simulate_path) or DEFAULT_AGENT_IO
        return self._agent_io

    @property
    def degraded_turns(self) -> list[dict[str, Any]]:
        """The turns this child could not answer, recorded as ``{turn_idx, subkind}``.

        The RECORD half of the degradation, and the half a log line cannot be: an empty turn
        that only ever reached a warning is invisible to ``transcript.json``, to
        ``run_summary.json`` and to every consumer downstream of them, so a dialog the RUNNER
        emptied is indistinguishable from an agent that chose to say nothing. Read by the
        orchestrator onto :class:`~plurai.orchestrator.ThreadResult` and persisted from there.
        """
        return list(self._degraded_turns)

    @property
    def ipc_trimmed_turns(self) -> list[dict[str, Any]]:
        """The turns whose reply the child TRIMMED to fit one IPC line, ``{turn_idx}`` each (closing turn included).

        The reply's ``ai_text`` and ``agent_actions`` are the agent's own, but its message chain and the bulk of its
        agent state never crossed the pipe, so a judge reading that turn's tool trail reads less than the agent did.
        Read by the orchestrator onto :class:`~plurai.orchestrator.ThreadResult`; the runner folds it into
        ``measurement_clean`` and an ``ipc.reply_trimmed`` finding.
        """
        return list(self._ipc_trimmed_turns)

    @property
    def turns_driven(self) -> int:
        """How many turns this child ANSWERED — with a reply, or with an emptied one.

        The DENOMINATOR: without it "3 degraded turns" cannot be told from "3 degraded turns out
        of 3", and only the second is the case where nothing about the agent was measured at all.

        Counted at the two RETURNS rather than at the send, so a turn that raised is not in it.
        The CLOSING turn sits under a best-effort ``except`` in the orchestrator, so a count of
        sends would read one more than the turns answered whenever it failed, and an
        all-degraded conversation would not trip the escalation. A turn that raised produced no reply for a judge to score, so it is not part
        of "how much of this dialog did the agent actually say".
        """
        return self._turns_driven

    def _scoreable_empty_turn(self, err: AgentError, *, turn_idx: int) -> AgentResponse | None:
        """An EMPTY assistant turn for an adapter condition that must be scored, else ``None``.

        Only the turn path has this door. An adapter error is either something the customer must
        fix in ``simulate.py`` (fatal — see :data:`~plurai._protocol.FATAL_ADAPTER_SUBKINDS`) or
        something the AGENT did that the evals exist to catch: it produced no reply, or a
        condition this build cannot name. The second class is a verdict, and turning it into a
        raise costs the scenario its whole measurement: one silent agent turn would discard the signal of every
        judge for that scenario.

        The empty ``ai_text`` is the point: it rides /user-turn and /finalize like any other
        reply, so ``response_not_empty`` and every judge downstream of it grade the silence
        instead of never running. ``init`` deliberately does NOT route here — there is no
        conversation to score before the agent has been constructed.

        Every degradation is RECORDED as well as logged (see :attr:`degraded_turns`). A warning
        alone would trade one silent failure for another: the run's
        artifacts would say the agent went quiet for twelve turns when the truth is that a
        mis-wired ``simulate.py`` never handed its reply over, and the judges' 0/5 would be
        published under the agent's name.
        """
        if err.kind != 'adapter' or not adapter_error_is_scoreable(err.subkind):
            return None
        self._degraded_turns.append({'turn_idx': turn_idx, 'subkind': err.subkind})
        logger.warning(
            '[plurai] scoring an EMPTY agent turn on thread %s (turn %d): %s. Underlying: %s: %s',
            self._thread_id,
            turn_idx + 1,
            render_adapter_error_message(err.subkind or '', thread_id=self._thread_id, turn_idx=turn_idx),
            err.exc_class,
            err.message,
        )
        # LABELLED, not left blank. This is the one place an empty reply is manufactured by the
        # PARENT rather than produced by the agent, so it is the producer `REPLY_SOURCE_EMPTY`
        # exists for — and an unlabelled turn is counted as "this producer did not classify",
        # which would file a known-empty turn beside an older child's silence.
        #
        # The reply is empty; the TRAIL is not thrown away with it. The child ships its
        # result on the error envelope, so the chain and the agent's structured state survive
        # into transcript.json and into ``agent_state`` — the difference between "the runner
        # emptied this turn and here is what the agent did" and a turn that reads, on every
        # artifact, as if the agent had done nothing at all.
        #
        # ``agent_actions`` is deliberately left UNFILLED, which is ``None``: this is the one
        # path on which the runner KNOWS it failed to read the turn, so claiming ``[]`` — "I can
        # observe this agent's actions and there were none" — would assert observability from
        # the one producer that has none (see ``AgentResponse``, and ``orchestrator._agent_actions_of``).
        return AgentResponse(
            ai_text='',
            reply_source=REPLY_SOURCE_EMPTY,
            full_messages=list(err.full_messages),
            agent_state=dict(err.agent_state),
            # What the simulated user can see after the turn the agent DID run, and how her act went.
            turn_world=err.turn_world,
            user_action_result=err.user_action_result,
            channel_results=err.channel_results,
        )

    async def finalize(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]] | None, bool]:
        """Send Finalize; await FinalizeResponse — returns (world_state, tool_events, unanswered_mock_calls, capture_retryable).

        ``tool_events`` is ``None`` when the child read no tool channel at all (it held no
        plurai session, so it took no snapshot). That is NOT the same answer as ``{}`` or
        ``{'events': []}``, which is a channel that was read and recorded nothing, and the
        caller must keep the two apart all the way to ``/finalize`` — a judge grading a tool
        trail fails an agent for the second and abstains on the first.

        ``world_state`` is ``None`` on exactly the same condition and for exactly the same
        reason: no session, so no ``get_full_state`` call, so no claim. ``{'state': {}}`` is a
        world that WAS read and holds nothing — a measurement a world-reading policy may fail an
        agent on — and collapsing the two would pass that policy vacuously on a store nobody
        snapshotted. See ``_protocol.FinalizeResponse`` for the one rule both channels obey.

        ``unanswered_mock_calls`` is every tool call ``/mock/execute`` did not answer during this
        session (``None`` = the child claims nothing); ``capture_retryable`` says a snapshot
        read failed in a way one more read can clear (a mock restart, a timeout). See ``_protocol.FinalizeResponse`` for both.
        """
        response = await self._send_and_wait(Finalize(), timeout=self.agent_turn_timeout)
        if isinstance(response, FinalizeResponse):
            self.unread_inputs = response.unread_inputs
            self.unread_parts = response.unread_parts
            self.world_values_coerced = response.world_values_coerced
            self.interception_changed = response.interception_changed
            return response.final_world_state, response.tool_events, response.unanswered_mock_calls, bool(response.capture_retryable)
        if isinstance(response, AgentError):
            # The event log held a fault this process never saw (a call the agent's spawned process made).
            self._raise_for_error(response, turn_idx=response.turn_idx)
        raise RuntimeError(f'unexpected child response to finalize: {type(response).__name__}')

    async def pipeline_invoke(
        self,
        inputs: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> PipelineResult:
        """Send PipelineInvoke; await PipelineResult (one-shot pipeline dispatch).

        ``timeout`` defaults to ``agent_turn_timeout`` so a caller may omit it,
        but callers should pass an explicit ``pipeline_invoke_timeout`` because
        one-shot pipelines have different wall-clock characteristics than
        dialog turns (typically longer per-call, no recursion budget).

        On ``PipelineError`` from the child, this method raises
        :class:`OrchestratorAdapterError` carrying the child's exc_class +
        message. Unlike ``user_turn``, we do NOT route through the
        recursion-error classifier — one-shot has no recursion budget;
        any error is just an error.
        """
        wall = timeout if timeout is not None else self.agent_turn_timeout
        response = await self._send_and_wait(
            PipelineInvoke(inputs=inputs),
            timeout=wall,
        )
        if isinstance(response, PipelineResult):
            return response
        if isinstance(response, PipelineError):
            if response.kind == 'mock_runtime_fault':
                raise MockRuntimeFaultRaised(
                    f'mock runtime fault during invoke_pipeline on thread {self._thread_id}: {response.exc_class}: {response.message}',
                    transient=response.transient is True,
                    cause=response.message,
                    route_cause=response.route_cause,
                    credit=credit_block_from_body(response.credit),
                    capacity=capacity_block_from_body(response.capacity),
                    unavailable=unavailable_block_from_record(response.unavailable),
                    child_exc_class=response.exc_class,
                    thread_id=self._thread_id,
                )
            if response.kind == PROVIDER_INFRA_KIND:
                raise _provider_infra(f'pipeline_invoke failed on thread {self._thread_id}', response, thread_id=self._thread_id)
            if response.kind == 'harness_fault':
                # The harness's own refusal, never an agent outcome: typed so error.json reads not_valid.
                raise HarnessFaultRaised(
                    f'harness fault during invoke_pipeline on thread {self._thread_id}: {response.exc_class}: {response.message}',
                    child_exc_class=response.exc_class,
                    child_message=response.message,
                    thread_id=self._thread_id,
                )
            raise OrchestratorAdapterError(
                f'pipeline_invoke failed on thread {self._thread_id}: '
                f'{response.exc_class}: {response.message}' + (' [timed out]' if response.timed_out else ''),
                # Structured attrs so consumers (local_runner's
                # pipeline_error dict) can filter / route by the
                # CHILD-side exception class without parsing the message
                # text. Without these, every pipeline crash looks like
                # OrchestratorAdapterError in error.json regardless of
                # the actual underlying failure.
                child_exc_class=response.exc_class,
                child_message=response.message,
                timed_out=response.timed_out,
            )
        raise RuntimeError(f'unexpected child response to pipeline_invoke: {type(response).__name__}')

    # ----- internals ---------------------------------------------------------

    async def _send_and_wait(self, msg: Message, *, timeout: float) -> Message:
        """Write one envelope; await one response OR the child's exit, whichever first.

        Timeout raises :class:`OrchestratorAdapterError` (after killing the
        child); a child that died — noticed on the write here or on the wait
        below — raises :class:`HarnessChildExited` carrying the exit code, so
        the runner classifies it as a harness fault rather than an agent one.
        """
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError('ChildProcess not started')
        line = encode(msg)
        if len(line) > IPC_LINE_LIMIT_BYTES:
            # The child's reader would die on it and surface as an unexplained exit: refused here,
            # typed, before anything is written.
            raise HarnessIpcOverrun(
                f'the {msg.type} envelope is {len(line)} bytes, over the {IPC_LINE_LIMIT_BYTES}-byte IPC line limit, '
                f'so it cannot be sent to the child (an oversized initial world or agent context)',
                envelope_type=msg.type,
                direction='sent',
                limit=IPC_LINE_LIMIT_BYTES,
                thread_id=None if self._thread_id == _UNINITIALIZED_THREAD_ID else self._thread_id,
            )
        try:
            proc.stdin.write(line)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            # The child closed stdin / died before we finished writing — the SAME condition
            # ``_await_response_or_exit`` types below, reached by the other door. WHICH door
            # a death comes through is a race: the parent spends the /user-turn round trip
            # between envelopes, so a child reaped in that window is discovered by the next
            # WRITE rather than by ``proc.wait()``. Typed identically, or the same OOM kill
            # would reach ``error.json`` as an agent verdict half the time.
            await self._proc_wait_brief()
            await self._drain_stderr_briefly()
            raise HarnessChildExited(
                f'child subprocess closed stdin before {msg.type} could be sent '
                f'(returncode={proc.returncode}){_child_death_hint(proc.returncode)}: {exc}{self._stderr_tail_block()}',
                returncode=proc.returncode,
                envelope_type=msg.type,
                thread_id=None if self._thread_id == _UNINITIALIZED_THREAD_ID else self._thread_id,
            ) from exc

        return await self._await_response_or_exit(timeout=timeout, envelope_type=msg.type)

    async def _await_response_or_exit(self, *, timeout: float, envelope_type: str) -> Message:
        proc = self._proc
        assert proc is not None  # noqa: S101  # _send_and_wait already checked
        queue_get = asyncio.create_task(self._response_queue.get())
        proc_wait = asyncio.create_task(proc.wait())
        try:
            done, _ = await asyncio.wait(
                {queue_get, proc_wait},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not queue_get.done():
                queue_get.cancel()
            if not proc_wait.done():
                proc_wait.cancel()

        if queue_get in done:
            result = queue_get.result()
            if isinstance(result, OversizedIpcLine):
                raise HarnessIpcOverrun(
                    f'the child answered {envelope_type!r} with a line over the {result.limit}-byte IPC line limit; the envelope '
                    f'could not be read. A reply or world that large cannot cross the pipe.',
                    envelope_type=envelope_type,
                    direction='received',
                    limit=result.limit,
                    thread_id=None if self._thread_id == _UNINITIALIZED_THREAD_ID else self._thread_id,
                )
            if isinstance(result, _DecodeFailure):
                # Stray non-JSON output (typically a customer ``print(...)``
                # in simulate.py) — surface immediately with the offending
                # bytes so they can route prints to stderr.
                snippet = result.raw_line[:200].decode('utf-8', errors='replace').rstrip()
                raise RuntimeError(
                    f'Child wrote non-JSON to stdout while awaiting response '
                    f'to {envelope_type!r} (decode error: {result.error}). '
                    f'This usually means simulate.py contains a `print(...)` '
                    f'statement — route prints to stderr via '
                    f'`print(..., file=sys.stderr)`. Offending line: {snippet!r}'
                )
            return result
        if proc_wait in done:
            await self._drain_stderr_briefly()
            # TYPED, because an untyped raise here is indistinguishable from the agent
            # crashing: the runner stamps ``kind`` off the raising class, and without one a
            # reaped child reaches the report and the suite critic as an agent verdict.
            raise HarnessChildExited(
                f'child subprocess exited with code {proc.returncode} before responding to '
                f'{envelope_type}{_child_death_hint(proc.returncode)}{self._stderr_tail_block()}',
                returncode=proc.returncode,
                envelope_type=envelope_type,
                thread_id=None if self._thread_id == _UNINITIALIZED_THREAD_ID else self._thread_id,
            )
        # Timeout — kill the child so it doesn't outlive the parent.
        await self._kill()
        # Scan recent stderr for the Gemini-3 thinking-streaming fingerprint
        # before raising. If the child was calling ChatGoogleGenerativeAI with
        # thinking enabled AND streaming (the langchain default), the SDK
        # emits ``thoughtSignature`` frames that some idle-watchdogs don't
        # recognise as activity — the call looks like a hang but the model
        # is still working. The customer's coding_llm config needs
        # ``model_kwargs={"disable_streaming": True}`` to fix it.
        # References: langchain-google#1382, openclaw#76071.
        hint = ''
        if _looks_like_gemini_streaming_hang(self._stderr_tail):
            hint = (
                ' Possible cause: ChatGoogleGenerativeAI with a Gemini-3 '
                'thinking model AND default streaming. The SDK emits '
                'thoughtSignature frames during the thinking phase that '
                "the idle-watchdog doesn't count as activity, so the call "
                'looks like a wall-clock hang. Try '
                "``model_kwargs={'disable_streaming': True}`` on your "
                'coding_llm config (langchain-google#1382, openclaw#76071).'
            )
        # Raised as the TYPED AgentTurnTimeout (a subclass, so an
        # ``except OrchestratorAdapterError`` catches it too), because the re-drive
        # classifier may only recognise the condition from the TYPE, never from this
        # message.
        #
        # ``envelope_type`` is carried, not just rendered into the text: this method
        # serves init / user_turn / finalize alike, and only a stalled user_turn is the
        # transient stall a re-drive can beat. The classifier needs to tell them apart
        # from a field. ``thread_id`` rides along for the runner's /fail + error.json;
        # the pre-init placeholder is passed as None so it can never outrank the real
        # id the orchestrator recorded at create.
        raise AgentTurnTimeout(
            f'agent turn exceeded {timeout}s on thread {self._thread_id} (envelope={envelope_type!r}); child was terminated.{hint}',
            envelope_type=envelope_type,
            budget_seconds=timeout,
            thread_id=None if self._thread_id == _UNINITIALIZED_THREAD_ID else self._thread_id,
        )

    def _raise_for_error(self, err: InitError | AgentError, *, turn_idx: int) -> NoReturn:
        if isinstance(err, InitError) and err.kind == 'mock_unreachable':
            # Connect-mode mock could not be reached/verified by
            # ``plurai.configure(...)`` in the child. This is an actionable
            # operational condition, NOT an opaque adapter bug — render a
            # targeted message that points at the mock URL/deploy state. The
            # child's underlying message (from plurai.configure) already carries
            # the specific remediation hint (refused/DNS/401/410/...), so we
            # preserve it verbatim and frame it.
            #
            # Raised as the TYPED MockUnreachableError (a subclass, so an
            # ``except OrchestratorAdapterError`` catches it too) and carrying
            # ``thread_id``: the runner needs the id to call /fail and to stamp
            # error.json, and the restart classifier needs the TYPE — it must never
            # match on this message text, which is prose.
            raise MockUnreachableError(
                f'Could not reach the deployed mock for thread {self._thread_id} in '
                f"mock_mode='connect'. Is your mock URL reachable and deployed? "
                f'If this agent has NO mockable tools, the posture is probably wrong rather than the URL: '
                f"'connect' is the default when nothing declares otherwise, so a workdir missing its "
                f'``.run_mode`` sidecar dials a mock that was never deployed. Re-run with '
                f'``--mock-mode mockless`` to confirm — that is the posture for an agent with '
                f'nothing to mock, and it is what the sidecar should have said. '
                f'Underlying error: {err.exc_class}: {err.message}',
                child_exc_class=err.exc_class,
                thread_id=self._thread_id,
                raised_at='init',
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        if err.kind == 'adapter' and err.subkind is not None:
            message = render_adapter_error_message(
                err.subkind,
                thread_id=self._thread_id,
                turn_idx=turn_idx,
            )
            cause = RuntimeError(f'{err.exc_class}: {err.message}')
            # Use the specialised wrap for graph_recursion so callers can
            # ``isinstance(e, GraphRecursionWrap)`` to programmatically
            # branch (e.g. auto-run the verifier on cause_class hints)
            # rather than substring-matching the message. The in-process
            # path in orchestrator.py raises GraphRecursionWrap too; the two
            # paths must stay symmetric, or a caller sees the specialised type
            # on one path and a bare adapter error on the other.
            if err.subkind == 'graph_recursion':
                raise GraphRecursionWrap(
                    message,
                    cause_class='unknown',
                    thread_id=self._thread_id,
                    turn_idx=turn_idx,
                ) from cause
            raise OrchestratorAdapterError(message) from cause
        if err.kind == 'reset_contract':
            # The harness's reset hook could not be called as the reset contract requires. Typed
            # beside its seed_state sibling below and for the same reason: error.json must carry
            # a kind the run report attributes to the TEMPLATE, and the two hooks are different
            # halves of simulate.py, so they get different words.
            raise HarnessResetContract(
                f'child reported {err.exc_class} during {err.type}: {err.message}', child_exc_class=err.exc_class, thread_id=self._thread_id
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        if err.kind == 'seed_state_contract':
            # The harness's seed_state hook could not be called as the seed contract requires.
            # Typed so error.json carries a kind the run report attributes to the TEMPLATE, not
            # the agent; the message keeps the child's own sentence, which names the file and
            # the signature to change it to.
            raise HarnessSeedStateContract(
                f'child reported {err.exc_class} during {err.type}: {err.message}', child_exc_class=err.exc_class, thread_id=self._thread_id
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        if err.kind == 'mock_runtime_fault':
            # The backend's mock runtime could not answer a mocked tool: not_valid like a harness fault,
            # but typed apart, with the runtime's cause and whether one more try could answer.
            where = 'scenario setup (reset / seed_state)' if isinstance(err, InitError) else f'turn {turn_idx}'
            raise MockRuntimeFaultRaised(
                f'mock runtime fault during {where}: {err.exc_class}: {err.message}',
                transient=err.transient is True,
                cause=err.message,
                route_cause=err.route_cause,
                credit=credit_block_from_body(err.credit),
                capacity=capacity_block_from_body(err.capacity),
                unavailable=unavailable_block_from_record(err.unavailable),
                child_exc_class=err.exc_class,
                thread_id=self._thread_id,
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        if err.kind == 'harness_fault':
            # The harness raised ``plurai.HarnessFault``: a refusal it tagged as its own, so
            # error.json carries a ``harness_*`` kind the run report counts as not_valid. The
            # operator reads this line: it names the hook in words, never the envelope type.
            where = 'scenario setup (reset / seed_state)' if isinstance(err, InitError) else f'turn {turn_idx}'
            raise HarnessFaultRaised(
                f'harness fault during {where}: {err.exc_class}: {err.message}',
                child_exc_class=err.exc_class,
                child_message=err.message,
                thread_id=self._thread_id,
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        if isinstance(err, AgentError) and (raising := _AGENT_TURN_RAISES.get(err.kind)) is not None:
            raise raising(
                f'child reported {err.exc_class} during {err.type}: {err.message}',
                child_exc_class=err.exc_class,
                child_message=err.message,
                thread_id=self._thread_id,
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        if err.kind == PROVIDER_INFRA_KIND:
            raise _provider_infra(f'child reported {err.exc_class} during {err.type}: {err.message}', err, thread_id=self._thread_id)
        if isinstance(err, InitError) and err.kind == 'mock_init_state_fault':
            # The mock answered enough to be DIALED (unlike 'mock_unreachable', which fires
            # before that) but a state-shaped call during init/reset/seed (typically a /state
            # read) then failed — a symptom of the mock being slow/overloaded under backend
            # load, not a client/harness bug. Typed apart from mock_unreachable so the
            # rendered remedy never reuses its connect-posture guidance, which does not fit a
            # mock that was reachable enough to accept the init call.
            raise MockInitStateFault(
                f'The deployed mock accepted the connection but a state operation during scenario setup failed for '
                f'thread {self._thread_id}: {err.exc_class}: {err.message}. This usually means the mock is up but slow '
                f'or overloaded (a /state read timing out), not a client/harness bug — re-run, and if it repeats check '
                f'the mock deploy/host load.',
                child_exc_class=err.exc_class,
                thread_id=self._thread_id,
                raised_at='init',
            ) from RuntimeError(f'{err.exc_class}: {err.message}')
        failure = RuntimeError(f'child reported {err.exc_class} during {err.type}: {err.message}')
        if isinstance(err, InitError):
            # WHEN it fired, as the typed raisers above say it (``raised_at``): ``error.json`` carries it, and an
            # init failure on a trial that ran beside its siblings is re-run serially (``run_simulation``).
            failure.raised_at = 'init'  # type: ignore[attr-defined]
        raise failure

    async def _read_stdout(self) -> None:
        """Drain child stdout, decode JSONL, push envelopes onto the response queue.

        On decode failure (e.g., a customer ``print(...)`` leaked into stdout)
        we push a :class:`_DecodeFailure` sentinel onto the queue so the
        in-flight ``_send_and_wait`` raises with the actual cause instead of
        the agent-turn timeout that would otherwise fire a whole budget later.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while True:
            # An overrun pushes its sentinel, so the in-flight envelope raises HarnessIpcOverrun at once
            # instead of waiting out the whole turn budget for a reply that was sent and could not be read.
            line = await read_ipc_line(proc.stdout)
            if isinstance(line, OversizedIpcLine):
                await self._response_queue.put(line)
                continue
            if not line:
                return  # EOF — child closed stdout
            try:
                msg = decode(line)
            except ProtocolDecodeError as exc:
                logger.warning(
                    '[plurai] child stdout decode error: %s; line=%r',
                    exc,
                    line,
                )
                await self._response_queue.put(_DecodeFailure(raw_line=line, error=str(exc)))
                continue
            await self._response_queue.put(msg)

    async def _read_stderr(self) -> None:
        """Tee child stderr to the parent's logger.

        Lines matching :data:`_STDERR_WARN_RE` go to WARNING; everything else
        goes to DEBUG. Keeps customer logs from drowning the console under
        default verbosity.
        """
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        in_traceback = False
        warning_tail = 0
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            # a credential value of the agent's CLI (an ``env`` dump, a failed-auth trace) goes as its name, in
            # the log this tees to and in the tail a death message carries (error.json, /fail, re-drive)
            text = redact_cli_credentials(line.decode('utf-8', errors='replace').rstrip())
            self._stderr_tail.append(text)
            # Strip leading whitespace before pattern matching so that lines
            # like "  [plurai] HTTP 401 ..." (the mock-env HTTP client prints
            # with a two-space indent) still get promoted to WARNING.
            stripped = text.lstrip()
            if _TRACEBACK_HEADER_RE.match(stripped):
                in_traceback = True
            if _WARNING_HEADER_RE.match(stripped):
                # A rendered warning is the header plus at most its source line and the
                # tracemalloc hint. BOUNDED, unlike the traceback flag, because nothing
                # terminates a warning the way an exception summary closes a trace — an
                # unbounded tail promotes whatever indented block happens to follow, and a
                # logging StreamHandler writing multi-line records to this same stderr is
                # exactly that.
                warning_tail = 2
            elif warning_tail and stripped == text:
                # No leading whitespace ⇒ not a continuation of the warning.
                warning_tail = 0
            elif warning_tail:
                warning_tail -= 1
            promote = in_traceback or warning_tail > 0 or bool(_STDERR_WARN_RE.match(stripped))
            if in_traceback and _EXCEPTION_SUMMARY_RE.match(stripped):
                # Summary line closes the trace. A chained "During handling
                # of..." block starts a fresh Traceback header that will
                # flip the flag back on, so we don't need extra bookkeeping.
                in_traceback = False
            if promote:
                logger.warning('[plurai.child] %s', text)
            else:
                logger.debug('[plurai.child] %s', text)

    async def _drain_stderr_briefly(self) -> None:
        """Give the stderr task ~200ms to flush any final lines before raising."""
        if self._stderr_task is None or self._stderr_task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=0.2)
        except TimeoutError:
            pass

    def _stderr_tail_block(self) -> str:
        """The child's last stderr lines, rendered for a death message (empty when it said nothing).

        Shared by both child-death raises: a child reaped by the OS writes nothing, so the block
        is empty there and the returncode's sign carries the classification instead.

        The message goes on to ``error.json`` (an upload member), the ``/fail`` body and the
        re-drive record, and a traceback's frames spell paths through the home: scrubbed here,
        with the harness's repo (the root above the harness) allowed. Credential values were
        redacted as each line was read (:meth:`_read_stderr`).
        """
        tail = '\n'.join(self._stderr_tail).rstrip()
        if not tail:
            return ''
        tail = scrub_parent_record(tail, repo=Path(os.path.normpath(self.simulate_path.absolute())).parent.parent)
        return f'\n--- last child stderr ---\n{tail}\n--- end stderr ---'

    async def _proc_wait_brief(self) -> None:
        """Wait up to 0.5s for the child to finish exiting (so returncode populates)."""
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=0.5)
        except TimeoutError:
            pass


_GEMINI_STREAMING_HANG_MARKERS = (
    # langchain_google_genai's logger banners + the SDK's per-call AFC log.
    # Either one indicates we're calling Gemini; combined with thinking-
    # related markers this fingerprints the streaming-thoughtSignature path.
    'ChatGoogleGenerativeAI',
    'langchain_google_genai',
    'google.genai',
    'AFC is enabled',
)

_GEMINI_THINKING_MARKERS = (
    # Identifiers we expect when the customer enabled thinking on a Gemini-3
    # model. We're permissive (substring match anywhere in the tail) because
    # different langchain/genai versions emit slightly different log shapes.
    'thinking_level',
    'thinking_budget',
    'gemini-3',
    'thoughtSignature',
    'thought_signature',
)


def _looks_like_gemini_streaming_hang(stderr_tail) -> bool:  # noqa: ANN001
    """True iff recent child stderr suggests Gemini-3 + thinking + streaming.

    A heuristic over the rolling stderr buffer — requires BOTH a Gemini-SDK
    marker AND a thinking-model marker so we don't false-positive on
    customers using GPT-4 or non-thinking Gemini variants. False negatives
    are acceptable: when in doubt, the hint just doesn't appear and the
    generic timeout message still surfaces.

    Accepts any iterable of strings so callers can pass a ``deque``,
    ``list``, or test fixture without coercion.
    """
    try:
        joined = '\n'.join(line for line in stderr_tail if isinstance(line, str))
    except TypeError:
        return False
    has_gemini = any(m in joined for m in _GEMINI_STREAMING_HANG_MARKERS)
    has_thinking = any(m in joined for m in _GEMINI_THINKING_MARKERS)
    return has_gemini and has_thinking


def _child_death_hint(returncode: int | None) -> str:
    """The remedy clause for a child that died, keyed on the SIGN of its returncode.

    A negative code is ``-signal`` (POSIX): the child did not choose to exit, something
    reaped it. SIGKILL is the one an operator cannot see any other way — nothing is
    written to the child's stderr, so the tail block is empty and the raise carries only a
    number. Under a parallel run that number is almost always the OS OOM reaper picking
    the largest child, which is why the hint names the knob (``--concurrency 1``) rather
    than only the signal.

    A POSITIVE code is the child exiting on its own (an ImportError in the customer's
    venv, a bad module path) and its stderr tail already says why — so no hint, rather
    than a guess that would misdirect.
    """
    if returncode is None or returncode >= 0:
        return ''
    signal_number = -returncode
    reaped = f' (killed by signal {signal_number}'
    if signal_number == 9:
        return f'{reaped}/SIGKILL — under a parallel run this is usually the OS out-of-memory reaper; re-run with --concurrency 1 to confirm)'
    return f'{reaped} — the child was killed rather than exiting on its own, so its stderr may say nothing)'


__all__ = ['ChildProcess', 'OrchestratorAdapterError']
