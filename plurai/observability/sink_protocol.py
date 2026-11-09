"""Observability-sink CONTRACT (two role Protocols) + NoOpSink.

The orchestrator drives ONE observability sink per run (single active sink — see
:mod:`plurai.observability`). Rather than branch on a sink name string at every
call site, the orchestrator holds ONE sink instance and calls its methods
directly; when no sink is configured it holds a :class:`NoOpSink` whose every
method is a no-op, so the call sites never branch on ``None`` / disabled.

Two roles, two Protocols
------------------------
A sink is used in TWO structurally distinct ways, so the contract is split into
two ``@runtime_checkable`` :class:`typing.Protocol` surfaces — a consumer depends
on the SHAPE it actually calls, not on a class hierarchy:

* :class:`InstrumentationSink` — the PER-TURN TRACING role the Orchestrator drives
  for every scenario (``prepare_run`` / ``record_transcript`` / ``post_feedback`` /
  async ``drain_run`` / async ``close``). The orchestrator seams annotate against THIS
  role. Each sink still drains eagerly inside its own seams (Arize force_flushes at the
  end of ``record_transcript``; Langfuse/Braintrust drain in ``post_feedback``), but the
  orchestrator ALSO calls the idempotent per-scenario ``drain_run`` on its scenario-scope
  exit — so the drain is guaranteed even on the crash path where ``post_feedback`` never
  runs (e.g. an eagerly-opened Braintrust root span is still ended).
* :class:`DatasetSink` — the GENERATION-TIME dataset-management role
  (``sync_scenarios``) the dataset-upload path drives once per scenario-generation
  pass. The upload path annotates against THIS role.

Plus FOUR OPTIONAL, additive roles:

* :class:`ExperimentLifecycleSink` — the SCENARIO-SET lifecycle role
  (``begin_experiment`` / ``end_experiment``) the runner drives ONCE around the whole
  ``asyncio.gather`` over scenarios, so a sink can model "the entire set run = one
  Experiment / Dataset-Run spanning N scenarios". This role is DELIBERATELY NOT folded
  into :class:`ObservabilitySink` and is NOT part of its conformance: it is purely
  additive, so a sink that does not model set-level experiments is simply not an
  :class:`ExperimentLifecycleSink` and the runner skips it (the runner ``isinstance``-
  gates the dispatch and wraps it best-effort). See that class's docstring for WHY a
  separate protocol rather than a default on the existing roles.

* :class:`ExperimentRunnerSink` — the EXPERIMENT-RUNNER role
  (``run_experiment_set``) the strategy selector drives ONCE from the main thread (no
  running event loop) to hand iteration of the full scenario set to the sink. The sink
  uses Arize's ``AsyncExecutor``-backed ``experiments.run`` internally; the method is
  SYNC and returns a failure count. Like :class:`ExperimentLifecycleSink`, this is NOT
  a base of :class:`ObservabilitySink` — it is purely additive. See that class's
  docstring for the full rationale.

* :class:`DatasetUrlSink` — the PLATFORM-LINK role (``dataset_url``) the runner drives
  ONCE at the end of a run to print the human URL of the provider dataset the run
  recorded against (the "where do I look" line). Same
  separate-protocol reasoning as the two roles above: purely additive, best-effort
  (never raises; unresolvable ⇒ ``None``), ``isinstance``-gated by the runner.

* :class:`RunLinksSink` — the RUN-LINK role (``run_links``) the runner drives ONCE per run, at
  the same point as :class:`DatasetUrlSink`, to print the links to the run's experiment and to
  each scenario's trace. Same separate-protocol reasoning and the same best-effort contract.

:class:`ObservabilitySink` is the combined alias
(``ObservabilitySink(InstrumentationSink, DatasetSink, Protocol)``) for the few
sites that genuinely need BOTH roles — most importantly the
:func:`plurai.observability.create_observability_sink` factory's return type
(the orchestrator holds one instance used for tracing AND, via the dataset path,
for upload), and the conformance check that every registered sink class is a
full ObservabilitySink.

No shared base class (NOT an ABC)
---------------------------------
The contract is NEITHER an ``abc.ABC`` with ``@abstractmethod`` declarations NOR
a shared concrete base — each sink is a STANDALONE class that satisfies the
Protocols structurally. Conformance is enforced two ways that an ABC could not:
STATICALLY (pyright fails if a registered sink class misses a protocol method —
see the registry typing in :mod:`plurai.observability.factory`)
and at RUNTIME (the factory ``isinstance``-asserts the constructed sink against
the combined ``@runtime_checkable`` protocol; see the note on
:func:`plurai.observability.create_observability_sink`).

The four concrete sinks (LangSmith / Langfuse / Arize / Braintrust) are each
self-contained: they declare ``name``, implement the role methods + an async
``close`` lifecycle drain, and lean on the module-level
:func:`plurai.observability.errors.swallow` best-effort guard rather than a base
method. :class:`NoOpSink` is likewise standalone and stubs every method, so it
satisfies BOTH protocols. Each concrete sink holds ALL of its state as INSTANCE
attributes (the lazily-created client, availability flags, braintrust's
thread-span slots, arize's process-init flag); there are no module-level mutable
globals.

The role methods, by contract:

* ``prepare_run(...) -> RunContext`` — open sink-side resources for one
  scenario and return a SINK-SPECIFIC run context (NOT flattened to one shape).
  LangSmith's context carries ``example_id`` (the child-handshake invariant);
  Langfuse / Braintrust carry a parent-side handle; Arize returns a marker.
* ``post_feedback(thread, finalize, run_ctx, ...)`` — post per-policy feedback +
  the rollup metrics. Consumes the run-local ``final_world_state`` +
  ``tool_events`` PARAMS (never ``ThreadResult.tool_events`` — empty in-process).
* ``record_transcript(run_ctx, *, thread_id, scenario_name, turns, ...)`` — emit
  the uniform 3-level span tree (scenario -> turn -> {agent, client}) from
  orchestrator-owned ``turns`` (a list of :class:`TranscriptTurn`); the scenario
  anchor is also the single eval-attachment target.
* ``sync_scenarios(...)`` — the generation-time dataset-upload entry.
* async ``drain_run(run_ctx)`` — the per-scenario teardown drain (best-effort,
  IDEMPOTENT). The orchestrator's scenario scope calls it on exit (normal AND
  exception), guaranteeing a drain even when ``post_feedback`` is skipped on a crash;
  because each sink also drains in its own seam on the happy path, ``drain_run`` must be
  safe to fire twice (Braintrust's ``slot.ended`` guard, Arize's repeat-safe
  ``force_flush``, etc.).
* async ``close()`` — the once-per-process teardown drain (best-effort), for state that
  outlives a single scenario (Braintrust's bg logger). Distinct from ``drain_run``, which
  is per-scenario.

Every method is best-effort: a sink failure can NEVER break a simulation run.
The module-level :func:`plurai.observability.errors.swallow` (logged best-effort
wrapper) is what the concrete sinks lean on; the rollup metric helpers are
imported from the package namespace by the concrete sinks as needed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

from plurai.observability.dataset_types import ProviderSyncResult
from plurai.observability.model_calls import ModelCall
from plurai.observability.tool_call_recovery import RecoveredToolCall
from plurai.orchestrator.models import CreateThreadResponse, FinalizeResponse, ScenarioUploadEntry


@runtime_checkable
class ScenarioResult(Protocol):
    """Structural protocol for the value returned by the ``run_one`` callable passed to
    :meth:`ExperimentRunnerSink.run_experiment_set`.

    Captures the fields that :func:`plurai.observability.arize._build_experiment_output`
    and :func:`plurai.observability.arize._coerce_output` actually read.
    :class:`~plurai.orchestrator.orchestrator.ThreadResult` satisfies this structurally.

    ``sink_protocol`` deliberately does NOT import ``ThreadResult`` directly — that would
    create a sink_protocol → orchestrator.orchestrator import cycle.  The Protocol serves
    as the structural seam instead.
    """

    # Read-only properties (not plain attributes): a plain attribute in a Protocol
    # demands settability, which the frozen ThreadResult dataclass can't provide.
    @property
    def thread_id(self) -> str: ...

    @property
    def finalize(self) -> FinalizeResponse: ...

    # ``None`` when NO tool channel was read on this scenario — a different answer from ``{}``
    # / ``{'events': []}``, which is a channel that was read and recorded nothing. The reader
    # (``arize._build_experiment_output``) already guards it; the type has to admit it or
    # ``ThreadResult`` stops satisfying this protocol.
    @property
    def tool_events(self) -> dict[str, Any] | None: ...

    @property
    def chatbot_messages(self) -> list[str]: ...

    @property
    def chatbot_full_messages(self) -> list[dict[str, Any]]: ...

    @property
    def unanswered_mock_calls(self) -> list[dict[str, Any]] | None: ...


class TranscriptTurn(TypedDict):
    """One outer-conversation turn, captured parent-side from orchestrator-owned strings.

    The orchestrator accumulates one of these per loop iteration; the list is then
    replayed by :meth:`InstrumentationSink.record_transcript` into a uniform span
    tree (scenario -> turn -> {agent -> tool*, client}) that looks identical in every
    sink, with no dependency on the agent's framework. ``agent_tool_calls`` are the
    tool calls the agent made DURING this turn, rendered as child nodes under the
    ``agent`` node so the "turn -> assistant -> tool calls (args + result)"
    tree materializes; a turn with no tool calls keeps the plain agent/client pair.

    For turn ``i`` (``input_user_message`` is the message the agent saw):

    ``user_messages[0] == initial_message`` does NOT hold in general: a scenario whose bench
    declares ``agent.io: {may_open: true}`` has no opener at all, so ``user_messages`` starts
    empty and its ``[0]``, when one eventually exists, is the user's REPLY to the agent rather
    than the opener. By ROLE: ``user_messages`` holds one entry per USER turn that happened, in
    order, and ``initial_message`` is the first of them ONLY when the scenario had a user first
    line. An agent-first turn 0 carries ``input_user_message: None`` — nothing was fed to the agent, which
    is a fact about the turn and not a missing value.

    * ``input_user_message`` — the user message fed to the agent this turn, or ``None`` when the
      agent opened and there was none.
    * ``agent_response`` — the agent's reply (``ai_text``).
    * ``output_user_message`` — the next user message the backend generated (or the
      terminating one).
    * ``agent_tool_calls`` — the tool calls this turn issued, each carrying ``tool`` /
      ``args`` / ``response`` / ``status``. Per-turn granularity pairs each call's args
      with its result within the one turn that issued it. Filled from THIS turn's
      serialized message chain (:func:`derive_tool_calls_from_messages`) where the agent's
      tool structure reaches the transcript, and otherwise grafted post-finalize by
      :func:`attribute_tool_events_to_turns` from the turn-stamped ``tool_events`` —
      which covers both the out-of-band ``code`` / ``cli_http`` channels and a
      self-looping agent's ``agent_state`` trail, neither of which the message chain
      can see. A ``NotRequired`` key: the orchestrator always sets it (possibly ``[]``),
      so it is present in production; older/offline turn fixtures may omit it and sinks
      read it as ``turn.get('agent_tool_calls') or []``.
    * ``agent_model_calls`` — the model calls this turn made, one per assistant message in the
      turn's chain (:func:`~plurai.observability.model_calls.derive_model_calls`), each with the
      messages it was shown, its text, the tool calls it requested, and its model and token
      usage where the framework reports them. A sink that models generations (Langfuse) or
      ``llm`` runs (LangSmith) renders one per call beneath the ``agent`` node. Empty when the
      chain carries no assistant message (a harness that returns only a string). A
      ``NotRequired`` key: the orchestrator always sets it, older/offline turn fixtures may
      omit it, and sinks read it as ``turn.get('agent_model_calls') or []``.
    * ``closing_turn`` — ``True`` ONLY on the final CLOSING-exchange entry (the
      delivered reply to the terminating user message). That exchange is a
      real turn to observability (the sinks span it like any other), but it is NOT a
      ``/user-turn`` cycle: no next user message exists (``output_user_message`` is
      empty) and ``turn_count`` carries the last cycle's server count. A
      ``NotRequired`` key so ordinary cycle turns simply omit it; consumers read it
      as ``turn.get('closing_turn', False)``.
    """

    turn_idx: int
    #: NULLABLE. ``None`` is an agent-first turn — nothing was fed to the agent —
    #: and it is not ``''``, which is a user turn that carried no text. A sink that INDEXES users
    #: by this field must key on the ROLE, not on the position (see the class docstring).
    #:
    #: The three JSON sinks (langfuse, langsmith, braintrust) pass the ``None`` straight through
    #: as a JSON null, which says exactly the right thing; nothing reads this field as
    #: ``… or ''``. The Arize sink omits the key rather than handing OTel a ``None`` it would
    #: drop while logging once per agent-first turn.
    #:
    #: The third shape is a LIST of content parts: a turn whose user side carried an attachment
    #: or a structured block.
    #:
    #: THE UNION IS THE CONTRACT, so each sink states what it does with the list rather than
    #: discovering it: the three JSON sinks carry it structured (it is a JSON value and says more
    #: than its text would), and arize PROJECTS it, because an OTel attribute value must be a
    #: primitive or a homogeneous sequence of primitives, and a list of parts is neither. A sink
    #: that renders this still reads the text through ``content_text``; a sink that
    #: INDEXES users by it keys on the ROLE, not the position (see the class docstring).
    input_user_message: str | list[dict[str, Any]] | None
    agent_response: str
    output_user_message: str
    should_terminate: bool
    turn_count: int
    agent_tool_calls: NotRequired[list[RecoveredToolCall]]
    agent_model_calls: NotRequired[list[ModelCall]]
    closing_turn: NotRequired[bool]
    #: WHERE ``agent_response`` came from (``plurai.child.REPLY_SOURCES``): the agent's own words,
    #: a tool result a graph ended on, a typed interrupt, an empty turn, or a suffix the
    #: classifier could not read. Labelling only — no verdict moves on it — but without it a raw
    #: tool payload and the agent's own answer are one indistinguishable string on every surface
    #: that reads this record. ``NotRequired``: turn fixtures written before it simply omit it,
    #: and an absent label is never read as ``assistant``.
    reply_source: NotRequired[str]


@runtime_checkable
class RunContext(Protocol):
    """The per-scenario run-context CONTRACT every sink's ``prepare_run`` returns.

    ``prepare_run`` returns a SINK-SPECIFIC run context (LangSmith carries
    ``example_id`` + ``anchor_run_id``; Arize an ``anchor_span_id`` + provider;
    Braintrust a span slot; Langfuse a client) — there is deliberately no flattened
    shape. The ONE thing every context must expose uniformly is ``enabled``: whether
    tracing is LIVE for this scenario (the sink was configured AND its per-scenario
    setup succeeded). ``record_transcript`` / ``post_feedback`` gate on it. Each sink's
    ``enabled`` returns EXACTLY the condition that sink already uses to decide whether
    to trace — exposing the existing decision, never changing what gets traced.

    ``@runtime_checkable`` so a context can be ``isinstance``-checked if needed.
    """

    @property
    def enabled(self) -> bool:
        """True when tracing is LIVE for this scenario (the sink's existing gate)."""
        ...


@runtime_checkable
class InstrumentationSink(Protocol):
    """The per-turn TRACING role: what the Orchestrator drives for each scenario.

    A consumer that only records traces (the Orchestrator's prepare / callback /
    feedback / drain seams) depends on THIS shape, not on the concrete class.
    ``@runtime_checkable`` so the factory can ``isinstance``-assert conformance at
    construction.
    """

    #: Registry name of the sink (identifies it in logs).
    name: str

    async def warm_once(self) -> None:
        """Perform the sink's ONE-TIME process init exactly once, on the event-loop thread.

        Called ONCE at the entry point AFTER the sink is built and BEFORE any
        ``asyncio.gather`` over scenarios. Because it runs single-threaded (no
        ``to_thread``) there is no race on the sink's process-global install (Arize's
        OTel tracer provider, Braintrust's process Logger). Each sink performs exactly
        its own one-time init here (Arize -> ``initialize``; Braintrust -> ``warm``;
        Langfuse / LangSmith build their lazy client if any, else no-op) and MUST be
        IDEMPOTENT — a second call is a cheap no-op. The lazy per-scenario guards remain
        as defense-in-depth (and are themselves lock-guarded), so a sink whose
        ``warm_once`` is skipped still self-inits, just without the single-thread
        guarantee. Best-effort: never raises.
        """
        ...

    async def prepare_run(
        self,
        *,
        agent_name: str,
        experiment_name: str,
        scenario_name: str,
        thread_id: str,
        scenario_folder: str | None = None,
        dataset_name: str | None = None,
        example_inputs: dict[str, Any] | None = None,
        agent_version: str,
    ) -> RunContext:
        """Open sink-side resources for one scenario; ALWAYS return a sink-specific context.

        ``scenario_name`` and ``scenario_folder`` are the scenario's TWO identities and are
        NOT interchangeable. The rule for an implementer: **label with ``scenario_name``,
        join with ``scenario_folder``**, and resolve the latter through
        :func:`plurai.observability.dataset_types.dataset_join_key` — which owns the
        distinction and the ``None`` fallback — rather than reading either field directly.

        prepare_run NEVER returns ``None``: a disabled / unconfigured sink (or a swallowed
        per-scenario setup failure) returns its CONCRETE context with ``enabled=False`` —
        the uniform disabled SENTINEL the Protocol's ``enabled`` was built for, so every
        downstream seam (and the orchestrator) gates on ``run_ctx.enabled`` and never on
        ``None``. The returned context is sink-specific but always conforms to
        :class:`RunContext` (exposes ``enabled``). ``agent_version`` is the operator's
        REQUIRED agent-build tag (CLI ``--agent-version``, defaulting to ``git describe`` /
        ``"unknown"`` — never ``None``) — a sink that stamps record metadata carries it
        onto its run context. Async: the blocking sink-SDK work runs
        inside the method via ``asyncio.to_thread`` so the orchestrator can ``await`` it
        directly (no orchestrator-side thread hop). Best-effort + self-guarding
        (``swallow``): a sink failure yields a disabled context and NEVER raises.
        """
        ...

    async def post_feedback(
        self,
        thread: CreateThreadResponse,
        finalize: FinalizeResponse,
        run_ctx: RunContext,
        *,
        final_world_state: dict[str, Any] | None = None,
        tool_events: dict[str, Any] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post per-policy feedback + the rollup metrics for one scenario.

        Async: the blocking sink-SDK work runs inside the method via
        ``asyncio.to_thread``. Best-effort + self-guarding (``swallow``): a sink
        failure returns ``None`` and NEVER raises.
        """
        ...

    async def record_transcript(
        self,
        run_ctx: RunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit the uniform 3-level span tree for one scenario (best-effort).

        Builds, from orchestrator-owned strings only, a scenario anchor with one
        child per turn and an ``agent`` + ``client`` span under each turn — the same
        shape in every sink, with no dependency on the client's framework. The
        scenario anchor is also the single eval-attachment target. Derives ALL state
        from ``run_ctx`` (never sink-instance or process-global mutable state), gates
        on ``run_ctx.enabled``, and routes through ``swallow`` so a failure can never
        break a run.

        Returns ``None``: the anchor it builds is NOT handed back — each sink stashes
        whatever ``post_feedback`` needs (the anchor span/run id, trace handle, span
        slot) onto ``run_ctx`` so the post-feedback seam re-targets evals from the
        per-scenario context, never from a return value.

        There is deliberately NO shared default: every sink must implement this
        so pyright fails a sink that omits it (conformance is the contract).

        Async: the blocking span-emit work runs inside the method via
        ``asyncio.to_thread`` (the Arize/Braintrust/Langfuse/LangSmith span SDKs
        are sync), so the orchestrator awaits it directly. Best-effort.
        """
        ...

    async def drain_run(self, run_ctx: RunContext) -> None:
        """Per-scenario teardown drain (best-effort, IDEMPOTENT, async).

        The orchestrator's scenario scope calls this on exit — on the normal path AND
        when the scenario raises — so the per-scenario drain is GUARANTEED even when
        ``post_feedback`` never runs (e.g. an eagerly-opened Braintrust root span is
        still ended + flushed). Derives all state from ``run_ctx`` (a disabled context —
        ``enabled=False`` — ⇒ no-op, the sink was disabled for this run). MUST be idempotent: each sink also drains in
        its own seam on the happy path, so ``drain_run`` may fire after that drain has already happened (the sinks guard with ``slot.ended`` / repeat-safe
        ``force_flush`` / a repeat-safe client ``flush`` that just drains an empty queue).
        Distinct from ``close``, which is the once-per-process teardown.

        Async: the blocking SDK drain runs inside the method via ``asyncio.to_thread`` so
        the event loop isn't blocked. Best-effort: never raises.
        """
        ...

    async def close(self) -> None:
        """Process-teardown drain for the sink (best-effort, async).

        Per-scenario draining is COUPLED into the sink's own seams (Arize
        force_flushes at the end of ``record_transcript``; Langfuse/Braintrust
        drain in ``post_feedback``; LangSmith flushes buffered submissions in
        ``drain_run``) — there is NO separate ``flush`` step. ``close`` is the
        once-per-process teardown drain; it ``await``s the blocking SDK shutdown via
        ``asyncio.to_thread`` so the event loop isn't blocked.
        """
        ...


@runtime_checkable
class DatasetSink(Protocol):
    """The generation-time DATASET-MANAGEMENT role: what the upload path drives.

    A consumer that only uploads a generated scenario batch (the dataset-upload
    path) depends on THIS shape. ``@runtime_checkable`` for the factory's
    construction-time conformance assert.
    """

    #: Registry name of the sink (identifies it in logs).
    name: str

    def sync_scenarios(
        self,
        *,
        dataset_name: str,
        dataset_description: str | None,
        scenarios: list[ScenarioUploadEntry],
        existing_metadata: dict[str, Any],
    ) -> ProviderSyncResult:
        """Upload a generated scenario batch into the sink's dataset (generation-time).

        Always returns a :class:`~plurai.observability.dataset_types.ProviderSyncResult`.
        A true no-op (no sink / SDK missing / empty batch) is represented as a result
        with ``ok=True, uploaded=0, failed=0`` — ``None`` is NEVER returned.
        See :class:`~plurai.observability.dataset_types.ProviderSyncResult` for the
        full success / partial / failure contract.
        """
        ...


@runtime_checkable
class ExperimentLifecycleSink(Protocol):
    """OPTIONAL scenario-SET lifecycle role: bracket the whole gather as ONE experiment.

    The runner (:func:`plurai.run_simulation._gather_run`) drives this ONCE per run around the entire
    ``asyncio.gather`` over scenarios (once per trial, each on its own sink, under ``--trials N``) — ``begin_experiment`` BEFORE the loop and
    ``end_experiment`` AFTER it — so a sink that models set-level analytics can open one
    Experiment / Dataset-Run that the N per-scenario traces attach to, then close it.

    Why a SEPARATE protocol (not a default on :class:`InstrumentationSink`)
    ----------------------------------------------------------------------
    The contract is enforced structurally via ``@runtime_checkable`` ``isinstance``
    (the factory's RUNTIME backstop). Adding a
    method to :class:`InstrumentationSink` / :class:`ObservabilitySink` — *even with a
    default body* — makes a sink that doesn't define it FAIL that ``isinstance`` check,
    because a runtime-checkable protocol's membership test looks only at attribute
    PRESENCE on the instance and a structurally-conforming sink does NOT inherit Protocol
    defaults. So a default on the existing roles would silently break every sink that
    hasn't yet implemented set-level experiments. Keeping this an INDEPENDENT protocol
    that is NOT a base of :class:`ObservabilitySink` makes the seam purely ADDITIVE: a
    sink opts in simply by defining both methods (then ``isinstance(sink,
    ExperimentLifecycleSink)`` is True and the runner drives it); a sink that doesn't is
    skipped, with its :class:`ObservabilitySink` conformance untouched.

    Best-effort: the runner wraps both calls so a failure here can NEVER abort the run.
    ``@runtime_checkable`` so the runner can ``isinstance``-dispatch the optional seam.
    """

    async def begin_experiment(self, *, dataset_name: str, run_name: str) -> None:
        """Open the set-level Experiment / Dataset-Run BEFORE the scenario gather (best-effort).

        ``dataset_name`` identifies the scenario set / resolved dataset the run spans;
        ``run_name`` is the experiment handle (the per-run label the N scenario traces
        roll up under). Called exactly once, before any ``prepare_run``, and — when a
        pre-flight dataset upload runs for this invocation — after it, so a sink that pins
        its experiment to a dataset version already sees every example. Best-effort: the runner guards the call, but
        a conforming sink should also self-guard and never raise.
        """
        ...

    async def end_experiment(self) -> None:
        """Close the set-level Experiment / Dataset-Run AFTER the gather completes (best-effort).

        Called exactly once, after every scenario has finished, while the sink is still
        open — before ``close``'s process drain. (The dataset upload does NOT run here: it
        runs ahead of the gather, since the dataset defines the experiment's inputs.)
        Best-effort: never raises.
        """
        ...


@runtime_checkable
class ExperimentRunnerSink(Protocol):
    """OPTIONAL experiment-runner role: the VENDOR drives the scenario-set as ONE experiment.

    The strategy selector (:func:`plurai.run_simulation.main`) drives this ONCE per
    invocation — ``run_experiment_set`` replaces the native ``asyncio.gather`` loop,
    handing iteration entirely to the sink's vendor SDK (Arize: ``experiments.run``;
    the shape fits any provider whose SDK owns iteration). The caller performs async
    setup (dataset upload, policy name fetch) BEFORE calling this method, then calls
    it SYNCHRONOUSLY from the main thread (no running event loop).

    Why a SEPARATE protocol (not a default on :class:`InstrumentationSink`)
    ----------------------------------------------------------------------
    Same reasoning as :class:`ExperimentLifecycleSink`: a ``@runtime_checkable`` protocol
    with no methods on :class:`ObservabilitySink` makes this seam PURELY ADDITIVE. A sink
    opts in simply by defining ``run_experiment_set``; every other sink — INCLUDING
    :class:`NoOpSink`, which deliberately does NOT stub this method — remains a plain
    ``ObservabilitySink``, so the selector's ``isinstance`` check alone picks the strategy.

    SYNC: ``run_experiment_set`` is a regular (non-async) method, and the caller must
    invoke it from the MAIN thread with NO running event loop. This is a vendor-imposed
    contract, not a style choice: arize's ``get_executor_on_sync_context`` falls back to
    a SERIAL ``SyncExecutor`` (no concurrency, no per-task timeout) off the main thread,
    and its sync wrapper raises ``TypeError`` on an async task — so offloading this call
    to a thread or nesting it in a running loop breaks it outright.

    ``@runtime_checkable`` so the strategy selector can ``isinstance``-dispatch.
    """

    def run_experiment_set(
        self,
        *,
        upload_metadata: Mapping[str, Any],
        default_dataset_name: str,
        run_name: str,
        scenario_ids: list[str],
        run_one: Callable[[str], Awaitable[ScenarioResult]],
        evaluator_names: list[str],
        concurrency: int,
        timeout: int,
    ) -> int:
        """Run the full scenario set as a single provider experiment (SYNC, main-thread only).

        ``upload_metadata`` is the provider-keyed mapping persisted by the dataset
        upload (the sink's own ``sync_scenarios`` ``metadata_updates``) — the caller
        round-trips it OPAQUELY; the sink resolves its own dataset handle from it,
        falling back to ``default_dataset_name`` when the mapping lacks one.
        ``run_name`` is the unique per-invocation experiment handle; ``scenario_ids``
        is the ordered list of scenario folder names to run. ``run_one`` is an async
        callable that runs ONE scenario by id and returns a ``ThreadResult``-compatible
        mapping. ``evaluator_names`` is the list of policy names fetched from the backend.
        ``concurrency`` and ``timeout`` bound the vendor executor.

        Returns the failure count: 0 = all passed; positive = that many scenarios failed.
        Best-effort: a conforming sink should self-guard and never raise.
        """
        ...


@runtime_checkable
class DatasetUrlSink(Protocol):
    """OPTIONAL platform-link role: resolve the human URL of the run's provider dataset.

    The runner (:mod:`plurai.run_simulation`) drives this ONCE at the end of a run —
    right before the summary print — so the operator gets a clickable "where do I look"
    link to the dataset/experiment the run recorded against, instead of only local
    counts.

    Why a SEPARATE protocol (not a default on the existing roles)
    -------------------------------------------------------------
    Same reasoning as :class:`ExperimentLifecycleSink`: a method added to
    :class:`InstrumentationSink` / :class:`ObservabilitySink` — even with a default
    body — would make every structurally-conforming sink that doesn't define it fail
    the factory's runtime ``isinstance`` conformance check (Protocol defaults are not
    inherited structurally). Keeping this independent makes the seam purely ADDITIVE:
    a sink opts in by defining ``dataset_url``; every other sink is skipped by the
    runner's ``isinstance`` gate.

    Best-effort: a conforming sink must self-guard and NEVER raise — an unresolvable
    URL (no creds, dataset missing, transport failure, no well-defined vendor URL
    shape) returns ``None``, and the runner additionally wraps the call so a
    misbehaving sink can never fail the run.
    """

    def dataset_url(self, dataset_name: str) -> str | None:
        """Best-effort platform URL for *dataset_name*; ``None`` when unresolvable.

        SYNC by design (called once, after the run, outside any hot path). Must
        never raise and never print — the caller owns the presentation.
        """
        ...


@dataclass(frozen=True, slots=True)
class RunLinks:
    """Where a run lives on the platform: its experiment page and one trace per scenario.

    ``experiment_url`` is the page that lists the run's traces (``None`` when the platform has
    no such page or it could not be resolved). ``traces`` pairs each scenario folder the sink
    recorded with the URL of its trace; a scenario whose URL could not be built is absent.
    """

    experiment_url: str | None
    traces: tuple[tuple[str, str], ...]


@runtime_checkable
class RunLinksSink(Protocol):
    """OPTIONAL run-link role: resolve the URLs of the run's experiment and per-scenario traces.

    The runner (:mod:`plurai.run_simulation`) drives this ONCE per run, right before the sink
    closes (a closed client may stop serving reads) and next to :class:`DatasetUrlSink`, so the
    summary can print ``Experiment:`` and ``Trace:`` lines that take an operator from the
    terminal to the recorded run.

    Why a SEPARATE protocol
    -----------------------
    Same reasoning as :class:`DatasetUrlSink`: a method added to :class:`InstrumentationSink` /
    :class:`ObservabilitySink` would fail the factory's runtime ``isinstance`` check for every
    sink that has not implemented it. A sink opts in by defining ``run_links``; the runner skips
    the rest.

    Best-effort: a conforming sink must self-guard and NEVER raise. Nothing resolvable returns
    ``None``, and the runner additionally wraps the call.
    """

    def run_links(self, dataset_name: str) -> RunLinks | None:
        """Best-effort links for the scenarios this sink recorded; ``None`` when there are none.

        SYNC by design (called once, after the run, outside any hot path). Must never raise
        and never print — the caller owns the presentation.
        """
        ...


@runtime_checkable
class ObservabilitySink(InstrumentationSink, DatasetSink, Protocol):
    """Combined contract: a sink that fills BOTH roles (tracing AND dataset).

    The :func:`plurai.observability.create_observability_sink` factory returns
    this — the orchestrator holds ONE instance used for per-turn tracing AND (via
    the dataset path) for upload, and ``NoOpSink`` must satisfy both. Sites that
    need only ONE role should annotate against :class:`InstrumentationSink` or
    :class:`DatasetSink` instead, so they depend on the narrowest shape they use.

    ``@runtime_checkable`` so the factory can backstop construction with
    ``isinstance(sink, ObservabilitySink)`` — a malformed sink then fails at
    STARTUP rather than with a vanished mid-run ``AttributeError`` (see the
    factory's RUNTIME-BACKSTOP note).
    """


class _DisabledRunContext:
    """The always-disabled :class:`RunContext` sentinel that :class:`NoOpSink` returns.

    ``prepare_run`` NEVER returns ``None`` — when no sink is configured the orchestrator
    holds a :class:`NoOpSink`, whose ``prepare_run`` returns THIS sentinel
    (``enabled`` is always ``False``). So every call site gates uniformly on
    ``run_ctx.enabled`` and never branches on ``None``. The concrete sinks each return
    their OWN context with ``enabled=False`` for the disabled case; this is purely the
    no-sink fallback.
    """

    @property
    def enabled(self) -> bool:
        """Always ``False``: a NoOpSink run is never live."""
        return False


class NoOpSink:
    """The active sink when none is configured: every method is a no-op.

    Lets the orchestrator hold a non-None sink unconditionally and call its
    methods without branching on disabled / None. A STANDALONE class (no shared
    base): it sets ``name`` and stubs every role method plus the async ``close``
    drain as a no-op, so it satisfies BOTH the :class:`InstrumentationSink` and
    :class:`DatasetSink` protocols (hence the combined :class:`ObservabilitySink`)
    and ALSO the optional :class:`ExperimentLifecycleSink` seam. It needs no
    best-effort guard because nothing it does can raise.
    """

    name = 'noop'

    async def warm_once(self) -> None:  # noqa: D102
        return None

    async def prepare_run(
        self,
        *,
        agent_name: str,
        experiment_name: str,
        scenario_name: str,
        thread_id: str,
        scenario_folder: str | None = None,
        dataset_name: str | None = None,
        example_inputs: dict[str, Any] | None = None,
        agent_version: str,
    ) -> RunContext:  # noqa: D102
        return _DisabledRunContext()

    async def post_feedback(
        self,
        thread: CreateThreadResponse,
        finalize: FinalizeResponse,
        run_ctx: RunContext,
        *,
        final_world_state: dict[str, Any] | None = None,
        tool_events: dict[str, Any] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:  # noqa: D102
        return None

    async def record_transcript(
        self,
        run_ctx: RunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:  # noqa: D102
        return None

    async def drain_run(self, run_ctx: RunContext) -> None:  # noqa: D102
        return None

    def sync_scenarios(
        self,
        *,
        dataset_name: str,
        dataset_description: str | None,
        scenarios: list[ScenarioUploadEntry],
        existing_metadata: dict[str, Any],
    ) -> ProviderSyncResult:  # noqa: D102
        return ProviderSyncResult(provider='noop', metadata_updates={}, summary_line='NoOpSink: no sink configured — nothing uploaded.')

    async def close(self) -> None:  # noqa: D102
        return None

    # Optional scenario-SET lifecycle seam (ExperimentLifecycleSink): explicit no-op
    # stubs so NoOpSink also satisfies that protocol — the runner's optional dispatch
    # finds them and they do nothing, exactly like every other NoOpSink method.
    async def begin_experiment(self, *, dataset_name: str, run_name: str) -> None:  # noqa: D102
        return None

    async def end_experiment(self) -> None:  # noqa: D102
        return None

    # Optional platform-link seam (DatasetUrlSink): explicit None stub so a
    # no-sink run resolves no URL (and prints no dataset line) without crashing.
    def dataset_url(self, dataset_name: str) -> str | None:  # noqa: D102
        return None

    # Optional run-link seam (RunLinksSink): a no-sink run has no links to print.
    def run_links(self, dataset_name: str) -> RunLinks | None:  # noqa: D102
        return None

    # DELIBERATELY no run_experiment_set stub: ExperimentRunnerSink is an opt-in
    # capability, and a no-observability run must take the native gather path — the
    # strategy selector dispatches on bare isinstance(sink, ExperimentRunnerSink).
