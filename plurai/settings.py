"""Project-wide settings for the plurai SDK — one global :class:`Settings` object.

This module owns the SINGLE global :class:`Settings` carrier the SDK resolves from
the process environment. Its fields are the observability vendor configs (FLAT, one
field per vendor) PLUS a :class:`NetworkSettings` component — there is no
``observability`` sub-grouping; the four sink configs sit DIRECTLY on
:class:`Settings`.

Env contract (two namespaces, deliberately different):

  - SINK config is FLAT — each vendor config nests under its bare ``<SINK>__*``
    namespace: ``LANGSMITH__API_KEY``, ``LANGSMITH__ENABLED``, ``BRAINTRUST__ENABLED``,
    ``ARIZE__SPACE_ID`` … (the ``<SINK>`` prefix IS the field name; the ``__``
    delimiter descends into the per-sink config). Every sink and ``run_simulation``
    read this flat surface, so it stays flat.
  - NETWORK config is PREFIXED under ``network`` — ``NETWORK__HTTP_TIMEOUT``,
    ``NETWORK__AGENT_TURN_TIMEOUT``, ``NETWORK__SCENARIO_TIMEOUT``, ``NETWORK__RETRIES``,
    ``NETWORK__BACKOFF_BASE``, ``NETWORK__BACKOFF_CAP``
    (the ``network`` field name + ``__`` delimiter). It holds transport tuning only: which
    backend a process talks to, and with what token, is never a setting. It is resolved by
    :func:`plurai._endpoints.resolve` (``PLURAI_URL`` / ``PLURAI_TOKEN``, else the saved login)
    and read through :attr:`Settings.endpoint`.

Single-active-sink contract: at most ONE ``<SINK>__ENABLED`` flag may be set. The
``_compose`` after-validator fails fast on more than one explicitly-enabled sink and
resolves the single active sink name onto ``Settings.active`` (or ``None``). The
factory (:func:`plurai.observability.create_observability_sink`) reads
``settings.active`` and resolves the matching config via a typed ``SinkName``-keyed
lookup (direct attribute access, NOT ``getattr`` on a dynamic field name).

Import direction stays one-way: ``plurai.settings`` imports the per-sink config
classes from the sink modules; nothing under ``plurai.observability`` imports
``plurai.settings`` (that would close an import cycle), so the graph remains acyclic.
"""

from __future__ import annotations

import logging
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from plurai._dotenv import resolve_dotenv_path
from plurai._endpoints import Endpoint, resolve
from plurai._protocol import MockMode
from plurai.observability.arize import ArizeSinkConfig
from plurai.observability.braintrust import BraintrustSinkConfig
from plurai.observability.langfuse import LangfuseSinkConfig
from plurai.observability.langsmith import LangSmithSinkConfig

logger = logging.getLogger(__name__)

# The set of valid sink field names — used as the type of ``Settings.active`` so the
# single-active contract is enforced at the type level, not just at runtime.
SinkName = Literal['langsmith', 'langfuse', 'arize', 'braintrust']

# The run PHASE — a qualification pass is a fail-fast gate over a small sample, a full
# run is the scored deliverable. It selects DEFAULTS only (the per-turn budget below and
# the turn-timeout re-drive policy), so it always sits at the BOTTOM of the precedence
# chain and can never outrank a value an operator actually declared.
RunPhase = Literal['qualification', 'full']

# The OBSERVABILITY POSTURE of a run: does this invocation write to an external platform?
#
# ``isolated`` — every sink is off, whatever the environment asked for. ``recording`` — a
# sink enabled by ``<SINK>__ENABLED`` is honoured. The DEFAULT is ``isolated``, so recording
# is opt-in: a sink flag is discovered by walking up from the cwd to the nearest ``.env``, so
# a parent directory belonging to the CUSTOMER can turn a production sink on with nothing on
# the command line and nobody's decision behind it (a run that scored nothing would still write
# a dataset into the customer's platform workspace). Sinks are cheap to re-enable and impossible
# to un-write.
ObservabilityPosture = Literal['isolated', 'recording']

# Per-turn agent budgets, by phase. Both are wall-clock seconds for ONE child envelope, and
# each is the phase's BASE budget plus WORKFLOW_TURN_ALLOWANCE_SECONDS.
#
# QUALIFICATION (base 10 min) is deliberately aggressive: the gate exists to EXPOSE a stalled
# agent on a 3-scenario sample, and a stall that takes an hour to surface has defeated the
# point of running a sample at all. It also skips the turn-timeout re-drive (see
# ``redrive.classify_restart_failure``) so the wall is really one budget, not 2 x it.
#
# FULL (base 20 min) buys a slow-but-working agent room to finish the scored run.
#
# The base budget leaves a slow but healthy turn room to finish, so a working agent does not
# need its budget raised to complete a run. Turns beyond the budget are NOT accommodated by
# design — a turn that overruns is the finding, not a bug; raise it per workspace via the ``.turn_timeout`` sidecar when that is
# genuinely expected.
#
# The workflow allowance. A mocked tool whose answer the backend computes live (a
# "workflow" tool) may legitimately hold the turn. A SYNCHRONOUS workflow call blocks its tool
# call for up to 240s (the innermost bound of the /mock/execute chain, see
# ``plurai._mocking.client.DEFAULT_MOCK_EXECUTE_TIMEOUT``); an asynchronous job runs up to the backend's
# workflow job deadline (300s, its queue wait included). Both are copied
# here so the order can be tested. The allowance covers FOUR sync calls at their deadline in one
# turn, on top of the agent's own base budget, which also covers a depth-2 async chain at its
# deadline plus 60s of reveal and poll slack. Applied to every run (no per-env signal reaches the
# runner); the cost is that a genuine stall surfaces 16 minutes later.
BACKEND_SYNC_WORKFLOW_CALL_SECONDS = 240.0
BACKEND_WORKFLOW_JOB_DEADLINE_SECONDS = 300.0
WORKFLOW_TURN_SYNC_CALLS = 4
WORKFLOW_TURN_ALLOWANCE_SECONDS = WORKFLOW_TURN_SYNC_CALLS * BACKEND_SYNC_WORKFLOW_CALL_SECONDS
QUALIFICATION_AGENT_TURN_BASE_SECONDS = 600.0
FULL_RUN_AGENT_TURN_BASE_SECONDS = 1200.0
QUALIFICATION_AGENT_TURN_TIMEOUT = QUALIFICATION_AGENT_TURN_BASE_SECONDS + WORKFLOW_TURN_ALLOWANCE_SECONDS  # 1560s, 26 min
FULL_RUN_AGENT_TURN_TIMEOUT = FULL_RUN_AGENT_TURN_BASE_SECONDS + WORKFLOW_TURN_ALLOWANCE_SECONDS  # 2160s, 36 min

# The per-SCENARIO time budget each phase defaults to — the bound that LANDS the scenario.
# QUALIFICATION (25 min) is sized for the quality loop, where scenarios run in parallel and a
# single long tail stalls a whole round; FULL (40 min) buys the deliverable run room to finish
# a flow the loop already accepted as legitimately slow.
#
# Sized against measured runs, not taste. Expiry is not fatal: it lands the scenario with its
# artifacts, so the budget can be tight, where a budget that destroyed the scenario at expiry
# could not.
QUALIFICATION_SCENARIO_TIME_BUDGET = 1500.0
FULL_RUN_SCENARIO_TIME_BUDGET = 2400.0

# Extra room the internal HARD ceiling gets on top of every stage bound it must dominate (see
# ``derive_scenario_hard_ceiling``). Covers the closing turn, the child's finalize handshake and
# the transport legs around them — the landing path that runs AFTER the budget has already
# fired. Generous on purpose: this ceiling exists to catch a HANG, and a ceiling that fires on a
# slow-but-landing scenario cancels exactly the scenario the budget exists to land.
SCENARIO_HARD_CEILING_SLACK_SECONDS = 900.0

# Headroom added to the VENDOR executor's per-task timeout so OUR ceiling always fires first.
# Policy, not a settings field — the two budgets behave completely differently at expiry. Ours
# raises ScenarioTimeout inside the task, which the vendor's ``except Exception`` records as a
# failed row and moves on. The vendor's own timeout does NOT cancel the task and requeues the
# item at the same priority with no attempt counter (arize executors.py: "Worker timeout,
# requeuing"), so an orphaned run keeps going while a second attempt starts, deletes the
# first's error.json as a stale artifact, and can loop indefinitely at one ceiling per
# iteration. Set them equal and the vendor wins every race, because its timer starts first —
# which is why this margin must not be operator-tunable down.
VENDOR_TIMEOUT_HEADROOM_SECONDS = 300

# Liveness-probe shape for the pre-run /healthz check: a few quick connect attempts, so ONE
# transient blip (exactly the failure class the 503 retry budget exists to survive) can
# never hard-refuse a run. Policy, not settings fields — tuning these looser defeats the
# probe's job (answer "is the backend THERE" in seconds, before a run spends anything).
PREFLIGHT_ATTEMPTS = 3
PREFLIGHT_RETRY_DELAY = 1.0
PREFLIGHT_CONNECT_TIMEOUT = 5.0

# Floor for every seconds-valued field below (and their CLI flags, which build a validated
# NetworkSettings). Not a style preference: OrchestratorClient derives
# ``connect=min(10.0, timeout)``, so a sub-millisecond read ceiling collapses the CONNECT
# budget with it and every request dies before the socket is even dialed. No HTTP round trip
# completes in under a millisecond, so a smaller value is always a typo (a unit slip — ms
# passed where seconds are meant), and failing at validation beats losing every scenario to
# an unexplained timeout.
MIN_SECONDS = 0.001


def default_agent_turn_timeout(phase: RunPhase) -> float:
    """The per-turn budget a phase defaults to, absent any explicit declaration."""
    return QUALIFICATION_AGENT_TURN_TIMEOUT if phase == 'qualification' else FULL_RUN_AGENT_TURN_TIMEOUT


def default_scenario_time_budget(phase: RunPhase) -> float:
    """The per-scenario TIME budget a phase defaults to, absent any explicit declaration."""
    return QUALIFICATION_SCENARIO_TIME_BUDGET if phase == 'qualification' else FULL_RUN_SCENARIO_TIME_BUDGET


def derive_scenario_hard_ceiling(scenario_time_budget: float, net: NetworkSettings) -> float:
    """The INTERNAL hard ceiling — a hang detector, not a budget, and not operator-tunable.

    ``scenario_timeout`` is the bound an operator sets, and it LANDS the scenario: the dialog
    ends between turns, the closing turn is delivered, /finalize runs, the row is graded and
    marked ``incomplete``. This ceiling is the wholly different thing underneath it — the
    ``asyncio.wait_for`` that CANCELS the attempt and leaves an ``error.json``, with the partial
    conversation beside it, instead of a graded row. It exists for
    the case the budget cannot reach: a turn that never returns, a wedged transport, a child
    that stopped answering. Path-independent by design, so "no scenario may run forever" is not
    an accident of which observability sink is configured.

    DERIVED to dominate every stage bound on the landing path, because cancelling a scenario
    that was about to land is precisely the loss it exists to prevent. Worst legal case, term by
    term: the budget is read only BETWEEN turns, so a turn already in flight can run its full
    per-turn budget past the last check (1x ``agent_turn_timeout``) and its ``/user-turn`` leg
    can take the full read timeout (1x ``http_timeout``); the closing turn is then delivered
    (a 2nd ``agent_turn_timeout``); and /finalize submits and polls the judges (its own wall
    clock). Plus slack for the child handshake and the remaining transport legs. Sum it and
    anything this ceiling catches is a stage bound that itself failed to hold — which is the
    only thing it should ever catch.

    Deliberately NOT tunable, and deliberately not the sum of the tightest numbers that would
    "work": every term here is a bound something else already enforces, so being generous costs
    nothing on a healthy run and being tight would resurrect a fatal timeout. The price is paid
    only on the hang path, where a wedged attempt holds its slot for the full ceiling — the
    deliberate trade for never cancelling a scenario that was about to land.

    Takes the whole ``NetworkSettings`` rather than three loose floats, so the stage bounds
    cannot be passed in the wrong order: all three are ``float``, and a transposition would
    type-check.
    """
    return (
        scenario_time_budget + (2 * net.agent_turn_timeout) + net.http_timeout + net.finalize_poll_max_wallclock_seconds + SCENARIO_HARD_CEILING_SLACK_SECONDS
    )


class NetworkSettings(BaseModel):
    """Network / transport config for the orchestrator + its HTTP client.

    A frozen ``BaseModel`` (NOT ``BaseSettings``) mirroring the per-sink config
    conventions — typed fields whose defaults match the orchestrator/client values,
    Populated by the OUTER :class:`Settings` from the ``NETWORK__*`` prefixed env namespace.
    """

    model_config = ConfigDict(frozen=True)

    # The single READ timeout for ALL orchestrator calls — /threads, /user-turn AND
    # /finalize (connect is clamped separately at 10s in client.py, so a dead host still
    # fails in seconds). Set generously: /finalize runs all of the backend's judges (many LLM
    # calls in parallel) and routinely takes minutes, so a snappy ceiling would kill
    # scoring AFTER a good dialog; a fast per-turn call under a high ceiling is harmless.
    # 1200 and not less: a /user-turn read timeout is NOT retryable (see ``retries``
    # below), so this ceiling is the only thing standing between a slow turn and a lost
    # scenario. The server's own budget is smaller, so the headroom costs nothing on a
    # healthy box; no proxy ceiling sits above this, so the client is the binding limit.
    # Override via NETWORK__HTTP_TIMEOUT or run_simulation's --http-timeout.
    #
    # allow_inf_nan=False on every timeout below, not decoration: `gt=0` rejects NaN but
    # ACCEPTS +inf, so without it NETWORK__HTTP_TIMEOUT=inf builds a client that never times
    # out — the one setting whose whole job is to bound a wait, silently unbounded. The env
    # path must refuse it exactly as the CLI flag's own validator does; the env is the
    # channel an operator reaches for under pressure.
    http_timeout: float = Field(default=1200.0, ge=MIN_SECONDS, allow_inf_nan=False)
    # Per-turn agent cap — current Orchestrator agent_turn_timeout; fed to ChildProcess.
    # This is the child-process wall-clock budget, NOT an httpx timeout: it bounds the
    # laptop->customer-agent leg, while ``http_timeout`` above bounds laptop->backend. They
    # are DIFFERENT legs and their numbers do not interact — do not "reconcile" them.
    #
    # The default is the FULL-run budget. A qualification pass lowers it to
    # QUALIFICATION_AGENT_TURN_TIMEOUT via ``--run-phase``, which is a DEFAULT and so still
    # loses to the flag, the env var and the ``.turn_timeout`` sidecar. See the phase
    # constants above for how both numbers are sized.
    agent_turn_timeout: float = Field(default=FULL_RUN_AGENT_TURN_TIMEOUT, ge=MIN_SECONDS, allow_inf_nan=False)
    # Per-SCENARIO TIME BUDGET, and the ONE scenario-level bound an operator sets. It LANDS the
    # scenario rather than killing it: read between turns, it ends the dialog through the
    # harness's own ``ended_early`` channel (reason ``time_budget_exhausted``), so the closing
    # turn is still delivered, /finalize still runs, the judges grade the turns that WERE
    # reached, and the row is recorded ``incomplete``. Budget exhaustion is therefore a scored
    # outcome, not silent wall clock — which is what makes it safe to set tightly.
    #
    # ``max_turns`` cannot express this: one real-mode turn can legitimately cost minutes
    # (a scenario of a dozen turns can run most of an hour, with no hang and no retries), so a
    # turn count is not a time bound.
    #
    # ``None`` means FILL FROM THE RUN PHASE (25 min qualification / 40 min full, see
    # ``default_scenario_time_budget``); the CLI always fills it, and a library caller that
    # leaves it unset is bounded by the full-run default rather than running unbounded. The
    # ceiling that CANCELS an attempt is derived and internal
    # (``derive_scenario_hard_ceiling``) — a hang detector, not a knob.
    scenario_timeout: float | None = Field(default=None, ge=MIN_SECONDS, allow_inf_nan=False)
    # The deployed mock's URL, when this workspace PINS one. Resolved from
    # ``--mock-service-url`` > ``NETWORK__MOCK_SERVICE_URL`` > the workdir's
    # ``.mock_service_url`` sidecar (the same precedence ``mock_mode`` takes over ``.run_mode``);
    # ``None`` means nothing pinned it and the SERVER's per-thread url is the only one. Lives
    # here rather than only on the CLI namespace so the sidecar the setup flow already wrote is
    # a first-class source instead of a value the operator has to re-supply per invocation.
    mock_service_url: str | None = None
    # Retry budget for transient HTTP. What it covers is RISK-SCOPED, not blanket —
    # _retry.py:_is_retryable is the exact policy: connection errors always; 502/503/504
    # and timeouts only on requests the client marks idempotent, which is the GETs plus
    # the finalize submit/poll/cancel trio (the submit POST rides the server's
    # same-thread job dedup, so a retry returns the live job instead of spawning a
    # second); 503 ALSO on POST /threads and /user-turn, because only the gateway emits
    # one there and a request it never forwarded was provably not applied — but NOT on POST /fail, which takes no status/timeout
    # retries at all. A READ timeout on POST /user-turn is deliberately NOT retried —
    # the turn may already have been applied and the wire carries no turn-level dedupe,
    # so a retry would duplicate it. Raise http_timeout for that class, not this.
    # Default 6 gives 7 total attempts with jittered exponential back-off: waits of
    # 1, 2, 4, 8, 16 and 30s (+ up to 1s jitter on the first five) = a ~61-66s sleep window, sized to
    # outlast the window in which a backend upgrade rejects every request: a retry budget smaller than the
    # outage it exists to ride out is decoration, and a smaller budget would give up too early.
    # Set NETWORK__RETRIES=0 to disable retries (1 attempt, no retry at all). Left unset (``None``), the transient budget is the
    # default above and a 429 keeps the SDK-wide budget of 5; a count set here is the transient budget, and a 429 gets at least
    # that many (``RetryConfig.from_settings``).
    retries: int | None = Field(default=None, ge=0)
    # Jittered exponential back-off bounds (seconds); see _retry.py for the exact wait
    # shape. ``backoff_cap`` is a per-wait ceiling, NOT the window size — at the default
    # retry count the cap holds the 6th wait to 30s. These (plus ``retries``) are the env-driven source for the
    # orchestrator's RetryConfig — build_default_client maps them via
    # RetryConfig.from_settings. RetryConfig's own dataclass defaults MIRROR these
    # values for standalone construction; a test fences the two against drift.
    backoff_base: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    backoff_cap: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    # Mock-service posture for the orchestrated run path. ``'connect'`` (prod
    # default) makes the child connect to the server-supplied deployed mock and
    # RAISE if it's unreachable; ``'local'`` (explicit opt-in for dev / tests) makes
    # the child host its OWN in-process loopback mock with the registered
    # ``@plurai.mock`` handlers; ``'mockless'`` declares that NOTHING is mocked (a
    # tool-less agent, or one whose every tool runs ``mode: real``) and shares local's
    # idle stub. Threaded to the child via ``Init.mock_mode`` — the
    # posture is a declared policy, never inferred from a failed connect.
    mock_mode: MockMode = 'connect'
    # Async-finalize poll cadence — seconds between GET /threads/{id}/finalize/jobs/{job_id}
    # status polls. Override via NETWORK__FINALIZE_POLL_INTERVAL_SECONDS.
    finalize_poll_interval_seconds: float = Field(default=3.0, ge=MIN_SECONDS, allow_inf_nan=False)
    # Max wallclock budget (seconds) for the async finalize poll loop. On exceed,
    # FinalizePollTimeout is raised → error.json. Default 1800s (30 min) is well
    # above expected finalize p99. Override via NETWORK__FINALIZE_POLL_MAX_WALLCLOCK_SECONDS.
    finalize_poll_max_wallclock_seconds: float = Field(default=1800.0, ge=MIN_SECONDS, allow_inf_nan=False)


class Settings(BaseSettings):
    """The single project-wide settings carrier (flat sinks + a network component).

    Fields: the four per-sink configs (FLAT ``<SINK>__*`` env), the ``network``
    component (``NETWORK__*`` env), and the excluded resolved ``active`` sink name.
    The ``_compose`` after-validator enforces the single-active-sink contract and
    stores the resolved active sink name on ``active``.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra='ignore',
        env_nested_delimiter='__',
        env_ignore_empty=True,
    )

    # The four per-sink configs, populated FLAT from the ``<SINK>__*`` namespace.
    langsmith: LangSmithSinkConfig = Field(default_factory=LangSmithSinkConfig)
    langfuse: LangfuseSinkConfig = Field(default_factory=LangfuseSinkConfig)
    arize: ArizeSinkConfig = Field(default_factory=ArizeSinkConfig)
    braintrust: BraintrustSinkConfig = Field(default_factory=BraintrustSinkConfig)

    # The network / transport component, populated from the ``NETWORK__*`` namespace.
    network: NetworkSettings = Field(default_factory=NetworkSettings)

    # The resolved backend. A private attribute, not a field: the backend is never read from the
    # settings sources (environment names or ``.env``), only from the resolver.
    _endpoint: Endpoint | None = PrivateAttr(default=None)

    # The id of the command this run belongs to (``plurai.command_run``). Not a settings source either: the
    # runner resolves it from its flag and the environment and attaches it.
    _command_run_id: str | None = PrivateAttr(default=None)

    # The single active sink name (or ``None`` when none is enabled), resolved ONCE
    # by ``_compose`` (the after-validator). At most ONE sink is ever active — the
    # single-active contract raises on >1 — so this is ``str | None``, not a tuple.
    # NOT an env field (``exclude=True`` keeps it off the model dump / construction
    # surface); the factory reads it instead of recomputing. Defaults to ``None`` so
    # a ``model_construct`` carrier (which skips ``_compose``) still has the attribute.
    active: SinkName | None = Field(default=None, exclude=True)

    @model_validator(mode='after')
    def _compose(self) -> Settings:
        """Resolve the final enabled set ONCE (fail-fast on multiple actives)."""
        # Anchor the sink names to the ``SinkName`` literal via a typed registry of
        # ``SinkName -> config`` built with DIRECT attribute access (no ``getattr`` on
        # a dynamic field name). Adding a sink means extending ``SinkName`` + this
        # mapping — a pyright-checked edit, not an attribute sniff.
        sink_configs: dict[
            SinkName,
            LangSmithSinkConfig | LangfuseSinkConfig | ArizeSinkConfig | BraintrustSinkConfig,
        ] = {
            'langsmith': self.langsmith,
            'langfuse': self.langfuse,
            'arize': self.arize,
            'braintrust': self.braintrust,
        }
        explicit = tuple(name for name, cfg in sink_configs.items() if cfg.enabled)
        if len(explicit) > 1:
            flags = ', '.join(f'{n.upper()}__ENABLED' for n in explicit)
            raise ValueError(
                f'Multiple observability sinks explicitly enabled ({flags}). The '
                'single-active-sink contract allows at most one: two live sinks '
                'share one tracer context, which breaks trace trees. Unset all but '
                'one of these flags: '
                f'{", ".join(f"{n.upper()}__ENABLED" for n in get_args(SinkName))}.'
            )

        # Store the single active sink name ONCE (or None when none enabled). The
        # >1 case already raised above, so ``explicit`` holds 0 or 1 name here.
        # Use object.__setattr__ because the model is frozen (pydantic-frozen pattern).
        object.__setattr__(self, 'active', explicit[0] if explicit else None)
        return self

    @property
    def endpoint(self) -> Endpoint:
        """The backend this run talks to: the one :func:`plurai._endpoints.resolve` returns, resolved on first use and kept.

        Raises:
            plurai._endpoints.EndpointError: nothing, or only half, is configured.
        """
        if self._endpoint is None:
            self._endpoint = resolve()
        return self._endpoint

    @property
    def resolved_endpoint(self) -> Endpoint | None:
        """The endpoint if one is already resolved or attached, without resolving one."""
        return self._endpoint

    @property
    def command_run_id(self) -> str | None:
        """The id that groups this run's jobs on the backend, when the run was given one."""
        return self._command_run_id

    def with_command_run_id(self, command_run_id: str | None) -> Settings:
        """A copy that sends ``command_run_id`` with the requests that start work."""
        copy = self.model_copy()
        copy._command_run_id = command_run_id
        return copy

    def with_endpoint(self, endpoint: Endpoint) -> Settings:
        """A copy that talks to ``endpoint`` instead of resolving one."""
        copy = self.model_copy()
        copy._endpoint = endpoint
        return copy

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Point the dotenv source at the nearest ``.env`` (walking up from cwd).

        Lets the main ``plurai-run-simulation`` process pick up nested ``<SINK>__*``
        config that lives only in ``.env`` — it never imports the agent's
        ``simulate.py``, which is what seeds ``os.environ`` in the agent subprocess.
        The dotenv source stays BELOW ``env_settings``, so exported env vars still
        beat ``.env``; the override is inert when no ``.env`` is found or
        ``PLURAI_DISABLE_DOTENV=1`` is set (see :mod:`plurai._dotenv`).
        """
        env_path = resolve_dotenv_path()
        if env_path is not None:
            dotenv_settings = DotEnvSettingsSource(
                settings_cls,
                env_file=env_path,
                env_file_encoding='utf-8',
                case_sensitive=False,
                env_nested_delimiter='__',
                env_ignore_empty=True,
            )
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)
