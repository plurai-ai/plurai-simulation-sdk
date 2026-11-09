"""Braintrust integration — parent-side, span-grouped observability sink."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from plurai.observability.sink_protocol import RunContext, TranscriptTurn
from plurai.observability.dataset_types import ProviderSyncResult, dataset_join_key
from plurai.observability.eval_mapping import AGGREGATE, PASS_RATE, SCENARIO_PASSED
from plurai.observability.feedback import build_scenario_summary
from plurai.observability.rollup import (
    chatbot_messages_of,
    compute_rollup_metrics,
    eval_results,
    score_from_result,
)
from plurai.observability.errors import offload, swallow, swallowable_error_types
from plurai.orchestrator.models import CreateThreadResponse, FinalizeResponse, ScenarioUploadEntry

if TYPE_CHECKING:
    from braintrust import Dataset, Experiment, Logger, Span

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BraintrustRunContext:
    """Braintrust per-scenario run context — the single object ``prepare_run`` returns.

    Carries the genuine PER-SCENARIO run state directly (there is NO separate handle
    wrapper). ``span_slot`` holds the
    per-thread ROOT span (a mutable :class:`_ThreadSpanSlot` so this context can stay
    ``frozen=True``), or ``None`` — the DISABLED SENTINEL: ``prepare_run`` returns a
    context with ``span_slot=None`` (so ``enabled`` is False) instead of ``None`` when
    the sink is disabled / unwarmed, so ``prepare_run`` never returns ``None``. When a
    slot is present, its ``ended`` guard is what makes :func:`flush` — and therefore
    :meth:`BraintrustSink.drain_run` — IDEMPOTENT: the first drain ends the span and
    sets ``slot.ended``, and any later drain (e.g. the scenario-scope ``finally`` after
    ``post_feedback`` already flushed on the happy path) sees the guard and skips the
    second ``end()``. ``experiment`` / ``scenario_name`` / ``thread_id`` stamp the span
    metadata for cross-sink RunContext parity.

    ``experiment_obj`` / ``dataset_record`` / ``dataset_origin`` are the OPTIONAL
    scenario-SET experiment linkage (populated only after
    :meth:`BraintrustSink.begin_experiment` created a real Braintrust Experiment): when
    ``experiment_obj`` is non-``None`` the per-scenario ROOT span is opened on the
    Experiment (``type='eval'``) carrying the dataset record's ``input`` / ``expected``
    plus the ``dataset_origin`` link to that record; when it is ``None`` the project-logs
    ``Logger`` root (``type='task'``) is used instead. All three default to ``None``.
    """

    span_slot: _ThreadSpanSlot | None
    experiment: str
    scenario_name: str
    thread_id: str
    experiment_obj: Experiment | None = None
    dataset_record: dict[str, Any] | None = None
    dataset_origin: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        """True when Braintrust tracing is LIVE for this scenario — exposes the EXISTING gate.

        ``record_transcript`` / ``_post_feedback`` already short-circuit on
        ``run_ctx.span_slot is None``. :meth:`BraintrustSink.prepare_run` always returns
        a :class:`BraintrustRunContext`; ``span_slot=None`` is the disabled sentinel
        (produced when the sink is disabled or :meth:`BraintrustSink.warm` did not
        produce a Logger). This property surfaces that existing decision uniformly;
        it does not change what gets traced.
        """
        return self.span_slot is not None


class BraintrustSinkConfig(BaseModel):
    """Braintrust observability sink config (mirrors the Plurai backend's Braintrust settings)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    api_key: str | None = None
    project: str | None = None

    @field_validator('api_key', 'project', mode='before')
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        """Coerce an empty / whitespace-only credential string to ``None``."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode='after')
    def _require_creds_when_enabled(self) -> BraintrustSinkConfig:
        """When the sink is ACTIVE, require ``api_key`` to be non-``None``.

        GATED on ``enabled``: ``Settings`` ALWAYS instantiates this
        nested config (even for an INACTIVE sink), so a bare required field would
        wrongly force creds for a sink that is off. The conditional-on-enabled shape
        is the only correct one — when Braintrust is the active sink ``api_key`` must
        be defined exactly once and non-``None``; an inactive config stays
        unconstrained.
        """
        if self.enabled and self.api_key is None:
            raise ValueError('Braintrust is enabled but BRAINTRUST__API_KEY is missing. Set BRAINTRUST__API_KEY (nested) when BRAINTRUST__ENABLED is set.')
        return self


# Rollup metric keys whose value is a genuine 0–1 QUALITY score (so it belongs in Braintrust
# ``scores``, which the API constrains to [0, 1]): the pass/fail verdict plus the two quality
# signals. Every OTHER rollup goes to ``metrics`` — the event field Braintrust models for
# arbitrary numerics — and for two different reasons: the tool-call / entity counts and the
# dimension-graph magnitudes routinely exceed 1, and ``scenario_status``, though it IS 0.0/1.0,
# is EVALUATION COVERAGE rather than quality (1.0 on a failed scenario), so charting it beside
# the real scores would read as a green. Its verdict WORD reaches Braintrust through the
# per-scenario summary's ``verdict`` field instead. Keyed by metric NAME so the routing is
# deterministic regardless of a run's actual count — and built from the SHARED eval_mapping
# routing-key constants (never string literals) so this consumer cannot drift from the keys
# rollup.compute_rollup_metrics actually emits.
_SCORE_ROLLUP_KEYS = frozenset({SCENARIO_PASSED, PASS_RATE, AGGREGATE})


def _braintrust_error_types() -> tuple[type[BaseException], ...]:
    """Return the Braintrust / HTTP exception classes worth swallowing.

    Braintrust transport is ``requests`` + ``urllib3``; surfaced server errors
    come back as ``braintrust.util.AugmentedHTTPError``. Looked up dynamically so
    the SDK can be imported without ``braintrust`` installed; falls back to
    ``(OSError,)`` when nothing resolves. ``ValueError`` (bad-cred login) is
    deliberately EXCLUDED so a config error propagates through direct calls;
    programmer bugs (TypeError / AttributeError / KeyError) propagate by
    construction (narrow tuple).
    """
    return swallowable_error_types(
        ('requests.exceptions', 'RequestException'),
        ('urllib3.exceptions', 'HTTPError'),
        ('braintrust.util', 'AugmentedHTTPError'),
    )


def _braintrust_available() -> bool:
    return importlib.util.find_spec('braintrust') is not None


def _span_type_attribute() -> Any:
    """Resolve braintrust's ``SpanTypeAttribute`` enum resiliently across SDK versions.

    The enum lives canonically in ``braintrust.span_types``, but whether the top-level
    package re-exports it is version-dependent — some releases drop the
    ``from braintrust import SpanTypeAttribute`` name. Try the top-level name first
    and fall back to the canonical submodule. Lazy — only reached on
    the enabled span-building path, where braintrust is guaranteed present.
    """
    try:
        from braintrust import SpanTypeAttribute
    except ImportError:
        from braintrust.span_types import SpanTypeAttribute
    return SpanTypeAttribute


@dataclass(slots=True)
class _ThreadSpanSlot:
    """Mutable single-slot holder for a thread's root span.

    Opened EAGERLY by ``prepare_run`` (via :func:`_ensure_thread_span`) and ended at
    ``flush``. Kept as a SEPARATE mutable object so
    :class:`BraintrustRunContext` itself can stay ``frozen=True`` (concurrent-read
    safe like its siblings) — only this slot's ``span`` / ``ended`` attributes
    mutate, never the run context's field graph. Per-thread: each thread owns its own
    run context + slot, and a thread's HTTP calls are sequential, so there is no
    intra- or cross-thread race on the slot (the check-then-set in
    :func:`_ensure_thread_span` has no ``await`` between the guard and the
    assignment).
    """

    span: Span | None = None
    ended: bool = False  # idempotency guard for end()
    turn_count: int = 0  # faithful per-scenario turn count, stashed by record_transcript


def _build_logger(config: BraintrustSinkConfig, *, project: str) -> Logger:
    """Construct (but do not log in / register) the process Logger.

    ``set_current=False`` is load-bearing: it avoids mutating the SDK's
    ``_cv_logger`` / ``_local_logger`` globals, so the handler's explicit
    ``logger=`` is the ONLY path to our logger. The config-construction validator
    (``_require_creds_when_enabled``) guarantees ``api_key`` is non-``None`` whenever
    the sink is enabled (and ``warm`` only calls this when enabled), so it passes
    straight through.
    """
    from braintrust import init_logger

    return init_logger(project=project, api_key=config.api_key, set_current=False)


def _dataset_origin(dataset_id: str | None, record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build the Braintrust span ``origin`` link to a dataset record (or ``None``).

    Mirrors the framework's Eval linkage (braintrust ``framework.py`` origin block): a
    span carries ``origin = {object_type, object_id, id, _xact_id, created}`` to bind it
    to the dataset record it was run against — what drives the dataset⇄experiment
    comparison grid + per-record score columns. The link REQUIRES both the record ``id``
    AND its server-assigned ``_xact_id`` (plus the dataset id); ``_xact_id`` / ``created``
    are ONLY available by FETCHING the dataset back (``insert()`` returns only the id), so
    :meth:`BraintrustSink.begin_experiment` iterates the Dataset to populate them. When any
    is missing the link degrades to ``None`` (exactly as the framework does), so the span
    still lands — just unlinked from the record.
    """
    if not record or not dataset_id:
        return None
    record_id = record.get('id')
    xact_id = record.get('_xact_id')
    if not record_id or not xact_id:
        return None
    return {
        'object_type': 'dataset',
        'object_id': dataset_id,
        'id': record_id,
        '_xact_id': xact_id,
        'created': record.get('created'),
    }


def _ensure_thread_span(run_ctx: BraintrustRunContext, bt_logger: Logger, *, thread_id: str, scenario_name: str) -> Span | None:
    """Open the per-thread ROOT span exactly once (idempotent via the slot guard).

    The transcript span subtree nests under this span → one trace tree (shared
    ``root_span_id``). ``bt_logger`` is the SHARED process Logger (the sink's
    ``_logger``) — passed in because it lives on the sink, not the per-scenario run
    context. ``thread_id`` / ``scenario_name`` are supplied by the callsites:
    ``BraintrustSink._prepare_run`` opens the root EAGERLY, and ``record_transcript``
    reuses that same open span. The slot guard makes this idempotent within a
    thread — the first call opens the span and any later call reuses it; the span's
    OWN metadata is single-nested (the clean filter key).

    ROOT parent selection (the ONLY branch): when ``run_ctx.experiment_obj`` is set (a
    scenario-SET Experiment was opened by ``begin_experiment``) the root is an
    ``Experiment.start_span(type='eval', ...)`` carrying the matched dataset record's
    ``input`` / ``expected`` + the ``origin`` link, so the run becomes a per-record
    experiment event in the comparison grid; otherwise the project-logs
    ``Logger.start_span(type='task', ...)`` root is opened. Either parent
    returns a normal ``SpanImpl``, so ``record_transcript``'s child nesting is identical.
    """
    # Lazy SDK import (like ``init_logger`` / ``init_dataset`` below) so this module
    # never imports the optional ``braintrust`` package at top level — keeps
    # ``import plurai`` working when the ``[braintrust]`` extra is absent. Only reached
    # on the enabled path, where braintrust is guaranteed present. Resolved through
    # :func:`_span_type_attribute` so a version that drops the top-level re-export still works.
    SpanTypeAttribute = _span_type_attribute()

    slot = run_ctx.span_slot
    if slot is None:
        return None  # disabled sentinel (enabled=False) — no slot to open a span into
    if slot.span is None:
        experiment_obj = run_ctx.experiment_obj
        if experiment_obj is not None:
            # Experiment-bound root: a per-record eval event under the scenario-SET
            # Experiment. ``input`` / ``expected`` come from the FETCHED dataset record
            # (so they match the record exactly) and ``origin`` links back to it; both
            # degrade gracefully (None) when no record matched this scenario.
            record = run_ctx.dataset_record or {}
            event: dict[str, Any] = {
                'name': scenario_name,
                'type': 'eval',
                'metadata': {
                    'thread_id': thread_id,
                    'scenario': scenario_name,
                    'experiment': run_ctx.experiment,
                },
            }
            if record:
                event['input'] = record.get('input')
                event['expected'] = record.get('expected')
            if run_ctx.dataset_origin is not None:
                event['origin'] = run_ctx.dataset_origin
            slot.span = experiment_obj.start_span(**event)
        else:
            # bt_logger.start_span(name=, type='task', metadata=...) → toplevel span under the Logger.
            # Queues a LazyValue row — no synchronous login.
            slot.span = bt_logger.start_span(
                name=scenario_name,
                type=SpanTypeAttribute.TASK,
                metadata={
                    'thread_id': thread_id,
                    'scenario': scenario_name,
                    'experiment': run_ctx.experiment,
                },
            )
    return slot.span


def record_transcript(
    run_ctx: BraintrustRunContext,
    bt_logger: Logger,
    *,
    thread_id: str,
    scenario_name: str,
    turns: list[TranscriptTurn],
) -> Span | None:
    """Nest the uniform 3-level span subtree under the OPEN thread ROOT span.

    Builds, under ``run_ctx.span_slot.span`` — the scenario anchor opened by
    ``prepare_run`` via :func:`_ensure_thread_span` and STILL OPEN because the
    orchestrator drives ``record_transcript`` BEFORE ``post_feedback`` (whose
    terminal :func:`flush` ends/drains it) — one child ``turn`` span per turn and,
    under each, an ``agent`` span (input = the user message the agent saw, output =
    the agent's reply) plus a sibling ``client`` span (input = the agent's reply,
    output = the next user message the backend generated). This is the same 3-level
    shape every sink emits, sourced only from orchestrator-owned strings. ``bt_logger``
    is the SHARED process Logger (the sink's ``_logger``), passed through only to open
    the root span via :func:`_ensure_thread_span`.

    The root open is idempotent via the slot guard, so a prior eager open (the real
    run path) is reused rather than reopened. Each created span is ``end()``-ed so
    its row is complete; the ROOT span is intentionally NOT ended here — ``flush``
    (driven by ``post_feedback``) owns the root's terminal ``end()``, and ending it
    early would orphan the score row logged onto it.
    """
    # Lazy SDK import (see :func:`_ensure_thread_span`) — keeps top-level ``import
    # plurai`` working without the optional ``[braintrust]`` extra; resolved via
    # :func:`_span_type_attribute` for cross-version re-export resilience.
    SpanTypeAttribute = _span_type_attribute()

    root = _ensure_thread_span(run_ctx, bt_logger, thread_id=thread_id, scenario_name=scenario_name)
    # Stash the faithful turn count on the mutable slot so ``post_feedback`` can report it
    # WITHOUT re-deriving from the thread: in production the sink's ``thread`` is a
    # ``CreateThreadResponse`` (no transcript), so ``chatbot_messages_of(thread)`` is empty.
    # This is Braintrust's native turn source, the analogue of langsmith ``turn_runs`` /
    # langfuse ``replay['turns']``. Set even when the root failed to open (below).
    if run_ctx.span_slot is not None:
        run_ctx.span_slot.turn_count = len(turns)
    if root is None:
        return None
    for turn in turns:
        turn_span = root.start_span(
            name='turn',
            type=SpanTypeAttribute.TASK,
            metadata={
                'turn_idx': turn['turn_idx'],
                'should_terminate': turn['should_terminate'],
                'turn_count': turn['turn_count'],
            },
        )
        agent_span = turn_span.start_span(
            name='agent',
            type=SpanTypeAttribute.LLM,
            # A content-part LIST rides STRUCTURED here rather than projected: this input is a
            # JSON value to this sink, and the parts say more than their text does. ``None``
            # rides as a JSON null (an agent-first turn). Only Arize projects, because an OTel
            # attribute takes no sequence of mappings.
            input=turn['input_user_message'],
            output=turn['agent_response'],
        )
        agent_span.end()
        client_span = turn_span.start_span(
            name='client',
            type=SpanTypeAttribute.LLM,
            input=turn['agent_response'],
            output=turn['output_user_message'],
        )
        client_span.end()
        turn_span.end()
    return root


def flush(run_ctx: BraintrustRunContext | None, bt_logger: Logger) -> None:
    """End the thread span (idempotent) THEN drain the bg logger (swallowed).

    Called at the end of every terminal path so the thread trace is CLOSED and
    its spans land promptly — crashed scenarios are the highest-value traces.
    ``span.end()`` logs the end-time metric; ``bt_logger.flush()`` drains the bg
    logger (``bt_logger`` is the SHARED process Logger off the sink's ``_logger`` —
    it lives on the sink, not the per-scenario run context, so it is passed in).
    Idempotent: the ``slot.ended`` guard prevents a second ``end()`` (the SDK's own
    ``_logged_end_time`` guard is a backstop). Errors are swallowed: observability
    must never abort a sim.
    """
    if run_ctx is None:
        return
    swallow = _braintrust_error_types()
    slot = run_ctx.span_slot
    if slot is not None and slot.span is not None and not slot.ended:
        try:
            slot.span.end()  # close the thread trace
        except swallow:
            logger.exception('Braintrust thread-span end failed.')
        slot.ended = True
    if bt_logger is not None:
        try:
            bt_logger.flush()  # drain bg logger
        except swallow:
            logger.exception('Braintrust flush failed; pending spans may be delayed or lost.')


#: The uuid5 namespace seed. Every stored item id derives from its value, so a different value maps the same scenario to a
#: different item id.
_ITEM_ID_SEED = 'plurai-braintrust-dataset-item'


def _stable_item_id(dataset_name: str, folder_name: str) -> str:
    """Derive a deterministic Braintrust dataset-item id from the scenario folder.

    Braintrust upserts items by ``id``, so a stable, folder-derived id makes
    re-running scenario generation idempotent (overwrite-in-place). uuid5 keeps the
    id URL-safe regardless of folder-name characters and namespaces it by dataset so
    the same folder in two datasets never collides.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{_ITEM_ID_SEED}:{dataset_name}:{folder_name}'))


def sync_scenarios_to_braintrust(
    config: BraintrustSinkConfig,
    *,
    dataset_name: str,
    dataset_description: str | None,
    scenarios: list,
    existing_items: dict | None = None,
) -> ProviderSyncResult:
    """Upsert a scenario batch into a Braintrust Dataset.

    Each entry in ``scenarios`` must have ``folder_name`` and ``inputs``;
    ``outputs`` and ``metadata`` are optional (``outputs`` is empty by design
    today — there is no ground truth to attach, not a gap).

    Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`.
    A true no-op (Braintrust disabled / unavailable) returns ``ok=True, uploaded=0``.
    A dataset get-or-create transport error ⇒ ``ok=False`` (logged at ``error``
    level); per-item failures are TALLIED (``failed`` / ``errors``) rather than
    silently dropped, and an all-items-failed batch is ``ok=False``.

    ``init_dataset(project=, name=, description=)`` is get-or-create. ``insert``
    UPSERTS on ``id`` and returns the row id string — the field is ``expected=``.
    Inserts are batched/async, so ``dataset.flush()`` after the loop is REQUIRED.

    Idempotency: items are upserted via a stable folder-derived ``id``
    (:func:`_stable_item_id`), so re-running the generator overwrites in place
    instead of duplicating. ``existing_items`` (from a prior run) takes precedence
    over the derived id.
    """
    if not config.enabled or not _braintrust_available():
        return ProviderSyncResult(provider='braintrust', metadata_updates={}, summary_line='Braintrust: disabled or SDK not installed — nothing uploaded.')

    from braintrust import init_dataset

    swallow = _braintrust_error_types()

    try:
        dataset = init_dataset(
            project=config.project or 'plurai',
            name=dataset_name,
            description=dataset_description,
            api_key=config.api_key,
        )
    except swallow as exc:
        # Dataset get-or-create transport failure is a genuine FAILURE, not a no-op.
        logger.error('Braintrust dataset get-or-create for %r failed: %s', dataset_name, exc)
        return ProviderSyncResult(
            provider='braintrust',
            metadata_updates={},
            summary_line=f'Braintrust: dataset get-or-create for {dataset_name!r} failed: {exc}',
            ok=False,
            errors=(f'dataset get-or-create: {exc}',),
        )

    existing_items = existing_items or {}
    result: dict = {}
    errors: list[str] = []

    for scen in scenarios:
        folder = scen['folder_name']
        item_id = existing_items.get(folder) or _stable_item_id(dataset_name, folder)
        try:
            returned = dataset.insert(
                input=scen['inputs'],
                expected=scen.get('outputs') or {},
                metadata={**scen.get('metadata', {}), 'scenario_folder': folder},
                id=item_id,
            )
            result[folder] = str(returned)
        except swallow as exc:
            # Best-effort per item: tally the loss (don't abort the loop).
            logger.error('Braintrust dataset-item upsert failed for scenario %r: %s', folder, exc)
            errors.append(f'{folder}: {exc}')

    try:
        dataset.flush()
    except swallow:
        logger.exception('Braintrust flush failed after syncing dataset %r.', dataset_name)

    failed = len(errors)
    total = len(scenarios)
    ok = not (total > 0 and failed == total)
    summary = f'Braintrust: uploaded {len(result)}/{total} item(s) to dataset {dataset_name!r}.'
    if failed:
        summary = f'Braintrust: uploaded {len(result)}/{total} item(s) to dataset {dataset_name!r} ({failed} failed — see log).'
    return ProviderSyncResult(
        provider='braintrust',
        metadata_updates={
            'braintrust_dataset_name': dataset_name,
            'braintrust_dataset_items': result,
        },
        summary_line=summary,
        ok=ok,
        uploaded=len(result),
        failed=failed,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# BraintrustSink — the ObservabilitySink instance for the parent-side,
# thread-span-grouped Braintrust sink. The process Logger slot + the warn-once
# flag live as INSTANCE state (``_logger`` / ``_warned``); there are NO
# module-level mutable globals. The stateless ``_ensure_thread_span`` /
# ``record_transcript`` / ``flush`` module functions take the per-scenario
# ``BraintrustRunContext`` plus the SHARED process Logger (``self._logger``)
# explicitly — the Logger lives on the sink, the run state on the context.
# ---------------------------------------------------------------------------


class BraintrustSink:
    """Braintrust sink (parent-side; per-thread span groups one trace tree).

    ``warm`` builds the process Logger once at startup; ``prepare_run`` warms,
    binds a per-thread :class:`BraintrustRunContext`, then EAGERLY opens the thread
    root span (so the real run path — which drives only ``prepare_run`` +
    ``record_transcript`` + ``post_feedback`` — reports). ``record_transcript`` nests
    the uniform 3-level span tree under that open thread root. ``post_feedback`` logs
    per-policy + rollup scores onto the thread span via ``span.log(scores=...)`` then
    flushes. ``close`` drains the bg logger.
    """

    name = 'braintrust'

    def __init__(self, config: BraintrustSinkConfig) -> None:
        self._config = config
        self._logger: Logger | None = None  # the process braintrust Logger (lazy, built by warm)
        self._warned = False  # dedups the bind-without-warm warning
        # Guards the lazy None-check-then-build in warm() so the process Logger is built
        # exactly once even if the lazy path ever runs under asyncio.to_thread.
        self._warm_lock = threading.Lock()
        # OPTIONAL scenario-SET experiment state (ExperimentLifecycleSink seam). All None
        # until begin_experiment() opens a real Experiment bound to the run's Dataset; then
        # the per-scenario root span is opened on the Experiment (see _ensure_thread_span).
        # Shared across the gather — one Experiment per scenario-SET run.
        self._experiment: Experiment | None = None  # the braintrust Experiment (created by begin_experiment)
        self._dataset: Dataset | None = None  # the bound braintrust Dataset
        self._dataset_id: str | None = None  # cached dataset.id for the origin link
        # {scenario_folder | record_id -> fetched dataset row} for the per-scenario origin
        # lookup. Value typed Any (the opaque SDK row), matching the rest of this file.
        self._record_index: dict[str, Any] = {}

    async def warm_once(self) -> None:
        """Build the process Braintrust Logger ONCE, on the event-loop thread (idempotent).

        Called at the entry point BEFORE any gather, so the process Logger build happens
        single-threaded. Idempotent: :meth:`warm` only builds when ``self._logger`` is
        still ``None`` (double-checked under the warm lock), so a second call — or a
        later lazy ``prepare_run`` self-warm — is a cheap no-op. Best-effort: never raises.
        """
        if self._logger is not None:
            return
        await offload(self.warm, sink=self.name, what='warm_once')

    def warm(self, *, project_name: str | None = None) -> None:
        """Process-scoped warm-up: build the Braintrust Logger (idempotent + locked).

        Builds the process Logger exactly once. The startup project is ``project_name``
        or the stable ``config.project or 'plurai'`` default. No-op when disabled,
        absent, or already built. Thread-safe: the None-check-then-build is guarded by
        ``self._warm_lock`` with double-checked locking, so even if the lazy
        ``prepare_run`` self-warm ever runs under ``asyncio.to_thread`` the Logger is
        built once. Belt and suspenders: ``warm_once`` builds it single-threaded at the
        entry point and this lock backstops the lazy path.
        """
        if not self._config.enabled or not _braintrust_available():
            return
        if self._logger is not None:
            return
        with self._warm_lock:
            if self._logger is not None:
                return
            self._logger = _build_logger(self._config, project=project_name or self._config.project or 'plurai')

    def _configure_tracing(self, *, experiment_name: str, scenario_name: str, thread_id: str, scenario_folder: str | None = None) -> BraintrustRunContext:
        """Bind a fresh per-thread :class:`BraintrustRunContext` (PURE BIND).

        Returns a DISABLED sentinel (``span_slot=None`` ⇒ ``enabled=False``) — NEVER
        ``None`` — when disabled, or when :meth:`warm` never ran (``self._logger`` is
        ``None``), with ONE warn-once notice in the latter. This is a pure bind: it gates
        on the cached ``self._logger`` but must NOT build the Logger (``warm`` owns that) —
        the live run context carries only the per-scenario state (a fresh
        :class:`_ThreadSpanSlot`, the ``experiment`` datum, plus the scenario identity),
        while the SHARED Logger stays on the sink (``self._logger``). The thread root span
        is opened EAGERLY later by ``prepare_run``.
        """
        # Disabled SENTINEL (span_slot=None ⇒ enabled=False): returned INSTEAD of None when
        # the sink is disabled or unwarmed, so prepare_run never returns None.
        disabled = BraintrustRunContext(span_slot=None, experiment=experiment_name, scenario_name=scenario_name, thread_id=thread_id)
        if not self._config.enabled:
            return disabled
        if self._logger is None:
            if not self._warned:
                logger.warning(
                    'Braintrust enabled but no Logger is initialized — call '
                    'warm() at your entry point (it builds the Logger once). '
                    'Falling back to no Braintrust observability for this run.'
                )
                self._warned = True
            return disabled
        # OPTIONAL experiment linkage: when a scenario-SET Experiment is live (opened by
        # begin_experiment) resolve THIS scenario's fetched dataset record + its origin
        # link so _ensure_thread_span opens the root on the Experiment. All None when no
        # experiment is active → the project-logs Logger root.
        experiment_obj = self._experiment
        record = origin = None
        if experiment_obj is not None:
            # _lookup_record's index is keyed on the upload-time FOLDER, never the display name.
            key = dataset_join_key(scenario_folder=scenario_folder, scenario_name=scenario_name)
            record = self._lookup_record(key)
            origin = _dataset_origin(self._dataset_id, record)
            if record is None and self._record_index:
                # A NON-EMPTY index that lacks this folder is unambiguous: the dataset has
                # rows, just not this scenario's. The empty case means no upload ran or the
                # fetch failed (already reported by begin_experiment), so it stays quiet —
                # a warning that fires when nothing is wrong is how the signal gets ignored.
                logger.warning(
                    f'Braintrust: scenario folder {key!r} matched no row in the experiment dataset '
                    f'({len(self._record_index)} indexed) — the root span will open WITHOUT its '
                    f'dataset-origin link, so this run loses its dataset provenance. Check the '
                    f'scenario was covered by the dataset upload.'
                )
        return BraintrustRunContext(
            span_slot=_ThreadSpanSlot(),
            experiment=experiment_name,
            scenario_name=scenario_name,
            thread_id=thread_id,
            experiment_obj=experiment_obj,
            dataset_record=record,
            dataset_origin=origin,
        )

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
    ) -> BraintrustRunContext:
        """Warm (process-once) then bind the per-thread run context; ALWAYS return it.

        ``agent_name`` / ``dataset_name`` / ``example_inputs`` / ``agent_version`` are
        accepted for the uniform orchestrator contract but unused here: the folded run
        context carries no project/dataset datum, and ``agent_version`` record tagging is
        a record-sink concern — LangSmith/Langfuse.

        Async: the blocking warm (Logger build) + eager thread-span open run inside
        via ``asyncio.to_thread`` so the orchestrator awaits this directly. Best-effort:
        a swallowed setup failure yields a DISABLED context (``span_slot=None`` ⇒
        ``enabled=False``) rather than ``None``.
        """
        return await offload(
            self._prepare_run,
            experiment_name=experiment_name,
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            thread_id=thread_id,
            sink=self.name,
            what='prepare_run',
        ) or BraintrustRunContext(span_slot=None, experiment=experiment_name, scenario_name=scenario_name, thread_id=thread_id)

    def _prepare_run(self, *, experiment_name: str, scenario_name: str, thread_id: str, scenario_folder: str | None = None) -> BraintrustRunContext:
        if self._logger is None:
            self.warm()
        run_ctx = self._configure_tracing(experiment_name=experiment_name, scenario_name=scenario_name, scenario_folder=scenario_folder, thread_id=thread_id)
        if not run_ctx.enabled:
            return run_ctx  # disabled sentinel (span_slot=None) — nothing to open
        # Eagerly open the thread ROOT span here so record_transcript can nest its
        # subtree and post_feedback has a span to log scores onto on the REAL
        # orchestrator run path — which drives only prepare_run + record_transcript +
        # post_feedback (there is no parent-side LangChain activity). Without this,
        # span_slot.span stays None and every score is silently dropped. record_transcript reuses this same span
        # via the slot guard rather than opening a second. The SHARED Logger comes off
        # the sink (self._logger) — it is not carried on the run ctx.
        _ensure_thread_span(run_ctx, self._logger, thread_id=thread_id, scenario_name=scenario_name)
        return run_ctx

    async def record_transcript(
        self,
        run_ctx: RunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Nest the uniform 3-level span tree under the open thread root (best-effort).

        Derives the per-scenario state from ``run_ctx`` (the run context): a ``None``
        ``run_ctx`` / ``span_slot`` means the sink is disabled for this run
        (``prepare_run`` returned ``None``), so this no-ops. The braintrust scenario
        anchor is the thread ROOT span ``prepare_run`` already opened with the
        scenario identity, so ``scenario_metadata`` is accepted for contract
        conformity but the anchor needs no further enrichment here. The SHARED Logger
        comes off the sink (``self._logger``) — it is not carried on the run context.
        Routed through :func:`swallow` so a sink failure can never abort a run.
        Children are added while the root is still open — the orchestrator calls this
        BEFORE ``post_feedback`` ends/flushes it.

        Async: the blocking span nesting runs inside via ``asyncio.to_thread`` so
        the orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, BraintrustRunContext) or run_ctx.span_slot is None:
            return
        await offload(
            record_transcript,
            run_ctx,
            self._logger,
            thread_id=thread_id,
            scenario_name=scenario_name,
            turns=turns,
            sink=self.name,
            what='record_transcript',
        )

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
        """Log per-policy + rollup scores onto the thread span; close + drain it.

        Async: the blocking metric-compute + ``span.log`` + flush run inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, BraintrustRunContext):
            return
        await asyncio.to_thread(
            self._post_feedback_sync,
            thread,
            finalize,
            run_ctx,
            final_world_state=final_world_state,
            tool_events=tool_events,
            scenario_metadata=scenario_metadata,
        )

    def _post_feedback_sync(
        self,
        thread: CreateThreadResponse,
        finalize: FinalizeResponse,
        run_ctx: BraintrustRunContext,
        *,
        final_world_state: dict[str, Any] | None = None,
        tool_events: dict[str, Any] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        thread_id = thread.thread_id
        policy_results = finalize.policy_evaluation_results or {}
        md = scenario_metadata
        if md is None:
            md = thread.scenario_metadata or {}
        scenario_name = run_ctx.scenario_name or md.get('scenario_name') or thread_id
        # Faithful Braintrust turn count = the outer-conversation turns recorded for this
        # scenario, stashed on the span slot by ``record_transcript`` (which always runs
        # FIRST) from the real ``turns`` list — the SAME count langsmith (``turn_runs``) and
        # langfuse (``replay['turns']``) report. It is NOT read off ``thread``: in production
        # ``thread`` is a ``CreateThreadResponse``, which carries no transcript, so
        # ``chatbot_messages_of(thread)`` would always be empty (turn_count stuck at 0).
        slot = run_ctx.span_slot
        turn_count = slot.turn_count if slot is not None else 0

        try:
            metrics = compute_rollup_metrics(
                policy_results=policy_results,
                final_world_state=final_world_state,
                tool_events=tool_events,
                chatbot_messages=chatbot_messages_of(thread),
                scenario_metadata=md,
                pass_rate=finalize.pass_rate,
                aggregate_score=float(finalize.aggregate_score),
            )
        except Exception:
            logger.exception('Observability rollup metric computation failed (scenario %s).', scenario_name)
            metrics = []

        swallow(
            self._post_feedback,
            run_ctx,
            policy_results=policy_results,
            metrics=metrics,
            final_world_state=final_world_state,
            scenario_metadata=md,
            turn_count=turn_count,
            sink=self.name,
            what='post_feedback',
        )

    def _post_feedback(
        self,
        run_ctx: BraintrustRunContext,
        *,
        policy_results: dict,
        metrics: list,
        final_world_state: dict[str, Any] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
        turn_count: int = 0,
    ) -> None:
        span_slot = run_ctx.span_slot
        if span_slot is None:
            return
        swallow = _braintrust_error_types()
        span = span_slot.span
        log = getattr(span, 'log', None)
        if log is not None:
            # Braintrust enforces that ``scores`` values are numbers in [0, 1]
            # (see braintrust.logger._validate_and_sanitize_experiment_log_partial_args:
            # a score > 1 raises ValueError and the whole row is
            # dropped). Per-policy results and the pass/fail rollup are genuine 0–1
            # scores; everything else goes to ``metrics`` — the count-style rollups
            # (tool-call / entity counts, the dimension-graph magnitudes) because they
            # routinely exceed 1, and ``scenario_status`` because its 0/1 is coverage,
            # not quality (see ``_SCORE_ROLLUP_KEYS``). Split by metric semantics
            # (not the runtime value) so the routing is deterministic regardless
            # of whether a given count happens to be 0 or 1 on a particular run.
            scores: dict = {}
            metric_values: dict = {}
            for policy_name, result in eval_results(policy_results).items():
                if isinstance(result, dict):
                    scores[policy_name] = score_from_result(result)
            # Score-type rollups (scenario_passed + the pass_rate / aggregate
            # quality signals — all in _SCORE_ROLLUP_KEYS) are [0, 1] by construction → ``scores``;
            # every other rollup is a raw count → ``metrics``. A score-type value that somehow falls
            # outside [0, 1] is defensively routed to ``metrics`` too: Braintrust DROPS the WHOLE
            # scores row for an out-of-range value, so this keeps it from taking the genuine 0–1
            # scores (per-policy + rollups) down with it.
            for m in metrics:
                if not isinstance(m.score, (int, float)) or isinstance(m.score, bool):
                    continue
                if m.key in _SCORE_ROLLUP_KEYS and 0.0 <= m.score <= 1.0:
                    scores[m.key] = float(m.score)
                else:
                    metric_values[m.key] = float(m.score)
            log_kwargs: dict = {}
            if scores:
                log_kwargs['scores'] = scores
            if metric_values:
                log_kwargs['metrics'] = metric_values
            # The per-scenario summary is the eval ``output`` — the same one-coherent-record
            # contract the langsmith / langfuse sinks set on their anchors:
            # {verdict, results, turn_count, persona, world_changes}. The final world-state
            # snapshot is NOT lost — ``build_scenario_summary`` folds it into ``world_changes``
            # (relocated into the structured summary). On the experiment root this is the
            # per-record experiment event's output column (alongside the dataset record's
            # ``expected``); on the Logger root it is a task-span output. The summary
            # is guaranteed non-empty (verdict / turn_count always present), so — matching the
            # other sinks — it is ALWAYS set.
            summary = build_scenario_summary(
                results=eval_results(policy_results),
                scenario_metadata=scenario_metadata,
                final_world_state=final_world_state,
                turn_count=turn_count,
                policy_results=policy_results,
            )
            log_kwargs['output'] = summary
            if log_kwargs:
                try:
                    log(**log_kwargs)
                except swallow:
                    logger.exception('Braintrust span.log(...) failed for thread %r.', run_ctx.thread_id)
        # Close + drain the thread span best-effort (idempotent in the reporter).
        # The SHARED Logger comes off the sink (self._logger) — not the run context.
        try:
            flush(run_ctx, self._logger)
        except swallow:
            logger.exception('Braintrust flush failed for thread %r.', run_ctx.thread_id)

    async def drain_run(self, run_ctx: RunContext) -> None:
        """Per-scenario teardown: end the thread root span + drain the bg logger (idempotent).

        Guaranteed by the orchestrator's scenario scope on exit (normal AND exception),
        so the eagerly-opened thread ROOT span (opened in ``prepare_run``) is ended +
        flushed even on the CRASH path where ``post_feedback`` never runs — without this
        a crashed scenario's root span would leak until the process-level ``close``.
        Idempotent: the module-level :func:`flush` guards the span ``end()`` with
        ``slot.ended``, so firing here after ``post_feedback`` already flushed on the
        happy path is a safe no-op on the span (a second ``bt_logger.flush()`` just
        drains an empty queue). No-op when ``run_ctx`` is ``None`` (sink disabled for
        this run). The SHARED Logger comes off the sink (``self._logger``).

        Async: the blocking :func:`flush` runs inside via ``asyncio.to_thread``, routed
        through the bare-``Exception`` :func:`~plurai.observability.errors.swallow` so
        that — unlike :func:`flush`'s own NARROW typed-except — even a non-transport SDK
        error during teardown can never escape into the orchestrator's scenario-scope
        ``finally`` and mask an in-flight exception. This never raises.
        """
        if not isinstance(run_ctx, BraintrustRunContext) or run_ctx.span_slot is None:
            return
        await offload(flush, run_ctx, self._logger, sink=self.name, what='drain_run')

    # -- OPTIONAL scenario-SET experiment lifecycle (ExperimentLifecycleSink seam) --

    def _lookup_record(self, scenario_folder: str) -> dict[str, Any] | None:
        """Resolve THIS scenario's fetched dataset record for the origin link (or None).

        The record index (built by :meth:`begin_experiment` from iterating the Dataset) is
        keyed by BOTH the upload-time ``scenario_folder`` metadata AND the record id — so
        the caller must pass the FOLDER, not the display ``scenario_name``; the two are NOT
        the same string. A miss is SILENT and degrades gracefully: the root span simply
        opens without its origin link, so the run looks fine while losing its dataset
        provenance — which is exactly why passing the wrong identity goes unnoticed.
        """
        return self._record_index.get(scenario_folder)

    async def begin_experiment(self, *, dataset_name: str, run_name: str) -> None:
        """Open the scenario-SET Experiment bound to the run's Dataset (best-effort, once).

        The :class:`~plurai.observability.sink_protocol.ExperimentLifecycleSink` seam the
        runner drives ONCE before the scenario gather. Creates ONE Braintrust Experiment
        (shared across the whole gather) bound to the Dataset named ``dataset_name`` and
        fetches the Dataset's records back so each per-scenario root span can carry the
        ``origin`` link (``id`` + server-assigned ``_xact_id`` are only available by
        FETCHING — ``insert()`` returns only the id). After this, ``prepare_run`` /
        ``_ensure_thread_span`` open each scenario's root on the Experiment (``type='eval'``)
        instead of the project-logs Logger. Best-effort: the blocking SDK work runs via
        ``asyncio.to_thread`` routed through the bare-``Exception``
        :func:`~plurai.observability.errors.swallow`, so a failure leaves the Experiment
        unset and the sink falls back to the Logger root — it NEVER raises.
        """
        await offload(
            self._begin_experiment_sync,
            dataset_name=dataset_name,
            run_name=run_name,
            sink=self.name,
            what='begin_experiment',
        )

    def _begin_experiment_sync(self, *, dataset_name: str, run_name: str) -> None:
        if not self._config.enabled or not _braintrust_available():
            return
        if self._logger is None:
            self.warm()
        from braintrust import init, init_dataset

        swallow = _braintrust_error_types()
        project = self._config.project or 'plurai'
        try:
            dataset = init_dataset(project=project, name=dataset_name, api_key=self._config.api_key)
        except swallow:
            logger.exception('Braintrust begin_experiment: init_dataset for %r failed.', dataset_name)
            return
        try:
            # set_current=False mirrors the Logger build: never mutate the SDK's global
            # current-experiment ContextVar (the per-scenario root holds the Experiment
            # explicitly via the run context).
            experiment = init(project=project, experiment=run_name, dataset=dataset, api_key=self._config.api_key, set_current=False)
        except swallow:
            logger.exception('Braintrust begin_experiment: init(experiment=%r) failed.', run_name)
            return
        # Fetch the Dataset's records back to populate the origin index. Separate guard:
        # a fetch failure must not discard the (usable) Experiment — origin just degrades.
        dataset_id: str | None = None
        # Value typed Any (not the SDK's DatasetEvent TypedDict): the records are opaque
        # fetched rows we only ever .get() from — consistent with how this file types every
        # other braintrust SDK object (span / logger / dataset) as Any.
        record_index: dict[str, Any] = {}
        try:
            dataset_id = dataset.id
            for record in dataset:
                metadata = record.get('metadata') or {}
                folder = metadata.get('scenario_folder')
                if folder:
                    record_index[str(folder)] = record
                record_id = record.get('id')
                if record_id:
                    record_index.setdefault(str(record_id), record)
        except swallow:
            logger.exception('Braintrust begin_experiment: dataset fetch for origin linkage failed (dataset %r).', dataset_name)
        self._experiment = experiment
        self._dataset = dataset
        self._dataset_id = dataset_id
        self._record_index = record_index

    async def end_experiment(self) -> None:
        """Summarize + flush the scenario-SET Experiment after the gather (best-effort, once).

        The :class:`~plurai.observability.sink_protocol.ExperimentLifecycleSink` counterpart
        the runner drives ONCE after every scenario finished (and after the opt-in dataset
        upload), while the sink is still open. ``Experiment.summarize()`` flushes pending
        spans, emits the experiment permalink, and triggers the comparison-vs-base. Resets
        the experiment state so a later run rebinds cleanly. Best-effort via
        ``asyncio.to_thread`` + bare-``Exception`` swallow — never raises.
        """
        await offload(self._end_experiment_sync, sink=self.name, what='end_experiment')

    def _end_experiment_sync(self) -> None:
        experiment = self._experiment
        if experiment is None:
            return
        swallow = _braintrust_error_types()
        try:
            experiment.summarize()  # flushes pending spans + prints the experiment URL
        except swallow:
            logger.exception('Braintrust end_experiment: summarize/flush failed.')
        finally:
            self._experiment = None
            self._dataset = None
            self._dataset_id = None
            self._record_index = {}

    async def close(self) -> None:
        """Module-level teardown drain for never-finalized threads (async, swallowed).

        Calls the module-level ``braintrust.flush()`` once the Logger has been built;
        a no-op when :meth:`warm` never ran. ``braintrust.flush()`` drains the SDK
        background queue — note that ``_build_logger`` uses ``set_current=False``, so
        ``self._logger`` is NOT the SDK global current logger; whether ``flush()``
        drains the same instance depends on the SDK's internal queuing.
        Per-scenario draining is coupled into ``post_feedback`` — this is the
        once-per-process teardown for threads that never reached ``post_feedback``.
        ``braintrust.flush()`` is a blocking sync SDK drain, so it runs in a worker
        thread via ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        if self._logger is None:
            return
        try:
            from braintrust import flush as bt_flush

            await asyncio.to_thread(bt_flush)  # module-level flush; drains SDK background queue
        except _braintrust_error_types():
            logger.exception('Braintrust teardown flush failed.')

    def sync_scenarios(
        self,
        *,
        dataset_name: str,
        dataset_description: str | None,
        scenarios: list[ScenarioUploadEntry],
        existing_metadata: dict[str, Any],
    ) -> ProviderSyncResult:
        """Upload a generated scenario batch into a Braintrust Dataset.

        Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`;
        a disabled/SDK-missing no-op returns ``ok=True, uploaded=0``.
        """
        return sync_scenarios_to_braintrust(
            self._config,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            scenarios=scenarios,
            existing_items=existing_metadata.get('braintrust_dataset_items'),
        )
