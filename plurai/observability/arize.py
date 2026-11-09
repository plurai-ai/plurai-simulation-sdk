"""Client-side Arize AX integration — OTel / OpenInference observability sink."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from plurai.content_parts import content_text
from plurai.observability.sink_protocol import RunContext, ScenarioResult, TranscriptTurn
from plurai.observability.dataset_types import ProviderSyncResult
from plurai.observability.errors import offload, swallowable_error_types
from plurai.observability.eval_mapping import SCENARIO_PASSED, turn_record_name
from plurai.observability.rollup import (
    chatbot_messages_of,
    compute_rollup_metrics,
    eval_results,
    score_from_result,
)
from plurai.orchestrator.models import CreateThreadResponse, FinalizeResponse, ScenarioUploadEntry

if TYPE_CHECKING:
    # Type-only imports of optional-extra SDK symbols (the `arize` extra + its OTel/
    # OpenInference deps). Guarded by TYPE_CHECKING + ``from __future__ import annotations``
    # so every annotation is a string at runtime and never forces the optional SDK to
    # import when the extra is absent (keeps ``import plurai`` working without ``[arize]``).
    import pandas as pd
    from arize.datasets.types import Dataset
    from openinference.instrumentation import TracerProvider


logger = logging.getLogger(__name__)


class EvalCell(TypedDict):
    """One eval entry produced by :func:`_build_experiment_output` and consumed by
    :class:`~plurai.observability.arize_experiments.PassthroughEvaluator`.

    ``score`` is always a finite ``float``; ``label`` and ``explanation`` are
    non-empty strings when present (``None`` is allowed at the producer, but
    ``PassthroughEvaluator`` coerces absent/empty to ``'absent'``).
    """

    score: float
    label: str | None
    explanation: str | None


@dataclass(slots=True)
class ArizeRunContext:
    """Arize per-scenario run context — the single object ``prepare_run`` returns."""

    provider: TracerProvider | None = None
    anchor_span_id: str | None = None
    trace_id: str | None = None

    @property
    def enabled(self) -> bool:
        """True when Arize tracing is LIVE for this scenario — exposes the EXISTING gate.

        ``record_transcript`` / ``post_feedback`` already short-circuit on
        ``run_ctx.provider is None`` (the provider is ``None`` when the sink is
        disabled or :meth:`ArizeSink.initialize` never ran). This property simply
        surfaces that existing decision uniformly; it does not change what gets traced.
        """
        return self.provider is not None


class ArizeSinkConfig(BaseModel):
    """Arize AX observability sink config (mirrors the Plurai backend's Arize settings)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    space_id: str | None = None
    api_key: str | None = None
    project_name: str | None = None
    endpoint: str | None = None

    @field_validator('space_id', 'api_key', 'project_name', 'endpoint', mode='before')
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        """Coerce an empty / whitespace-only credential string to ``None``."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode='after')
    def _require_creds_when_enabled(self) -> ArizeSinkConfig:
        """When the sink is ACTIVE, require ``space_id`` / ``api_key`` / ``endpoint``."""
        if self.enabled:
            missing = [name for name in ('space_id', 'api_key', 'endpoint') if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f'Arize is enabled but required credential field(s) are missing: {", ".join(missing)}. '
                    'Set ARIZE__SPACE_ID, ARIZE__API_KEY, and ARIZE__ENDPOINT (nested) when ARIZE__ENABLED is set.'
                )
        return self


# PROCESS-WIDE SIDE EFFECT (documented): register() builds an OTel tracer provider
# and installs it as the global OTel provider.


def _arize_error_types() -> tuple[type[BaseException], ...]:
    """Return the tracing-export exception classes worth swallowing.

    The default OTLP transport is gRPC (``arize-otel`` ``Transport.GRPC``), so
    ``grpc.RpcError`` covers it; ``requests.exceptions.RequestException`` covers the
    HTTP/HTTPS transports. ``SpanExportResult`` is an ENUM, not an exception — it does
    not belong here. The helper's ``(OSError,)`` fallback covers raw socket errors.
    Any other exception (TypeError / programmer bug) propagates.
    """
    return swallowable_error_types(
        ('grpc', 'RpcError'),
        ('requests.exceptions', 'RequestException'),
    )


def _arize_dataset_error_types() -> tuple[type[BaseException], ...]:
    """Return the DATASETS-client exception classes worth swallowing.

    Distinct from :func:`_arize_error_types` on purpose: the v8 datasets client is REST
    over urllib3 (``arize/_generated/api_client/rest.py``), NOT the gRPC/requests stack
    the tracing exporter uses. ``OpenApiException`` is the generated client's base class
    (``ApiException`` and friends subclass it). ``MissingAPIKeyError`` and ``ValueError``
    (e.g. empty dataset) are config/programmer errors and deliberately NOT swallowed.
    """
    return swallowable_error_types(
        ('arize._generated.api_client.exceptions', 'OpenApiException'),
        ('urllib3.exceptions', 'HTTPError'),
    )


def _arize_dataset_not_found_types() -> tuple[type[BaseException], ...]:
    """Return the dataset NOT-FOUND exception classes (the expected first-run signal).

    Get-by-NAME resolves through ``_find_dataset_id``, which raises
    ``arize.utils.resolve.NotFoundError`` when the name is absent; a get-by-ID 404 would
    raise the generated client's ``NotFoundException`` instead — catch both. Kept
    SEPARATE from :func:`_arize_dataset_error_types` because dataset-absent means
    "proceed to create, skip delete", while a transport error is logged-but-proceeds.
    Empty fallback: when the SDK is importable these resolve. Rename asymmetry: a future
    rename of ``NotFoundException`` degrades to the transport tuple (it subclasses
    ``OpenApiException``), but ``NotFoundError`` subclasses plain ``Exception`` — a
    rename there would PROPAGATE on a first-run not-found, not degrade.
    """
    return swallowable_error_types(
        ('arize.utils.resolve', 'NotFoundError'),
        ('arize._generated.api_client.exceptions', 'NotFoundException'),
        fallback=(),
    )


def _safe_eval_name(name: str) -> str:
    """Sanitize an eval name to Arize's ``eval.<name>.*`` column grammar.

    Arize's eval-column name regex is ``[A-Za-z0-9_\\s]+`` (``arize/spans/columns.py``):
    alphanumeric, underscore, and whitespace only — hyphens and other punctuation are
    rejected by ``validate=True``. Policy names can carry hyphens/punctuation, so replace
    every disallowed character with ``_`` and collapse runs into a single ``_``. Falls
    back to ``'eval'`` for a name that sanitizes to empty so the column is never the bare
    ``eval..score``.
    """
    cleaned = re.sub(r'[^A-Za-z0-9_\s]+', '_', name).strip()
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned or 'eval'


def _eval_column_name(raw: str, used: set[str]) -> str:
    """Sanitize *raw* and return a collision-free eval column name.

    Composes :func:`_safe_eval_name` with suffix dedup (``name``, ``name_2``,
    ``name_3``, …), recording the choice in *used* so two original names that
    collide after sanitization get distinct eval columns rather than silently
    overwriting each other. Shared by the experiment-output and live span-eval
    channels so their collision handling stays identical. (The evaluator SET
    deliberately uses bare :func:`_safe_eval_name` instead — there, collisions
    SHARE one evaluator.)
    """
    sanitized = _safe_eval_name(raw)
    if sanitized not in used:
        used.add(sanitized)
        return sanitized
    counter = 2
    while (candidate := f'{sanitized}_{counter}') in used:
        counter += 1
    used.add(candidate)
    return candidate


def _policy_eval_rows(policy_results: dict[str, Any], used: set[str]) -> Iterator[tuple[str, float, str | None, str | None]]:
    """Yield one ``(column_name, score, label, explanation)`` row per policy result.

    The SHARED per-policy pipeline for both eval channels (the experiment task
    output and the live span-eval row), so their name sanitization/dedup, label
    fallback (own label, else ``violated`` → ok/violated/unknown) and the
    100/10_000-char clamps stay identical by construction. ``label`` /
    ``explanation`` are ``None`` when absent — each channel applies its own
    presence rule.
    """
    for name, presult in eval_results(policy_results).items():
        if not isinstance(presult, dict):
            continue
        n = _eval_column_name(name, used)
        label = presult.get('label')
        if label is None:
            violated = presult.get('violated')
            label = 'violated' if violated is True else 'ok' if violated is False else 'unknown'
        reasoning = presult.get('reasoning')
        yield (
            n,
            score_from_result(presult),
            str(label).strip()[:100] or None,
            str(reasoning)[:10_000] if reasoning else None,
        )


#: The per-turn judge's source pool the Plurai backend stores beside the loaded sources (``source_grounding.reserve`` /
#: ``unit_order``, written only when the loader left units out). It can run to ~100 KB per scenario and is judge
#: input, not scenario description, so the anchor span never carries it.
_JUDGE_POOL_KEYS = ('reserve', 'unit_order')


def _without_judge_pool(metadata: dict[str, Any]) -> dict[str, Any]:
    grounding = metadata.get('source_grounding')
    if not isinstance(grounding, dict) or not any(k in grounding for k in _JUDGE_POOL_KEYS):
        return metadata
    return {**metadata, 'source_grounding': {k: v for k, v in grounding.items() if k not in _JUDGE_POOL_KEYS}}


def _build_experiment_output(result: ScenarioResult, *, scenario_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the JSON-serializable task output stored on the experiment run.

    Shape: ``{status, pass_rate, aggregate_score, final_world_state, evals: {name: {score,
    label, explanation}}}``. ``pass_rate`` is the headline quality signal; ``aggregate_score``
    is the secondary reliability number. The ``evals`` map carries one entry per policy result
    (sanitized name) plus one per rollup metric ``compute_rollup_metrics`` EMITTED, which is a
    variable set: ``scenario_status`` (always, the categorical verdict word) and
    ``aggregate`` (= ``finalize.aggregate_score``, secondary) are unconditional, while the two
    verdict axes ``scenario_passed`` and ``pass_rate`` are absent on any row the backend did not
    evaluate. Their COLUMNS are declared regardless (``arize_experiments.VERDICT_ROLLUP_NAMES``),
    so such a row reads ``label='absent'`` rather than losing the column. ``PassthroughEvaluator``
    (:mod:`plurai.observability.arize_experiments`)
    reads ``output['evals'][name]`` to surface each as an AX evaluator column. The
    per-policy rows come from :func:`_policy_eval_rows` — the SAME pipeline the live
    span-eval channel (:meth:`_post_feedback_sync`) consumes — so the experiment
    columns match it by construction, including the collision dedup (``_2``, ``_3``, …).
    """
    finalize = result.finalize
    # Wire model degrades default_factory dict/list fields to ``| None``; guard at read.
    policy_results = finalize.policy_evaluation_results or {}
    final_world_state = finalize.final_world_state or {}
    metrics = compute_rollup_metrics(
        policy_results=policy_results,
        final_world_state=final_world_state,
        tool_events=result.tool_events,
        chatbot_messages=chatbot_messages_of(result),
        scenario_metadata=scenario_metadata,
        pass_rate=finalize.pass_rate,
        aggregate_score=float(finalize.aggregate_score),
    )

    evals: dict[str, EvalCell] = {}
    used_eval_names: set[str] = set()

    # Per-policy evals (shared pipeline; this channel stores None for absent label/explanation).
    for n, score, label, explanation in _policy_eval_rows(policy_results, used_eval_names):
        evals[n] = EvalCell(score=score, label=label, explanation=explanation)

    # Rollup metrics: the score carries the eval; the comment becomes the explanation. A
    # rollup MUST carry a NON-NULL string label too — experiments.run materializes the
    # eval.<name>.label column unconditionally, and an all-null column is rejected by the
    # Flight server. scenario_passed -> passed/failed; the count/score rollups ->
    # str(value) (truncated). The explanation falls back to the label when there is no
    # comment, so that column is never all-null either. The branch key is the SHARED
    # eval_mapping routing-key constant — never a string literal — so this consumer cannot
    # drift from the key rollup.compute_rollup_metrics actually emits.
    for m in metrics:
        n = _eval_column_name(m.key, used_eval_names)
        if m.key == SCENARIO_PASSED:
            rollup_label: str = 'passed' if m.score >= 1.0 else 'failed'
        else:
            rollup_label = str(m.value)[:100]
        evals[n] = EvalCell(
            score=float(m.score),
            label=rollup_label,
            explanation=str(m.comment)[:10_000] if m.comment else rollup_label,
        )

    return {
        'status': finalize.status,
        # pass_rate headlines; aggregate_score is the secondary reliability number.
        # NULLABLE, verbatim: the backend returns null for a scenario it scored nothing on, and
        # coercing that to 0.0 would publish a full red on the headline quality signal. This is
        # the task OUTPUT payload (a JSON blob), not a Flight column, so a null is representable.
        # `_failed_experiment_output` keeps its 0.0 deliberately: that row is a scenario that
        # never ran, and its `evals` map is empty, so nothing reads the field as a measurement.
        'pass_rate': finalize.pass_rate,
        'aggregate_score': finalize.aggregate_score,
        'final_world_state': final_world_state,
        'evals': evals,
    }


def _resolve_scenario_id(dataset_row: dict[str, Any] | None, metadata: dict[str, Any] | None) -> str | None:
    """Recover the scenario id (= folder name) from an experiment example.

    The dataset-upload path (``sync_scenarios_to_arize``) stamps ``scenario_id`` as a
    TOP-LEVEL example field, so it surfaces directly in each ``dataset_row``; fall back
    to the ``scenario_folder`` the upload also keeps in the nested metadata map (the
    row's ``metadata`` map or the separately-bound ``metadata`` arg). ``None`` when
    nothing resolves — the task then returns a failed-output row.
    """
    if isinstance(dataset_row, dict):
        sid = dataset_row.get('scenario_id')
        if sid:
            return str(sid)
        row_meta = dataset_row.get('metadata') or dataset_row.get('attributes.metadata')
        if isinstance(row_meta, dict) and row_meta.get('scenario_folder'):
            return str(row_meta['scenario_folder'])
    if isinstance(metadata, dict) and metadata.get('scenario_folder'):
        return str(metadata['scenario_folder'])
    return None


def _failed_experiment_output(error: str, *, scenario_id: str | None = None) -> dict[str, Any]:
    """A JSON-serializable FAILED task output, shaped like :func:`_build_experiment_output`.

    Returned in place of a real output when a scenario can't run: ``experiments.run`` DROPS
    a row whose task returns ``None`` (``functions.py``), so a failed scenario must still
    return a dict. ``evals`` is empty, so every passthrough evaluator reads ``label='absent'``
    for this row; ``status='failed'`` makes it countable by :func:`_experiment_failure_messages`.
    """
    output: dict[str, Any] = {
        'status': 'failed',
        'pass_rate': 0.0,
        'aggregate_score': 0.0,
        'final_world_state': {},
        'evals': {},
        'error': error,
    }
    if scenario_id is not None:
        output['scenario_id'] = scenario_id
    return output


def _skipped_experiment_output(scenario_id: str) -> dict[str, Any]:
    """A JSON-serializable SKIPPED task output for a dataset row outside this run's scope.

    The dataset accumulates across scoped runs (the upload is idempotent by folder), so
    ``experiments.run`` hands the task rows an earlier run uploaded. Such a row must not
    run, and must not be dropped either (a ``None`` task result is): ``status='skipped'``
    keeps it, ``evals`` is empty (every passthrough evaluator reads ``label='absent'``) and
    :func:`_experiment_failure_messages` does not count it.
    """
    return {
        'status': 'skipped',
        'pass_rate': None,
        'aggregate_score': None,
        'final_world_state': {},
        'evals': {},
        'scenario_id': scenario_id,
    }


def _coerce_output(output: Any) -> dict[str, Any] | None:
    """Normalize an ``experiments.run`` results-frame ``output`` cell to a dict.

    The cell is either the dict our task returned OR — after ``experiments.run`` serializes
    the results for the Flight upload — a JSON STRING of that dict (the round-tripped shape
    the live frame actually carries). Returns the dict, or ``None`` when the cell is neither
    a dict nor a JSON-object string (a genuinely malformed / dropped output).
    """
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _experiment_failure_messages(results_df: pd.DataFrame) -> list[str] | None:
    """One pass over the ``experiments.run`` results frame — one message per failed row.

    A row fails when its task errored (``error`` column carries a string) or its output
    (a dict OR its round-tripped JSON STRING) does not report ``status == 'completed'``
    (the FAILED-output dict or a finalize-failed run); a ``'skipped'`` row (outside this
    run's scope, :func:`_skipped_experiment_output`) is not a failure. The messages surface WHICH
    scenarios failed and WHY — ``experiments.run`` swallows per-task failures into the
    frame (the full traceback is also logged at the task site via ``logger.exception``).
    Defensive about the frame shape: ``None`` when the ``output`` column is
    missing/unreadable (the run itself returned, so the caller counts 0 failures rather
    than over-reporting).
    """
    try:
        outputs = list(results_df['output'])
    except (KeyError, TypeError):
        return None
    try:
        errors = list(results_df['error'])
    except (KeyError, TypeError):
        errors = [None] * len(outputs)
    messages: list[str] = []
    for output, error in zip(outputs, errors, strict=True):
        parsed = _coerce_output(output)
        errored = isinstance(error, str) and bool(error)
        if not errored and parsed is not None and parsed.get('status') in ('completed', 'skipped'):
            continue
        if parsed is not None and parsed.get('status') == 'failed':
            messages.append(f'scenario {parsed.get("scenario_id", "?")} failed: {parsed.get("error", "no error detail")}')
        elif errored:
            messages.append(f'task run error: {error}')
        else:
            messages.append(f'unexpected non-completed task output: {output!r}')
    return messages


def sync_scenarios_to_arize(
    config: ArizeSinkConfig,
    *,
    dataset_name: str,
    dataset_description: str | None,
    scenarios: list[dict[str, Any]],
    existing_dataset_id: str | None = None,
) -> ProviderSyncResult:
    """Upsert a scenario batch into an Arize AX dataset."""
    if not config.enabled:
        return ProviderSyncResult(provider='arize', metadata_updates={}, summary_line='Arize: disabled or SDK not installed — nothing uploaded.')
    if not scenarios:
        # The v8 client raises ValueError('Cannot create an empty dataset') client-side;
        # an empty batch is a no-op here, not an error worth crashing the CLI over.
        logger.warning(f'No scenarios to sync to Arize dataset {dataset_name!r}; skipping upload.')
        return ProviderSyncResult(provider='arize', metadata_updates={}, summary_line=f'Arize: no scenarios to upload to dataset {dataset_name!r}.')
    from arize import ArizeClient

    def _fail(detail: str, *errors: str) -> ProviderSyncResult:
        """Log + build the FAILED sync result (the summary mirrors the log line)."""
        logger.error(f'Arize {detail}')
        return ProviderSyncResult(
            provider='arize',
            metadata_updates={},
            summary_line=f'Arize: {detail}',
            ok=False,
            errors=errors or (detail,),
        )

    swallow = _arize_dataset_error_types()
    not_found = _arize_dataset_not_found_types()

    # space_id is required by every call below (get/delete/create scope to it) — check
    # BEFORE touching the prior dataset so a broken config never deletes anything.
    space = config.space_id
    if not space:
        return _fail(f'dataset sync for {dataset_name!r} failed: space_id is required but missing from config.', 'missing space_id in config')
    client = ArizeClient(api_key=config.api_key)

    if existing_dataset_id:
        logger.info(f'Prior Arize dataset id {existing_dataset_id!r} for {dataset_name!r} will rotate (delete-then-recreate).')

    # 1) Find the prior dataset by name. Dataset-ABSENT is the EXPECTED first-run signal
    #    (proceed to create, skip delete) and is DISTINCT from a transport error on get,
    #    which is logged but must not abort the sync.
    # ``None`` here is LOAD-BEARING (not the prepare_run disabled-sentinel case): it means
    # the dataset doesn't exist yet → nothing to delete, proceed to create.
    existing: Dataset | None = None
    try:
        existing = client.datasets.get(dataset=dataset_name, space=space)
    except not_found:
        existing = None  # expected first-run path
    except swallow as exc:
        logger.warning(f'Arize dataset get for {dataset_name!r} failed (transport): {exc}; proceeding to create.')
    if existing is not None:
        try:
            client.datasets.delete(dataset=dataset_name, space=space)
        except (*not_found, *swallow) as exc:
            logger.warning(f'Arize dataset delete for {dataset_name!r} failed: {exc}; create may collide on the unique-name constraint.')

    # 2) Build dict examples (NOT a DataFrame — no hard pandas import; the v8 API accepts
    #    arbitrary user-defined JSON fields per example). Use the nested
    #    (input / expected_output / metadata) shape — do NOT spread scen['inputs']
    #    into top-level keys. System-managed keys (id/created_at/updated_at) must
    #    not appear.
    # ``scenario_id`` (= ``folder_name``) is stamped as a TOP-LEVEL example field
    # (the v8 dataset API turns it into its own column), so it survives into every example's
    # ``dataset_row`` — the experiment task resolves the scenario identity from there (see
    # ``run_experiment_set``), not only from ``metadata``. The scenario-SET id is NOT stamped
    # here: the run path already carries it via the CLI args (``create_thread``), so the
    # example only needs the per-scenario id.
    examples: list[dict[str, Any]] = [
        {
            'scenario_id': scen['folder_name'],
            'input': scen['inputs'],
            'expected_output': scen.get('outputs') or {},
            'metadata': {**scen.get('metadata', {}), 'scenario_folder': scen['folder_name']},
        }
        for scen in scenarios
    ]
    # 3) Create fresh from the full batch.
    try:
        dataset = client.datasets.create(name=dataset_name, space=space, examples=examples)
    except (*not_found, *swallow) as exc:
        # Create transport failure is a genuine FAILURE, not a no-op — the prior
        # dataset was already deleted at step (1), so there is nothing to fall back to.
        return _fail(f'dataset create for {dataset_name!r} failed: {exc}', f'dataset create: {exc}')

    # ``datasets.create`` returns a ``Dataset`` whose ``id`` is typed ``str`` (StrictStr) —
    # typed access, no getattr. The blank guard stays: ``str`` permits ``''``, so an
    # empty id is still possible (and, after the prior dataset was deleted, unrecoverable).
    dataset_id = dataset.id
    if not dataset_id:
        # A blank handle after the prior dataset was already deleted at step (1) is
        # UNRECOVERABLE: the old dataset is gone and we have no id for the new one.
        # Treat it as a FAILURE (do NOT return a success tuple with an empty id).
        return _fail(
            f'dataset create for {dataset_name!r} returned a blank dataset id (prior dataset already deleted — unrecoverable).',
            'dataset create returned a blank/missing dataset id',
        )
    return ProviderSyncResult(
        provider='arize',
        metadata_updates={
            'arize_dataset_id': dataset_id,
            'arize_dataset_name': dataset_name,
        },
        summary_line=f'Arize: synced {len(scenarios)} example(s) to dataset {dataset_name!r} (id={dataset_id}, recreated).',
        uploaded=len(scenarios),
    )


# ---------------------------------------------------------------------------
# ArizeSink — the ObservabilitySink instance for the OTel/OpenInference sink.
# The process-global tracer provider + the warn-once flag live as INSTANCE state
# (``_provider`` / ``_warned``); there are NO module-level mutable globals. The OTel
# register() global-provider install is process-wide BY NATURE — the instance owns the
# flag and the module docstring documents it.
# ---------------------------------------------------------------------------


class ArizeSink:
    """Arize AX sink"""

    name = 'arize'

    def __init__(self, config: ArizeSinkConfig) -> None:
        self._config = config
        self._provider: TracerProvider | None = None  # the OTel tracer provider (process-global side effect)
        self._warned = False  # dedups the bind-without-init warning
        # Guards the lazy None-check-then-install in initialize() so the PROCESS-GLOBAL
        # OTel tracer provider is installed exactly once even if the lazy path ever runs
        # under asyncio.to_thread (a concurrent double-install would leak/replace it).
        self._init_lock = threading.Lock()
        # True while ``run_experiment_set`` is driving the experiments.run path. The
        # experiment posts per-policy + rollup evals via passthrough EVALUATORS (the
        # task output), so ``post_feedback`` MUST skip its span-eval channel
        # (``spans.update_evaluations``) while this is set — otherwise every eval is
        # double-emitted.
        self._experiment_active = False

    async def warm_once(self) -> None:
        """Install the OTel tracer provider ONCE, on the event-loop thread (idempotent).

        Called at the entry point BEFORE any gather, so the process-global provider
        install happens single-threaded (no race). Idempotent: :meth:`initialize` only
        installs when ``self._provider`` is still ``None`` (double-checked under the
        init lock), so a second call — or a later lazy ``prepare_run`` self-init — is a
        cheap no-op. Best-effort: never raises.

        Emits a ONE-TIME startup notice when the sink is enabled but ``project_name`` is
        not configured: spans will be recorded but per-policy/rollup evaluations will NOT
        be attached (``update_evaluations`` requires a project name to scope the
        server-side span-id join). This is a static config gap, not a per-scenario
        transient, so the notice fires here (at most once) rather than per scenario.
        """
        if self._provider is not None:
            return
        await offload(self.initialize, sink=self.name, what='warm_once')
        if self._config.enabled and self._config.project_name is None:
            logger.warning(
                'Arize is enabled but ARIZE__PROJECT_NAME is not set. '
                'Spans will be recorded but per-policy and rollup evaluations will NOT be '
                'attached to those spans. Set ARIZE__PROJECT_NAME (matching the project name '
                'passed to register()) to enable eval attachment via update_evaluations.'
            )

    def initialize(self, *, project_name: str | None = None) -> None:
        """Process-scoped init: register the OTel tracer provider (idempotent + locked).

        Installs the PROCESS-GLOBAL OTel tracer provider exactly once. No-op when
        disabled or when the provider is already installed. FAIL-LOUD on a missing dep:
        when the tracing packages (``arize-otel`` / ``openinference``) aren't installed
        the lazy ``from arize.otel import register`` raises ``ImportError`` — there is no
        silent availability guard; the entry-point ``warm_once`` runs this under
        best-effort ``offload`` so a misconfigured run degrades rather than crashing. The
        provider's project comes from ``project_name`` or
        ``config.project_name``; when BOTH are ``None`` the kwarg is omitted so
        ``register()`` resolves its native default.

        Thread-safe: the None-check-then-install is guarded by ``self._init_lock``
        with double-checked locking, so even if the lazy ``prepare_run`` self-init ever
        runs under ``asyncio.to_thread`` across N scenario threads, the global-provider
        install happens exactly once (a concurrent double-install would leak/replace
        the process-global provider). Belt and suspenders: ``warm_once`` at the entry
        point installs it single-threaded BEFORE the gather, and this lock backstops
        the lazy path.

        It deliberately does NOT instrument LangChain: the SDK captures no white-box
        agent telemetry — the uniform transcript tree is emitted explicitly by
        :meth:`record_transcript`, so a process-global LangChain monkeypatch would add
        exactly the non-conformant agent-internals tier this sink must not emit.
        """
        if not self._config.enabled:
            return
        # Double-checked locking: the first (lock-free) read short-circuits the common
        # already-installed path; the install itself is serialized under the lock and
        # re-checks so only the first caller through builds the global provider.
        if self._provider is not None:
            return
        with self._init_lock:
            if self._provider is not None:
                return
            from arize.otel import register

            kwargs: dict[str, Any] = {
                # Arize installs itself as the global OTel tracer provider
                # (set_global_tracer_provider defaults True in register()).
                # register() prints a banner to stdout by default; keep the console quiet.
                'verbose': False,
            }
            provider_project = project_name or self._config.project_name
            if provider_project is not None:
                kwargs['project_name'] = provider_project
            # The config-construction validator (_require_creds_when_enabled) GUARANTEES
            # space_id / api_key / endpoint are non-None whenever the sink is enabled,
            # and this method early-returns when disabled — so by here all three are set
            # and pass straight through (no omit-on-None fallback needed).
            kwargs['space_id'] = self._config.space_id
            kwargs['api_key'] = self._config.api_key
            kwargs['endpoint'] = self._config.endpoint
            self._provider = register(**kwargs)

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
    ) -> ArizeRunContext:
        """Register the OTel provider once (if needed), then return the run context.

        The uniform ``prepare_run`` kwargs are accepted for protocol parity but Arize
        needs none of them here: the provider's project is set in :meth:`initialize`,
        and ``record_transcript`` receives ``thread_id`` / ``scenario_name`` directly
        (``agent_version`` record tagging is a record-sink concern — LangSmith/Langfuse).

        Async: the blocking ``register()`` provider build runs inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly. Best-effort: a
        swallowed failure yields a disabled ``ArizeRunContext()`` (``provider=None`` ⇒
        ``enabled=False``) rather than ``None`` — prepare_run never returns ``None``.
        """
        return await offload(self._prepare_run, sink=self.name, what='prepare_run') or ArizeRunContext()

    def _prepare_run(self) -> ArizeRunContext:
        """Register the OTel provider once (if needed) and bind the run context.

        ``ArizeRunContext.provider`` is exactly ``self._provider`` (process-global
        instance state) when the sink is enabled AND initialized — ``record_transcript``
        reads it back to ``get_tracer``, so the value is load-bearing. It is ``None`` when
        disabled or when ``initialize()`` could not build a provider (arize-otel absent /
        ``register()`` failed); in the latter case emit the ONE warn-once notice.
        """
        if self._provider is None:
            self.initialize()
        provider: TracerProvider | None = None
        if self._config.enabled:
            if self._provider is not None:
                provider = self._provider
            elif not self._warned:
                logger.warning(
                    'Arize enabled but no tracer provider is initialized — call '
                    'initialize() once at your process entry point (it builds the provider once). '
                    'Falling back to no Arize observability for this run.'
                )
                self._warned = True
        return ArizeRunContext(provider=provider)

    async def record_transcript(
        self,
        run_ctx: RunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit the uniform 3-level OTEL span tree (scenario -> turn -> {agent, client}).

        Builds the tree with EXPLICIT-parent OTEL spans only — NEVER
        ``start_as_current_span`` — so interleaved ``asyncio.gather`` scenarios never
        cross-parent through the ambient context. The scenario anchor is a
        deterministic trace ROOT (an empty ``Context()`` guarantees no parent); each
        turn / agent / client is parented by passing
        ``context=set_span_in_context(parent)``. All state (the tracer provider) comes
        from ``run_ctx.provider`` only — the REAL ``register()`` provider, never the OTel
        global no-op. OpenInference attrs (``input.value`` / ``output.value`` /
        ``session.id`` / ``openinference.span.kind``) make the spans surface in Arize.
        Best-effort via ``swallow``. There is NO separate orchestrator flush step:
        the per-scenario drain is COUPLED in here — ``_record_transcript``
        ``force_flush()``-es the buffered (BatchSpanProcessor) tracer provider at its
        END so each scenario's spans surface promptly. Returns ``None``: the anchor it
        builds is captured onto ``run_ctx`` (the 16-hex ``anchor_span_id`` + 32-hex
        ``trace_id``) so ``post_feedback`` keys evals from the context, not from a return.

        Async: the blocking span emit + terminal ``force_flush`` run inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, ArizeRunContext) or run_ctx.provider is None:
            return
        provider = run_ctx.provider
        anchor = await offload(
            self._record_transcript,
            provider,
            thread_id=thread_id,
            scenario_name=scenario_name,
            turns=turns,
            scenario_metadata=scenario_metadata,
            sink=self.name,
            what='record_transcript',
        )
        # Capture the anchor span's 16-hex OTEL id onto the run context so post_feedback
        # (handed the SAME run_ctx) keys evals to it via spans.update_evaluations.
        # record_transcript runs before post_feedback, so it is set by the time evals
        # post. Guard the get_span_context() read so a swallowed/None anchor leaves
        # anchor_span_id None (post_feedback then no-ops) rather than raising. run_ctx is
        # non-None here (guarded at the method top).
        if anchor is not None:
            try:
                span_context = anchor.get_span_context()
                run_ctx.anchor_span_id = format(span_context.span_id, '016x')
                # Also capture the 32-hex TRACE id so post_feedback can stamp
                # ``result.trace.id`` on the offline experiment-run row (cross-linking the
                # row back to this live OTLP trace). Distinct from the span id above.
                run_ctx.trace_id = format(span_context.trace_id, '032x')
            except Exception:
                logger.exception('Arize anchor span/trace-id capture failed; evals will be skipped for this scenario.')

    def _record_transcript(
        self,
        provider: TracerProvider,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None,
    ) -> Any | None:
        from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes  # local import (arize extra)
        from opentelemetry.context import Context, get_current  # local import (arize extra)
        from opentelemetry.trace import get_current_span, set_span_in_context  # local import (arize extra)

        tracer = provider.get_tracer('plurai.observability.arize')
        kind = SpanAttributes.OPENINFERENCE_SPAN_KIND
        chain = OpenInferenceSpanKindValues.CHAIN.value
        agent_kind = OpenInferenceSpanKindValues.AGENT.value
        client_kind = OpenInferenceSpanKindValues.LLM.value

        anchor_attrs: dict[str, Any] = {
            kind: chain,
            SpanAttributes.SESSION_ID: thread_id,
            'plurai.thread_id': thread_id,
            'plurai.scenario': scenario_name,
        }
        if scenario_metadata:
            anchor_attrs[SpanAttributes.METADATA] = json.dumps(_without_judge_pool(scenario_metadata), default=str)

        # Anchor PARENT selection (the deep link). When this runs inside an
        # ``experiments.run`` task, that task span is the AMBIENT OTel span — parent the
        # anchor under it so the AX UI deep-links the experiment run to this full
        # transcript trace (each parallel task carries its OWN ambient span via
        # contextvars, so siblings never cross-parent). When NO active span is present
        # (a trace-only run) fall back to an empty ``Context()``, a deterministic
        # fresh-trace root (no parent). Either way the anchor is started
        # via ``start_span`` (never ``start_as_current_span``), so it never leaks into a
        # sibling task's ambient context.
        current_context = get_current()
        anchor_context = current_context if get_current_span(current_context).get_span_context().is_valid else Context()
        scenario = tracer.start_span(name=scenario_name, context=anchor_context, attributes=anchor_attrs)
        try:
            scenario_ctx = set_span_in_context(scenario)
            for turn in turns:
                turn_span = tracer.start_span(
                    name=turn_record_name(turn['turn_idx']),
                    context=scenario_ctx,
                    attributes={
                        kind: chain,
                        SpanAttributes.SESSION_ID: thread_id,
                        'plurai.turn_idx': turn['turn_idx'],
                        'plurai.should_terminate': turn['should_terminate'],
                        'plurai.turn_count': turn['turn_count'],
                    },
                )
                try:
                    turn_ctx = set_span_in_context(turn_span)
                    tracer.start_span(
                        name='agent',
                        context=turn_ctx,
                        attributes={
                            kind: agent_kind,
                            SpanAttributes.SESSION_ID: thread_id,
                            # OMITTED on an agent-first turn, where there is no user input
                            # (``input_user_message`` is ``None``). An OTel attribute value must be
                            # a primitive or a homogeneous sequence of primitives, so a None would
                            # be dropped by the SDK with an "Invalid type NoneType" log per turn;
                            # the other sinks carry it as a JSON null. Not spelled ``or ''``: that
                            # would fabricate an empty user turn.
                            # A content-part LIST is projected for the same reason (a list of
                            # mappings is not a valid attribute value). ``content_text`` is the
                            # projection the backend shares, so the span reads what a judge reads.
                            **({} if turn['input_user_message'] is None else {SpanAttributes.INPUT_VALUE: content_text(turn['input_user_message'])}),
                            SpanAttributes.OUTPUT_VALUE: turn['agent_response'],
                        },
                    ).end()
                    tracer.start_span(
                        name='client',
                        context=turn_ctx,
                        attributes={
                            kind: client_kind,
                            SpanAttributes.SESSION_ID: thread_id,
                            SpanAttributes.INPUT_VALUE: turn['agent_response'],
                            SpanAttributes.OUTPUT_VALUE: turn['output_user_message'],
                        },
                    ).end()
                finally:
                    turn_span.end()
        finally:
            scenario.end()
        # Couple the per-scenario drain into record_transcript: Arize buffers spans
        # behind a BatchSpanProcessor, so without
        # this each scenario's spans would sit in the queue until process teardown.
        # force_flush the SAME provider that built the tree; swallow drain failures
        # so a flush error never loses the (already-built) anchor return value.
        try:
            provider.force_flush()
        except _arize_error_types():
            logger.exception('Arize tracer force_flush failed after record_transcript; spans may be delayed or lost.')
        return scenario

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
        """Post per-policy + rollup evals to the scenario anchor span (best-effort).

        Posts INLINE, with no deferred queue: this builds a single-row pandas DataFrame
        keyed by ``context.span_id`` (the anchor span's 16-hex OTEL id captured into
        ``run_ctx.anchor_span_id`` by :meth:`record_transcript`) with one
        ``eval.<name>.{score,label,explanation}`` group per policy result plus one per
        rollup metric, and ships it via ``ArizeClient.spans.update_evaluations`` — the
        separate span-id-keyed eval channel (OpenInference surfaces no eval/score span
        attribute, so this keyed API is the conformant path). Evals upsert-join to the
        already-flushed anchor span server-side by id (14-day window).

        No-ops (returns) when ``run_ctx`` is ``None``, the anchor id was never captured
        (tracing not live / record_transcript skipped), the sink is disabled, or the v8
        ``arize`` client is not importable. Best-effort: the blocking Flight gRPC call
        runs via ``asyncio.to_thread`` + ``swallow`` so a posting failure logs and never
        breaks the run.
        """
        if not isinstance(run_ctx, ArizeRunContext):
            return
        # Live span-eval channel: post per-policy + rollup evals to the anchor span via
        # update_evaluations, keyed by the 16-hex anchor span id. Gated only when the anchor
        # id was captured, the sink is enabled, and the v8 client is importable — AND NOT
        # under the experiments.run path: there the SAME evals are emitted via passthrough
        # evaluators (the task output), so posting them here too would DOUBLE-emit. The span
        # tree (record_transcript) still runs — only the eval post is skipped.
        span_id = run_ctx.anchor_span_id
        if span_id and self._config.enabled and not self._experiment_active:
            await offload(
                self._post_feedback_sync,
                thread,
                finalize,
                span_id,
                final_world_state=final_world_state,
                tool_events=tool_events,
                scenario_metadata=scenario_metadata,
                sink=self.name,
                what='post_feedback',
            )

    def _post_feedback_sync(
        self,
        thread: CreateThreadResponse,
        finalize: FinalizeResponse,
        span_id: str,
        *,
        final_world_state: dict | None,
        tool_events: dict | None,
        scenario_metadata: dict | None,
    ) -> None:
        import pandas as pd
        from arize import ArizeClient

        policy_results = finalize.policy_evaluation_results or {}
        md = scenario_metadata
        if md is None:
            md = thread.scenario_metadata or {}
        metrics = compute_rollup_metrics(
            policy_results=policy_results,
            final_world_state=final_world_state,
            tool_events=tool_events,
            chatbot_messages=chatbot_messages_of(thread),
            scenario_metadata=md,
            pass_rate=finalize.pass_rate,
            aggregate_score=float(finalize.aggregate_score),
        )

        row: dict[str, Any] = {'context.span_id': span_id}
        # Track sanitized names already written to this row so that two policies (or a
        # policy and a rollup metric) whose original names collide after sanitization get
        # distinct columns (_2, _3, …) rather than silently overwriting each other.
        used_eval_names: set[str] = set()

        # Per-policy evals -> eval.<sanitized>.{score,label,explanation} (shared pipeline;
        # every row carries a numeric score, so the label is never an orphan-explanation —
        # this channel OMITS the label/explanation keys when absent).
        for n, score, label, explanation in _policy_eval_rows(policy_results, used_eval_names):
            row[f'eval.{n}.score'] = score
            if label:
                row[f'eval.{n}.label'] = label
            if explanation:
                row[f'eval.{n}.explanation'] = explanation

        # Rollup metrics -> eval.<sanitized>.{score,label,explanation}. The score carries the
        # eval (so no orphan explanation); the comment becomes the explanation.
        for m in metrics:
            n = _eval_column_name(m.key, used_eval_names)
            row[f'eval.{n}.score'] = float(m.score)
            # A str-valued rollup IS a label. Without this line ``scenario_status`` (the
            # verdict word) and the comma-joined list rollups reach Arize as bare floats, which
            # is the one form a categorical exists to avoid.
            if isinstance(m.value, str) and m.value:
                row[f'eval.{n}.label'] = m.value[:100]
            if m.comment:
                row[f'eval.{n}.explanation'] = str(m.comment)[:10_000]

        if len(row) == 1:  # only the span-id key, no evals -> nothing to post
            return

        # update_evaluations REQUIRES a non-None space_id + project_name to scope the
        # server-side span-id join. When enabled the config validator guarantees
        # space_id, but project_name is optional (initialize() lets register() resolve a
        # native default the eval API cannot see) — so if it is unset here there is no
        # join scope to post against and we no-op rather than mis-scope or crash.
        space_id = self._config.space_id
        project_name = self._config.project_name
        if space_id is None or project_name is None:
            logger.warning(
                'Arize evals skipped: space_id/project_name unresolved (set ARIZE__SPACE_ID and '
                'ARIZE__PROJECT_NAME — the latter MUST match the registered tracing project).'
            )
            return

        df = pd.DataFrame([row])  # single row, one network call carrying every eval
        swallow_types = _arize_error_types()
        try:
            client = ArizeClient(api_key=self._config.api_key)
            client.spans.update_evaluations(
                space_id=space_id,
                project_name=project_name,  # MUST match register(project_name=...)
                dataframe=df,
                validate=True,
            )
        except swallow_types:
            logger.exception('Arize update_evaluations failed; evals not posted (simulation unaffected).')
            return
        logger.info('Arize evals posted for span %s: %d eval column(s).', span_id, len(row) - 1)

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
        """Run the full scenario set as ONE Arize experiment (SYNC, main-thread only).

        Implements the :class:`~plurai.observability.sink_protocol.ExperimentRunnerSink`
        protocol. Hands iteration to Arize's ``experiments.run`` (AsyncExecutor-backed): one
        ASYNC ``task`` per example resolves its scenario id from ``dataset_row`` (stamped
        as a top-level field by ``sync_scenarios_to_arize``), awaits
        ``run_one(sid)`` — which runs the scenario + writes all artifacts on the AsyncExecutor
        loop — and returns the :func:`_build_experiment_output` dict; a failed scenario returns
        a JSON-serializable FAILED-output dict (NEVER ``None`` / raise), so its row is kept (a
        ``None`` task result is dropped by ``experiments.run``). A row whose scenario is not in
        ``scenario_ids`` (an earlier run uploaded it) returns a SKIPPED output and never runs.
        Passthrough evaluators built from ``evaluator_names`` ∪ rollups read those outputs.

        Called SYNCHRONOUSLY from the main thread with NO running loop — ``experiments.run``
        drives ``asyncio.run`` internally via AsyncExecutor and requires exactly that. The
        async ``task`` + ``concurrency=N`` route execution onto that AsyncExecutor. Best-effort:
        self-guards and returns the failure count — an SDK-level failure counts the whole set.

        The dataset handle comes from ``upload_metadata`` — the mapping THIS sink persisted
        at upload time (``sync_scenarios`` ``metadata_updates``), round-tripped opaquely by
        the caller: ``arize_dataset_id`` is preferred; else ``arize_dataset_name``; else
        ``default_dataset_name`` (a name handle resolves via the configured space). The
        space comes from this sink's own config, not from the caller.
        """
        if not self._config.enabled:
            logger.warning('Arize experiment run skipped: sink disabled or SDK not installed.')
            return len(scenario_ids)

        from arize import ArizeClient

        # The experiments-path surface (PassthroughEvaluator subclasses the SDK's
        # CodeEvaluator) lives in its own module that imports the SDK at module
        # level — importable only with the [arize] extra, hence lazy HERE, next to
        # the ArizeClient import that carries the same requirement.
        from plurai.observability.arize_experiments import build_passthrough_evaluators

        evaluators = build_passthrough_evaluators(evaluator_names)
        # Resolve the dataset handle from the metadata this sink wrote at upload time;
        # id preferred (a name handle resolves via the configured space).
        handle = upload_metadata.get('arize_dataset_id') or upload_metadata.get('arize_dataset_name') or default_dataset_name
        # The dataset holds every scenario any run uploaded (the upload is idempotent by
        # folder), so a scoped run must skip the rows it does not cover.
        in_scope = set(scenario_ids)

        async def task(*, dataset_row: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None, **_kwargs: Any) -> dict[str, Any]:
            sid = _resolve_scenario_id(dataset_row, metadata)
            if not sid:
                return _failed_experiment_output('could not resolve scenario_id from dataset_row')
            if sid not in in_scope:
                return _skipped_experiment_output(sid)
            try:
                result = await run_one(sid)
            except Exception as exc:  # noqa: BLE001 — a task failure must RETURN a row, never raise (else the row is dropped)
                logger.exception('Arize experiment task failed for scenario %s.', sid)
                return _failed_experiment_output(f'{type(exc).__name__}: {exc}', scenario_id=sid)
            return _build_experiment_output(result, scenario_metadata=metadata if isinstance(metadata, dict) else None)

        # Mark the experiment path active so post_feedback skips its span-eval
        # channel — evals flow through the passthrough evaluators here; posting them
        # again via the spans.update_evaluations channel would double-emit.
        self._experiment_active = True
        swallow_types = _arize_error_types()
        try:
            client = ArizeClient(api_key=self._config.api_key)
            _experiment, results_df = client.experiments.run(
                name=run_name,
                dataset=handle,
                space=self._config.space_id,
                task=task,
                evaluators=evaluators,
                concurrency=concurrency,
                set_global_tracer_provider=False,
                exit_on_error=False,
                timeout=timeout,
            )
        except swallow_types:
            logger.exception('Arize experiments.run failed for run %r; treating all %d scenario(s) as failed.', run_name, len(scenario_ids))
            return len(scenario_ids)
        finally:
            self._experiment_active = False

        messages = _experiment_failure_messages(results_df)
        if messages is None:
            # The output column is missing — we cannot assess per-scenario status, so
            # treat every scenario as failed (conservative: the same policy as the
            # experiments.run-raised branch above).  Returning 0 here would incorrectly
            # map to exit 0 and hide a broken run.
            logger.warning(
                "experiments.run results frame lacks an 'output' column; cannot assess per-scenario status — treating all %d scenario(s) as failed.",
                len(scenario_ids),
            )
            return len(scenario_ids)
        for msg in messages:
            logger.error('Arize experiment %r: %s', run_name, msg)
        failures = len(messages)
        logger.info('Arize experiment %r completed: %d scenario(s), %d failed.', run_name, len(scenario_ids), failures)
        return failures

    async def drain_run(self, run_ctx: RunContext) -> None:
        """Per-scenario teardown: ``force_flush`` the buffered tracer provider (idempotent).

        Guaranteed by the orchestrator's scenario scope on exit (normal AND exception).
        Mirrors the terminal ``force_flush`` already coupled into ``record_transcript`` —
        firing here too is harmless (``force_flush`` is repeat-safe; nothing buffered ⇒
        a cheap no-op) and ensures the scenario's spans surface even on a path where
        ``record_transcript`` was skipped. No-op when tracing isn't live (``run_ctx``
        is not an :class:`ArizeRunContext` or its ``provider`` is ``None``). Distinct
        from ``close``, which shuts the provider down once per process.

        Async: the blocking ``force_flush`` runs inside via ``asyncio.to_thread``. Unlike
        the happy-path ``record_transcript`` (which uses the NARROW ``_arize_error_types``
        so a programmer bug surfaces), this guaranteed-``finally`` teardown swallows ANY
        exception — a teardown error must never escape into the orchestrator's
        scenario-scope ``finally`` and mask an in-flight exception. This never raises.
        """
        if not isinstance(run_ctx, ArizeRunContext) or run_ctx.provider is None:
            return
        provider = run_ctx.provider

        def _force_flush() -> None:
            try:
                provider.force_flush()
            except Exception:
                logger.exception('Arize force_flush failed at scenario drain; spans may be delayed or lost.')

        await asyncio.to_thread(_force_flush)

    async def close(self) -> None:
        """Process-teardown drain (async). Idempotent + swallowed best-effort.

        A no-op when Arize was never configured; the SDK's own ``shutdown()`` is
        flag-guarded, so repeated calls are safe. ``shutdown()`` is a blocking sync
        SDK drain, so it runs in a worker thread via ``asyncio.to_thread`` to avoid
        blocking the event loop. Per-scenario draining is coupled into
        ``record_transcript``'s terminal ``force_flush`` — this is only the
        once-per-process teardown shutdown.
        """
        if self._provider is None:
            return
        try:
            await asyncio.to_thread(self._provider.shutdown)
        except _arize_error_types():
            logger.exception('Arize tracer shutdown failed at teardown.')

    def sync_scenarios(
        self,
        *,
        dataset_name: str,
        dataset_description: str | None,
        scenarios: list[ScenarioUploadEntry],
        existing_metadata: dict[str, Any],
    ) -> ProviderSyncResult:
        """Upload a generated scenario batch into an Arize AX dataset (delete-then-recreate).

        Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`;
        a disabled/SDK-missing/empty-batch no-op returns ``ok=True, uploaded=0``. The
        resulting ``arize_dataset_id`` rides back in ``metadata_updates`` (persisted by the
        upload path); the experiments.run driver reads it from there, not from the instance.
        """
        return sync_scenarios_to_arize(
            self._config,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            scenarios=scenarios,  # type: ignore[arg-type]  # ScenarioUploadEntry duck-types the {folder_name, inputs, ...} dict shape
            existing_dataset_id=existing_metadata.get('arize_dataset_id'),
        )
