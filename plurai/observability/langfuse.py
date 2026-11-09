"""Client-side Langfuse integration for plurai.Orchestrator."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from plurai.observability.model_calls import ModelCall, openai_tool_calls
from plurai.observability.sink_protocol import RunContext, RunLinks, TranscriptTurn
from plurai.observability.tool_call_recovery import tool_span_metadata
from plurai.observability.dataset_types import ProviderSyncResult, dataset_join_key
from plurai.observability.eval_mapping import (
    TypedScore,
    build_scenario_metadata,
    eval_scope,
    filter_scenario_metadata,
    map_eval_score,
    per_turn_results,
    scenario_record_name,
    turn_record_name,
)
from plurai.observability.feedback import build_scenario_summary, map_rollup_score
from plurai.observability.rollup import (
    chatbot_messages_of,
    compute_rollup_metrics,
    eval_results,
)
from plurai.observability.errors import offload, swallow, swallowable_error_types
from plurai.observability.score_correction import ScoreOverwrite, ScoreOverwriteOutcome, eval_outcome, eval_score_placements, every_eval
from plurai.orchestrator.models import CreateThreadResponse, FinalizeResponse, ScenarioUploadEntry

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from langfuse import Langfuse
    from langfuse.experiment import Evaluation, ExperimentItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural typing for the v4 SESSION-ANCHOR surface this sink probes defensively.
#
# ``propagate_attributes`` is a MODULE-LEVEL function (``from langfuse import
# propagate_attributes``), NOT a client method — so ``client.propagate_attributes``
# genuinely is NOT reachable through the ``Langfuse`` type. It (with its companion
# ``start_as_current_observation`` and the yielded span's ``update``) is therefore modelled
# with a ``@runtime_checkable`` Protocol and narrowed via ``isinstance``: an absent surface
# fails ``isinstance`` ⇒ an early ``return``, never an UNswallowed ``AttributeError``, and the
# call is typed. Because the concrete ``Langfuse`` lacks ``propagate_attributes``, this is
# a clean STRUCTURAL narrow (no "unsafe overlap").
#
# Every OTHER site uses DIRECT TYPED ACCESS, because the attribute IS
# declared on (or inferred by the type checker from) the concretely-typed object:
# ``create_score`` / ``flush`` / ``create_dataset`` / ``create_dataset_item`` /
# ``start_observation`` / ``run_experiment`` / ``get_dataset`` on ``Langfuse``; the
# ``DatasetItem.id`` and ``DatasetClient.items`` / span ``trace_id`` / span ``id`` fields of
# their returned handles. A ``hasattr(...)`` runtime guard precedes direct access wherever a
# surface the TYPE declares may be missing at runtime on a real/older client (or a test stub) (``client.start_observation``, a span's
# ``trace_id`` / ``id``, ``client.run_experiment`` / ``client.get_dataset``). A runtime_checkable-Protocol
# ``isinstance`` is NOT usable for these: the checker already sees the attribute on the
# concrete type and rejects it as an "unsafe overlap", and a widening ``x: object = client``
# does not help (pyright re-narrows the assignment back to the concrete type). ``hasattr``
# keeps the never-raise contract without a ``cast`` / ``# type: ignore``.
# ---------------------------------------------------------------------------


@runtime_checkable
class _SpanUpdate(Protocol):
    """A span handle exposing ``update`` — the session-anchor metadata stamp."""

    def update(self, *, metadata: Any) -> Any: ...


@runtime_checkable
class _SupportsSessionAnchor(Protocol):
    """Client surface for the v4 session-anchor open in :func:`open_session_trace`.

    Requires BOTH ``propagate_attributes`` AND ``start_as_current_observation``. A
    real v4 ``Langfuse`` lacks ``propagate_attributes`` (it is the module-level
    ``from langfuse import propagate_attributes``), so ``isinstance`` is ``False``
    there and the open no-ops.
    """

    def propagate_attributes(self, *, session_id: str, trace_name: str) -> AbstractContextManager[Any]: ...
    def start_as_current_observation(self, *, name: str, as_type: str) -> AbstractContextManager[Any]: ...


@dataclass(frozen=True, slots=True)
class LangfuseRunContext:
    """Langfuse per-scenario run context — the single marker ``prepare_run`` returns.

    Carries the two per-run data ``LangfuseSink`` consumes downstream directly (no
    intermediate handle wrapper): the ``Langfuse`` ``client`` (a process-wide
    singleton in SDK v3/v4 — constructing it registers it in the SDK's client
    registry; read by :func:`open_session_trace`,
    :meth:`LangfuseSink._record_transcript`, and :meth:`LangfuseSink._post_feedback`)
    and the run's ``experiment`` label (stamped onto the session-anchor +
    scenario-anchor span metadata). ``client`` is genuine per-scenario state — NOT a
    duplicate of any sink-instance slot, since ``LangfuseSink`` caches only
    ``self._config`` and the SDK's own registry dedupes by credentials. Capture is
    PARENT-side only via this ``client``: ``record_transcript`` emits the span tree
    and ``post_feedback`` logs per-policy + rollup scores via
    ``client.create_score(session_id=thread_id, ...)`` — there is NO LangChain
    handler.

    ``thread_id`` is never read off the context in langfuse (the orchestrator passes
    it as a kwarg) but is KEPT for the uniform cross-sink RunContext shape that the
    orchestrator relies on polymorphically.
    """

    client: Langfuse | None
    experiment: str
    scenario_name: str
    thread_id: str
    #: The scenario FOLDER — the dataset JOIN key every sink's upload stamps as
    #: ``metadata.scenario_folder``. Distinct from ``scenario_name``, which is the DISPLAY
    #: label; see :func:`plurai.observability.dataset_types.dataset_join_key`. Read by
    #: ``_buffer_replay_row`` to key the row against the uploaded item. ``None`` ⇒ the join
    #: falls back to ``scenario_name``.
    scenario_folder: str | None = None
    #: The operator's agent-build tag (CLI ``--agent-version``), stamped into the anchor
    #: trace metadata by ``record_transcript`` ONLY when supplied (None ⇒ untagged).
    agent_version: str | None = None
    #: Mutable per-scenario holder for the captured RICH-transcript trace handle.
    #: ``record_transcript`` stashes the scenario-anchor ``trace_id`` (+ anchor
    #: ``observation_id``) here so the later ``post_feedback`` can re-target its per-item
    #: scores onto that trace (the Sessions/Traces-view badges). The dataclass is frozen, so
    #: the field REFERENCE never changes — but this dict's CONTENTS are mutated in place,
    #: keeping the per-scenario state on the per-scenario context (no shared sink-instance
    #: mutable state in the bounded-parallel gather). Empty when no trace was captured (older
    #: client surface / ``record_transcript`` skipped) ⇒ scoring degrades to session-only.
    #: (The per-SET native Dataset Run is registered separately, in ``end_experiment``.)
    trace_link: dict[str, Any] = field(default_factory=dict)
    #: Mutable per-scenario holder for the run_experiment REPLAY payload.
    #: ``record_transcript`` stashes the scenario's final agent ``output``, the full ``turns``
    #: transcript, and the ``thread_id`` (session) here; the later ``post_feedback`` reads them
    #: (alongside the computed scores) and appends ONE row to the sink's per-SET replay buffer,
    #: which ``end_experiment`` feeds to ``client.run_experiment`` to register the native
    #: Dataset Run. The replay task returns the ``output`` AND re-emits the ``turns`` under the
    #: native ``experiment-item-run`` span (stamping ``thread_id`` as its session) — so each
    #: experiment item renders the SDK's full shape: ``sdk-experiment`` env +
    #: ``langfuse.experiment.*`` attrs + scores + the turn-by-turn tree + a session link.
    #: Single-writer per scenario (record→post run sequentially for one run), so — unlike the
    #: cross-scenario sink buffer — this needs no lock. Empty when no transcript was recorded ⇒
    #: that scenario contributes an output-only replay.
    replay: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """True when Langfuse tracing is LIVE for this scenario — exposes the EXISTING gate.

        ``record_transcript`` / ``post_feedback`` already short-circuit on
        ``run_ctx.client is None``. :meth:`LangfuseSink.prepare_run` always returns a
        :class:`LangfuseRunContext`; ``client=None`` is the disabled sentinel (produced
        when the sink is disabled or langfuse isn't installed). This property surfaces
        that existing decision uniformly; it does not change what gets traced.
        """
        return self.client is not None


class LangfuseSinkConfig(BaseModel):
    """Langfuse observability sink config (mirrors the Plurai backend's Langfuse settings).

    Owned by this sink module: a
    frozen pydantic ``BaseModel`` that ``Settings`` nests so its fields
    source from the ``LANGFUSE__*`` nested env namespace (``LANGFUSE__ENABLED`` /
    ``LANGFUSE__PUBLIC_KEY`` / ``LANGFUSE__SECRET_KEY`` / ``LANGFUSE__HOST`` /
    ``LANGFUSE__ENVIRONMENT`` / ``LANGFUSE__PROJECT`` / ``LANGFUSE__DATASET``). Keys are PLAIN strings — NOT
    ``SecretStr`` — because the SDK resolves them straight from the client
    environment and passes them RAW to the Langfuse client.

    ``enabled`` is a native ``bool`` field coerced via pydantic's standard
    case-insensitive boolean coercion (e.g. ``1`` / ``true`` / ``yes`` / ``on`` /
    ``t`` / ``y`` are truthy; ``0`` / ``false`` / ``no`` / ``off`` / ``f`` / ``n`` are
    falsy) and an empty env value is ignored (``env_ignore_empty`` on ``Settings``),
    leaving it ``False``.
    Credential fields coerce empty-string -> ``None``. When ``host`` is left ``None`` the
    Langfuse SDK natively resolves its own ``LANGFUSE_HOST`` / ``LANGFUSE_BASE_URL``
    at client-construction time (vendor behaviour, outside this config surface);
    pass ``LANGFUSE__HOST`` to override it explicitly.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    environment: str | None = None
    project: str | None = None
    #: Explicit dataset override. HEAD of the dataset cascade at the three sites that
    #: resolve one for THIS sink — the upload, the set-level Dataset Run, and the UI link —
    #: so they cannot name different datasets. ``None`` ⇒ the caller's name is used.
    #: Does NOT reach the ``dataset_name`` recorded in each transcript's run context, which
    #: still carries the caller's name (as it does for the LangSmith override).
    dataset: str | None = None

    @field_validator('public_key', 'secret_key', 'host', 'environment', 'project', 'dataset', mode='before')
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        """Coerce an empty / whitespace-only credential string to ``None``."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode='after')
    def _require_creds_when_enabled(self) -> LangfuseSinkConfig:
        """When the sink is ACTIVE, require ``public_key`` AND ``secret_key`` non-``None``.

        GATED on ``enabled``: ``Settings`` ALWAYS instantiates this
        nested config (even for an INACTIVE sink), so bare required fields would
        wrongly force creds for a sink that is off. The conditional-on-enabled shape
        is the only correct one — when Langfuse is the active sink both keys must be
        defined exactly once and non-``None``; an inactive config stays unconstrained.
        """
        if self.enabled:
            missing = [name for name in ('public_key', 'secret_key') if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f'Langfuse is enabled but required credential field(s) are missing: {", ".join(missing)}. '
                    'Set LANGFUSE__PUBLIC_KEY and LANGFUSE__SECRET_KEY (nested) when LANGFUSE__ENABLED is set.'
                )
        return self


def _langfuse_error_types() -> tuple[type[BaseException], ...]:
    """Return the Langfuse / HTTP exception classes worth swallowing.

    Looked up dynamically so the SDK can be imported without ``langfuse``
    installed. Any other exception (TypeError / AttributeError / KeyError /
    ImportError / programmer bug) propagates so the caller sees the real cause.
    """
    return swallowable_error_types(
        ('langfuse.api.core', 'ApiError'),
        ('httpx', 'HTTPError'),
        ('requests.exceptions', 'RequestException'),
    )


def _langfuse_available() -> bool:
    return importlib.util.find_spec('langfuse') is not None


def _build_client(config: LangfuseSinkConfig) -> Langfuse:
    """Construct a Langfuse client from config.

    ``None`` constructor args defer to the SDK's native env resolution
    (``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST`` /
    ``LANGFUSE_BASE_URL``); explicit config fields override. The SDK reads its
    own env vars at construction time — this never mutates ``os.environ``.
    """
    from langfuse import Langfuse

    return Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        host=config.host,
        environment=config.environment,
    )


def warm(config: LangfuseSinkConfig) -> None:
    """Process-scoped warm-up: build (and registry-register) the Langfuse client once.

    Called ONCE per process so the Langfuse client (and its claim on the OTel
    global tracer provider) is built deterministically from config rather than
    on whichever invocation lands first. This keeps NO explicit cache slot —
    the SDK's OWN client registry dedupes by credentials, so the later per-run
    :func:`configure_tracing` resolves this same client rather than building a
    second one. No-ops when Langfuse is disabled or the package is absent.
    """
    if not config.enabled or not _langfuse_available():
        return
    _build_client(config)


def configure_tracing(
    config: LangfuseSinkConfig,
    *,
    agent_name: str,
    experiment_name: str,
    dataset_name: str | None = None,
) -> Langfuse | None:
    """Build the run's Langfuse client (gated), or ``None`` (PURE BIND).

    Deliberately sets NO environment variables and enables no ambient tracing —
    capture is parent-side only (there is no LangChain handler), so traces appear
    solely from this sink's explicit span/score calls (:func:`open_session_trace`,
    ``record_transcript``, ``post_feedback``). Its only "side effect" is constructing
    the ``Langfuse`` client, which the SDK's client registry DEDUPES against the one
    :func:`warm` built at the entry point — so the per-request path carries no
    process-level side effect beyond that deduped construction.

    ``None`` constructor args defer to the SDK's native env resolution
    (``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST`` /
    ``LANGFUSE_BASE_URL``); explicit config fields override. Auth is verified
    lazily by the SDK at first real use, not eagerly here.

    ``agent_name`` / ``experiment_name`` / ``dataset_name`` are accepted for
    caller/protocol parity but not consumed by the bind itself: the caller
    (:meth:`LangfuseSink._prepare_run`) carries ``experiment_name`` straight onto
    the :class:`LangfuseRunContext`; the Langfuse client sources its
    project/environment from ``config`` at construction time.

    Returns the gated ``Langfuse`` client, or ``None`` (no-op) when
    ``config.enabled`` is False or langfuse isn't installed.
    """
    if not config.enabled:
        return None
    if not _langfuse_available():
        logger.warning(
            'Langfuse enabled in config but the langfuse package is not importable '
            '(try `pip install plurai-simulation-sdk[langfuse]`). Falling back to no Langfuse observability.'
        )
        return None

    return _build_client(config)


def open_session_trace(client: Langfuse, *, experiment: str, thread_id: str, scenario_name: str) -> None:
    """Materialize the run's Langfuse SESSION so its session-scoped scores are visible.

    Opens one short root span inside ``propagate_attributes(session_id=thread_id)``
    and ends it immediately. A Langfuse Session is materialized implicitly by
    traces carrying its ``sessionId`` (there is no ``create_session``), and capture
    is parent-side only (no ``CallbackHandler``) — so without this no trace ever
    carries ``thread_id`` and the Sessions view has no row to surface the standalone
    ``create_score(session_id=...)`` rows ``post_feedback`` posts.
    (See the module docstring for the grounded data-model reasoning.)

    ``propagate_attributes`` sets the attribute on the currently-active span AND
    on spans created WITHIN its context, so the span is opened INSIDE the context
    manager to inherit ``session_id``. Best-effort: every failure is swallowed —
    observability must never abort a sim. v4 surface only.
    """
    # ``isinstance`` (NOT direct access) so a client lacking the v4 session-anchor
    # surface no-ops without raising. ``propagate_attributes`` is a MODULE-level
    # function, never a real-client method, so a genuine v4 ``Langfuse`` fails this
    # probe.
    if not isinstance(client, _SupportsSessionAnchor):
        return
    swallow = _langfuse_error_types()
    # Session-anchor trace + span carry the same ``scenario:<name>`` record name the
    # rich transcript trace uses, so both Sessions-view rows read identically.
    record_name = scenario_record_name(scenario_name)
    try:
        with (
            client.propagate_attributes(session_id=thread_id, trace_name=record_name),
            client.start_as_current_observation(name=record_name, as_type='span') as span,
        ):
            if isinstance(span, _SpanUpdate):
                # BARE identifying metadata via the shared builder (no backend-prefixed key in
                # anything shown in the UI) — the same {scenario_name, thread_id, experiment}
                # key shapes the scenario anchor's record metadata carries.
                metadata = build_scenario_metadata(scenario_metadata={'scenario_name': scenario_name}, thread_id=thread_id, experiment_name=experiment)
                span.update(metadata=metadata)
    except swallow:
        logger.exception('Langfuse session-anchor trace open failed for thread %r.', thread_id)


# The langfuse score-payload row shapes this sink emits, as a DISCRIMINATED union keyed on the
# data_type literal (last element): the SDK's ``create_score`` overloads correlate value-type
# with data_type (float ↔ NUMERIC/BOOLEAN, str ↔ CATEGORICAL), so the call site narrows each
# row on its discriminant and pyright matches the right overload — no cast. Declared once so
# the live create_score calls, the replay Evaluation rows, and the buffer typing all share it.
_NumericScorePayload = tuple[str, float, str, Literal['NUMERIC', 'BOOLEAN']]
_CategoricalScorePayload = tuple[str, str, str, Literal['CATEGORICAL']]
_ScorePayload = _NumericScorePayload | _CategoricalScorePayload


def _score_payloads(ts: TypedScore) -> list[_ScorePayload]:
    """Expand one :class:`TypedScore` into langfuse ``create_score`` payload rows
    ``(name, value, comment, data_type)`` — the ONE place the kind→data_type mapping
    lives, shared by the live ``create_score`` axes and the replay ``Evaluation``s so
    the two surfaces can never disagree.

    * 'boolean' → one BOOLEAN row carrying the numeric 1.0/0.0 form (langfuse BOOLEAN
      scores take the numeric value; the True/False rendering is server-side).
    * 'number' → one NUMERIC row carrying the raw number.
    * 'categorical' → the native string label as a CATEGORICAL row PLUS a separate
      NUMERIC companion row (``<key>_score``): langfuse charts/aggregates only numeric
      scores, so a categorical without the companion would be invisible in score charts.
    * 'unknown' → one NUMERIC 0.5 row (ambiguous — emitted, never dropped, so a
      malformed PolicyResult stays visible).
    """
    if ts.kind == 'categorical' and isinstance(ts.value, str):
        return [(ts.key, ts.value, ts.comment, 'CATEGORICAL'), (f'{ts.key}_score', ts.score, ts.comment, 'NUMERIC')]
    if ts.kind == 'boolean':
        return [(ts.key, ts.score, ts.comment, 'BOOLEAN')]
    return [(ts.key, ts.score, ts.comment, 'NUMERIC')]


def _paged(fetch: Callable[..., Any], **filters: Any) -> Iterator[Any]:
    """Every row of a cursor-paged langfuse list endpoint (``scores_v3.get_many_v3`` / ``observations.get_many``)."""
    cursor = None
    while True:
        page = fetch(cursor=cursor, **filters)
        yield from page.data
        cursor = page.meta.cursor
        if not cursor:
            return


@dataclass(frozen=True, slots=True)
class _RecordedScore:
    """A score a run posted: the score row, the subject to re-post it on, and the name of the observation it sits on."""

    score: Any
    target: dict[str, str]
    observation_name: str | None


def _recorded_scores(client: Langfuse, thread_id: str, names: set[str]) -> list[_RecordedScore]:
    """Every score named in ``names`` that a run posted for ``thread_id``.

    A run scores the session itself and each trace in it (the scenario trace and the
    experiment-replay trace), a turn-scoped score sitting on that trace's ``turn N``
    observation. Langfuse mints the score ids, so they are found here through the thread's
    session; nothing about them can be derived.
    """
    api = client.api
    wanted = ','.join(sorted(names))
    found: dict[str, _RecordedScore] = {}
    for score in _paged(api.scores_v3.get_many_v3, session_id=thread_id, name=wanted, fields='subject,details', limit=100):
        if score.subject is not None and score.subject.kind == 'session':
            found.setdefault(score.id, _RecordedScore(score, {'session_id': score.subject.id}, None))
    # Observations API v2 has no trace entity: the session's traces are the trace ids of the rows
    # it filters to, and each trace's own rows name the observations its scores sit on.
    session_filter = json.dumps([{'type': 'string', 'column': 'sessionId', 'operator': '=', 'value': thread_id}])
    trace_ids = dict.fromkeys(o.trace_id for o in _paged(api.observations.get_many, filter=session_filter, fields='core', limit=1000) if o.trace_id)
    for trace_id in trace_ids:
        observation_names = {o.id: o.name for o in _paged(api.observations.get_many, trace_id=trace_id, fields='core,basic', limit=1000)}
        for score in _paged(api.scores_v3.get_many_v3, trace_id=trace_id, name=wanted, fields='subject,details', limit=100):
            subject = score.subject
            if subject is not None and subject.kind == 'trace':
                found.setdefault(score.id, _RecordedScore(score, {'trace_id': subject.id}, None))
            elif subject is not None and subject.kind == 'observation':
                target = {'trace_id': subject.trace_id or trace_id, 'observation_id': subject.id}
                found.setdefault(score.id, _RecordedScore(score, target, observation_names.get(subject.id)))
    return list(found.values())


def _overwrite_eval(client: Langfuse, recorded: Sequence[_RecordedScore], placements: Sequence[tuple[int | None, TypedScore]]) -> int:
    """Re-post every recorded score of one eval with its corrected payload, by its own id on its own subject.

    A score is matched by name and place: on the ``turn N`` observation for a turn placement,
    anywhere else (session, trace, scenario anchor) for a scenario placement. Re-posting an
    existing ``score_id`` is an upsert, so the corrected value replaces the old one in place.
    Returns how many scores were overwritten.
    """
    turn_names = {turn_record_name(k) for k, _ in placements if k is not None}
    written = 0
    for turn_idx, ts in placements:
        place = None if turn_idx is None else turn_record_name(turn_idx)
        for payload in _score_payloads(ts):
            for rec in recorded:
                rec_place = rec.observation_name if rec.observation_name in turn_names else None
                if rec.score.name != payload[0] or rec_place != place:
                    continue
                kw: dict[str, Any] = {'score_id': rec.score.id, 'comment': payload[2], 'environment': rec.score.environment, **rec.target}
                # Branch on the data_type discriminant so pyright matches the create_score overload.
                if payload[3] == 'CATEGORICAL':
                    client.create_score(name=payload[0], value=payload[1], data_type=payload[3], **kw)
                else:
                    client.create_score(name=payload[0], value=payload[1], data_type=payload[3], **kw)
                written += 1
    return written


def _langfuse_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version('langfuse')
    except PackageNotFoundError:
        return 'unknown'


def _rename_active_experiment_span(name: str) -> None:
    """Rename langfuse's auto-generated generic ``experiment-item-run`` span to *name*.

    The gotcha: ``client.run_experiment`` opens every item's root span with the
    SDK-internal name ``experiment-item-run``, so the experiment view renders N
    identical generic records. Langfuse exposes no rename knob on the experiment
    surface, but the span IS the current OpenTelemetry span while the replay task runs
    (``_process_experiment_item`` invokes the task synchronously inside it) — so
    renaming the ACTIVE OTel span before it ends is the one seam that retitles the
    record to the shared ``scenario:<name>`` convention.

    Best-effort + offline-safe: ``opentelemetry`` is imported FUNCTION-locally (it
    arrives only with the langfuse extra — module scope must import no vendor SDK) and
    an absent package / no active span (``get_current_span`` then returns the no-op
    ``INVALID_SPAN``, whose ``update_name`` is a no-op) degrades to the generic name
    rather than raising.
    """
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return
    otel_trace.get_current_span().update_name(name)


#: LangChain ``usage_metadata`` key -> Langfuse ``usage_details`` key.
_USAGE_KEYS = {'input_tokens': 'input', 'output_tokens': 'output', 'total_tokens': 'total'}


def _generation_fields(call: ModelCall) -> dict[str, Any]:
    """The ``start_observation(as_type='generation')`` keywords for one model call.

    ``output`` is the completion text, or the assistant message (``content`` + OpenAI-shaped
    ``tool_calls``) when the call requested tools — a tool-requesting completion is usually
    empty text, and an empty generation would hide what the model decided. ``model`` and
    ``usage_details`` are passed only when the framework reported them; usage keeps the integer
    counts it knows (``input`` / ``output`` / ``total``).
    """
    output: Any = call['output']
    if call['tool_calls']:
        output = {'role': 'assistant', 'content': call['output'], 'tool_calls': openai_tool_calls(call['tool_calls'])}
    fields: dict[str, Any] = {'name': call.get('model') or 'model', 'input': call['input'], 'output': output}
    if model := call.get('model'):
        fields['model'] = model
    usage = {
        target: value
        for source, target in _USAGE_KEYS.items()
        if isinstance(value := (call.get('usage') or {}).get(source), int) and not isinstance(value, bool)
    }
    if usage:
        fields['usage_details'] = usage
    return fields


class _MillisecondClock:
    """Holds each observation of a tree to a later millisecond than the one before it.

    The tree is written after the conversation it records, in a few milliseconds, and langfuse
    stamps an observation's start when it is created (``start_observation`` takes no start
    time) with millisecond precision. Sibling turns then share start times and the UI orders the
    ties arbitrarily; waiting for the next millisecond before each observation keeps the
    conversation's order, at about a millisecond per observation.
    """

    def __init__(self) -> None:
        self._last = time.time_ns() // 1_000_000

    def tick(self) -> None:
        while (now := time.time_ns() // 1_000_000) <= self._last:
            time.sleep(0.0002)
        self._last = now


def _emit_agent_span_with_tools(turn_span: Any, turn: TranscriptTurn, clock: _MillisecondClock) -> None:
    """Emit the per-turn ``agent`` observation (the generation itself for a one-call turn) and its ``tool`` children.

    Shared by the rich ``record_transcript`` trace AND the experiment-replay
    decoration (``_emit_replay_turns``) so both render the
    "turn -> assistant -> tool calls (args + result)" tree identically — the
    experiment view stays self-contained and matches the default-env trace. Tool
    calls come from the turn's recovered ``agent_tool_calls`` (``tool`` / ``args`` /
    ``response`` / ``status``); a turn that called no tools keeps the plain agent
    span. An errored call surfaces at Langfuse ``level='ERROR'`` so it renders red.
    A turn with several model calls (``agent_model_calls``) keeps an ``agent`` span with one
    ``generation`` child per call, ahead of the tools, so prompt, completion, model and token
    usage render natively. ``clock`` holds every observation to its own millisecond.
    A tool span fed by a ``/mock/execute`` record carries ``metadata={'mocked': True}`` (:func:`tool_span_metadata`).
    ``turn_span`` is a runtime ``LangfuseSpan`` (typed ``Any`` here, matching this
    module's other span handling); the caller already wraps emission in the
    ``swallow`` tuple, so this stays a thin builder.
    """
    # One model call IS the agent's step, so the agent observation itself is that generation; a
    # child with the same prompt and reply would only repeat it. Several calls (call -> tool ->
    # call ...) keep an ``agent`` span with one generation per call.
    calls = turn.get('agent_model_calls') or []
    clock.tick()
    if len(calls) == 1:
        agent_span = turn_span.start_observation(as_type='generation', **{**_generation_fields(calls[0]), 'name': 'agent'})
    else:
        agent_span = _agent_span(turn_span, turn)
        for model_call in calls:
            clock.tick()
            agent_span.start_observation(as_type='generation', **_generation_fields(model_call)).end()
    for tc in turn.get('agent_tool_calls') or []:
        errored = tc['status'] == 'error'
        clock.tick()
        agent_span.start_observation(
            name=tc['tool'],
            as_type='tool',
            input=tc['args'],
            output=tc['response'],
            level='ERROR' if errored else 'DEFAULT',
            status_message=f'tool call {tc["tool"]!r} errored' if errored else None,
            **({'metadata': marker} if (marker := tool_span_metadata(tc)) else {}),
        ).end()
    agent_span.end()


def _agent_span(turn_span: Any, turn: TranscriptTurn) -> Any:
    """The plain ``agent`` span of a turn whose model calls hang under it (or that has none)."""
    return turn_span.start_observation(
        name='agent',
        as_type='span',
        # A content-part LIST rides STRUCTURED here rather than projected: this input is a JSON
        # value to this sink, and the parts say more than their text does. ``None`` (an
        # agent-first turn) is OMITTED rather than sent as a null: langfuse's ``_serialize``
        # returns ``None`` for it and ``span.py`` drops every attribute whose value is ``None``.
        # Only Arize projects, because an OTel attribute takes no sequence of mappings.
        input=turn['input_user_message'],
        output=turn['agent_response'],
    )


#: The uuid5 namespace seed. Every stored item id derives from its value, so a different value maps the same scenario to a
#: different item id.
_ITEM_ID_SEED = 'plurai-langfuse-dataset-item'


def _stable_item_id(dataset_name: str, folder_name: str) -> str:
    """Derive a deterministic Langfuse dataset-item id from the scenario folder.

    Langfuse upserts items by ``id`` (same id => overwrite), so a stable,
    folder-derived id makes re-running the generator idempotent (overwrite in
    place). uuid5 keeps the id URL-safe regardless of folder-name characters and
    namespaces it by dataset so the same folder in two datasets never collides.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{_ITEM_ID_SEED}:{dataset_name}:{folder_name}'))


def sync_scenarios_to_langfuse(
    config: LangfuseSinkConfig,
    *,
    dataset_name: str,
    dataset_description: str | None,
    scenarios: list[dict[str, Any]],
    existing_items: dict[str, str] | None = None,
) -> ProviderSyncResult:
    """Upsert a scenario batch into a Langfuse Dataset.

    Each entry in ``scenarios`` must have ``folder_name`` and ``inputs``;
    ``outputs`` and ``metadata`` are optional (``outputs`` is empty by design
    today — there is no ground truth to attach, not a gap).

    Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`.
    A true no-op (Langfuse disabled / unavailable) returns ``ok=True, uploaded=0``.
    A dataset create/upsert transport error ⇒ ``ok=False`` (logged at ``error``
    level); per-item failures are TALLIED (``failed`` / ``errors``) rather than
    silently dropped, and an all-items-failed batch is ``ok=False``.

    Idempotency: items are upserted via a stable folder-derived ``id``
    (:func:`_stable_item_id`), so re-running the generator overwrites in place
    instead of duplicating. ``existing_items`` (from a prior run) takes
    precedence over the derived id so the mapping survives any future id-scheme
    change. NOTE: this is item upsert only — no run<->item experiment linkage.
    """
    if not config.enabled or not _langfuse_available():
        return ProviderSyncResult(provider='langfuse', metadata_updates={}, summary_line='Langfuse: disabled or SDK not installed — nothing uploaded.')

    client = _build_client(config)
    swallow = _langfuse_error_types()

    try:
        # Langfuse's POST /api/public/v2/datasets upserts by name, so calling it
        # on an existing dataset is a safe no-op update (get-or-create semantics).
        client.create_dataset(name=dataset_name, description=dataset_description)
    except swallow as exc:
        # Dataset op transport failure is a genuine FAILURE, not a no-op.
        logger.error('Langfuse dataset create/upsert for %r failed: %s', dataset_name, exc)
        return ProviderSyncResult(
            provider='langfuse',
            metadata_updates={},
            summary_line=f'Langfuse: dataset create/upsert for {dataset_name!r} failed: {exc}',
            ok=False,
            errors=(f'dataset create/upsert: {exc}',),
        )

    existing_items = existing_items or {}
    result: dict[str, str] = {}
    errors: list[str] = []

    for scen in scenarios:
        folder = scen['folder_name']
        inputs = scen['inputs']
        outputs = scen.get('outputs') or {}
        # Item metadata is the scenario DEFINITION, ALLOW-LIST filtered (``filter_scenario_metadata``:
        # no run facts, since an item outlives every run that replays it), plus ``scenario_folder``:
        # the run<->item JOIN key ``_run_experiment_replay`` matches on — internal plumbing, deliberately kept.
        metadata = {**filter_scenario_metadata(scen.get('metadata') or {}), 'scenario_folder': folder}
        item_id = existing_items.get(folder) or _stable_item_id(dataset_name, folder)
        try:
            item = client.create_dataset_item(
                dataset_name=dataset_name,
                input=inputs,
                expected_output=outputs,
                metadata=metadata,
                id=item_id,
            )
            # ``create_dataset_item`` returns a concretely-typed ``DatasetItem`` whose
            # ``id`` is a declared ``str`` field — direct typed access (no getattr).
            result[folder] = str(item.id)
        except swallow as exc:
            # Best-effort per item: tally the loss (don't abort the loop).
            logger.error('Langfuse dataset-item upsert failed for scenario %r: %s', folder, exc)
            errors.append(f'{folder}: {exc}')

    try:
        client.flush()
    except swallow:
        logger.exception('Langfuse flush failed after syncing dataset %r.', dataset_name)

    failed = len(errors)
    total = len(scenarios)
    ok = not (total > 0 and failed == total)
    summary = f'Langfuse: uploaded {len(result)}/{total} item(s) to dataset {dataset_name!r}.'
    if failed:
        summary = f'Langfuse: uploaded {len(result)}/{total} item(s) to dataset {dataset_name!r} ({failed} failed — see log).'
    return ProviderSyncResult(
        provider='langfuse',
        metadata_updates={
            'langfuse_dataset_name': dataset_name,
            'langfuse_dataset_items': result,
        },
        summary_line=summary,
        ok=ok,
        uploaded=len(result),
        failed=failed,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# LangfuseSink — the ObservabilitySink instance for the parent-side Langfuse
# sink. Holds the config as INSTANCE state (Langfuse keeps NO module-level
# mutable global — the SDK's own client registry dedupes by credentials). The
# stateless module functions above (warm / configure_tracing /
# open_session_trace / sync_scenarios_to_langfuse) are reused as they are.
# ---------------------------------------------------------------------------


class LangfuseSink:
    """Langfuse sink (parent-side; per-policy + rollup scores + a NATIVE Dataset Run).

    ``prepare_run`` warms (process-once), builds the parent-side Langfuse client,
    then eagerly materializes the run's SESSION via :func:`open_session_trace` (so
    the real run path's session-scoped scores are visible — see module
    docstring) and returns a :class:`LangfuseRunContext` carrying that ``client`` +
    ``experiment`` directly. ``record_transcript`` emits the uniform 3-level span
    tree under the session-scoped trace (the RICH transcript, kept in the normal
    Traces/Sessions view), capturing the scenario-anchor ``trace_id`` AND the final
    agent ``output`` onto the run context. ``post_feedback`` logs per-policy + rollup
    scores via ``client.create_score(...)`` against TWO axes — ``session_id`` (Sessions
    view) and the per-item ``trace_id`` (+ anchor ``observation_id``) — each emitted as a
    SINGLE-TARGET call (langfuse v4 drops a score event that mixes object links); it also
    buffers ``{output, scores}`` for the set.

    The per-SET Dataset Run is registered NATIVELY in ``end_experiment`` via
    ``client.run_experiment`` (a REPLAY task that returns each scenario's captured output +
    an evaluator that returns its scores). Doing it through the SDK's own experiment
    machinery is what makes the run render fully in the langfuse experiment UI —
    ``sdk-experiment`` environment, the ``langfuse.experiment.*`` span attributes, and
    item-level scores in the Scores view — which the lower-level
    ``client.api.dataset_run_items.create`` linkage does NOT produce (the UI does not
    aggregate an externally-anchored run). The replay task additionally re-emits each
    scenario's turn-by-turn span tree under its ``experiment-item-run`` span and stamps its
    session, so every experiment item shows the same turn detail + session link as the rich
    transcript trace ``record_transcript`` keeps in the default env (Traces/Sessions view).

    Implements the additive
    :class:`~plurai.observability.sink_protocol.ExperimentLifecycleSink` seam
    (``begin_experiment`` / ``end_experiment``) so the whole scenario set maps to ONE
    Langfuse Dataset Run.
    """

    name = 'langfuse'

    def __init__(self, config: LangfuseSinkConfig) -> None:
        self._config = config
        # Scenario-SET experiment handles for the ExperimentLifecycleSink seam: set ONCE by
        # ``begin_experiment`` BEFORE the gather and read-only during it, they name the per-SET
        # Langfuse Dataset Run. ``None`` until/unless the runner drives the seam.
        self._experiment_dataset_name: str | None = None
        self._experiment_run_name: str | None = None
        # Per-SET REPLAY buffer feeding the native ``run_experiment`` registration:
        # ``{scenario_folder: {'output': str, 'scores': [(name, value, comment), ...]}}``.
        # ``begin_experiment`` resets it (before the gather); each scenario's ``post_feedback``
        # body inserts ONE row keyed by its UNIQUE folder from OFF the event loop
        # (``asyncio.to_thread``); ``end_experiment`` snapshots + clears it (after the gather).
        # No lock: concurrent inserts hit DISTINCT keys, and a single ``dict[key] = value`` is
        # atomic under the GIL (CPython never releases it mid-``__setitem__``).
        # The reset/snapshot run single-threaded (begin before, end after the gather), so they
        # never overlap an insert. NOTE: this leans on the orchestrator's one-set-at-a-time
        # begin->gather->end sequencing AND the GIL; a free-threaded build would need a lock.
        self._replay_rows: dict[str, dict[str, Any]] = {}
        # What ``run_links`` reports: the Dataset Run page ``client.run_experiment`` returns
        # (set by ``end_experiment`` after the gather) and ``(scenario_folder, trace_id)`` per
        # recorded scenario. The appends come from concurrent ``record_transcript`` threads;
        # ``list.append`` is atomic under the GIL, so no lock (same reasoning as ``_replay_rows``).
        self._experiment_url: str | None = None
        self._recorded_traces: list[tuple[str, str]] = []

    async def warm_once(self) -> None:
        """Build (and registry-register) the Langfuse client ONCE, on the event-loop thread.

        Called at the entry point BEFORE any gather. Defers to the module-level
        :func:`warm`, which constructs the client deterministically from config so the
        SDK's OWN client registry (which dedupes by credentials) resolves the same
        client on the later per-scenario :func:`configure_tracing` build rather than
        racing on whichever scenario lands first. Idempotent — the SDK registry dedupes,
        so a repeat build returns the same client. No-op when disabled / absent.
        Best-effort: never raises.
        """
        await offload(warm, self._config, sink=self.name, what='warm_once')

    # -- dataset_url (optional DatasetUrlSink role) -------------------------

    def dataset_url(self, dataset_name: str) -> str | None:
        """Best-effort UI URL of the run's Langfuse dataset page (never raises).

        Constructed as ``<host>/project/<project_id>/datasets/<name>`` — the same
        base-url + project-id shape the vendor's own ``get_trace_url`` builds its links
        from. The project id comes from the client's persisted ``_get_project_id``
        lookup (an underscore surface, hence the defensive ``getattr``: SDK drift
        degrades to no link, never a crash); the base url prefers the client's own
        normalized ``_base_url`` over the raw config host. When either is unavailable
        we return ``None`` rather than fabricate a URL.

        ``LANGFUSE__DATASET`` (``cfg.dataset``) heads the cascade here too, so the printed
        link lands on the dataset the upload and the run actually targeted.
        """
        if not self._config.enabled or not _langfuse_available():
            return None
        resolved_name = self._config.dataset or dataset_name
        try:
            client = _build_client(self._config)
            resolve_project_id = getattr(client, '_get_project_id', None)
            project_id = resolve_project_id() if callable(resolve_project_id) else None
            base = getattr(client, '_base_url', None) or self._config.host
            if not project_id or not base:
                return None
            return f'{str(base).rstrip("/")}/project/{project_id}/datasets/{urllib.parse.quote(str(resolved_name), safe="")}'
        except Exception:
            logger.debug('Langfuse dataset_url resolution failed for %r.', resolved_name, exc_info=True)
            return None

    # -- run_links (optional RunLinksSink role) -----------------------------

    def run_links(self, dataset_name: str) -> RunLinks | None:
        """Best-effort links to the Dataset Run and to each recorded scenario's trace (never raises).

        The experiment link is the ``dataset_run_url`` the native ``run_experiment`` registration
        returned (``None`` when no experiment ran or the SDK reported none); a trace link is the
        client's ``get_trace_url`` for the scenario anchor's trace id. A scenario whose trace
        URL cannot be built is left out.
        """
        recorded = dict(self._recorded_traces)  # a re-driven scenario's latest attempt wins
        if not self._config.enabled or not _langfuse_available() or not (recorded or self._experiment_url):
            return None
        traces: list[tuple[str, str]] = []
        try:
            client = _build_client(self._config)
            if hasattr(client, 'get_trace_url'):
                for folder, trace_id in sorted(recorded.items()):
                    if url := client.get_trace_url(trace_id=trace_id):
                        traces.append((folder, url))
        except Exception:
            logger.debug('Langfuse run_links resolution failed for %r.', dataset_name, exc_info=True)
        return RunLinks(experiment_url=self._experiment_url, traces=tuple(traces)) if traces or self._experiment_url else None

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
    ) -> LangfuseRunContext:
        """Warm + build the parent-side Langfuse client; ALWAYS return its run context.

        Async: the blocking warm + session-anchor open run inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly. Best-effort: a
        swallowed setup failure yields a DISABLED context (``client=None`` ⇒
        ``enabled=False``) rather than ``None``.
        """
        return await offload(
            self._prepare_run,
            agent_name=agent_name,
            experiment_name=experiment_name,
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            thread_id=thread_id,
            dataset_name=dataset_name,
            agent_version=agent_version,
            sink=self.name,
            what='prepare_run',
        ) or LangfuseRunContext(client=None, experiment=experiment_name, scenario_name=scenario_name, thread_id=thread_id)

    def _prepare_run(
        self,
        *,
        agent_name: str,
        experiment_name: str,
        scenario_name: str,
        thread_id: str,
        dataset_name: str | None,
        agent_version: str,
        scenario_folder: str | None = None,
    ) -> LangfuseRunContext:
        warm(self._config)
        # DATASET cascade: the EXPLICIT config dataset (LANGFUSE__DATASET) wins so an
        # override is never clobbered by the ``dataset_name`` the orchestrator passes
        # (the scenario set's name — the SAME name the upload path targets).
        client = configure_tracing(
            self._config,
            agent_name=agent_name,
            experiment_name=experiment_name,
            dataset_name=self._config.dataset or dataset_name,
        )
        if client is None:
            # Disabled SENTINEL (client=None ⇒ enabled=False): returned INSTEAD of None
            # when the sink is disabled / langfuse absent, so prepare_run never returns None.
            return LangfuseRunContext(client=None, experiment=experiment_name, scenario_name=scenario_name, thread_id=thread_id)
        # Eagerly materialize the SESSION here (capture is parent-side only with no
        # CallbackHandler, so no trace would otherwise carry this session_id and the
        # session-scoped scores post_feedback posts would have no visible
        # Sessions-view row). Best-effort — open_session_trace swallows internally.
        open_session_trace(client, experiment=experiment_name, thread_id=thread_id, scenario_name=scenario_name)
        return LangfuseRunContext(
            client=client,
            experiment=experiment_name,
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            thread_id=thread_id,
            agent_version=agent_version,
        )

    async def record_transcript(
        self,
        run_ctx: RunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit the uniform 3-level span tree under a SESSION-scoped Langfuse trace.

        Builds ``scenario -> turn[i] -> {agent, client}`` natively via
        ``client.start_observation(...)`` then ``span.start_observation(...)`` nesting
        — the same shape every sink produces, from orchestrator-owned strings only.
        Every span is created INSIDE a ``propagate_attributes(session_id=thread_id,
        trace_name=scenario_name)`` context so the scenario anchor (and its children)
        carry ``session_id`` — the v4 session-scoping mechanism. It is a MODULE-LEVEL
        function (``from langfuse import propagate_attributes``), NOT a client method, so
        reaching it as ``getattr(client, 'propagate_attributes')`` resolves to ``None``
        and leaves session scoping dead.

        The scenario anchor is the scenario-level eval-attachment target (turn-scoped
        evals attach to their captured turn span); it is also what gives the standalone
        ``create_score(session_id=...)`` rows ``post_feedback`` posts a materialized
        Sessions-view row to surface under. The anchor span is left OPEN
        and stashed on ``run_ctx`` — ``post_feedback`` sets the per-scenario summary as
        its output and ends it (``drain_run`` ends it on the crash path). State is
        derived from ``run_ctx`` only; the whole build routes through ``swallow``
        (best-effort — a span failure can never break a sim). Returns the scenario
        anchor or ``None``.

        Async: the blocking span build runs inside via ``asyncio.to_thread`` so the
        orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, LangfuseRunContext) or run_ctx.client is None:
            return
        await offload(
            self._record_transcript,
            run_ctx,
            thread_id=thread_id,
            scenario_name=scenario_name,
            turns=turns,
            scenario_metadata=scenario_metadata,
            sink=self.name,
            what='record_transcript',
        )

    def _record_transcript(
        self,
        run_ctx: LangfuseRunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list[TranscriptTurn],
        scenario_metadata: dict[str, Any] | None,
    ) -> Any | None:
        client = run_ctx.client
        # ``start_observation`` is a declared ``Langfuse`` method → DIRECT TYPED ACCESS.
        # The ``hasattr`` runtime guard (plus the ``None`` check) makes a ``None`` client or a
        # surface lacking ``start_observation`` (older SDK / test stub) no-op without raising. (An
        # ``isinstance`` against a runtime_checkable Protocol is rejected here as an
        # "unsafe overlap" — the checker already sees ``start_observation`` on ``Langfuse`` —
        # and a widening ``client: object`` does not help, as pyright re-narrows the
        # assignment back to ``Langfuse``.)
        if client is None or not hasattr(client, 'start_observation'):
            return None
        from langfuse import propagate_attributes  # MODULE-level v4 session mechanism

        # ALLOW-LIST filtered record metadata (the shared ``build_scenario_metadata``):
        # exactly the allow-listed scenario-metadata keys + thread/experiment/agent_version run
        # identifiers, FLAT on the anchor observation — the IDENTICAL mapping the
        # LangSmith anchor emits. No backend-prefixed keys and no nested raw
        # scenario-metadata dump: that leaks prompt-engineering internals
        # (``tool_verbs`` / ``knowledge_plan``) into the trace view.
        anchor_metadata = build_scenario_metadata(
            scenario_metadata=scenario_metadata,
            thread_id=thread_id,
            experiment_name=run_ctx.experiment,
            agent_version=run_ctx.agent_version,
        )
        # Enter propagate_attributes BEFORE creating any span so the scenario anchor
        # and EVERY child inherit session_id (it sets the attribute on spans created
        # within the OTel context). session_id/trace_name are the only v4 trace-level
        # knobs; scenario_metadata rides the observation's own metadata (unconstrained).
        # The trace + anchor are named ``scenario:<name>`` (the shared record-name convention)
        # so the Traces view rows read as scenario records at a glance.
        record_name = scenario_record_name(scenario_name)
        turn_ids: dict[int, Any] = {}
        link = run_ctx.trace_link
        with propagate_attributes(session_id=thread_id, trace_name=record_name):
            scenario = client.start_observation(name=record_name, as_type='span', metadata=anchor_metadata)
            # Stash the OPEN anchor on the run context IMMEDIATELY — before any turn span
            # is built — so a mid-loop failure (swallowed by the offload guard) still
            # leaves the handle reachable: ``post_feedback`` / ``drain_run`` can end it
            # and no span leaks on the partial-failure path. The anchor deliberately
            # stays OPEN here: an OTel span cannot be updated after ``end()`` (langfuse
            # v4's ``update``/``set_trace_io`` no-op on a non-recording span), and the
            # per-scenario SUMMARY — the anchor's output — only exists once
            # ``post_feedback`` has the finalize verdicts. Same
            # root-stays-open-until-feedback lifecycle braintrust uses.
            link['anchor_span'] = scenario
            clock = _MillisecondClock()
            for turn in turns:
                clock.tick()
                turn_span = scenario.start_observation(
                    name=turn_record_name(turn['turn_idx']),
                    as_type='span',
                    metadata={
                        'turn_idx': turn['turn_idx'],
                        'should_terminate': turn['should_terminate'],
                        'turn_count': turn['turn_count'],
                    },
                )
                # Capture the turn span's observation id so turn-SCOPED evals can anchor on
                # this exact node (post_feedback's scope routing). ``id`` is a declared
                # ``LangfuseSpan`` field → direct typed access; the ``hasattr`` guard keeps
                # the never-raise skip for a surface (older SDK / test stub) that lacks it —
                # an uncaptured turn degrades that eval to the scenario anchor.
                if hasattr(turn_span, 'id') and turn_span.id is not None:
                    turn_ids[turn['turn_idx']] = turn_span.id
                _emit_agent_span_with_tools(turn_span, turn, clock)
                clock.tick()
                turn_span.start_observation(
                    name='client',
                    as_type='span',
                    input=turn['agent_response'],
                    output=turn['output_user_message'],
                ).end()
                turn_span.end()
        # Capture the scenario-anchor trace handle so ``post_feedback`` can link this run's
        # trace to its dataset item (materializing the set's Dataset Run) and re-target its
        # scores. ``trace_id`` / ``id`` are declared ``LangfuseSpan`` fields → DIRECT TYPED
        # ACCESS, guarded by ``hasattr`` so a span surface that exposes neither (older SDK /
        # test stub) leaves the holder empty ⇒ linkage degrades to session-only. Stored on
        # ``run_ctx`` (per-scenario state), never on the shared sink instance.
        if hasattr(scenario, 'trace_id') and scenario.trace_id is not None:
            link['trace_id'] = scenario.trace_id
            folder = (scenario_metadata or {}).get('scenario_folder') or dataset_join_key(scenario_folder=run_ctx.scenario_folder, scenario_name=scenario_name)
            self._recorded_traces.append((str(folder), str(scenario.trace_id)))
        if hasattr(scenario, 'id') and scenario.id is not None:
            link['observation_id'] = scenario.id
        if turn_ids:
            link['turn_observations'] = turn_ids
        # Stash the replay payload the native run_experiment registration consumes (via
        # ``post_feedback`` -> the per-SET buffer -> ``end_experiment``): the scenario's final
        # agent response as the item ``output`` column, PLUS the full ``turns`` transcript and
        # ``thread_id`` (session). The replay task re-emits those ``turns`` under the native
        # ``experiment-item-run`` span and stamps ``thread_id`` as its session — so each
        # experiment item shows the same turn-by-turn detail + session link this rich trace
        # carries here. ``thread_id`` is always
        # available; ``turns`` is omitted on an empty transcript ⇒ output-only replay.
        run_ctx.replay['thread_id'] = thread_id
        # The allow-listed scenario metadata (same mapping the session-trace anchor carries) is
        # stashed for the replay so the experiment-run trace gets it too: metadata sits on both
        # the dataset item and the trace.
        run_ctx.replay['metadata'] = anchor_metadata
        if turns:
            run_ctx.replay['output'] = turns[-1]['agent_response']
            run_ctx.replay['turns'] = turns
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
        """Log per-policy + rollup TYPED scores against the run's axes (best-effort).

        Per-policy results route through ``eval_mapping.map_eval_score`` so every
        ``create_score`` carries a proper ``data_type`` (BOOLEAN / NUMERIC, or
        CATEGORICAL + a numeric companion); scope routing places turn-scoped scores on
        their turn observation ONLY. Also sets the per-scenario summary as the anchor's
        output and ends the stashed anchor span.

        Async: the blocking metric-compute + ``create_score`` POSTs run inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, LangfuseRunContext):
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

    async def drain_run(self, run_ctx: RunContext) -> None:
        """Per-scenario teardown: flush the Langfuse client (idempotent).

        Guaranteed by the orchestrator's scenario scope on exit (normal AND exception).
        Mirrors the ``flush`` already coupled into ``post_feedback`` (where the
        ``create_score`` POSTs flush against the session) — firing here too is harmless
        (``flush`` is repeat-safe; nothing buffered ⇒ a cheap no-op) and ensures the
        scenario's traces surface even on a path where ``post_feedback`` was skipped.
        No-op when tracing isn't live (``run_ctx`` is ``None`` or its ``client`` is
        ``None``). Distinct from ``close``, which shuts the client down once per process.

        Async: the blocking ``flush`` runs inside via ``asyncio.to_thread``; errors
        are swallowed so this never raises.
        """
        if not isinstance(run_ctx, LangfuseRunContext) or run_ctx.client is None:
            return
        client = run_ctx.client
        link = run_ctx.trace_link

        def _flush() -> None:
            # CRASH-path closure of the scenario anchor: ``record_transcript`` leaves the
            # anchor span OPEN (so ``post_feedback`` can set the summary output before
            # ending it). On the happy path ``post_feedback`` pops + ends it, so this pop
            # finds nothing (idempotent); on the crash path where ``post_feedback`` never
            # ran, ending it here guarantees no span leaks past the scenario scope.
            anchor = link.pop('anchor_span', None)
            if anchor is not None:
                try:
                    anchor.end()
                except Exception:
                    logger.exception('Langfuse anchor-span end failed at scenario drain; span may be lost.')
            try:
                client.flush()
            except Exception:
                logger.exception('Langfuse flush failed at scenario drain; traces may be delayed or lost.')

        await asyncio.to_thread(_flush)

    async def close(self) -> None:
        """Process-teardown drain (async, no-op).

        Langfuse couples its per-scenario draining into ``post_feedback`` (the
        ``create_score`` POSTs flush against the run's session as they are made),
        so there is no buffered process-wide state left to drain at teardown.
        """

    async def begin_experiment(self, *, dataset_name: str, run_name: str) -> None:
        """Open the set-level Dataset Run BEFORE the scenario gather (additive seam).

        Implements the
        :class:`~plurai.observability.sink_protocol.ExperimentLifecycleSink` STRUCTURALLY
        (defined directly on the sink, no inheritance — adding methods never affects
        ``ObservabilitySink`` conformance). The Dataset Run is NOT created here: it is
        registered NATIVELY at the END of the gather by :meth:`_run_experiment_replay`. This
        STASHES the per-SET handles the runner supplies — ``dataset_name`` (the dataset whose
        items the run replays over) and ``run_name`` (the run label) — and RESETS the per-SET
        replay buffer for a fresh set. Set once, before any ``prepare_run``; read-only during
        the bounded-parallel gather (the buffer-reset here runs single-threaded, before any
        concurrent insert). A plain assignment can't raise; the runner also ``isinstance``-gates
        + wraps this best-effort.

        ``LANGFUSE__DATASET`` (``cfg.dataset``) heads the cascade, so the Dataset Run
        replays over the SAME dataset the upload filled.
        """
        self._experiment_dataset_name = self._config.dataset or dataset_name
        self._experiment_run_name = run_name
        self._replay_rows = {}
        self._experiment_url = None

    async def end_experiment(self) -> None:
        """Register the set-level Dataset Run NATIVELY AFTER the gather (best-effort).

        Counterpart to :meth:`begin_experiment`. Snapshots + clears the per-SET replay buffer
        (safe without a lock — this runs AFTER the gather, so no ``post_feedback`` insert is in
        flight), then — if the seam was driven and any scenario contributed a row — dispatches
        :meth:`_run_experiment_replay` via ``asyncio.to_thread`` (the synchronous
        ``client.run_experiment`` must run off the event loop). That call is what makes the run
        render as a first-class experiment in the langfuse UI. Always clears the stashed handles
        for a clean teardown; never raises.
        """
        run_name = self._experiment_run_name
        dataset_name = self._experiment_dataset_name
        rows = self._replay_rows
        self._replay_rows = {}
        self._experiment_dataset_name = None
        self._experiment_run_name = None
        if not run_name or not dataset_name or not rows:
            return
        await asyncio.to_thread(self._run_experiment_replay, dataset_name=dataset_name, run_name=run_name, rows=rows)

    def _post_feedback_sync(
        self,
        thread: CreateThreadResponse,
        finalize: FinalizeResponse,
        run_ctx: LangfuseRunContext,
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
            thread_id=thread_id,
            policy_results=policy_results,
            metrics=metrics,
            final_world_state=final_world_state,
            scenario_metadata=md,
            sink=self.name,
            what='post_feedback',
        )

    def _post_feedback(
        self,
        run_ctx: LangfuseRunContext,
        *,
        thread_id: str,
        policy_results: dict,
        metrics: list,
        final_world_state: dict[str, Any] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        client = run_ctx.client
        # ``client`` is the disabled sentinel (``None``) when tracing isn't live; the
        # ``create_score`` / ``flush`` calls below are DECLARED methods of the concretely
        # typed ``Langfuse`` client, so direct typed access (no getattr) is used once
        # ``client`` is known non-``None``.
        if client is None:
            return
        swallow = _langfuse_error_types()
        create_score = client.create_score

        # Score the RICH transcript axes only: ``session_id`` (Sessions view + the always-present
        # default) and the per-item ``trace_id`` (+ anchor ``observation_id``) captured by
        # ``record_transcript``. The per-SET Dataset Run and its experiment-VIEW item scores are
        # registered NATIVELY by ``end_experiment`` via ``run_experiment`` — so post_feedback
        # links no items here (no ``dataset_run_items.create`` / ``dataset_run_id`` axis).
        # Each score is still a SINGLE-TARGET call: a langfuse v4 score references exactly ONE
        # object — the server DROPS an event that carries more than one of {session_id, trace_id,
        # dataset_run_id}, so a multi-linked call would silently lose the score.
        link = run_ctx.trace_link
        trace_id = link.get('trace_id')
        observation_id = link.get('observation_id')
        turn_observations: dict[int, Any] = link.get('turn_observations') or {}
        scenario_targets: list[dict[str, Any]] = [{'session_id': thread_id}]
        if trace_id is not None:
            item_target: dict[str, Any] = {'trace_id': trace_id}
            if observation_id is not None:
                item_target['observation_id'] = observation_id
            scenario_targets.append(item_target)

        # Materialize the TYPED score rows ONCE — (TypedScore, turn_target) pairs reused for the
        # live ``create_score`` axes and (scenario-scoped rows only) the per-SET replay buffer
        # ``end_experiment`` feeds to ``run_experiment``. Per-policy results route through the
        # shared ``map_eval_score``; rollups RE-SHAPE through ``map_rollup_score`` (their
        # numeric/native split already matches TypedScore) and are ALWAYS scenario-scoped — they
        # summarize the whole run, so the scenario pass/fail verdict posts exactly ONCE per axis.
        # SCOPE routing (the per-turn stream goes live) — EXCLUSIVE placement (an eval is
        # placed in exactly one place, never posted twice):
        #   * A scenario-scoped policy posts ONCE on the scenario axes (turn_target=None).
        #   * A turn-scoped policy posts one score per captured per-turn entry (``turns``
        #     sub-list) on that turn's observation and NOT on the scenario — N entries ⇒ N turn
        #     scores. ``turn['turn_idx']`` is the SAME 0-based key ``record_transcript`` stored
        #     in ``turn_observations`` (langfuse.py turn_ids[turn['turn_idx']]) and the SDK
        #     loop's ``range(max_turns)`` index — off-by-one is impossible by construction.
        #   * A per-turn entry whose turn was never captured (idx absent from turn_observations)
        #     degrades to scenario level (turn_target=None) rather than silently dropped; a
        #     turn-scoped policy that emitted no per-turn stream falls back to its rolled-up
        #     verdict on the scenario so it is never dropped.
        #   * The single overall scenario pass/fail is the separate ``scenario_passed`` rollup.
        rows: list[tuple[TypedScore, dict[str, Any] | None]] = []
        # REPLAY turn scores: a turn-scoped policy's per-turn scores keyed by turn_idx, so the
        # experiment-run (replay) trace can place them on ITS OWN re-emitted turn observations
        # (the Experiments-tab trace shows turn evals on their turn). Collected
        # INDEPENDENTLY of the live session-trace placement above (which needs the captured
        # session turn observation) — the replay re-emits every turn fresh, indexed by turn_idx.
        replay_turn_scores: dict[int, list[_ScorePayload]] = {}
        for policy_name, result in eval_results(policy_results).items():
            if not isinstance(result, dict):
                continue
            per_turn = per_turn_results(policy_name, result) if eval_scope(result) == 'turn' else []
            if per_turn and trace_id is not None:
                for ts in per_turn:
                    turn_target: dict[str, Any] | None = None
                    if ts.turn_idx in turn_observations:
                        turn_target = {'trace_id': trace_id, 'observation_id': turn_observations[ts.turn_idx]}
                    rows.append((ts.score, turn_target))
            else:
                rows.append((map_eval_score(policy_name, result), None))
            for ts in per_turn:
                replay_turn_scores.setdefault(ts.turn_idx, []).extend(_score_payloads(ts.score))
        for m in metrics:
            rows.append((map_rollup_score(m.key, m.score, m.value, m.comment), None))

        for ts, turn_target in rows:
            for target in [turn_target] if turn_target is not None else scenario_targets:
                for payload in _score_payloads(ts):
                    try:
                        # Branch on the data_type DISCRIMINANT so pyright narrows the payload
                        # union to the matching create_score overload (float ↔ NUMERIC/BOOLEAN,
                        # str ↔ CATEGORICAL) — both branches make the identical call shape.
                        if payload[3] == 'CATEGORICAL':
                            create_score(name=payload[0], value=payload[1], comment=payload[2], data_type=payload[3], **target)
                        else:
                            create_score(name=payload[0], value=payload[1], comment=payload[2], data_type=payload[3], **target)
                    except swallow:
                        logger.exception('Langfuse create_score failed for %r (target=%r).', payload[0], sorted(target))

        # The per-scenario SUMMARY (one-coherent-record contract) — built ONCE and reused as the
        # output of BOTH the live session trace (the anchor span below) AND the experiment-run
        # replay trace (the ``_task`` return, via the replay buffer).
        summary = build_scenario_summary(
            results=eval_results(policy_results),
            scenario_metadata=scenario_metadata,
            final_world_state=final_world_state,
            turn_count=len(run_ctx.replay.get('turns') or []),
            policy_results=policy_results,
        )

        # Replay buffer feeds the experiment-run trace: SCENARIO-scoped
        # payloads become the item-level (trace-root) evaluations; the TURN-scoped payloads
        # (``replay_turn_scores``) are placed on the replay trace's own turn observations; and the
        # SUMMARY becomes the trace output. So the Experiments-tab trace carries whole-conversation
        # evals on the scenario, turn evals on their turn, and output = summary — all on ONE trace.
        replay_payloads: list[_ScorePayload] = []
        for ts, turn_target in rows:
            if turn_target is None:
                replay_payloads.extend(_score_payloads(ts))
        self._buffer_replay_row(run_ctx, replay_payloads, turn_scores=replay_turn_scores, summary=summary)

        # Per-scenario SUMMARY as the trace output: set on the STILL-OPEN anchor span stashed by
        # ``record_transcript`` (langfuse derives the trace's output from its root observation),
        # then END it — the anchor cannot be ended at record time because ``update`` no-ops on a
        # non-recording span and the summary needs the finalize verdicts only available here.
        # ``pop`` keeps the close idempotent against ``drain_run``'s crash-path end.
        anchor = link.pop('anchor_span', None)
        if anchor is not None and hasattr(anchor, 'end'):
            try:
                if hasattr(anchor, 'update'):
                    anchor.update(output=summary)
                anchor.end()
            except swallow:
                logger.exception('Langfuse anchor summary update/end failed for session %r.', thread_id)

        try:
            client.flush()
        except swallow:
            logger.exception('Langfuse flush failed after scoring session %r.', thread_id)

    def _buffer_replay_row(
        self,
        run_ctx: LangfuseRunContext,
        rows: list[_ScorePayload],
        *,
        turn_scores: dict[int, list[_ScorePayload]] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Append this scenario's ``{summary, scores, turn_scores, turns, thread_id}`` to the per-SET buffer.

        Keyed by the scenario FOLDER (``dataset_join_key`` over the run context) — the same join key the
        dataset items carry in ``metadata['scenario_folder']`` (set by
        :func:`sync_scenarios_to_langfuse`), so :meth:`_run_experiment_replay`'s task /
        evaluator can map each fetched ``DatasetItem`` back to its captured ``output`` (the
        item column) + ``scores`` (the evaluations) + ``turns`` / ``thread_id`` (re-emitted as
        the experiment item's turn tree + session). No-op unless the set-level seam is active
        (``begin_experiment`` stashed the run + dataset names). Concurrent scenarios run this OFF
        the event loop (``post_feedback`` -> ``asyncio.to_thread``) and share the dict, but the
        insert needs no lock: each writes a DISTINCT folder key and a single ``dict[key] = value``
        is atomic under the GIL (see ``__init__``).
        """
        run_name = self._experiment_run_name
        dataset_name = self._experiment_dataset_name
        folder = dataset_join_key(scenario_folder=run_ctx.scenario_folder, scenario_name=run_ctx.scenario_name)
        if not run_name or not dataset_name:
            return  # the set-level experiment seam is inactive — the normal non-experiment path
        if not folder:
            # Distinct from the above: we HAVE an experiment but no join key, so this
            # scenario's whole replay row (summary, scores, turn tree, session) is dropped.
            # Near-unreachable while the orchestrator threads the folder, but a
            # silent total loss is not something to leave to an invariant nothing asserts.
            logger.warning(
                f'Langfuse: no scenario folder or name for thread {run_ctx.thread_id!r}; dropping its replay row — '
                f'its scores and turn tree will be missing from experiment run {run_name!r}.'
            )
            return
        replay = run_ctx.replay
        row: dict[str, Any] = {
            # The SUMMARY is the experiment-run trace output;
            # the bare last-message output is kept as a fallback for a summary-less crash path.
            'summary': summary,
            'output': replay.get('output', ''),
            'scores': rows,
            'turn_scores': turn_scores or {},
            'turns': replay.get('turns') or [],
            'thread_id': replay.get('thread_id') or '',
            # Allow-listed scenario metadata stamped onto the experiment-run trace.
            'metadata': replay.get('metadata') or {},
        }
        self._replay_rows[folder] = row

    def _run_experiment_replay(self, *, dataset_name: str, run_name: str, rows: dict[str, dict[str, Any]]) -> None:
        """Register the per-SET Dataset Run NATIVELY by replaying the captured rows.

        Calls ``client.run_experiment`` with a REPLAY task (returns each item's captured
        ``output`` AND re-emits its turn-by-turn span tree + session under the native
        ``experiment-item-run`` span — see :meth:`_emit_replay_turns`) and an evaluator (returns
        each item's ``scores`` as :class:`~langfuse.experiment.Evaluation` objects).
        Running it through the SDK's own experiment machinery is what makes the run render fully
        in the langfuse experiment UI (``sdk-experiment`` environment, ``langfuse.experiment.*``
        span attributes, item-level scores in the Scores view, plus — via the task — the
        per-item turn tree and session link) — the lower-level ``dataset_run_items.create``
        linkage does NOT (the UI does not aggregate an externally-anchored run as a first-class
        run).

        Maps fetched ``DatasetItem``s back to their captured rows via
        ``metadata['scenario_folder']`` (the deterministic join key both sides share). Fully
        self-guarding + best-effort: builds the gated client (``None`` ⇒ skip), ``hasattr``-guards
        the declared ``run_experiment`` / ``get_dataset`` methods so an older surface / test stub
        no-ops WITHOUT raising (an ``AttributeError`` is not swallowable), and wraps the SDK calls
        in the narrow ``swallow`` tuple. Already OFF the event loop (``end_experiment`` dispatches
        this via ``asyncio.to_thread``), so the synchronous ``run_experiment`` is safe here.
        """
        # Build with the config environment (NOT the experiment trace's forced
        # ``sdk-experiment``): the item scores langfuse stamps from this client's env must share
        # the env of the SESSION + the session-attached score copies + the rich transcript trace
        # (all ``config.environment``) so they stay visible in the env the user browses — pinning
        # them to ``sdk-experiment`` would hide them from the trace's env-scoped Scores tab.
        client = configure_tracing(self._config, agent_name=run_name, experiment_name=run_name, dataset_name=dataset_name)
        if client is None:
            return
        # ``run_experiment`` / ``get_dataset`` are declared ``Langfuse`` methods → DIRECT TYPED
        # ACCESS; the ``hasattr`` guards preserve a never-raise skip for a surface (older SDK /
        # test stub) that lacks them. (An ``isinstance`` against a runtime_checkable Protocol is
        # rejected as an "unsafe overlap" — the checker already sees the methods on ``Langfuse``.)
        if not hasattr(client, 'run_experiment') or not hasattr(client, 'get_dataset'):
            return
        swallow = _langfuse_error_types()
        from langfuse import Evaluation

        try:
            dataset = client.get_dataset(dataset_name)
        except swallow:
            logger.exception('Langfuse get_dataset(%r) failed; skipping native experiment registration.', dataset_name)
            return

        def _folder_of(item: ExperimentItem) -> str | None:
            md = item.get('metadata') if isinstance(item, dict) else item.metadata
            return md.get('scenario_folder') if isinstance(md, dict) else None

        def _display_name_of(item: ExperimentItem) -> str | None:
            md = item.get('metadata') if isinstance(item, dict) else item.metadata
            return md.get('scenario_name') if isinstance(md, dict) else None

        def _task(*, item: ExperimentItem, **_: Any) -> Any:
            folder = _folder_of(item)
            if folder is not None:
                # Retitle the ACTIVE generic ``experiment-item-run`` span to the shared
                # ``scenario:<name>`` record name (langfuse offers no
                # rename knob, but this task runs synchronously inside that span).
                #
                # Titled with the DISPLAY name, joined on the FOLDER. Using the folder for
                # both would make the Experiments view disagree with the Sessions/Traces
                # view for the same scenario — those rows are titled from ``scenario_name``
                # (``open_session_trace`` / ``record_transcript``), and the invariant that
                # they read identically is stated on ``open_session_trace``. Falls back to
                # the folder when the item carries no display name.
                _rename_active_experiment_span(scenario_record_name(_display_name_of(item) or folder))
            row = rows.get(folder) if folder is not None else None
            if not row:
                return ''
            # Stamp the allow-listed scenario metadata onto the experiment-run TRACE, so it sits
            # on both the dataset item and the trace. The task runs inside the
            # ``experiment-item-run`` root span, so updating the current span's metadata sets the
            # trace's metadata — mirroring how ``record_transcript`` sets it on the session
            # trace's root observation. Best-effort: a surface lacking the v4 method no-ops.
            md = row.get('metadata') or {}
            if md and hasattr(client, 'update_current_span'):
                client.update_current_span(metadata=md)
            # Re-emit the turn-by-turn tree + session INSIDE the active ``experiment-item-run``
            # span (langfuse's ``_process_experiment_item`` runs this task with that span open
            # and the experiment-attribute context propagating) so the experiment item shows the
            # same turn detail + session link the rich transcript trace carries — not just a thin
            # output-only span. The TURN-scoped scores are placed on these re-emitted turn
            # observations (turn evals on their turn). See :meth:`_emit_replay_turns`.
            self._emit_replay_turns(
                client,
                thread_id=str(row.get('thread_id') or ''),
                turns=row.get('turns') or [],
                turn_scores=row.get('turn_scores') or {},
            )
            # The experiment-run TRACE OUTPUT is the per-scenario SUMMARY; fall back to the bare last-message output if no summary was captured.
            summary = row.get('summary')
            return summary if summary is not None else str(row.get('output', ''))

        def _evaluator(*, input: Any, output: Any, expected_output: Any, metadata: dict[str, Any] | None, **_: Any) -> list[Evaluation]:
            folder = metadata.get('scenario_folder') if isinstance(metadata, dict) else None
            row = rows.get(folder) if folder is not None else None
            if not row:
                return []
            # The buffered payloads already carry the TYPED form (the same ``_score_payloads``
            # expansion the live axes posted — incl. the CATEGORICAL string + its numeric
            # companion), so the experiment view renders the identical typed scores.
            return [Evaluation(name=name, value=value, comment=comment, data_type=data_type) for name, value, comment, data_type in row.get('scores', [])]

        # Richer, self-describing run description (shown in the dataset's Runs tab): name the
        # source, the scenario count, and the per-scenario eval-score count so the run is legible
        # at a glance without opening an item. ``rows`` is non-empty here (``end_experiment``
        # guards), so the first row is a safe sample of the (uniform) score count.
        scenario_count = len(rows)
        score_count = len(next(iter(rows.values()), {}).get('scores', []))
        description = (
            f'Plurai client-side scenario-set run over dataset {dataset_name!r}: '
            f'{scenario_count} scenario(s), {score_count} eval score(s) each '
            f'(policy evaluations + rollup metrics), replayed natively via run_experiment.'
        )

        try:
            result = client.run_experiment(
                name=run_name,
                run_name=run_name,
                description=description,
                # Only the items this run recorded: the dataset accumulates every scenario any run
                # uploaded, and replaying the rest would add empty rows to this run.
                data=[item for item in dataset.items if _folder_of(item) in rows],
                task=_task,
                evaluators=[_evaluator],
                metadata={'source': 'plurai-sdk', 'scenario_count': str(scenario_count)},
            )
            self._experiment_url = getattr(result, 'dataset_run_url', None) or None
            client.flush()
        except swallow:
            logger.exception('Langfuse run_experiment replay failed (run=%r, dataset=%r).', run_name, dataset_name)

    def _emit_replay_turns(
        self,
        client: Langfuse,
        *,
        thread_id: str,
        turns: list[TranscriptTurn],
        turn_scores: dict[int, list[_ScorePayload]] | None = None,
    ) -> None:
        """Re-emit the turn-by-turn span tree INSIDE the native ``experiment-item-run`` span.

        Invoked from the :meth:`_run_experiment_replay` REPLAY task, which langfuse's own
        ``_process_experiment_item`` runs SYNCHRONOUSLY while ``experiment-item-run`` is the
        ACTIVE span and an experiment-attribute propagation context is open. Creating spans here
        therefore (a) nests them under ``experiment-item-run`` (OTEL current-span parenting), and
        (b) inherits the ``sdk-experiment`` environment + ``langfuse.experiment.*`` attrs (the
        span processor stamps the propagated experiment context onto every span at start). The
        wrapping ``propagate_attributes(session_id=...)`` additionally stamps the SESSION onto
        ``experiment-item-run`` AND every child — so the experiment item renders both the
        turn-by-turn detail and a session link, mirroring the rich transcript trace ``record_transcript`` keeps in the default
        env. (This re-emits a SECOND copy of the transcript, in the experiment env — intentional:
        it makes the experiment view self-contained.)

        Best-effort: no-ops on an empty transcript / missing session, or a client surface lacking
        ``start_observation`` (older SDK / test stub — the same ``hasattr`` guard + typed access
        pattern ``record_transcript`` uses), and routes the span build through the narrow
        ``swallow`` tuple so a replay-decoration failure never aborts the experiment.
        """
        if not turns or not thread_id or not hasattr(client, 'start_observation'):
            return
        from langfuse import propagate_attributes  # MODULE-level v4 session mechanism

        scores = turn_scores or {}
        swallow = _langfuse_error_types()
        try:
            with propagate_attributes(session_id=thread_id):
                clock = _MillisecondClock()
                for turn in turns:
                    clock.tick()
                    turn_span = client.start_observation(
                        name=turn_record_name(turn['turn_idx']),
                        as_type='span',
                        metadata={
                            'turn_idx': turn['turn_idx'],
                            'should_terminate': turn['should_terminate'],
                            'turn_count': turn['turn_count'],
                        },
                    )
                    _emit_agent_span_with_tools(turn_span, turn, clock)
                    clock.tick()
                    turn_span.start_observation(
                        name='client',
                        as_type='span',
                        input=turn['agent_response'],
                        output=turn['output_user_message'],
                    ).end()
                    # Place this turn's TURN-scoped scores on its observation (turn evals on
                    # their turn), keyed by turn_idx — single-target (trace_id + observation_id of
                    # this replay turn), mirroring the live session-trace turn scoring. Branch on
                    # the data_type DISCRIMINANT (indexed, not destructured) so pyright narrows the
                    # payload union to the matching create_score overload (str ↔ CATEGORICAL, float
                    # ↔ NUMERIC/BOOLEAN).
                    target: dict[str, Any] = {'trace_id': turn_span.trace_id, 'observation_id': turn_span.id}
                    for payload in scores.get(turn['turn_idx'], []):
                        if payload[3] == 'CATEGORICAL':
                            client.create_score(name=payload[0], value=payload[1], comment=payload[2], data_type=payload[3], **target)
                        else:
                            client.create_score(name=payload[0], value=payload[1], comment=payload[2], data_type=payload[3], **target)
                    turn_span.end()
        except swallow:
            logger.exception('Langfuse replay turn-span emission failed (session=%r).', thread_id)

    def overwrite_scores(self, items: Sequence[ScoreOverwrite]) -> list[ScoreOverwriteOutcome]:
        """Overwrite each corrected eval's recorded scores IN PLACE; never raises.

        Per thread: look up every score of the corrected evals' names the run posted
        (:func:`_recorded_scores`), then re-post each one with its OWN ``score_id`` on its OWN
        subject (:func:`_overwrite_eval`). The data type never changes: a correction flips
        ``violated`` and leaves the cell's ``label``, which is what fixes the kind.
        """
        if not self._config.enabled or not _langfuse_available():
            return every_eval(items, 'failed', 'langfuse is not enabled or not installed')
        client = _build_client(self._config)
        swallow = _langfuse_error_types()
        outcomes: list[ScoreOverwriteOutcome] = []
        for item in items:
            placements = {ev.policy: eval_score_placements(ev.policy, ev.cell) for ev in item.evals}
            names = {payload[0] for rows in placements.values() for _, ts in rows for payload in _score_payloads(ts)}
            try:
                recorded = _recorded_scores(client, item.thread_id, names)
            except AttributeError:
                # langfuse 3.x has no scores_v3 / v2 observations API, and 5.x removed scores_v3.
                outcomes.extend(every_eval([item], 'failed', f'langfuse {_langfuse_version()} cannot look scores up; install langfuse>=4,<5'))
                continue
            except swallow as exc:
                outcomes.extend(every_eval([item], 'failed', f'score lookup failed: {exc}'))
                continue
            for ev in item.evals:
                written = _overwrite_eval(client, recorded, placements[ev.policy])
                outcomes.append(eval_outcome(item.thread_id, ev.policy, written=written, failures=()))
        try:
            client.flush()
        except swallow:
            logger.exception('Langfuse flush failed after overwriting scores.')
        return outcomes

    def sync_scenarios(
        self,
        *,
        dataset_name: str,
        dataset_description: str | None,
        scenarios: list[ScenarioUploadEntry],
        existing_metadata: dict[str, Any],
    ) -> ProviderSyncResult:
        """Upload a generated scenario batch into a Langfuse Dataset.

        Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`;
        a disabled/SDK-missing no-op returns ``ok=True, uploaded=0``.

        ``LANGFUSE__DATASET`` (``cfg.dataset``) wins over the passed name, matching the
        head-of-cascade precedence the per-scenario anchor, the Dataset Run and
        ``dataset_url`` use — otherwise the upload fills one dataset while the run
        references another.

        The prior-handle guard is the other half: a recorded item id is upserted BY ID,
        which would update the item in the OLD dataset instead of creating one here —
        while still reporting "uploaded N/N". It fires only on EVIDENCE of divergence (a
        recorded name that differs); an ABSENT name is unknown provenance, and dropping
        handles there would duplicate items in the same dataset.
        """
        resolved_name = self._config.dataset or dataset_name
        recorded_name = existing_metadata.get('langfuse_dataset_name')
        prior_items = existing_metadata.get('langfuse_dataset_items')
        if recorded_name is not None and recorded_name != resolved_name:
            logger.warning(
                f'Langfuse: the recorded dataset items belong to dataset {recorded_name!r} but this upload '
                f'targets {resolved_name!r} (LANGFUSE__DATASET override). Creating FRESH items in '
                f'{resolved_name!r}; the items in {recorded_name!r} are left untouched and are no longer tracked.'
            )
            prior_items = None
        return sync_scenarios_to_langfuse(
            self._config,
            dataset_name=resolved_name,
            dataset_description=dataset_description,
            scenarios=scenarios,  # type: ignore[arg-type]  # ScenarioUploadEntry duck-types the {folder_name, inputs, ...} dict shape
            existing_items=prior_items,
        )
