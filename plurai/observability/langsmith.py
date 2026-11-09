"""Client-side LangSmith Experiment recording for plurai.Orchestrator.

After ``Orchestrator.run()`` finalizes a scenario, the backend returns
``policy_evaluation_results`` in the FinalizeResponse. The orchestrator owns the
whole outer conversation as plain strings and the customer's LangSmith key lives
client-side, so recording MUST happen client-side — the server doesn't (and
shouldn't) hold the customer's LangSmith key.

Recording starts with ONE set-level step, then a four-step pipeline per scenario:

  0. **Pre-flight (set-level, before any scenario):** the runner uploads the whole
     scenario set as Examples — see ``requires_preflight_dataset`` on
     :class:`LangSmithSink` and ``run_simulation._run_gathered_async``. This is not an
     optimization. A LangSmith experiment session pins its example set to the dataset
     version as of session CREATION, and neither ``reference_dataset_id`` nor that
     version can be set post-hoc; so any example created after the first scenario opens
     its session is invisible to the Experiments grid, which then renders "No results
     found" forever while the header aggregates still compute (they read runs directly).
     Every example must exist before any project does.

Then, per scenario:
  1. Get-or-create a **Dataset** (named after experiment_name / agent_name).
  2. Get-or-create an **Example** within that dataset, deduplicated by
     ``metadata.scenario_name`` so re-runs of the same scenario reuse the
     same Example — then, and only then, get-or-create the **Project** carrying
     ``reference_dataset_id`` (order matters, per step 0). After a pre-flight both
     are pure reads.
  3. ``record_transcript`` builds a 3-level **RunTree** from orchestrator-owned
     strings (scenario anchor -> per-turn -> {agent, client}); the anchor is
     constructed with ``reference_example_id=example_id`` so it shows up in the
     LangSmith Experiments grid for that Example, and with the pre-allocated
     ``run_ctx.anchor_run_id`` so feedback can target it directly.
  4. Post per-policy + rollup ``Feedback`` rows, TYPED via the shared
     ``eval_mapping.map_eval_score`` (numeric score + native value + reasoning
     comment) and SCOPE-routed: scenario-scoped rows (the default) land on
     the **scenario anchor root**; turn-scoped rows land on their **turn node**
     (captured per-turn run ids) — exactly one placement per row. The anchor's
     OUTPUT is then set to the per-scenario summary (write-only ``update_run``).

When ``langsmith`` isn't installed every public function silently no-ops; the
optional extra is ``pip install plurai-simulation-sdk[langsmith]``.

Name-collision note: this module is ``plurai.observability.langsmith`` and shares
its leaf name with the third-party ``langsmith`` package it lazily imports. Under
Python 3 absolute-import semantics a bare ``import langsmith`` here resolves to the
installed site-packages distribution, NOT to this module. Every WITHIN-package import is fully
qualified (``from plurai.observability... import ...``) so nothing self-shadows.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from plurai.observability.errors import offload, swallow
from plurai.observability.eval_mapping import build_scenario_metadata, eval_scope, map_eval_score, per_turn_results, scenario_record_name, turn_record_name
from plurai.observability.feedback import build_scenario_summary, map_rollup_score
from plurai.observability.score_correction import ScoreOverwrite, ScoreOverwriteOutcome, eval_outcome, eval_score_placements, every_eval
from plurai.observability.model_calls import ModelCall, openai_tool_calls
from plurai.observability.sink_protocol import RunContext, RunLinks
from plurai.observability.tool_call_recovery import tool_span_metadata
from plurai.observability.dataset_types import SCENARIO_FOLDER_KEY, ProviderSyncResult, dataset_join_key
from plurai.observability.rollup import (
    agent_actions_of,
    agent_states_of,
    chatbot_messages_of,
    compute_rollup_metrics,
    eval_results,
)
from plurai.orchestrator.models import CreateThreadResponse, FinalizeResponse, ScenarioUploadEntry

if TYPE_CHECKING:
    import langsmith
    from langsmith.schemas import Dataset, Example

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LangSmithRunContext:
    """LangSmith per-scenario run context.

    ``example_id`` is the dataset-Example linkage key: ``record_transcript``
    constructs the scenario anchor ``RunTree`` with ``reference_example_id=example_id``
    so the anchor shows up in the LangSmith Experiments grid for that Example.
    ``anchor_run_id`` is the pre-allocated id of that anchor root — generated in
    ``prepare_run`` BEFORE the anchor is built so ``post_feedback`` can attach
    per-policy + rollup feedback to the single scenario anchor unconditionally
    (no run lookup, no race). ``client`` is the per-scenario LangSmith ``Client``
    (built fresh in ``_prepare_run`` — never cached on the sink); ``project_name``
    is the experiment project the anchor is posted into.
    """

    client: langsmith.Client | None  # built fresh per scenario, NOT cached on the sink
    project_name: str | None
    example_id: uuid.UUID | str | None  # opaque to the orchestrator
    scenario_name: str
    thread_id: str
    anchor_run_id: uuid.UUID | str | None = None  # pre-allocated scenario-anchor root id (eval target)
    tags: tuple[str, ...] = ()
    #: The operator's agent-build tag (CLI ``--agent-version``), stamped into the anchor
    #: record metadata by ``record_transcript`` ONLY when supplied (None ⇒ untagged).
    agent_version: str | None = None
    #: The dataset Example's inputs (the full scenario: first_talk + world-state + agent context +
    #: allow-listed metadata). ``record_transcript`` sets these as the anchor RUN's inputs so the
    #: experiment row shows the FULL scenario input (matching the linked Example and the Langfuse
    #: trace), not a bare ``{scenario_name, thread_id}`` stub.
    example_inputs: dict[str, Any] = field(default_factory=dict)
    #: Mutable per-scenario holder mapping ``turn_idx`` -> the POSTED per-turn run id.
    #: ``record_transcript`` fills it as it builds the turn children; ``post_feedback``
    #: reads it to anchor turn-SCOPED evals on their turn node (scope routing in
    #: ``post_experiment_feedback`` — ``turn_runs.get(turn_idx, anchor)`` per per-turn entry).
    #: Same frozen-dataclass-with-mutable-holder pattern
    #: as Langfuse's ``trace_link``: the field REFERENCE never changes, the dict
    #: CONTENTS are per-scenario state mutated in place. Empty when no transcript was
    #: recorded ⇒ every eval degrades to the scenario anchor.
    turn_runs: dict[int, uuid.UUID | str] = field(default_factory=dict)
    #: Mutable 1-element holder for the scenario-anchor ``RunTree``. ``record_transcript`` builds
    #: it (so children can be ``create_child``'d and ``turn_runs`` captured) and STASHES it here
    #: WITHOUT ending/posting it; ``_post_summary`` (in ``post_feedback``) ends it with the
    #: per-scenario summary as its outputs and posts it. This is required because an experiment
    #: run (``reference_example_id`` set) accepts its outputs ONLY at CREATE — ``update_run`` to
    #: outputs is ignored — so the summary must ride the anchor's create. Same frozen-dataclass-
    #: with-mutable-holder pattern as ``turn_runs``. Empty when no transcript was recorded.
    anchor_holder: list[Any] = field(default_factory=list)
    #: The scenario's folder (its join identity, :func:`dataset_join_key`) when the caller passed one:
    #: the ``Trace:`` link label for a record written without run facts, as on the Langfuse sink.
    scenario_folder: str | None = None

    @property
    def enabled(self) -> bool:
        """True only when the LangSmith experiment pipeline is fully set up."""
        return self.client is not None and self.example_id is not None and self.project_name is not None


class LangSmithSinkConfig(BaseModel):
    """LangSmith observability sink config (mirrors the Plurai backend's LangSmith settings).

    Owned by this sink module: a
    frozen pydantic ``BaseModel`` that ``Settings`` nests so its
    fields source from the ``LANGSMITH__*`` nested env namespace
    (``LANGSMITH__ENABLED`` / ``LANGSMITH__API_KEY`` / ``LANGSMITH__PROJECT`` /
    ``LANGSMITH__DATASET`` / ``LANGSMITH__E2E_JUDGE_ENABLED``). Keys are PLAIN
    strings — NOT ``SecretStr`` — because the SDK resolves them straight from the
    client environment and passes them RAW to the LangSmith client.

    ``enabled`` is a native ``bool`` field coerced via pydantic's standard
    case-insensitive boolean coercion (e.g. ``1`` / ``true`` / ``yes`` / ``on`` /
    ``t`` / ``y`` are truthy; ``0`` / ``false`` / ``no`` / ``off`` / ``f`` / ``n`` are
    falsy) and an empty env value is ignored (``env_ignore_empty`` on ``Settings`` —
    the force-disable trick blanks ``LANGSMITH__ENABLED`` to fall back to the
    ``False`` default).
    Credential fields coerce empty-string -> ``None``. ``e2e_judge_enabled`` is a
    native ``bool`` defaulting ``True`` (an empty env value is ignored; an explicit
    falsy spelling disables it). LangSmith enables UNIFORMLY via the explicit
    ``LANGSMITH__ENABLED`` flag (like the other three sinks); there is no
    key-presence auto-enable.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    api_key: str | None = None
    project: str | None = None
    dataset: str | None = None
    e2e_judge_enabled: bool = True

    @field_validator('api_key', 'project', 'dataset', mode='before')
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        """Coerce an empty / whitespace-only credential string to ``None``."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode='after')
    def _require_creds_when_enabled(self) -> LangSmithSinkConfig:
        """When the sink is ACTIVE, require ``api_key`` non-``None``.

        GATED on ``enabled``: ``Settings`` ALWAYS instantiates this
        nested config (even for an INACTIVE sink), so a bare required field would
        wrongly force a key for a sink that is off. The conditional-on-enabled shape
        is the only correct one — when LangSmith is the active sink the key must be
        defined; an inactive config stays unconstrained. Mirrors the arize /
        braintrust / langfuse validators (LangSmith enables via its explicit flag, so it
        needs the same required-when-enabled guard).
        """
        if self.enabled and self.api_key is None:
            raise ValueError('LangSmith is enabled but LANGSMITH__API_KEY is missing. Set LANGSMITH__API_KEY (nested) when LANGSMITH__ENABLED is set.')
        return self


def _langsmith_error_types() -> tuple[type[BaseException], ...]:
    """Return the LangSmith / HTTP exception classes worth swallowing.

    Looked up dynamically so the SDK can be imported without ``langsmith``
    installed. Any other exception (TypeError / AttributeError / programmer
    bug) propagates so the caller sees the real cause.
    """
    types: list[type[BaseException]] = []
    try:
        from langsmith.utils import LangSmithError

        types.append(LangSmithError)
    except ImportError:
        pass
    try:
        from requests.exceptions import RequestException

        types.append(RequestException)
    except ImportError:
        pass
    return tuple(types) or (OSError,)  # OSError covers raw socket errors as a last resort


def _langsmith_not_found_type() -> type[BaseException] | None:
    """Return ``LangSmithNotFoundError`` if the SDK is importable, else None.

    Pulled out as its own helper so callers can branch on "not found" cleanly
    (get-or-create) versus other LangSmith API errors (which still get
    swallowed + logged).
    """
    try:
        from langsmith.utils import LangSmithNotFoundError

        return LangSmithNotFoundError
    except ImportError:
        return None


def _langsmith_available() -> bool:
    return importlib.util.find_spec('langsmith') is not None


def build_handle(*, api_key: str) -> langsmith.Client | None:
    """Construct the per-scenario LangSmith ``Client`` the orchestrator records through.

    Returns ``None`` (no-op) when:
      - ``api_key`` is empty / None (caller can't authenticate), or
      - the ``langsmith`` package isn't importable (optional extra not installed).

    Does NOT mutate process env vars — the ``api_key`` is passed straight to the
    ``langsmith.Client`` constructor. We just construct the ``Client`` (built fresh per
    scenario; never cached on the sink) and hand it back for the RunContext.
    """
    if not api_key:
        return None
    if not _langsmith_available():
        return None

    import langsmith

    return langsmith.Client(api_key=api_key)


# The PolicyResult -> typed-score mapping is the shared
# ``eval_mapping.map_eval_score`` (imported above): every per-policy feedback row
# carries BOTH the numeric ``score`` and the typed native ``value`` it produces.


# Run and feedback ids are DERIVED from the thread id, so a later score correction
# (``LangSmithSink.overwrite_scores``) can address the exact feedback row a run posted without
# a lookup: the experiment project a lookup would need is derived on conflict and never
# persisted. A thread id is unique per scenario attempt, so the ids never collide across runs.
_PLURAI_NS = uuid.uuid5(uuid.NAMESPACE_URL, 'https://plurai.ai/sdk/observability/langsmith')


def _anchor_run_id(thread_id: str) -> uuid.UUID:
    """The scenario-anchor run id of ``thread_id``."""
    return uuid.uuid5(_PLURAI_NS, thread_id)


def _turn_run_id(thread_id: str, turn_idx: int) -> uuid.UUID:
    """The run id of turn ``turn_idx`` (0-based) of ``thread_id``."""
    return uuid.uuid5(_PLURAI_NS, f'{thread_id}/turn/{turn_idx}')


def _feedback_id(run_id: uuid.UUID | str, key: str) -> uuid.UUID:
    """The id of the ``key`` feedback row on ``run_id``."""
    return uuid.uuid5(_PLURAI_NS, f'{run_id}/{key}')


# ---------------------------------------------------------------------------
# Dataset / Example get-or-create helpers — idempotent across re-runs.
# ---------------------------------------------------------------------------


def _get_or_create_dataset(client: langsmith.Client, *, name: str) -> Dataset:
    """Read a Dataset by name; create it on first call.

    Idempotent across re-runs of the same suite: the second call hits the
    ``read_dataset`` happy path and never invokes ``create_dataset``. Only
    ``LangSmithNotFoundError`` triggers the create branch — every other
    LangSmith error propagates so the caller's ``prepare_experiment_run``
    swallow logs it and gives up on linkage cleanly.

    Race-safe under ``asyncio.gather`` parallelism: when N scenarios race
    to create the same dataset, exactly one wins and (N−1) hit a conflict
    on ``create_dataset``. The losers re-read instead of propagating —
    LangSmith doesn't expose an ``upsert`` flag on ``create_dataset``
    (unlike ``create_project``), so we close the TOCTOU window manually.
    """
    not_found = _langsmith_not_found_type()
    if not_found is None:
        # No typed not-found available (langsmith package missing or moved
        # the symbol). Let read_dataset raise on miss — the outer swallow in
        # prepare_experiment_run handles it as "dataset setup failed".
        return client.read_dataset(dataset_name=name)
    try:
        return client.read_dataset(dataset_name=name)
    except not_found:
        pass
    try:
        return client.create_dataset(
            dataset_name=name,
            description=f'Plurai simulation suite for {name}.',
        )
    except _langsmith_error_types() as create_exc:
        # Likely a parallel-scenario race: another task created the
        # dataset between our read-miss and our create. Re-read; if it's
        # there now, return it. If the re-read STILL fails with a TYPED
        # LangSmith/HTTP error (genuine outage / permissions problem),
        # surface the original create error — that's the meaningful failure
        # for the operator to debug. Narrowed to the module's typed-error
        # tuple (NOT bare ``Exception``) so a NON-network re-read bug
        # surfaces its own traceback instead of being masked by create_exc.
        try:
            return client.read_dataset(dataset_name=name)
        except _langsmith_error_types():
            raise create_exc  # noqa: B904  # surface the original create error, not the re-read miss


def _reference_dataset_is_empty(client: langsmith.Client, *, dataset_id: uuid.UUID | str) -> bool | None:
    """Whether ``dataset_id`` currently holds NO examples. ``None`` == could not tell.

    Must be sampled BEFORE this scenario's own example is created, or the answer is
    always "not empty" — the caller would be observing its own write. That timing is
    the whole reason this is a separate function rather than an inline check inside
    :func:`_ensure_experiment_project`.

    Self-guarded on purpose: a DIAGNOSTIC read must never bubble into
    ``prepare_experiment_run``'s handler, which would disable the run context and cost
    the run its Experiments linkage. A warning is never worth linkage. But the two
    failure modes get different volumes — a transport blip is noise (DEBUG), while an
    unexpected error means the probe itself is broken (``TypeError`` from a renamed
    ``list_examples`` kwarg after an SDK bump, say). The latter is a defect in OUR
    code and stays visible, because silence here reads as "the dataset is healthy".
    """
    try:
        return next(iter(client.list_examples(dataset_id=dataset_id, limit=1)), None) is None
    except _langsmith_error_types():
        logger.debug(f'LangSmith: reference-dataset probe for {dataset_id!r} failed (transport); skipping the empty-grid diagnostic.', exc_info=True)
        return None
    except Exception:  # noqa: BLE001 — never cost the run its linkage, but a probe DEFECT must be visible
        logger.warning(
            f'LangSmith: reference-dataset probe for {dataset_id!r} raised unexpectedly — the empty-grid diagnostic is NOT running for this run.',
            exc_info=True,
        )
        return None


def _read_or_create_linked_project(
    client: langsmith.Client,
    *,
    project_name: str,
    dataset_id: uuid.UUID | str,
    dataset_was_empty: bool | None,
) -> bool:
    """One read-or-create pass for ``project_name`` with ``reference_dataset_id=dataset_id``.

    Returns True when the project is properly linked for experiments (read with the
    right linkage, or freshly created carrying it), False when it exists WITHOUT the
    linkage — the caller (:func:`_ensure_experiment_project`) decides what to do about
    that: warn and honor the name, or derive a fresh one and run this pass again.
    """
    not_found = _langsmith_not_found_type()
    try:
        proj = client.read_project(project_name=project_name)
    except Exception as exc:
        if not_found is not None and isinstance(exc, not_found):
            # Race-safe under parallelism: ``upsert=True`` makes
            # ``create_project`` return the existing project instead of
            # 409-ing if another task won the race. The reference_dataset_id
            # in the upsert call is only honored when this call actually
            # creates the project; if a sibling task already created it
            # WITH the same reference_dataset_id, upsert returns that.
            # (If a sibling raced and forgot reference_dataset_id, we'd
            # land in the post-read warning branch on a subsequent run —
            # the simulation completes either way.)
            client.create_project(
                project_name=project_name,
                reference_dataset_id=dataset_id,
                description=f'Plurai simulation experiment for dataset {project_name}.',
                upsert=True,
            )
            return True
        raise
    if proj.reference_dataset_id == dataset_id:
        # REUSING a project whose reference dataset was empty when we looked is the
        # visible half of the "No results found" failure: the session is pinned to a
        # version with nothing in it. Only ``True`` warns — ``None`` means the probe
        # could not tell, and guessing would cry wolf about a healthy project.
        if dataset_was_empty is True:
            logger.warning(
                f'LangSmith project {project_name!r} was REUSED and its reference dataset '
                f'({dataset_id!r}) held NO examples when this run started — the Experiments grid '
                f'will show "No results found" for it. A project pins its example set at CREATION '
                f'time and cannot be re-pointed afterwards, so uploading examples now does not '
                f'repair THIS project: delete it (or pick a new PLURAI_EXPERIMENT_NAME) and re-run.'
            )
        return True
    return False


def _unlinked_project_warning(project_name: str, dataset_id: uuid.UUID | str) -> str:
    """The exists-without-linkage warning — the degraded-but-surviving outcome."""
    return (
        f'LangSmith project {project_name!r} already exists without '
        f'reference_dataset_id={dataset_id!r} — runs will be linked to '
        f'examples but will NOT appear in the Dataset Experiments tab. '
        f'Delete the project (or pick a new PLURAI_EXPERIMENT_NAME) '
        f'and re-run to get the experiment registration.'
    )


def _ensure_experiment_project(
    client: langsmith.Client,
    *,
    project_name: str,
    dataset_id: uuid.UUID | str,
    dataset_was_empty: bool | None = None,
    derive_on_conflict: bool = False,
) -> str:
    """Make sure a LangSmith project exists AND carries ``reference_dataset_id=dataset_id``;
    return the EFFECTIVE project name runs should be posted into.

    Why this matters: LangSmith's "Experiments" tab on a Dataset only lists
    projects whose ``reference_dataset_id`` points at that dataset. Projects
    auto-created when the tracer writes the first run land WITHOUT this
    field set — so traces show up in the project (and link to examples via
    ``reference_example_id``) but the project never appears as an
    Experiment for the dataset. The fix is to pre-create the project with
    ``reference_dataset_id`` set, BEFORE the tracer's first write.

    LangSmith's API only accepts ``reference_dataset_id`` at project creation time —
    ``update_project`` cannot set it after the fact — so a same-named project that
    already exists WITHOUT the linkage can never be repaired in place. What happens
    then depends on ``derive_on_conflict``:

    * ``False`` (the caller's name was CHOSEN BY THE CUSTOMER — ``LANGSMITH__PROJECT``
      or ``PLURAI_EXPERIMENT_NAME``): silently renaming a name the customer chose
      would scatter their runs, so log the warning and return ``project_name``
      unchanged (runs still land there and link to examples; they just never surface
      on the Experiments tab).
    * ``True`` (the name fell back to the resolved dataset name — nobody chose it): derive
      ``f'{project_name} · exp-{str(dataset_id)[:8]}'`` and run the same
      read-or-create pass ONCE against it. The suffix is deterministic per dataset,
      so parallel tasks and future runs all converge on ONE derived project — which
      keeps the Experiments grid aggregating per-scenario over time exactly as reusing
      one same-named project does. If the derived name is ALSO squatted without linkage
      (pathological), log the original warning and return ``project_name``.

    Reuse of an existing properly-linked project stays unconditional and intentional:
    one durable project accumulating runs across invocations is what makes the
    Experiments grid aggregate per-scenario over time.

    ``dataset_was_empty`` is the caller's pre-write sample from
    :func:`_reference_dataset_is_empty` (``None`` == unknown). It can only be taken
    BEFORE the caller creates its own example, which is why it arrives as an argument
    instead of being probed here.
    """
    if _read_or_create_linked_project(client, project_name=project_name, dataset_id=dataset_id, dataset_was_empty=dataset_was_empty):
        return project_name
    # Project exists but isn't tagged for this dataset (or any dataset), and
    # reference_dataset_id cannot be set post-hoc.
    if not derive_on_conflict:
        logger.warning(
            _unlinked_project_warning(project_name, dataset_id) + ' Or unset LANGSMITH__PROJECT to let the SDK derive a linked project automatically.'
        )
        return project_name
    derived = f'{project_name} · exp-{str(dataset_id)[:8]}'
    # WARNING, not info: the derivation CHANGES WHERE RUNS LAND — the operator looking
    # for their runs under the resolved name needs to learn the redirect happened.
    logger.warning(
        f'LangSmith project {project_name!r} exists without reference_dataset_id={dataset_id!r} '
        f'(the linkage can only be set at project creation) — deriving {derived!r} so runs land '
        f'in a project that DOES appear on the dataset Experiments tab. The suffix is '
        f'deterministic per dataset, so parallel tasks and future runs converge on that one project.'
    )
    if _read_or_create_linked_project(client, project_name=derived, dataset_id=dataset_id, dataset_was_empty=dataset_was_empty):
        return derived
    # Pathological: the DERIVED name is also squatted without linkage. No further
    # derivation — one deterministic fallback, then degrade as the non-derive path does. The
    # warning names BOTH squatted names so the operator sees the full picture: the
    # original conflict AND the derived candidate that was tried and found taken too.
    logger.warning(
        _unlinked_project_warning(project_name, dataset_id)
        + f' (The auto-derived fallback {derived!r} ALSO exists without that linkage, so it was not used either.)'
    )
    return project_name


def _get_or_create_example(
    client: langsmith.Client,
    *,
    dataset_id: uuid.UUID | str,
    scenario_name: str,
    inputs: dict[str, Any],
    scenario_folder: str | None = None,
) -> Example:
    """Read an Example by its dataset JOIN KEY within a dataset; create on miss.

    Identity = ``(dataset_id, folder)`` where ``folder`` is
    :func:`~plurai.observability.dataset_types.dataset_join_key` — the scenario
    folder, NOT the display ``scenario_name``. Re-running the same scenario points
    additional Runs at the same Example, which is what makes the LangSmith Experiments
    grid aggregate per-scenario across runs.

    Inputs are snapshotted on first create; subsequent calls return the existing Example
    unchanged so the comparison baseline stays stable even if the scenario YAML drifts.

    Resolution PREFERS an example created by the dataset-upload path
    (:func:`sync_scenarios_to_langsmith`), which keys on ``scenario_folder`` and snapshots
    the RICH scenario inputs (``persona``, ``agent_context``, ``initial_world_state``,
    ``first_talk``). Linking the experiment run to that uploaded example keeps this grid's
    Inputs identical to LangFuse — whose ``run_experiment`` replays the uploaded dataset
    item directly. Only when no uploaded example exists do we create one from the leaner
    runtime ``inputs``, so a bare ``run_simulation`` still gets a usable example.

    **Folder first; the name-keyed read is a fallback.** Every uploaded population this
    sink writes carries ``scenario_folder``, so filter 1 resolves any healthy dataset in a
    single read. Filter 2 exists only for datasets polluted with lean examples minted at
    runtime carrying ONLY ``{'scenario_name': <inner name>}`` and no folder stamp.

    Two properties of that shim are worth stating rather than discovering:

    * It is skipped entirely when the two identities coincide, so it costs a healthy
      dataset nothing — and it is only reachable at all when filter 1 missed, i.e. when
      the pre-flight did not cover this scenario.
    * It can MIS-JOIN. If two scenarios in a set share an inner name but differ by folder,
      it can link one scenario's run to the other's lean example. It is not safe, which is
      why it runs last.

    Two further filters must NOT be added — one is unreachable, the other unsafe:
    ``{'scenario_name': folder}`` can only match a row whose name equals its folder, and
    every writer that produces one also writes ``scenario_folder``, so filter 1 already
    returned; and ``{'scenario_folder': scenario_name}`` requires a folder stamp equal to
    some *display* name, which no writer produces for its own scenario — it could only ever
    match a DIFFERENT scenario's row, making it a mis-join surface with no upside.
    """
    folder = dataset_join_key(scenario_folder=scenario_folder, scenario_name=scenario_name)
    filters = [{SCENARIO_FOLDER_KEY: folder}]
    if scenario_name != folder:
        filters.append({'scenario_name': scenario_name})  # lean-example fallback
    for meta_filter in filters:
        example = next(iter(client.list_examples(dataset_id=dataset_id, metadata=meta_filter, limit=1)), None)
        if example is not None:
            return example
    # A create HERE means no example was pre-flighted for this scenario, and that has a
    # consequence the operator cannot otherwise see: a session pins its example set at
    # creation, so an example minted mid-gather may never render as a grid row. A silent
    # path here makes that loss invisible.
    #
    # Deliberately does NOT assert that a pre-flight ran and missed: ``Orchestrator.run``
    # is public SDK surface with no pre-flight of its own, and this function's contract
    # above calls that a supported path. Naming a fault that isn't there is how a team
    # learns to filter the one warning that must survive to catch the next regression.
    logger.warning(
        f'LangSmith: no dataset example for scenario folder {folder!r}; creating one mid-run. '
        f'A session pins its example set at creation, so this row may be MISSING from the '
        f'Experiments grid (the run is still traced and scored correctly). If a pre-flight '
        f'upload ran, it did not cover this scenario — check the scenario set is complete; '
        f'if none ran, pre-flight the set with --upload-datasets before the run.'
    )
    # outputs={} is intentional: the backend evaluates the transcript through its
    # PolicyManager, not against a fixed reference output. The Example here
    # is purely an input snapshot for the Experiments grid; agent outputs
    # live on the linked Runs (one per turn).
    #
    # BOTH keys are stamped so a later upload pass (which keys on the folder) finds and
    # updates this row instead of minting a rich duplicate beside it.
    return client.create_example(
        inputs=inputs,
        outputs={},
        dataset_id=dataset_id,
        metadata={'scenario_name': scenario_name, SCENARIO_FOLDER_KEY: folder},
    )


# ---------------------------------------------------------------------------
# Pre-run + post-run public entry points (split because the scenario anchor's
# ``reference_example_id`` must be set at RunTree-construction time — the
# linkage to the dataset Example is established when ``record_transcript``
# builds the anchor, which means we must know the example id BEFORE that).
# ---------------------------------------------------------------------------


class PreparedExperiment(NamedTuple):
    """What :func:`prepare_experiment_run` hands back on success.

    ``project_name`` is the EFFECTIVE experiment project — usually the caller's own
    ``project_name``, but the auto-derived fallback when the requested name was
    squatted without dataset linkage (see :func:`_ensure_experiment_project`). The
    caller MUST post runs into this name, or the derivation is dead code.
    """

    example: Example
    project_name: str


def prepare_experiment_run(
    client: langsmith.Client | None,
    *,
    dataset_name: str,
    project_name: str,
    scenario_name: str,
    example_inputs: dict[str, Any],
    scenario_folder: str | None = None,
    project_is_explicit: bool = False,
) -> PreparedExperiment | None:
    """Get-or-create the Dataset + experiment Project + Example for this scenario.

    Called BEFORE ``record_transcript`` so the orchestrator's run context carries
    ``example.id`` — that's what ``record_transcript`` sets as the anchor
    ``RunTree``'s ``reference_example_id`` to link the scenario to this Example.

    Three resources need to be in place before the anchor is posted, and the ORDER
    is load-bearing:
      1. **Dataset** — holds Examples (one per scenario).
      2. **Example** — what the anchor links to via ``reference_example_id``.
      3. **Project** — must be created with ``reference_dataset_id=dataset.id``
         so the anchor run shows up in the Dataset's Experiments tab. Without
         this the runs land in an untagged project and never appear there.

    **Why Example BEFORE Project.** LangSmith pins an experiment session's example
    set to the dataset version as of session CREATION, and neither
    ``reference_dataset_id`` nor its version can be set post-hoc. Creating the
    project first therefore pins the session to a version that predates every
    example — the Experiments grid renders "No results found" permanently, while
    the header aggregates still compute (they read the runs directly). The normal
    fix is the set-level PRE-FLIGHT upload in ``run_simulation._run_gathered_async``, which puts every example in place
    before any scenario reaches this function; this ordering is the second line of
    defence, so that even with no pre-flight the session pins a version holding at
    least the winning scenario's own example instead of zero.

    This function stays SELF-SUFFICIENT — both get-or-creates remain — because the
    pre-flight can be skipped, fail, or simply not cover a scenario. When the
    pre-flight did run, every step here degrades to a pure read.

    ``project_is_explicit`` says whether ``project_name`` was chosen by the customer —
    ``LANGSMITH__PROJECT`` *or* ``PLURAI_EXPERIMENT_NAME`` (the project cascade's
    ``experiment_name`` leg is customer-supplied too) — rather than falling back to
    the resolved dataset name; only that dataset-name fallback may be re-derived when
    it exists without dataset linkage (``derive_on_conflict=not project_is_explicit``
    — see :func:`_ensure_experiment_project`).

    Returns a :class:`PreparedExperiment` on success — the Example (caller reads
    ``.id``) plus the EFFECTIVE project name to post runs into — or ``None`` when:
      - ``client`` is None (LangSmith disabled or unavailable)
      - any setup step raises a LangSmith / HTTP error (logged + swallowed;
        the orchestrator continues without linkage so the simulation still
        completes cleanly)
    """
    if client is None:
        return None
    try:
        dataset = _get_or_create_dataset(client, name=dataset_name)
        # Sampled BEFORE our own example lands, or we would just be observing our own
        # write and the diagnostic could never fire (it is the reuse-of-an-empty-dataset
        # case we want to catch, and by the next statement the dataset is never empty).
        dataset_was_empty = _reference_dataset_is_empty(client, dataset_id=dataset.id)
        example = _get_or_create_example(
            client,
            dataset_id=dataset.id,
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            inputs=example_inputs,
        )
    except _langsmith_error_types():
        logger.exception(
            f'LangSmith dataset/example setup failed for scenario {scenario_name!r} (dataset={dataset_name!r}); proceeding without Experiments linkage.'
        )
        return None
    # The PROJECT (per-invocation experiment) is registered against the durable DATASET —
    # that linkage is what surfaces the run on the dataset's Experiments tab, and tagging
    # here is the only way reference_dataset_id ever gets populated (update_project rejects
    # it post-hoc). Created LAST, so the session's pinned dataset version already contains
    # the examples (see the ordering note above).
    #
    # Its OWN handler, outside the block above: once the Example exists, a project failure
    # must not throw it away. Returning None here would disable the whole run context —
    # no transcript, no feedback, and an orphan Example left in the dataset — over a
    # degradation the project helper itself treats as survivable (its wrong-linkage branch
    # logs and carries on with the run intact). Same principle as the probe: a missing
    # Experiments-tab registration is never worth the run's linkage. On this swallow
    # path the ORIGINAL project name stands — no derivation happened.
    effective_project = project_name
    try:
        effective_project = _ensure_experiment_project(
            client,
            project_name=project_name,
            dataset_id=dataset.id,
            dataset_was_empty=dataset_was_empty,
            derive_on_conflict=not project_is_explicit,
        )
    except _langsmith_error_types():
        logger.exception(
            f'LangSmith experiment-project setup failed for {project_name!r} (dataset={dataset_name!r}); '
            f'keeping the Example linkage — runs will be traced and linked, but will not appear on the '
            f'dataset Experiments tab.'
        )
    return PreparedExperiment(example=example, project_name=effective_project)


def feedback_value_for_kind(kind: str, score: float | None, value: Any) -> Any:
    """The ``value`` to post on a per-policy LangSmith feedback row, given the TypedScore ``kind``.

    LangSmith's Experiments grid renders a feedback column by the TYPE of ``value``: a
    STRING value becomes a categorical column (label + distribution — what we want for a
    boolean true/false and for category evals), while a bool / number value collapses to a
    numeric ``AVG`` column. A ``feedback_config`` does NOT override this. So a 'boolean' per-policy verdict posts the STRING ``'true'`` / ``'false'`` label —
    the numeric 1/0 still rides on ``score`` (Monitor charts it). Categorical evals already carry a native string ``value`` (rendered as their label);
    numbers keep their numeric ``value``.

    Applied ONLY to per-policy evals (:func:`post_experiment_feedback`). The generated ROLLUP
    metrics post their numeric value verbatim — ``scenario_passed`` and the other numeric
    rollups render as the numeric 0.0/1.0.
    """
    if kind == 'boolean':
        return 'true' if (score is not None and score >= 0.5) else 'false'
    return value


def _post_feedback_rows(
    client: langsmith.Client,
    *,
    scenario_name: str,
    rows: Sequence[tuple[uuid.UUID | str | None, str, float | None, str, Any]],
) -> None:
    """Post each ``(run_id, key, score, comment, value)`` row onto its RESOLVED target run.

    The single shared posting path both the per-policy (:func:`post_experiment_feedback`)
    and rollup (:meth:`LangSmithSink._post_feedback`) feedback sites route through. Each
    row carries EXACTLY ONE target run id (the scenario anchor for a scenario / rollup
    verdict, or the per-turn entry's turn node — ``turn_runs.get(turn_idx, anchor)`` in
    :func:`post_experiment_feedback`), so there is still NO run lookup and NO closing-turn
    heuristic. The posted ``value`` is computed by the CALLER (per-policy booleans are mapped
    to a ``'true'`` / ``'false'`` label via :func:`feedback_value_for_kind`; rollups stay
    numeric). Best-effort: each row's failure is logged and swallowed so one bad row never
    blocks the rest. Rows whose target is ``None`` (the anchor was never built) are skipped.
    Each row is posted under its derived :func:`_feedback_id`, the id a correction overwrites.
    """
    swallow = _langsmith_error_types()
    for run_id, key, score, comment, value in rows:
        if run_id is None:
            continue
        try:
            client.create_feedback(run_id=run_id, key=key, score=score, comment=comment, value=value, feedback_id=_feedback_id(run_id, key))
        except swallow:
            logger.exception('LangSmith create_feedback failed for %r on scenario %r (run_id=%s).', key, scenario_name, run_id)


def post_experiment_feedback(
    client: langsmith.Client | None,
    *,
    anchor_run_id: uuid.UUID | str | None,
    scenario_name: str,
    policy_results: dict[str, Any],
    turn_runs: dict[int, uuid.UUID | str] | None = None,
) -> None:
    """Post per-policy TYPED ``Feedback`` rows on their scope-resolved target runs.

    Called AFTER finalize. Each PolicyResult routes through the shared
    :func:`~plurai.observability.eval_mapping.map_eval_score`, so every
    ``create_feedback`` row carries the numeric ``score`` (the chartable form) and a
    typed ``value`` (a 'pass'/'fail' label for booleans so the grid renders pass/fail, the
    native category string for categoricals, the number for numbers), with the reasoning as
    the comment. PLACEMENT is EXCLUSIVE (an eval is never posted twice): a scenario-scoped
    policy posts ONCE on the scenario anchor (built by ``record_transcript`` with
    ``run_ctx.anchor_run_id``); a turn-scoped policy posts ONE feedback per per-turn entry (the
    typed :class:`~plurai.observability.eval_mapping.TurnScore` stream) on that
    turn's node via the ``turn_runs`` capture (``turn_runs.get(turn_idx, anchor)``) and NOT on
    the anchor — N entries ⇒ N feedbacks, an uncaptured turn degrading to the anchor — with no
    run lookup and no closing-turn heuristic. A turn-scoped policy that emitted no per-turn
    stream falls back to its rolled-up verdict on the anchor so it is never dropped. The single
    overall scenario pass/fail is the separate ``scenario_passed`` rollup, not a per-policy copy.

    No-op on:
      - ``client`` is None
      - empty ``policy_results``
      - no ``anchor_run_id`` (the anchor was never built)
    """
    if client is None or not policy_results or anchor_run_id is None:
        return

    results = eval_results(policy_results)
    if not results:
        return

    turn_runs_map = turn_runs or {}
    rows: list[tuple[uuid.UUID | str | None, str, float | None, str, Any]] = []
    for policy_name, result in results.items():
        if not isinstance(result, dict):
            continue  # a malformed (non-dict) PolicyResult has no mappable fields — skip the row, keep the rest
        # PLACEMENT is EXCLUSIVE (an eval is placed in exactly one place, never posted
        # twice): a turn-scoped policy posts ONLY on its turns; a
        # scenario-scoped policy posts ONLY on the anchor. The single overall scenario
        # pass/fail lives in the SEPARATE ``scenario_passed`` rollup, not in a per-policy
        # rolled-up copy on the anchor. A boolean verdict posts a 'true'/'false' LABEL value
        # (so the grid renders true/false, not a numeric AVG) via ``feedback_value_for_kind``.
        per_turn = per_turn_results(policy_name, result) if eval_scope(result) == 'turn' else []
        if per_turn:
            # Turn-scoped: one feedback per captured per-turn entry on its turn
            # run — N entries ⇒ N feedbacks; an uncaptured turn degrades to the anchor.
            for ts in per_turn:
                target = turn_runs_map.get(ts.turn_idx, anchor_run_id)
                rows.append((target, ts.score.key, ts.score.score, ts.score.comment, feedback_value_for_kind(ts.score.kind, ts.score.score, ts.score.value)))
        else:
            # Scenario-scoped, OR turn-scoped with NO per-turn stream available (e.g. a policy
            # that terminated before emitting one): place the rolled-up verdict on the anchor so
            # the eval is never dropped.
            ts = map_eval_score(policy_name, result)
            rows.append((anchor_run_id, ts.key, ts.score, ts.comment, feedback_value_for_kind(ts.kind, ts.score, ts.value)))
    _post_feedback_rows(client, scenario_name=scenario_name, rows=rows)


# ---------------------------------------------------------------------------
# Generation-time dataset upload.
#
# Distinct from the per-scenario run-recording pipeline above: this is the
# BATCH "push the generated scenarios into a LangSmith Dataset" step that runs
# once per generate-scenarios pass. It holds to the SDK conventions: lazy langsmith
# import, stdlib logging, no backend config import, plain api_key arg, and NO
# os.environ mutation (the client constructs ``langsmith.Client(api_key=...)``
# directly).
# ---------------------------------------------------------------------------


def sync_scenarios_to_langsmith(
    *,
    api_key: str | None,
    dataset_name: str,
    dataset_description: str | None,
    scenarios: list[dict[str, Any]],
    existing_examples: dict[str, str] | None = None,
) -> ProviderSyncResult:
    """Upsert a scenario batch into a LangSmith Dataset.

    Each entry in ``scenarios`` must have ``folder_name`` and ``inputs``;
    ``outputs`` and ``metadata`` are optional.

    Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`.
    A true no-op (no ``api_key`` / SDK not installed) returns ``ok=True, uploaded=0``.

    * dataset get-or-create transport error ⇒ ``ProviderSyncResult(ok=False)``
      (logged at ``error`` level; never swallowed-as-no-op — a real failure must NOT
      look like a no-op to the operator);
    * per-item failures are TALLIED (``failed`` / ``errors``) and threaded onto
      the otherwise-successful result rather than silently dropped; if EVERY item
      failed the result is ``ok=False``.

    Example identity is the scenario FOLDER, and it is resolved from TWO sources so a
    re-run updates in place (a new Dataset version pinned to the same example ids) rather
    than minting duplicates:

    1. ``existing_examples`` — the ``{folder: example_id}`` handle map persisted by a prior
       upload into ``<out>/dataset_upload_metadata.json``.
    2. Failing that, a folder-keyed read of the DATASET itself.

    The second source is not redundant: ``--out`` is required and typically a per-run
    directory, so the handle map is routinely absent even though the examples exist. Without
    it, every invocation would see an empty map and re-create the whole dataset, and a
    LangSmith run pre-flights implicitly on EVERY invocation. The dataset is the durable
    record; the file is a cache.

    This does NOT mutate ``os.environ`` — the ``api_key`` is passed straight into the
    ``langsmith.Client`` constructor (the customer supplies the key
    client-side).
    """
    if not api_key:
        return ProviderSyncResult(provider='langsmith', metadata_updates={}, summary_line='LangSmith: no API key configured — nothing uploaded.')
    if not _langsmith_available():
        return ProviderSyncResult(provider='langsmith', metadata_updates={}, summary_line='LangSmith: SDK not installed — nothing uploaded.')
    import langsmith

    client = langsmith.Client(api_key=api_key)
    swallow = _langsmith_error_types()

    try:
        if client.has_dataset(dataset_name=dataset_name):
            dataset = client.read_dataset(dataset_name=dataset_name)
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description=dataset_description,
            )
    except swallow as exc:
        # Transport failure on the dataset op is a genuine FAILURE, not a no-op:
        # log at error level and return an ok=False result so the upload path
        # reports FAILED (returning None here would mis-report it as "skipped").
        logger.error(f'LangSmith dataset create/read for {dataset_name!r} failed: {exc}')
        return ProviderSyncResult(
            provider='langsmith',
            metadata_updates={},
            summary_line=f'LangSmith: dataset create/read for {dataset_name!r} failed: {exc}',
            ok=False,
            errors=(f'dataset create/read: {exc}',),
        )

    dataset_id = str(dataset.id)
    existing_examples = existing_examples or {}
    result: dict[str, str] = {}
    errors: list[str] = []

    for scen in scenarios:
        folder = scen['folder_name']
        inputs = scen['inputs']
        outputs = scen.get('outputs') or {}
        # Stamp ONLY the folder, under the join key the runtime get-or-create reads, and
        # otherwise pass the server's scenario metadata through UNTOUCHED.
        #
        # Do NOT also force ``'scenario_name': folder``: the runtime path joins on
        # ``scenario_folder`` (the two identities are NOT the same string), so such a stamp
        # buys nothing — and it writes the FOLDER into a field named for the scenario,
        # leaving anything that reads ``metadata.scenario_name`` off a LangSmith example
        # with the folder, not the name. Overwriting a key inside a passthrough metadata
        # blob is the sink's business only for the key it owns: ``scenario_folder``.
        metadata = {**scen.get('metadata', {}), SCENARIO_FOLDER_KEY: folder}
        prior_id = existing_examples.get(folder)
        # Everything below is inside ONE per-item guard, including the existence probe. The
        # probe must NOT get a `try` of its own above this one: an exception outside the
        # swallowed tuple (the SDK parses rows into pydantic models lazily, so `list()` is
        # where a malformed row raises) would escape the whole loop and discard every folder
        # already upserted into ``result``. A best-effort upload must lose at most the item
        # it is on.
        try:
            if prior_id is None:
                # ``existing_examples`` comes from ``<out>/dataset_upload_metadata.json``,
                # and ``--out`` is typically a per-run directory — so a fresh out dir makes
                # every folder look new and re-mints the whole dataset. Since a LangSmith
                # run pre-flights implicitly on EVERY invocation, that silently regrows
                # duplicate examples run after run. Ask the DATASET, the durable record.
                #
                # A FAILED probe is not "no example exists" — it is "unknown", and creating
                # on unknown defaults to the damaging answer (a duplicate: the very symptom
                # this probe exists to prevent). So it is reported at ``warning`` AND
                # tallied into ``errors``: this package configures no logging, so
                # ``logger.info`` would emit nothing under the WARNING lastResort handler,
                # and an untallied failure leaves ``failed=0`` and a clean 'uploaded N/N'
                # summary over a dataset that just grew a duplicate. Rate limiting is the
                # realistic trigger — this probe fires once per scenario in a tight loop.
                existing = next(iter(client.list_examples(dataset_id=dataset_id, metadata={SCENARIO_FOLDER_KEY: folder}, limit=1)), None)
                if existing is not None:
                    prior_id = str(existing.id)
            if prior_id is not None:
                # Update creates a new modified_at — LangSmith's native versioning.
                # If the example was deleted server-side, fall through to create.
                try:
                    # ``outputs`` is OMITTED when this scenario carries none, rather than
                    # sent as {}. Scenarios never carry reference outputs, and {} is a real
                    # value to LangSmith — it would BLANK outputs a customer curated on the
                    # example in the UI.
                    client.update_example(
                        prior_id,
                        inputs=inputs,
                        metadata=metadata,
                        **({'outputs': outputs} if outputs else {}),
                    )
                    result[folder] = prior_id
                    continue
                except swallow as exc:
                    logger.info(f'update_example({prior_id}) for scenario {folder!r} failed ({exc}); creating a fresh example.')
            example = client.create_example(
                dataset_id=dataset_id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            result[folder] = str(example.id)
        except swallow as exc:
            # Best-effort per item: tally the loss (don't abort the loop), so
            # "uploaded N item(s)" never hides a silently-dropped scenario.
            logger.error(f'LangSmith upsert failed for scenario {folder!r}: {exc}')
            errors.append(f'{folder}: {exc}')
        except Exception as exc:  # noqa: BLE001 — see below
            # Deliberately BROADER than the typed tuple above. The vendor SDK raises
            # outside its own error hierarchy in reachable places — ``list_examples`` is a
            # generator that parses rows into pydantic models, so a malformed row surfaces
            # as a ``ValidationError`` here — and the enclosing caller turns ANY escape into
            # a total-loss ``UploadResult(ok=False)``, throwing away every folder already
            # upserted. Tallying it costs one item instead of the batch.
            logger.error(f'LangSmith upsert failed unexpectedly for scenario {folder!r}: {exc!r}')
            errors.append(f'{folder}: unexpected {type(exc).__name__}: {exc}')

    failed = len(errors)
    total = len(scenarios)
    # ALL items failed (and there were items) ⇒ the attempt failed outright.
    ok = not (total > 0 and failed == total)
    summary = f'LangSmith: uploaded {len(result)}/{total} example(s) to dataset {dataset_name!r} (id={dataset_id}).'
    if failed:
        summary = f'LangSmith: uploaded {len(result)}/{total} example(s) to dataset {dataset_name!r} (id={dataset_id}) ({failed} failed — see log).'
    return ProviderSyncResult(
        provider='langsmith',
        metadata_updates={
            'langsmith_dataset_id': dataset_id,
            'langsmith_dataset_name': dataset_name,
            'langsmith_examples': result,
        },
        summary_line=summary,
        ok=ok,
        uploaded=len(result),
        failed=failed,
        errors=tuple(errors),
    )


class _Timeline:
    """Strictly increasing timestamps for a run tree written after the conversation it records.

    The whole tree is built in a few milliseconds once the scenario is over, so the clock alone
    gives sibling turns equal start times and the UI orders the ties arbitrarily. Each node takes
    the next millisecond after the root's start instead, in the order the conversation happened.
    """

    def __init__(self, start: datetime) -> None:
        self._at = start

    def next(self) -> datetime:
        self._at += timedelta(milliseconds=1)
        return self._at


def _model_call_run(parent: Any, call: ModelCall, *, name: str | None, timeline: _Timeline) -> tuple[Any, dict[str, Any]]:
    """An ``llm`` child of ``parent`` for one model call, and the outputs to end it with.

    ``inputs`` / ``outputs`` use the chat shape LangSmith renders as a conversation: the messages
    shown to the model, and a ``choices`` list holding the assistant message. Model name rides
    ``ls_model_name``, the provider ``ls_provider`` and token usage ``usage_metadata`` — the keys
    LangSmith reads for model and cost columns (cost needs the provider too, so it stays empty
    when the framework did not name one). Left open, so a caller can hang tool runs under it
    before it ends.
    """
    model = call.get('model')
    metadata = {key: value for key, value in (('ls_model_name', model), ('ls_provider', call.get('provider'))) if value}
    message: dict[str, Any] = {'role': 'assistant', 'content': call['output']}
    if call['tool_calls']:
        message['tool_calls'] = openai_tool_calls(call['tool_calls'])
    outputs: dict[str, Any] = {'choices': [{'message': message}]}
    if usage := call.get('usage'):
        outputs['usage_metadata'] = usage
    llm_run = parent.create_child(
        name=name or model or 'model',
        run_type='llm',
        inputs={'messages': call['input']},
        start_time=timeline.next(),
        **({'extra': {'metadata': metadata}} if metadata else {}),
    )
    return llm_run, outputs


# ---------------------------------------------------------------------------
# LangSmithSink — the ObservabilitySink instance for the LangSmith experiment
# pipeline. Holds ONLY the config as INSTANCE state (no module-level globals, no
# cached client); the per-scenario LangSmith ``Client`` is built fresh in
# ``_prepare_run`` and threaded out on the ``LangSmithRunContext``, and the
# stateless module functions above remain functions and are reused unchanged. The
# factory constructs exactly one of these when LangSmith is the single active sink.
# ---------------------------------------------------------------------------


class LangSmithSink:
    """LangSmith Experiments sink (dataset + example + per-policy / rollup feedback).

    The customer's agent emits its own traces into the customer's LangSmith
    account, so recording happens CLIENT-side. ``prepare_run`` get-or-creates the
    Dataset + experiment Project + Example and returns a
    :class:`LangSmithRunContext` carrying the ``example_id`` the orchestrator
    threads into the child handshake. ``post_feedback`` posts per-policy + rollup
    feedback on the scenario anchor root (pre-allocated ``anchor_run_id``).
    """

    name = 'langsmith'

    #: Ask the runner to upload the scenario dataset BEFORE the gather, even without
    #: ``--upload-datasets``. LangSmith pins an experiment session's example set to the
    #: dataset version as of session creation, and that pin cannot be changed afterwards
    #: — so examples created concurrently with (or after) the first scenario's session
    #: are invisible to the Experiments grid, which renders "No results found" while the
    #: header aggregates still compute off the runs. Populating the dataset up front is
    #: the only way to get a correct grid, which makes it a REQUIREMENT of this sink
    #: rather than an operator preference. Read by ``run_simulation._run_gathered_async``
    #: via ``getattr``; every other sink simply lacks the attribute.
    requires_preflight_dataset = True

    def __init__(self, config: LangSmithSinkConfig) -> None:
        self._config = config
        # ``(scenario_folder, project_name, anchor_run_id)`` per recorded scenario, read by
        # ``run_links``. The sink is shared by every concurrent scenario's ``record_transcript``
        # thread; ``list.append`` is atomic under the GIL, so no lock.
        self._recorded: list[tuple[str, str, str]] = []

    async def warm_once(self) -> None:
        """No-op: LangSmith has NO process-global init to warm.

        Unlike Arize (OTel global provider) / Braintrust (process Logger) / Langfuse
        (SDK client registry), the LangSmith ``Client`` is built FRESH per scenario in
        ``_prepare_run`` and threaded out on the run context — never cached on the sink
        and carrying no process-wide side effect — so there is nothing to install once.
        Defined for uniform ``warm_once`` conformance across sinks.
        """

    # -- dataset_url (optional DatasetUrlSink role) -------------------------

    def dataset_url(self, dataset_name: str) -> str | None:
        """Best-effort UI URL of the run's LangSmith Dataset (never raises).

        Resolves the SAME dataset identity ``prepare_run`` registers runs against
        (the explicit ``LANGSMITH__DATASET`` override wins over the scenario set's
        ``dataset_name``) and reads it by name — the SDK's ``Dataset`` schema carries
        the hosted ``url``. Any failure (no creds / SDK absent / dataset missing /
        transport error) degrades to ``None``: the summary line simply omits the link.
        """
        try:
            client = build_handle(api_key=self._config.api_key or '')
            if client is None:
                return None
            dataset = client.read_dataset(dataset_name=self._config.dataset or dataset_name)
            return getattr(dataset, 'url', None) or None
        except Exception:
            logger.debug('LangSmith dataset_url resolution failed for %r.', dataset_name, exc_info=True)
            return None

    # -- run_links (optional RunLinksSink role) -----------------------------

    def run_links(self, dataset_name: str) -> RunLinks | None:
        """Best-effort links to the run's experiment and to each recorded scenario's trace (never raises).

        The experiment is the dataset's compare view over the experiment project(s) the scenarios
        were posted into; a trace is the scenario anchor run inside its project. Both hang off the
        dataset's hosted URL (``<org>/datasets/<id>``) — the SDK exposes no project or run URL —
        so a dataset that cannot be read yields no links. A project that cannot be read drops its
        scenarios' traces and is left out of the experiment link.
        """
        # A re-driven scenario records twice under one folder; the latest attempt is its trace.
        recorded = {folder: (project, run_id) for folder, project, run_id in list(self._recorded)}
        if not recorded:
            return None
        try:
            client = build_handle(api_key=self._config.api_key or '')
            if client is None:
                return None
            dataset_url = getattr(client.read_dataset(dataset_name=self._config.dataset or dataset_name), 'url', None)
            if not dataset_url or '/datasets/' not in dataset_url:
                return None
            org_url = dataset_url.split('/datasets/')[0]
            project_ids: dict[str, str] = {}
            for project_name in dict.fromkeys(project for project, _ in recorded.values()):
                try:
                    project_ids[project_name] = str(client.read_project(project_name=project_name).id)
                except Exception:
                    logger.debug('LangSmith project %r could not be read for run links.', project_name, exc_info=True)
            experiment_url = f'{dataset_url}/compare?selectedSessions={",".join(project_ids.values())}' if project_ids else None
            traces = sorted(
                (folder, f'{org_url}/projects/p/{project_ids[project]}/r/{run_id}') for folder, (project, run_id) in recorded.items() if project in project_ids
            )
            return RunLinks(experiment_url=experiment_url, traces=tuple(traces))
        except Exception:
            logger.debug('LangSmith run_links resolution failed for %r.', dataset_name, exc_info=True)
            return None

    # -- prepare_run --------------------------------------------------------

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
    ) -> LangSmithRunContext:
        """Get-or-create Dataset/Project/Example; ALWAYS return a context (``enabled`` gates).

        Async: the blocking LangSmith get-or-create work runs inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly. Best-effort: any
        sink error is logged + swallowed, yielding a DISABLED context (``enabled=False`` —
        client/example unset) rather than ``None``, so the simulation runs without
        observability for this scenario and prepare_run NEVER raises or returns ``None``.
        """
        return await offload(
            self._prepare_run,
            agent_name=agent_name,
            experiment_name=experiment_name,
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            thread_id=thread_id,
            dataset_name=dataset_name,
            example_inputs=example_inputs or {},
            agent_version=agent_version,
            sink=self.name,
            what='prepare_run',
        ) or LangSmithRunContext(client=None, project_name=None, example_id=None, scenario_name=scenario_name, thread_id=thread_id)

    def _prepare_run(
        self,
        *,
        agent_name: str,
        experiment_name: str,
        scenario_name: str,
        thread_id: str,
        dataset_name: str | None,
        example_inputs: dict[str, Any],
        agent_version: str,
        scenario_folder: str | None = None,
    ) -> LangSmithRunContext:
        cfg = self._config
        # DATASET cascade: the EXPLICIT LangSmith config dataset (LANGSMITH__DATASET,
        # via cfg.dataset) wins so a configured override is never clobbered by the
        # ``dataset_name`` the orchestrator passes (the scenario set's dataset name —
        # the SAME name the upload path targets); then the server-side experiment
        # name, then the agent name.
        resolved_dataset = cfg.dataset or dataset_name or experiment_name or agent_name
        # PROJECT cascade — a SEPARATE identity from the dataset. The experiment
        # PROJECT is per-invocation (LangSmith renders one project = one experiment);
        # the DATASET is the durable scenario-set dataset. Registering the
        # per-invocation project against the durable dataset is what makes the run
        # appear on that dataset's Experiments tab. LANGSMITH__PROJECT (cfg.project)
        # wins; else the per-invocation experiment identity; else fall back to the
        # dataset name, so project and dataset share one name.
        project_name = cfg.project or experiment_name or resolved_dataset
        # Disabled SENTINEL (enabled=False, client/example unset): returned INSTEAD of
        # None when the client can't be built or the experiment setup fails, so
        # prepare_run never returns None — downstream seams gate on ``enabled``.
        disabled = LangSmithRunContext(client=None, project_name=project_name, example_id=None, scenario_name=scenario_name, thread_id=thread_id)
        client = build_handle(api_key=cfg.api_key or '')
        if client is None:
            return disabled
        prepared = prepare_experiment_run(
            client,
            dataset_name=resolved_dataset,
            project_name=project_name,
            scenario_name=scenario_name,
            scenario_folder=scenario_folder,
            example_inputs=example_inputs,
            # A project name is CUSTOMER-CHOSEN when either knob is set: an explicit
            # LANGSMITH__PROJECT (cfg.project) OR a PLURAI_EXPERIMENT_NAME
            # (experiment_name — the cascade's middle leg). Both are honored verbatim
            # (with a warning) when squatted; only the resolved-DATASET fallback —
            # a name nobody chose — may be re-derived. See _ensure_experiment_project.
            project_is_explicit=bool(cfg.project or experiment_name),
        )
        if prepared is None:
            return disabled
        example = prepared.example
        return LangSmithRunContext(
            client=client,
            # The EFFECTIVE project (possibly derived): record_transcript posts the
            # anchor into it AND stamps it as the record's experiment_name.
            project_name=prepared.project_name,
            example_id=example.id,
            scenario_name=scenario_name,
            thread_id=thread_id,
            # Pre-allocate the scenario-anchor root id NOW so ``record_transcript``
            # can build the anchor with it AND ``post_feedback`` can target the
            # same run unconditionally (no lookup, no race). Derived from the thread
            # so a later score correction can find it too.
            anchor_run_id=_anchor_run_id(thread_id),
            tags=(resolved_dataset, f'thread:{thread_id}'),
            agent_version=agent_version,
            scenario_folder=scenario_folder,
            # Mirror the RESOLVED example's inputs onto the anchor run:
            # ``record_transcript`` uses ``run_ctx.example_inputs`` as the root run's
            # inputs, so when the experiment linked to the RICH uploaded example this
            # makes the run's displayed input match its reference example (persona,
            # agent_context, initial_world_state, first_talk) — parity with LangFuse,
            # whose run_experiment feeds the run the uploaded item. Falls back to the
            # leaner orchestrator inputs when the example was freshly created from them.
            example_inputs=example.inputs or example_inputs,
        )

    # -- post_feedback ------------------------------------------------------

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
        """Post per-policy feedback + the rollup metrics on the scenario anchor.

        Both per-policy and rollup feedback attach to the single scenario anchor
        root (``run_ctx.anchor_run_id``, built by ``record_transcript``). The rollup
        consumes the PASSED-IN ``final_world_state`` + ``tool_events`` (run-local)
        and ``thread.chatbot_messages`` — NEVER ``ThreadResult.tool_events`` (empty
        on the in-process path). No-op when ``run_ctx`` is ``None`` or the experiment
        pipeline wasn't fully set up.

        Async: the blocking metric-compute + feedback POSTs run inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, LangSmithRunContext) or not run_ctx.enabled:
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
        """Per-scenario teardown: flush any buffered LangSmith client state (idempotent).

        Guaranteed by the orchestrator's scenario scope on exit (normal AND exception).
        LangSmith's tracing client may buffer run-tree submissions; this drain ensures
        spans surface even on a path where ``record_transcript`` was skipped (crash
        path). No-op when tracing isn't live (``run_ctx`` is ``None`` or its ``client``
        is ``None``). Distinct from ``close``, which shuts the client down once per
        process.

        Async: any blocking flush work runs inside via ``asyncio.to_thread``; errors
        are swallowed so this never raises.
        """
        if not isinstance(run_ctx, LangSmithRunContext) or run_ctx.client is None:
            return
        client = run_ctx.client

        def _flush() -> None:
            try:
                client.flush()
            except Exception:
                logger.exception('LangSmith flush failed at scenario drain; traces may be delayed or lost.')

        await asyncio.to_thread(_flush)

    async def close(self) -> None:
        """Process-teardown drain (async, no-op).

        Per-scenario draining is handled by ``drain_run`` (which flushes the SDK's
        background run-tree tracing queue via ``client.flush()`` at scenario-scope exit);
        feedback rows are posted per-call in ``post_feedback``. No process-wide state
        outlives a scenario on this sink, so teardown has nothing left to drain.
        """

    def _post_feedback_sync(
        self,
        thread: CreateThreadResponse,
        finalize: FinalizeResponse,
        run_ctx: LangSmithRunContext,
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
                agent_states=agent_states_of(thread),
                agent_actions_by_turn=agent_actions_of(thread),
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
            scenario_name=scenario_name,
            policy_results=policy_results,
            metrics=metrics,
            final_world_state=final_world_state,
            scenario_metadata=md,
            sink=self.name,
            what='post_feedback',
        )

    def _post_feedback(
        self,
        run_ctx: LangSmithRunContext,
        *,
        scenario_name: str,
        policy_results: dict,
        metrics: list,
        final_world_state: dict[str, Any] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        client = run_ctx.client
        anchor_run_id = run_ctx.anchor_run_id
        if client is None:
            return  # disabled context (enabled gates earlier; this also narrows the type for the helpers below)
        # POST THE ANCHOR FIRST, ended with the per-scenario summary as its CREATE outputs (the
        # one-coherent-record contract). MUST precede the feedback rows so the anchor run exists
        # before feedback attaches to ``anchor_run_id``.
        self._post_summary(
            run_ctx,
            scenario_name=scenario_name,
            policy_results=policy_results,
            final_world_state=final_world_state,
            scenario_metadata=scenario_metadata,
        )
        # Per-policy TYPED feedback — scope-routed: turn-scoped rows attach to their turn
        # node (run_ctx.turn_runs capture), everything else to the scenario anchor root.
        post_experiment_feedback(
            client,
            anchor_run_id=anchor_run_id,
            scenario_name=scenario_name,
            policy_results=policy_results,
            turn_runs=run_ctx.turn_runs,
        )
        # Rollup metrics — through the SAME TypedScore posting path as the per-policy rows
        # (map_rollup_score RE-SHAPES the RollupMetric's existing numeric/native split; no
        # re-mapping). Rollups are ALWAYS scenario-scoped — they summarize the whole run —
        # so every row targets the anchor; the scenario pass/fail verdict therefore posts
        # exactly ONCE, on the anchor. The list includes `aggregate` (SECONDARY reliability
        # number, always) and `pass_rate` (HEADLINE quality signal, only when the backend published
        # a rate for this scenario), produced by compute_rollup_metrics, so they ride the same
        # path with no special-casing.
        # ROLLUPS post their NATIVE value: the numeric ones verbatim (no true/false label) —
        # scenario_passed renders as 0.0/1.0 — and the str-valued ones (`scenario_status`, whose five-word
        # verdict IS the point, plus the comma-joined list rollups) as a categorical label. The
        # true/false label is a per-policy-eval treatment only.
        rollup_rows: list[tuple[uuid.UUID | str | None, str, float | None, str, Any]] = []
        for m in metrics:
            ts = map_rollup_score(m.key, m.score, m.value, m.comment)
            rollup_rows.append((anchor_run_id, ts.key, ts.score, ts.comment, ts.value))
        _post_feedback_rows(client, scenario_name=scenario_name, rows=rollup_rows)

    def _post_summary(
        self,
        run_ctx: LangSmithRunContext,
        *,
        scenario_name: str,
        policy_results: dict,
        final_world_state: dict[str, Any] | None,
        scenario_metadata: dict[str, Any] | None,
    ) -> None:
        """End the stashed scenario-anchor RunTree with the per-scenario summary, and POST it.

        The one-coherent-record contract: the scenario anchor's OUTPUT is a DISTINCT, non-empty
        summary ({verdict, results, turn_count, persona, world_changes}). It is set as the
        anchor's CREATE outputs — NOT via ``update_run`` — because an experiment run (one
        carrying ``reference_example_id``) accepts outputs ONLY at create time; ``update_run``
        to outputs is silently IGNORED on such runs. So ``record_transcript``
        defers ending/posting the root and stashes it here; we end it with the summary and post
        it now. No read-back (LangSmith ingestion lags writes). No-op when no anchor was stashed
        (record_transcript was skipped — e.g. a crash path).
        """
        if not run_ctx.anchor_holder:
            return
        root = run_ctx.anchor_holder[0]
        summary = build_scenario_summary(
            results=eval_results(policy_results),
            scenario_metadata=scenario_metadata,
            final_world_state=final_world_state,
            turn_count=len(run_ctx.turn_runs),
            policy_results=policy_results,
        )
        try:
            root.end(outputs=summary)
            root.post()
        except _langsmith_error_types():
            logger.exception('LangSmith summary anchor post failed for scenario %r (anchor_run_id=%s).', scenario_name, run_ctx.anchor_run_id)

    # -- overwrite_scores (score correction) -------------------------

    def overwrite_scores(self, items: Sequence[ScoreOverwrite]) -> list[ScoreOverwriteOutcome]:
        """Overwrite each corrected eval's feedback rows IN PLACE; never raises.

        Each row the run posted sits at a derived id (:func:`_feedback_id` on the
        :func:`_anchor_run_id` / :func:`_turn_run_id` it targets), so the correction is an
        ``update_feedback`` of that id with the corrected cell's score, value and comment —
        posted exactly as :func:`post_experiment_feedback` would have. A row the platform does
        not have (observability off for that run, or a run recorded without derived ids)
        answers ``not_recorded``.
        """
        client = build_handle(api_key=self._config.api_key or '')
        if client is None:
            return every_eval(items, 'failed', 'langsmith is not installed or LANGSMITH__API_KEY is not set')
        swallow = _langsmith_error_types()
        not_found = _langsmith_not_found_type()
        outcomes: list[ScoreOverwriteOutcome] = []
        for item in items:
            for ev in item.evals:
                written = 0
                failures: list[str] = []
                for turn_idx, ts in eval_score_placements(ev.policy, ev.cell):
                    run_id = _anchor_run_id(item.thread_id) if turn_idx is None else _turn_run_id(item.thread_id, turn_idx)
                    try:
                        client.update_feedback(
                            _feedback_id(run_id, ts.key),
                            score=ts.score,
                            value=feedback_value_for_kind(ts.kind, ts.score, ts.value),
                            comment=ts.comment,
                        )
                        written += 1
                    except swallow as exc:
                        if not_found is None or not isinstance(exc, not_found):
                            failures.append(f'update_feedback failed: {exc}')
                outcomes.append(eval_outcome(item.thread_id, ev.policy, written=written, failures=failures))
        return outcomes

    # -- record_transcript --------------------------------------------------

    async def record_transcript(
        self,
        run_ctx: RunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list,
        scenario_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit the 3-level RunTree (scenario -> turn -> {agent, client}) for one scenario.

        The scenario root is built with ``reference_example_id=run_ctx.example_id``
        (Experiments-grid linkage) and the pre-allocated ``run_ctx.anchor_run_id``
        (so feedback can target it unconditionally). Best-effort + ``run_ctx.enabled``
        gated; all state derives from ``run_ctx``. Returns the anchor root id, or
        ``None``.

        Async: the blocking RunTree build + POSTs run inside via
        ``asyncio.to_thread`` so the orchestrator awaits this directly.
        """
        if not isinstance(run_ctx, LangSmithRunContext) or not run_ctx.enabled:
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
        run_ctx: LangSmithRunContext,
        *,
        thread_id: str,
        scenario_name: str,
        turns: list,
        scenario_metadata: dict[str, Any] | None,
    ) -> Any:
        from langsmith.run_trees import RunTree

        client = run_ctx.client
        # ALLOW-LIST filtered record metadata (the shared ``build_scenario_metadata``):
        # exactly the allow-listed scenario-metadata keys + thread/experiment/agent_version run
        # identifiers — the IDENTICAL mapping the Langfuse anchor emits, so no
        # prompt-engineering internals (``tool_verbs`` / ``knowledge_plan``) reach the
        # Experiments grid.
        root_metadata = build_scenario_metadata(
            scenario_metadata=scenario_metadata,
            thread_id=thread_id,
            experiment_name=run_ctx.project_name,
            agent_version=run_ctx.agent_version,
        )
        # Scenario anchor: id + reference_example_id set AT CONSTRUCTION so the
        # Experiments grid aggregates it and feedback targets run_ctx.anchor_run_id.
        # Named ``scenario:<name>`` (the shared record-name convention) so the grid rows read
        # as scenario records at a glance.
        # The anchor RUN's inputs are the FULL scenario (the linked Example's inputs) so the
        # experiment row shows the complete input — first_talk + world-state + agent context —
        # matching the Langfuse trace, not a bare {scenario_name, thread_id} stub. Fall back to
        # that stub only when no example inputs were captured (degraded prepare_run).
        root_inputs = run_ctx.example_inputs or {'scenario_name': scenario_name, 'thread_id': thread_id}
        root = RunTree(
            id=run_ctx.anchor_run_id,
            name=scenario_record_name(scenario_name),
            run_type='chain',
            inputs=root_inputs,
            reference_example_id=run_ctx.example_id,
            extra={'metadata': root_metadata},
            ls_client=client,
            project_name=run_ctx.project_name,
            tags=list(run_ctx.tags),
        )
        # Build the tree top-down. The ROOT is NOT posted here — it is stashed on the run
        # context and posted by ``_post_summary`` (in the later ``post_feedback`` pass) ended
        # with the per-scenario summary as its outputs. This is REQUIRED because an experiment
        # run (one carrying ``reference_example_id``) takes its outputs ONLY at CREATE time —
        # ``update_run`` to outputs is IGNORED on such runs, so the summary must
        # ride the anchor's create. The children are posted now (they have no such constraint);
        # LangSmith assembles the tree by id even though the parent posts last.
        nodes: list[Any] = []
        # ``getattr``: a stub tree (tests, an older SDK surface) may carry no start time; now() orders the same.
        timeline = _Timeline(getattr(root, 'start_time', None) or datetime.now(UTC))
        for turn in turns:
            turn_run = root.create_child(
                name=turn_record_name(turn['turn_idx']),
                run_type='chain',
                run_id=_turn_run_id(thread_id, turn['turn_idx']),
                start_time=timeline.next(),
                # A content-part LIST rides STRUCTURED: these inputs are JSON to LangSmith, and
                # the parts say more than their text does. Only Arize projects.
                inputs={'input_user_message': turn['input_user_message']},
                extra={
                    'metadata': {
                        'turn_idx': turn['turn_idx'],
                        'should_terminate': turn['should_terminate'],
                        'turn_count': turn['turn_count'],
                    }
                },
            )
            # Capture the turn run's id (allocated by RunTree at construction) so
            # turn-SCOPED evals can anchor on this exact node (post_experiment_feedback's
            # per-turn placement). Mutates the per-scenario holder on
            # run_ctx — never sink-instance state.
            run_ctx.turn_runs[turn['turn_idx']] = turn_run.id
            # One model call IS the agent's step, so the agent node itself is that ``llm`` run; a
            # second node with the same prompt and reply would only repeat it. Several calls
            # (call -> tool -> call ...) keep a ``chain`` agent with one ``llm`` child per call.
            calls = turn.get('agent_model_calls') or []
            llm_runs: list[Any] = []
            if len(calls) == 1:
                agent_run, agent_outputs = _model_call_run(turn_run, calls[0], name='agent', timeline=timeline)
            else:
                agent_run = turn_run.create_child(name='agent', run_type='chain', inputs={'input': turn['input_user_message']}, start_time=timeline.next())
                agent_outputs = {'output': turn['agent_response']}
                for call in calls:
                    llm_run, llm_outputs = _model_call_run(agent_run, call, name=None, timeline=timeline)
                    llm_run.end(outputs=llm_outputs, end_time=timeline.next())
                    llm_runs.append(llm_run)
            # Tool calls the agent made THIS turn become ``run_type='tool'`` children of
            # the agent node — the "turn -> assistant -> tool calls (args + result)"
            # tree. ``args`` are the call inputs; ``response`` the tool result; an errored
            # call (``status == 'error'``) ends with ``error`` set so it renders red. A call
            # recorded by ``/mock/execute`` carries ``mocked: True`` in its run metadata. Posted
            # AFTER ``agent_run`` (parent-first) below.
            tool_runs = []
            for tc in turn.get('agent_tool_calls') or []:
                tool_run = agent_run.create_child(
                    name=tc['tool'],
                    run_type='tool',
                    inputs={'args': tc['args']},
                    start_time=timeline.next(),
                    **({'extra': {'metadata': marker}} if (marker := tool_span_metadata(tc)) else {}),
                )
                tool_run.end(
                    outputs={'result': tc['response']},
                    error=f'tool call {tc["tool"]!r} errored' if tc['status'] == 'error' else None,
                    end_time=timeline.next(),
                )
                tool_runs.append(tool_run)
            agent_run.end(outputs=agent_outputs, end_time=timeline.next())
            client_run = turn_run.create_child(name='client', run_type='chain', inputs={'input': turn['agent_response']}, start_time=timeline.next())
            client_run.end(outputs={'output': turn['output_user_message']}, end_time=timeline.next())
            turn_run.end(outputs={'output_user_message': turn['output_user_message']}, end_time=timeline.next())
            nodes.extend([turn_run, agent_run, *llm_runs, *tool_runs, client_run])
        # Post the CHILDREN now; STASH the root to be ended (with the summary) + posted by
        # ``_post_summary``. The root is intentionally NOT ended/posted here — see above.
        for node in nodes:
            node.post()
        run_ctx.anchor_holder.append(root)
        # The folder is the scenario's join identity (the run facts carry it); a record written
        # without run facts falls back to the join key, the same label the Langfuse sink prints.
        folder = (scenario_metadata or {}).get(SCENARIO_FOLDER_KEY) or dataset_join_key(scenario_folder=run_ctx.scenario_folder, scenario_name=scenario_name)
        if run_ctx.project_name:
            self._recorded.append((str(folder), run_ctx.project_name, str(root.id)))
        return root.id

    # -- sync ---------------------------------------------------------------

    def sync_scenarios(
        self,
        *,
        dataset_name: str,
        dataset_description: str | None,
        scenarios: list[ScenarioUploadEntry],
        existing_metadata: dict[str, Any],
    ) -> ProviderSyncResult:
        """Upload a generated scenario batch into a LangSmith Dataset.

        Always returns a :class:`plurai.observability.dataset_types.ProviderSyncResult`;
        a disabled/SDK-missing no-op returns ``ok=True, uploaded=0``.

        ``LANGSMITH__DATASET`` (``cfg.dataset``) wins over the passed name, matching the
        SAME head-of-cascade precedence the per-scenario anchor uses (``_prepare_run``)
        and ``dataset_url``. Without this the caller's name (the scenario set's own
        ``dataset_name``) would send the upload to a DIFFERENT dataset than the one the
        experiment project references — examples in one dataset, an empty dataset behind
        the grid. That divergence makes the pre-flight upload useless for anyone who sets the override.

        The prior-handle guard is the other half: ``sync_scenarios_to_langsmith`` treats
        a known example id as an in-place ``update_example``, WITHOUT checking which
        dataset that example lives in. Handles recorded against a different dataset name
        would therefore silently update the OLD dataset and create nothing in the new one
        — while still reporting "uploaded N/N".

        The guard fires only on EVIDENCE of divergence — a recorded name that differs.
        An ABSENT name is unknown provenance, not a mismatch, and dropping handles there
        would re-create examples that already exist in the same dataset, destroying the
        idempotency this metadata exists to provide. (Every upload writes
        ``langsmith_dataset_name``, so absent means hand-edited or foreign metadata —
        exactly where the conservative read is right.)
        """
        resolved_name = self._config.dataset or dataset_name
        recorded_name = existing_metadata.get('langsmith_dataset_name')
        prior_examples = existing_metadata.get('langsmith_examples')
        if recorded_name is not None and recorded_name != resolved_name:
            # Never silent: dropping handles means CREATING a second copy of the suite in
            # another dataset. An operator who changed LANGSMITH__DATASET by accident (stale
            # shell, wrong env file) would otherwise see an ordinary "uploaded N/N" line.
            logger.warning(
                f'LangSmith: the recorded example handles belong to dataset {recorded_name!r} but this upload '
                f'targets {resolved_name!r} (LANGSMITH__DATASET override). Creating FRESH examples in '
                f'{resolved_name!r}; the examples in {recorded_name!r} are left untouched and are no longer tracked.'
            )
            prior_examples = None
        return sync_scenarios_to_langsmith(
            api_key=self._config.api_key,
            dataset_name=resolved_name,
            dataset_description=dataset_description,
            scenarios=scenarios,  # type: ignore[arg-type]  # ScenarioUploadEntry duck-types the {folder_name, inputs, ...} dict shape
            existing_examples=prior_examples,
        )
