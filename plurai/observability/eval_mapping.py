"""Shared typed-score / scope / naming core for the observability sinks.

A LEAF module (sibling to ``rollup.py``, NO vendor SDK imports, so it imports
with no extra installed): every sink needs the same three answers about a backend
PolicyResult before it can post feedback, and this module is the single owner of
all three so the answers cannot drift per-vendor:

1. *What kind of score is this, and what are its numeric + native forms?* —
   :func:`map_eval_score` → :class:`TypedScore`. Vendors disagree on what a
   "score" is (LangSmith wants value+score, Braintrust wants 0–1, Arize wants
   label+score columns), so the mapping produces BOTH a numeric ``score``
   (always present, the cross-vendor comparable form) and the native ``value``
   (bool / number / category string) and lets each sink pick what it can carry.
2. *What does the score attach to?* — :func:`eval_scope` / :func:`turn_index`.
   The backend either emits ``scope`` / ``turn_idx`` or omits them, and the
   defensive default ('scenario', None) reproduces an omitting server's behavior
   byte-for-byte — so the backend can start emitting per-turn results without a
   client release.
3. *What is the UI-facing name?* — :func:`clean_name` +
   :data:`BARE_BUILTIN_NAMES`. The backend's ``plurai_`` name prefix arrives on the wire;
   this is the single source of truth for stripping it, and the bare tuple is
   the one list the all-four-sinks rename (and its invariant tests) import.

Plus the two RECORD-shape answers the record sinks (LangSmith + Langfuse) need
identically at trace time:

4. *What is a scenario record called in a vendor UI?* — :func:`scenario_record_name`
   (the ``scenario:<name>`` convention) so the LangSmith anchor RunTree, the
   Langfuse trace, and the Langfuse experiment-item record all read identically
   and no sink hand-rolls the prefix.
5. *Which scenario-metadata keys ride along on a record?* —
   :func:`build_scenario_metadata` + :data:`SCENARIO_METADATA_ALLOWED_KEYS`, the
   allow-list filter that keeps prompt-engineering internals (``tool_verbs``,
   ``knowledge_plan``) out of vendor UIs while both sinks emit the IDENTICAL
   mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Recognized name prefixes, stripped (once, first match) by :func:`clean_name`. ``plurai_`` is the
# wire prefix the backend puts on built-in rollup names; it never reaches a vendor UI.
_NAME_PREFIXES: tuple[str, ...] = ('plurai_',)

# Deterministic categorical→numeric companion mapping, used ONLY when a str label arrives
# WITHOUT a ``violated`` bool (when ``violated`` is present it is authoritative — see
# :func:`map_eval_score`). The sets are intentionally tiny and fixed: the point is a STABLE
# numeric companion for the common pass/fail vocabularies so vendor charts stay comparable
# across runs, not a fuzzy classifier. Anything unrecognized maps to 0.5 (ambiguous), the same
# convention ``rollup.score_from_result`` already uses for unknowns.
_CATEGORICAL_PASS: frozenset[str] = frozenset({'pass', 'passed', 'true', 'yes', 'success'})
_CATEGORICAL_FAIL: frozenset[str] = frozenset({'fail', 'failed', 'false', 'no', 'violation', 'violated'})

# The built-in rollup ROUTING KEYS, bare (UI-facing) form, one named constant each. These are
# not labels: braintrust routes them between its scores/metrics channels BY NAME, arize derives
# the passed/failed label BY NAME, and arize_experiments builds evaluator columns from them —
# so producer (rollup.py) and every consumer reference THESE constants, never a string literal,
# making producer/consumer drift unrepresentable.
SCENARIO_STATUS = 'scenario_status'
TOOL_CALLS = 'tool_calls'
FINAL_ENTITIES = 'final_entities'
AGGREGATE = 'aggregate'
SCENARIO_PASSED = 'scenario_passed'
PASS_RATE = 'pass_rate'  # noqa: S105 — a metric routing key, not a credential (S105 keys on the 'pass' substring)
CHALLENGE_LEVEL = 'challenge_level'
FLOW_COMPLEXITY = 'flow_complexity'
TARGET_FLOWS = 'target_flows'
COMPLICATIONS_COUNT = 'complications_count'

# Every key ``rollup.compute_rollup_metrics`` can emit. Ordered: the four always-present
# rollups first (``rollup.ALWAYS_PRESENT_ROLLUP_NAMES`` is the [:4] slice of THIS tuple), then
# the two VERDICT-AXIS rollups — posted only when the axis the backend publishes is non-null, so a
# scenario nothing evaluated posts no number at all instead of a 1.0 nobody measured — then the
# four conditional dimension-graph rollups. Single source for the prefixed→bare
# rename — a name added to rollup.py without updating this tuple fails a cross-check test.
#
# RESERVED NAMES — eval authors, read this: every name in this tuple is also a per-scenario
# row the sinks GENERATE automatically (the rollup/routing keys, posted under their bare
# names). A user eval named identically to one of them — e.g. a policy literally called
# ``scenario_passed``, ``pass_rate``, ``tool_calls`` — name-COLLIDES with the generated row
# on the vendor dashboards: the two share a key and become indistinguishable (and may
# overwrite each other). The bare names ARE the routing keys, so this is not cosmetic. Do not
# name a user eval after any built-in rollup name; this tuple is the authoritative list of the
# reserved set.
BARE_BUILTIN_NAMES: tuple[str, ...] = (
    SCENARIO_STATUS,
    TOOL_CALLS,
    FINAL_ENTITIES,
    AGGREGATE,
    SCENARIO_PASSED,
    PASS_RATE,
    CHALLENGE_LEVEL,
    FLOW_COMPLEXITY,
    TARGET_FLOWS,
    COMPLICATIONS_COUNT,
)


@dataclass(frozen=True, slots=True)
class TypedScore:
    """One typed eval score: numeric form ALWAYS present, native form preserved.

    ``score`` is the cross-vendor comparable number (every vendor can chart a
    float); ``value`` is the native form (bool for pass/fail, the raw number,
    the category string) for vendors whose UI can render it, or ``None`` when
    the result carried nothing usable (kind 'unknown' — still emitted, never
    dropped, so a malformed PolicyResult is VISIBLE in the sink as an ambiguous
    0.5 instead of silently missing).
    """

    key: str
    kind: Literal['boolean', 'categorical', 'number', 'unknown']
    score: float
    value: float | str | bool | None
    comment: str


@dataclass(frozen=True, slots=True)
class TurnScore:
    """One per-turn verdict: a :class:`TypedScore` plus the 0-based turn it attaches to.

    The typed pairing the sinks iterate for per-turn placement — ``turn_idx``
    is the validated 0-based index (the same key that anchors the turn observation), ``score``
    is derived by the SAME :func:`map_eval_score` path scenario-level scores use, so a per-turn
    and a scenario score are typed identically.
    """

    turn_idx: int
    score: TypedScore

    def __post_init__(self) -> None:
        # Enforce the invariant ``per_turn_results`` already guarantees (via :func:`turn_index`)
        # at the TYPE boundary too, so a future second construction site cannot mint a turn
        # score no sink can anchor: ``turn_idx`` must be a real 0-based index, never a bool
        # (an int subclass that would alias turn 0/1) and never negative.
        if isinstance(self.turn_idx, bool) or self.turn_idx < 0:
            raise ValueError(f'TurnScore.turn_idx must be a non-negative, non-bool int, got {self.turn_idx!r}')


def map_eval_score(name: str, result: dict[str, Any]) -> TypedScore:
    """Map one backend PolicyResult to a :class:`TypedScore`. Total — never raises, never drops.

    CANONICAL INPUT SHAPE (the one this module understands): the backend per-eval result dict
    ``{violated: bool?, label: bool|int|float|str?, reasoning: str?}`` plus the future optional
    scope fields ``{scope?, turn_idx?}`` read by :func:`eval_scope` / :func:`turn_index`. This is
    deliberately NOT polymorphic over ``rollup.RollupMetric`` (accepting both shapes would force a
    rollup import here and make the leaf circular); rollup consumers adapt at the call site — a
    RollupMetric maps directly: key→key, score→score, value→value, comment→comment, with kind
    'boolean' for the pass/fail rollup (:data:`SCENARIO_PASSED`), 'categorical' for a str-valued
    one (:data:`SCENARIO_STATUS`, ``target_flows``, ``complications_count``) and 'number' for
    the rest. :func:`~plurai.observability.feedback.map_rollup_score` is that adapter.

    The LABEL's type drives the ``kind`` (a str label stays categorical even when ``violated``
    is also present); ``violated`` — the canonical PolicyResult field — drives the pass/fail
    SCORE wherever the label is not itself the number:

    - bool label / bool ``violated`` → 'boolean': score 1.0 (pass) / 0.0 (fail), value True/False.
      When both are present ``violated`` wins (it is authoritative in the backend — consistent with
      ``rollup.score_from_result``, which also treats ``violated`` as primary).
    - int/float label → 'number': the label IS the score (e.g. a 0–10 rating); value keeps the raw number.
    - str label → 'categorical': value = the label; the numeric companion comes from ``violated``
      when present (1.0 not-violated / 0.0 violated), else the deterministic
      :data:`_CATEGORICAL_PASS` / :data:`_CATEGORICAL_FAIL` mapping (unrecognized → 0.5).
    - neither → 'unknown': score 0.5 (ambiguous), value None — emitted, NEVER dropped.
    """
    comment = result.get('reasoning') or ''
    key = clean_name(name)
    label = result.get('label')
    violated = result.get('violated')
    if isinstance(label, bool):
        # bool label is already pass/fail — but ``violated`` stays authoritative when both exist.
        passed = (not violated) if isinstance(violated, bool) else label
        return TypedScore(key, 'boolean', 1.0 if passed else 0.0, passed, comment)
    if isinstance(label, (int, float)):
        return TypedScore(key, 'number', float(label), label, comment)
    if isinstance(label, str):
        if isinstance(violated, bool):
            score = 0.0 if violated else 1.0
        elif label.lower() in _CATEGORICAL_PASS:
            score = 1.0
        elif label.lower() in _CATEGORICAL_FAIL:
            score = 0.0
        else:
            score = 0.5
        return TypedScore(key, 'categorical', score, label, comment)
    if isinstance(violated, bool):
        passed = not violated
        return TypedScore(key, 'boolean', 1.0 if passed else 0.0, passed, comment)
    return TypedScore(key, 'unknown', 0.5, None, comment)


def eval_scope(result: dict[str, Any]) -> Literal['turn', 'scenario']:
    """Which level a PolicyResult attaches to. Defensive: only the exact string 'turn'
    opts into per-turn attachment; everything else (missing, junk, future values) stays
    'scenario'. A server that omits ``scope`` lands on the default, which IS its behavior."""
    return 'turn' if result.get('scope') == 'turn' else 'scenario'


def turn_index(result: dict[str, Any]) -> int | None:
    """The 0-based turn a turn-scoped result points at, or None when absent/unusable.

    Defensive on the same axis as :func:`eval_scope`: only a non-bool, non-negative int is
    trusted (bool is an int subclass and would silently alias turn 0/1; a negative index is
    junk no sink can anchor). A server that omits ``turn_idx`` yields None — the norm there.
    """
    idx = result.get('turn_idx')
    if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
        return idx
    return None


def per_turn_results(name: str, result: dict[str, Any]) -> list[TurnScore]:
    """The per-turn verdict STREAM carried on a turn-scoped policy cell,
    as TYPED :class:`TurnScore` objects.

    The backend emits a turn_based policy's cell as a SUPERSET: the rolled-up
    scenario verdict stays at the top level (read unchanged by ``map_eval_score`` /
    ``build_scenario_summary``) PLUS a ``turns`` sub-list of per-turn entries, each
    ``{turn_idx, value, type, reasoning, violated}``. Each entry becomes a :class:`TurnScore`
    so a sink can place ONE typed score per turn on its matching turn observation WITHOUT any
    stringly-typed dict access.

    Each entry's :class:`TypedScore` is derived through the SAME :func:`map_eval_score` path
    scenario-level scores use — the entry's typed ``value`` is fed in as the ``label`` the
    mapping keys off (``type`` is informational; the kind is inferred from ``value``), with
    ``violated`` authoritative for the pass/fail score — so per-turn and scenario scores can
    never disagree on kind/score/value.

    Defensive + total (never raises): a missing / non-list ``turns`` ⇒ ``[]`` (the default for
    a scenario-scoped policy and for a server that omits ``turns`` alike — no per-turn
    placement), and an entry without a usable :func:`turn_index` (missing / bool / negative /
    non-int) is dropped — a sink can only anchor on a real 0-based turn key. **Tolerates a
    list SHORTER than ``max_turns``**
    (early policy termination yields fewer entries): callers iterate exactly what is present
    and never assume a fixed length.
    """
    turns = result.get('turns')
    if not isinstance(turns, list):
        return []
    out: list[TurnScore] = []
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        idx = turn_index(entry)
        if idx is None:
            continue
        score = map_eval_score(name, {'label': entry.get('value'), 'violated': entry.get('violated'), 'reasoning': entry.get('reasoning')})
        out.append(TurnScore(turn_idx=idx, score=score))
    return out


# The record-metadata ALLOW-LIST: exactly the scenario-metadata keys that may surface on a
# vendor record (LangSmith anchor RunTree ``extra.metadata``; Langfuse dataset item + trace
# metadata). Copying THROUGH this tuple — never copying the raw dict — is what keeps the
# prompt-engineering internals the backend also ships in ``scenario_metadata`` (``tool_verbs``,
# ``knowledge_plan``) out of customer UIs: an unlisted key is unrepresentable in the output,
# so a new server-side field stays private until it is deliberately added here.
SCENARIO_METADATA_ALLOWED_KEYS: tuple[str, ...] = (
    'scenario_name',
    'scenario_target',
    'success_criteria',
    'target_flows',
    'challenge_level',
    'flow_complexity',
    'complexity_level',
    'subtlety_level',
    'current_elements',
    'scenario_type',
    'is_knowledge_scenario',
)


# Facts about the RUN that the orchestrator injects into the scenario_metadata mapping it hands
# the sinks. The backend never sends these and they are not part of a scenario's definition, so they
# ride their own list rather than the allow-list above: they belong on a run RECORD and NOT in
# the dataset Example inputs, which ``filter_scenario_metadata`` builds from the definition keys
# alone. ``ended_early`` is the typed reason the dialog ended for something other than the user
# reaching their goal: the budget ran out (``orchestrator.TURN_CAP_REASON``), every turn was
# emptied by the parent (``ADAPTER_DEGRADED_REASON``), or the simulated user gave up
# (``USER_GAVE_UP_REASON``, reported by the backend on the /user-turn response). Without it such a
# scenario renders as an ordinary short one, beside evals that are low precisely because the run
# never reached them. Every producer here either knows its fact first-hand or relays it from the
# one party that does. ``ended_early_cause`` is the simulator's own
# stated cause for its give-up (``orchestrator.GIVE_UP_CAUSE_*``), relayed on the same terms:
# it is what separates "the agent refused" from "the user never saw the reply" on the record.
# ``trial_index`` / ``trial_count`` / ``trial_group`` name WHICH trial of a ``--trials N`` run the
# record belongs to (the runner passes them to ``Orchestrator.run``): each trial is a run of its own
# on every sink, and these are what tell its records apart even when a sink's project is shared.
# ``run_id`` is the per-invocation run handle (the ``run_name`` in ``run_summary.json`` and in
# the transcript metadata), ``scenario_id`` / ``scenario_folder`` name the scenario the way the
# runner addresses it, ``scenario_set_id`` the set it belongs to and ``sdk_version`` the build
# that ran it: together they join a trace in the vendor UI back to the local run artifacts. Both
# scenario keys ride because readers search the trace store by either spelling.
RUN_FACT_KEYS: tuple[str, ...] = (
    'ended_early',
    'ended_early_cause',
    'trial_index',
    'trial_count',
    'trial_group',
    'run_id',
    'scenario_id',
    'scenario_folder',
    'scenario_set_id',
    'sdk_version',
)


def filter_scenario_metadata(scenario_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """The allow-listed scenario-DEFINITION subset (no run identifiers).

    Like :func:`build_scenario_metadata` but WITHOUT the thread/experiment/agent_version
    run identifiers — for the dataset Example INPUT snapshot, which is the scenario
    definition, not a run record. Copying THROUGH :data:`SCENARIO_METADATA_ALLOWED_KEYS`
    keeps the prompt-engineering internals the backend ships in ``scenario_metadata``
    (``tool_verbs`` / ``knowledge_plan``) and the backend's ``experiment`` run handle out of the UI's
    Inputs column. Total: never raises, never mutates its input.
    """
    md = scenario_metadata or {}
    return {key: md[key] for key in SCENARIO_METADATA_ALLOWED_KEYS if key in md}


def turn_record_name(turn_idx: int) -> str:
    """The UI-facing name of one turn observation — the 1-based ``turn N`` convention.

    ``turn_idx`` is the 0-based wire key (kept verbatim everywhere it indexes scoring /
    turn-observation lookups); only the DISPLAY label is 1-based, so ``turn_idx=0`` renders as
    ``turn 1``. THE single source of this label, shared by every sink's turn span/run name (and
    their tests), so the four surfaces can never drift apart.
    """
    return f'turn {turn_idx + 1}'


def scenario_record_name(scenario_name: str) -> str:
    """The UI-facing name of one scenario record — the ``scenario:<name>`` convention.

    THE single source of the prefix, shared by the LangSmith anchor RunTree name, the
    Langfuse trace name, and the Langfuse experiment-item record rename (and their
    tests), so the three surfaces can never drift apart.
    """
    return f'scenario:{scenario_name}'


def build_scenario_metadata(
    *,
    scenario_metadata: dict[str, Any] | None,
    thread_id: str | None = None,
    experiment_name: str | None = None,
    agent_version: str | None = None,
) -> dict[str, Any]:
    """Build the ONE record-metadata mapping both record sinks emit (allow-list filtered).

    Copies EXACTLY the :data:`SCENARIO_METADATA_ALLOWED_KEYS` present in the thread's
    ``scenario_metadata`` (a key the backend did not send is simply absent — never
    None-padded) plus any :data:`RUN_FACT_KEYS` the orchestrator injected, then adds the
    run identifiers ONLY when supplied: ``thread_id``, ``experiment`` (the per-invocation
    experiment/run handle) and ``agent_version`` (the operator's ``--agent-version`` tag —
    absent entirely when untagged, so old records are distinguishable from records of an
    unversioned run). Total: never raises, never mutates its input.
    """
    md = scenario_metadata or {}
    out: dict[str, Any] = {key: md[key] for key in SCENARIO_METADATA_ALLOWED_KEYS if key in md}
    out.update({key: md[key] for key in RUN_FACT_KEYS if key in md})
    if thread_id:
        out['thread_id'] = thread_id
    if experiment_name:
        out['experiment'] = experiment_name
    if agent_version:
        out['agent_version'] = agent_version
    return out


def clean_name(raw: str) -> str:
    """Strip the backend's ``plurai_`` wire prefix — THE single source of truth
    for UI-facing metric/eval names, so no sink hand-rolls its own stripping.

    Strips at most one prefix (the names are singly-prefixed on the wire), and refuses to
    strip down to an empty string (a name that is only the prefix is returned unchanged rather than
    producing an unusable empty key).
    """
    for prefix in _NAME_PREFIXES:
        if raw.startswith(prefix) and len(raw) > len(prefix):
            return raw.removeprefix(prefix)
    return raw
