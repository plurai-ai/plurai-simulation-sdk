"""Shared post-feedback presentation core for the record sinks (LangSmith + Langfuse).

A LEAF module (imports only :mod:`plurai.observability.eval_mapping` — no vendor
SDKs, so it imports with no extra installed) owning the two POST-FEEDBACK answers
both record sinks need identically, so they cannot drift per-vendor:

1. *How does a rollup metric enter the typed-score path?* — :func:`map_rollup_score`.
   ``rollup.RollupMetric`` already carries the numeric/native split
   :class:`~plurai.observability.eval_mapping.TypedScore` models (``score`` +
   ``value``), so the conversion is a RE-SHAPE, not a re-mapping: a str-valued
   rollup (target_flows / complications_count carry comma-joined lists) becomes
   'categorical' (native string preserved), everything else 'number'. Routing
   rollups through the same posting path as the per-policy evals gives them the
   identical data_type treatment with no per-sink special-casing.

2. *What is the per-scenario record output?* — :func:`build_scenario_summary`.
   The one-coherent-record-per-scenario summary both sinks set as the scenario
   anchor's OUTPUT (LangSmith ``update_run(outputs=...)``; Langfuse anchor-span
   ``update(output=...)``): verdict + verdict_reason + per-eval typed results +
   turn_count + persona + world_changes. Built from the SAME ``map_eval_score`` mapping the
   live scores use for the per-eval table, and the SAME
   :func:`~plurai.observability.scenario_verdict.read_scenario_verdict` the rollup uses for the
   verdict — so the summary and the posted feedback can never disagree about either.

Deliberately does NOT import ``rollup.py``: ``rollup`` references
``eval_mapping`` (the bare-name routing constants), so a feedback->rollup import
edge would point the wrong way for this layer; callers pass the already-unwrapped
``evaluation_results`` mapping (``rollup.eval_results``) and the RollupMetric
FIELDS (key/score/value/comment scalars) instead.
"""

from __future__ import annotations

from typing import Any, Literal

from plurai.observability.eval_mapping import SCENARIO_PASSED, TypedScore, map_eval_score
from plurai.observability.scenario_verdict import read_scenario_verdict


def map_rollup_score(key: str, score: float, value: float | str, comment: str) -> TypedScore:
    """Re-shape one ``RollupMetric``'s fields into the :class:`TypedScore` posting shape.

    ``key`` passes through VERBATIM — the prefixed→bare rename happens at the producer
    (``rollup.compute_rollup_metrics``, sourced from the ``eval_mapping`` constants),
    never here. The numeric ``score`` is always the chartable form. Kind, per the canonical
    ``map_eval_score`` recipe (the rollup adapter is at THIS call site because
    ``eval_mapping`` must stay a leaf): the scenario pass/fail rollup
    (:data:`~plurai.observability.eval_mapping.SCENARIO_PASSED` — its 0.0/1.0 IS a
    verdict) is 'boolean'; a str ``value`` (the comma-joined list rollups, and
    :data:`~plurai.observability.eval_mapping.SCENARIO_STATUS`, whose five-word vocabulary
    is exactly what a categorical is for) keeps its native string as a 'categorical' value;
    everything else is a plain 'number'.
    """
    kind: Literal['boolean', 'categorical', 'number']
    if key == SCENARIO_PASSED:
        kind = 'boolean'
    elif isinstance(value, str):
        kind = 'categorical'
    else:
        kind = 'number'
    return TypedScore(key=key, kind=kind, score=score, value=value, comment=comment)


def build_scenario_summary(
    *,
    results: dict[str, Any],
    scenario_metadata: dict[str, Any] | None,
    final_world_state: dict[str, Any] | None,
    turn_count: int,
    policy_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the per-scenario summary record both sinks set as the anchor's OUTPUT.

    ``results`` is the already-unwrapped ``evaluation_results`` mapping
    (``rollup.eval_results(policy_results)``); ``policy_results`` is the whole mapping it came
    from, which is where the backend rides the verdict. The summary is guaranteed NON-EMPTY
    (``verdict`` / ``turn_count`` are always present) and DISTINCT per scenario in
    practice (per-eval reasoning + world_changes are scenario-specific by
    construction):

    * ``verdict`` — the backend's own ``scenario_verdict`` status, READ never re-derived:
      'passed' / 'failed' / 'not_evaluated' / 'unclean' / 'unknown'. Deriving it client-side
      from "any policy violated" would read an EMPTY policy set, or a scenario whose every
      judge crashed, as 'passed'. ``policy_results`` is REQUIRED
      (explicitly nullable, never defaulted): a caller with none must say so, because the
      alternative is a fifth sink silently posting 'unknown' for every scenario of a real run.
    * ``verdict_reason`` — one line saying WHY that status, so a 'not_evaluated' record does
      not send its reader hunting a crash that never happened.
    * ``results`` — per-eval typed entries via ``map_eval_score`` (kind / numeric
      score / native value / reasoning), keyed by the cleaned eval name. UNCHANGED by the
      verdict: a standard red stays visible in this table even on a scenario that passed.
    * ``turn_count`` — outer-conversation turns recorded for this scenario.
    * ``persona`` — from scenario metadata (``None`` when the backend sent none).
    * ``world_changes`` — the final world-state entity mapping; accepts both the
      ``{state: {...}}`` (SDK envelope) and bare ``{entities: {...}}``
      shapes, the same dual-envelope rule ``rollup.compute_rollup_metrics`` applies.
    """
    per_eval: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        if not isinstance(result, dict):
            continue
        ts = map_eval_score(name, result)
        per_eval[ts.key] = {'kind': ts.kind, 'score': ts.score, 'value': ts.value, 'reasoning': ts.comment}

    md = scenario_metadata or {}
    world_changes: dict[str, Any] = {}
    if isinstance(final_world_state, dict):
        state = final_world_state.get('state') or final_world_state.get('entities') or {}
        if isinstance(state, dict):
            world_changes = state

    verdict = read_scenario_verdict(policy_results)
    return {
        'verdict': verdict.status,
        'verdict_reason': verdict.note,
        'results': per_eval,
        'turn_count': turn_count,
        'persona': md.get('persona'),
        'world_changes': world_changes,
    }
