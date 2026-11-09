"""Vendor-neutral rollup-metric computation for the observability sinks.

The metric math ONLY — the per-sink posting is sink-specific and lives on each sink
class. Data sources are the client-side shapes: the run-local ``final_world_state`` +
``tool_events`` params + ``chatbot_messages`` (NEVER ``ThreadResult.tool_events``,
empty in-process).

This is a LEAF module (depends only on ``tool_call_recovery`` and the
``eval_mapping`` naming core); every sink module imports the shapes/functions it
needs from here, so the rollup math has a single owner and no sink-specific
knowledge leaks in. The rollup KEYS are the bare routing-key constants from
``eval_mapping`` (single source — never string literals), so the producer cannot
drift from the braintrust/arize consumers that route by these names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from plurai.observability.eval_mapping import (
    AGGREGATE,
    BARE_BUILTIN_NAMES,
    CHALLENGE_LEVEL,
    COMPLICATIONS_COUNT,
    FINAL_ENTITIES,
    FLOW_COMPLEXITY,
    PASS_RATE,
    SCENARIO_PASSED,
    SCENARIO_STATUS,
    TARGET_FLOWS,
    TOOL_CALLS,
)
from plurai.observability.scenario_verdict import read_scenario_verdict
from plurai.observability.tool_call_recovery import TOOL_CALL_COUNT_FLOOR_KEY, merged_tool_calls_with_floor


@dataclass(frozen=True, slots=True)
class RollupMetric:
    """One rollup feedback row: a key, a numeric score, a comment, and a value.

    ``value`` is honestly ``float | str``: most rollups carry the numeric ``score``
    again, but the list-valued ones (``target_flows`` /
    ``complications_count``) carry a comma-joined string. (The larger
    score/display split is intentionally out of scope.)
    """

    key: str
    score: float
    comment: str
    value: float | str


def score_from_result(result: dict) -> float:
    """Map a backend PolicyResult to a 0.0–1.0 score for sink feedback.

    Prefer ``violated`` since every PolicyResult has it; fall back to numeric
    label values when present. The point of the score is comparability across
    policies in the sink UI — a binary "passed/failed" framing matches how
    the backend already aggregates.

    DIVERGENCE NOTE: this is the braintrust/arize scoring path; langfuse/langsmith
    score the same cell through :func:`eval_mapping.map_eval_score`. The two AGREE on
    the bool/``violated`` case (the overwhelming majority), but diverge for a *str-label*
    ("categorical") policy that carries NO ``violated`` bool: ``map_eval_score`` maps a
    recognized pass/fail vocabulary word (e.g. "pass"→1.0) via its categorical branch,
    whereas this function has no categorical branch and returns 0.5 (ambiguous). Both
    treat ``violated`` as authoritative, so a normal backend cell (which always carries
    ``violated``) scores identically everywhere; the gap only shows for label-only
    categorical evals. Kept deliberately: braintrust and arize score an unknown label 0.5.
    """
    violated = result.get('violated')
    if violated is True:
        return 0.0
    if violated is False:
        return 1.0
    # Unknown — fall back to numeric label if it's a number, else 0.5 (ambiguous).
    label = result.get('label')
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return float(label)
    return 0.5


def eval_results(policy_results: dict) -> dict:
    """Unwrap the evaluation_results map from a backend policy-results dict.

    Accepts both the flat ``{evaluation_results: {...}}`` shape and the nested
    ``{aggregated_results: {evaluation_results: {...}}}`` shape."""
    return policy_results.get('evaluation_results') or (policy_results.get('aggregated_results') or {}).get('evaluation_results') or {}


# The always-present rollups — DECLARED for every scenario regardless of metadata. Single source
# of truth for "which built-in rollup COLUMNS always exist", and
# ``arize_experiments.build_passthrough_evaluators`` imports
# this tuple (with :data:`VERDICT_ROLLUP_NAMES`) to build the experiment's evaluator columns — a
# column for a CONDITIONAL dimension rollup would be all-0.0 noise on runs that lack it.
# The [:4] slice of the shared ``BARE_BUILTIN_NAMES`` tuple, whose documented order puts the
# always-present four first — a test pins the slice
# against the explicit names so a reorder of the source tuple fails loudly instead of silently
# changing this set.
#
# DECLARED, not unconditionally emitted, and the distinction is one metric wide: ``tool_calls``
# is WITHHELD on a run whose tool channel was never read and whose recovery lanes found nothing
# (:func:`compute_rollup_metrics` — an unread channel is not a zero). Its column stays here
# because a column that disappears on those runs would delete "how many tools" from the grid for
# everyone; an absent CELL is a distinguishable value, a missing COLUMN is not.
#
# ``scenario_passed`` and ``pass_rate`` are NOT in it; they live in
# :data:`VERDICT_ROLLUP_NAMES`. They are the two VERDICT AXES, and the backend publishes each as
# nullable: a scenario nothing evaluated has no pass/fail, and a scenario whose measurement did
# not hold has no rate worth publishing. They are posted only when their axis is non-null, and
# :data:`~plurai.observability.eval_mapping.SCENARIO_STATUS` — which IS always present — carries
# what happened in every case.
ALWAYS_PRESENT_ROLLUP_NAMES: tuple[str, ...] = BARE_BUILTIN_NAMES[:4]

# The two verdict-axis rollups: emitted only when the backend published that axis, but DECLARED as
# experiment columns all the same. An arize experiment materializes one column per declared
# evaluator, so leaving these undeclared would delete "did it pass" from the grid for every run,
# including a fully-graded one — while declaring them costs only a ``label='absent'`` cell on the
# rows that have no verdict, which is a distinguishable value rather than a manufactured verdict
# (``arize_experiments.PassthroughEvaluator`` scores an absent entry 0.0 and labels it 'absent').
VERDICT_ROLLUP_NAMES: tuple[str, ...] = BARE_BUILTIN_NAMES[4:6]


def tool_call_floor(tool_events: dict | None, local_floor: dict) -> dict[str, int]:
    """``{tool: rows suppressed}`` — every suppression this rollup can see, as a LOWER BOUND.

    Two producers, because the payload reaches here already merged. The orchestrator unions the
    out-of-band lanes into ``tool_events`` before ``/finalize`` and stamps what the name rule
    could not verify under :data:`TOOL_CALL_COUNT_FLOOR_KEY`. The union THIS module runs over that
    payload floors its own rows and reports them too.

    Combined by MAX per tool, never by sum: the local union re-runs the message lane against
    events that already contain it, so its drops are largely re-detections of the upstream ones
    and adding them would publish a hole twice the size of the real one. A floor is a lower
    bound, and the max of two lower bounds is still one.
    """
    published = (tool_events or {}).get(TOOL_CALL_COUNT_FLOOR_KEY)
    upstream = published.get('by_tool') if isinstance(published, dict) else None
    combined: dict[str, int] = {}
    for source in (upstream if isinstance(upstream, dict) else {}, local_floor):
        for name, dropped in source.items():
            if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped > 0:
                combined[str(name)] = max(combined.get(str(name), 0), dropped)
    return dict(sorted(combined.items()))


def compute_rollup_metrics(
    *,
    policy_results: dict,
    final_world_state: dict | None,
    tool_events: dict | None,
    chatbot_messages: list | None,
    scenario_metadata: dict | None,
    agent_states: object = None,
    agent_actions_by_turn: object = None,
    pass_rate: float | None = None,
    aggregate_score: float = 0.0,
) -> list:
    """Compute the per-scenario rollup metrics.

    The four always-on rollups (:data:`ALWAYS_PRESENT_ROLLUP_NAMES`: scenario_status /
    tool_calls / final_entities / aggregate [the secondary reliability number, from the
    FinalizeResponse via ``aggregate_score``]) — ``tool_calls`` with ONE exception, below — the
    two VERDICT-AXIS rollups when their axis is non-null (scenario_passed / pass_rate), plus the
    dimension-graph metadata metrics when present.

    THE TOOL CHANNEL HAS THREE STATES, and only two of them are a count. ``tool_events`` is
    ``None`` when the run took no tool snapshot at all (a child with no session; a hop reaped at
    its turn budget), ``{}``/``{'events': []}`` when it looked and found nothing. The first is not
    a measurement, so when it is ``None`` AND neither recovery lane found a call, ``tool_calls``
    is WITHHELD rather than posted as 0.0 — the same treatment the pass axis gets below, for the
    same reason, and the column is DECLARED regardless
    (:data:`ALWAYS_PRESENT_ROLLUP_NAMES`), so such a row reads absent rather than losing the
    column. A read-and-empty channel still posts its zero: that one IS a measurement.

    Every sink emits whatever this returns, so all of it lands uniformly with no
    per-sink special-casing. ``tool_events`` + ``chatbot_messages`` + ``agent_states``
    feed :func:`merged_tool_calls_with_floor` (the same union recovery the backend uses);
    ``final_world_state`` is the run-local snapshot, accepted in both the ``{state: {...}}`` and
    bare ``{entities: {...}}`` envelopes.

    THE VERDICT IS READ, NEVER DERIVED. Deriving "did the scenario pass?" from "was any policy
    violated?" would score an EMPTY policy set 1.0, and a scenario whose only judge crashed
    would post ``scenario_passed 1.0`` while ``run_summary.json`` said the measurement was
    unclean. The backend owns that answer (it knows which policy owns the goal and whether the
    measurement held); :func:`~plurai.observability.scenario_verdict.read_scenario_verdict` reads it.

    A NULL AXIS IS POSTED AS ONE CATEGORICAL, NEVER BY OMISSION. ``scenario_status`` is always
    present and carries ``passed`` / ``failed`` / ``not_evaluated`` / ``unclean`` / ``unknown``.
    Omitting a metric to signal a null is survivorship bias — an omitted row is indistinguishable
    from a sink outage and vendor aggregates silently drop it, so a null-heavy run would read
    BETTER than a graded one. Its numeric companion is deliberately NOT a pass score (the pass
    axis is ``scenario_passed``, which is absent exactly when there is no pass to report): it is
    1.0 iff the scenario reached a conclusive goal verdict, so the column's mean is EVALUATION
    COVERAGE and a not-evaluated row can never chart as a red. The one number it must never be
    is 0.5 — ``map_eval_score`` already spends that on an unknown per-eval cell, where it charts
    as half a pass.

    ``agent_states`` is the harness's self-reported tool trail, per turn — the THIRD recovery
    lane, and the only one that sees an agent running its own tool loop. Optional, because a sink that never built
    the per-turn list must not be forced to invent one: ``None`` reads as "this caller has no
    trail", never as "the agent made no out-of-band calls". ``agent_actions_by_turn`` is the typed
    action lane (renders, handoffs) under the same rule: ``None`` means this caller has no transcript rows.

    ``pass_rate`` is likewise ``None``-able and forwarded from ``FinalizeResponse.pass_rate``
    VERBATIM: coercing it with ``or 0.0`` would turn "the backend scored nothing" into a posted
    0.0, a manufactured red. It is used only when
    the verdict block published no ``score`` of its own.
    """
    results = eval_results(policy_results)
    verdict = read_scenario_verdict(policy_results)

    # Tool-call count — union of tool_events.events + chatbot_messages + agent-trail recovery,
    # WITH the union's floor. `tc_count` is a LOWER BOUND whenever the name rule had to drop a
    # row it could not verify, and this metric is a sink column a human reads as a measurement:
    # publishing `2 tool_call events` for a run whose union suppressed one is the same silent
    # under-count on the reporting side that `union_by_call_identity` exists to end on the
    # grading side.
    tool_calls, local_floor = merged_tool_calls_with_floor(tool_events, chatbot_messages, agent_states, agent_actions_by_turn=agent_actions_by_turn)
    tc_count = len(tool_calls)
    floored = tool_call_floor(tool_events, local_floor)
    # THE UNREAD CHANNEL, and it is not a zero. ``tool_events`` is ``dict | None`` (``None`` =
    # the child took no tool snapshot at all) and ``merged_tool_calls_with_floor(None, …)``
    # returns an empty union, so publishing it would claim "0 tool_call events" — a
    # measurement, in the one column a human reads as a measurement, about a run nobody
    # watched. Same reasoning as ``pass_rate`` above: "scored nothing" is not a 0.0.
    #
    # AND ONLY when the two recovery lanes also came back empty: one recovered call IS an
    # observation, and a count built out of it is a real lower bound worth publishing.
    tool_channel_unread = tool_events is None and tc_count == 0
    # No floor → no note: a floor note on every clean run is noise that hides the real ones.
    tc_comment = f'{tc_count} tool_call events'
    if floored:
        named = ', '.join(f'{name}x{n}' for name, n in floored.items())
        tc_comment = (
            f'{tc_count} tool_call events (floor: {sum(floored.values())} suppressed — {named}); the two recovery lanes '
            'could not be matched by call_id, so this count is a LOWER BOUND, not a measurement'
        )

    # Final entity count. Accept both {state: {...}} (SDK envelope) and
    # bare {entities: {...}} (direct callers).
    entity_count = 0
    if isinstance(final_world_state, dict):
        state = final_world_state.get('state') or final_world_state.get('entities') or {}
        if isinstance(state, dict):
            entity_count = sum(1 for v in state.values() if isinstance(v, dict))

    metrics: list = [
        RollupMetric(
            SCENARIO_STATUS,
            1.0 if verdict.conclusive else 0.0,
            f'scenario_status={verdict.status} over {len(results)} policies — {verdict.note}. '
            '1.0 = a conclusive goal verdict was reached (the PASS axis is scenario_passed, absent when there is none)',
            verdict.status,
        ),
        RollupMetric(FINAL_ENTITIES, float(entity_count), f'{entity_count} entities in final world-state', float(entity_count)),
        # The always-on quality signal from the FinalizeResponse — emitted through the same
        # channel as the other rollups so every sink lands it (no per-sink special-casing).
        RollupMetric(AGGREGATE, aggregate_score, f'aggregate_score={aggregate_score} (secondary reliability number)', aggregate_score),
    ]
    if not tool_channel_unread:
        # Position 1, where every positional reader of this list expects it — inserted rather than
        # appended so positional readers of the list find it where they expect it.
        metrics.insert(1, RollupMetric(TOOL_CALLS, float(tc_count), tc_comment, float(tc_count)))

    # THE PASS AXIS — posted only when the backend reached a verdict. ``not_evaluated`` / ``unclean``
    # / ``unknown`` post NO number here at all; ``scenario_status`` above already said which of
    # the three it was, so nothing is lost and no sink charts a null as a pass or as a red.
    if verdict.conclusive:
        passed_score = 1.0 if verdict.status == 'passed' else 0.0
        detail = f'catastrophic: {", ".join(verdict.catastrophic_violations)}' if verdict.catastrophic_violations else verdict.note
        metrics.append(RollupMetric(SCENARIO_PASSED, passed_score, f'{len(results)} policies; {detail}', passed_score))

    # THE SCORE AXIS — the POLICY rate, which is a different question from the goal: the
    # conversation's weighted score from THE rule (:mod:`plurai.scoring`, read through the verdict
    # reader as a 0-1 rate), independent of the pass axis — a catastrophic red leaves it high, so
    # ``failed`` beside ``pass_rate 0.9`` is normal and must not be floored. It survives a null GOAL
    # verdict for the same reason: the judges really did grade the turns.
    #
    # Two ways it is withheld, and only two:
    #   * NOBODY PUBLISHED ONE. No stamped eval was graded (an older backend's cells carry no band)
    #     AND the FinalizeResponse's own ``pass_rate`` is null. The fallback is on the VALUE, not the
    #     status, so a measurement the backend made is never dropped silently. The comment names which
    #     source it came from.
    #   * THE MEASUREMENT IS UNCLEAN. the backend says the grading itself did not hold, and this
    #     metric is labelled the headline quality signal — publishing a 1.0 headline for a row
    #     the backend declared untrustworthy would be a vacuous green.
    #     ``scenario_status`` says ``unclean`` in its place, so nothing is silently missing.
    if verdict.score is not None:
        rate, source = verdict.score, 'plurai.scoring'
    else:
        rate, source = pass_rate, 'finalize.pass_rate'
    if rate is not None and verdict.status != 'unclean':
        metrics.append(RollupMetric(PASS_RATE, float(rate), f'pass_rate={rate} from {source} (headline quality signal)', float(rate)))

    md = scenario_metadata or {}
    # Guard float() against non-numeric metadata: a single bad value must not crash
    # the whole computation (callers wrap this in a bare ``except Exception``, which
    # would drop ALL rollup metrics). Skip that one metric instead. ``bool`` is
    # excluded to mirror ``score_from_result``'s numeric check.
    cl = md.get('challenge_level')
    if isinstance(cl, (int, float)) and not isinstance(cl, bool):
        metrics.append(RollupMetric(CHALLENGE_LEVEL, float(cl), f'walk path_sum = {cl}', float(cl)))
    fc = md.get('flow_complexity')
    if isinstance(fc, (int, float)) and not isinstance(fc, bool):
        metrics.append(RollupMetric(FLOW_COMPLEXITY, float(fc), f'walk visited {int(fc) + 1} flow(s); {int(fc)} edge(s) traversed', float(fc)))
    if (tf := md.get('target_flows')) and isinstance(tf, list):
        metrics.append(RollupMetric(TARGET_FLOWS, float(len(tf)), ', '.join(str(f) for f in tf), ', '.join(str(f) for f in tf)))
    if (els := md.get('current_elements')) and isinstance(els, list):
        metrics.append(RollupMetric(COMPLICATIONS_COUNT, float(len(els)), ', '.join(str(e) for e in els), ', '.join(str(e) for e in els)))

    return metrics


@runtime_checkable
class _TranscriptCarrier(Protocol):
    """Structural protocol for objects that carry an agent transcript.

    ``@runtime_checkable`` means ``isinstance`` performs a ``hasattr`` check on
    each declared attribute — type accuracy is not verified at runtime.
    Objects with neither attribute (e.g. ``CreateThreadResponse``) fail the
    check and fall through to ``return []``.
    """

    chatbot_full_messages: list[Any]
    chatbot_messages: list[Any]


@runtime_checkable
class _TurnCarrier(Protocol):
    """Anything exposing the orchestrator's per-turn transcript records (``ThreadResult``)."""

    transcript_turns: list[Any]


def agent_states_of(thread: object) -> list[dict[str, Any]]:
    """Each turn's ``agent_state`` at the index of the turn that produced it, or ``[]``.

    The third recovery lane (:func:`merged_tool_calls_with_floor` reads a state's LIST POSITION
    as its turn number). Same rule as the orchestrator's ``_agent_states_by_turn`` — keyed on
    each record's authoritative ``turn_idx``, gaps padded with an empty state, an unusable index
    skipped rather than collapsed onto turn 0 — restated here because the orchestrator imports
    this package and cannot be imported back. An object with no ``transcript_turns`` (the
    ``CreateThreadResponse`` handshake) yields ``[]``: no trail lane, and the floor still reaches
    the rollup through the payload receipt (:func:`tool_call_floor`).
    """
    turns = getattr(thread, 'transcript_turns', None) if isinstance(thread, _TurnCarrier) else None
    if not isinstance(turns, list):
        return []
    by_turn: dict[int, dict[str, Any]] = {}
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        idx = entry.get('turn_idx')
        if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
            state = entry.get('agent_state')
            by_turn[idx] = state if isinstance(state, dict) else {}
    return [by_turn.get(i, {}) for i in range(max(by_turn) + 1)] if by_turn else []


def agent_actions_of(thread: object) -> list[list[dict[str, Any]]]:
    """Each turn's ``agent_actions`` at the index of the turn that filed them, or ``[]``.

    The typed action lane of :func:`merged_tool_calls_with_floor`, read off the same transcript
    records and by the same ``turn_idx`` rule as :func:`agent_states_of`.
    """
    turns = getattr(thread, 'transcript_turns', None) if isinstance(thread, _TurnCarrier) else None
    if not isinstance(turns, list):
        return []
    by_turn: dict[int, list[dict[str, Any]]] = {}
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        idx = entry.get('turn_idx')
        if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
            actions = entry.get('agent_actions')
            by_turn[idx] = [a for a in actions if isinstance(a, dict)] if isinstance(actions, list) else []
    return [by_turn.get(i, []) for i in range(max(by_turn) + 1)] if by_turn else []


def chatbot_messages_of(thread: object) -> list:
    """Best-effort extraction of the agent transcript for tool-call recovery.

    Accepts the orchestrator's ``CreateThreadResponse`` (no transcript) or any
    object exposing ``chatbot_messages`` / ``chatbot_full_messages``. Prefers
    the full (structured) messages since :func:`merged_tool_calls` recovers
    tool_use blocks from them; falls back to plain strings, then ``[]``.
    """
    if isinstance(thread, _TranscriptCarrier):
        full = thread.chatbot_full_messages
        if isinstance(full, list) and full:
            return full
        msgs = thread.chatbot_messages
        if isinstance(msgs, list):
            return msgs
    return []


# THE ZERO-ROW CLASSIFIER, shared with the Plurai plugin's post-run report, which owns it (its
# `zero_result_tool_calls` headline). The
# runner reads it for one decision: which ok tool calls came back as an EMPTY READ, the half of
# `run_simulation._world_unread_ids` that says "nothing came back". The SDK cannot import the
# plugin, so the bytes between the markers are the plugin's, byte for byte, and a twin test fails
# when the two copies differ. The docstring inside speaks for the
# plugin's headline; the runner's use is the one named here.
#
# The fenced bytes are another repo's, so this repo's tools may not restyle them: the formatter is
# switched off around the fence, and PLR1714 (which would rewrite the first comparison inside) is
# suppressed for this module on the next line.
# ruff: noqa: PLR1714
# fmt: off
# --- BEGIN shared zero-result classifier (mirror) --------------------------------------------
#: Keys whose value ANNOTATES a result set rather than BEING one — the shapes the live mock
#: actually emits beside an empty list (a prose `note`, a `count: 0`, the echoed `query`).
#: Everything else populated is data. See `_zero_result`.
_ANNOTATION_KEYS = frozenset({
    'note', 'notes', 'message', 'msg', 'detail', 'details', 'reason', 'status', 'error', 'errors',
    'warning', 'warnings', 'query', 'filters', 'count', 'total', 'total_count', 'num_results',
    'page', 'offset', 'limit', 'cursor', 'next_cursor', 'has_more',
})


def _json_container(value: object) -> object:
    """`value` decoded when it is JSON text of a list or dict; anything else unchanged."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        if isinstance(decoded, (list, dict)):
            return decoded
    return value


def _zero_result(payload: object) -> bool:
    """True when a tool RESPONSE is an empty list/dict/zero-row shape — the call succeeded
    and returned nothing. The live mock's zero-row shape is a dict whose list-valued keys
    are all empty beside non-list metadata (`{"results": [], "note": "…"}`), so a dict
    counts when it HAS list values, every one of them is empty, and NOTHING ELSE in it
    carries data — never when it merely contains an empty list beside populated ones.

    That last clause is the guard: `{"warnings": [], "balance": 500}` is a complete answer
    that happens to carry one empty list, and counting it as a zero-row retrieval failure
    inflates the very number this counter exists to measure. A populated non-list value is
    DATA unless its key annotates the result set (`_ANNOTATION_KEYS`); an annotation key
    this set does not know therefore UNDERcounts, which is the safe direction for a headline.

    It judges what the agent was HANDED: a mock success envelope (`status: success` + `content`)
    is judged on its `content`, and counts outright when its runtime sibling (`_plurai_runtime`)
    says `status: not_found` — the tool's declared nothing-found answer, whatever its shape. A
    runtime latency `not_found` (`grounding: not_found_no_repair_budget`) counts too: the mock
    could not read the world either. An empty map from a success envelope is a write ACK, not an
    empty read. JSON text of a list or dict is decoded ONLY in what a success envelope hands over
    (its content, and each of that map's values, one level). Every other shape — bare, or an
    error envelope (`status: error`, `_plurai_handler_error`, `_plurai_unregistered_tool`) — is
    judged by the rule above exactly as before, so no error envelope becomes countable.
    """
    if (isinstance(payload, dict) and payload.get('status') == 'success' and 'content' in payload
            and '_plurai_handler_error' not in payload and '_plurai_unregistered_tool' not in payload):
        runtime = payload.get('_plurai_runtime')
        if isinstance(runtime, dict) and runtime.get('status') == 'not_found':
            return True
        payload = _json_container(payload['content'])
        if payload == {}:
            return False  # a write ACK — the success envelope handed back nothing to read
        if isinstance(payload, dict):
            payload = {k: _json_container(v) for k, v in payload.items()}
    if payload == [] or payload == {}:
        return True
    if not isinstance(payload, dict):
        return False
    lists = [v for v in payload.values() if isinstance(v, list)]
    if not lists or any(lists):   # no list at all, or one of them has rows — not a zero-row shape
        return False
    return all(not v or k in _ANNOTATION_KEYS for k, v in payload.items() if not isinstance(v, list))
# --- END shared zero-result classifier (mirror) ----------------------------------------------
# fmt: on
