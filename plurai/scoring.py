"""THE ONE SCORING MODULE: a 0-100 score and a pass/fail per conversation, per scenario and for the suite, and the
agent's ship verdict — all computed from the attempts' ``verdicts.json`` files, never stored in them.

This module holds the ONLY copy of the weights (:data:`WEIGHTS`) and the cutoff (:data:`CUTOFF`). Nothing else
weighs a verdict or compares a score to 85; every surface (``run.json``, ``analytics.md``, the release
hand-off, the report) copies the ``scores`` block this module writes into ``run_summary.json``.

INPUTS. Each attempt's verdicts, read through :mod:`plurai.verdicts` (its one writer stamps nothing derived): every
eval's ``state``, the ``severity`` band and ``tier`` the backend stamped on its cell (:func:`plurai.verdicts.stamped_band`),
the attempt's ``status``, its ``goal`` block (the backend's ``scenario_verdict``) and its ``history``. From the run's
``run_summary.json``: the population (``scenarios``), the trial folders (``trials.dirs``) and each scenario's
likelihood (``scenario_likelihoods``, the ``path_probability`` the scenario set served; the runner records it).

CONVERSATION (one trial attempt of one scenario).

* Weight per eval by its band: critical 8, high 4, medium 2, low 1. ``score = 100 * passed weight / graded weight``
  over the evals in state ``pass`` or ``fail``. ``not_applicable`` and ``ungraded`` evals are left out and counted
  separately; a graded eval whose cell carries no band (the backend holds no body for it) is left out too and named
  under ``unweighted``. The value is unrounded; compare it as is, round only to display.
* It FAILS on any of (every reason that holds is recorded, in this order): ``low_score`` (score < 85),
  ``critical_failure`` (an eval of band/tier critical failed), ``failing_tier`` (an eval of tier failing failed;
  the goal owners — ``goal.target_judge_owners`` — are the goal axis, not this one), ``goal_missed``
  (``goal.goal_achieved`` is false; null is the status ``goal_undecided``, scored normally), ``crashed`` (status
  ``agent_error``: scored 0, its cells still name any critical failure).
* Status ``harness_error`` and ``incomplete`` are left out: ``result: not_scored``, no score, nothing counted.
* A critical eval that is no longer ``fail`` is RELEASED only when the history entry that moved it out of ``fail``
  is a red-cell audit ``FALSE_RED`` row with a typed cause (false positive: rubric/type/prompt/target_clause; test
  setup: scenario/harness/env/scan_fact) — a ``correction`` event, or a ``backend_corrections`` row of a rescore.
  Any other flip out of ``fail`` by a correction leaves it a critical failure. A rescore that re-judged the eval
  without re-applying a correction is a new verdict, not a release.

SCENARIO. The mean of its trial attempts' scores (a crash counts as 0; ``not_scored`` attempts are left out; a
``stability_retry/`` second sample is never read — it is not an attempt). It FAILS when the mean is < 85
(``low_score``), or when any attempt failed for a non-score reason (critical, failing tier, goal, crash).
``result: not_scored`` when no attempt was scored, or when every scored attempt is a crash with no graded weight: such
a scenario measured nothing, so it is unscored like any other (see AGENT) and its 0 reaches no mean. It keeps
``crashed`` under ``fail_reasons`` and its attempts keep their own 0, so a reader can say why it went unmeasured. A
crash beside an attempt that graded something, or with checks graded itself, stays a 0 in the mean.

SUITE. The mean of the scored scenarios' scores, weighted by likelihood when EVERY scored scenario has one (a
positive number), else equal weights (``equal_because`` names the scenarios without one). ``weights`` is each
scenario's effective share.

AGENT. ``no_verdict`` when nothing was scored (``nothing_scored``) and no scenario crashed (``agent_crashed``,
below). ``dont_ship`` when the suite is < 85 with every scenario scored (``suite_below_cutoff``), when some scenario
is unscored and even the BEST CASE suite is < 85 (``best_case_below_cutoff``; never ``suite_below_cutoff``, which
always means the measured score), or when any critical failure anywhere is not released (``critical_failure``). The
best case
(``best_case_score``) is the suite rule re-applied to the whole population with every unscored scenario (no scored
attempt) at 100 — the exact ceiling over the unscored scenarios (a partly scored scenario, some trials lost, counts
at its scored mean as final, as everywhere else): the suite is a mean whose weights do not depend on the scores, and
the re-applied rule follows the weighting switching to equal when an unscored scenario has no likelihood. With nothing
unscored it is the suite score itself. When a scenario is unscored, ``dont_ship`` also lists ``unscored_scenarios``
(named) as a run-integrity note; otherwise an unscored scenario gives ``no_verdict`` (``unscored_scenarios``): its
worst case is always a critical failure, so Ship is never said while any scenario is unscored. Otherwise ``ship``.
So a run whose summary says ``scored: false`` reads either No verdict or a Don't ship that no unscored result can
change, and in both the unscored scenarios are named.
Two more rules say ``dont_ship`` whatever the score, scored or not:

* ``agent_crashed``: a scenario where the agent's own code raised (status ``agent_error``) on 2 or more of its
  attempts, or on every attempt, whatever was graded — named under ``crashed_scenarios``.
* ``reproduced_high``: a required check (tier failing) of band high that failed on 2 or more scored conversations,
  in one scenario or across several — every such failure listed under ``blocking_high``.

Critical failures, ``agent_crashed`` and ``reproduced_high`` block Ship. Every failing-tier failure (the
reproduced high ones included) and undecided goals are listed under ``non_blocking``.
``non_blocking.replay_turn_only`` counts the undecided conversations that are replays of one recorded turn (the
attempt's own ``finalize_record.dialog_outcome`` says ``replay_cut``; :func:`plurai.trials.attempt_replay_mode`):
such a replay has no goal of its own. It is a SUBSET of ``goal_undecided``, which stays the total.
Every blocking and every released critical failure is listed, whatever the verdict.

THE ``scores`` BLOCK (``run_summary.json`` → ``scores``, schema :data:`SCORES_SCHEMA`)::

    {schema, weights: {critical: 8, high: 4, medium: 2, low: 1}, cutoff: 85.0,
     scenarios: {<id>: {score, result, fail_reasons, likelihood, regression,
                        conversations: [{trial, trial_dir, item, status, score, result, fail_reasons, passed_weight,
                                         graded_weight, counts: {pass, fail, not_applicable, ungraded}, unweighted,
                                         critical_failed, critical_released, failing_failed, failing_high,
                                         replay_mode: full|single_hop}]}},
     suite: {score, result, weighting: likelihood|equal|null, equal_because: {scenarios: [...]}|null, weights},
     agent: {verdict: ship|dont_ship|no_verdict, reasons, blocking_critical, blocking_high, released_critical,
             non_blocking: {failing_tier, goal_undecided, replay_turn_only}, unscored_scenarios, crashed_scenarios,
             best_case_score}}

``result`` is ``pass`` / ``fail`` / ``not_scored``; ``score`` is ``null`` when not scored. ``best_case_score`` is
``null`` unless some scenario is unscored and something was scored. A conversation's
``status`` is its ``verdicts.json`` status, or ``missing`` when the trial folder holds no readable file.
``critical_released[]`` rows are ``{eval, turn_idx, verdict, cause, quote, reason, source, seq}``;
``blocking_critical[]`` / ``blocking_high[]`` / ``failing_tier[]`` rows are ``{scenario, trial, eval}``;
``released_critical[]`` rows add the release row's fields. A conversation's ``failing_high`` names its failed
failing-tier evals of band high (a subset of ``failing_failed``).

CLI (for the plugin, which reads the block and never recomputes it)::

    plurai scores refresh --run-dir <run dir>

Recomputes ``scores`` from the run's ``verdicts.json`` files and writes it into ``<run dir>/run_summary.json``
atomically (every other key untouched but the pass count, :func:`align_totals`); prints the block as ONE JSON object on stdout. Call it after every
``plurai verdicts`` write (rescore, correction, import) so the block never goes stale. ``<run dir>`` is the
run's ``$OUT`` (trial 1; its summary lists the other trials). Exit 0 written; 66 (``INPUT_UNREADABLE``) when ``run_summary.json`` is missing,
unreadable or not an object (nothing written, the reason on stderr).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from plurai import exit_codes
from plurai import verdicts
from plurai.observability.scenario_verdict import VERDICT_KEY
from plurai.run_artifacts import RUN_SUMMARY_FILENAME, TRIAL_DIR_PREFIX
from plurai.trials import attempt_replay_mode, is_regression_scenario, listed_trial_dirs

#: THE weights, by the band the backend stamps on a cell. The only copy.
WEIGHTS: Final[Mapping[str, int]] = {'critical': 8, 'high': 4, 'medium': 2, 'low': 1}
#: THE cutoff, the same for a conversation, a scenario and the suite. The only copy.
CUTOFF: Final = 85.0

#: ``run_summary.json`` → ``scores``' schema tag.
SCORES_SCHEMA: Final = 'scores-v1'
#: The ``run_summary.json`` key the runner records each scenario's served ``path_probability`` under.
LIKELIHOODS_KEY: Final = 'scenario_likelihoods'

LOW_SCORE: Final = 'low_score'
CRITICAL_FAILURE: Final = 'critical_failure'
FAILING_TIER: Final = 'failing_tier'
GOAL_MISSED: Final = 'goal_missed'
CRASHED: Final = 'crashed'
#: Every fail reason, in the order they are recorded.
FAIL_REASONS: Final = (LOW_SCORE, CRITICAL_FAILURE, FAILING_TIER, GOAL_MISSED, CRASHED)
#: The reasons that fail a scenario whatever its average.
NON_SCORE_REASONS: Final = frozenset(reason for reason in FAIL_REASONS if reason != LOW_SCORE)

#: The statuses that are scored (``agent_error`` is scored 0); ``harness_error`` and ``incomplete`` are left out.
_SCORED_STATUSES: Final = frozenset({'scored', 'goal_undecided'})
_CRASHED_STATUS: Final = 'agent_error'

#: The red-cell audit ruling that releases a critical failure, and the typed causes it must carry:
#: the eval was wrong (false positive), or the test setup was (unmeasured).
RELEASE_VERDICT: Final = 'FALSE_RED'
FALSE_POSITIVE_CAUSES: Final = frozenset({'rubric', 'type', 'prompt', 'target_clause'})
TEST_SETUP_CAUSES: Final = frozenset({'scenario', 'harness', 'env', 'scan_fact'})
RELEASE_CAUSES: Final = FALSE_POSITIVE_CAUSES | TEST_SETUP_CAUSES

_CRITICAL: Final = 'critical'
_FAILING: Final = 'failing'
_HIGH: Final = 'high'
#: The agent verdict's own reasons beside the cutoff and critical ones (see AGENT).
AGENT_CRASHED: Final = 'agent_crashed'
REPRODUCED_HIGH: Final = 'reproduced_high'
#: How many attempts (crashes) or scored conversations (high failures) make a failure reproduced.
REPRODUCED: Final = 2
_MISSING_STATUS: Final = 'missing'


def _ordered(reasons: set[str]) -> list[str]:
    return [reason for reason in FAIL_REASONS if reason in reasons]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _result(score: float | None, reasons: list[str]) -> str:
    if score is None:
        return 'not_scored'
    return 'fail' if reasons else 'pass'


# ---------------------------------------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------------------------------------


def _is_critical(band: verdicts.Band) -> bool:
    return _CRITICAL in (band.severity, band.tier)


def _release_row(entry: Mapping[str, Any], *, source: str, seq: Any) -> dict[str, Any]:
    return {
        'eval': entry.get('eval'),
        'turn_idx': entry.get('turn_idx'),
        'verdict': entry.get('verdict'),
        'cause': entry.get('cause'),
        'quote': entry.get('quote'),
        'reason': entry.get('reason'),
        'source': source,
        'seq': seq,
    }


def _flips_out_of_fail(history: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per eval, the LATEST history row that moved it (or one of its turns) out of ``fail`` and still stands: a
    ``correction`` / ``local_check`` event, or a rescore's ``backend_corrections`` row. A later flip back into
    ``fail``, or a rescore that re-judged the eval without re-applying a correction, clears it."""
    flips: dict[str, dict[str, Any]] = {}
    for entry in history:
        source, seq = entry.get('source'), entry.get('seq')
        if source == 'rescore':
            corrected = {row.get('eval') for row in entry.get('backend_corrections') or () if isinstance(row, Mapping)}
            for name in entry.get('rejudged') or ():
                if name not in corrected:
                    flips.pop(name, None)
            for row in entry.get('backend_corrections') or ():
                if isinstance(row, Mapping) and (row.get('rescored') or {}).get('state') == 'fail' and row.get('corrected') != 'fail':
                    flips[str(row.get('eval'))] = _release_row(row, source='backend_correction', seq=seq)
        elif source in ('correction', 'local_check'):
            name = str(entry.get('eval'))
            old, new = (entry.get('old') or {}).get('state'), (entry.get('new') or {}).get('state')
            if old == 'fail' and new != 'fail':
                flips[name] = {**_release_row(entry, source=str(source), seq=seq), '_replaced_cell': entry.get('replaced_cell')}
            elif new == 'fail':
                flips.pop(name, None)
    return flips


def _releases(verdict: Any, cause: Any) -> bool:
    spelled = str(verdict or '').strip().upper().replace(' ', '_')
    return spelled == RELEASE_VERDICT and cause in RELEASE_CAUSES


def conversation_score(doc: Mapping[str, Any]) -> dict[str, Any]:
    """One attempt's score, result and every fail reason, from its ``verdicts.json`` document (:func:`plurai.verdicts.read`)."""
    status = doc['status']['value']
    evals: Mapping[str, Mapping[str, Any]] = doc.get('evals') or {}
    goal: Mapping[str, Any] = doc.get('goal') or {}
    attempt = doc.get('attempt') or {}
    out: dict[str, Any] = {'item': attempt.get('item'), 'status': status}
    if status not in _SCORED_STATUSES and status != _CRASHED_STATUS:
        return {**out, **_not_scored()}
    raw_owners = goal.get('target_judge_owners')
    owners = {str(name) for name in raw_owners} if isinstance(raw_owners, list) else set()
    counts = dict.fromkeys(verdicts.EVAL_STATES, 0)
    passed_weight = graded_weight = 0
    unweighted: list[str] = []
    critical_failed: list[str] = []
    failing_failed: list[str] = []
    failing_high: list[str] = []
    for name, entry in sorted(evals.items()):
        state = entry['state']
        counts[state] += 1
        if state not in ('pass', 'fail'):
            continue
        band = verdicts.stamped_band(entry)
        if band.severity not in WEIGHTS:
            unweighted.append(name)
            continue
        graded_weight += WEIGHTS[band.severity]
        if state == 'pass':
            passed_weight += WEIGHTS[band.severity]
        elif _is_critical(band):
            critical_failed.append(name)
        elif band.tier == _FAILING and name not in owners:
            failing_failed.append(name)
            if band.severity == _HIGH:
                failing_high.append(name)
    critical_released: list[dict[str, Any]] = []
    for name, row in sorted(_flips_out_of_fail(doc.get('history') or ()).items()):
        entry = evals.get(name)
        if entry is None or entry['state'] == 'fail':
            continue  # the eval left the file, or it is still red (counted above)
        if not _is_critical(verdicts.stamped_band(entry)) and not _is_critical(verdicts.stamped_band({'cell': row.get('_replaced_cell')})):
            continue
        public = {k: v for k, v in row.items() if k != '_replaced_cell'}
        if _releases(row.get('verdict'), row.get('cause')):
            critical_released.append(public)
        else:
            critical_failed.append(name)
    reasons: set[str] = set()
    if status == _CRASHED_STATUS:
        score: float | None = 0.0
        reasons.add(CRASHED)
    else:
        score = 100.0 * passed_weight / graded_weight if graded_weight else None
        if score is not None and score < CUTOFF:
            reasons.add(LOW_SCORE)
    if critical_failed:
        reasons.add(CRITICAL_FAILURE)
    if failing_failed:
        reasons.add(FAILING_TIER)
    if goal.get('goal_achieved') is False:
        reasons.add(GOAL_MISSED)
    fail_reasons = _ordered(reasons)
    return {
        **out,
        'score': score,
        'result': _result(score, fail_reasons),
        'fail_reasons': fail_reasons,
        'passed_weight': passed_weight,
        'graded_weight': graded_weight,
        'counts': counts,
        'unweighted': unweighted,
        'critical_failed': sorted(critical_failed),
        'critical_released': critical_released,
        'failing_failed': failing_failed,
        'failing_high': failing_high,
    }


def _not_scored() -> dict[str, Any]:
    return {
        'score': None,
        'result': 'not_scored',
        'fail_reasons': [],
        'passed_weight': 0,
        'graded_weight': 0,
        'counts': dict.fromkeys(verdicts.EVAL_STATES, 0),
        'unweighted': [],
        'critical_failed': [],
        'critical_released': [],
        'failing_failed': [],
        'failing_high': [],
    }


def payload_result(per: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`conversation_score` over a ``/finalize``-shaped payload (``policy_evaluation_results``, flat or under
    ``aggregated_results``) read as a scored dialog: its cells' states (:func:`plurai.verdicts.evals_of`, the one
    mapping) and its goal block. For a reader that holds the payload, not the attempt's file — the scenario-verdict
    reader. The attempt's status (a crash, a harness loss) is the file's; this never infers one."""
    per = per if isinstance(per, Mapping) else {}
    nested = per.get('aggregated_results')
    source = per if 'evaluation_results' in per or not isinstance(nested, Mapping) else nested
    goal = source.get(VERDICT_KEY, per.get(VERDICT_KEY))
    doc = {'status': {'value': 'scored'}, 'evals': verdicts.evals_of(source), 'goal': goal if isinstance(goal, Mapping) else None, 'history': []}
    return conversation_score(doc)


def missing_conversation() -> dict[str, Any]:
    """A trial folder with no readable ``verdicts.json``: the attempt measured nothing."""
    return {'item': None, 'status': _MISSING_STATUS, **_not_scored()}


# ---------------------------------------------------------------------------------------------------------
# Scenario, suite, agent
# ---------------------------------------------------------------------------------------------------------


def _likelihood(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _is_ungraded_crash(conversation: Mapping[str, Any]) -> bool:
    """A crash whose 0 stands on no weighted check: the agent raised before any eval it was judged on was graded."""
    return conversation['status'] == _CRASHED_STATUS and not conversation['graded_weight']


def scenario_score(scenario_id: str, conversations: Sequence[Mapping[str, Any]], *, likelihood: Any = None) -> dict[str, Any]:
    """A scenario over its trial attempts (``conversations``, in trial order; never a stability retry)."""
    scored = [c for c in conversations if c['score'] is not None]
    score = None if all(_is_ungraded_crash(c) for c in scored) else _mean([c['score'] for c in scored])
    reasons = {reason for c in conversations for reason in c['fail_reasons'] if reason in NON_SCORE_REASONS}
    if score is not None and score < CUTOFF:
        reasons.add(LOW_SCORE)
    fail_reasons = _ordered(reasons)
    return {
        'score': score,
        'result': _result(score, fail_reasons),
        'fail_reasons': fail_reasons,
        'likelihood': _likelihood(likelihood),
        'regression': is_regression_scenario(scenario_id),
        'conversations': [dict(c) for c in conversations],
    }


def suite_score(scenarios: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The suite over the scored scenarios: likelihood-weighted only when every one of them has a likelihood."""
    scored = {sid: s for sid, s in sorted(scenarios.items()) if s['score'] is not None}
    if not scored:
        return {'score': None, 'result': 'not_scored', 'weighting': None, 'equal_because': None, 'weights': {}}
    without = sorted(sid for sid, s in scored.items() if s['likelihood'] is None)
    raw = {sid: (1.0 if without else s['likelihood']) for sid, s in scored.items()}
    total = sum(raw.values())
    weights = {sid: w / total for sid, w in raw.items()}
    score = sum(scored[sid]['score'] * w for sid, w in weights.items())
    return {
        'score': score,
        'result': 'fail' if score < CUTOFF else 'pass',
        'weighting': 'equal' if without else 'likelihood',
        'equal_because': {'scenarios': without} if without else None,
        'weights': weights,
    }


def agent_verdict(suite: Mapping[str, Any], scenarios: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Ship / Don't ship / No verdict, with every reason and every critical failure, blocking or released."""
    blocking: list[dict[str, Any]] = []
    released: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    high: list[dict[str, Any]] = []
    crashed: list[str] = []
    undecided = replay_turn_only = 0
    for sid, scenario in sorted(scenarios.items()):
        crashes = sum(c['status'] == _CRASHED_STATUS for c in scenario['conversations'])
        if crashes >= REPRODUCED or (crashes and crashes == len(scenario['conversations'])):
            crashed.append(sid)
        for c in scenario['conversations']:
            where = {'scenario': sid, 'trial': c.get('trial')}
            blocking += [{**where, 'eval': name} for name in c['critical_failed']]
            released += [{**where, **row} for row in c['critical_released']]
            failing += [{**where, 'eval': name} for name in c['failing_failed']]
            high += [{**where, 'eval': name} for name in c.get('failing_high') or ()]
            undecided += c['status'] == 'goal_undecided'
            # A SUBSET of the undecided, never subtracted from it: a replay of one recorded turn has no goal of its own.
            replay_turn_only += c['status'] == 'goal_undecided' and c.get('replay_mode') == 'single_hop'
    unscored = sorted(sid for sid, s in scenarios.items() if s['score'] is None)
    reproduced = {name for name in (row['eval'] for row in high) if sum(row['eval'] == name for row in high) >= REPRODUCED}
    blocking_high = [row for row in high if row['eval'] in reproduced]
    reasons: list[str] = []
    best: float | None = None
    if suite['score'] is None:
        verdict, reasons = 'no_verdict', ['nothing_scored']
        if crashed:  # the agent's own raise is evidence: never "nothing scored, run again"
            verdict, reasons = 'dont_ship', [AGENT_CRASHED, 'unscored_scenarios']
    else:
        # The best case: the suite rule over the whole population, every unscored scenario at its ceiling of 100.
        best = suite_score({**scenarios, **{sid: {**scenarios[sid], 'score': 100.0} for sid in unscored}})['score'] if unscored else None
        if best is None and suite['score'] < CUTOFF:
            reasons.append('suite_below_cutoff')
        elif best is not None and best < CUTOFF:
            reasons.append('best_case_below_cutoff')
        if blocking:
            reasons.append(CRITICAL_FAILURE)
        if blocking_high:
            reasons.append(REPRODUCED_HIGH)
        if crashed:
            reasons.insert(0, AGENT_CRASHED)
        if reasons:
            verdict = 'dont_ship'
            reasons += ['unscored_scenarios'] if unscored else []
        else:  # an unscored scenario's worst case is a critical failure: it could still turn Ship into Don't ship
            verdict, reasons = ('no_verdict', ['unscored_scenarios']) if unscored else ('ship', [])
    return {
        'verdict': verdict,
        'reasons': reasons,
        'blocking_critical': blocking,
        'blocking_high': blocking_high,
        'released_critical': released,
        'non_blocking': {'failing_tier': failing, 'goal_undecided': undecided, 'replay_turn_only': replay_turn_only},
        'unscored_scenarios': unscored,
        'crashed_scenarios': crashed,
        'best_case_score': best,
    }


def scores_of(scenarios: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The whole ``scores`` block from per-scenario results (:func:`scenario_score`)."""
    suite = suite_score(scenarios)
    return {
        'schema': SCORES_SCHEMA,
        'weights': dict(WEIGHTS),
        'cutoff': CUTOFF,
        'scenarios': {sid: dict(s) for sid, s in sorted(scenarios.items())},
        'suite': suite,
        'agent': agent_verdict(suite, scenarios),
    }


# ---------------------------------------------------------------------------------------------------------
# The run dir
# ---------------------------------------------------------------------------------------------------------


def _trial_dirs(summary: Mapping[str, Any]) -> list[str]:
    """``''`` (trial 1, the run dir itself) and the ``trial_<k>`` folders the summary lists, in trial order."""
    listed = sorted(listed_trial_dirs(dict(summary)), key=lambda d: int(d.removeprefix(TRIAL_DIR_PREFIX)))
    return ['', *listed]


def run_scores(run_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """The ``scores`` block of the run at ``run_dir``, over the population ``summary`` names (the dict the runner is
    about to write, or the file on disk): each scenario's attempt in each listed trial folder, read through
    :func:`plurai.verdicts.read_or_none` (a folder without a readable file is a ``missing`` attempt)."""
    population = listed if isinstance(listed := summary.get('scenarios'), Mapping) else {}
    likelihoods = served if isinstance(served := summary.get(LIKELIHOODS_KEY), Mapping) else {}
    dirs = _trial_dirs(summary)
    scenarios: dict[str, dict[str, Any]] = {}
    for sid in sorted(str(s) for s in population):
        conversations = []
        for index, sub in enumerate(dirs, start=1):
            item = Path(run_dir) / sub / sid if sub else Path(run_dir) / sid
            doc = verdicts.read_or_none(item)
            conversation = conversation_score(doc) if doc is not None else missing_conversation()
            conversations.append({'trial': index, 'trial_dir': sub, **conversation, 'replay_mode': attempt_replay_mode(item)})
        scenarios[sid] = scenario_score(sid, conversations, likelihood=likelihoods.get(sid))
    return scores_of(scenarios)


def align_totals(summary: dict[str, Any]) -> dict[str, Any]:
    """``summary`` with its run totals' ``passed_valid`` (``totals``, and ``trials.run.totals`` when present) counted off
    its ``scores`` block: the valid trials whose conversation reads ``pass`` — the one pass rule, never the goal axis.
    The block wins only when it decided something (a conversation reads ``pass`` or ``fail``): with no block, or a block
    that scored no conversation (no weighted check graded), there is nothing to count off it, and the count
    :func:`plurai.trials.run_totals` made stands. Returns ``summary`` (mutated in place)."""
    scenarios = (summary.get('scores') or {}).get('scenarios') if isinstance(summary.get('scores'), Mapping) else None
    if not isinstance(scenarios, Mapping):
        return summary
    results = {
        (str(sid), c.get('trial')): c.get('result')
        for sid, s in scenarios.items()
        if isinstance(s, Mapping)
        for c in s.get('conversations') or ()
        if isinstance(c, Mapping)
    }
    if not any(result in ('pass', 'fail') for result in results.values()):
        return summary
    trials = summary.get('trials')
    run = trials.get('run') if isinstance(trials, Mapping) else None
    for totals in (summary.get('totals'), run.get('totals') if isinstance(run, Mapping) else None):
        if isinstance(totals, dict) and isinstance(totals.get('scenarios'), Mapping) and 'passed_valid' in totals:
            totals['passed_valid'] = sum(
                1
                for sid, entry in totals['scenarios'].items()
                if isinstance(entry, Mapping)
                for row in entry.get('trials') or ()
                if isinstance(row, Mapping) and row.get('validity') == 'valid' and results.get((str(sid), row.get('trial'))) == 'pass'
            )
    return summary


class ScoresError(Exception):
    """``run_summary.json`` could not be read: nothing was written."""


def refresh(run_dir: Path) -> dict[str, Any]:
    """Recompute ``scores`` for ``run_dir`` and write it into its ``run_summary.json`` (atomically, every other key kept)."""
    path = Path(run_dir) / RUN_SUMMARY_FILENAME
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ScoresError(f'{path} is missing or unreadable: {exc}') from exc
    if not isinstance(summary, dict):
        raise ScoresError(f'{path} is not a JSON object')
    scores = run_scores(Path(run_dir), summary)
    summary['scores'] = scores
    align_totals(summary)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.scores.tmp')
    tmp.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)
    return scores


def run_cli(args: argparse.Namespace) -> int:
    """``plurai scores refresh``: exit 0 written (the block on stdout), ``INPUT_UNREADABLE`` when ``run_summary.json`` is missing or unreadable."""
    try:
        scores = refresh(args.run_dir)
    except ScoresError as exc:
        print(f'plurai scores refresh: {exc}', file=sys.stderr)
        return exit_codes.INPUT_UNREADABLE
    print(json.dumps(scores, sort_keys=True))
    return exit_codes.OK


def add_parser(sub: argparse._SubParsersAction, *, path_type: Any = Path) -> None:
    """Register ``scores refresh`` on the ``plurai`` parser."""
    scores = sub.add_parser('scores', help="The run's conversation / scenario / suite scores and the agent's ship verdict (run_summary.json → scores).")
    verbs = scores.add_subparsers(dest='verb', required=True)
    refresh_verb = verbs.add_parser('refresh', help="Recompute run_summary.json → scores from the run's verdicts.json files and print it.")
    refresh_verb.add_argument('--run-dir', required=True, type=path_type, metavar='DIR', help="The run's $OUT (trial 1; its summary lists the trials).")
    refresh_verb.set_defaults(func=run_cli)


__all__ = [
    'CUTOFF',
    'FAIL_REASONS',
    'LIKELIHOODS_KEY',
    'RELEASE_CAUSES',
    'SCORES_SCHEMA',
    'WEIGHTS',
    'ScoresError',
    'add_parser',
    'agent_verdict',
    'align_totals',
    'conversation_score',
    'missing_conversation',
    'payload_result',
    'refresh',
    'run_scores',
    'scenario_score',
    'scores_of',
    'suite_score',
]
