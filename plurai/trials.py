"""Independent TRIALS of one run: the typed per-trial cells and the ONE aggregate over them.

``plurai-run-simulation --trials N`` runs the same scenario set N times, concurrently, as N
complete runs: trial 1 is ``$OUT`` itself and trial ``k`` is ``$OUT/trial_k/`` — each with its
own ``run_summary.json`` and scenario folders, exactly the shape a single run has. This module is
the one owner of what the N runs SAY TOGETHER, published under ``run_summary.json`` → ``trials``
(schema :data:`TRIALS_SCHEMA`). Readers copy that block; they never re-derive it.

TWO halves, kept apart so the arithmetic is testable without a disk:

* :func:`cells_from_run_dir` reads ONE trial's TYPED cells — each attempt's ``verdicts.json``
  (:mod:`plurai.verdicts`, its one writer: the cells, the scenario verdict the backend served beside them and
  the attempt's status) — never the report's decorated row strings, where a turn-scoped red and a pass
  can print the same name, and never a transcript.
* :func:`aggregate_trials` is pure: per scenario and per policy, how many valid trials passed,
  failed, were not applicable, were deferred to an owner policy or were inconclusive, the label the counts earn
  (:func:`cell_label`) and which trial each outcome was (``by_trial``); then the run-level
  headline — pass^N (a scenario counts only when it passed on EVERY valid trial, with at least
  two valid, out of the scenarios whose goal was measured), the per-trial pass rate with a
  Wilson 95% interval, and beside it how often the goal itself was reached (a decisive red fails
  a scenario whose goal was reached). The STRICT pair (``trial_goal_strict``,
  ``goal_all_trials_strict``) is the same headline over every drawn trial and every scenario with a
  goal to grade: an invalid trial stays in the denominator as not passed, its count typed by reason,
  so a report leads with it and never re-counts the invalid trials itself; N/A stays its own axis.

The wording a label licenses is "inconsistent across trials", never a diagnosis of "flaky agent":
each trial's judges sample too, and nothing here separates judge noise from agent variance. A
downstream renderer's own short label (e.g. "Flaky") is fine as long as it keeps that disclaimer
beside it — what this module forbids is ATTRIBUTING the inconsistency to the agent as settled fact.

REPLAY RATES beside the trials block, over the same typed cells. Each cell also carries its
policies' outcome AT THE CUT (turn 0 — the first live turn, which a resumed regression starts at)
and a per-turn tally. :func:`aggregate_regression_replays` counts, per recorded regression
scenario and per policy, how often the check caught the defect at the cut and anywhere in the
replay, with a Wilson interval on the failed rate and the clean replays a fix would need
(:func:`clean_replays_to_clear`); :func:`aggregate_policy_turn_rates` is the check's red rate per
judged turn (per conversation for a scenario-scoped check) over every scenario. Both read only the
scenario folder's own transcript: a ``stability_retry/`` second sample is never counted, and the
recorded reply that earned the folder is not a turn of the replay. :func:`record_time_rate` is the
same arithmetic over the replies sampled when the folder was recorded.

A LEAF: stdlib, ``scenario_verdict``'s reader, ``run_artifacts``' constants and :mod:`plurai.verdicts`' reader, so a
consumer can import it without the runner.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from plurai import exit_codes
from plurai._platform import utf8_stdio
from plurai._protocol import (
    AGENT_RECURSION_BUDGET_KIND,
    PROVIDER_INFRA_KIND,
    TRAIL_REFUSAL_KEY,
    TURN_BUDGET_EXHAUSTED_KIND,
    HarnessChildExited,
    is_harness_owned,
)
from plurai.gates.apply import make_finding
from plurai.gates.catalogue import DIALOG_PROVIDER_INFRA_INVALID, DIALOG_TURN_BUDGET_INVALID, DIALOG_USER_SIM_REPEAT_INVALID
from plurai.gates.contract import GateFinding
from plurai.gates.invocation import current as current_gate_authority
from plurai.host_effects import HOST_EFFECTS_KEY
from plurai.observability.scenario_verdict import GOAL_SOURCE_NONE, ScenarioVerdict, _unwrap, read_scenario_verdict
from plurai.observability.tool_call_recovery import AGENT_STATE_TOOL_TRAIL_KEY, tool_event_errored
from plurai.run_artifacts import (
    CUT_AT_ISSUE_TURN_CAUSE,
    REGRESSION_SCENARIO_PREFIX,
    REPLAY_CUT_REASON,
    RESUME_STATE_DROPPED_KEY,
    RUN_SUMMARY_FILENAME,
    SEED_RESTORE_KEY,
    TRIAL_DIR_PREFIX,
    TRIGGER_ABSENT_GUARD_KIND,
    TRIGGER_ABSENT_REASON,
)
from plurai.verdicts import CONVERTER_COMMAND, as_policy_results
from plurai.verdicts import read_or_none as read_verdicts_or_none

logger = logging.getLogger(__name__)

#: The ``trials`` block's schema tag. A reader that does not know it treats the block as absent.
TRIALS_SCHEMA: Final = 'trials-v1'

#: ``run_summary.json`` → ``regression_replays``' schema tag (:func:`aggregate_regression_replays`).
REGRESSION_REPLAYS_SCHEMA: Final = 'regression-replays-v1'

#: ``run_summary.json`` → ``policy_turn_rates``' schema tag (:func:`aggregate_policy_turn_rates`).
POLICY_TURN_RATES_SCHEMA: Final = 'policy-turn-rates-v1'

#: :func:`record_time_rate`'s schema tag, for the record the plugin copies into ``replays.json``.
RECORD_TIME_RATE_SCHEMA: Final = 'record-time-rate-v1'

#: The turn a regression is cut at, in its own replay's numbering: the seeded history is not a
#: turn, so turn 0 is the first LIVE turn for a resumed folder and for an opener-only one alike.
CUT_TURN: Final = 0

#: Which of a regression's two cells its CLEAR verdict reads (:func:`clear_axis`): the cut, for a replay that
#: resumed the recorded conversation and so met the issue turn as recorded; or the whole replay, for one that did
#: not resume — an opener-only or opener-rooted folder plays its conversation live from the recorded opener, so
#: the defect can land on any turn and only the whole replay says whether it recurred.
CLEAR_AT_CUT: Final = 'at_cut'
CLEAR_ANYWHERE: Final = 'anywhere'

#: The most clean replays :func:`clean_replays_to_clear` will ask for; beyond it a fix cannot be
#: shown at this cut and the answer is ``None`` (re-cut the regression instead).
MAX_CLEAN_REPLAYS: Final = 100

#: The two-sided Wilson z for the 95% interval every published range and retirement (:func:`clean_replays_to_clear`)
#: read, and for the 80% interval only the fixed / not-fixed DECISION reads (:func:`clean_replays_to_decide`).
Z95: Final = 1.959963984540054
Z80: Final = 1.2815515655446004

#: The fewest clean draws any "fixed" decision stands on: at 80% a single clean draw's upper bound (0 of 1: 0.622)
#: already sits below 3 of 3's lower bound, so without a floor one draw could decide.
DECISION_MIN_CLEAN: Final = 3

#: Where one trial's single cell landed. ``na`` is its own axis (the policy did not apply), and so is
#: ``deferred`` (the judge decided, and the backend handed its verdict to the named owner policies); neither
#: is folded into ``inconclusive`` (a judge that ran and did not decide, errored, or never reported).
Outcome = Literal['passed', 'failed', 'na', 'deferred', 'inconclusive']

#: What the counts of one cell across the valid trials earn — see :func:`cell_label`.
Label = Literal['consistent_pass', 'consistent_fail', 'inconsistent', 'insufficient', 'not_applicable', 'deferred']

#: Why a trial measured nothing for a scenario (``not_run``: the trial never produced it at all).
#: ``provider_infra`` / ``turn_budget``: the dialog LANDED on a cut it does not own — the agent's provider
#: failing, or the per-turn wall — so the turns it reached are not counted unless an overrule lifts
#: :func:`harness_cut_finding`. ``invalid_run``: the simulated user repeated itself, so the dialog is not
#: counted unless an overrule lifts :func:`repeat_invalid_finding`.
InvalidReason = Literal['errored', 'quarantined', 'harness_fault', 'host_dependency_missing', 'invalid_run', 'not_run', 'provider_infra', 'turn_budget']


#: What one trial of one scenario was: measured (``valid``; an agent that raised in its own turn is
#: valid and failed), or not, and then whose side could not run it.
Validity = Literal['valid', 'invalid_ours', 'invalid_agent']

#: Why an invalid trial measured nothing. ``agent_crash``: the agent's own process exited on its own —
#: the one ``invalid_agent`` cause. ``provider_infra``: the agent's model provider refused or failed the call
#: (a rate limit, an outage) — infrastructure, not the agent, so ``invalid_ours`` (not the agent's).
#: ``harness_fault``: our harness, mock or suite refused or broke it. ``ours_infra``: our backend, budgets or
#: load (an untyped error included — nothing typed it the agent's).
InvalidCause = Literal['ours_infra', 'harness_fault', 'provider_infra', 'agent_crash']

#: :data:`InvalidCause`, in print order.
INVALID_CAUSES: Final[tuple[InvalidCause, ...]] = ('ours_infra', 'harness_fault', 'provider_infra', 'agent_crash')

#: ``run_summary.json`` → ``totals``' schema tag (:func:`run_totals`).
RUN_TOTALS_SCHEMA: Final = 'run-totals-v1'

#: Per typed invalid reason, its cause when no error record says more (a landed cut, a voided dialog). TOTAL over
#: :data:`InvalidReason` (a test pins it): every reason a trial can carry has a cause.
_REASON_CAUSE: Final[Mapping[InvalidReason, InvalidCause]] = {
    'errored': 'ours_infra',
    'quarantined': 'harness_fault',
    'harness_fault': 'harness_fault',
    # The harness could not run the agent on this host (a missing host requirement).
    'host_dependency_missing': 'harness_fault',
    'invalid_run': 'harness_fault',
    'not_run': 'ours_infra',
    'provider_infra': 'provider_infra',
    'turn_budget': 'ours_infra',
}


def _error_cause(error: Mapping[str, Any]) -> InvalidCause:
    """The :data:`InvalidCause` of an ``error.json`` record, off its typed ``kind`` (the first attempt's when a
    re-drive's last attempt left none) and, for a lost child, its return code's sign — never its message."""
    kind = error.get('kind')
    if kind is None and isinstance(error.get('redrive'), dict):
        kind = error['redrive'].get('original_kind')
    if kind == PROVIDER_INFRA_KIND:
        return 'provider_infra'
    if kind == HarnessChildExited.kind:
        returncode = error.get('returncode')
        # A child that exited on its own is the agent's process; one reaped by a signal died under our load.
        return 'agent_crash' if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode > 0 else 'ours_infra'
    # A recursion crash (``GraphRecursionWrap``) with nothing read is an empty seed or an empty mock: ours.
    if kind == AGENT_RECURSION_BUDGET_KIND or is_harness_owned(kind):
        return 'harness_fault'
    return 'ours_infra'


def _tool_refusals(scenario_dir: Path) -> int:
    """How many of the scenario folder's recorded tool calls a gate REFUSED (typed ``refusal: {by, reason}``)."""
    doc = _read_json(scenario_dir / 'tool_events.json')
    events = doc.get('events') if isinstance(doc, dict) else None
    return sum(1 for e in events if isinstance(e, dict) and isinstance(e.get(TRAIL_REFUSAL_KEY), dict)) if isinstance(events, list) else 0


def _tool_errors(scenario_dir: Path) -> int:
    """How many of the scenario folder's recorded tool calls FAILED (:func:`tool_event_errored`, either lane)."""
    doc = _read_json(scenario_dir / 'tool_events.json')
    events = doc.get('events') if isinstance(doc, dict) else None
    if not isinstance(events, list):
        return 0
    return sum(1 for e in events if isinstance(e, dict) and e.get('type', 'tool_call') == 'tool_call' and tool_event_errored(e))


#: A run's own tokens in an error message (hex ids, counts) folded, so one fault reads the same in every trial.
_VOLATILE: Final = re.compile(r'0x[0-9a-fA-F]+|[0-9a-fA-F]{8,}|\d+')


def _repeat_signature(item_dir: Path) -> tuple[Any, ...] | None:
    """An attempt's loss as ``(exception, folded message, turns_reached)``, or ``None`` unless ``verdicts.json`` holds it
    a harness error of THIS ``error.json``'s thread with no typed kind, and the record types no ``transient`` or
    ``route_cause`` — a typed loss carries its own remedy."""
    judged, record = read_verdicts_or_none(item_dir), _read_json(item_dir / 'error.json')
    if judged is None or not isinstance(record, dict) or judged['status'].get('value') != 'harness_error':
        return None
    if record.get('thread_id') != judged['attempt'].get('thread_id') or (judged['status'].get('cause') or {}).get('kind') is not None:
        return None
    if 'transient' in record or record.get('route_cause') is not None:
        return None
    return (record.get('exception'), _VOLATILE.sub('#', str(record.get('error') or '')), record.get('turns_reached'))


def repeated_losses(attempts: Mapping[str, Sequence[Path]]) -> list[str]:
    """The scenarios (``{scenario: its attempt folder per trial}``) every one of at least two trials lost to one untyped
    fault (:func:`_repeat_signature`). Read off disk on every compose, never stamped on a record. Trials that ran side by
    side can share one outage, so a reader re-runs a named scenario once, apart in time, before saying a re-run reproduces it."""
    return [sid for sid, dirs in attempts.items() if len(dirs) >= 2 and len(sigs := {_repeat_signature(d) for d in dirs}) == 1 and None not in sigs]


#: Per ``ended_early`` reason a landed dialog is INVALID on: the trial's invalid reason. Its holding
#: finding is built by :func:`harness_cut_finding`, one ``make_finding`` per gate.
_HARNESS_CUTS: Mapping[str, InvalidReason] = {PROVIDER_INFRA_KIND: 'provider_infra', TURN_BUDGET_EXHAUSTED_KIND: 'turn_budget'}


def harness_cut_finding(reason: str | None, scenario_id: str, turn_failure: Mapping[str, Any]) -> GateFinding | None:
    """THE holding finding for a dialog that landed on a cut it does not own (:data:`_HARNESS_CUTS`), or
    ``None`` for any other reason.

    One builder for both of its readers — the runner that emits it and :func:`cells_from_summary`,
    which counts the cell only when an overrule given to this invocation names its fingerprint — so the
    two compute one fingerprint. Evidence is the failure's typed position and class, never its text.
    """
    evidence = {
        'turn_idx': turn_failure.get('turn_idx'),
        'closing': bool(turn_failure.get('closing')),
        'child_exc_class': turn_failure.get('child_exc_class'),
    }
    observed = {'error': turn_failure.get('error')}
    if reason == PROVIDER_INFRA_KIND:
        return make_finding(
            DIALOG_PROVIDER_INFRA_INVALID,
            subject=scenario_id,
            evidence=evidence,
            observed=observed,
            remedy='check the agent provider (quota, rate limit, status) and re-run; overrule only when the turns the dialog reached are worth counting',
            draft_ref=None,
        )
    if reason == TURN_BUDGET_EXHAUSTED_KIND:
        return make_finding(
            DIALOG_TURN_BUDGET_INVALID,
            subject=scenario_id,
            evidence=evidence,
            observed=observed,
            remedy='raise --agent-turn-timeout if a turn of this agent is legitimately that slow, then re-run; overrule only when the turns it reached count',
            draft_ref=None,
        )
    return None


def _harness_cut_invalid(reason: Any, scenario_id: str, turn_failure: Any) -> InvalidReason | None:
    """The trial's invalid reason for a landed harness cut, or ``None`` when the dialog is measured (no
    cut, or an overrule given to this invocation lifted the hold on this scenario's cell). The finding is
    recorded here too, once per fingerprint — a cell read back from an earlier run dir (a ``--scenario-id``
    re-run) was never emitted by this invocation's runner, so its fingerprint would otherwise be nowhere."""
    if not isinstance(reason, str) or reason not in _HARNESS_CUTS:
        return None
    finding = harness_cut_finding(reason, scenario_id, turn_failure if isinstance(turn_failure, dict) else {})
    if finding is not None:
        authority = current_gate_authority()
        if not any(f.fingerprint == finding.fingerprint for f in authority.findings):
            authority.emit(finding)
        if any((o.gate_id, o.fingerprint) == (finding.gate_id, finding.fingerprint) for o in authority.local_overrules):
            return None
    return _HARNESS_CUTS[reason]


def _basis_points(value: Any) -> int | None:
    """A recorded similarity or threshold as an integer in basis points — a fingerprint holds no float."""
    return round(float(value) * 10_000) if isinstance(value, int | float) and not isinstance(value, bool) else None


def repeat_invalid_finding(scenario_id: str, user_sim: Mapping[str, Any]) -> GateFinding | None:
    """THE holding finding for a dialog the simulated user invalidated by repeating itself (``run_valid: false``), or
    ``None`` for any other dialog.

    Evidence is each repeated pair the run recorded (``anomalies.<turn>.degenerate_repeat``): the turn, the kind, the
    similarity and the threshold it was judged against, in basis points — and who produced the repeat. Never the
    message text, so the fingerprint is stable across reads of one run.
    """
    if user_sim.get('run_valid') is not False:
        return None
    anomalies = user_sim.get('anomalies')
    pairs: list[dict[str, Any]] = []
    owners: set[str] = set()
    for turn, block in sorted(anomalies.items() if isinstance(anomalies, Mapping) else (), key=lambda item: (len(str(item[0])), str(item[0]))):
        record = block.get('degenerate_repeat') if isinstance(block, Mapping) else None
        if not isinstance(record, Mapping):
            continue
        owners.add(str(record.get('owner') or 'user_sim'))
        pairs.append(
            {
                'turn': str(turn),
                'kind': record.get('kind') if isinstance(record.get('kind'), str) else None,
                'similarity_bp': _basis_points(record.get('similarity')),
                'threshold_bp': _basis_points(record.get('threshold')),
            }
        )
    return make_finding(
        DIALOG_USER_SIM_REPEAT_INVALID,
        subject=scenario_id,
        evidence={'pairs': pairs, 'owners': sorted(owners)},
        observed={'invalid_reason': user_sim.get('invalid_reason')},
        remedy=(
            're-run the scenario; overrule only after reading the repeated turns and finding that the agent answered the conversation the scenario specified'
        ),
        draft_ref=None,
    )


def _repeat_invalid(scenario_id: str, user_sim: Any) -> InvalidReason | None:
    """``invalid_run`` for a dialog the simulated user invalidated, unless an overrule given to this invocation lifted
    its finding. The finding is recorded here — where the trial cell is decided — once per fingerprint."""
    finding = repeat_invalid_finding(scenario_id, user_sim) if isinstance(user_sim, Mapping) else None
    if finding is None:
        return None
    authority = current_gate_authority()
    if not any(f.fingerprint == finding.fingerprint for f in authority.findings):
        authority.emit(finding)
    if any((o.gate_id, o.fingerprint) == (finding.gate_id, finding.fingerprint) for o in authority.local_overrules):
        return None
    return 'invalid_run'


#: What one count of :func:`policy_turn_tally` is: a judged turn, or a whole conversation for a
#: check the backend scores once per scenario.
Unit = Literal['turn', 'conversation']

#: How a regression was replayed: whole conversations, or only its issue turn (``--replay-turn-only``),
#: read per attempt off the backend's ``dialog_outcome`` (:func:`replay_mode_of`).
ReplayMode = Literal['full', 'single_hop']

#: The backend's outcome for one reply sampled at a regression's cut (its ``SampledReplyOutcome``):
#: the owning check caught the defect again, did not, or could not grade the reply.
SampledOutcome = Literal['caught', 'missed', 'not_measured']

#: The goal labels that MEASURED the goal (two or more valid trials concluded it): the pass^N denominator.
_MEASURED_GOAL_LABELS: Final = frozenset({'consistent_pass', 'consistent_fail', 'inconsistent'})

#: The backend's typed ``inconclusive_reason`` values for a policy that does not apply — the same set as
#: its contract block ``not_applicable_reasons``. ``mixed_not_applicable`` (the judge's samples split on
#: whether the premise arose, and not-applicable won) is N/A there, so it is N/A here.
_NOT_APPLICABLE_REASONS: Final = frozenset({'not_applicable', 'mixed_not_applicable'})

#: The backend's ``verdict_axis.deferred_reasons``: matched EXACTLY, never inside a '+'-joined reason.
_DEFERRED_REASONS: Final = frozenset({'deferred'})

#: The outcome counts every cell and tally carries, in print order.
_OUTCOME_KEYS: Final[tuple[Outcome, ...]] = ('passed', 'failed', 'na', 'deferred', 'inconclusive')


def trial_dir_name(index: int) -> str:
    """Trial ``index``'s directory RELATIVE to ``$OUT`` — ``''`` for trial 1, which is ``$OUT``."""
    return '' if index == 1 else f'{TRIAL_DIR_PREFIX}{index}'


def listed_trial_dirs(summary: Any) -> frozenset[str]:
    """The ``trial_<k>`` directories a ``run_summary.json`` of schema :data:`TRIALS_SCHEMA` lists
    (``trials.dirs``, trial 1's ``''`` aside); ``frozenset()`` for any other summary."""
    block = summary.get('trials') if isinstance(summary, dict) else None
    dirs = block.get('dirs') if isinstance(block, dict) and block.get('schema') == TRIALS_SCHEMA else None
    if not isinstance(dirs, list):
        return frozenset()
    return frozenset(d for d in dirs if isinstance(d, str) and d.startswith(TRIAL_DIR_PREFIX) and d.removeprefix(TRIAL_DIR_PREFIX).isdigit())


def trial_run_name(run_name: str, index: int) -> str:
    """Trial ``index``'s own run / experiment name: each trial is a run of its own on every sink."""
    return f'{run_name}-t{index}'


@dataclass(frozen=True, slots=True)
class ScenarioCells:
    """One scenario in ONE trial: whether that trial measured it, and each typed cell it landed."""

    #: ``None`` when the trial measured the scenario; else the typed reason it did not.
    invalid_reason: InvalidReason | None
    thread_id: str | None
    goal: Outcome
    goal_achieved: Outcome
    policies: dict[str, Outcome] = field(default_factory=dict)
    #: The agent raised in its own turn: the goal failed and no policy was judged.
    agent_raised: bool = False
    #: Each policy's outcome at :data:`CUT_TURN` (:func:`policy_turn_outcome`), keyed like ``policies``.
    at_cut: dict[str, Outcome] = field(default_factory=dict)
    #: Each policy's per-turn counts on this trial (:func:`policy_turn_tally`).
    turn_tally: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: How this attempt replayed the scenario (:func:`replay_mode_of`); for an attempt whose agent raised (no
    #: verdict to read it off), the run's own ``replay_mode`` from its summary. ``None`` when neither says.
    replay_mode: ReplayMode | None = None
    #: The policies whose :attr:`at_cut` outcome is an abstention the backend TYPED as the single-hop cut
    #: (:func:`policy_cut_abstained`) — never a contested verdict, a guard skip or an error.
    cut_abstained: frozenset[str] = frozenset()
    #: An attempt that resumed with less than the recorded state (:func:`restore_reduced`), in EVERY mode: on a single
    #: hop every :attr:`at_cut` outcome is then ``inconclusive``; a whole replay keeps its outcomes and is caveated.
    restore_reduced: bool = False
    #: Whether this attempt resumed a recorded conversation, off the transcript's typed
    #: ``injected_inputs.resume_seed_present``; ``None`` when the transcript does not say.
    resumed: bool | None = None
    #: Whose side failed when :attr:`invalid_reason` is set (:func:`trial_validity`); ``None`` when valid.
    invalid_cause: InvalidCause | None = None
    #: The tool calls a gate refused on this attempt (:func:`_tool_refusals`).
    tool_refusals: int = 0
    #: The tool calls that failed on this attempt (:func:`_tool_errors`).
    tool_errors: int = 0
    #: The policies whose single-hop :attr:`at_cut` ``passed`` is a TYPED trigger-absent N/A (:func:`_single_hop_at_cut`):
    #: not caught, and no evidence the defect is gone — the premise never arose on the replayed turn.
    trigger_absent: frozenset[str] = frozenset()
    #: Per policy whose outcome is ``deferred``: the owner policies the backend named on its cell (``deferred_to``).
    deferred_to: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


#: The policy lists that decide a policy's outcome whatever cell it carries, applied in this order
#: (so a policy named in both reads as errored).
_NAMED_OUTCOMES: Final[tuple[tuple[str, Outcome], ...]] = (('skipped_policies', 'na'), ('errored_policies', 'inconclusive'))


def policy_outcomes(per: Mapping[str, Any]) -> dict[str, Outcome]:
    """Each policy's outcome on one attempt, off the backend's TYPED ``policy_evaluation_results``.

    Only a ``success`` cell is a verdict (the backend's own rule): a boolean ``violated`` then decides
    pass or fail, and a verdict without one is ``inconclusive``, never a pass. An ``inconclusive``
    cell whose reason is an N/A one (``not_applicable``, ``mixed_not_applicable``) is N/A, one whose
    reason is exactly ``deferred`` is ``deferred``; any other non-verdict (a judge that abstained, a
    weakly agreed cell, an errored one) is ``inconclusive``. A policy named in
    ``skipped_policies`` (its applicability guard or flow filter kept it from running) is N/A and
    one named in ``errored_policies`` is inconclusive, whatever cell it also carries — the runner's
    own graded set (``_graded_policies``) excludes both the same way.
    """
    outcomes: dict[str, Outcome] = {str(name): _verdict(cell) for name, cell in _evaluation_cells(per).items()}
    for key, outcome in _NAMED_OUTCOMES:
        names = per.get(key)
        for name in names if isinstance(names, list) else ():
            outcomes[str(name)] = outcome
    return outcomes


def _evaluation_cells(per: Mapping[str, Any]) -> Mapping[str, Any]:
    cells = per.get('evaluation_results')
    return cells if isinstance(cells, dict) else {}


def _verdict(cell: Any) -> Outcome:
    """One judged unit — a whole cell, or one entry of a turn-scoped cell's ``turns[]`` — read on
    the backend's rule: only ``success`` with a boolean ``violated`` decides, an N/A reason is N/A, the
    deferred reason is deferred, and anything else (a judge that abstained, a contested unit) is inconclusive."""
    if not isinstance(cell, dict):
        return 'inconclusive'
    if cell.get('status') == 'success':
        violated = cell.get('violated')
        return ('failed' if violated else 'passed') if isinstance(violated, bool) else 'inconclusive'
    reason = cell.get('inconclusive_reason')
    if reason in _NOT_APPLICABLE_REASONS:
        return 'na'
    return 'deferred' if reason in _DEFERRED_REASONS else 'inconclusive'


def _named_outcome(per: Mapping[str, Any], policy: str) -> Outcome | None:
    """The outcome ``skipped_policies`` / ``errored_policies`` impose on ``policy``, else ``None``."""
    for key, outcome in reversed(_NAMED_OUTCOMES):
        names = per.get(key)
        if isinstance(names, list) and policy in (str(name) for name in names):
            return outcome
    return None


def _is_turn_scoped(cell: Mapping[str, Any]) -> bool:
    """A cell the backend judged turn by turn: it says so, or it carries the per-turn list."""
    return cell.get('scope') == 'turn' or isinstance(cell.get('turns'), list)


def _int_turns(values: Any) -> set[int]:
    return {value for value in values if isinstance(value, int) and not isinstance(value, bool)} if isinstance(values, list) else set()


def policy_turn_outcome(per: Mapping[str, Any], policy: str, turn: int) -> Outcome:
    """``policy``'s outcome on ONE turn of one attempt, off the same typed cells as :func:`policy_outcomes`.

    A policy named in ``skipped_policies`` is N/A and one in ``errored_policies`` inconclusive, as
    there. A TURN-SCOPED cell is read at its ``turns[]`` entry for ``turn`` (the same verdict rule as a
    whole cell); a turn it lists in ``skipped_turn_idxs`` instead is N/A, and a turn it does not list
    at all is inconclusive. A SCENARIO-SCOPED cell has one verdict for the whole dialog, so it is that
    verdict — except a red, which is failed on ``turn`` only when a finding names that turn: findings
    that name only OTHER turns place the red elsewhere (passed here), and a red whose findings name no
    turn cannot be placed (inconclusive). A policy with no cell is inconclusive.
    """
    named = _named_outcome(per, policy)
    if named is not None:
        return named
    cell = _evaluation_cells(per).get(policy)
    if not isinstance(cell, dict):
        return 'inconclusive'
    if _is_turn_scoped(cell):
        turns = cell.get('turns')
        entry = next((e for e in turns if isinstance(e, dict) and e.get('turn_idx') == turn), None) if isinstance(turns, list) else None
        if entry is not None:
            return _verdict(entry)
        return 'na' if turn in _int_turns(cell.get('skipped_turn_idxs')) else 'inconclusive'
    outcome = _verdict(cell)
    if outcome != 'failed':
        return outcome
    findings = cell.get('findings')
    named_turns = _int_turns([f.get('turn_idx') for f in findings if isinstance(f, dict)] if isinstance(findings, list) else None)
    if turn in named_turns:
        return 'failed'
    return 'passed' if named_turns else 'inconclusive'


def replay_mode_of(per: Mapping[str, Any]) -> ReplayMode:
    """``single_hop`` when the backend's ``dialog_outcome`` says this attempt ended on the single-hop cut, else ``full``."""
    outcome = per.get('dialog_outcome')
    return 'single_hop' if isinstance(outcome, dict) and outcome.get('ended_early') == REPLAY_CUT_REASON else 'full'


def recorded_attempt(item_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(transcript.json, its finalize_record)`` of one attempt's folder — the ONE read of the conversation facts the
    backend recorded for it. ``({}, None)`` when the transcript is missing or unreadable; ``None`` record when absent."""
    transcript = _read_json(item_dir / 'transcript.json')
    if not isinstance(transcript, dict):
        transcript = {}
    record = transcript.get('finalize_record')
    return transcript, record if isinstance(record, dict) else None


def attempt_replay_mode(item_dir: Path) -> ReplayMode:
    """:func:`replay_mode_of` over one attempt's own recorded ``finalize_record`` (:func:`recorded_attempt`)."""
    return replay_mode_of(recorded_attempt(item_dir)[1] or {})


#: The ``agent_state`` keys a turn reports about ITSELF under the harness contract, rebuilt every turn and never
#: state the agent routes on: its tool trail, its :func:`plurai.confine` host-effects record, what its own code
#: added to its model input (:func:`plurai.model_input_context`; the backend reads it as the turn's input context),
#: the product text its own code appended to the reply (``appended_output``; the text itself stays in ``messages`` by
#: contract, the backend reads the tag per turn) and its run info (the plugin's harness templates). A regression pin
#: is the recorded turn's whole return, so it carries them, and a ``seed_state`` that skips only these restored the
#: agent's state (:func:`restore_reduced`).
TURN_REPORT_KEYS: Final = frozenset({AGENT_STATE_TOOL_TRAIL_KEY, HOST_EFFECTS_KEY, 'input_context', 'appended_output', 'run_info'})


def _drops_state(dropped: Any, seed: Any) -> bool:
    """Whether ``resume_state_dropped`` names state the agent met the cut without: any key outside
    :data:`TURN_REPORT_KEYS`, or a non-empty ``tool_trail`` beside seed messages that carry no tool exchange — the
    trail is then the only record of the last turn's tool results (a harness that folds it back into the history
    restores it; one that skips it does not). A report of another shape, or no seed to read, counts as state."""
    if not isinstance(dropped, list):
        return bool(dropped)
    if any(not isinstance(key, str) or key not in TURN_REPORT_KEYS for key in dropped):
        return True
    if AGENT_STATE_TOOL_TRAIL_KEY not in dropped:
        return False
    if not isinstance(seed, Mapping):
        return True
    state, messages = seed.get('agent_state'), seed.get('messages')
    if isinstance(state, Mapping) and not state.get(AGENT_STATE_TOOL_TRAIL_KEY):
        return False
    if not isinstance(messages, list):
        return True
    return not any(isinstance(m, Mapping) and ('tool' in (m.get('type'), m.get('role')) or m.get('tool_calls')) for m in messages)


def restore_reduced(transcript: Mapping[str, Any]) -> bool:
    """Whether a resumed attempt met its cut with LESS state than the recording had, off its ``resume_fidelity``.

    Reduced when the harness's ``seed_state`` reported any shortfall — a ``resume_state_dropped`` naming state
    (:func:`_drops_state`) or any other truthy key of its own — or the runner measured a ``partial`` restore. The
    runner's ``undetermined`` alone is not a loss (every agent that returns only its new messages reads that way),
    and neither is an empty report or a drop of the turn's own report keys (:data:`TURN_REPORT_KEYS`). A dropped
    state key is a skip for RECORDING (the conversation was restored whole), but it is a loss for a measurement AT
    THE CUT: the agent met the recorded user message without the state it had then (a pending question it was
    waiting on, say), so whatever it did there is no reading of the recorded defect. ``transcript`` needs only
    ``resume_fidelity`` and, for a dropped ``tool_trail``, the ``resume_seed`` it restored.
    """
    fidelity = transcript.get('resume_fidelity')
    if not isinstance(fidelity, Mapping):
        return bool(fidelity)
    seed = transcript.get('resume_seed')
    if any(_drops_state(value, seed) if key == RESUME_STATE_DROPPED_KEY else value for key, value in fidelity.items() if key != SEED_RESTORE_KEY):
        return True
    reading = fidelity.get(SEED_RESTORE_KEY)
    return isinstance(reading, Mapping) and reading.get('state') == 'partial'


def _unit_at(per: Mapping[str, Any], policy: str, turn: int) -> Any:
    """``policy``'s judged unit on ``turn``: a turn-scoped cell's ``turns[]`` entry, else the cell itself."""
    unit = _evaluation_cells(per).get(policy)
    if isinstance(unit, dict) and _is_turn_scoped(unit):
        turns = unit.get('turns')
        return next((e for e in turns if isinstance(e, dict) and e.get('turn_idx') == turn), None) if isinstance(turns, list) else None
    return unit


def policy_cut_abstained(per: Mapping[str, Any], policy: str, turn: int) -> bool:
    """Whether ``policy``'s outcome on ``turn`` is an abstention the backend TYPED as the single-hop cut:
    an inconclusive unit (the cell, or a turn-scoped cell's ``turns[]`` entry) whose ``not_measured_cause``
    is :data:`~plurai.run_artifacts.CUT_AT_ISSUE_TURN_CAUSE`. A policy ``skipped_policies`` /
    ``errored_policies`` names is never one."""
    if _named_outcome(per, policy) is not None:
        return False
    unit = _unit_at(per, policy, turn)
    return isinstance(unit, dict) and _verdict(unit) == 'inconclusive' and unit.get('not_measured_cause') == CUT_AT_ISSUE_TURN_CAUSE


def trigger_absent(per: Mapping[str, Any], policy: str, turn: int) -> bool:
    """Whether ``policy``'s N/A on ``turn`` is TYPED as "the trigger did not arise" — one of the two shapes
    the backend publishes (``replay_cut_vocabulary.trigger_absent``): every ``guard_suppressions`` record for the
    policy is a guard that read the state and returned False, or its unit's ``inconclusive_reason`` is exactly
    the judge's own not-applicable. ``mixed_not_applicable`` (contested applicability), a guard that raised or
    could not read its input, a judge that emitted no cell, a turn that was never graded and a skipped goal
    judge are NOT: each measured nothing."""
    records = per.get('guard_suppressions')
    kinds = {str(r.get('kind')) for r in records if isinstance(r, dict) and str(r.get('policy')) == policy} if isinstance(records, list) else set()
    if kinds == {TRIGGER_ABSENT_GUARD_KIND}:
        return True
    unit = _unit_at(per, policy, turn)
    return isinstance(unit, dict) and unit.get('inconclusive_reason') == TRIGGER_ABSENT_REASON


def _single_hop_at_cut(per: Mapping[str, Any], at_cut: dict[str, Outcome]) -> dict[str, Outcome]:
    """A single-hop replay's outcomes at the cut, with a TYPED trigger-absent N/A read as NOT CAUGHT (``passed``).

    The replay's input is the recorded one, byte for byte: when the check's premise demonstrably did not arise
    on the replayed turn (:func:`trigger_absent`), the defect did not recur — a clean replay. Every other N/A
    stays N/A: nothing was measured, so it is never a clean replay.
    """
    return {name: 'passed' if outcome == 'na' and trigger_absent(per, name, CUT_TURN) else outcome for name, outcome in at_cut.items()}


def policy_turn_tally(per: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per policy with a cell on this attempt: its outcome counts, one per judged UNIT.

    A turn-scoped cell counts each turn it lists (in ``turns[]`` or ``skipped_turn_idxs``), each
    read by :func:`policy_turn_outcome`; a scenario-scoped cell counts its one conversation at its
    :func:`policy_outcomes` outcome. A policy named in ``skipped_policies`` / ``errored_policies``
    WITHOUT a cell has no unit to count and is absent here — it never touches a failed rate.
    """
    scenario = policy_outcomes(per)
    tally: dict[str, dict[str, Any]] = {}
    for raw_name, cell in _evaluation_cells(per).items():
        name = str(raw_name)
        row: dict[str, Any] = {'unit': 'conversation', **dict.fromkeys(_OUTCOME_KEYS, 0)}
        if isinstance(cell, dict) and _is_turn_scoped(cell):
            row['unit'] = 'turn'
            turns = cell.get('turns')
            entries = turns if isinstance(turns, list) else []
            listed = _int_turns([e.get('turn_idx') for e in entries if isinstance(e, dict)]) | _int_turns(cell.get('skipped_turn_idxs'))
            for turn in sorted(listed):
                row[policy_turn_outcome(per, name, turn)] += 1
        else:
            row[scenario[name]] += 1
        tally[name] = row
    return tally


def _unconcluded(verdict: ScenarioVerdict) -> Outcome:
    """A verdict that concluded nothing: N/A when the eval set declared no goal owner, else inconclusive."""
    return 'na' if verdict.goal_source == GOAL_SOURCE_NONE else 'inconclusive'


def goal_outcome(per: Mapping[str, Any]) -> Outcome:
    """The scenario goal on one attempt, read through the ONE scenario-verdict reader.

    ``passed`` / ``failed`` are the reader's conclusive statuses — ``goal_achieved`` plus the backend's red lists and THE
    conversation rule (:mod:`plurai.scoring`), so a decisive (catastrophic) or failing-tier red, or a low score, fails it
    even beside an achieved goal. Otherwise a scenario whose
    eval set declared no goal owner has no goal to reach, so it is N/A; every other status (nothing
    concluded, an unclean measurement, no readable block at all) is inconclusive.
    """
    verdict = read_scenario_verdict(dict(per))
    if verdict.status in ('passed', 'failed'):
        return verdict.status
    return _unconcluded(verdict)


def goal_achieved_outcome(per: Mapping[str, Any]) -> Outcome:
    """Whether the goal was REACHED on one attempt — the verdict's ``goal_achieved`` axis, which a
    decisive red does not move (:func:`goal_outcome` is the scenario verdict, which it does). An
    unclean measurement reaches no goal on either axis — unless the backend typed every breach as
    sparing the goal (:func:`_breaches_spare_the_goal`), when this axis reads the goal as measured."""
    verdict = read_scenario_verdict(dict(per))
    if verdict.status == 'unclean' and not _breaches_spare_the_goal(per):
        return 'inconclusive'
    if verdict.goal_achieved is not None:
        return 'passed' if verdict.goal_achieved else 'failed'
    return _unconcluded(verdict)


def _breaches_spare_the_goal(per: Mapping[str, Any]) -> bool:
    """Whether the backend typed EVERY integrity breach as sparing the goal: ``measurement_integrity``
    readable, ``clean`` False, ``breaches`` a non-empty list, and each breach ``voids_goal is False``
    (the backend decides it against the policy set the run was scored with). Anything
    else — an older backend with no flag, one breach without it, a ``True`` — is ``False``: the
    measurement reaches no goal. The SDK applies no local rule of its own."""
    block = _unwrap(dict(per), 'measurement_integrity')
    if not isinstance(block, dict) or block.get('clean') is not False:
        return False
    breaches = block.get('breaches')
    return isinstance(breaches, list) and bool(breaches) and all(isinstance(b, dict) and b.get('voids_goal') is False for b in breaches)


def cells_from_run_dir(run_dir: Path) -> dict[str, ScenarioCells]:
    """ONE trial's typed cells, per scenario its own ``run_summary.json`` lists.

    The summary names the population (so a scenario that errored in this trial is still counted,
    as invalid); each scenario's ``verdicts.json`` supplies the cells AND the status, computed once by
    its writer (:func:`plurai.verdicts.status_of_error` / :func:`plurai.verdicts.status_of_landed`) — this
    reader never re-classifies an ``error.json`` or an ``ended_early``. A scenario is INVALID in this
    trial — it measured nothing — when it has no ``verdicts.json`` or its status is ``harness_error``
    (``harness_fault`` when the harness failed it, ``quarantined`` when the suite refused it, else ``errored``), or when the simulated user invalidated
    the dialog (``invalid_run``, held by :func:`repeat_invalid_finding` until an overrule counts it). An
    ``agent_error`` status — the agent raised in its own turn, or hit its recursion limit after reading
    real data, whether the attempt then failed or the dialog landed — is a measured, FAILED trial: its
    goal failed, or is N/A when the trial's measured scenarios say the eval set declared no goal owner.
    ``{}`` when the trial wrote no readable summary.

    Reads the summary off disk and hands it to :func:`cells_from_summary`.
    """
    return cells_from_summary(run_dir, _read_json(run_dir / RUN_SUMMARY_FILENAME))


def cells_from_summary(run_dir: Path, summary: Any) -> dict[str, ScenarioCells]:
    """:func:`cells_from_run_dir` over a summary already in hand — the dict the single-run writer is
    about to write, so the population is the one that file will name. Only ``scenarios``,
    ``scenario_verdicts``, ``replay_mode`` and ``single_hop_scenarios`` are read off it; every cell comes from ``run_dir``'s scenario folders.
    ``{}`` when ``summary`` lists no scenarios."""
    if not isinstance(summary, dict) or not isinstance(summary.get('scenarios'), dict):
        return {}
    verdicts: dict[str, Any] = raw if isinstance(raw := summary.get('scenario_verdicts'), dict) else {}
    cells: dict[str, ScenarioCells] = {}
    raised: dict[str, Any] = {}
    goal_sources: set[str | None] = set()
    # The run's own mode (``run_summary.json`` → ``replay_mode``), for an attempt with no verdict to read it off.
    declared_mode = summary.get('replay_mode')
    run_mode: ReplayMode | None = declared_mode if declared_mode in ('full', 'single_hop') else None
    # The scenarios a replay plan played single-hop beside the others (``run_summary.json`` → ``single_hop_scenarios``).
    hops = {str(sid) for sid in raw_hops} if isinstance(raw_hops := summary.get('single_hop_scenarios'), list) else set()
    for sid, row in summary['scenarios'].items():
        # THE verdicts and THE status of this attempt (``plurai.verdicts``, its one writer); the transcript is read
        # only for the conversation facts (``finalize_record``, ``metadata``, ``injected_inputs``).
        judged = read_verdicts_or_none(run_dir / sid)
        if judged is None and any((run_dir / sid / name).is_file() for name in ('transcript.json', 'error.json')):
            # A folder with a transcript or an error record and no verdicts.json is counted as measuring nothing, never
            # read off its transcript — and said, because a re-run into such a run dir would otherwise lose its rows in silence.
            logger.warning('%s holds an attempt with no verdicts.json; it counts as measuring nothing. Convert the run: %s', run_dir / sid, CONVERTER_COMMAND)
        status: dict[str, Any] = judged['status'] if judged is not None else {}
        cause: dict[str, Any] = status.get('cause') or {}
        if status.get('value') == 'agent_error':
            # The agent crashed — raised in its own turn, or hit its recursion limit after reading real data — in
            # an attempt that failed or in a landed dialog: a FAILED trial, never an invalid one (its goal is set below).
            raised[sid] = judged['attempt']['thread_id'] if judged is not None else None
            continue
        refusals = _tool_refusals(run_dir / sid)
        if judged is None or status.get('value') == 'harness_error':
            unmeasured: InvalidReason = 'harness_fault' if verdicts.get(sid) == 'not_valid' else 'quarantined' if row == 'incomplete' else 'errored'
            cells[sid] = ScenarioCells(
                invalid_reason=unmeasured,
                thread_id=None,
                goal='inconclusive',
                goal_achieved='inconclusive',
                # The status's typed cause says whose side it was; without a file, the typed reason does.
                invalid_cause=_error_cause(cause) if judged is not None else _REASON_CAUSE[unmeasured],
                tool_refusals=refusals,
            )
            continue
        transcript, record = recorded_attempt(run_dir / sid)
        per = as_policy_results(judged, record=record)
        metadata: dict[str, Any] = raw if isinstance(raw := transcript.get('metadata'), dict) else {}
        reason = _repeat_invalid(sid, per.get('user_sim_integrity'))
        if reason is None:
            reason = _harness_cut_invalid(metadata.get('ended_early'), sid, metadata.get('turn_failure'))
        thread_id = metadata.get('thread_id')
        goal_sources.add(read_scenario_verdict(dict(per)).goal_source)
        policies = policy_outcomes(per)
        mode = replay_mode_of(per)
        at_cut: dict[str, Outcome] = {name: policy_turn_outcome(per, name, CUT_TURN) for name in policies}
        # Computed in every mode. A single hop that dropped resumed state measured nothing at its cut: no outcome there is
        # "not caught". A whole replay keeps its outcomes, caveated by the flag (the runner warns on it).
        reduced = restore_reduced(transcript)
        injected = transcript.get('injected_inputs')
        seeded = injected.get('resume_seed_present') if isinstance(injected, dict) else None
        absent: frozenset[str] = frozenset()
        if mode == 'single_hop':
            not_measured: Outcome = 'inconclusive'
            absent = frozenset() if reduced else frozenset(name for name, outcome in at_cut.items() if outcome == 'na' and trigger_absent(per, name, CUT_TURN))
            at_cut = dict.fromkeys(at_cut, not_measured) if reduced else _single_hop_at_cut(per, at_cut)
        cells[sid] = ScenarioCells(
            invalid_reason=reason,
            thread_id=thread_id if isinstance(thread_id, str) else None,
            goal=goal_outcome(per),
            goal_achieved=goal_achieved_outcome(per),
            policies=policies,
            at_cut=at_cut,
            turn_tally=policy_turn_tally(per),
            replay_mode=mode,
            cut_abstained=frozenset(name for name in policies if policy_cut_abstained(per, name, CUT_TURN)),
            restore_reduced=reduced,
            resumed=seeded if isinstance(seeded, bool) else None,
            invalid_cause=_REASON_CAUSE[reason] if reason is not None else None,
            tool_refusals=refusals,
            tool_errors=_tool_errors(run_dir / sid),
            trigger_absent=absent,
            deferred_to=_deferred_owners(per, policies),
        )
    # One eval set per trial: its goal owner is read off the scenarios this trial measured.
    raised_goal: Outcome = 'na' if goal_sources == {GOAL_SOURCE_NONE} else 'failed'
    for sid, thread_id in raised.items():
        cells[sid] = ScenarioCells(
            invalid_reason=None,
            thread_id=thread_id,
            goal=raised_goal,
            goal_achieved=raised_goal,
            agent_raised=True,
            replay_mode='single_hop' if sid in hops else run_mode,
            tool_refusals=_tool_refusals(run_dir / sid),
        )
    return {sid: cells[sid] for sid in summary['scenarios'] if sid in cells}


def _deferred_owners(per: Mapping[str, Any], policies: Mapping[str, Outcome]) -> dict[str, tuple[str, ...]]:
    """Per policy whose outcome is ``deferred``, the owners its cell names in ``deferred_to``."""
    cells = _evaluation_cells(per)
    owners: dict[str, tuple[str, ...]] = {}
    for name, outcome in policies.items():
        cell = cells.get(name)
        named = cell.get('deferred_to') if outcome == 'deferred' and isinstance(cell, dict) else None
        if isinstance(named, list):
            owners[name] = tuple(str(owner) for owner in named)
    return owners


def cell_label(passed: int, failed: int, na: int, inconclusive: int, *, deferred: int = 0) -> Label:
    """The label one cell's counts over the VALID trials earn (measured = passed + failed):

    1. nothing measured, nothing inconclusive and some deferred → ``deferred``;
    2. nothing measured, some N/A and nothing inconclusive or deferred → ``not_applicable``;
    3. fewer than two measured → ``insufficient``;
    4. no fail → ``consistent_pass``; 5. no pass → ``consistent_fail``; 6. else ``inconsistent``.
    """
    measured = passed + failed
    if measured == 0 and inconclusive == 0 and deferred > 0:
        return 'deferred'
    if measured == 0 and na > 0 and inconclusive == 0:
        return 'not_applicable'
    if measured < 2:
        return 'insufficient'
    if failed == 0:
        return 'consistent_pass'
    if passed == 0:
        return 'consistent_fail'
    return 'inconsistent'


def _counts(by_trial: Sequence[Outcome | None]) -> dict[str, Any]:
    """One cell: the counts over its VALID trials, their label, and ``by_trial`` — the outcome on each
    REQUESTED trial in order, ``None`` where that trial was invalid (so it counts nowhere)."""
    tally = dict.fromkeys(_OUTCOME_KEYS, 0)
    for outcome in by_trial:
        if outcome is not None:
            tally[outcome] += 1
    label = cell_label(tally['passed'], tally['failed'], tally['na'], tally['inconclusive'], deferred=tally['deferred'])
    return {**tally, 'label': label, 'by_trial': list(by_trial)}


def wilson(passed: int, of: int, z: float) -> list[float] | None:
    """The two-sided Wilson score interval at ``z`` for ``passed`` of ``of``, rounded to 3 places; ``None`` at 0."""
    if of <= 0:
        return None
    p = passed / of
    denom = 1 + z * z / of
    centre = p + z * z / (2 * of)
    half = z * math.sqrt(p * (1 - p) / of + z * z / (4 * of * of))
    return [round(max(0.0, (centre - half) / denom), 3), round(min(1.0, (centre + half) / denom), 3)]


def wilson95(passed: int, of: int) -> list[float] | None:
    """The Wilson score 95% interval for ``passed`` of ``of``, rounded to 3 places; ``None`` at 0."""
    return wilson(passed, of, Z95)


def wilson80(passed: int, of: int) -> list[float] | None:
    """The Wilson score 80% interval — the fixed / not-fixed decision's level, never a published range."""
    return wilson(passed, of, Z80)


#: Why ``trial_goal_strict`` publishes no rate. ``no_trial_drawn``: the run drew no trial at all;
#: ``no_goal_to_grade``: every drawn trial's goal was N/A (the eval set declared no goal owner), so
#: there is nothing to rate; ``no_trial_decided``: some goal was gradable, but no valid trial reached a
#: verdict on it (every such trial was invalid or inconclusive).
NoRateReason = Literal['no_trial_drawn', 'no_goal_to_grade', 'no_trial_decided']


def _strict_goal(scenarios: Mapping[str, Mapping[str, Any]], n: int) -> dict[str, Any]:
    """The goal pass rate over EVERY drawn trial with a goal to grade: ``of`` is ``n`` × scenarios minus the
    ``na`` trials — N/A is its own axis (the backend said there is no goal to grade), reported beside the rate
    and never counted against it (``scenario_verdict``'s rule: no ``0 of 7`` for a goal-less eval set).
    What did not pass is broken down by TYPE — the valid trials' ``failed`` / ``inconclusive`` goals and
    the invalid trials per :data:`InvalidReason` — so ``passed`` + the parts == ``of`` and a reader copies
    each count instead of re-counting rows.

    ``invalid['host_dependency_missing']`` counts EVERY trial ruled so, including those of a scenario no
    trial of which was valid (``unmeasured_host_dependency``) — unlike ``host_dependency_excluded_trials``,
    which names only the scenarios that still have a valid trial.

    No rate (``rate``/``wilson95`` ``None``, ``no_rate_reason`` typed) when no valid trial decided a goal:
    a rate over zero verdicts is not a measurement."""
    invalid: dict[str, int] = {}
    for row in scenarios.values():
        for bad in row['invalid']:
            invalid[bad['reason']] = invalid.get(bad['reason'], 0) + 1
    goal = {key: sum(row['goal'][key] for row in scenarios.values()) for key in ('passed', 'failed', 'inconclusive', 'na')}
    of = n * len(scenarios) - goal['na']
    decided = goal['passed'] + goal['failed']
    reason: NoRateReason | None = None if decided else 'no_trial_drawn' if n * len(scenarios) == 0 else 'no_goal_to_grade' if of == 0 else 'no_trial_decided'
    return {
        'passed': goal['passed'],
        'of': of,
        'rate': round(goal['passed'] / of, 3) if reason is None else None,
        'wilson95': wilson95(goal['passed'], of) if reason is None else None,
        'no_rate_reason': reason,
        'na': goal['na'],
        'not_passed': {'failed': goal['failed'], 'inconclusive': goal['inconclusive'], 'invalid': dict(sorted(invalid.items()))},
    }


def _strict_all_trials(scenarios: Mapping[str, Mapping[str, Any]], n: int) -> dict[str, Any]:
    """pass^N over every scenario with a goal to grade: a scenario counts only when its goal passed on EVERY
    drawn trial (an invalid trial is not a pass). A scenario whose every drawn trial was N/A is out of ``of``
    and counted beside it as ``na``."""
    goal_less = sum(1 for row in scenarios.values() if n and row['goal']['na'] == n)
    return {'passed': sum(1 for row in scenarios.values() if n and row['goal']['passed'] == n), 'of': len(scenarios) - goal_less, 'na': goal_less}


def _invalid_trials(per_trial: Sequence[ScenarioCells | None]) -> list[dict[str, Any]]:
    """The trials that measured nothing for one scenario, each with its typed reason (``not_run``:
    the trial never produced the scenario)."""
    return [
        {'trial': index, 'reason': cells.invalid_reason if cells is not None else 'not_run'}
        for index, cells in enumerate(per_trial, start=1)
        if cells is None or cells.invalid_reason is not None
    ]


def trial_validity(cells: ScenarioCells | None) -> tuple[Validity, InvalidCause | None]:
    """One trial's :data:`Validity` and, when invalid, its :data:`InvalidCause` (``None`` cells: the trial
    never produced the scenario — ours)."""
    if cells is None:
        return 'invalid_ours', 'ours_infra'
    if cells.invalid_reason is None:
        return 'valid', None
    cause = cells.invalid_cause or _REASON_CAUSE[cells.invalid_reason]
    return ('invalid_agent' if cause == 'agent_crash' else 'invalid_ours'), cause


def run_totals(cells_by_trial: Sequence[Mapping[str, ScenarioCells]]) -> dict[str, Any]:
    """THE run's numbers: over every drawn trial of every scenario (one mapping per requested trial, in
    order), how many were valid and how many were not, by cause; how many valid trials passed their goal
    (``passed_valid`` over ``valid_trials`` is the headline); the tool calls a gate refused; and per scenario each
    trial's validity. Pure. Every surface copies these counts; none recounts them."""
    by_cause: dict[str, int] = dict.fromkeys(INVALID_CAUSES, 0)
    valid = passed = refusals = 0
    scenarios: dict[str, Any] = {}
    for sid in dict.fromkeys(sid for trial in cells_by_trial for sid in trial):
        rows: list[dict[str, Any]] = []
        for index, trial in enumerate(cells_by_trial, start=1):
            cells = trial.get(sid)
            validity, cause = trial_validity(cells)
            refused = cells.tool_refusals if cells is not None else 0
            refusals += refused
            if cause is None:
                valid += 1
                passed += int(cells is not None and cells.goal == 'passed')
            else:
                by_cause[cause] += 1
            rows.append({'trial': index, 'validity': validity, 'cause': cause, 'tool_refusals': refused})
        scenarios[sid] = {'trials': rows}
    return {
        'schema': RUN_TOTALS_SCHEMA,
        'valid_trials': valid,
        'invalid_trials': sum(by_cause.values()),
        'invalid_by_cause': by_cause,
        'passed_valid': passed,
        'tool_refusals': refusals,
        'scenarios': scenarios,
    }


def aggregate_trials(
    cells_by_trial: Sequence[Mapping[str, ScenarioCells]], *, run_name: str, measurement: Sequence[tuple[bool | None, Sequence[str]]]
) -> dict[str, Any]:
    """The ``trials`` block over ``cells_by_trial`` — one mapping per REQUESTED trial, in order
    (an empty mapping for a trial that produced nothing) — and ``measurement``, each trial's own
    ``(measurement_clean, measurement_unclean_because)`` in the same order. Pure: no I/O, no clock.

    Every count is over the scenario's VALID trials; an invalid trial is listed with its typed
    reason and counts nowhere else (its ``by_trial`` entry is ``None``) — except the two STRICT
    run-level figures, ``trial_goal_strict`` and ``goal_all_trials_strict``, which keep every drawn
    trial with a goal to grade in the denominator and count an invalid one as not passed, by type. A policy a valid trial did
    not report at all is counted ``inconclusive`` there (no verdict), never N/A (N/A is a statement
    the backend makes).
    """
    n = len(cells_by_trial)
    order = list(dict.fromkeys(sid for trial in cells_by_trial for sid in trial))
    scenarios: dict[str, Any] = {}
    for sid in order:
        per_trial = [trial.get(sid) for trial in cells_by_trial]
        valid_or_none = [cells if cells is not None and cells.invalid_reason is None else None for cells in per_trial]
        valid = [cells for cells in valid_or_none if cells is not None]
        invalid = _invalid_trials(per_trial)
        names = {name for cells in valid for name in cells.policies}
        policies: dict[str, Any] = {}
        for name in sorted(names):
            row = _counts([cells.policies.get(name, 'inconclusive') if cells is not None else None for cells in valid_or_none])
            owners = sorted({owner for cells in valid if cells.policies.get(name) == 'deferred' for owner in cells.deferred_to.get(name, ())})
            policies[name] = {**row, **({'deferred_to': owners} if owners else {})}
        scenarios[sid] = {
            'valid': len(valid),
            'invalid': invalid,
            'agent_raised': [index for index, cells in enumerate(per_trial, start=1) if cells is not None and cells.agent_raised],
            'thread_ids': [cells.thread_id if cells is not None else None for cells in per_trial],
            'goal': _counts([cells.goal if cells is not None else None for cells in valid_or_none]),
            'goal_achieved': _counts([cells.goal_achieved if cells is not None else None for cells in valid_or_none]),
            'policies': policies,
        }
    goal_measured_scenarios = sum(1 for row in scenarios.values() if row['goal']['label'] in _MEASURED_GOAL_LABELS)
    goal_passed = sum(row['goal']['passed'] for row in scenarios.values())
    goal_measured = sum(row['goal']['passed'] + row['goal']['failed'] for row in scenarios.values())
    return {
        'schema': TRIALS_SCHEMA,
        'n': n,
        'dirs': [trial_dir_name(index) for index in range(1, n + 1)],
        'run_names': [trial_run_name(run_name, index) for index in range(1, n + 1)],
        'measurement': [
            {'trial': index, 'clean': clean if isinstance(clean, bool) else None, 'because': [str(why) for why in because]}
            for index, (clean, because) in enumerate(measurement, start=1)
        ],
        'scenarios': scenarios,
        'run': {
            'scenarios': len(scenarios),
            # pass^N: the goal passed on EVERY valid trial, and at least two trials were valid — over the
            # scenarios whose goal was MEASURED; an insufficient or N/A goal is counted beside them.
            'goal_all_trials': {
                'passed': sum(1 for row in scenarios.values() if row['valid'] >= 2 and row['goal']['passed'] == row['valid']),
                'of': goal_measured_scenarios,
                'unmeasured': len(scenarios) - goal_measured_scenarios,
            },
            # STRICT, the headline a report prints FIRST: pass^N over every scenario and the pass rate over
            # every drawn trial with a goal to grade, where a trial that measured nothing counts as not
            # passed (never dropped); N/A trials are counted beside it, not against it.
            'goal_all_trials_strict': _strict_all_trials(scenarios, n),
            'trial_goal_strict': _strict_goal(scenarios, n),
            # Among the trials that DECIDED the goal (passed + failed): the rate above with the invalid, the
            # inconclusive and the N/A trials out of the denominator. Not "the valid trials": a valid trial can be inconclusive.
            'trial_goal': {
                'passed': goal_passed,
                'of': goal_measured,
                'rate': round(goal_passed / goal_measured, 3) if goal_measured else None,
                'wilson95': wilson95(goal_passed, goal_measured),
            },
            # The goal axis beside the pass axis above: a decisive red fails a scenario whose goal was reached.
            'trial_goal_achieved': {
                'passed': sum(row['goal_achieved']['passed'] for row in scenarios.values()),
                'of': sum(row['goal_achieved']['passed'] + row['goal_achieved']['failed'] for row in scenarios.values()),
            },
            'inconsistent_scenarios': [
                sid
                for sid, row in scenarios.items()
                if row['goal']['label'] == 'inconsistent' or any(cell['label'] == 'inconsistent' for cell in row['policies'].values())
            ],
            # Each policy's counts summed over scenarios x valid trials — the one across-trial number a
            # report prints. No label: repeats of the same scenarios are not a wider sample.
            'policies': {
                name: {key: sum(row['policies'][name][key] for row in scenarios.values() if name in row['policies']) for key in _OUTCOME_KEYS}
                for name in sorted({name for row in scenarios.values() for name in row['policies']})
            },
            # The run totals over the same cells, so a refold of this block recomputes them too.
            'totals': run_totals(cells_by_trial),
        },
    }


# ---------------------------------------------------------------------------
# Replay rates: regressions at their cut, and each check's per-turn red rate
# ---------------------------------------------------------------------------


def clean_replays_to_clear(low: float | None) -> int | None:
    """How many CLEAN replays (0 caught of n) it takes before the clean interval's upper bound falls below
    ``low`` — the lower bound of the failed rate measured before a fix — so the two ranges are apart.

    Found by iterating :func:`wilson95` itself, the same rounded bound a comparison reads, never by a
    closed form that could disagree with it at the third decimal. ``None`` when there is nothing to
    clear (``low`` is ``None`` or 0: the defect was never shown to recur) or when more than
    :data:`MAX_CLEAN_REPLAYS` would be needed.
    """
    if low is None or low <= 0:
        return None
    for n in range(1, MAX_CLEAN_REPLAYS + 1):
        interval = wilson95(0, n)
        if interval is not None and interval[1] < low:
            return n
    return None


def clean_replays_to_decide(low: float | None) -> int | None:
    """:func:`clean_replays_to_clear` at the DECISION level: the clean draws (0 caught of n) before
    :func:`wilson80`'s upper bound falls below ``low`` — the before's 80% lower bound — and never fewer than
    :data:`DECISION_MIN_CLEAN`. ``None`` on :func:`clean_replays_to_clear`'s two grounds: nothing to decide (``low``
    is ``None`` or 0), or more than :data:`MAX_CLEAN_REPLAYS` would be needed.
    """
    if low is None or low <= 0:
        return None
    for n in range(1, MAX_CLEAN_REPLAYS + 1):
        interval = wilson80(0, n)
        if interval is not None and interval[1] < low:
            return max(DECISION_MIN_CLEAN, n)
    return None


def _rating(failed: int, of: int) -> dict[str, Any]:
    """The two rating pairs over ``failed`` of ``of``: 95% (published, retirement) and 80% (the decision)."""
    interval = wilson95(failed, of)
    decision = wilson80(failed, of)
    return {
        'wilson95_failed': interval,
        'clean_replays_to_clear': clean_replays_to_clear(interval[0] if interval else None),
        'wilson80_failed': decision,
        'clean_replays_to_decide': clean_replays_to_decide(decision[0] if decision else None),
    }


def rated_cell(counts: Mapping[str, Any]) -> dict[str, Any]:
    """A cell's counts plus the Wilson intervals on its FAILED rate (over passed + failed) and the clean
    replays each lower bound asks for.

    THE one rating every k-of-K comparison reads: a regression replay's cell here, and an exposing
    scenario's cell pooled over a fix's before and after runs (the plugin's ``--verify``), so a
    before rated by one and an after rated by the other are comparable to the third decimal. The
    fixed / not-fixed decision reads the 80% pair (``wilson80_failed``, ``clean_replays_to_decide``);
    retirement and every published range read the 95% pair (``wilson95_failed``, ``clean_replays_to_clear``).
    """
    return {**counts, **_rating(counts['failed'], counts['passed'] + counts['failed'])}


def is_regression_scenario(scenario_id: str) -> bool:
    """Whether a scenario is a recorded regression replayed from its cut (the recorder's folder prefix)."""
    return scenario_id.startswith(REGRESSION_SCENARIO_PREFIX)


def clear_axis(valid: Sequence[ScenarioCells]) -> str:
    """The cell a regression's clear verdict reads: :data:`CLEAR_ANYWHERE` when EVERY valid replay says it did not
    resume (:attr:`ScenarioCells.resumed` is ``False``), else :data:`CLEAR_AT_CUT`. A replay that does not say
    keeps the cut."""
    return CLEAR_ANYWHERE if valid and all(cells.resumed is False for cells in valid) else CLEAR_AT_CUT


def aggregate_regression_replays(cells_by_trial: Sequence[Mapping[str, ScenarioCells]]) -> dict[str, Any]:
    """The ``regression_replays`` block: per recorded regression scenario, per policy, how its check
    landed on the replays — AT THE CUT (:data:`CUT_TURN`) and ANYWHERE in the dialog.

    ``cells_by_trial`` is :func:`aggregate_trials`' input (one mapping per requested trial). Pure.
    Each policy cell is :func:`_counts` (``passed`` / ``failed`` / ``na`` / ``deferred`` / ``inconclusive``,
    ``label``, ``by_trial``) plus :func:`rated_cell`'s two pairs: ``wilson95_failed`` / ``clean_replays_to_clear``
    (published, retirement) and ``wilson80_failed`` / ``clean_replays_to_decide`` (the decision). ``anywhere`` is
    the policy's ordinary scenario outcome (:func:`policy_outcomes`). EVERY policy is listed, so the
    reader picks the regression's owning check. An invalid trial is listed with its reason and counts
    nowhere; a scenario without the regression prefix gets no entry; ``regressions`` is ``{}`` when no
    regression ran.

    SINGLE-HOP replays (``replay_mode``, per regression: ``single_hop`` / ``full`` from its valid attempts,
    ``None`` when none says) read a TYPED trigger-absent N/A at the cut as not caught (:func:`_single_hop_at_cut`), and each
    ``at_cut`` cell counts its TYPED cut abstentions as ``cut_abstained`` (inside ``inconclusive``, never in
    the rate). ``restore_reduced`` counts the valid attempts, in every mode, that resumed with less than the recorded
    state (:func:`restore_reduced`): a single hop's at_cut outcomes are then all ``inconclusive``; a whole replay's are
    kept and caveated. The plugin routes such a folder to whole replays before its next run.

    Each ``at_cut`` cell also counts its ``trigger_absent`` passes (inside ``passed``): the premise never arose on the
    replayed turn, so a reader that retires a regression on clean replays does not count them as clean. The entry sums
    its valid replays' failed (``tool_errors``) and refused (``tool_refusals``) tool calls.

    ``clear_axis`` (:func:`clear_axis`) names the cell — ``at_cut`` or ``anywhere`` — whose ``clean_replays_to_clear``
    and interval decide whether a fix cleared the regression: the whole replay for a folder that replays from its
    recorded opener, the cut for a resumed one. Both cells are always published.
    """
    regressions: dict[str, Any] = {}
    order = list(dict.fromkeys(sid for trial in cells_by_trial for sid in trial if is_regression_scenario(sid)))
    for sid in order:
        per_trial = [trial.get(sid) for trial in cells_by_trial]
        valid_or_none = [cells if cells is not None and cells.invalid_reason is None else None for cells in per_trial]
        names = sorted({name for cells in valid_or_none if cells is not None for name in cells.policies})
        valid = [cells for cells in valid_or_none if cells is not None]
        policies = {
            name: {
                'at_cut': {
                    **rated_cell(_counts([cells.at_cut.get(name, 'inconclusive') if cells is not None else None for cells in valid_or_none])),
                    'cut_abstained': sum(1 for cells in valid if name in cells.cut_abstained),
                    'trigger_absent': sum(1 for cells in valid if name in cells.trigger_absent),
                },
                'anywhere': rated_cell(_counts([cells.policies.get(name, 'inconclusive') if cells is not None else None for cells in valid_or_none])),
            }
            for name in names
        }
        modes = {cells.replay_mode for cells in valid if cells.replay_mode is not None}
        regressions[sid] = {
            'valid': len(valid),
            'invalid': _invalid_trials(per_trial),
            'replay_mode': modes.pop() if len(modes) == 1 else None,
            'restore_reduced': sum(1 for cells in valid if cells.restore_reduced),
            'tool_errors': sum(cells.tool_errors for cells in valid),
            'tool_refusals': sum(cells.tool_refusals for cells in valid),
            'clear_axis': clear_axis(valid),
            'policies': policies,
        }
    return {'schema': REGRESSION_REPLAYS_SCHEMA, 'cut_turn': CUT_TURN, 'replays': len(cells_by_trial), 'regressions': regressions}


def aggregate_policy_turn_rates(cells_by_trial: Sequence[Mapping[str, ScenarioCells]]) -> dict[str, Any]:
    """The ``policy_turn_rates`` block: each check's red rate per judged unit, summed over every scenario
    and every VALID trial. Pure.

    The unit is the check's (:func:`policy_turn_tally`): a judged turn, or a conversation for a check
    the backend scores once per scenario; the first one seen names it. ``wilson95_failed`` is over
    passed + failed — N/A, deferred and inconclusive units are counted beside it, never in it. This is the
    CHECK's rate: a red of the check is not necessarily the defect a regression recorded.
    """
    totals: dict[str, dict[str, Any]] = {}
    for trial in cells_by_trial:
        for cells in trial.values():
            if cells.invalid_reason is not None:
                continue
            for name, row in cells.turn_tally.items():
                entry = totals.setdefault(name, {'unit': row['unit'], **dict.fromkeys(_OUTCOME_KEYS, 0)})
                for key in _OUTCOME_KEYS:
                    entry[key] += row[key]
    policies = {name: {**entry, 'wilson95_failed': wilson95(entry['failed'], entry['passed'] + entry['failed'])} for name, entry in sorted(totals.items())}
    return {'schema': POLICY_TURN_RATES_SCHEMA, 'policies': policies}


def record_time_rate(outcomes: Sequence[str]) -> dict[str, Any]:
    """The rate a regression recurs at its cut, from the replies SAMPLED when it was recorded.

    ``outcomes`` is the backend's :data:`SampledOutcome` per sampled reply, in order. ``k`` caught of ``n``
    graded (``caught`` + ``missed``); ``not_measured`` — and any value the backend did not type — is
    counted beside them and in neither. The recorded reply that earned the folder is NOT an input: it
    was kept because it fired, so it says nothing about how often it fires. Both rating pairs
    (``wilson95_failed`` / ``clean_replays_to_clear``, ``wilson80_failed`` / ``clean_replays_to_decide``) are the
    same arithmetic the replay blocks use, so a record-time rate is a valid "before" for a later replay batch;
    ``label`` is :func:`cell_label` over the same counts.
    """
    k = sum(1 for outcome in outcomes if outcome == 'caught')
    missed = sum(1 for outcome in outcomes if outcome == 'missed')
    n = k + missed
    return {
        'schema': RECORD_TIME_RATE_SCHEMA,
        'cut_turn': CUT_TURN,
        'k': k,
        'n': n,
        'not_measured': len(outcomes) - n,
        'label': cell_label(missed, k, 0, len(outcomes) - n),
        **_rating(k, n),
    }


def _sampled_outcomes(payload: Any) -> list[str] | None:
    """The outcomes in a ``record-rate`` input: a list of outcome strings, or ``score_sampled_replies``'
    result (``{"outcomes": [{"outcome": ...}, ...]}``). ``None`` for anything else."""
    rows = payload.get('outcomes') if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    outcomes = [row.get('outcome') if isinstance(row, dict) else row for row in rows]
    return [str(outcome) for outcome in outcomes] if all(isinstance(outcome, str) for outcome in outcomes) else None


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m plurai.trials record-rate``: JSON on stdin (see :func:`_sampled_outcomes`) →
    :func:`record_time_rate`'s record as one JSON line on stdout. Exit 2 on unreadable input."""
    utf8_stdio()
    parser = exit_codes.UsageParser(prog='python -m plurai.trials', description='Replay-rate arithmetic over sampled regression replies.')
    parser.add_argument('command', choices=('record-rate',))
    parser.parse_args(argv)
    try:
        outcomes = _sampled_outcomes(json.loads(sys.stdin.read()))
    except ValueError:
        outcomes = None
    if outcomes is None:
        print('record-rate: stdin is not a list of outcomes nor a score_sampled_replies result', file=sys.stderr)
        return exit_codes.INPUT_UNREADABLE
    print(json.dumps(record_time_rate(outcomes)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
