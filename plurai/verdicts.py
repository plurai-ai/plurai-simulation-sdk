"""THE ONE WRITER of ``verdicts.json``: how one conversation attempt was judged, in one file.

Every conversation attempt — a scenario folder, a ``trial_<k>/`` copy of it, its
``stability_retry/`` — gets ONE ``verdicts.json`` beside its ``transcript.json`` / ``error.json``. It is
the only place a verdict is read from. ``transcript.json`` is the record of what HAPPENED (turns,
injected inputs, metadata, and the conversation facts the backend re-reads under ``finalize_record``) and is
never rewritten; this file is how it was JUDGED, and it is complete every time: every eval of the eval
set, each with an explicit state. No partial files, nothing to merge, no freshness rule.

Three events write it, all through this module (a source-scanning guard keeps it the only writer):

1. :func:`write_initial` — run end, from the same ``/finalize`` response as the transcript; an attempt
   that failed gets a file too (``graded: false``, ``evals: {}``) so its status is single-source.
2. :func:`apply_rescore` — a ``score_run`` result replaces the evals it re-judged and keeps the rest.
3. :func:`apply_correction` — one audited unit (an eval, or one turn of it) flipped by the plugin's rules.

:func:`import_legacy` is the old-run converter's entry; :func:`read` is the reader. The plugin decides
WHAT changes (summary refusals, lane pins, unit splitting, sibling propagation) and hands a typed event;
this module only checks the result is complete, that the event was built from the file as it is now,
appends ``history`` and writes atomically. It never re-derives ``scenario_verdict`` (``goal``) and never
computes a score: no ``pass_rate``, no weights. Each cell keeps the backend's own stamps verbatim — the
``severity`` band and ``tier`` (:func:`stamped_band`) are what the scoring module (``plurai/scoring.py``)
weights by; nothing here reads them for arithmetic.

STATES. Each eval is ``pass`` / ``fail`` / ``not_applicable`` / ``ungraded``, and the last two carry one
reason from a closed list (:data:`NOT_APPLICABLE_REASONS`, :data:`UNGRADED_REASONS`, or
``inconclusive:<gate>``) — so "everything N/A" and "nothing could be graded" are never one count. Turn
entries of a turn-scoped cell carry the same pair.

STATUS. One of :data:`STATUSES`, precedence ``agent_error`` > ``harness_error`` > ``incomplete`` >
``goal_undecided`` > ``scored``; computed ONCE here, so no reader opens ``error.json`` or ``ended_early``
to classify again. ``agent_error`` is the agent crashing anywhere in the conversation (scored 0);
``harness_error`` is ours (re-run it; left out of the score), with ``retry: true`` on a transient loss;
a measurement breach that voids the scenario is ``harness_error`` (a non-voiding one, such as one errored
judge, leaves the dialog scored with that eval ``ungraded``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from plurai import exit_codes
from plurai._protocol import AGENT_RAISED_KIND, AGENT_RECURSION_BUDGET_KIND
from plurai.observability.scenario_verdict import GOAL_SOURCE_NONE, VERDICT_KEY, _unwrap
from plurai.run_artifacts import STABILITY_RETRY_DIRNAME, VERDICTS_FILENAME

#: The file's schema tag. A reader that does not know it refuses the file.
VERDICTS_SCHEMA: Final = 1

Status = Literal['scored', 'agent_error', 'harness_error', 'incomplete', 'goal_undecided']
EvalState = Literal['pass', 'fail', 'not_applicable', 'ungraded']
AttemptKind = Literal['trial', 'stability_retry']

STATUSES: Final[tuple[Status, ...]] = get_args(Status)
EVAL_STATES: Final[tuple[EvalState, ...]] = get_args(EvalState)

#: Why an eval is ``not_applicable``: its judge said the premise never arose (``not_applicable``, or the
#: samples split and N/A won), its guard kept it from running (``guard_skipped``), its per-turn gate did on
#: every turn, or it handed its charge to a named owner (``deferred``).
NOT_APPLICABLE_REASONS: Final = frozenset({'not_applicable', 'mixed_not_applicable', 'guard_skipped', 'turn_skipped_every_turn', 'deferred'})
#: Why an eval is ``ungraded``: the dialog never reached what it judges (``not_measured``), the judge broke
#: (``judge_error``), or — any ``inconclusive:<gate>`` — the judge ran and did not decide.
UNGRADED_REASONS: Final = frozenset({'not_measured', 'judge_error'})
INCONCLUSIVE_PREFIX: Final = 'inconclusive:'

#: The ``/finalize`` keys that are CONVERSATION FACTS, not verdicts: they stay in ``transcript.json`` under
#: ``finalize_record`` because the backend's re-score reads them back (how the dialog ended, the simulated user's
#: integrity). Every other key is the verdict side, so an unknown new key lands here, where a rescore
#: rewrites it and it cannot go stale. ``what_happened`` is the backend's one-sentence account of the conversation, written
#: from the transcript alone (no verdict in its input), so no rescore can make it stale; it is kept in ``finalize_record`` for the report page.
RECORD_KEYS: Final = frozenset(
    {'dialog_outcome', 'user_sim_integrity', 'scenario_dimensions', 'origin_basis', 'should_terminate', 'world_state_trustworthy', 'what_happened'}
)
#: The ``/finalize`` / ``score_run`` keys that are DERIVED from the cells (counts, rates, name lists) and so
#: are not stored: they belong to the one scorer. ``skipped_policies`` / ``errored_policies`` become states.
DERIVED_KEYS: Final = frozenset(
    {
        'aggregate_score',
        'pass_rate',
        'inconclusive_count',
        'inconclusive_rate',
        'not_applicable_count',
        'not_measured_count',
        'deferred_count',
        'conclusive_count',
        'policies_evaluated',
        'violated_policies',
        'inconclusive_policies',
        'errored_policies',
        'skipped_policies',
        'not_applicable_policies',
        'not_measured_policies',
        'deferred_policies',
        'high_confidence_violations',
        'blocking_violations',
    }
)
#: Identity keys: they land in ``judged_against`` (or a history entry), never in ``judging``.
_IDENTITY_KEYS: Final = frozenset({'eval_revision', 'gt_revision', 'gt_fingerprint', 'eval_set_id', 'scenario_set_id', 'scenario_name', 'run_upload_id'})
#: ``score_run``'s envelope: consumed by :func:`apply_rescore` itself, never stored as a judging fact.
_RESCORE_ENVELOPE_KEYS: Final = frozenset({'cells', 'not_scored', 'not_scored_kinds', 'changed', 'baseline_cells', 'gate_authority', 'warnings'})
_CELLS_KEY: Final = 'evaluation_results'

#: Cell keys that are cost and timing, not judgement: dropped from every stored cell.
_NOISE_CELL_KEYS: Final = frozenset({'usage_metadata', 'latency'})

#: ``score_run``'s typed ``not_scored_kinds``: these mean the eval does not apply, so its old cell leaves.
DROP_KINDS: Final = frozenset({'guard_skipped', 'turn_skipped_every_turn'})
#: …and these mean nobody re-judged it here, so its current entry stays (``kept_from_earlier_revision``).
KEEP_KINDS: Final = frozenset({'no_agent_turns', 'turn_refused', 'guard_raised'})

#: The PLUGIN's command that converts a run written before this file existed. Named in every refusal to read one.
CONVERTER_COMMAND: Final = 'python3 "${CLAUDE_PLUGIN_ROOT}/scripts/convert_verdicts.py" <run dir>'

PRE_CRASH_REAL_DATA: Final = 'real_data'


class VerdictsError(Exception):
    """A refusal: nothing was written. ``code`` is the CLI exit code (``REFUSED``, or ``INPUT_UNREADABLE`` for a file that cannot be read)."""

    def __init__(self, message: str, *, code: int = exit_codes.REFUSED) -> None:
        super().__init__(message)
        self.code = code


class VerdictsMissingError(VerdictsError):
    """The attempt folder holds no ``verdicts.json`` (a run written before it existed: convert it)."""


# ---------------------------------------------------------------------------------------------------------
# The file's shape
# ---------------------------------------------------------------------------------------------------------


class Attempt(BaseModel):
    """Which conversation attempt this file judged."""

    model_config = ConfigDict(extra='forbid')

    scenario: str
    item: str
    kind: AttemptKind
    trial_index: int = Field(ge=1)
    trial_count: int = Field(ge=1)
    thread_id: str | None


class Cause(BaseModel):
    """The typed fact behind a non-``scored`` status: an ``error.json`` ``kind`` (``None`` when untyped), an
    ``ended_early`` reason, ``measurement_unclean``, ``not_graded`` or ``pipeline_error``."""

    model_config = ConfigDict(extra='forbid')

    kind: str | None
    transient: bool = False
    #: The orchestration server's HTTP status (``backend_http``), when the failure carried one.
    status: int | None = None
    #: A lost child's exit code: its sign says exited (>0) vs reaped (<0).
    returncode: int | None = None


class StatusBlock(BaseModel):
    model_config = ConfigDict(extra='forbid')

    value: Status
    cause: Cause | None = None
    #: ``harness_error`` on a transient loss: re-running is the remedy.
    retry: bool = False
    #: The harness's cut reason when it cut the dialog (``incomplete``, unless a voiding breach outranks
    #: it) — kept so a re-judging that clears the outranking status lands back on ``incomplete``.
    cut_by: str | None = None


class JudgedAgainst(BaseModel):
    """The file-level revisions every verdict in it was judged against. Per-change revisions live in ``history``."""

    model_config = ConfigDict(extra='forbid')

    eval_set_id: str | None = None
    #: The backend's digest over the set's per-policy fingerprints (severity, tier, rubric).
    eval_revision: str | None = None
    scenario_set_id: str | None = None
    #: The ground-truth digest the judges graded against (the live run never knows ``gt_revision``).
    gt_fingerprint: str | None = None
    #: The set-level counter, ``None`` until a rescore knows it.
    gt_revision: int | None = None
    #: Evals a rescore did NOT re-judge and the backend did NOT carry (a keep-kind): judged at an earlier revision.
    kept_from_earlier_revision: list[str] = Field(default_factory=list)


def _reason_ok(state: str, reason: str | None) -> bool:
    if state in ('pass', 'fail'):
        return reason is None
    if state == 'not_applicable':
        return reason in NOT_APPLICABLE_REASONS
    return reason in UNGRADED_REASONS or (isinstance(reason, str) and reason.startswith(INCONCLUSIVE_PREFIX) and len(reason) > len(INCONCLUSIVE_PREFIX))


class EvalEntry(BaseModel):
    """One eval: its state, the reason when it is not a verdict, and the backend's cell (``None`` when no cell
    exists — a guard skipped it on every turn). The cell's ``turns[]`` entries carry the same pair."""

    model_config = ConfigDict(extra='forbid')

    state: EvalState
    reason: str | None = None
    cell: dict[str, Any] | None

    @model_validator(mode='after')
    def _closed_reason(self) -> EvalEntry:
        if not _reason_ok(self.state, self.reason):
            raise ValueError(f'reason {self.reason!r} is not one this file allows for state {self.state!r}')
        return self


class VerdictsDoc(BaseModel):
    """The whole file. Validated before every write, so a malformed or partial file is never written."""

    model_config = ConfigDict(extra='forbid')

    verdicts_schema: Literal[1]
    attempt: Attempt
    #: ``False`` for an attempt that produced no grading (it failed, or finalize returned no cells); its
    #: ``evals`` is then ``{}`` — never padded with ``ungraded`` cells.
    graded: bool
    status: StatusBlock
    judged_against: JudgedAgainst
    #: The backend's ``scenario_verdict`` block, verbatim; ``None`` when there is none.
    goal: dict[str, Any] | None
    #: Verdict-side finalize facts not derivable from ``evals`` (measurement_integrity, guard records, …).
    judging: dict[str, Any]
    evals: dict[str, EvalEntry]
    #: One entry per change; the first is the run's own verdicts (states only). Never in the report bundle.
    history: list[dict[str, Any]] = Field(min_length=1)

    @model_validator(mode='after')
    def _coherent(self) -> VerdictsDoc:
        if not self.graded and self.evals:
            raise ValueError('an ungraded attempt carries no evals')
        if self.history[0].get('source') != 'run':
            raise ValueError("history[0] must be the run's own verdicts (source 'run')")
        if [entry.get('seq') for entry in self.history] != list(range(len(self.history))):
            raise ValueError('history seq must count 0..n-1 in order (append-only)')
        for name, entry in self.evals.items():
            turns = entry.cell.get('turns') if entry.cell is not None else None
            for turn in turns if isinstance(turns, list) else ():
                if isinstance(turn, dict) and not (turn.get('state') in EVAL_STATES and _reason_ok(turn['state'], turn.get('reason'))):
                    raise ValueError(f'eval {name!r}: turn {turn.get("turn_idx")!r} carries no valid state/reason')
        return self


class Band(NamedTuple):
    """The weight inputs the backend stamps on a cell, verbatim: its ``severity`` band (``critical|high|medium|low``)
    and its ``tier`` (``critical|failing|standard``)."""

    severity: str | None
    tier: str | None


def stamped_band(entry: Mapping[str, Any] | EvalEntry) -> Band:
    """THE SEAM for the scoring module: what the backend stamped on this eval's cell, under the cell keys ``severity``
    and ``tier`` (``None`` when it stamped nothing, or there is no cell). Read, never computed — the weights
    themselves live only in ``plurai.scoring``."""
    cell = entry.cell if isinstance(entry, EvalEntry) else entry.get('cell')
    if not isinstance(cell, Mapping):
        return Band(None, None)
    severity, tier = cell.get('severity'), cell.get('tier')
    return Band(severity if isinstance(severity, str) else None, tier if isinstance(tier, str) else None)


# ---------------------------------------------------------------------------------------------------------
# States: THE one mapping from a backend cell to (state, reason)
# ---------------------------------------------------------------------------------------------------------


def state_of(unit: Any) -> tuple[EvalState, str | None]:
    """``(state, reason)`` of one judged unit — a whole cell, or one ``turns[]`` entry of a turn-scoped cell.

    Only ``success`` with a boolean ``violated`` is a verdict. ``error`` (and any status the backend does not
    name, or a success without a boolean) is the judge breaking: ``ungraded``/``judge_error``. An
    ``inconclusive`` unit's ``inconclusive_reason`` decides the rest: an N/A reason or ``deferred`` is
    ``not_applicable``, ``not_measured`` is ``ungraded``/``not_measured``, and any other reason is a judge
    that did not decide, ``ungraded``/``inconclusive:<reason>``. The raw ``not_applicable`` status (before
    the backend demotes it) is ``not_applicable``.
    """
    if not isinstance(unit, Mapping):
        return 'ungraded', 'judge_error'
    status = unit.get('status')
    if status == 'success':
        violated = unit.get('violated')
        if isinstance(violated, bool):
            return ('fail' if violated else 'pass'), None
        return 'ungraded', 'judge_error'
    if status == 'not_applicable':
        return 'not_applicable', 'not_applicable'
    if status == 'inconclusive':
        reason = unit.get('inconclusive_reason')
        reason = reason if isinstance(reason, str) and reason else 'unstated'
        if reason in ('not_applicable', 'mixed_not_applicable', 'deferred'):
            return 'not_applicable', reason
        if reason == 'not_measured':
            return 'ungraded', 'not_measured'
        return 'ungraded', f'{INCONCLUSIVE_PREFIX}{reason}'
    return 'ungraded', 'judge_error'


def _stamped_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The backend's cell verbatim minus cost/timing, with ``state``/``reason`` stamped on each ``turns[]`` entry."""
    out = {k: copy.deepcopy(v) for k, v in cell.items() if k not in _NOISE_CELL_KEYS}
    turns = out.get('turns')
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                turn['state'], turn['reason'] = state_of(turn)
    return out


def _all_turns_skipped(cell: Mapping[str, Any]) -> bool:
    turns = cell.get('turns')
    return (cell.get('scope') == 'turn' or isinstance(turns, list)) and not turns and bool(cell.get('skipped_turn_idxs'))


def entry_of(cell: Any, *, skipped: bool = False, errored: bool = False) -> dict[str, Any]:
    """One eval's entry. ``errored`` (the backend's ``errored_policies``) wins, then ``skipped`` (``skipped_policies``:
    its guard kept it from running), then the cell's own state."""
    stored = _stamped_cell(cell) if isinstance(cell, Mapping) else None
    if errored:
        return {'state': 'ungraded', 'reason': 'judge_error', 'cell': stored}
    if skipped:
        reason = 'turn_skipped_every_turn' if stored is not None and _all_turns_skipped(stored) else 'guard_skipped'
        return {'state': 'not_applicable', 'reason': reason, 'cell': stored}
    if stored is None:
        return {'state': 'ungraded', 'reason': 'judge_error', 'cell': None}
    state, reason = state_of(stored)
    return {'state': state, 'reason': reason, 'cell': stored}


def _names(raw: Any) -> set[str]:
    return {str(n) for n in raw} if isinstance(raw, list) else set()


def evals_of(per: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every eval ``/finalize`` named, each with its state: the cells, plus a policy named only in
    ``skipped_policies`` / ``errored_policies`` (a guard skipped it on every turn: no cell)."""
    cells = per.get(_CELLS_KEY)
    cells = cells if isinstance(cells, Mapping) else {}
    skipped, errored = _names(per.get('skipped_policies')), _names(per.get('errored_policies'))
    names = sorted({str(n) for n in cells} | skipped | errored)
    return {name: entry_of(cells.get(name), skipped=name in skipped, errored=name in errored) for name in names}


def _judging_of(source: Mapping[str, Any]) -> dict[str, Any]:
    """The verdict-side facts of a ``/finalize`` or ``score_run`` payload that are neither cells, goal,
    identity, envelope, a derived number nor a conversation fact. Unknown keys land here."""
    skip = RECORD_KEYS | DERIVED_KEYS | _IDENTITY_KEYS | _RESCORE_ENVELOPE_KEYS | {_CELLS_KEY, VERDICT_KEY, 'aggregated_results'}
    return {k: copy.deepcopy(v) for k, v in source.items() if k not in skip}


def finalize_record(per: Mapping[str, Any] | None) -> dict[str, Any]:
    """The conversation facts of a ``/finalize`` payload (:data:`RECORD_KEYS`) — the transcript's ``finalize_record``."""
    return {k: per[k] for k in sorted(RECORD_KEYS) if isinstance(per, Mapping) and k in per}


def _gt_fingerprint(source: Mapping[str, Any], cells: Mapping[str, Any]) -> str | None:
    """The file-level ground-truth digest: the backend's own top-level stamp, else the one value every cell carries."""
    top = source.get('gt_fingerprint')
    if isinstance(top, str) and top:
        return top
    stamps = {c.get('gt_fingerprint') for c in cells.values() if isinstance(c, Mapping) and c.get('gt_fingerprint')}
    return next(iter(stamps)) if len(stamps) == 1 else None


# ---------------------------------------------------------------------------------------------------------
# Status: THE one mapping from the run's signals to the five statuses
# ---------------------------------------------------------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _agent_owned(kind: Any, pre_crash_tool_activity: Any) -> bool:
    """The agent crashed: it raised in its own turn, or its graph hit its recursion limit AFTER it had read
    real data (an empty seed or mock loops the same way, so without that evidence the crash is not the agent's)."""
    return kind == AGENT_RAISED_KIND or (kind == AGENT_RECURSION_BUDGET_KIND and pre_crash_tool_activity == PRE_CRASH_REAL_DATA)


def status_of_error(record: Mapping[str, Any]) -> dict[str, Any]:
    """The status of an attempt that wrote ``error.json`` instead of a transcript: ``agent_error`` when the
    agent crashed (:func:`_agent_owned`), else ``harness_error`` — every other kind (``harness_*``, backend,
    provider, a quarantined scenario, an untyped failure) is ours. ``kind`` falls back to a
    re-drive's ``original_kind``; ``retry`` is a transient loss."""
    kind = record.get('kind')
    if kind is None and isinstance(record.get('redrive'), Mapping):
        kind = record['redrive'].get('original_kind')
    transient = record.get('transient') is True
    cause = {
        'kind': kind if isinstance(kind, str) else None,
        'transient': transient,
        'status': _int_or_none(record.get('status')),
        'returncode': _int_or_none(record.get('returncode')),
    }
    if _agent_owned(kind, record.get('pre_crash_tool_activity')):
        return {'value': 'agent_error', 'cause': cause, 'retry': False}
    return {'value': 'harness_error', 'cause': cause, 'retry': transient}


def _cause(kind: str) -> dict[str, Any]:
    return {'kind': kind, 'transient': False, 'status': None, 'returncode': None}


def status_of_landed(
    per: Mapping[str, Any], *, graded: bool, ended_early: str | None, turn_failure: Mapping[str, Any] | None, cut_by_harness: bool
) -> dict[str, Any]:
    """The status of a dialog that LANDED (finalized), in precedence order.

    ``agent_error``: it ended on the agent crashing (:func:`_agent_owned`, with a landed recursion's own
    ``pre_crash_tool_activity``), or a one-shot hop raised (goal source ``pipeline_error``). ``harness_error``:
    a recursion with no evidence the agent read anything, nothing graded, or a measurement breach that voids the
    scenario (:func:`_voids_scenario`; a non-voiding one, such as one errored judge, leaves the dialog scored with
    that eval ``ungraded``).
    ``incomplete``: the harness cut it (``cut_by_harness`` — the runner's one mapping of its cut reasons).
    ``goal_undecided``: a goal owner exists and reached no verdict. ``scored``: everything else, a goal-less
    eval set included.
    """
    failure = turn_failure if isinstance(turn_failure, Mapping) else {}
    if _agent_owned(ended_early, failure.get('pre_crash_tool_activity')):
        return {'value': 'agent_error', 'cause': _cause(str(ended_early)), 'retry': False}
    if _goal_facts(per)[1] == 'pipeline_error':
        return {'value': 'agent_error', 'cause': _cause('pipeline_error'), 'retry': False}
    if ended_early == AGENT_RECURSION_BUDGET_KIND:
        return {'value': 'harness_error', 'cause': _cause(ended_early), 'retry': False}
    return _judged_status(verdict_of=per, graded=graded, cut_by=str(ended_early) if cut_by_harness else None)


def _goal_facts(source: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    """``(goal_achieved, goal_source)`` read straight off the backend's ``scenario_verdict`` block — never off its
    ``passed`` / ``score``, which the block need not carry. ``(None, None)`` when there is no block; a block
    that names no source has none (``GOAL_SOURCE_NONE``)."""
    block = _unwrap(dict(source), VERDICT_KEY)
    if not isinstance(block, Mapping):
        return None, None
    achieved, goal_source = block.get('goal_achieved'), block.get('goal_source')
    return (achieved if isinstance(achieved, bool) else None), (goal_source if isinstance(goal_source, str) and goal_source else GOAL_SOURCE_NONE)


def _voids_scenario(source: Mapping[str, Any]) -> bool:
    """Whether the backend's ``measurement_integrity`` makes the whole measurement untrustworthy: ``clean`` is not
    ``true`` and a breach VOIDS the scenario (``voids_scenario: true``). A breach the backend typed ``voids_scenario:
    false`` (one judge errored) does not — its eval is ``ungraded`` and the dialog is still scored. A claim this
    cannot read as non-voiding (``clean`` not a bool, no typed breach list, a breach without the flag — an older
    backend) is read as voiding: an untyped breach is never a scored measurement."""
    block = _unwrap(dict(source), 'measurement_integrity')
    if not isinstance(block, Mapping) or 'clean' not in block or block['clean'] is True:
        return False
    breaches = block.get('breaches')
    if block['clean'] is not False or not isinstance(breaches, list) or not breaches:
        return True
    return any(not isinstance(breach, Mapping) or breach.get('voids_scenario') is not False for breach in breaches)


def _judged_status(*, verdict_of: Mapping[str, Any], graded: bool, cut_by: str | None) -> dict[str, Any]:
    """The part of a landed status a re-judging can move: ungraded / a voiding breach / cut / goal undecided / scored.
    ``goal_undecided`` keys on ``goal_achieved`` being null with a goal owner, never on ``passed``."""
    goal_achieved, goal_source = _goal_facts(verdict_of)
    if not graded:
        status = {'value': 'harness_error', 'cause': _cause('not_graded')}
    elif _voids_scenario(verdict_of):
        status = {'value': 'harness_error', 'cause': _cause('measurement_unclean')}
    elif cut_by is not None:
        status = {'value': 'incomplete', 'cause': _cause(cut_by)}
    elif goal_achieved is None and goal_source != GOAL_SOURCE_NONE:
        status = {'value': 'goal_undecided', 'cause': None}
    else:
        status = {'value': 'scored', 'cause': None}
    return {**status, 'retry': False, 'cut_by': cut_by}


#: The ``harness_error`` causes a re-judging can clear (every other non-``scored`` status is a fact of the dialog).
_JUDGED_CAUSES: Final = frozenset({'not_graded', 'measurement_unclean'})


def _rejudged_status(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The status after a rescore or correction moved ``goal`` / ``judging``: a fact of the dialog (an agent crash,
    a failed attempt, a recursion with no evidence) stays; the judged part is recomputed from the new verdicts, the
    harness cut (``cut_by``) included."""
    status = doc['status']
    cause = status.get('cause') or {}
    if status['value'] == 'agent_error' or (status['value'] == 'harness_error' and cause.get('kind') not in _JUDGED_CAUSES):
        return status
    view = {**doc['judging'], **({VERDICT_KEY: doc['goal']} if doc['goal'] is not None else {})}
    return _judged_status(verdict_of=view, graded=doc['graded'], cut_by=status.get('cut_by'))


# ---------------------------------------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------------------------------------


def attempt_of(item_dir: Path, *, scenario: str, thread_id: str | None, run_context: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``attempt`` block of the folder the runner is writing: a ``stability_retry/`` folder is that kind of
    attempt of its parent scenario folder; anything else is a trial (trial 1 of 1 outside ``--trials``)."""
    retry = item_dir.name == STABILITY_RETRY_DIRNAME
    context = run_context or {}
    return {
        'scenario': scenario,
        'item': item_dir.parent.name if retry else item_dir.name,
        'kind': 'stability_retry' if retry else 'trial',
        'trial_index': _int_or_none(context.get('trial_index')) or 1,
        'trial_count': _int_or_none(context.get('trial_count')) or 1,
        'thread_id': thread_id,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def _states(evals: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: entry['state'] for name, entry in sorted(evals.items())}


# ---------------------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------------------


def path_of(item_dir: Path) -> Path:
    """Where an attempt folder's file lives."""
    return Path(item_dir) / VERDICTS_FILENAME


def _validated(doc: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return VerdictsDoc.model_validate(doc).model_dump(mode='json')
    except ValidationError as exc:
        raise VerdictsError(f'the verdicts document is incomplete or malformed, so nothing was written: {exc}') from exc


def _write(item_dir: Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, then write atomically (a sibling temp, then ``os.replace``): a reader sees the whole old file or the whole new one."""
    valid = _validated(doc)
    target = path_of(item_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'{target.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(valid, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(target)
    return valid


def read(item_dir: Path) -> dict[str, Any]:
    """The attempt folder's verdicts, validated. Raises :class:`VerdictsMissingError` (naming the converter) when there
    is no file, and :class:`VerdictsError` on an unreadable file or an unknown schema. No fallback to the transcript."""
    target = path_of(item_dir)
    if not target.is_file():
        raise VerdictsMissingError(
            f'{target} does not exist — a run written before verdicts.json existed. Convert it first with the plurai-simulation '
            f'plugin converter: {CONVERTER_COMMAND}'
        )
    try:
        raw = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise VerdictsError(f'{target} is unreadable: {exc}', code=exit_codes.INPUT_UNREADABLE) from exc
    if not isinstance(raw, dict) or raw.get('verdicts_schema') != VERDICTS_SCHEMA:
        schema = raw.get('verdicts_schema') if isinstance(raw, dict) else None
        raise VerdictsError(f'{target} carries verdicts_schema {schema!r}; this SDK reads {VERDICTS_SCHEMA}')
    return _validated(raw)


def read_or_none(item_dir: Path) -> dict[str, Any] | None:
    """:func:`read`, or ``None`` when the folder holds no readable file of this schema — for a reader that counts
    such a folder as an attempt that produced nothing (the runner's own projections)."""
    try:
        return read(item_dir)
    except VerdictsError:
        return None


def discard_with_error_record(item_dir: Path) -> None:
    """Remove a FAILED attempt's file together with its ``error.json``, when a new attempt takes the folder.
    Only the caller that is dropping that error record calls this; a measured attempt's file moves with its
    transcript instead (``previous_attempt/``)."""
    target = path_of(item_dir)
    if target.exists():
        target.unlink()


# ---------------------------------------------------------------------------------------------------------
# Event 1: run end
# ---------------------------------------------------------------------------------------------------------


def write_initial(
    item_dir: Path,
    *,
    attempt: Mapping[str, Any],
    identity: Mapping[str, Any],
    per: Mapping[str, Any] | None = None,
    ended_early: str | None = None,
    turn_failure: Mapping[str, Any] | None = None,
    cut_by_harness: bool = False,
    finalize_job_id: str | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the attempt's first ``verdicts.json``: from the ``/finalize`` payload ``per`` of a landed dialog,
    or from the ``error`` record of one that failed (``graded: false``, ``evals: {}``). Exactly one of the two.
    ``identity`` carries ``eval_set_id`` / ``scenario_set_id`` as the run knew them."""
    if (per is None) == (error is None):
        raise VerdictsError('write_initial takes exactly one of a finalize payload or an error record')
    if error is not None:
        evals: dict[str, dict[str, Any]] = {}
        status = status_of_error(error)
        source: Mapping[str, Any] = {}
        goal = None
    else:
        source = per or {}
        evals = evals_of(source)
        status = status_of_landed(source, graded=bool(evals), ended_early=ended_early, turn_failure=turn_failure, cut_by_harness=cut_by_harness)
        raw_goal = source.get(VERDICT_KEY)
        goal = copy.deepcopy(raw_goal) if isinstance(raw_goal, dict) else None
    cells = source.get(_CELLS_KEY) if isinstance(source.get(_CELLS_KEY), Mapping) else {}
    revision = source.get('eval_revision') if isinstance(source.get('eval_revision'), str) else None
    fingerprint = _gt_fingerprint(source, cells)
    doc = {
        'verdicts_schema': VERDICTS_SCHEMA,
        'attempt': dict(attempt),
        'graded': bool(evals),
        'status': status,
        'judged_against': {
            'eval_set_id': identity.get('eval_set_id'),
            'eval_revision': revision,
            'scenario_set_id': identity.get('scenario_set_id'),
            'gt_fingerprint': fingerprint,
            'gt_revision': None,
            'kept_from_earlier_revision': [],
        },
        'goal': goal,
        'judging': _judging_of(source),
        'evals': evals,
        'history': [
            {
                'seq': 0,
                'source': 'run',
                'at': _now(),
                'eval_revision': revision,
                'gt_fingerprint': fingerprint,
                'finalize_job_id': finalize_job_id,
                'states': _states(evals),
            }
        ],
    }
    return _write(item_dir, doc)


# ---------------------------------------------------------------------------------------------------------
# Event 2: rescore
# ---------------------------------------------------------------------------------------------------------


def _current(item_dir: Path, history_len: int) -> dict[str, Any]:
    """The file as it is now, refusing an event built from any other version of it (append-only history)."""
    doc = read(item_dir)
    if history_len != len(doc['history']):
        raise VerdictsError(
            f'the event was built from a verdicts.json with {history_len} history entries, but {path_of(item_dir)} now has '
            f'{len(doc["history"])}: something changed it since. Rebuild the event from the current file.'
        )
    return doc


#: The flag the backend sets on a cell it carried forward from the baseline: carrying is not a change.
_CARRIED_KEY: Final = 'carried_forward'

#: The evidence an ``audit_corrections[]`` row carries beside the backend's ruling (the plugin adds it from the run's audit).
_AUDIT_EVIDENCE: Final = ('verdict', 'quote', 'quote_turn', 'quote_file', 'reason')


def _same_cell(old: Mapping[str, Any] | None, new: Mapping[str, Any] | None) -> bool:
    def judged(entry: Mapping[str, Any] | None) -> tuple[Any, Any]:
        cell = (entry or {}).get('cell')
        body = {k: v for k, v in cell.items() if k != _CARRIED_KEY} if isinstance(cell, Mapping) else cell
        return (entry or {}).get('state'), body

    return (old is None) == (new is None) and judged(old) == judged(new)


def apply_rescore(item_dir: Path, result: Mapping[str, Any], *, history_len: int) -> dict[str, Any]:
    """Apply one ``score_run`` result (``summary=false``) to the attempt's file.

    Every eval of the result's ``cells`` replaces the stored one. A ``not_scored`` eval is read by its typed
    kind: a guard skip (:data:`DROP_KINDS`) is ``not_applicable`` and its old cell leaves; a keep-kind
    (:data:`KEEP_KINDS`) keeps the stored entry and is listed in ``kept_from_earlier_revision``. An eval the
    result REMOVED (``changed[].new`` null, no cell) leaves the file. Every other stored eval must be in the
    result, or the result is partial and refused — as is an untyped or unknown not_scored kind. ``goal`` and
    ``judging`` are replaced from the result. Each unit the backend re-applied an audit correction to (the result's
    ``audit_corrections[]``) is one ``backend_corrections`` row, with the evidence the plugin added from the run's
    audit (:data:`_AUDIT_EVIDENCE`), so the history keeps why the verdict moved.
    """
    doc = _current(item_dir, history_len)
    if not doc['graded']:
        raise VerdictsError(f'{path_of(item_dir)} records an attempt that produced no grading ({doc["status"]["value"]}); there is nothing to rescore')
    cells = result.get('cells')
    not_scored = result.get('not_scored')
    changed = result.get('changed')
    goal = result.get(VERDICT_KEY)
    if not isinstance(cells, Mapping) or not isinstance(not_scored, Mapping) or not isinstance(changed, list) or not isinstance(goal, dict):
        raise VerdictsError('a score_run result must carry `cells`, `not_scored`, `changed` and `scenario_verdict`; this one is partial')
    kinds = result.get('not_scored_kinds') or {}
    if not isinstance(kinds, Mapping):
        raise VerdictsError('`not_scored_kinds` must be a policy -> kind mapping')
    removed = {str(c.get('policy_name')) for c in changed if isinstance(c, Mapping) and c.get('new') is None and str(c.get('policy_name')) not in cells}
    old_evals: dict[str, dict[str, Any]] = doc['evals']
    roster = (set(old_evals) - removed) | {str(n) for n in cells} | {str(n) for n in not_scored}
    new_evals: dict[str, dict[str, Any]] = {}
    kept: list[str] = []
    dropped: dict[str, str] = dict.fromkeys(sorted(removed & set(old_evals)), 'removed')
    for name in sorted(roster):
        if name in cells:
            new_evals[name] = entry_of(cells[name])
        elif name in not_scored:
            kind = kinds.get(name)
            if kind in DROP_KINDS:
                new_evals[name] = {'state': 'not_applicable', 'reason': kind, 'cell': None}
                if (old_evals.get(name) or {}).get('cell') is not None:
                    dropped[name] = str(kind)
            elif kind in KEEP_KINDS and name in old_evals:
                new_evals[name] = old_evals[name]
                kept.append(name)
            else:
                raise VerdictsError(f'eval {name!r} is not_scored with kind {kind!r}, which this writer cannot place (untyped, unknown, or nothing to keep)')
        else:
            raise VerdictsError(
                f'eval {name!r} is in {path_of(item_dir)} but absent from the result (not re-judged, carried, skipped or removed): a partial result'
            )
    changes = []
    for name in sorted(set(old_evals) | set(new_evals)):
        old, new = old_evals.get(name), new_evals.get(name)
        if _same_cell(old, new):
            continue
        changes.append(
            {
                'eval': name,
                'turn_idx': None,
                'old': {'state': old['state']} if old else None,
                'new': {'state': new['state']} if new else None,
                'replaced_cell': old.get('cell') if old else None,
            }
        )
    audit_corrections = result.get('audit_corrections') or []
    if not isinstance(audit_corrections, list) or not all(isinstance(row, Mapping) for row in audit_corrections):
        raise VerdictsError('`audit_corrections` must be a list of corrected units')
    backend_corrections = [
        {
            'eval': row.get('policy'),
            'turn_idx': row.get('turn_idx'),
            'cause': row.get('cause'),
            'corrected': row.get('corrected'),
            'rescored': {'state': state_of(row.get('rescored'))[0]},
            **{key: row.get(key) for key in _AUDIT_EVIDENCE},
        }
        for row in audit_corrections
    ]
    revision = result.get('eval_revision') if isinstance(result.get('eval_revision'), str) else doc['judged_against']['eval_revision']
    gt_revision = _int_or_none(result.get('gt_revision'))
    judged = dict(doc['judged_against'])
    judged.update(
        eval_revision=revision,
        gt_revision=gt_revision,
        gt_fingerprint=_gt_fingerprint(result, cells) or judged['gt_fingerprint'],
        kept_from_earlier_revision=sorted(kept),
        **{key: result[key] for key in ('eval_set_id', 'scenario_set_id') if isinstance(result.get(key), str)},
    )
    updated = {**doc, 'evals': new_evals, 'goal': copy.deepcopy(goal), 'judging': _judging_of(result), 'judged_against': judged}
    updated['status'] = _rejudged_status(updated)
    updated['history'] = [
        *doc['history'],
        {
            'seq': len(doc['history']),
            'source': 'rescore',
            'at': _now(),
            'eval_revision': revision,
            'gt_revision': gt_revision,
            'run_upload_id': result.get('run_upload_id'),
            # The upload ITEM it scored (``<scenario>/stability_retry`` for a retry scored by reference): with
            # ``run_upload_id`` it tells a retry's own sample from its first attempt scored twice.
            'scenario_name': result.get('scenario_name'),
            'rejudged': sorted(str(n) for n, c in cells.items() if not (isinstance(c, Mapping) and c.get(_CARRIED_KEY))),
            'carried': sum(1 for c in cells.values() if isinstance(c, Mapping) and c.get(_CARRIED_KEY)),
            'dropped': dropped,
            'changes': changes,
            'backend_corrections': backend_corrections,
            **({'warnings': copy.deepcopy(result['warnings'])} if result.get('warnings') else {}),
        },
    ]
    return _write(item_dir, updated)


# ---------------------------------------------------------------------------------------------------------
# Event 3: audit correction (and the plugin's own local check)
# ---------------------------------------------------------------------------------------------------------


def _unit_state(entry: Mapping[str, Any] | None, turn_idx: int | None) -> str | None:
    """The state of one unit of an eval: the eval itself, or the ``turns[]`` entry for ``turn_idx``. ``None`` when absent."""
    if entry is None:
        return None
    if turn_idx is None:
        return entry['state']
    turns = (entry.get('cell') or {}).get('turns')
    match = next((t for t in turns if isinstance(t, Mapping) and t.get('turn_idx') == turn_idx), None) if isinstance(turns, list) else None
    return match.get('state') if match is not None else None


#: ``quote_file`` is optional: the path of the file the agent wrote that the quote is anchored on (``None`` otherwise),
#: kept as :func:`apply_rescore` keeps it on a backend correction. ``call_id`` is optional too: the tool call the agent
#: made that the ruling cites instead of a reply quote. ``clause`` is the rubric clause the audit row cites (the
#: corrected cell's customer-facing reasoning names it; the trail keeps it beside the auditor's own ``reason``).
_CORRECTION_FACTS: Final = ('verdict', 'cause', 'quote', 'quote_turn', 'quote_file', 'call_id', 'reason', 'clause', 'propagated_from')


def apply_correction(item_dir: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ONE corrected unit (an eval, or one turn of it) the plugin's rules decided.

    ``event``: ``{history_len, source: 'correction'|'local_check', eval, turn_idx, old: {state}, new: {state},
    cell, goal?, verdict, cause, quote, quote_turn, quote_file?, call_id?, reason, clause?, propagated_from}``. ``cell`` is the corrected backend
    cell (built by the plugin; this writer never re-derives one); ``goal`` the recomputed ``scenario_verdict`` when
    the correction moves it. Refused unless ``old.state`` is the unit's state in the file now, the event was built
    from the file's current history, and ``cell`` really puts the unit in ``new.state``. Only a ``local_check`` may
    add an eval the file does not hold (``old.state`` null).
    """
    history_len = _int_or_none(event.get('history_len'))
    if history_len is None:
        raise VerdictsError('a correction event must carry `history_len`: the length of the history it was built from')
    doc = _current(item_dir, history_len)
    if not doc['graded']:
        raise VerdictsError(f'{path_of(item_dir)} records an attempt that produced no grading; there is nothing to correct')
    source = event.get('source', 'correction')
    if source not in ('correction', 'local_check'):
        raise VerdictsError(f'a correction event names source {source!r}; only correction / local_check are corrections')
    name = event.get('eval')
    turn_idx = _int_or_none(event.get('turn_idx'))
    old, new, cell = event.get('old'), event.get('new'), event.get('cell')
    if not isinstance(name, str) or not isinstance(old, Mapping) or not isinstance(new, Mapping) or not isinstance(cell, Mapping):
        raise VerdictsError('a correction event must carry `eval`, `old`, `new` and the corrected `cell`')
    current = doc['evals'].get(name)
    if current is None and source != 'local_check':
        raise VerdictsError(f'eval {name!r} is not in {path_of(item_dir)}; a correction flips an existing unit')
    if _unit_state(current, turn_idx) != old.get('state'):
        raise VerdictsError(
            f'stale correction: eval {name!r}{f" turn {turn_idx}" if turn_idx is not None else ""} is '
            f'{_unit_state(current, turn_idx)!r} in the file, not {old.get("state")!r} as the event says'
        )
    entry = entry_of(cell)
    if _unit_state(entry, turn_idx) != new.get('state'):
        raise VerdictsError(f'the corrected cell puts eval {name!r} in {_unit_state(entry, turn_idx)!r}, not {new.get("state")!r} as the event says')
    updated = {**doc, 'evals': {**doc['evals'], name: entry}}
    if isinstance(event.get('goal'), dict):
        updated['goal'] = copy.deepcopy(event['goal'])
    updated['status'] = _rejudged_status(updated)
    updated['history'] = [
        *doc['history'],
        {
            'seq': len(doc['history']),
            'source': source,
            'at': _now(),
            'eval': name,
            'turn_idx': turn_idx,
            'old': {'state': old.get('state')},
            'new': {'state': new.get('state')},
            **{key: event.get(key) for key in _CORRECTION_FACTS},
            'replaced_cell': current.get('cell') if current else None,
        },
    ]
    return _write(item_dir, updated)


# ---------------------------------------------------------------------------------------------------------
# The old-run converter's entry
# ---------------------------------------------------------------------------------------------------------


def import_legacy(item_dir: Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Write a whole converted document for an attempt folder that has none (the plugin's converter builds it by
    the current rules, ``history`` = ``[run, convert]``). Refused when the folder already holds one: converting twice
    would rewrite a history."""
    if path_of(item_dir).exists():
        raise VerdictsError(f'{path_of(item_dir)} already exists; a converted document is written once, never over a live one')
    if doc.get('verdicts_schema') != VERDICTS_SCHEMA:
        raise VerdictsError(f'the converted document carries verdicts_schema {doc.get("verdicts_schema")!r}; this SDK writes {VERDICTS_SCHEMA}')
    return _write(item_dir, doc)


# ---------------------------------------------------------------------------------------------------------
# Readers' projection
# ---------------------------------------------------------------------------------------------------------


def as_policy_results(doc: Mapping[str, Any], *, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The file in the shape of a ``/finalize`` ``policy_evaluation_results`` block, for a reader whose pure
    helpers read that shape (``plurai.trials``, the stability projection): the stored cells, ``goal`` as
    ``scenario_verdict``, the judging facts, the conversation facts of the transcript's ``finalize_record``,
    and ``violated_policies`` / ``skipped_policies`` / ``errored_policies`` re-stated from the states. Read from THIS
    file only."""
    evals: Mapping[str, Mapping[str, Any]] = doc.get('evals') or {}
    return {
        **dict(record or {}),
        **copy.deepcopy(doc.get('judging') or {}),
        _CELLS_KEY: {name: copy.deepcopy(entry['cell']) for name, entry in evals.items() if entry.get('cell') is not None},
        **({VERDICT_KEY: copy.deepcopy(doc['goal'])} if doc.get('goal') is not None else {}),
        'violated_policies': sorted(n for n, e in evals.items() if e.get('state') == 'fail'),
        'skipped_policies': sorted(n for n, e in evals.items() if e.get('reason') in ('guard_skipped', 'turn_skipped_every_turn')),
        'errored_policies': sorted(n for n, e in evals.items() if e.get('reason') == 'judge_error'),
        'eval_revision': (doc.get('judged_against') or {}).get('eval_revision'),
    }


# ---------------------------------------------------------------------------------------------------------
# CLI: `plurai verdicts apply-rescore | apply-correction | import-legacy`
# ---------------------------------------------------------------------------------------------------------


def _load_json_object(path: Path, what: str) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise VerdictsError(f'cannot read {what} {path}: {exc}', code=exit_codes.INPUT_UNREADABLE) from exc
    if not isinstance(raw, dict):
        raise VerdictsError(f'{what} {path} is not a JSON object', code=exit_codes.INPUT_UNREADABLE)
    return raw


def run_cli(args: argparse.Namespace) -> int:
    """One ``plurai verdicts`` verb. Exit 0 written (one ``wrote <path> (history <n>)`` line on stdout); 2 an input
    file is unreadable; 3 refused — partial result, stale event, unknown kind, missing or existing file. A refusal
    writes nothing and names why on stderr."""
    try:
        if args.verb == 'apply-rescore':
            doc = apply_rescore(args.item_dir, _load_json_object(args.result, 'score_run result'), history_len=args.history_len)
        elif args.verb == 'apply-correction':
            doc = apply_correction(args.item_dir, _load_json_object(args.event, 'correction event'))
        else:
            doc = import_legacy(args.item_dir, _load_json_object(args.doc, 'converted document'))
    except VerdictsError as exc:
        print(f'plurai verdicts {args.verb}: refused: {exc}', file=sys.stderr)
        return exc.code
    print(f'wrote {path_of(args.item_dir)} (history {len(doc["history"])})')
    return exit_codes.OK


def add_parser(sub: argparse._SubParsersAction, *, path_type: Any = Path) -> None:
    """Register ``verdicts`` and its three verbs on the ``plurai`` parser."""
    verdicts = sub.add_parser('verdicts', help="Write a conversation attempt's verdicts.json — the one writer (rescore, correction, legacy import).")
    verbs = verdicts.add_subparsers(dest='verb', required=True)
    rescore = verbs.add_parser('apply-rescore', help="Apply a score_run result (summary=false) to an attempt folder's verdicts.json.")
    rescore.add_argument('--item-dir', required=True, type=path_type, metavar='DIR', help='The attempt folder (holds verdicts.json).')
    rescore.add_argument('--result', required=True, type=path_type, metavar='FILE', help='The score_run result, as JSON.')
    rescore.add_argument(
        '--history-len', required=True, type=int, metavar='N', help="len(history) of the verdicts.json the score_run's baseline_cells were read from."
    )
    correction = verbs.add_parser('apply-correction', help="Apply one corrected unit (an eval, or one turn of it) to an attempt folder's verdicts.json.")
    correction.add_argument('--item-dir', required=True, type=path_type, metavar='DIR', help='The attempt folder (holds verdicts.json).')
    correction.add_argument('--event', required=True, type=path_type, metavar='FILE', help='The correction event, as JSON.')
    legacy = verbs.add_parser('import-legacy', help='Write a converted verdicts.json into an attempt folder that has none.')
    legacy.add_argument('--item-dir', required=True, type=path_type, metavar='DIR', help='The attempt folder.')
    legacy.add_argument('--doc', required=True, type=path_type, metavar='FILE', help='The whole converted document, as JSON.')
    for parser in (rescore, correction, legacy):
        parser.set_defaults(func=run_cli)


__all__ = [
    'CONVERTER_COMMAND',
    'DROP_KINDS',
    'EVAL_STATES',
    'KEEP_KINDS',
    'NOT_APPLICABLE_REASONS',
    'RECORD_KEYS',
    'STATUSES',
    'UNGRADED_REASONS',
    'VERDICTS_SCHEMA',
    'Band',
    'VerdictsDoc',
    'VerdictsError',
    'VerdictsMissingError',
    'add_parser',
    'apply_correction',
    'apply_rescore',
    'as_policy_results',
    'attempt_of',
    'discard_with_error_record',
    'entry_of',
    'evals_of',
    'finalize_record',
    'import_legacy',
    'path_of',
    'read',
    'read_or_none',
    'stamped_band',
    'state_of',
    'status_of_error',
    'status_of_landed',
    'write_initial',
]
