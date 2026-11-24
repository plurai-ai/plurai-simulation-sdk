"""The one automatic retry for infra causes: which scenarios a second pass is for, and merging that pass into the first.

After the trials finish, ``plurai ci run`` re-runs once, with the same trial count, the scenarios whose gating scenario ×
policy could not be measured for an infra cause that a second pass may clear, or lost some trials to one while every trial it
did measure was violated. The retry is capped at a share of the scenarios that ran, and one scenario is always allowed:
above the cap the box is unhealthy, not the scenarios, and the retry is skipped. A skipped or failed retry never hides a gating
failure of the first pass: the run then ends on that failure with the retry problem as a note. There is no switch to turn the
retry off.

The second pass fills in only the trials the first pass lost: a trial the first pass decided keeps its verdict, so a violation
never turns green on the rerun.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from plurai.report.outcome import DEFAULT_FAIL_ON_TAGS
from plurai.report.render import Stopped
from plurai.report.run import Cell, Run, read_run, retriable_cause

#: The retry covers at most this share of the scenarios that ran.
RETRY_CAP_PERCENT: Final = 20

#: What an invalid trial entry says: which trial, why, and a credit refusal's code and reset time.
_INVALID_FIELDS: Final = ('trial', 'reason', 'code', 'resets_at')


@dataclass(frozen=True)
class RetrySelection:
    """The scenarios to run a second time, and whether there are too many of them."""

    #: In run order. Empty when nothing is worth retrying.
    scenarios: tuple[str, ...]
    #: How many scenarios ran in the first pass: the denominator of the cap.
    ran: int

    @property
    def over_cap(self) -> bool:
        """Above the cap, except one scenario: a suite of four or fewer scenarios would otherwise never retry a single blip."""
        return len(self.scenarios) > 1 and 100 * len(self.scenarios) > RETRY_CAP_PERCENT * self.ran


def _worth_retrying(cell: Cell) -> bool:
    """A gating scenario × policy that no rule ignores and a second pass may measure: unmeasured for infra causes alone, every
    one of them retriable, or violated in every measured trial with the others lost to such causes."""
    result = cell.outcome
    if cell.ignored is not None or result.level != 'gating':
        return False
    if cell.lost_to_infra:
        return True
    if result.outcome != 'not_measured' or result.classes != ('infra',):
        return False
    return all(retriable_cause(cause) for cause in dict.fromkeys(c for c in cell.causes if c))


def select_retries(run: Run) -> RetrySelection:
    """The scenarios of ``run`` (the first pass) to run again: those with a gating scenario × policy that is not measured for
    an infra cause a second pass may clear, or that failed in every measured trial and lost the others to one. Suite and
    credits causes, a mock that needs a redeploy, an unknown cause, advisory scenario × policy and ignored ones never select a
    scenario.

    A run with a credits cell retries nothing: the refusal stopped the run, so a second pass would be refused the same way,
    and the scenarios it kept from running are not infra failures to count against the cap."""
    counted = [cell for cell in run.cells if cell.ignored is None]
    if any('credits' in cell.outcome.classes for cell in counted):
        return RetrySelection((), len({cell.scenario for cell in counted}))
    chosen = tuple(dict.fromkeys(cell.scenario for cell in counted if _worth_retrying(cell)))
    return RetrySelection(chosen, len({cell.scenario for cell in counted}))


def cap_reason(selection: RetrySelection) -> str:
    """Why a selection above the cap is not retried."""
    return f'{len(selection.scenarios)} of {selection.ran} scenarios unmeasured, over the {RETRY_CAP_PERCENT}% retry cap'


def retry_stop(selection: RetrySelection) -> Stopped:
    """The stop for a selection above the cap: not measured, infra. Nothing is retried."""
    return Stopped.not_measured('infra', cap_reason(selection), 'the backend box looks unhealthy and not the scenarios, so run the job again once it recovers.')


def retry_note(why: str) -> str:
    """The note on results that carry a gating failure when the retry could not happen for ``why``: the run exits on the failure
    and says what the retry would have measured."""
    return f'no retry: {why}; the results are those of the first pass'


def _bake_unclean(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The scenario rows of ``doc`` with the trials its ``measurement`` block marks unclean written into ``by_trial`` as
    ``unclean``. The block lists trial numbers for the whole run, so a row that moves into another run carries the mark
    itself; the reason of an inconclusive trial written as ``unclean`` is dropped with it."""
    block = doc['trials']
    unclean = {row.get('trial') for row in block.get('measurement') or [] if isinstance(row, dict) and row.get('clean') is False}
    rows = copy.deepcopy(block['scenarios'])
    for row in rows.values():
        for entry in (row.get('policies') or {}).values():
            reasons = list(entry.get('inconclusive_reasons') or [])
            kept = 0
            for index, recorded in enumerate(entry.get('by_trial') or [], start=1):
                if recorded != 'inconclusive':
                    continue
                if index in unclean:
                    entry['by_trial'][index - 1] = 'unclean'
                    if kept < len(reasons):
                        del reasons[kept]
                else:
                    kept += 1
            if 'inconclusive_reasons' in entry:
                entry['inconclusive_reasons'] = reasons
    return rows


def _own_invalid(row: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {entry['trial']: entry for entry in row.get('invalid') or [] if isinstance(entry, dict) and isinstance(entry.get('trial'), int)}


def _policy_of(row: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """The policy ``entry`` of ``row`` with each invalid trial carrying its own ``invalid`` entry, so the policy no longer
    depends on the scenario's list of them and can sit in a scenario with policies of the other pass."""
    invalid = _own_invalid(row)
    kept = copy.deepcopy(dict(entry))
    kept['by_trial'] = [
        invalid[index] if recorded is None and index in invalid else recorded for index, recorded in enumerate(entry.get('by_trial') or [], start=1)
    ]
    return kept


def _replaced_trials(cell: Cell) -> frozenset[int]:
    """The 1-based trials of a first-pass cell that a second pass may measure: those lost to an infra cause it may clear."""
    return frozenset(
        index
        for index, (verdict, cause) in enumerate(zip(cell.verdicts, cell.causes, strict=True), start=1)
        if verdict == 'invalid_infra' and cause and retriable_cause(cause)
    )


def _reason_at(entry: Mapping[str, Any], index: int) -> str | None:
    """The reason the judges gave for inconclusive trial ``index`` of ``entry``: the one at its rank among the inconclusive trials."""
    by_trial = entry.get('by_trial') or []
    rank = sum(1 for recorded in by_trial[: index - 1] if recorded == 'inconclusive')
    reasons = entry.get('inconclusive_reasons') or []
    return reasons[rank] if rank < len(reasons) else None


def _merge_trials(old: Mapping[str, Any], new: Mapping[str, Any], replace: frozenset[int], configured: int) -> dict[str, Any]:
    """The policy ``old`` of the first pass with each trial of ``replace`` taken from ``new``, trial by trial. The first pass's
    findings stay when it has any; a violation only the second pass saw brings its own."""
    old_trials, new_trials = old.get('by_trial') or [], new.get('by_trial') or []
    by_trial: list[Any] = []
    reasons: list[str] = []
    for index in range(1, configured + 1):
        source, trials = (new, new_trials) if index in replace else (old, old_trials)
        recorded = trials[index - 1] if index <= len(trials) else None
        by_trial.append(recorded)
        if recorded == 'inconclusive' and (reason := _reason_at(source, index)) is not None:
            reasons.append(reason)
    merged = copy.deepcopy(dict(old))
    merged['by_trial'] = by_trial
    if reasons or 'inconclusive_reasons' in old or 'inconclusive_reasons' in new:
        merged['inconclusive_reasons'] = reasons
    if 'findings' not in old and any(recorded == 'failed' for index, recorded in enumerate(by_trial, start=1) if index in replace):
        merged.update({key: copy.deepcopy(new[key]) for key in ('findings', 'reasoning', 'agreement') if key in new})
    owners = sorted({*(old.get('deferred_to') or ()), *((new.get('deferred_to') or ()) if 'deferred' in by_trial else ())})
    if owners:
        merged['deferred_to'] = owners
    return merged


def merge_retry(first: Mapping[str, Any], second: Mapping[str, Any], *, fail_on: Iterable[str] = DEFAULT_FAIL_ON_TAGS) -> dict[str, Any]:
    """The ``run.json`` of the first pass with the second pass merged in, scenario × policy by scenario × policy.

    ``first`` and ``second`` are ``run.json`` documents of the same trial count; ``second`` holds only the retried
    scenarios. A scenario × policy that the first pass could not measure for an infra cause, or measured as violated in every
    trial but the ones it lost to such a cause, takes those lost trials from the second pass and is listed under ``retried``;
    every trial the first pass decided keeps its verdict, so a failure of the first pass never turns green on the rerun. Every
    other scenario × policy of those scenarios keeps its first-pass trials. The result adds the threads and the duration of
    both passes and folds the unclean marks of each pass into its own trials. Neither argument is changed. ``fail_on`` is the
    run's gating tags. Raises ``ValueError`` for another trial count or for a scenario the first pass never ran.
    """
    configured = first['trials']['n']
    if second['trials']['n'] != configured:
        raise ValueError(f'the retry ran {second["trials"]["n"]} trials; the first pass ran {configured}, and the trial count must match')
    before, after = _bake_unclean(first), _bake_unclean(second)
    unknown = sorted(set(after) - set(before))
    if unknown:
        raise ValueError(f'the retry ran {", ".join(unknown)}, which the first pass did not run')
    replaceable = {
        cell.key: _replaced_trials(cell)
        for cell in read_run(first, fail_on=fail_on).cells
        if cell.lost_to_infra or (cell.outcome.outcome == 'not_measured' and cell.outcome.classes == ('infra',))
    }
    rows = dict(before)
    retried: list[dict[str, str]] = list(first.get('retried') or [])
    for scenario, new_row in after.items():
        old_row = before[scenario]
        old_policies, new_policies = old_row.get('policies') or {}, new_row.get('policies') or {}
        new_invalid = [entry for entry in new_row.get('invalid') or [] if isinstance(entry, dict)]
        policies = {}
        for name in dict.fromkeys([*old_policies, *new_policies]):
            replace = replaceable.get((scenario, name), frozenset())
            # A scenario the second pass could not judge at all (a credit refusal stopped it) has no policy, only invalid trials.
            new_entry = new_policies.get(name) or ({'by_trial': [None] * configured} if new_invalid and replace else None)
            if name not in old_policies:
                source_row, source = new_row, new_policies[name]
                if replace:
                    retried.append({'scenario': scenario, 'policy': name})
            elif new_entry is not None and replace:
                baked_old, baked_new = _policy_of(old_row, old_policies[name]), _policy_of(new_row, new_entry)
                policies[name] = _merge_trials(baked_old, baked_new, replace, configured)
                retried.append({'scenario': scenario, 'policy': name})
                continue
            else:
                source_row, source = old_row, old_policies[name]
            policies[name] = _policy_of(source_row, source)
        # A scenario with no policy in either pass keeps the trials it lost: the cause is all that says why, a credit stop included.
        kept = [] if policies else (new_invalid or old_row.get('invalid') or [])
        rows[scenario] = {**old_row, 'invalid': kept, 'policies': policies}
    merged = copy.deepcopy(dict(first))
    merged['trials']['scenarios'] = rows
    merged['trials'].pop('measurement', None)
    merged['retried'] = list({(item['scenario'], item['policy']): item for item in retried}.values())
    if models := sorted({*first.get('agent_models', ()), *second.get('agent_models', ())}):
        merged['agent_models'] = models
    for key in ('threads', 'duration_seconds'):
        if isinstance(first.get(key), int) and isinstance(second.get(key), int):
            merged[key] = first[key] + second[key]
    return merged
