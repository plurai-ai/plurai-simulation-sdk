"""The ``run.json`` of one pass of the runner, composed from the artifacts the runner left in its output directory.

The runner's ``run_summary.json`` carries the ``trials`` block (``trials-v1``): per scenario and policy, the outcome on each
trial. This module keeps what the report reads of it, adds what the judges said (read from each attempt's verdicts through :mod:`plurai.verdicts`)
and refines the cause of an invalid trial from the attempt's ``error.json``: a backend 503 or 504, a gateway 402 with its
code, a timeout, a mock the box restarted. Pure file reading: no network, no clock.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from plurai import client_handshake
from plurai.observability.model_calls import models_of
from plurai.report import RUN_SCHEMA
from plurai.report.run import CREDIT_CODES
from plurai.run_artifacts import RUN_SUMMARY_FILENAME, TRIAL_DIR_PREFIX, load_transcript
from plurai.trials import aggregate_trials, cells_from_run_dir, cells_from_summary
from plurai.verdicts import path_of as verdicts_path
from plurai.verdicts import read_or_none as read_verdicts_or_none

#: The judge samples behind a verdict; the backend reports agreement as a fraction of them.
JUDGE_SAMPLES: Final = 3

#: The mock's own failure to answer, by its typed cause, as the cause of the trial.
_ROUTE_CAUSE: Final = {'transport': 'box_restart', 'held_for_edit': 'awaiting_edit_finalize', 'not_deployed': 'needs_redeploy'}
_TIMEOUT_KINDS: Final = frozenset({'harness_scenario_timeout', 'harness_turn_timeout', 'harness_finalize_poll_timeout'})


def _load(path: Path) -> tuple[Any, str | None]:
    """The JSON in ``path`` and ``None``, or ``None`` and why there is none: the file is missing, cannot be read, or is no JSON."""
    try:
        return json.loads(path.read_text(encoding='utf-8')), None
    except FileNotFoundError:
        return None, f'there is no {path.name}'
    except OSError as failure:
        return None, f'{path.name} cannot be read ({type(failure).__name__})'
    except ValueError:
        return None, f'{path.name} is not valid JSON'


def _read(path: Path) -> Any:  # noqa: ANN401
    return _load(path)[0]


def runner_stop_words(out: Path, exit_code: int) -> str:
    """Why a pass into ``out`` has no ``run.json``, in plain words: the runner's exit code and, when its summary is the problem,
    what is wrong with it. An exit of 0 with no usable summary is the summary's fault, not a stopped runner."""
    summary, problem = _load(out / RUN_SUMMARY_FILENAME)
    if problem is None and not isinstance(summary, dict):
        problem = f'{RUN_SUMMARY_FILENAME} is not a JSON object'
    stopped = f'the runner stopped with exit {exit_code}'
    if problem is None:
        return stopped
    return f'{stopped}; {problem}' if exit_code else f'the runner exited 0 but {problem}'


def trial_dir(out: Path, index: int) -> Path:
    """Where trial ``index`` of a run into ``out`` wrote its scenario folders (trial 1 is ``out`` itself)."""
    return out if index == 1 else out / f'{TRIAL_DIR_PREFIX}{index}'


def agent_models(out: Path, scenarios: Collection[str], trials: int) -> list[str]:
    """The distinct model ids the agent's replies name in the transcripts the pass left, sorted.

    A reply names its model in its ``response_metadata`` when the agent framework reports one; an agent that does not
    contributes nothing, so the list can be empty.
    """
    found: set[str] = set()
    for index in range(1, trials + 1):
        for scenario in scenarios:
            found.update(models_of(load_transcript(trial_dir(out, index) / scenario / 'transcript.json').raw.get('chatbot_messages')))
    return sorted(found)


def _credit_cause(credit: Mapping[str, Any]) -> dict[str, Any]:
    """The ``credits`` cause of an ``error.json`` ``credit`` block ``{code, resets_at?}``. A code outside :data:`CREDIT_CODES`
    carries no ``code``, which the report reads as an infra refusal."""
    entry: dict[str, Any] = {'reason': 'credits'}
    if credit.get('code') in CREDIT_CODES:
        entry['code'] = credit['code']
        if isinstance(credit.get('resets_at'), str):
            entry['resets_at'] = credit['resets_at']
    return entry


def error_cause(error: Mapping[str, Any]) -> dict[str, Any] | None:
    """The cause an ``error.json`` earns beyond the runner's generic ``errored``: a dict with ``reason`` and, for a credit
    refusal (its ``credit`` block, whatever route refused), ``code`` and ``resets_at``. ``None`` when the record says nothing finer."""
    kind = error.get('kind')
    status = error.get('status')
    if isinstance(error.get('credit'), Mapping):
        return _credit_cause(error['credit'])
    if kind == 'backend_http' and isinstance(status, int):
        if status == 402:
            return {'reason': 'credits'}
        if status in (429, 502, 503):
            return {'reason': 'unavailable'}
        if status == 504:
            return {'reason': 'gateway_timeout'}
    if kind in _TIMEOUT_KINDS:
        return {'reason': 'timeout'}
    cause = _ROUTE_CAUSE.get(str(error.get('route_cause')))
    return {'reason': cause} if cause else None


def credentials_status(error: Mapping[str, Any]) -> int | None:
    """401 or 403 when the attempt failed because the backend refused the token; else ``None``."""
    status = error.get('status')
    if error.get('kind') == 'backend_http' and status in (401, 403):
        return int(status)
    return 401 if error.get('route_cause') == 'auth' else None


def upgrade_refusal(error: Mapping[str, Any]) -> client_handshake.UpgradeRefusal | None:
    """The version refusal an attempt failed on (HTTP 426), from the ``upgrade`` block of its ``error.json``; ``None`` for
    any other failure."""
    if error.get('kind') != 'backend_http' or error.get('status') != client_handshake.UPGRADE_STATUS:
        return None
    return client_handshake.UpgradeRefusal.from_record(error.get('upgrade'))


def attempt_errors(out: Path, trials: int) -> list[tuple[int, str, dict[str, Any]]]:
    """Every ``error.json`` the pass left, as ``(trial, scenario, record)``."""
    found = []
    for index in range(1, trials + 1):
        base = trial_dir(out, index)
        for path in sorted(base.glob('*/error.json')):
            record = _read(path)
            if isinstance(record, dict):
                found.append((index, path.parent.name, record))
    return found


def pass_trials_block(out: Path, trials: int) -> dict[str, Any] | None:
    """The ``trials`` block of a pass: the summary's own when it has one, else every trial's, aggregated here. A run stopped for
    credit leaves each trial's own summary and no cross-trial block. ``None`` when the runner left no summary."""
    summary = _read(out / RUN_SUMMARY_FILENAME)
    if not isinstance(summary, dict):
        return None
    block = summary.get('trials')
    if isinstance(block, dict):
        return block
    indexes = range(1, trials + 1)
    own = [_read(trial_dir(out, index) / RUN_SUMMARY_FILENAME) for index in indexes]
    measurement = [(s.get('measurement_clean'), s.get('measurement_unclean_because') or []) if isinstance(s, dict) else (None, []) for s in own]
    cells = [cells_from_run_dir(trial_dir(out, index)) for index in indexes]
    return aggregate_trials(cells, run_name=str(summary.get('run_name') or 'ci'), measurement=measurement)


def _judged(out: Path, scenario: str, trial: int) -> Mapping[str, Mapping[str, Any]]:
    doc = read_verdicts_or_none(trial_dir(out, trial) / scenario)
    evals = doc.get('evals') if isinstance(doc, dict) else None
    return evals if isinstance(evals, dict) else {}


def _agreement(cell: Mapping[str, Any]) -> list[int] | None:
    value = cell.get('agreement')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        return None
    return [round(value * JUDGE_SAMPLES), JUDGE_SAMPLES]


def _findings(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = cell.get('findings')
    rows: list[Any] = raw if isinstance(raw, list) else []
    found = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        entry: dict[str, Any] = {'kind': str(row.get('kind') or 'finding')}
        if isinstance(row.get('why'), str):
            entry['why'] = row['why']
        if isinstance(row.get('turn_idx'), int) and not isinstance(row.get('turn_idx'), bool):
            entry['turn_idx'] = row['turn_idx']
        found.append(entry)
    return found


def _policy_entry(out: Path, scenario: str, policy: str, counts: Mapping[str, Any], trials: int, *, raised: Collection[int] = ()) -> dict[str, Any]:
    """One policy of one scenario: the outcomes per trial, and what the judges said about the first violated one.

    A trial in ``raised`` is one whose agent raised: no policy was judged there, so it is invalid (the scenario's ``invalid`` list
    gives its cause) whatever the aggregate counted for it."""
    # A deferred trial stays "deferred", never "na": the report counts it in neither failed, unstable nor not measured, and
    # names the sibling policies in ``deferred_to``.
    by_trial = [None if index in raised else outcome for index, outcome in enumerate(counts.get('by_trial') or [], start=1)]
    entry: dict[str, Any] = {'by_trial': by_trial}
    owners = counts.get('deferred_to')
    if isinstance(owners, list) and 'deferred' in by_trial:
        entry['deferred_to'] = [str(owner) for owner in owners]
    reasons: list[str] = []
    said = False
    for index in range(1, trials + 1):
        outcome = by_trial[index - 1] if index <= len(by_trial) else None
        cell = (_judged(out, scenario, index).get(policy) or {}).get('cell')
        if not isinstance(cell, Mapping):
            cell = {}
        if outcome == 'inconclusive':
            reason = cell.get('inconclusive_reason')
            reasons.append(reason if isinstance(reason, str) and reason else 'inconclusive')
        if outcome == 'failed' and not said:
            said = True
            if findings := _findings(cell):
                entry['findings'] = findings
            if isinstance(cell.get('reasoning'), str):
                entry['reasoning'] = cell['reasoning']
            if (agreement := _agreement(cell)) is not None:
                entry['agreement'] = agreement
    if reasons:
        entry['inconclusive_reasons'] = reasons
    return entry


def credit_stop(out: Path) -> dict[str, Any] | None:
    """The ``credits`` cause of the credit refusal that stopped the pass into ``out``, from the ``aborted`` block of its
    summary; ``None`` when the pass was not stopped for credit. A code outside :data:`CREDIT_CODES` carries no ``code``."""
    summary = _read(out / RUN_SUMMARY_FILENAME)
    aborted = summary.get('aborted') if isinstance(summary, dict) else None
    if not isinstance(aborted, Mapping) or aborted.get('reason') != 'credit':
        return None
    return _credit_cause(aborted)


def _invalid(out: Path, scenario: str, rows: Sequence[Mapping[str, Any]], stop: Mapping[str, Any] | None, raised: Collection[int] = ()) -> list[dict[str, Any]]:
    """The invalid trials of ``scenario``. A trial that never ran because a credit refusal stopped the pass (``stop``) is a credits
    trial, whichever scenario met the refusal; a trial with an ``error.json`` of its own keeps the cause that record names; a
    trial whose agent raised (``raised``) has the cause ``agent_raised``."""
    entries: list[dict[str, Any]] = [{'trial': index, 'reason': 'agent_raised'} for index in sorted(raised)]
    for row in rows:
        if row['trial'] in raised:
            continue
        entry: dict[str, Any] = {'trial': row['trial'], 'reason': row['reason']}
        error = _read(trial_dir(out, int(row['trial'])) / scenario / 'error.json')
        if isinstance(error, dict) and (finer := error_cause(error)) is not None and row['reason'] in ('errored', 'not_run', 'harness_fault'):
            entry = {'trial': row['trial'], **finer}
        elif stop is not None and row['reason'] == 'not_run':
            entry = {'trial': row['trial'], **stop}
        entries.append(entry)
    return entries


def _compose(
    out: Path,
    block: Mapping[str, Any],
    *,
    trials: int,
    identity: Mapping[str, Any],
    evals: Sequence[tuple[str, str]],
    pre_quarantined: Sequence[str],
    threads: int | None,
    duration_seconds: int | None,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    stop = credit_stop(out)
    for scenario, row in (block.get('scenarios') or {}).items():
        raised = [index for index in row.get('agent_raised') or [] if isinstance(index, int) and 1 <= index <= trials]
        scenarios[scenario] = {
            'invalid': _invalid(out, scenario, row.get('invalid') or [], stop, raised),
            'policies': {policy: _policy_entry(out, scenario, policy, counts, trials, raised=raised) for policy, counts in (row.get('policies') or {}).items()},
        }
    for name in pre_quarantined:
        scenarios.setdefault(name, {'invalid': [{'trial': i, 'reason': 'quarantined'} for i in range(1, trials + 1)], 'policies': {}})
    summary = _read(out / RUN_SUMMARY_FILENAME)
    backend = summary.get('backend_build') if isinstance(summary, dict) else None
    doc: dict[str, Any] = {
        'schema': RUN_SCHEMA,
        **identity,
        'evals': [{'id': name, 'verdict_tier': tier} for name, tier in evals],
        'trials': {
            'schema': block.get('schema', 'trials-v1'),
            'n': block.get('n', trials),
            'measurement': block.get('measurement') or [],
            'scenarios': scenarios,
        },
    }
    if isinstance(backend, Mapping) and isinstance(backend.get('build_id'), str):
        doc['backend_build'] = backend['build_id']
    if models := agent_models(out, scenarios, trials):
        doc['agent_models'] = models
    if threads is not None:
        doc['threads'] = threads
    if duration_seconds is not None:
        doc['duration_seconds'] = duration_seconds
    return doc


def pass_doc(
    out: Path,
    *,
    trials: int,
    identity: Mapping[str, Any],
    evals: Sequence[tuple[str, str]],
    pre_quarantined: Sequence[str] = (),
    threads: int | None = None,
    duration_seconds: int | None = None,
) -> dict[str, Any] | None:
    """The ``run.json`` of the pass that ran into ``out``, or ``None`` when the runner left no summary.

    ``identity`` is the run's identity fields (ids, repo, selection, versions, ...), copied into the document. ``evals`` lists
    the eval set's policies with their effective tiers. ``pre_quarantined`` are scenarios the artifact read stamped
    unsolvable: they were not run, and each is recorded as quarantined on every trial.
    """
    block = pass_trials_block(out, trials)
    if block is None:
        return None
    return _compose(
        out, block, trials=trials, identity=identity, evals=evals, pre_quarantined=pre_quarantined, threads=threads, duration_seconds=duration_seconds
    )


def partial_doc(
    out: Path,
    *,
    scenarios: Sequence[str],
    trials: int,
    identity: Mapping[str, Any],
    evals: Sequence[tuple[str, str]],
    pre_quarantined: Sequence[str] = (),
    threads: int | None = None,
    duration_seconds: int | None = None,
) -> dict[str, Any] | None:
    """The ``run.json`` of a pass that was cut short, read from what it left in ``out``: ``None`` when no scenario was judged
    on any trial.

    The runner writes its summary when a pass ends, so a pass cut short has none. The population is ``scenarios`` on every
    trial; a trial with a verdict file or an ``error.json`` is recorded as the runner left it, and every other trial was
    still running or had not started, which the cause ``cancelled`` records.
    """
    per_trial = []
    undecided: dict[str, set[int]] = {}
    for index in range(1, trials + 1):
        base = trial_dir(out, index)
        per_trial.append(cells_from_summary(base, {'scenarios': dict.fromkeys(scenarios, 'not_run')}))
        for name in scenarios:
            if not (verdicts_path(base / name).is_file() or (base / name / 'error.json').is_file()):
                undecided.setdefault(name, set()).add(index)
    if not any(cells.invalid_reason is None for trial in per_trial for cells in trial.values()):
        return None
    block = aggregate_trials(per_trial, run_name='ci', measurement=[(None, [])] * trials)
    doc = _compose(
        out, block, trials=trials, identity=identity, evals=evals, pre_quarantined=pre_quarantined, threads=threads, duration_seconds=duration_seconds
    )
    for name, indexes in undecided.items():
        for entry in doc['trials']['scenarios'][name]['invalid']:
            trial = entry['trial']
            if trial in indexes:
                entry.clear()
                entry.update(trial=trial, reason='cancelled')
    return doc
