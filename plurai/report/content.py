"""What of a ``run.json`` may leave the machine when run content is not included.

The same switch covers the transcripts in the artifacts and the judge text in the report (see
:func:`plurai.report.resolve_include_content`): auto includes content for a private repository only. Without content the
report shows ids, policy descriptions, outcomes, counts, causes, finding kinds and turns; the uploaded ``run.json`` carries exactly that and
nothing else. It is built by allowlist, so a field a producer adds later (a judge's reasoning, a quote, a transcript) stays
out until it is listed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plurai.report.render import Visibility, resolve_include_content

#: Top-level fields that name the run, its suite and its tooling. ``scenario_titles`` is not here: a title is text the suite
#: author or the backend wrote about the scenario's content, so it is run content.
_RUN_KEYS = (
    'schema',
    'scenario_set_id',
    'eval_set_id',
    'env_id',
    'corpus_id',
    'env_alias',
    'repo',
    'selection',
    'commit',
    'backend_build',
    'threads',
    'duration_seconds',
    'finished_on',
    'suite_scenarios',
    'retried',
    'cancelled',
    'refusal',
    'versions',
    'agent_models',
    'pinned_digests',
    'credit_preflight',
    'token_expires',
    'token_days_left',
    'policy_descriptions',
)
_EVAL_KEYS = ('id', 'verdict_tier')
_TRIALS_KEYS = ('schema', 'n')
_MEASUREMENT_KEYS = ('trial', 'clean')
#: ``why`` is the judge's sentence about the transcript, so it is not here; the kind and the turn are.
_FINDING_KEYS = ('kind', 'turn_idx')
_POLICY_KEYS = ('by_trial', 'effective_tier', 'agreement', 'inconclusive_reasons', 'deferred_to')
_INVALID_KEYS = ('trial', 'reason', 'code', 'resets_at')


def _pick(source: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if isinstance(source, Mapping) and key in source}


def _rows(value: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return [_pick(row, keys) for row in value] if isinstance(value, list) else []


def _trial(recorded: Any) -> Any:
    """A trial of ``by_trial``: an outcome string, ``None``, or the invalid entry of a merged run."""
    return _pick(recorded, _INVALID_KEYS) if isinstance(recorded, Mapping) else recorded


def _policy(entry: Any) -> dict[str, Any]:
    kept = _pick(entry, _POLICY_KEYS)
    if isinstance(kept.get('by_trial'), list):
        kept['by_trial'] = [_trial(recorded) for recorded in kept['by_trial']]
    if isinstance(entry, Mapping) and 'findings' in entry:
        kept['findings'] = _rows(entry['findings'], _FINDING_KEYS)
    return kept


def _scenario(row: Any) -> dict[str, Any]:
    policies = row.get('policies') if isinstance(row, Mapping) else None
    kept: dict[str, Any] = {'invalid': _rows(row.get('invalid') if isinstance(row, Mapping) else None, _INVALID_KEYS)}
    if isinstance(policies, Mapping):
        kept['policies'] = {name: _policy(entry) for name, entry in policies.items()}
    return kept


def without_content(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``doc`` with only the fields the report shows without content: the allowlist in this module. Outcomes, ids,
    causes, counts, finding kinds and turns stay, so the copy reads to the same report as ``doc`` rendered without content."""
    kept = _pick(doc, _RUN_KEYS)
    if isinstance(doc.get('evals'), list):
        kept['evals'] = _rows(doc['evals'], _EVAL_KEYS)
    trials = doc.get('trials')
    if isinstance(trials, Mapping):
        block = _pick(trials, _TRIALS_KEYS)
        if 'measurement' in trials:
            block['measurement'] = _rows(trials['measurement'], _MEASUREMENT_KEYS)
        scenarios = trials.get('scenarios')
        if isinstance(scenarios, Mapping):
            block['scenarios'] = {name: _scenario(row) for name, row in scenarios.items()}
        kept['trials'] = block
    return kept


def run_json_for_upload(doc: Mapping[str, Any], *, include_content: bool | None, visibility: Visibility | None) -> dict[str, Any]:
    """The ``run.json`` to upload: ``doc`` whole when content is included (an explicit ``include_content``, else a private
    repository), else :func:`without_content`."""
    return dict(doc) if resolve_include_content(include_content, visibility) else without_content(doc)
