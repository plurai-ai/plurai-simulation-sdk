"""Reading a versioned ``run.json`` into cells, each judged by the one outcome function.

The reader knows the schema versions in :data:`KNOWN_RUN_SCHEMAS` and refuses every other. The per-trial verdicts of a
cell come from the ``trials`` block (``trials-v1``): ``trials.scenarios[*].policies[*].by_trial`` holds the outcome the
judges gave on each configured trial, and ``None`` where the trial was invalid. A run without that block has no per-trial
results, so it is refused rather than rendered from its decorated row strings.

A policy entry may also carry what the judge said, read when present: ``findings`` (``[{kind, why, turn_idx}]``),
``reasoning``, ``agreement`` (``[k, n]``), ``inconclusive_reasons`` (one per inconclusive trial) and the
backend's ``effective_tier`` for that scenario × policy.

A trial the reader cannot place (an outcome or a reason it does not know) is invalid with an infra cause and is named in
:attr:`Run.unknown_causes`, so a backend release cannot silently change a cell's class. A scenario quarantined as
unsolvable is not run, so it has no cells; it is listed in :attr:`Run.quarantined`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, NamedTuple

from plurai.report.outcome import DEFAULT_FAIL_ON_TAGS, CellOutcome, NotMeasuredClass, TrialVerdict, cell_outcome, tags_of

RUN_SCHEMA: Final = 'plurai-run/1'
KNOWN_RUN_SCHEMAS: Final[tuple[str, ...]] = (RUN_SCHEMA,)
KNOWN_TRIALS_SCHEMAS: Final[tuple[str, ...]] = ('trials-v1',)

#: The tiers a policy declares, as the backend names them.
TIERS: Final[tuple[str, ...]] = ('catastrophic', 'failing', 'standard')

#: The judges' outcome on one trial, as ``by_trial`` records it, for the outcomes that decide a trial.
_DECIDED: Final[dict[str, TrialVerdict]] = {'failed': 'violated', 'passed': 'clear', 'na': 'not_applicable'}

#: The mock answered 410: a human undeployed it, so a retry cannot fix it. The cause is infra, and the report names the fix.
NEEDS_REDEPLOY: Final = 'needs_redeploy'

#: The reason on an invalid trial of a scenario the suite refused to run because it cannot be solved.
QUARANTINED: Final = 'quarantined'


class Cause(NamedTuple):
    """What the report knows about one cause of an undecided trial."""

    #: The class it earns.
    cls: NotMeasuredClass
    #: The cause in plain words, for a reader who has not seen the backend's token.
    words: str
    #: What to do about it, a phrase that follows "re-run": cause-specific because the class alone does not decide the fix.
    next_step: str


_RERUN: Final = 're-run the job'
_FIX_SUITE: Final = 'fix the policy text or the scenario'

#: Every cause the backend names for a trial that was not decided. A cause not in this table is unknown and counts as infra.
#: An inconclusive judge is a suite cause, except when a judge sample errored: a provider error that outlived its retries
#: demotes a clear majority to inconclusive, and a retry can fix that, so it is infra.
CAUSES: Final[dict[str, Cause]] = {
    'errored': Cause('infra', 'the run errored', _RERUN),
    QUARANTINED: Cause('infra', 'the scenario is quarantined as unsolvable', 'fix or remove the scenario the suite cannot solve'),
    'harness_fault': Cause('infra', 'the test harness faulted', _RERUN),
    'host_dependency_missing': Cause('infra', 'a dependency of the agent is missing on the host', "install it in the job's environment"),
    'invalid_run': Cause('infra', 'the run was invalid', _RERUN),
    'not_run': Cause('infra', 'the trial did not run', _RERUN),
    'provider_infra': Cause('infra', 'the model provider failed', _RERUN),
    'turn_budget': Cause('infra', 'the agent used every turn', 'check why the agent used every turn'),
    NEEDS_REDEPLOY: Cause('infra', 'the mock environment is not deployed', 'deploy the mock environment again with `/use-env`; it is not retried'),
    'unclean': Cause('infra', 'the measurement was not clean', _RERUN),
    'sample_error': Cause('infra', 'a judge sample errored', _RERUN),
    'not_evaluated': Cause('suite', 'the judges did not evaluate the trial', _FIX_SUITE),
    'inconclusive': Cause('suite', 'the judges could not decide', _FIX_SUITE),
    'weak_agreement': Cause('suite', 'the judges disagreed', _FIX_SUITE),
    'low_confidence': Cause('suite', 'the judges were not confident', _FIX_SUITE),
    'no_evidence': Cause('suite', 'the judges found no evidence', _FIX_SUITE),
    'contested_violation': Cause('suite', 'the judges contested the violation', _FIX_SUITE),
    'contested_applicability': Cause('suite', 'the judges disagreed on whether the policy applies', _FIX_SUITE),
    'not_measured': Cause('suite', 'the judges did not measure the trial', _FIX_SUITE),
    'credits': Cause('credits', 'out of credits', 'top up credits in the Plurai dashboard'),
}

#: What a cause the reader does not know is called, and what to do about it.
UNKNOWN_CAUSE: Final = Cause('infra', 'an unrecognised cause', 'update the SDK; the backend named a cause it does not know')

#: A GitHub repository as ``owner/name``: what ``--github`` takes and what an expiry note quotes.
REPO_SHAPE: Final = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')

#: The suite ids a run records, by the name the report uses.
_SUITE_ID_KEYS: Final[dict[str, str]] = {'scenarios': 'scenario_set_id', 'evals': 'eval_set_id', 'env': 'env_id', 'corpus': 'corpus_id'}


class RunJsonError(ValueError):
    """The ``run.json`` cannot be read: an unknown schema, a missing block, a value that is not what the schema says."""


class Credits(NamedTuple):
    """The gateway's credit refusal behind a credits cause: its code, and when it resets when it says."""

    code: str | None
    resets_at: str | None


@dataclass(frozen=True)
class CellDetail:
    """What is known about one scenario × policy beyond the run's counts: the judge's words and why trials were inconclusive."""

    #: The kind of the judge's first finding, its reason (``findings[0].why``) and every 0-based turn the findings point at.
    kind: str | None = None
    why: str | None = None
    turns: tuple[int, ...] = ()
    #: The judge's full reasoning, and how many of its samples agreed out of how many.
    reasoning: str | None = None
    agreement: tuple[int, int] | None = None
    #: The reason of each inconclusive trial, in trial order.
    inconclusive_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ('kind', 'why', 'reasoning'):
            if isinstance(value := getattr(self, name), str):
                object.__setattr__(self, name, scrub_surrogates(value))
        object.__setattr__(self, 'inconclusive_reasons', tuple(scrub_surrogates(r) for r in self.inconclusive_reasons))


@dataclass(frozen=True)
class Cell:
    """One scenario paired with one policy, with its per-trial verdicts and the outcome they earn."""

    scenario: str
    policy: str
    tags: tuple[str, ...]
    verdicts: tuple[TrialVerdict, ...]
    #: Per trial, why it is invalid (a typed reason, or the judge's inconclusive reason); ``None`` for a valid trial.
    causes: tuple[str | None, ...]
    outcome: CellOutcome
    detail: CellDetail = field(default_factory=CellDetail)
    credits: Credits | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.scenario, self.policy)


@dataclass(frozen=True)
class Run:
    """A read ``run.json``: its cells and the identity and facts the report header and footer carry."""

    schema: str
    #: The configured trial count (T).
    trials: int
    cells: tuple[Cell, ...]
    versions: Mapping[str, str] = field(default_factory=dict)
    #: The causes of invalid trials that the reader does not know; they are counted as infra.
    unknown_causes: tuple[str, ...] = ()
    #: Scenarios quarantined as unsolvable: not run, so not cells and not counted.
    quarantined: tuple[str, ...] = ()
    #: The ids of the suite the run scored, by kind, for each id the run carries.
    suite_ids: Mapping[str, str] = field(default_factory=dict)
    #: The name the project gives the mock environment (``ci.yaml``'s ``env`` alias), which the footer shows beside its id.
    env_alias: str | None = None
    repo: str | None = None
    selection: str | None = None
    #: The agent's commit, which names this run among a repository's runs.
    commit: str | None = None
    agent_models: tuple[str, ...] = ()
    backend_build: str | None = None
    threads: int | None = None
    duration_seconds: int | None = None


_SURROGATES = re.compile('[\ud800-\udfff]')


def scrub_surrogates(text: str) -> str:
    """``text`` with lone surrogates (which no encoder accepts) replaced."""
    return _SURROGATES.sub('\ufffd', text)


def _text(value: object) -> str | None:
    return scrub_surrogates(value) if isinstance(value, str) and value else None


def _object(value: object, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RunJsonError(f'run.json {where} is not an object')
    return value


def _array(value: object, where: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RunJsonError(f'run.json {where} is not an array')
    return value


def _objects(value: object, where: str) -> list[Mapping[str, Any]]:
    return [item if isinstance(item, dict) else _object(False, f'{where}[{index}]') for index, item in enumerate(_array(value, where))]


def _repo(value: object) -> str | None:
    """The run's ``owner/name``; ``None`` when absent. Any other shape is refused: the report quotes it in a command."""
    if value is None or value == '':
        return None
    if not isinstance(value, str) or not REPO_SHAPE.fullmatch(value):
        raise RunJsonError('run.json repo is not an owner/name')
    return value


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _agreement(value: object, where: str) -> tuple[int, int] | None:
    """``k`` of ``n`` judge samples agreeing, from ``[k, n]``; ``None`` when absent."""
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 2 and all(_count(part) is not None for part in value) and value[1] > 0:
        return (value[0], value[1])
    raise RunJsonError(f'run.json {where} agreement is not [k, n]')


def _findings_detail(findings: object, where: str) -> CellDetail:
    """The detail of a list of ``{kind, why, turn_idx}`` findings: the first finding's kind and why, every turn."""
    rows = _objects(findings, f'{where} findings')
    turns = tuple(dict.fromkeys(f['turn_idx'] for f in rows if isinstance(f.get('turn_idx'), int)))
    first = rows[0] if rows else {}
    return CellDetail(kind=_text(first.get('kind')), why=_text(first.get('why')), turns=turns)


def _entry_detail(entry: Mapping[str, Any], where: str) -> CellDetail:
    """The detail a policy entry carries itself."""
    return replace(
        _findings_detail(entry.get('findings'), where),
        reasoning=_text(entry.get('reasoning')),
        agreement=_agreement(entry.get('agreement'), where),
        inconclusive_reasons=_strings(entry.get('inconclusive_reasons'), f'{where} inconclusive_reasons'),
    )


def _strings(value: object, where: str) -> tuple[str, ...]:
    """The strings of an array; any other item is refused. An inconclusive reason is the gates that fired joined with ``+``."""
    items = _array(value, where)
    if not all(isinstance(item, str) for item in items):
        raise RunJsonError(f'run.json {where} has an item that is not a string')
    return tuple(items)


def _tier_by_policy(doc: Mapping[str, Any]) -> dict[str, str]:
    return {str(row['id']): row['verdict_tier'] for row in _objects(doc.get('evals'), 'evals') if row.get('verdict_tier') in TIERS and 'id' in row}


def _unclean_trials(doc: Mapping[str, Any], configured: int) -> frozenset[int]:
    """The trials whose measurement the run records as not clean (``trials.measurement[*].clean`` is false). Each row names an
    integer trial in ``1..configured`` once; any other row says the run disagrees with itself, so the run is refused."""
    rows = _objects(doc['trials'].get('measurement'), 'trials.measurement')
    seen: set[int] = set()
    for row in rows:
        trial = row.get('trial')
        if not isinstance(trial, int) or isinstance(trial, bool):
            raise RunJsonError('run.json trials.measurement has a row with no integer trial')
        if not 1 <= trial <= configured:
            raise RunJsonError(f'run.json trials.measurement names trial {trial}, outside 1..{configured}')
        if trial in seen:
            raise RunJsonError(f'run.json trials.measurement names trial {trial} twice')
        seen.add(trial)
    return frozenset(row['trial'] for row in rows if row.get('clean') is False)


def classify(cause: str) -> tuple[TrialVerdict, bool]:
    """The verdict an undecided trial earns from its ``cause``, and whether the reader knows every part of the cause.

    The backend joins every inconclusive gate that fired with ``+``, so a cause can name several parts. Any part that is
    infra or unknown makes the trial infra; else a credits part makes it credits; else it is suite.
    """
    causes = [CAUSES.get(part) for part in cause.split('+')]
    unknown = None in causes
    classes = [c.cls for c in causes if c is not None]
    if unknown or 'infra' in classes:
        return 'invalid_infra', unknown
    return ('invalid_credits' if 'credits' in classes else 'invalid_suite'), False


def _trial_verdicts(
    by_trial: Sequence[object], invalid: Mapping[int, Mapping[str, Any]], inconclusive: Sequence[str], unclean: frozenset[int], configured: int
) -> tuple[tuple[TrialVerdict, ...], tuple[str | None, ...], frozenset[str]]:
    """Per configured trial, its verdict and the cause when it was not decided, and the causes the reader does not know."""
    verdicts: list[TrialVerdict] = []
    causes: list[str | None] = []
    unknown: set[str] = set()
    reasons = iter(inconclusive)
    for index in range(1, configured + 1):
        recorded = by_trial[index - 1] if index <= len(by_trial) else None
        if isinstance(recorded, str) and recorded in _DECIDED:
            verdicts.append(_DECIDED[recorded])
            causes.append(None)
            continue
        if recorded == 'inconclusive':
            reason = next(reasons, 'inconclusive')
            # An unclean measurement reaches no verdict on either axis, whatever the judges then said.
            cause = 'unclean' if index in unclean else reason
        elif recorded is None:
            cause = str(invalid.get(index, {}).get('reason') or 'not_run')
        else:
            cause = str(recorded)
        cause = scrub_surrogates(cause)
        verdict, is_unknown = classify(cause)
        verdicts.append(verdict)
        causes.append(cause)
        if is_unknown:
            unknown.add(cause)
    return tuple(verdicts), tuple(causes), frozenset(unknown)


def _credits(invalid: Mapping[int, Mapping[str, Any]], verdicts: Sequence[TrialVerdict]) -> Credits | None:
    """The credit refusal of the first of the cell's own trials that stopped for credits."""
    for index, verdict in enumerate(verdicts, start=1):
        if verdict == 'invalid_credits':
            bad = invalid.get(index, {})
            return Credits(_text(bad.get('code')), _text(bad.get('resets_at')))
    return None


def _run_schema(doc: Mapping[str, Any]) -> str:
    """The schema ``doc`` names, which must be one of :data:`KNOWN_RUN_SCHEMAS`."""
    if not isinstance(doc, Mapping):
        raise RunJsonError('run.json is not a JSON object')
    schema = doc.get('schema')
    if schema is None:
        raise RunJsonError(f'run.json names no schema; this reader reads {", ".join(KNOWN_RUN_SCHEMAS)}')
    if schema not in KNOWN_RUN_SCHEMAS:
        raise RunJsonError(f'run.json schema {schema!r} is not one this reader knows; it reads {", ".join(KNOWN_RUN_SCHEMAS)}')
    return str(schema)


def _validated(doc: Mapping[str, Any]) -> tuple[str, int, dict[str, Any]]:
    """The run's schema, its configured trial count and its scenario rows, or :class:`RunJsonError`."""
    schema = _run_schema(doc)
    block = doc.get('trials')
    if not isinstance(block, dict):
        raise RunJsonError('run.json has no trials block, so it carries no per-trial results; run the suite with trials')
    if block.get('schema') not in KNOWN_TRIALS_SCHEMAS:
        raise RunJsonError(f'run.json trials schema {block.get("schema")!r} is not one this reader knows; it reads {", ".join(KNOWN_TRIALS_SCHEMAS)}')
    configured = block.get('n')
    rows = block.get('scenarios')
    if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1 or not isinstance(rows, dict):
        raise RunJsonError('run.json trials block carries no valid trial count or scenario rows')
    return schema, configured, rows


def _invalid_entries(row: Mapping[str, Any], configured: int) -> dict[int, Mapping[str, Any]]:
    """The scenario's ``invalid`` entries by trial. Each names an integer trial in ``1..configured`` once: an entry outside
    that range or a repeated index says the run disagrees with itself, so the run is refused."""
    entries: dict[int, Mapping[str, Any]] = {}
    for bad in _objects(row.get('invalid'), 'trials.scenarios[*].invalid'):
        trial = bad.get('trial')
        if not isinstance(trial, int) or isinstance(trial, bool):
            raise RunJsonError('run.json trials.scenarios[*].invalid has an entry with no integer trial')
        if not 1 <= trial <= configured:
            raise RunJsonError(f'run.json trials.scenarios[*].invalid names trial {trial}, outside 1..{configured}')
        if trial in entries:
            raise RunJsonError(f'run.json trials.scenarios[*].invalid names trial {trial} twice')
        entries[trial] = bad
    return entries


def _is_quarantined(row: Mapping[str, Any], configured: int, invalid: Mapping[int, Mapping[str, Any]]) -> bool:
    """Whether the suite refused the scenario as unsolvable in every trial and no policy recorded a verdict.

    Quarantine is claimed only when every trial index has an ``invalid`` entry of reason ``quarantined`` and nothing was
    decided, so a decided trial cannot make a scenario vanish from the counts.
    """
    covered = sum(1 for bad in invalid.values() if bad.get('reason') == QUARANTINED)
    entries = _object(row.get('policies'), 'policies').values()
    decided = any(v is not None for entry in entries if isinstance(entry, dict) for v in _array(entry.get('by_trial'), 'by_trial'))
    return covered == configured and not decided


def _universe(doc: Mapping[str, Any], rows: Mapping[str, Any]) -> list[str]:
    """The policies of the run: those any scenario recorded, else the eval set's. A scenario that measured nothing names
    none of its own, and still owes a not-measured cell for each."""
    recorded = {name for row in rows.values() if isinstance(row, dict) for name in _object(row.get('policies'), 'policies')}
    return sorted(recorded or {str(r['id']) for r in _objects(doc.get('evals'), 'evals') if 'id' in r})


def read_run(
    doc: Mapping[str, Any],
    *,
    fail_on: Iterable[str] = DEFAULT_FAIL_ON_TAGS,
) -> Run:
    """The cells of ``doc`` (a parsed ``run.json``), with each outcome computed by :func:`cell_outcome`.

    A cell's tier is the cell's own ``effective_tier`` in the run (the tier the backend resolved for that cell), else the
    policy's declared ``verdict_tier``. A cell with no known tier carries no tier tag and is advisory unless its scenario
    is a regression. What the judge said comes from the policy entry. A scenario has a cell
    for each policy it records; one that records none owes a cell for every policy of the run. A policy recorded for some
    scenarios and absent from another is not a cell of that scenario: the run says nothing about it. Raises
    :class:`RunJsonError` on an unknown or missing schema or a run that carries no ``trials`` block.
    """
    schema, configured, rows = _validated(doc)
    unclean = _unclean_trials(doc, configured)
    declared = _tier_by_policy(doc)
    universe = _universe(doc, rows)
    cells: list[Cell] = []
    quarantined: list[str] = []
    unknown: set[str] = set()
    for sid, raw in rows.items():
        row = _object(raw, f'trials.scenarios.{sid}')
        invalid = _invalid_entries(row, configured)
        if _is_quarantined(row, configured, invalid):
            quarantined.append(scrub_surrogates(sid))
            continue
        recorded = _object(row.get('policies'), f'trials.scenarios.{sid}.policies')
        for name in sorted(recorded or universe):
            where = f'trials.scenarios.{sid}.policies.{name}'
            entry = _object(recorded.get(name), where)
            detail = _entry_detail(entry, where)
            by_trial = _array(entry.get('by_trial'), f'{where}.by_trial')
            if len(by_trial) > configured:
                raise RunJsonError(f'run.json {where}.by_trial has {len(by_trial)} entries for {configured} trials')
            verdicts, causes, unknown_here = _trial_verdicts(by_trial, invalid, detail.inconclusive_reasons, unclean, configured)
            unknown |= unknown_here
            tier = _text(entry.get('effective_tier')) or declared.get(name)
            tags = tags_of(tier if tier in TIERS else None, sid)
            outcome = cell_outcome(tags=tags, trials=verdicts, fail_on=fail_on)
            credits = _credits(invalid, verdicts)
            cells.append(
                Cell(
                    scenario=scrub_surrogates(sid),
                    policy=scrub_surrogates(name),
                    tags=tags,
                    verdicts=verdicts,
                    causes=causes,
                    outcome=outcome,
                    detail=detail,
                    credits=credits,
                )
            )
    if rows and not cells and not quarantined:
        raise RunJsonError('run.json records no policy for any scenario, so there is no scenario × policy to judge')
    return _run_of(doc, schema, configured, cells, unknown=tuple(sorted(unknown)), quarantined=tuple(quarantined))


def _run_of(doc: Mapping[str, Any], schema: str, configured: int, cells: list[Cell], *, unknown: tuple[str, ...], quarantined: tuple[str, ...]) -> Run:
    versions = doc.get('versions')
    models = doc.get('agent_models')
    return Run(
        schema=schema,
        trials=configured,
        cells=tuple(cells),
        versions={str(k): str(v) for k, v in versions.items()} if isinstance(versions, dict) else {},
        unknown_causes=unknown,
        quarantined=quarantined,
        suite_ids={kind: doc[key] for kind, key in _SUITE_ID_KEYS.items() if _text(doc.get(key))},
        env_alias=_text(doc.get('env_alias')),
        repo=_repo(doc.get('repo')),
        selection=_text(doc.get('selection')),
        commit=_text(doc.get('commit')),
        agent_models=tuple(str(m) for m in models if _text(m)) if isinstance(models, list) else (),
        backend_build=_text(doc.get('backend_build')),
        threads=_count(doc.get('threads')),
        duration_seconds=_count(doc.get('duration_seconds')),
    )
